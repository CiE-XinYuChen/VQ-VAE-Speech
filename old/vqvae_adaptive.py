import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.vector_quantizer_ema import VectorQuantizerEMA


class AdaptiveVectorQuantizerEMA(VectorQuantizerEMA):
    """自适应VQ-EMA，能自动确定最优码本数量"""
    
    def __init__(self, embedding_dim, commitment_cost, decay, device,
                 initial_num_embeddings=None,
                 min_usage_threshold=0.01,
                 merge_distance_threshold=0.5,
                 prune_interval=50,
                 stable_epochs_required=100,
                 epsilon=1e-5):
        """
        Args:
            initial_num_embeddings: 初始码本数量，None表示自动确定
            min_usage_threshold: 最小使用率阈值，低于此值的码本会被删除
            merge_distance_threshold: 合并距离阈值，距离小于此值的码本会被合并
            prune_interval: 剪枝检查间隔
            stable_epochs_required: 判定稳定所需的epoch数
        """
        
        # 如果没有指定初始码本数，使用默认策略
        if initial_num_embeddings is None:
            initial_num_embeddings = 20  # 默认从20个开始
        
        super().__init__(
            num_embeddings=initial_num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
            device=device,
            epsilon=epsilon
        )
        
        self.min_usage_threshold = min_usage_threshold
        self.merge_distance_threshold = merge_distance_threshold
        self.prune_interval = prune_interval
        self.stable_epochs_required = stable_epochs_required
        
        # 追踪码本使用情况
        self.register_buffer('_ema_usage', torch.zeros(initial_num_embeddings))
        self.register_buffer('_active_codes', torch.ones(initial_num_embeddings, dtype=torch.bool))
        
        # 追踪稳定性
        self.epoch_count = 0
        self.last_active_count = initial_num_embeddings
        self.stable_epochs = 0
        self.pruning_history = []
        
    def forward(self, inputs, compute_distances_if_possible=True, record_codebook_stats=False):
        """前向传播，包含自适应调整"""
        
        # 只使用活跃的码本
        active_indices = torch.where(self._active_codes)[0]
        
        if len(active_indices) == 0:
            # 如果没有活跃码本，重新初始化一个
            self._active_codes[0] = True
            active_indices = torch.tensor([0], device=self._device)
        
        # 获取活跃的嵌入
        active_embeddings = self._embedding.weight[active_indices]
        
        # 计算到活跃码本的距离
        flat_input = inputs.permute(1, 2, 0).reshape(-1, self._embedding_dim)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(active_embeddings**2, dim=1)
                    - 2 * torch.matmul(flat_input, active_embeddings.t()))
        
        # 找到最近的码本
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], len(active_indices), device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)
        
        # 映射回原始索引
        original_indices = active_indices[encoding_indices.squeeze()]
        original_encodings = torch.zeros(
            encodings.shape[0], self._num_embeddings, device=inputs.device
        )
        original_encodings[:, active_indices] = encodings
        
        # 量化
        quantized = torch.matmul(encodings, active_embeddings).view(
            inputs.shape[1], inputs.shape[2], inputs.shape[0]
        ).permute(2, 0, 1)
        
        # 在训练模式下更新统计和执行剪枝
        if self.training:
            self.epoch_count += 1
            
            # 更新使用统计 - 累积批次中每个码本的使用次数
            batch_usage = torch.zeros(self._num_embeddings, device=self._device)
            for idx in original_indices:
                batch_usage[idx] += 1
            batch_usage = batch_usage / len(original_indices)  # 归一化
            self._ema_usage = self._ema_usage * self._decay + (1 - self._decay) * batch_usage
            
            # 更新EMA权重（只更新活跃码本）
            self._ema_cluster_size[active_indices] = (
                self._ema_cluster_size[active_indices] * self._decay +
                (1 - self._decay) * torch.sum(encodings, dim=0)
            )
            
            # 获取编码的总和
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w[active_indices] = (
                self._ema_w[active_indices] * self._decay + (1 - self._decay) * dw
            )
            
            # 更新嵌入权重
            n = torch.sum(self._ema_cluster_size[active_indices])
            updated_cluster_size = (
                (self._ema_cluster_size[active_indices] + self._epsilon) /
                (n + len(active_indices) * self._epsilon) * n
            )
            self._embedding.weight.data[active_indices] = (
                self._ema_w[active_indices] / updated_cluster_size.unsqueeze(1)
            )
            
            # 定期执行剪枝
            if self.epoch_count % self.prune_interval == 0:
                self._prune_and_merge()
        
        # 计算损失
        e_latent_loss = nn.functional.mse_loss(quantized.detach(), inputs)
        commitment_loss = self._commitment_cost * nn.functional.mse_loss(quantized, inputs.detach())
        vq_loss = commitment_loss + e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        # 计算perplexity（只考虑活跃码本）
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # 返回标准格式的结果
        return (vq_loss, quantized, perplexity, original_encodings, distances, 
                original_indices.unsqueeze(1), {}, None, None, None, quantized)
    
    def _prune_and_merge(self):
        """剪枝未使用的码本并合并相似的码本"""
        
        with torch.no_grad():
            active_indices = torch.where(self._active_codes)[0]
            
            # 1. 剪枝低使用率的码本
            usage_rates = self._ema_usage / (self._ema_usage.sum() + 1e-10)
            low_usage = usage_rates < self.min_usage_threshold
            
            # 标记要删除的码本
            to_prune = low_usage & self._active_codes
            
            # 2. 合并相似的码本
            if len(active_indices) > 1:
                active_embeddings = self._embedding.weight[active_indices]
                
                # 计算活跃码本之间的距离
                distances = torch.cdist(active_embeddings, active_embeddings)
                
                # 找到距离小于阈值的码本对
                close_pairs = (distances < self.merge_distance_threshold) & (distances > 0)
                
                # 合并相似码本（保留使用率更高的）
                for i in range(len(active_indices)):
                    for j in range(i + 1, len(active_indices)):
                        if close_pairs[i, j]:
                            idx_i, idx_j = active_indices[i], active_indices[j]
                            
                            # 保留使用率更高的码本
                            if self._ema_usage[idx_i] > self._ema_usage[idx_j]:
                                to_prune[idx_j] = True
                                # 合并权重（加权平均）
                                total_usage = self._ema_usage[idx_i] + self._ema_usage[idx_j]
                                if total_usage > 0:
                                    weight_i = self._ema_usage[idx_i] / total_usage
                                    weight_j = self._ema_usage[idx_j] / total_usage
                                    self._embedding.weight.data[idx_i] = (
                                        weight_i * self._embedding.weight.data[idx_i] +
                                        weight_j * self._embedding.weight.data[idx_j]
                                    )
                                    self._ema_usage[idx_i] = total_usage
                            else:
                                to_prune[idx_i] = True
                                total_usage = self._ema_usage[idx_i] + self._ema_usage[idx_j]
                                if total_usage > 0:
                                    weight_i = self._ema_usage[idx_i] / total_usage
                                    weight_j = self._ema_usage[idx_j] / total_usage
                                    self._embedding.weight.data[idx_j] = (
                                        weight_i * self._embedding.weight.data[idx_i] +
                                        weight_j * self._embedding.weight.data[idx_j]
                                    )
                                    self._ema_usage[idx_j] = total_usage
            
            # 执行剪枝
            self._active_codes[to_prune] = False
            self._ema_usage[to_prune] = 0
            
            # 记录剪枝历史
            num_pruned = to_prune.sum().item()
            current_active = self._active_codes.sum().item()
            self.pruning_history.append({
                'epoch': self.epoch_count,
                'pruned': num_pruned,
                'active': current_active
            })
            
            # 检查稳定性
            if current_active == self.last_active_count:
                self.stable_epochs += self.prune_interval
            else:
                self.stable_epochs = 0
                self.last_active_count = current_active
            
            if num_pruned > 0:
                print(f"Epoch {self.epoch_count}: Pruned {num_pruned} codes, "
                      f"{current_active} active codes remaining")
    
    def get_active_codes_count(self):
        """获取当前活跃的码本数量"""
        return self._active_codes.sum().item()
    
    def get_optimal_num_embeddings(self):
        """获取最优的码本数量"""
        return self.get_active_codes_count()
    
    def is_stable(self):
        """判断码本数量是否已稳定"""
        return self.stable_epochs >= self.stable_epochs_required
    
    def set_data_based_initial_codes(self, data_size):
        """基于数据规模自动设置初始码本数"""
        # 使用经验公式：sqrt(数据点数) 或 log(数据点数) * 2
        suggested_num = min(30, max(5, int(np.sqrt(data_size))))
        
        # 重新初始化
        self._num_embeddings = suggested_num
        self._embedding = nn.Embedding(suggested_num, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._ema_w = nn.Parameter(self._embedding.weight.data.clone(), requires_grad=False)
        self._ema_cluster_size = nn.Parameter(torch.zeros(suggested_num), requires_grad=False)
        self._ema_usage = torch.zeros(suggested_num).to(self._device)
        self._active_codes = torch.ones(suggested_num, dtype=torch.bool).to(self._device)
        
        return suggested_num


def train_adaptive_vqvae(data, encoder, decoder, n_epochs=300, learning_rate=1e-3):
    """训练自适应VQ-VAE"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 准备数据 - 创建序列数据
    def create_sequence_data(data, seq_length=10):
        n_samples = len(data)
        n_sequences = n_samples - seq_length + 1
        sequences = []
        
        for i in range(n_sequences):
            seq = data[i:i+seq_length]
            sequences.append(seq)
        
        return torch.stack(sequences)
    
    sequences = create_sequence_data(data, seq_length=10)
    sequences = sequences.to(device)
    batch_size, seq_length, dim = sequences.shape
    
    # 创建自适应VQ层
    vq_layer = AdaptiveVectorQuantizerEMA(
        embedding_dim=2,
        commitment_cost=0.25,
        decay=0.99,
        device=device,
        initial_num_embeddings=None,  # 自动确定
        min_usage_threshold=0.001,  # 降低阈值，减少过度剪枝
        merge_distance_threshold=0.3,  # 降低合并阈值，只合并非常相似的码本
        prune_interval=50,  # 增加剪枝间隔
        stable_epochs_required=80  # 增加稳定要求
    ).to(device)
    
    # 基于数据规模设置初始码本数
    initial_num = vq_layer.set_data_based_initial_codes(len(data))
    print(f"Initial number of codes: {initial_num}")
    
    # K-means初始化
    with torch.no_grad():
        flat_data = sequences.reshape(-1, 2).cpu().numpy()
        kmeans = KMeans(n_clusters=initial_num, random_state=42, n_init=10)
        kmeans.fit(flat_data)
        initial_centers = torch.FloatTensor(kmeans.cluster_centers_).to(device)
        vq_layer._embedding.weight.data[:initial_num] = initial_centers
        vq_layer._ema_w.data[:initial_num] = initial_centers.clone()
    
    # 优化器
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate
    )
    
    criterion = nn.MSELoss()
    
    train_history = {
        'total_loss': [],
        'vq_loss': [],
        'recon_loss': [],
        'perplexity': [],
        'active_codes': [],
        'pruning_events': []
    }
    
    for epoch in range(n_epochs):
        encoder.train()
        decoder.train()
        vq_layer.train()
        
        # 编码
        flat_sequences = sequences.reshape(-1, 2)
        encoded = encoder(flat_sequences)
        encoded = encoded.reshape(batch_size, seq_length, 2).permute(2, 1, 0)
        
        # VQ前向传播
        vq_loss, quantized, perplexity, encodings, distances, encoding_indices, \
            _, _, _, _, _ = vq_layer(encoded)
        
        # 解码
        quantized_permuted = quantized.permute(2, 1, 0).reshape(-1, 2)
        reconstructed = decoder(quantized_permuted)
        reconstructed = reconstructed.reshape(batch_size, seq_length, 2)
        
        # 计算损失
        reconstruction_loss = criterion(reconstructed, sequences)
        total_loss = vq_loss + reconstruction_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 记录
        train_history['total_loss'].append(total_loss.item())
        train_history['vq_loss'].append(vq_loss.item())
        train_history['recon_loss'].append(reconstruction_loss.item())
        train_history['perplexity'].append(perplexity.item())
        train_history['active_codes'].append(vq_layer.get_active_codes_count())
        
        if (epoch + 1) % 20 == 0:
            active_codes = vq_layer.get_active_codes_count()
            print(f"Epoch [{epoch+1}/{n_epochs}] "
                  f"Loss: {total_loss.item():.4f}, "
                  f"Active Codes: {active_codes}, "
                  f"Perplexity: {perplexity.item():.2f}")
        
        # 检查是否稳定
        if vq_layer.is_stable():
            print(f"\nCodebook size stabilized at {vq_layer.get_optimal_num_embeddings()} codes")
            print(f"Training stopped at epoch {epoch+1}")
            break
    
    # 记录剪枝事件
    train_history['pruning_events'] = vq_layer.pruning_history
    
    return vq_layer, train_history


def visualize_adaptive_results(data, labels, vq_layer, train_history):
    """可视化自适应VQ-VAE结果"""
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 活跃码本数量变化
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(train_history['active_codes'], linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Active Codes')
    ax1.set_title('Active Codebook Size Evolution')
    ax1.grid(True)
    
    # 标记剪枝事件
    for event in train_history['pruning_events']:
        ax1.axvline(x=event['epoch'], color='r', linestyle='--', alpha=0.3)
    
    # 2. 训练损失
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(train_history['total_loss'], label='Total', alpha=0.7)
    ax2.plot(train_history['vq_loss'], label='VQ', alpha=0.7)
    ax2.plot(train_history['recon_loss'], label='Recon', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Losses')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Perplexity
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(train_history['perplexity'])
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Perplexity')
    ax3.set_title('Codebook Usage (Perplexity)')
    ax3.grid(True)
    
    # 4. 原始数据
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', 
                         alpha=0.6, s=20)
    ax4.set_title('Original Data (True Clusters)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    plt.colorbar(scatter, ax=ax4)
    
    # 5. 最终码本位置
    ax5 = plt.subplot(2, 3, 5)
    
    # 获取活跃码本
    active_indices = torch.where(vq_layer._active_codes)[0]
    if len(active_indices) > 0:
        active_embeddings = vq_layer._embedding.weight[active_indices].detach().cpu().numpy()
        
        # 绘制数据和码本
        ax5.scatter(data[:, 0], data[:, 1], c='gray', alpha=0.3, s=10)
        ax5.scatter(active_embeddings[:, 0], active_embeddings[:, 1], 
                   c='red', s=200, marker='*', edgecolors='black', linewidth=2)
        
        for i, pos in enumerate(active_embeddings):
            ax5.annotate(f'{i}', pos, fontsize=10, ha='center', va='center', 
                        color='white', weight='bold')
    
    ax5.set_title(f'Final Active Codes: {len(active_indices)}')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    
    # 6. 剪枝历史
    ax6 = plt.subplot(2, 3, 6)
    if train_history['pruning_events']:
        epochs = [e['epoch'] for e in train_history['pruning_events']]
        pruned = [e['pruned'] for e in train_history['pruning_events']]
        active = [e['active'] for e in train_history['pruning_events']]
        
        ax6.bar(epochs, pruned, width=20, alpha=0.5, label='Pruned')
        ax6.plot(epochs, active, 'o-', color='green', label='Active')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Number of Codes')
        ax6.set_title('Pruning History')
        ax6.legend()
        ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig('adaptive_vqvae_results.png', dpi=150, bbox_inches='tight')
    print("Results saved to adaptive_vqvae_results.png")
    
    # 打印最终统计
    print(f"\n=== Final Statistics ===")
    print(f"Initial codes: {len(train_history['active_codes']) and train_history['active_codes'][0]}")
    print(f"Final active codes: {vq_layer.get_optimal_num_embeddings()}")
    print(f"Reduction: {(1 - vq_layer.get_optimal_num_embeddings() / train_history['active_codes'][0]) * 100:.1f}%")
    print(f"Final perplexity: {train_history['perplexity'][-1]:.2f}")
    
    return fig


def main():
    """主函数"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=== Adaptive VQ-VAE with Automatic Codebook Size ===\n")
    
    # 生成2D聚类数据
    def generate_2d_clusters(n_points=700, n_clusters=7, noise=0.25, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        points_per_cluster = n_points // n_clusters
        data = []
        labels = []
        
        # 生成聚类中心
        centers = np.random.uniform(-2, 2, size=(n_clusters, 2))
        
        for i in range(n_clusters):
            cluster_data = np.random.randn(points_per_cluster, 2) * noise + centers[i]
            data.append(cluster_data)
            labels.extend([i] * points_per_cluster)
        
        data = np.vstack(data)
        labels = np.array(labels)
        
        # 打乱数据
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]
        
        return torch.FloatTensor(data), labels
    
    # 测试不同复杂度的数据
    print("Testing with 7 clusters...")
    data, labels = generate_2d_clusters(n_points=700, n_clusters=7, noise=0.25)
    
    # 创建简单的编码器和解码器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    ).to(device)
    
    decoder = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    ).to(device)
    
    # 训练自适应VQ-VAE
    vq_layer, train_history = train_adaptive_vqvae(
        data, encoder, decoder, n_epochs=300, learning_rate=5e-3
    )
    
    # 可视化结果
    visualize_adaptive_results(data, labels, vq_layer, train_history)
    
    print("\n" + "="*50)
    print("The model automatically determined the optimal number of codes!")
    print(f"For {len(np.unique(labels))} true clusters, "
          f"the model found {vq_layer.get_optimal_num_embeddings()} codes are optimal.")


if __name__ == "__main__":
    main()