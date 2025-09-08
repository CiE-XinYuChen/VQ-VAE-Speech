import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict


class AdaptiveVectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        embedding_dim,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
        device='cpu',
        initial_num_embeddings=None,
        min_usage_rate=0.02,  # 提高最小使用率阈值到2%
        merge_similarity_threshold=0.5,  # 提高合并阈值，更积极地合并相似码本
        prune_check_interval=30,  # 更频繁地检查
        stable_epochs=50  # 稳定所需的epoch数
    ):
        super().__init__()
        
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._epsilon = epsilon
        self._device = device
        
        # 自适应参数
        self.min_usage_rate = min_usage_rate
        self.merge_similarity_threshold = merge_similarity_threshold
        self.prune_check_interval = prune_check_interval
        self.stable_epochs = stable_epochs
        
        # 初始码本数（如果未指定，后续会根据数据确定）
        if initial_num_embeddings is not None:
            self._num_embeddings = initial_num_embeddings
            self._initialize_embeddings()
        else:
            self._num_embeddings = None
            self._embedding = None
    
    def _initialize_embeddings(self):
        """初始化嵌入和统计信息"""
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        
        # EMA统计
        self.register_buffer('_ema_cluster_size', torch.zeros(self._num_embeddings))
        self.register_buffer('_ema_w', torch.zeros(self._num_embeddings, self._embedding_dim))
        
        # 使用统计（用于自适应）
        self.register_buffer('_usage_count', torch.zeros(self._num_embeddings))
        self.register_buffer('_total_usage', torch.tensor(0.0))
        
        # 活跃码本标记
        self.register_buffer('_active_codes', torch.ones(self._num_embeddings, dtype=torch.bool))
        
        # 训练统计
        self.epoch_count = 0
        self.last_num_active = self._num_embeddings
        self.stable_count = 0
    
    def initialize_from_data(self, data, n_init_codes=None):
        """根据数据初始化码本"""
        if n_init_codes is None:
            # 自动确定初始码本数：sqrt(n_samples)，上限32
            n_samples = len(data)
            n_init_codes = min(int(np.sqrt(n_samples)), 32)
        
        self._num_embeddings = n_init_codes
        self._initialize_embeddings()
        
        # 使用K-means初始化
        if len(data) >= n_init_codes:
            kmeans = KMeans(n_clusters=n_init_codes, random_state=42, n_init=3)
            kmeans.fit(data.cpu().numpy() if isinstance(data, torch.Tensor) else data)
            self._embedding.weight.data = torch.FloatTensor(kmeans.cluster_centers_).to(self._device)
        
        print(f"Initialized with {n_init_codes} codes from data")
        return n_init_codes
    
    def forward(self, inputs):
        # 确保已初始化
        if self._embedding is None:
            raise RuntimeError("Embeddings not initialized. Call initialize_from_data first.")
        
        # 输入形状处理
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # 只使用活跃的码本
        active_indices = torch.where(self._active_codes)[0]
        if len(active_indices) == 0:
            raise RuntimeError("No active codes remaining!")
        
        active_embeddings = self._embedding.weight[active_indices]
        
        # 计算距离（只对活跃码本）
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True) +
            torch.sum(active_embeddings**2, dim=1) -
            2 * torch.matmul(flat_input, active_embeddings.t())
        )
        
        # 找到最近的码本
        encoding_indices = torch.argmin(distances, dim=1)
        
        # 映射回原始索引
        original_indices = active_indices[encoding_indices]
        
        # One-hot编码（对原始索引）
        encodings = torch.zeros(len(flat_input), self._num_embeddings, device=self._device)
        encodings.scatter_(1, original_indices.unsqueeze(1), 1)
        
        # 量化
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # 计算损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # 更新统计（训练时）
        if self.training:
            # 更新使用计数
            batch_usage = torch.zeros(self._num_embeddings, device=self._device)
            unique_indices, counts = torch.unique(original_indices, return_counts=True)
            batch_usage[unique_indices] = counts.float()
            
            # EMA更新使用统计
            self._usage_count = self._decay * self._usage_count + (1 - self._decay) * batch_usage
            self._total_usage = self._decay * self._total_usage + (1 - self._decay) * len(flat_input)
            
            # 更新EMA权重
            self._ema_cluster_size = self._decay * self._ema_cluster_size + (1 - self._decay) * torch.sum(encodings, dim=0)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = self._decay * self._ema_w + (1 - self._decay) * dw
            
            # 更新嵌入权重（只更新使用过的）
            n = torch.sum(self._ema_cluster_size)
            updated_cluster_size = (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n
            
            mask = self._ema_cluster_size > 0
            self._embedding.weight.data[mask] = self._ema_w[mask] / updated_cluster_size[mask].unsqueeze(1)
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # 计算perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return vq_loss, quantized, perplexity, encodings, original_indices
    
    def adaptive_update(self):
        """自适应更新码本（剪枝和合并）"""
        self.epoch_count += 1
        
        # 只在特定间隔检查
        if self.epoch_count % self.prune_check_interval != 0:
            return False
        
        with torch.no_grad():
            # 计算使用率
            if self._total_usage > 0:
                usage_rates = self._usage_count / self._total_usage
            else:
                usage_rates = torch.zeros_like(self._usage_count)
            
            active_indices = torch.where(self._active_codes)[0]
            n_active_before = len(active_indices)
            
            # 1. 剪枝低使用率码本
            low_usage_mask = usage_rates < self.min_usage_rate
            to_prune = low_usage_mask & self._active_codes
            
            # 计算有效使用的码本数（基于perplexity的估计）
            avg_usage = usage_rates[self._active_codes]
            if len(avg_usage) > 0:
                perplexity = torch.exp(-torch.sum(avg_usage * torch.log(avg_usage + 1e-10)))
                estimated_clusters = int(perplexity.item())
            else:
                estimated_clusters = 2
            
            if torch.any(to_prune):
                # 保留至少estimated_clusters个码本，但不少于2个
                n_to_keep = max(2, min(estimated_clusters, torch.sum(~to_prune).item()))
                if n_to_keep < len(active_indices):
                    # 只保留使用率最高的码本
                    _, top_indices = torch.topk(usage_rates, n_to_keep)
                    new_active = torch.zeros_like(self._active_codes)
                    new_active[top_indices] = True
                    self._active_codes = new_active
                    
                    # 重置被剪枝码本的统计
                    pruned_mask = ~self._active_codes
                    self._usage_count[pruned_mask] = 0
                    self._ema_cluster_size[pruned_mask] = 0
                    self._ema_w[pruned_mask] = 0
            
            # 2. 合并相似码本（可选）
            active_indices = torch.where(self._active_codes)[0]
            if len(active_indices) > 2:
                active_embeddings = self._embedding.weight[active_indices]
                
                # 计算码本间距离
                distances = torch.cdist(active_embeddings, active_embeddings)
                
                # 找到最相似的码本对
                distances.fill_diagonal_(float('inf'))
                min_dist, min_idx = torch.min(distances.view(-1), dim=0)
                
                if min_dist < self.merge_similarity_threshold:
                    # 合并最相似的两个码本
                    i, j = min_idx // len(active_indices), min_idx % len(active_indices)
                    idx_i, idx_j = active_indices[i], active_indices[j]
                    
                    # 保留使用率更高的，合并到它
                    if self._usage_count[idx_i] >= self._usage_count[idx_j]:
                        keep_idx, remove_idx = idx_i, idx_j
                    else:
                        keep_idx, remove_idx = idx_j, idx_i
                    
                    # 合并嵌入（加权平均）
                    weight_keep = self._usage_count[keep_idx] / (self._usage_count[keep_idx] + self._usage_count[remove_idx] + 1e-10)
                    weight_remove = 1 - weight_keep
                    
                    self._embedding.weight.data[keep_idx] = (
                        weight_keep * self._embedding.weight.data[keep_idx] +
                        weight_remove * self._embedding.weight.data[remove_idx]
                    )
                    
                    # 合并统计
                    self._usage_count[keep_idx] += self._usage_count[remove_idx]
                    self._ema_cluster_size[keep_idx] += self._ema_cluster_size[remove_idx]
                    
                    # 移除被合并的码本
                    self._active_codes[remove_idx] = False
            
            # 检查是否稳定
            n_active_after = torch.sum(self._active_codes).item()
            
            if n_active_after == self.last_num_active:
                self.stable_count += 1
            else:
                self.stable_count = 0
                print(f"Epoch {self.epoch_count}: {n_active_before} -> {n_active_after} active codes")
            
            self.last_num_active = n_active_after
            
            return self.stable_count >= self.stable_epochs // self.prune_check_interval
    
    def get_active_codes(self):
        """获取活跃码本"""
        active_indices = torch.where(self._active_codes)[0]
        return self._embedding.weight[active_indices], active_indices


def train_adaptive_vqvae(data, encoder, decoder, n_epochs=500, batch_size=32):
    """训练自适应VQ-VAE"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建序列数据
    def create_sequences(data, seq_len=10):
        sequences = []
        for i in range(len(data) - seq_len + 1):
            sequences.append(data[i:i+seq_len])
        return torch.stack(sequences)
    
    sequences = create_sequences(data, seq_len=10).to(device)
    
    # 初始化自适应VQ层
    vq_layer = AdaptiveVectorQuantizerEMA(
        embedding_dim=2,
        commitment_cost=0.25,
        decay=0.99,
        device=device,
        min_usage_rate=0.02,  # 2%的最小使用率
        merge_similarity_threshold=0.3,  # 更积极地合并
        prune_check_interval=30,
        stable_epochs=60
    ).to(device)
    
    # 用数据初始化码本
    flat_data = sequences.reshape(-1, 2)
    n_init = vq_layer.initialize_from_data(flat_data)
    
    # 优化器
    optimizer = optim.Adam(
        list(encoder.parameters()) + 
        list(decoder.parameters()) + 
        list(vq_layer.parameters()),
        lr=1e-3
    )
    
    # 训练历史
    history = {
        'total_loss': [],
        'vq_loss': [],
        'recon_loss': [],
        'perplexity': [],
        'n_active_codes': [],
        'pruning_events': []
    }
    
    for epoch in range(n_epochs):
        total_loss_epoch = 0
        vq_loss_epoch = 0
        recon_loss_epoch = 0
        perplexity_epoch = 0
        n_batches = 0
        
        # 训练模式
        encoder.train()
        decoder.train()
        vq_layer.train()
        
        # 批次训练
        indices = torch.randperm(len(sequences))
        for i in range(0, len(sequences), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = sequences[batch_indices]
            
            # 前向传播
            z_e = encoder(batch.reshape(-1, 2))
            vq_loss, z_q, perplexity, _, _ = vq_layer(z_e)
            x_recon = decoder(z_q)
            
            # 重建损失
            recon_loss = F.mse_loss(x_recon, batch.reshape(-1, 2))
            
            # 总损失
            loss = vq_loss + recon_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录
            total_loss_epoch += loss.item()
            vq_loss_epoch += vq_loss.item()
            recon_loss_epoch += recon_loss.item()
            perplexity_epoch += perplexity.item()
            n_batches += 1
        
        # 自适应更新
        is_stable = vq_layer.adaptive_update()
        
        # 记录历史
        n_active = torch.sum(vq_layer._active_codes).item()
        history['total_loss'].append(total_loss_epoch / n_batches)
        history['vq_loss'].append(vq_loss_epoch / n_batches)
        history['recon_loss'].append(recon_loss_epoch / n_batches)
        history['perplexity'].append(perplexity_epoch / n_batches)
        history['n_active_codes'].append(n_active)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] "
                  f"Loss: {history['total_loss'][-1]:.4f}, "
                  f"Active Codes: {n_active}, "
                  f"Perplexity: {history['perplexity'][-1]:.2f}")
        
        # 检查是否稳定
        if is_stable:
            print(f"\nCodebook size stabilized at {n_active} codes")
            print(f"Training stopped at epoch {epoch+1}")
            break
    
    return vq_layer, history


def visualize_results(data, labels, vq_layer, encoder, history):
    """可视化结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 码本数量变化
    ax = axes[0, 0]
    ax.plot(history['n_active_codes'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Active Codes')
    ax.set_title('Active Codebook Size Evolution')
    ax.grid(True)
    
    # 标记剪枝事件
    for i, event in enumerate(history.get('pruning_events', [])):
        ax.axvline(x=event, color='red', linestyle='--', alpha=0.5)
    
    # 2. 训练损失
    ax = axes[0, 1]
    ax.plot(history['total_loss'], label='Total', linewidth=2)
    ax.plot(history['vq_loss'], label='VQ', linewidth=2)
    ax.plot(history['recon_loss'], label='Recon', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True)
    
    # 3. Perplexity
    ax = axes[0, 2]
    ax.plot(history['perplexity'], linewidth=2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity')
    ax.set_title('Codebook Usage (Perplexity)')
    ax.grid(True)
    
    # 4. 原始数据
    ax = axes[1, 0]
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', s=30, alpha=0.6)
    ax.set_title('Original Data (True Clusters)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(scatter, ax=ax)
    
    # 5. 最终码本和分配
    ax = axes[1, 1]
    
    # 获取活跃码本
    active_embeddings, active_indices = vq_layer.get_active_codes()
    active_embeddings = active_embeddings.detach().cpu().numpy()
    
    # 编码数据点
    with torch.no_grad():
        vq_layer.eval()
        encoder.eval()
        z_e = encoder(data)
        _, z_q, _, _, indices = vq_layer(z_e)
        indices = indices.cpu().numpy()
    
    # 绘制数据点（按分配的码本着色）
    ax.scatter(data[:, 0], data[:, 1], c=indices, cmap='tab20', s=30, alpha=0.6)
    
    # 绘制码本
    ax.scatter(active_embeddings[:, 0], active_embeddings[:, 1], 
              c='black', marker='*', s=200, edgecolors='red', linewidths=2)
    
    ax.set_title(f'Final Active Codes: {len(active_embeddings)}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # 6. 使用率分布
    ax = axes[1, 2]
    usage_rates = vq_layer._usage_count[vq_layer._active_codes] / (vq_layer._total_usage + 1e-10)
    usage_rates = usage_rates.detach().cpu().numpy()
    
    ax.bar(range(len(usage_rates)), sorted(usage_rates, reverse=True))
    ax.set_xlabel('Code Index (sorted)')
    ax.set_ylabel('Usage Rate')
    ax.set_title('Code Usage Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adaptive_vqvae_fixed.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return fig


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=== Improved Adaptive VQ-VAE ===\n")
    
    # 生成测试数据
    def generate_2d_clusters(n_points=700, n_clusters=7, noise=0.3):
        points_per_cluster = n_points // n_clusters
        data = []
        labels = []
        
        # 生成更分散的聚类中心
        angles = np.linspace(0, 2*np.pi, n_clusters, endpoint=False)
        centers = np.array([[2*np.cos(a), 2*np.sin(a)] for a in angles])
        
        for i in range(n_clusters):
            cluster_data = np.random.randn(points_per_cluster, 2) * noise + centers[i]
            data.append(cluster_data)
            labels.extend([i] * points_per_cluster)
        
        data = np.vstack(data)
        labels = np.array(labels)
        
        # 打乱
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]
        
        return torch.FloatTensor(data), labels
    
    # 生成数据
    data, labels = generate_2d_clusters(n_points=700, n_clusters=7, noise=0.3)
    
    # 创建编码器和解码器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    ).to(device)
    
    decoder = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    ).to(device)
    
    # 训练
    print("Training adaptive VQ-VAE...")
    vq_layer, history = train_adaptive_vqvae(data, encoder, decoder, n_epochs=500)
    
    # 可视化
    print("\nGenerating visualization...")
    visualize_results(data, labels, vq_layer, encoder, history)
    
    # 最终统计
    n_final = torch.sum(vq_layer._active_codes).item()
    print(f"\n{'='*50}")
    print(f"Final Statistics:")
    print(f"  True clusters: {len(np.unique(labels))}")
    print(f"  Initial codes: {vq_layer._num_embeddings}")
    print(f"  Final active codes: {n_final}")
    print(f"  Reduction: {(1 - n_final/vq_layer._num_embeddings)*100:.1f}%")
    print(f"  Final perplexity: {history['perplexity'][-1]:.2f}")


if __name__ == "__main__":
    main()