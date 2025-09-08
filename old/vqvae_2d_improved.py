import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os
from sklearn.cluster import KMeans

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.vector_quantizer_ema import VectorQuantizerEMA


def generate_2d_clusters(n_points=500, n_clusters=5, noise=0.3):
    """生成二维聚类数据"""
    np.random.seed(42)
    
    # 生成聚类中心
    angles = np.linspace(0, 2*np.pi, n_clusters, endpoint=False)
    centers = np.array([[np.cos(angle)*3, np.sin(angle)*3] for angle in angles])
    
    # 生成数据点
    data = []
    labels = []
    points_per_cluster = n_points // n_clusters
    
    for i, center in enumerate(centers):
        # 为每个聚类生成高斯分布的点
        cluster_points = np.random.randn(points_per_cluster, 2) * noise + center
        data.append(cluster_points)
        labels.extend([i] * points_per_cluster)
    
    data = np.vstack(data)
    labels = np.array(labels)
    
    # 打乱数据
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]
    
    return torch.FloatTensor(data), torch.LongTensor(labels)


def create_sequence_data(data, seq_length=10):
    """将二维点转换为序列数据格式 (batch, seq_length, dim)"""
    n_points = len(data)
    n_sequences = n_points // seq_length
    
    # 重塑为序列格式
    sequences = data[:n_sequences * seq_length].reshape(n_sequences, seq_length, 2)
    return sequences


class ImprovedVectorQuantizerEMA(VectorQuantizerEMA):
    """改进的VQ-EMA，添加码本重置机制和斥力机制"""
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, device, 
                 epsilon=1e-5, reset_threshold=0.01, reset_interval=50, 
                 repulsion_strength=0.1, min_distance=0.5):
        super().__init__(num_embeddings, embedding_dim, commitment_cost, decay, device, epsilon)
        self.reset_threshold = reset_threshold
        self.reset_interval = reset_interval
        self.register_buffer('_ema_usage', torch.zeros(num_embeddings))
        self.step_count = 0
        self.repulsion_strength = repulsion_strength  # 斥力强度
        self.min_distance = min_distance  # 码本之间的最小距离
        
    def forward(self, inputs, compute_distances_if_possible=True, record_codebook_stats=False):
        # 调用父类的forward方法
        result = super().forward(inputs, compute_distances_if_possible, record_codebook_stats)
        
        # 在训练时更新使用统计和检查是否需要重置
        if self.training:
            self.step_count += 1
            vq_loss, quantized, perplexity, encodings, distances, encoding_indices = result[:6]
            
            # 更新使用统计
            # encodings shape: (batch_size * seq_length, num_embeddings)
            batch_usage = torch.sum(encodings, dim=0)
            # 确保batch_usage是一维的
            if batch_usage.dim() > 1:
                batch_usage = batch_usage.sum(dim=1) if batch_usage.shape[0] == self._num_embeddings else batch_usage.sum(dim=0)
            if batch_usage.shape[0] != self._num_embeddings:
                # 如果还是不匹配，创建正确大小的张量
                correct_usage = torch.zeros(self._num_embeddings, device=batch_usage.device)
                correct_usage[:min(batch_usage.shape[0], self._num_embeddings)] = batch_usage[:min(batch_usage.shape[0], self._num_embeddings)]
                batch_usage = correct_usage
            self._ema_usage = self._ema_usage * self._decay + (1 - self._decay) * batch_usage
            
            # 应用码本之间的斥力，防止过度聚集
            self.apply_codebook_repulsion()
            
            # 定期检查并重置未使用的码本
            if self.step_count % self.reset_interval == 0:
                self.reset_dead_codes(inputs)
        
        return result
    
    def apply_codebook_repulsion(self):
        """应用码本之间的斥力，使码本向外扩散"""
        with torch.no_grad():
            embeddings = self._embedding.weight.data
            n = embeddings.shape[0]
            
            # 计算码本中心（用于向外扩散）
            center = embeddings.mean(dim=0, keepdim=True)
            
            # 计算所有码本对之间的距离
            distances = torch.cdist(embeddings, embeddings)
            
            # 对每个码本计算斥力
            for i in range(n):
                # 1. 计算其他码本对当前码本的斥力
                repulsion_force = torch.zeros_like(embeddings[i])
                
                for j in range(n):
                    if i != j:
                        dist = distances[i, j]
                        if dist < self.min_distance and dist > 0:
                            # 斥力方向：从j指向i
                            direction = embeddings[i] - embeddings[j]
                            direction = direction / (dist + 1e-8)
                            
                            # 斥力大小：距离越近，斥力越大
                            force_magnitude = self.repulsion_strength * (self.min_distance - dist) / self.min_distance
                            repulsion_force += direction * force_magnitude
                
                # 2. 添加向外扩散的力（从中心向外）
                outward_direction = embeddings[i] - center[0]
                outward_norm = torch.norm(outward_direction)
                if outward_norm > 0:
                    outward_direction = outward_direction / outward_norm
                    # 向外扩散的力，帮助码本分散到数据边缘
                    outward_force = outward_direction * self.repulsion_strength * 0.5
                    repulsion_force += outward_force
                
                # 应用组合力，移动码本
                embeddings[i] += repulsion_force
            
            # 更新EMA权重
            self._ema_w.data = embeddings.clone()
    
    def reset_dead_codes(self, inputs):
        """重置使用率低的码本"""
        # 找出使用率低的码本
        dead_codes = self._ema_usage < self.reset_threshold
        
        if dead_codes.any():
            # 将输入数据展平
            flat_input = inputs.permute(1, 2, 0).reshape(-1, self._embedding_dim)
            
            # 随机选择一些输入点来重新初始化死码本
            n_dead = dead_codes.sum().item()
            if n_dead > 0 and len(flat_input) > 0:
                # 随机选择输入点
                random_indices = torch.randperm(len(flat_input))[:n_dead]
                new_embeddings = flat_input[random_indices]
                
                # 重置死码本
                dead_indices = torch.where(dead_codes)[0]
                for i, idx in enumerate(dead_indices):
                    if i < len(new_embeddings):
                        self._embedding.weight.data[idx] = new_embeddings[i]
                        self._ema_w.data[idx] = new_embeddings[i]
                        self._ema_cluster_size[idx] = 1.0
                        self._ema_usage[idx] = 0.1  # 给一个小的初始使用值


def train_improved_vq(data, n_epochs=200, num_embeddings=8, learning_rate=1e-3):
    """使用改进的VQ-EMA进行训练"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建序列数据
    sequences = create_sequence_data(data, seq_length=10)
    sequences = sequences.to(device)
    batch_size, seq_length, dim = sequences.shape
    
    # 添加简单的编码器和解码器来提供梯度
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
    
    # 初始化改进的VQ层
    vq_layer = ImprovedVectorQuantizerEMA(
        num_embeddings=num_embeddings,
        embedding_dim=2,
        commitment_cost=0.25,
        decay=0.99,
        device=device,
        epsilon=1e-5,
        reset_threshold=0.01,  # 使用率低于1%的码本会被重置
        reset_interval=30,  # 每30步检查一次
        repulsion_strength=0.05,  # 增强斥力强度，让码本更分散
        min_distance=1.5  # 增大码本之间的最小期望距离
    ).to(device)
    
    # 使用K-means初始化码本，而不是随机初始化
    with torch.no_grad():
        # 将序列数据展平
        flat_data = sequences.reshape(-1, 2).cpu().numpy()
        
        # 使用K-means找到初始聚类中心
        kmeans = KMeans(n_clusters=num_embeddings, random_state=42, n_init=10)
        kmeans.fit(flat_data)
        initial_centers = torch.FloatTensor(kmeans.cluster_centers_).to(device)
        
        # 初始化码本为K-means中心
        vq_layer._embedding.weight.data = initial_centers
        vq_layer._ema_w.data = initial_centers.clone()
        
        # 初始化cluster size为均匀分布
        vq_layer._ema_cluster_size.data = torch.ones(num_embeddings) * (len(flat_data) / num_embeddings)
    
    # 优化器：优化编码器和解码器参数
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate
    )
    
    # 重建损失函数
    criterion = nn.MSELoss()
    
    train_history = {
        'total_loss': [],
        'vq_loss': [],
        'recon_loss': [],
        'perplexity': [],
        'codebook_positions': [],
        'input_positions': [],
        'usage_stats': []
    }
    
    for epoch in range(n_epochs):
        encoder.train()
        decoder.train()
        vq_layer.train()
        
        # 将数据reshape为 (batch*seq_length, dim) 以通过编码器
        flat_sequences = sequences.reshape(-1, 2)
        
        # 编码
        encoded = encoder(flat_sequences)
        encoded = encoded.reshape(batch_size, seq_length, 2).permute(2, 1, 0)  # (dim, seq_length, batch)
        
        # VQ前向传播
        vq_loss, quantized, perplexity, encodings, distances, encoding_indices, \
            vq_loss_dict, _, _, _, _ = vq_layer(encoded)
        
        # 解码
        quantized_permuted = quantized.permute(2, 1, 0).reshape(-1, 2)  # (batch*seq_length, dim)
        reconstructed = decoder(quantized_permuted)
        reconstructed = reconstructed.reshape(batch_size, seq_length, 2)
        
        # 计算重建损失
        reconstruction_loss = criterion(reconstructed, sequences)
        
        # 总损失 = VQ损失 + 重建损失（与原始代码一致）
        total_loss = vq_loss + reconstruction_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 记录历史
        train_history['total_loss'].append(total_loss.item())
        train_history['vq_loss'].append(vq_loss.item())
        train_history['recon_loss'].append(reconstruction_loss.item())
        train_history['perplexity'].append(perplexity.item())
        
        # 定期记录状态
        if epoch % 10 == 0:
            codebook_pos = vq_layer.embedding.weight.data.cpu().numpy().copy()
            train_history['codebook_positions'].append(codebook_pos)
            
            with torch.no_grad():
                train_history['input_positions'].append(sequences.reshape(-1, 2).cpu().numpy())
                
                # 记录使用统计
                usage = vq_layer._ema_usage.cpu().numpy().copy()
                train_history['usage_stats'].append(usage)
        
        if (epoch + 1) % 20 == 0:
            usage_info = vq_layer._ema_usage.cpu().numpy()
            active_codes = (usage_info > 0.01).sum()
            print(f"Epoch [{epoch+1}/{n_epochs}] "
                  f"Total: {total_loss.item():.4f}, "
                  f"VQ: {vq_loss.item():.4f}, "
                  f"Recon: {reconstruction_loss.item():.4f}, "
                  f"Perplexity: {perplexity.item():.2f}, "
                  f"Active: {active_codes}/{num_embeddings}")
    
    return encoder, decoder, vq_layer, train_history


def visualize_improved_clustering(data, labels, encoder, decoder, vq_layer, train_history):
    """可视化改进的VQ聚类结果"""
    device = next(vq_layer.parameters()).device
    
    # 准备数据
    sequences = create_sequence_data(data, seq_length=10)
    sequences = sequences.to(device)
    
    vq_layer.eval()
    
    with torch.no_grad():
        # 直接使用原始数据
        inputs_vq = sequences.permute(2, 1, 0)
        
        # VQ推理
        _, quantized, _, _, _, encoding_indices, _, _, _, _, _ = vq_layer(
            inputs_vq, compute_distances_if_possible=False
        )
        
        # 获取结果
        data_points = sequences.reshape(-1, 2).cpu().numpy()
        quantized_points = quantized.permute(2, 1, 0).reshape(-1, 2).cpu().numpy()
        cluster_assignments = encoding_indices.squeeze().cpu().numpy().flatten()
        codebook = vq_layer.embedding.weight.data.cpu().numpy()
        usage_stats = vq_layer._ema_usage.cpu().numpy()
    
    # 创建图形
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 训练损失
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(train_history['vq_loss'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('VQ Loss')
    ax1.set_title('Training VQ Loss')
    ax1.grid(True)
    
    # 2. Perplexity
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(train_history['perplexity'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title(f'Codebook Usage (Final: {train_history["perplexity"][-1]:.2f})')
    ax2.grid(True)
    
    # 3. 原始数据和真实标签
    ax3 = plt.subplot(3, 3, 3)
    scatter = ax3.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', 
                         alpha=0.6, s=20)
    ax3.set_title('Original Data (True Clusters)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(scatter, ax=ax3)
    
    # 4. 数据和VQ聚类结果
    ax4 = plt.subplot(3, 3, 4)
    scatter = ax4.scatter(data_points[:, 0], data_points[:, 1], 
                         c=cluster_assignments, cmap='tab10', alpha=0.6, s=20)
    
    # 根据使用率调整码本显示大小
    sizes = 100 + usage_stats * 1000  # 大小根据使用率调整
    ax4.scatter(codebook[:, 0], codebook[:, 1], c='red', s=sizes, 
               marker='*', edgecolors='black', linewidth=2, label='Codebook', alpha=0.8)
    
    for i, (pos, usage) in enumerate(zip(codebook, usage_stats)):
        if usage > 0.01:  # 只标注活跃的码本
            ax4.annotate(f'{i}', pos, fontsize=10, ha='center', va='center', color='white')
    
    ax4.set_title('Data & VQ Clustering (size ∝ usage)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.legend()
    plt.colorbar(scatter, ax=ax4)
    
    # 5. Voronoi图（只显示活跃码本）
    ax5 = plt.subplot(3, 3, 5)
    
    # 创建网格点
    x_min, x_max = data_points[:, 0].min() - 0.5, data_points[:, 0].max() + 0.5
    y_min, y_max = data_points[:, 1].min() - 0.5, data_points[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    
    # 只考虑活跃的码本
    active_indices = np.where(usage_stats > 0.01)[0]
    if len(active_indices) > 0:
        active_codebook = codebook[active_indices]
        distances = torch.cdist(grid_points, torch.FloatTensor(active_codebook))
        nearest_codebook_idx = distances.argmin(dim=1).numpy()
        nearest_codebook = active_indices[nearest_codebook_idx].reshape(xx.shape)
        
        ax5.contourf(xx, yy, nearest_codebook, levels=len(codebook)-1, 
                     cmap='tab10', alpha=0.3)
    
    ax5.scatter(data_points[:, 0], data_points[:, 1], 
               c='gray', alpha=0.2, s=10)
    
    # 只显示活跃的码本
    for i, (pos, usage) in enumerate(zip(codebook, usage_stats)):
        if usage > 0.01:
            ax5.scatter(pos[0], pos[1], c='red', s=200, 
                       marker='*', edgecolors='black', linewidth=2)
            ax5.annotate(f'{i}', pos, fontsize=10, ha='center', va='center', 
                        color='white', weight='bold')
    
    ax5.set_title('Active Voronoi Regions')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    
    # 6. 码本使用统计
    ax6 = plt.subplot(3, 3, 6)
    bars = ax6.bar(range(len(codebook)), usage_stats)
    colors = plt.cm.tab10(np.linspace(0, 1, len(codebook)))
    for i, (bar, usage) in enumerate(zip(bars, usage_stats)):
        if usage > 0.01:
            bar.set_color(colors[i])
        else:
            bar.set_color('lightgray')
    
    ax6.axhline(y=0.01, color='r', linestyle='--', label='Reset Threshold')
    ax6.set_xlabel('Codebook Index')
    ax6.set_ylabel('Usage Rate')
    ax6.set_title('Codebook Usage Distribution')
    ax6.set_xticks(range(len(codebook)))
    ax6.legend()
    ax6.grid(True, axis='y')
    
    # 7. 码本使用率演化
    ax7 = plt.subplot(3, 3, 7)
    if len(train_history['usage_stats']) > 0:
        usage_evolution = np.array(train_history['usage_stats']).T
        epochs = np.arange(0, len(train_history['usage_stats'])) * 10
        
        for i in range(len(codebook)):
            if usage_stats[i] > 0.01:  # 只显示活跃码本的演化
                ax7.plot(epochs, usage_evolution[i], label=f'Code {i}')
        
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Usage Rate')
        ax7.set_title('Codebook Usage Evolution')
        ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax7.grid(True)
    
    # 8. 聚类质量对比
    ax8 = plt.subplot(3, 3, 8)
    unique, counts = np.unique(cluster_assignments, return_counts=True)
    
    # 创建混淆矩阵风格的显示
    confusion = np.zeros((len(np.unique(labels)), len(codebook)))
    for true_label, pred_code in zip(labels[:len(cluster_assignments)], cluster_assignments):
        confusion[true_label, pred_code] += 1
    
    im = ax8.imshow(confusion, cmap='Blues', aspect='auto')
    ax8.set_xlabel('VQ Code Index')
    ax8.set_ylabel('True Cluster')
    ax8.set_title('Cluster Assignment Matrix')
    plt.colorbar(im, ax=ax8)
    
    # 9. 聚类指标
    ax9 = plt.subplot(3, 3, 9)
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    
    data_subset = data[:len(cluster_assignments)].numpy()
    labels_subset = labels[:len(cluster_assignments)].numpy()
    
    sil_score = silhouette_score(data_subset, cluster_assignments)
    ari_score = adjusted_rand_score(labels_subset, cluster_assignments)
    active_codes = (usage_stats > 0.01).sum()
    
    metrics_text = f"Clustering Metrics:\n\n"
    metrics_text += f"Active Codes: {active_codes}/{len(codebook)}\n"
    metrics_text += f"Silhouette Score: {sil_score:.3f}\n"
    metrics_text += f"Adjusted Rand Index: {ari_score:.3f}\n"
    metrics_text += f"Final Perplexity: {train_history['perplexity'][-1]:.2f}\n"
    metrics_text += f"Final VQ Loss: {train_history['vq_loss'][-1]:.4f}"
    
    ax9.text(0.1, 0.5, metrics_text, transform=ax9.transAxes, 
             fontsize=12, verticalalignment='center')
    ax9.axis('off')
    
    plt.tight_layout()
    plt.savefig('vqvae_improved_clustering.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to vqvae_improved_clustering.png")
    
    return fig


def main():
    """主函数"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=== Improved VQ-EMA 2D Clustering ===\n")
    
    # 1. 生成二维数据
    print("1. Generating 2D cluster data...")
    n_points = 700
    n_clusters = 7
    data, labels = generate_2d_clusters(n_points=n_points, n_clusters=n_clusters, noise=0.25)
    print(f"   Generated {n_points} points in {n_clusters} clusters")
    
    # 2. 训练改进的VQ-EMA
    print("\n2. Training Improved VQ-EMA...")
    num_embeddings = 7  # 使用稍多的码本，让算法自己选择
    n_epochs = 2500
    
    encoder, decoder, vq_layer, train_history = train_improved_vq(
        data,
        n_epochs=n_epochs,
        num_embeddings=num_embeddings,
        learning_rate=5e-3
    )
    
    # 3. 可视化结果
    print("\n3. Visualizing results...")
    fig = visualize_improved_clustering(data, labels, encoder, decoder, vq_layer, train_history)
    
    # 4. 最终分析
    print("\n=== Final Analysis ===")
    
    sequences = create_sequence_data(data, seq_length=10)
    with torch.no_grad():
        device = next(vq_layer.parameters()).device
        sequences = sequences.to(device)
        inputs_vq = sequences.permute(2, 1, 0)
        _, _, perplexity, _, _, encoding_indices, _, _, _, _, _ = vq_layer(
            inputs_vq, compute_distances_if_possible=False
        )
        cluster_assignments = encoding_indices.squeeze().cpu().numpy().flatten()
    
    from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
    
    data_subset = data[:len(cluster_assignments)].numpy()
    labels_subset = labels[:len(cluster_assignments)].numpy()
    
    sil_score = silhouette_score(data_subset, cluster_assignments)
    ari_score = adjusted_rand_score(labels_subset, cluster_assignments)
    nmi_score = normalized_mutual_info_score(labels_subset, cluster_assignments)
    
    usage_stats = vq_layer._ema_usage.cpu().numpy()
    active_codes = (usage_stats > 0.01).sum()
    
    print(f"Final VQ Loss: {train_history['vq_loss'][-1]:.4f}")
    print(f"Final Perplexity: {train_history['perplexity'][-1]:.2f}")
    print(f"Active Codes: {active_codes}/{num_embeddings}")
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Adjusted Rand Index: {ari_score:.3f}")
    print(f"Normalized Mutual Information: {nmi_score:.3f}")
    
    print("\nResults saved:")
    print("  - vqvae_improved_clustering.png")


if __name__ == "__main__":
    main()