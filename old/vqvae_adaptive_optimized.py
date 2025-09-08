import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.patches as mpatches


class OptimizedAdaptiveVQVAE(nn.Module):
    def __init__(
        self,
        embedding_dim,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
        device='cpu',
        target_perplexity_ratio=0.7,  # 目标perplexity比率
    ):
        super().__init__()
        
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._epsilon = epsilon
        self._device = device
        self.target_perplexity_ratio = target_perplexity_ratio
        
        self._num_embeddings = None
        self._embedding = None
        
        # 跟踪统计
        self.usage_history = []
        self.prune_history = []
    
    def initialize_from_data(self, data, n_clusters_hint=None):
        """使用数据初始化，可以提供聚类数量提示"""
        if n_clusters_hint is not None:
            # 如果有提示，初始化为提示数量的2倍（留余地剪枝）
            self._num_embeddings = int(n_clusters_hint * 2)
        else:
            # 否则使用sqrt规则
            self._num_embeddings = min(int(np.sqrt(len(data)) * 0.5), 20)
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        
        # 使用K-means++初始化，确保码本分布良好
        data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
        kmeans = KMeans(n_clusters=self._num_embeddings, init='k-means++', 
                       random_state=42, n_init=10, max_iter=300)
        kmeans.fit(data_np)
        
        # 设置初始码本
        self._embedding.weight.data = torch.FloatTensor(kmeans.cluster_centers_).to(self._device)
        
        # 添加小的随机噪声防止相同初始化
        self._embedding.weight.data += torch.randn_like(self._embedding.weight.data) * 0.01
        
        # 初始化EMA统计
        self.register_buffer('_ema_cluster_size', torch.zeros(self._num_embeddings))
        self.register_buffer('_ema_w', torch.zeros(self._num_embeddings, self._embedding_dim))
        self.register_buffer('_active_mask', torch.ones(self._num_embeddings, dtype=torch.bool))
        
        # 使用统计
        self.register_buffer('_usage_count', torch.zeros(self._num_embeddings))
        self.register_buffer('_total_count', torch.tensor(0.0))
        
        print(f"Initialized with {self._num_embeddings} codes")
        return self._num_embeddings
    
    def forward(self, inputs):
        # 确保形状正确
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # 获取活跃码本
        active_indices = torch.where(self._active_mask)[0]
        if len(active_indices) == 0:
            raise RuntimeError("No active codes!")
        
        active_embeddings = self._embedding.weight[active_indices]
        
        # 计算距离
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True) +
            torch.sum(active_embeddings**2, dim=1) -
            2 * torch.matmul(flat_input, active_embeddings.t())
        )
        
        # 找最近的码本
        encoding_indices = torch.argmin(distances, dim=1)
        original_indices = active_indices[encoding_indices]
        
        # One-hot编码
        encodings = torch.zeros(len(flat_input), self._num_embeddings, device=self._device)
        encodings.scatter_(1, original_indices.unsqueeze(1), 1)
        
        # 量化
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # 计算损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # 训练时更新
        if self.training:
            # 统计使用
            batch_usage = torch.zeros(self._num_embeddings, device=self._device)
            unique, counts = torch.unique(original_indices, return_counts=True)
            batch_usage[unique] = counts.float()
            
            self._usage_count = self._decay * self._usage_count + (1 - self._decay) * batch_usage
            self._total_count = self._decay * self._total_count + (1 - self._decay) * len(flat_input)
            
            # EMA更新
            self._ema_cluster_size = self._decay * self._ema_cluster_size + (1 - self._decay) * torch.sum(encodings, dim=0)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = self._decay * self._ema_w + (1 - self._decay) * dw
            
            # 更新嵌入
            n = torch.sum(self._ema_cluster_size)
            updated_cluster_size = (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n
            
            mask = self._ema_cluster_size > self._epsilon
            self._embedding.weight.data[mask] = self._ema_w[mask] / updated_cluster_size[mask].unsqueeze(1)
        
        # Straight-through
        quantized = inputs + (quantized - inputs).detach()
        
        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return vq_loss, quantized, perplexity, encodings, original_indices
    
    def adaptive_prune(self, epoch, interval=30):
        """自适应剪枝"""
        if epoch % interval != 0 or epoch < 50:  # 前50个epoch不剪枝
            return False
        
        with torch.no_grad():
            # 计算使用率
            if self._total_count > 0:
                usage_rates = self._usage_count / self._total_count
            else:
                return False
            
            # 找出低使用率的码本
            active_indices = torch.where(self._active_mask)[0]
            n_active = len(active_indices)
            
            if n_active <= 3:  # 保持至少3个码本
                return False
            
            # 计算使用率阈值（保留使用率前70%的码本）
            active_usage = usage_rates[self._active_mask]
            sorted_usage, _ = torch.sort(active_usage, descending=True)
            
            # 动态阈值：保留累积使用率达到95%的码本
            cumsum_usage = torch.cumsum(sorted_usage, dim=0)
            cumsum_usage = cumsum_usage / cumsum_usage[-1]
            n_keep = torch.where(cumsum_usage >= 0.95)[0]
            
            if len(n_keep) > 0:
                n_keep = max(3, n_keep[0].item() + 1)  # 至少保留3个
            else:
                n_keep = n_active
            
            if n_keep < n_active:
                # 保留使用率最高的n_keep个码本
                _, top_indices = torch.topk(usage_rates, n_keep)
                new_active = torch.zeros_like(self._active_mask)
                new_active[top_indices] = True
                
                n_pruned = n_active - n_keep
                self._active_mask = new_active
                
                # 清理被剪枝的码本
                pruned_mask = ~self._active_mask
                self._usage_count[pruned_mask] = 0
                self._ema_cluster_size[pruned_mask] = 0
                
                print(f"Epoch {epoch}: Pruned {n_pruned} codes, {n_keep} remaining")
                self.prune_history.append(epoch)
                return True
        
        return False
    
    def get_active_codebook(self):
        """获取活跃码本"""
        active_indices = torch.where(self._active_mask)[0]
        return self._embedding.weight[active_indices], active_indices


def create_voronoi_visualization(data, labels, vq_model, encoder, decoder, history):
    """创建包含Voronoi区域的可视化"""
    device = data.device
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 码本数量变化
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(history['n_active'], linewidth=2, color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Active Codes')
    ax1.set_title('Codebook Size Evolution')
    ax1.grid(True, alpha=0.3)
    
    # 标记剪枝事件
    for event in vq_model.prune_history:
        ax1.axvline(x=event, color='red', linestyle='--', alpha=0.5)
    
    # 2. 训练损失
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(history['total_loss'], label='Total', linewidth=2)
    ax2.plot(history['vq_loss'], label='VQ', linewidth=2)
    ax2.plot(history['recon_loss'], label='Recon', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Losses')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Perplexity
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(history['perplexity'], linewidth=2, color='green')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Perplexity')
    ax3.set_title(f'Codebook Usage (Final: {history["perplexity"][-1]:.2f})')
    ax3.grid(True, alpha=0.3)
    
    # 4. 原始数据（真实聚类）
    ax4 = plt.subplot(3, 3, 4)
    scatter = ax4.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', s=30, alpha=0.7)
    ax4.set_title(f'Original Data ({len(np.unique(labels))} True Clusters)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    plt.colorbar(scatter, ax=ax4)
    
    # 5. 编码空间中的数据和码本
    ax5 = plt.subplot(3, 3, 5)
    
    with torch.no_grad():
        vq_model.eval()
        encoder.eval()
        
        # 编码数据
        z_e = encoder(data)
        _, z_q, _, _, indices = vq_model(z_e)
        z_e = z_e.cpu().numpy()
        indices = indices.cpu().numpy()
        
        # 获取活跃码本
        active_codebook, active_indices = vq_model.get_active_codebook()
        active_codebook = active_codebook.cpu().numpy()
        
        # 绘制编码后的数据点
        scatter = ax5.scatter(z_e[:, 0], z_e[:, 1], c=indices, cmap='tab20', s=30, alpha=0.7)
        
        # 绘制码本
        ax5.scatter(active_codebook[:, 0], active_codebook[:, 1], 
                   c='red', s=200, marker='*', edgecolors='black', linewidth=2)
        
        for i, pos in enumerate(active_codebook):
            ax5.annotate(f'{i}', pos, fontsize=8, ha='center', va='center', color='white')
    
    ax5.set_title(f'Encoded Space ({len(active_codebook)} Active Codes)')
    ax5.set_xlabel('Encoded X')
    ax5.set_ylabel('Encoded Y')
    plt.colorbar(scatter, ax=ax5)
    
    # 6. Voronoi区域（编码空间）
    ax6 = plt.subplot(3, 3, 6)
    
    # 创建网格
    x_min, x_max = z_e[:, 0].min() - 0.5, z_e[:, 0].max() + 0.5
    y_min, y_max = z_e[:, 1].min() - 0.5, z_e[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    
    # 计算Voronoi区域
    distances = torch.cdist(grid_points, torch.FloatTensor(active_codebook))
    nearest_codebook = distances.argmin(dim=1).numpy().reshape(xx.shape)
    
    # 绘制Voronoi区域
    contour = ax6.contourf(xx, yy, nearest_codebook, levels=len(active_codebook)-1, 
                           cmap='tab20', alpha=0.3)
    
    # 添加数据点
    ax6.scatter(z_e[:, 0], z_e[:, 1], c='gray', alpha=0.2, s=10)
    
    # 添加码本
    ax6.scatter(active_codebook[:, 0], active_codebook[:, 1], 
               c='red', s=200, marker='*', edgecolors='black', linewidth=2)
    
    for i, pos in enumerate(active_codebook):
        ax6.annotate(f'{i}', pos, fontsize=10, ha='center', va='center', 
                    color='white', weight='bold')
    
    ax6.set_title('Voronoi Regions (Encoded Space)')
    ax6.set_xlabel('Encoded X')
    ax6.set_ylabel('Encoded Y')
    
    # 7. 原始空间的聚类结果
    ax7 = plt.subplot(3, 3, 7)
    
    # 解码码本到原始空间
    with torch.no_grad():
        decoder.eval()
        decoded_codebook = decoder(torch.FloatTensor(active_codebook).to(device))
        decoded_codebook = decoded_codebook.cpu().numpy()
    
    # 绘制数据点（按VQ分配着色）
    scatter = ax7.scatter(data[:, 0], data[:, 1], c=indices, cmap='tab20', s=30, alpha=0.7)
    
    # 绘制解码后的码本
    ax7.scatter(decoded_codebook[:, 0], decoded_codebook[:, 1], 
               c='red', s=200, marker='*', edgecolors='black', linewidth=2)
    
    ax7.set_title('VQ Clustering (Original Space)')
    ax7.set_xlabel('X')
    ax7.set_ylabel('Y')
    plt.colorbar(scatter, ax=ax7)
    
    # 8. 原始空间的Voronoi区域
    ax8 = plt.subplot(3, 3, 8)
    
    # 创建原始空间的网格
    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    grid_points_orig = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    
    # 通过编码器处理网格点
    with torch.no_grad():
        z_grid = encoder(grid_points_orig)
        _, _, _, _, grid_indices = vq_model(z_grid)
        grid_indices = grid_indices.cpu().numpy().reshape(xx.shape)
    
    # 绘制Voronoi区域
    contour = ax8.contourf(xx, yy, grid_indices, levels=len(active_codebook)-1, 
                           cmap='tab20', alpha=0.3)
    
    # 添加数据点
    ax8.scatter(data[:, 0], data[:, 1], c='gray', alpha=0.2, s=10)
    
    # 添加解码后的码本
    ax8.scatter(decoded_codebook[:, 0], decoded_codebook[:, 1], 
               c='red', s=200, marker='*', edgecolors='black', linewidth=2)
    
    for i in range(len(decoded_codebook)):
        ax8.annotate(f'{i}', decoded_codebook[i], fontsize=10, 
                    ha='center', va='center', color='white', weight='bold')
    
    ax8.set_title('Voronoi Regions (Original Space)')
    ax8.set_xlabel('X')
    ax8.set_ylabel('Y')
    
    # 9. 聚类质量指标
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # 计算聚类指标（只在有多个聚类时计算）
    n_unique_indices = len(np.unique(indices))
    if n_unique_indices > 1:
        sil_score = silhouette_score(data.cpu().numpy(), indices)
    else:
        sil_score = 0.0
    ch_score = calinski_harabasz_score(data.cpu().numpy(), indices)
    
    metrics_text = f"""
    === Clustering Metrics ===
    
    True Clusters: {len(np.unique(labels))}
    VQ Codes: {len(active_codebook)}
    
    Silhouette Score: {sil_score:.3f}
    Calinski-Harabasz: {ch_score:.1f}
    
    Final VQ Loss: {history['vq_loss'][-1]:.4f}
    Final Perplexity: {history['perplexity'][-1]:.2f}
    
    Initial Codes: {vq_model._num_embeddings}
    Reduction: {(1 - len(active_codebook)/vq_model._num_embeddings)*100:.1f}%
    """
    
    ax9.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('Adaptive VQ-VAE with Voronoi Visualization', fontsize=14, y=0.98)
    plt.tight_layout()
    
    return fig


def train_optimized_vqvae(data, labels, n_epochs=300, n_clusters_hint=None):
    """训练优化的自适应VQ-VAE"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    
    # 创建更简单的编码器和解码器（避免过度变换）
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
    
    # 创建VQ模型
    vq_model = OptimizedAdaptiveVQVAE(
        embedding_dim=2,
        commitment_cost=0.25,
        decay=0.99,
        device=device,
        target_perplexity_ratio=0.8  # 目标是perplexity的80%作为码本数
    ).to(device)
    
    # 初始化
    vq_model.initialize_from_data(data, n_clusters_hint=n_clusters_hint)
    
    # 优化器（降低学习率）
    optimizer = optim.Adam(
        list(encoder.parameters()) + 
        list(decoder.parameters()) + 
        list(vq_model.parameters()),
        lr=1e-3
    )
    
    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # 训练历史
    history = {
        'total_loss': [],
        'vq_loss': [],
        'recon_loss': [],
        'perplexity': [],
        'n_active': []
    }
    
    # 创建序列数据
    def create_sequences(data, seq_len=10):
        sequences = []
        for i in range(len(data) - seq_len + 1):
            sequences.append(data[i:i+seq_len])
        return torch.stack(sequences)
    
    sequences = create_sequences(data, seq_len=10)
    
    print("\nTraining...")
    stable_count = 0
    last_n_active = vq_model._num_embeddings
    
    for epoch in range(n_epochs):
        # 训练模式
        encoder.train()
        decoder.train()
        vq_model.train()
        
        # 批次训练
        batch_size = 32
        total_loss = 0
        vq_loss_sum = 0
        recon_loss_sum = 0
        perplexity_sum = 0
        n_batches = 0
        
        indices = torch.randperm(len(sequences))
        for i in range(0, len(sequences), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch = sequences[batch_idx]
            
            # 前向传播
            z_e = encoder(batch.reshape(-1, 2))
            vq_loss, z_q, perplexity, _, _ = vq_model(z_e)
            x_recon = decoder(z_q)
            
            # 损失
            recon_loss = F.mse_loss(x_recon, batch.reshape(-1, 2))
            loss = vq_loss + recon_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            
            optimizer.step()
            
            # 记录
            total_loss += loss.item()
            vq_loss_sum += vq_loss.item()
            recon_loss_sum += recon_loss.item()
            perplexity_sum += perplexity.item()
            n_batches += 1
        
        # 自适应剪枝
        vq_model.adaptive_prune(epoch, interval=30)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        n_active = torch.sum(vq_model._active_mask).item()
        history['total_loss'].append(total_loss / n_batches)
        history['vq_loss'].append(vq_loss_sum / n_batches)
        history['recon_loss'].append(recon_loss_sum / n_batches)
        history['perplexity'].append(perplexity_sum / n_batches)
        history['n_active'].append(n_active)
        
        # 打印进度
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] "
                  f"Loss: {history['total_loss'][-1]:.4f}, "
                  f"Active: {n_active}, "
                  f"Perplexity: {history['perplexity'][-1]:.2f}")
        
        # 检查稳定性
        if n_active == last_n_active:
            stable_count += 1
        else:
            stable_count = 0
        
        last_n_active = n_active
        
        # 早停
        if stable_count >= 30:
            print(f"\nConverged at epoch {epoch+1} with {n_active} codes")
            break
    
    return vq_model, encoder, decoder, history


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=== Optimized Adaptive VQ-VAE with Voronoi Visualization ===\n")
    
    # 生成测试数据
    def generate_2d_clusters(n_points=700, n_clusters=7, noise=0.25):
        points_per_cluster = n_points // n_clusters
        data = []
        labels = []
        
        # 生成分散的聚类中心
        angles = np.linspace(0, 2*np.pi, n_clusters, endpoint=False)
        radius = 2.5
        centers = np.array([[radius*np.cos(a), radius*np.sin(a)] for a in angles])
        
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
    
    # 生成数据
    print("Generating data with 7 clusters...")
    data, labels = generate_2d_clusters(n_points=700, n_clusters=7, noise=0.25)
    
    # 训练模型（提供聚类数量提示）
    vq_model, encoder, decoder, history = train_optimized_vqvae(
        data, labels, n_epochs=300, n_clusters_hint=7
    )
    
    # 创建可视化
    print("\nCreating visualization...")
    fig = create_voronoi_visualization(data, labels, vq_model, encoder, decoder, history)
    
    # 保存结果
    plt.savefig('vqvae_adaptive_optimized.png', dpi=150, bbox_inches='tight')
    print("Results saved to vqvae_adaptive_optimized.png")
    
    plt.show()


if __name__ == "__main__":
    main()