import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.spatial.distance import cdist


class SmartAdaptiveVQVAE(nn.Module):
    """智能自适应VQ-VAE，基于聚类质量动态调整码本数量"""
    
    def __init__(
        self,
        embedding_dim,
        initial_codes=16,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
        device='cpu'
    ):
        super().__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = initial_codes
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._epsilon = epsilon
        self._device = device
        
        # 初始化嵌入
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        
        # EMA参数
        self.register_buffer('_ema_cluster_size', torch.zeros(self._num_embeddings))
        self.register_buffer('_ema_w', torch.zeros(self._num_embeddings, self._embedding_dim))
        
        # 活跃码本掩码
        self.register_buffer('_active_mask', torch.ones(self._num_embeddings, dtype=torch.bool))
        
        # 使用统计
        self.register_buffer('_usage_count', torch.zeros(self._num_embeddings))
        self.register_buffer('_batch_count', torch.tensor(0.0))
        
        # 历史记录
        self.merge_history = []
        self.quality_history = []
    
    def initialize_from_data(self, data):
        """使用K-means++初始化码本"""
        data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
        
        # 使用K-means++确保良好的初始分布
        kmeans = KMeans(n_clusters=self._num_embeddings, init='k-means++', 
                       random_state=42, n_init=10)
        kmeans.fit(data_np)
        
        # 设置初始码本
        self._embedding.weight.data = torch.FloatTensor(kmeans.cluster_centers_).to(self._device)
        
        print(f"Initialized {self._num_embeddings} codes using K-means++")
    
    def forward(self, inputs):
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
        
        # 更新统计（训练时）
        if self.training:
            # 更新使用计数
            batch_usage = torch.zeros(self._num_embeddings, device=self._device)
            for idx in original_indices:
                batch_usage[idx] += 1
            
            self._usage_count = self._decay * self._usage_count + (1 - self._decay) * batch_usage
            self._batch_count = self._batch_count + 1
            
            # EMA更新
            self._ema_cluster_size = self._decay * self._ema_cluster_size + (1 - self._decay) * torch.sum(encodings, dim=0)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = self._decay * self._ema_w + (1 - self._decay) * dw
            
            # 更新嵌入权重
            n = torch.sum(self._ema_cluster_size)
            updated_cluster_size = (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n
            
            # 只更新使用过的码本
            mask = self._ema_cluster_size > self._epsilon
            self._embedding.weight.data[mask] = self._ema_w[mask] / updated_cluster_size[mask].unsqueeze(1)
        
        # Straight-through
        quantized = inputs + (quantized - inputs).detach()
        
        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return vq_loss, quantized, perplexity, encodings, original_indices
    
    def merge_similar_codes(self, threshold=0.5):
        """合并相似的码本"""
        with torch.no_grad():
            active_indices = torch.where(self._active_mask)[0]
            if len(active_indices) <= 2:
                return 0
            
            active_embeddings = self._embedding.weight[active_indices]
            
            # 计算码本间距离
            distances = torch.cdist(active_embeddings, active_embeddings)
            distances.fill_diagonal_(float('inf'))
            
            merged_count = 0
            merged_mask = torch.zeros(len(active_indices), dtype=torch.bool)
            
            for i in range(len(active_indices)):
                if merged_mask[i]:
                    continue
                    
                # 找到与码本i距离小于阈值的所有码本
                close_indices = torch.where(distances[i] < threshold)[0]
                
                if len(close_indices) > 0:
                    # 选择使用率最高的作为保留码本
                    all_indices = torch.cat([torch.tensor([i]), close_indices])
                    all_indices = all_indices[~merged_mask[all_indices]]
                    
                    if len(all_indices) > 1:
                        # 获取实际的码本索引
                        actual_indices = active_indices[all_indices]
                        usage = self._usage_count[actual_indices]
                        
                        # 保留使用率最高的
                        keep_local_idx = all_indices[torch.argmax(usage)]
                        keep_idx = active_indices[keep_local_idx]
                        
                        # 合并其他码本到保留的码本
                        for local_idx in all_indices:
                            if local_idx != keep_local_idx:
                                merge_idx = active_indices[local_idx]
                                
                                # 加权平均更新码本位置
                                w1 = self._usage_count[keep_idx] / (self._usage_count[keep_idx] + self._usage_count[merge_idx] + 1e-10)
                                w2 = 1 - w1
                                self._embedding.weight.data[keep_idx] = (
                                    w1 * self._embedding.weight.data[keep_idx] +
                                    w2 * self._embedding.weight.data[merge_idx]
                                )
                                
                                # 合并统计信息
                                self._usage_count[keep_idx] += self._usage_count[merge_idx]
                                self._ema_cluster_size[keep_idx] += self._ema_cluster_size[merge_idx]
                                
                                # 停用被合并的码本
                                self._active_mask[merge_idx] = False
                                self._usage_count[merge_idx] = 0
                                self._ema_cluster_size[merge_idx] = 0
                                
                                merged_mask[local_idx] = True
                                merged_count += 1
            
            return merged_count
    
    def prune_low_usage_codes(self, min_usage_rate=0.01):
        """剪枝低使用率的码本"""
        with torch.no_grad():
            if self._batch_count == 0:
                return 0
            
            active_indices = torch.where(self._active_mask)[0]
            if len(active_indices) <= 5:  # 保持至少5个码本
                return 0
            
            # 计算平均使用率
            avg_usage = self._usage_count / (self._batch_count * 32 + 1e-10)  # 假设批次大小为32
            
            # 找出低使用率的码本
            low_usage_mask = (avg_usage < min_usage_rate) & self._active_mask
            
            # 确保至少保留5个码本
            n_to_prune = torch.sum(low_usage_mask).item()
            n_active = torch.sum(self._active_mask).item()
            
            if n_active - n_to_prune < 5:
                # 保留使用率最高的5个
                _, top_indices = torch.topk(avg_usage, min(5, n_active))
                new_active = torch.zeros_like(self._active_mask)
                new_active[top_indices] = True
                n_pruned = n_active - min(5, n_active)
                self._active_mask = new_active
            else:
                self._active_mask[low_usage_mask] = False
                n_pruned = n_to_prune
            
            # 清理被剪枝的码本
            pruned_mask = ~self._active_mask
            self._usage_count[pruned_mask] = 0
            self._ema_cluster_size[pruned_mask] = 0
            
            return n_pruned
    
    def compute_clustering_quality(self, data, indices):
        """计算聚类质量指标"""
        n_unique = len(torch.unique(indices))
        if n_unique <= 1:
            return 0.0
        
        try:
            # 计算轮廓系数
            sil_score = silhouette_score(
                data.cpu().numpy(), 
                indices.cpu().numpy(),
                sample_size=min(1000, len(data))
            )
            return sil_score
        except:
            return 0.0
    
    def get_active_codebook(self):
        """获取活跃码本"""
        active_indices = torch.where(self._active_mask)[0]
        return self._embedding.weight[active_indices], active_indices


def train_smart_adaptive_vqvae(data, labels, n_epochs=500):
    """训练智能自适应VQ-VAE"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    
    # 简单的编码器和解码器（使用恒等映射附近的初始化）
    encoder = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    ).to(device)
    
    decoder = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    ).to(device)
    
    # 使用较小的权重初始化，让网络接近恒等映射
    with torch.no_grad():
        for param in encoder.parameters():
            param.data *= 0.1
        for param in decoder.parameters():
            param.data *= 0.1
    
    # 创建VQ模型，初始码本数为聚类数的2倍
    n_initial_codes = len(np.unique(labels)) * 2  # 根据真实聚类数初始化
    vq_model = SmartAdaptiveVQVAE(
        embedding_dim=2,
        initial_codes=n_initial_codes,
        commitment_cost=0.25,
        decay=0.99,
        device=device
    ).to(device)
    
    # 初始化码本
    vq_model.initialize_from_data(data)
    
    # 优化器
    optimizer = optim.Adam(
        list(encoder.parameters()) + 
        list(decoder.parameters()) + 
        list(vq_model.parameters()),
        lr=2e-3
    )
    
    # 创建序列
    def create_sequences(data, seq_len=10):
        sequences = []
        for i in range(len(data) - seq_len + 1):
            sequences.append(data[i:i+seq_len])
        return torch.stack(sequences)
    
    sequences = create_sequences(data, seq_len=10)
    
    # 训练历史
    history = {
        'total_loss': [],
        'vq_loss': [],
        'recon_loss': [],
        'perplexity': [],
        'n_active': [],
        'quality_score': []
    }
    
    print("\nTraining with smart adaptation...")
    best_quality = -1
    best_n_codes = 16
    no_improve_count = 0
    
    for epoch in range(n_epochs):
        # 训练
        encoder.train()
        decoder.train()
        vq_model.train()
        
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
            optimizer.step()
            
            # 记录
            total_loss += loss.item()
            vq_loss_sum += vq_loss.item()
            recon_loss_sum += recon_loss.item()
            perplexity_sum += perplexity.item()
            n_batches += 1
        
        # 评估聚类质量
        with torch.no_grad():
            vq_model.eval()
            encoder.eval()
            z_e = encoder(data)
            _, _, _, _, indices = vq_model(z_e)
            quality = vq_model.compute_clustering_quality(data, indices)
        
        # 自适应调整（在训练稳定后才开始）
        if epoch > 100 and epoch % 50 == 0:  # 更晚开始，更少频率
            n_active_before = torch.sum(vq_model._active_mask).item()
            
            # 只有在码本数量过多时才调整
            if n_active_before > 10:  # 保持合理的码本数量
                # 1. 合并非常相似的码本
                merge_threshold = 0.1  # 只合并非常相近的码本
                n_merged = vq_model.merge_similar_codes(threshold=merge_threshold)
                
                # 2. 剪枝极低使用率码本
                min_usage = 0.001  # 只剪枝几乎不用的码本
                n_pruned = vq_model.prune_low_usage_codes(min_usage_rate=min_usage)
            else:
                n_merged = 0
                n_pruned = 0
            
            n_active_after = torch.sum(vq_model._active_mask).item()
            
            if n_merged > 0 or n_pruned > 0:
                print(f"Epoch {epoch}: Merged {n_merged}, Pruned {n_pruned}, "
                      f"Active codes: {n_active_before} -> {n_active_after}, "
                      f"Quality: {quality:.3f}")
            
            # 检查质量是否改善
            if quality > best_quality:
                best_quality = quality
                best_n_codes = n_active_after
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # 记录历史
        n_active = torch.sum(vq_model._active_mask).item()
        history['total_loss'].append(total_loss / n_batches)
        history['vq_loss'].append(vq_loss_sum / n_batches)
        history['recon_loss'].append(recon_loss_sum / n_batches)
        history['perplexity'].append(perplexity_sum / n_batches)
        history['n_active'].append(n_active)
        history['quality_score'].append(quality)
        
        # 打印进度
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] "
                  f"Loss: {history['total_loss'][-1]:.4f}, "
                  f"Active: {n_active}, "
                  f"Perplexity: {history['perplexity'][-1]:.2f}, "
                  f"Quality: {quality:.3f}")
        
        # 早停（保持合理的码本数量）
        if no_improve_count >= 5 or (epoch > 200 and n_active >= 5 and n_active <= 10):
            print(f"\nConverged at epoch {epoch+1}")
            print(f"Final configuration: {n_active} codes with quality {quality:.3f}")
            print(f"Best configuration: {best_n_codes} codes with quality {best_quality:.3f}")
            break
    
    return vq_model, encoder, decoder, history


def visualize_smart_results(data, labels, vq_model, encoder, decoder, history):
    """可视化结果（包含Voronoi区域）"""
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 码本数量和质量变化
    ax1 = plt.subplot(2, 4, 1)
    ax1_twin = ax1.twinx()
    ax1.plot(history['n_active'], 'b-', label='Active Codes')
    ax1_twin.plot(history['quality_score'], 'r-', alpha=0.7, label='Quality')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Active Codes', color='b')
    ax1_twin.set_ylabel('Quality Score', color='r')
    ax1.set_title('Adaptation Progress')
    ax1.grid(True, alpha=0.3)
    
    # 2. 损失曲线
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(history['total_loss'], label='Total')
    ax2.plot(history['vq_loss'], label='VQ')
    ax2.plot(history['recon_loss'], label='Recon')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Losses')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Perplexity
    ax3 = plt.subplot(2, 4, 3)
    ax3.plot(history['perplexity'], color='green')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Perplexity')
    ax3.set_title(f'Codebook Usage (Final: {history["perplexity"][-1]:.2f})')
    ax3.grid(True, alpha=0.3)
    
    # 4. 原始数据
    ax4 = plt.subplot(2, 4, 4)
    scatter = ax4.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', s=30, alpha=0.7)
    ax4.set_title(f'Original Data ({len(np.unique(labels))} Clusters)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    plt.colorbar(scatter, ax=ax4)
    
    # 获取最终的码本和分配
    with torch.no_grad():
        vq_model.eval()
        encoder.eval()
        decoder.eval()
        
        z_e = encoder(data)
        _, z_q, _, _, indices = vq_model(z_e)
        indices = indices.cpu().numpy()
        
        active_codebook, _ = vq_model.get_active_codebook()
        active_codebook = active_codebook.cpu().numpy()
        
        # 解码码本
        decoded_codebook = decoder(torch.FloatTensor(active_codebook).to(data.device))
        decoded_codebook = decoded_codebook.cpu().numpy()
    
    # 5. VQ聚类结果
    ax5 = plt.subplot(2, 4, 5)
    scatter = ax5.scatter(data[:, 0], data[:, 1], c=indices, cmap='tab20', s=30, alpha=0.7)
    ax5.scatter(decoded_codebook[:, 0], decoded_codebook[:, 1], 
               c='red', s=200, marker='*', edgecolors='black', linewidth=2)
    ax5.set_title(f'VQ Clustering ({len(active_codebook)} Codes)')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    plt.colorbar(scatter, ax=ax5)
    
    # 6. 原始空间的Voronoi区域
    ax6 = plt.subplot(2, 4, 6)
    
    # 创建网格
    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(data.device)
    
    # 通过模型处理网格点
    with torch.no_grad():
        z_grid = encoder(grid_points)
        _, _, _, _, grid_indices = vq_model(z_grid)
        grid_indices = grid_indices.cpu().numpy().reshape(xx.shape)
    
    # 绘制Voronoi区域
    contour = ax6.contourf(xx, yy, grid_indices, levels=len(active_codebook)-1, 
                           cmap='tab20', alpha=0.3)
    ax6.scatter(data[:, 0], data[:, 1], c='gray', alpha=0.2, s=10)
    ax6.scatter(decoded_codebook[:, 0], decoded_codebook[:, 1], 
               c='red', s=200, marker='*', edgecolors='black', linewidth=2)
    
    for i in range(len(decoded_codebook)):
        ax6.annotate(f'{i}', decoded_codebook[i], fontsize=10, 
                    ha='center', va='center', color='white', weight='bold')
    
    ax6.set_title('Voronoi Regions')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    
    # 7. 使用率分布
    ax7 = plt.subplot(2, 4, 7)
    usage_rates = vq_model._usage_count[vq_model._active_mask] / (vq_model._batch_count * 32 + 1e-10)
    usage_rates = usage_rates.detach().cpu().numpy()
    
    ax7.bar(range(len(usage_rates)), sorted(usage_rates, reverse=True))
    ax7.set_xlabel('Code Index (sorted)')
    ax7.set_ylabel('Usage Rate')
    ax7.set_title('Code Usage Distribution')
    ax7.grid(True, alpha=0.3)
    
    # 8. 聚类指标
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # 计算最终指标
    n_unique = len(np.unique(indices))
    if n_unique > 1:
        sil_score = silhouette_score(data.cpu().numpy(), indices)
        ch_score = calinski_harabasz_score(data.cpu().numpy(), indices)
    else:
        sil_score = 0
        ch_score = 0
    
    metrics_text = f"""
    === Final Results ===
    
    True Clusters: {len(np.unique(labels))}
    Final Codes: {len(active_codebook)}
    Active/Used: {n_unique}
    
    Silhouette: {sil_score:.3f}
    CH Score: {ch_score:.1f}
    
    Final Loss: {history['total_loss'][-1]:.4f}
    Perplexity: {history['perplexity'][-1]:.2f}
    Quality: {history['quality_score'][-1]:.3f}
    """
    
    ax8.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('Smart Adaptive VQ-VAE with Quality-Guided Optimization', fontsize=14)
    plt.tight_layout()
    
    return fig


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=== Smart Adaptive VQ-VAE ===")
    print("Using clustering quality metrics to guide adaptation\n")
    
    # 生成测试数据
    def generate_2d_clusters(n_points=700, n_clusters=7, noise=0.3):
        points_per_cluster = n_points // n_clusters
        data = []
        labels = []
        
        # 生成清晰分离的聚类
        angles = np.linspace(0, 2*np.pi, n_clusters, endpoint=False)
        radius = 2.0
        centers = np.array([[radius*np.cos(a), radius*np.sin(a)] for a in angles])
        
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
    print("Generating data with 7 clusters...")
    data, labels = generate_2d_clusters(n_points=700, n_clusters=7, noise=0.3)
    
    # 训练
    vq_model, encoder, decoder, history = train_smart_adaptive_vqvae(data, labels, n_epochs=500)
    
    # 可视化
    print("\nCreating visualization...")
    fig = visualize_smart_results(data, labels, vq_model, encoder, decoder, history)
    
    # 保存
    plt.savefig('vqvae_adaptive_smart.png', dpi=150, bbox_inches='tight')
    print("Results saved to vqvae_adaptive_smart.png")
    
    plt.show()


if __name__ == "__main__":
    main()