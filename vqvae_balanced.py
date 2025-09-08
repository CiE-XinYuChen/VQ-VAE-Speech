import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class BalancedVQVAE(nn.Module):
    """平衡的VQ-VAE，确保码本均匀分布"""
    
    def __init__(self, embedding_dim, num_embeddings, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._epsilon = epsilon
        
        # 嵌入层
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        
        # EMA参数
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', torch.zeros(num_embeddings, embedding_dim))
        
        # 使用统计
        self.register_buffer('_usage_count', torch.zeros(num_embeddings))
        
    def forward(self, inputs):
        # 展平输入
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # 计算距离
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True) +
            torch.sum(self._embedding.weight**2, dim=1) -
            2 * torch.matmul(flat_input, self._embedding.weight.t())
        )
        
        # 找最近的嵌入
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self._num_embeddings).float()
        
        # 量化
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # 计算损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # 训练时更新EMA
        if self.training:
            # 统计使用情况
            for idx in encoding_indices:
                self._usage_count[idx] += 1
            
            # EMA更新
            self._ema_cluster_size = self._decay * self._ema_cluster_size + (1 - self._decay) * torch.sum(encodings, dim=0)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = self._decay * self._ema_w + (1 - self._decay) * dw
            
            # 更新嵌入权重
            n = torch.sum(self._ema_cluster_size)
            self._ema_cluster_size = (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n
            self._embedding.weight.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)
            
            # 重新初始化未使用的码本（重要！）
            usage_threshold = torch.mean(self._usage_count) * 0.1
            low_usage_indices = torch.where(self._usage_count < usage_threshold)[0]
            
            if len(low_usage_indices) > 0 and torch.sum(self._usage_count) > 100:
                # 将低使用率的码本重新初始化到高密度区域
                high_density_indices = torch.topk(self._usage_count, k=min(3, self._num_embeddings//2))[1]
                for low_idx in low_usage_indices:
                    # 在高使用率码本附近添加噪声创建新码本
                    high_idx = high_density_indices[torch.randint(len(high_density_indices), (1,))]
                    self._embedding.weight.data[low_idx] = (
                        self._embedding.weight.data[high_idx] + 
                        torch.randn_like(self._embedding.weight.data[high_idx]) * 0.5
                    )
                    # 重置统计
                    self._ema_cluster_size[low_idx] = self._ema_cluster_size[high_idx] / 2
                    self._ema_w[low_idx] = self._embedding.weight.data[low_idx] * self._ema_cluster_size[low_idx]
                    self._usage_count[low_idx] = self._usage_count[high_idx] / 2
        
        # Straight-through
        quantized = inputs + (quantized - inputs).detach()
        
        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return vq_loss, quantized, perplexity, encodings, encoding_indices


def train_balanced_vqvae(data, n_codes=8, n_epochs=300, learning_rate=1e-3):
    """训练平衡的VQ-VAE"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    
    # 直接在原始空间工作（不使用编码器/解码器）
    vq_model = BalancedVQVAE(
        embedding_dim=2,
        num_embeddings=n_codes,
        commitment_cost=0.25,
        decay=0.99
    ).to(device)
    
    # 使用K-means初始化码本
    kmeans = KMeans(n_clusters=n_codes, n_init=10, random_state=42)
    kmeans.fit(data.cpu().numpy())
    vq_model._embedding.weight.data = torch.FloatTensor(kmeans.cluster_centers_).to(device)
    
    # 优化器（只优化VQ层）
    optimizer = optim.Adam(vq_model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # 创建批次数据
    def create_batches(data, batch_size=32):
        n_samples = len(data)
        indices = torch.randperm(n_samples)
        batches = []
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batches.append(data[batch_indices])
        return batches
    
    # 训练历史
    history = {
        'vq_loss': [],
        'perplexity': [],
        'n_active': []
    }
    
    print(f"Training with {n_codes} codes...")
    
    for epoch in range(n_epochs):
        vq_model.train()
        
        epoch_loss = 0
        epoch_perplexity = 0
        n_batches = 0
        
        batches = create_batches(data, batch_size=32)
        
        for batch in batches:
            # 前向传播
            vq_loss, quantized, perplexity, _, indices = vq_model(batch)
            
            # 反向传播
            optimizer.zero_grad()
            vq_loss.backward()
            optimizer.step()
            
            epoch_loss += vq_loss.item()
            epoch_perplexity += perplexity.item()
            n_batches += 1
        
        scheduler.step()
        
        # 记录
        history['vq_loss'].append(epoch_loss / n_batches)
        history['perplexity'].append(epoch_perplexity / n_batches)
        
        # 计算活跃码本数
        with torch.no_grad():
            vq_model.eval()
            _, _, _, _, all_indices = vq_model(data)
            n_active = len(torch.unique(all_indices))
            history['n_active'].append(n_active)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] "
                  f"Loss: {history['vq_loss'][-1]:.4f}, "
                  f"Perplexity: {history['perplexity'][-1]:.2f}, "
                  f"Active: {n_active}/{n_codes}")
    
    return vq_model, history


def visualize_balanced_results(data, labels, vq_model, history):
    """可视化平衡的结果"""
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 训练损失
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history['vq_loss'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('VQ Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # 2. Perplexity
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(history['perplexity'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title(f'Codebook Usage (Final: {history["perplexity"][-1]:.2f})')
    ax2.grid(True, alpha=0.3)
    
    # 3. 活跃码本数
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(history['n_active'])
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Active Codes')
    ax3.set_title(f'Active Codebook Count')
    ax3.grid(True, alpha=0.3)
    
    # 4. 原始数据
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', s=30, alpha=0.7)
    ax4.set_title(f'Original Data ({len(np.unique(labels))} Clusters)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    plt.colorbar(scatter, ax=ax4)
    
    # 5. VQ聚类结果和码本
    ax5 = plt.subplot(2, 3, 5)
    
    with torch.no_grad():
        vq_model.eval()
        _, quantized, _, _, indices = vq_model(data)
        indices = indices.cpu().numpy()
        codebook = vq_model._embedding.weight.cpu().numpy()
    
    scatter = ax5.scatter(data[:, 0], data[:, 1], c=indices, cmap='tab20', s=30, alpha=0.7)
    ax5.scatter(codebook[:, 0], codebook[:, 1], 
               c='red', s=200, marker='*', edgecolors='black', linewidth=2)
    
    for i, pos in enumerate(codebook):
        ax5.annotate(f'{i}', pos, fontsize=10, ha='center', va='center', color='white')
    
    ax5.set_title(f'VQ Clustering ({len(codebook)} Codes)')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    plt.colorbar(scatter, ax=ax5)
    
    # 6. Voronoi区域
    ax6 = plt.subplot(2, 3, 6)
    
    # 创建网格
    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(data.device)
    
    with torch.no_grad():
        _, _, _, _, grid_indices = vq_model(grid_points)
        grid_indices = grid_indices.cpu().numpy().reshape(xx.shape)
    
    contour = ax6.contourf(xx, yy, grid_indices, levels=len(codebook)-1, 
                           cmap='tab20', alpha=0.3)
    ax6.scatter(data[:, 0], data[:, 1], c='gray', alpha=0.2, s=10)
    ax6.scatter(codebook[:, 0], codebook[:, 1], 
               c='red', s=200, marker='*', edgecolors='black', linewidth=2)
    
    for i, pos in enumerate(codebook):
        ax6.annotate(f'{i}', pos, fontsize=10, ha='center', va='center', 
                    color='white', weight='bold')
    
    ax6.set_title('Voronoi Regions')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    
    plt.suptitle('Balanced VQ-VAE Clustering', fontsize=14)
    plt.tight_layout()
    
    # 打印统计
    n_unique = len(np.unique(indices))
    if n_unique > 1:
        sil_score = silhouette_score(data.cpu().numpy(), indices)
    else:
        sil_score = 0
    
    print(f"\n=== Final Statistics ===")
    print(f"True clusters: {len(np.unique(labels))}")
    print(f"VQ codes: {len(codebook)}")
    print(f"Active codes: {n_unique}")
    print(f"Silhouette score: {sil_score:.3f}")
    print(f"Final perplexity: {history['perplexity'][-1]:.2f}")
    
    # 打印每个码本的使用率
    unique, counts = np.unique(indices, return_counts=True)
    usage_dict = dict(zip(unique, counts))
    print(f"\nCodebook usage:")
    for i in range(len(codebook)):
        usage = usage_dict.get(i, 0)
        print(f"  Code {i}: {usage} samples ({usage/len(data)*100:.1f}%)")
    
    return fig


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=== Balanced VQ-VAE ===\n")
    
    # 生成测试数据
    def generate_2d_clusters(n_points=2000, n_clusters=40, noise=0.25):
        points_per_cluster = n_points // n_clusters
        data = []
        labels = []
        
        # 环形分布的聚类中心
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
    
    # 测试不同数量的码本
    data, labels = generate_2d_clusters(n_points=20000, n_clusters=40, noise=0.25)
    
    # 尝试不同的码本数量
    for n_codes in [7, 8, 10,12,14,20,21,50]:
        print(f"\n{'='*50}")
        print(f"Testing with {n_codes} codes...")
        
        # 训练
        vq_model, history = train_balanced_vqvae(data, n_codes=n_codes, n_epochs=300)
        
        # 可视化
        fig = visualize_balanced_results(data, labels, vq_model, history)
        
        # 保存
        plt.savefig(f'vqvae_balanced_{n_codes}codes.png', dpi=150, bbox_inches='tight')
        print(f"Results saved to vqvae_balanced_{n_codes}codes.png")
        
        # 检查是否达到均匀分布
        with torch.no_grad():
            vq_model.eval()
            _, _, _, _, indices = vq_model(data)
            unique, counts = torch.unique(indices, return_counts=True)
            
            # 计算分布的均匀程度（标准差越小越均匀）
            usage_rates = counts.float() / len(data)
            uniformity = torch.std(usage_rates).item()
            
            print(f"Distribution uniformity (std): {uniformity:.4f}")
            
            # 如果分布足够均匀（标准差小于0.05），认为成功
            if uniformity < 0.02 and len(unique) >= 6:
                print(f"\n✓ Successfully achieved uniform distribution with {n_codes} codes!")
                plt.show()
                continue
    
    print("\nTrying adaptive approach...")
    plt.show()


if __name__ == "__main__":
    main()