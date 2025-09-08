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


class VQWithRepulsion(VectorQuantizerEMA):
    """带斥力机制的VQ-EMA"""
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, device, 
                 epsilon=1e-5, repulsion_strength=0.01):
        super().__init__(num_embeddings, embedding_dim, commitment_cost, decay, device, epsilon)
        self.repulsion_strength = repulsion_strength
        self.register_buffer('_ema_usage', torch.zeros(num_embeddings))
        
    def forward(self, inputs, compute_distances_if_possible=True, record_codebook_stats=False):
        # 调用父类的forward方法
        result = super().forward(inputs, compute_distances_if_possible, record_codebook_stats)
        
        # 在训练时应用斥力
        if self.training:
            vq_loss, quantized, perplexity, encodings, distances, encoding_indices = result[:6]
            
            # 更新使用统计
            batch_usage = torch.sum(encodings, dim=0)
            # 确保维度正确
            if batch_usage.dim() > 1:
                batch_usage = batch_usage.sum(dim=-1)
            self._ema_usage = self._ema_usage * self._decay + (1 - self._decay) * batch_usage
            
            # 应用斥力防止码本过度聚集
            self.apply_repulsion()
        
        return result
    
    def apply_repulsion(self):
        """应用码本间的斥力"""
        with torch.no_grad():
            embeddings = self._embedding.weight.data
            n = embeddings.shape[0]
            
            # 计算码本间距离
            distances = torch.cdist(embeddings, embeddings)
            
            # 应用斥力
            for i in range(n):
                force = torch.zeros_like(embeddings[i])
                for j in range(n):
                    if i != j:
                        dist = distances[i, j]
                        if dist > 0 and dist < 2.0:  # 只对近距离的码本应用斥力
                            direction = (embeddings[i] - embeddings[j]) / (dist + 1e-8)
                            magnitude = self.repulsion_strength * (2.0 - dist) / 2.0
                            force += direction * magnitude
                
                embeddings[i] += force
            
            self._ema_w.data = embeddings.clone()


def train_vq_direct(data, n_epochs=200, num_embeddings=8, learning_rate=1e-3):
    """直接训练VQ-EMA，不使用投影层"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建序列数据
    sequences = create_sequence_data(data, seq_length=10)
    sequences = sequences.to(device)
    
    # 初始化VQ层
    vq_layer = VQWithRepulsion(
        num_embeddings=num_embeddings,
        embedding_dim=2,
        commitment_cost=0.25,
        decay=0.99,
        device=device,
        epsilon=1e-5,
        repulsion_strength=0.02
    ).to(device)
    
    # 使用K-means初始化码本
    with torch.no_grad():
        flat_data = sequences.reshape(-1, 2).cpu().numpy()
        kmeans = KMeans(n_clusters=num_embeddings, random_state=42, n_init=10)
        kmeans.fit(flat_data)
        initial_centers = torch.FloatTensor(kmeans.cluster_centers_).to(device)
        
        vq_layer._embedding.weight.data = initial_centers
        vq_layer._ema_w.data = initial_centers.clone()
        vq_layer._ema_cluster_size.data = torch.ones(num_embeddings) * (len(flat_data) / num_embeddings)
    
    # 不使用投影层，直接优化VQ层
    optimizer = optim.SGD([vq_layer._embedding.weight], lr=0.0)  # 不优化embedding，让EMA更新
    
    train_history = {
        'vq_loss': [],
        'perplexity': [],
        'codebook_positions': [],
        'data_positions': []
    }
    
    for epoch in range(n_epochs):
        vq_layer.train()
        
        # 将数据转换为VQ格式 (dim, seq_length, batch)
        inputs_vq = sequences.permute(2, 1, 0)
        
        # VQ前向传播
        vq_loss, quantized, perplexity, encodings, distances, encoding_indices, \
            vq_loss_dict, _, _, _, _ = vq_layer(inputs_vq)
        
        # 只使用VQ损失
        total_loss = vq_loss
        
        # 反向传播（主要用于commitment loss）
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 记录历史
        train_history['vq_loss'].append(vq_loss.item())
        train_history['perplexity'].append(perplexity.item())
        
        # 定期记录码本位置
        if epoch % 10 == 0:
            codebook_pos = vq_layer.embedding.weight.data.cpu().numpy().copy()
            train_history['codebook_positions'].append(codebook_pos)
            train_history['data_positions'].append(sequences.reshape(-1, 2).cpu().numpy())
        
        if (epoch + 1) % 20 == 0:
            active_codes = (vq_layer._ema_usage > 0.01).sum().item()
            print(f"Epoch [{epoch+1}/{n_epochs}] "
                  f"VQ Loss: {vq_loss.item():.4f}, "
                  f"Perplexity: {perplexity.item():.2f}, "
                  f"Active: {active_codes}/{num_embeddings}")
    
    return vq_layer, train_history


def visualize_results(data, labels, vq_layer, train_history):
    """可视化结果"""
    device = next(vq_layer.parameters()).device
    
    # 准备数据
    sequences = create_sequence_data(data, seq_length=10)
    sequences = sequences.to(device)
    
    vq_layer.eval()
    
    with torch.no_grad():
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
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 训练损失
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(train_history['vq_loss'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('VQ Loss')
    ax1.set_title('Training VQ Loss')
    ax1.grid(True)
    
    # 2. Perplexity
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(train_history['perplexity'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title(f'Codebook Usage (Final: {train_history["perplexity"][-1]:.2f})')
    ax2.grid(True)
    
    # 3. 原始数据
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', 
                         alpha=0.6, s=20)
    ax3.set_title('Original Data (True Clusters)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(-5, 5)
    plt.colorbar(scatter, ax=ax3)
    
    # 4. VQ聚类结果（关键图）
    ax4 = plt.subplot(2, 3, 4)
    # 绘制数据点，根据VQ分配着色
    scatter = ax4.scatter(data_points[:, 0], data_points[:, 1], 
                         c=cluster_assignments, cmap='tab10', alpha=0.5, s=20)
    
    # 绘制码本，大小根据使用率
    sizes = 100 + usage_stats * 500
    for i, (pos, usage) in enumerate(zip(codebook, usage_stats)):
        if usage > 0.01:
            ax4.scatter(pos[0], pos[1], c='red', s=sizes[i], 
                       marker='*', edgecolors='black', linewidth=2, alpha=0.9)
            ax4.annotate(f'{i}', pos, fontsize=10, ha='center', va='center', 
                        color='white', weight='bold')
    
    ax4.set_title('Data Points & VQ Codebook')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_xlim(-5, 5)
    ax4.set_ylim(-5, 5)
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4)
    
    # 5. Voronoi区域
    ax5 = plt.subplot(2, 3, 5)
    
    # 创建网格
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    
    # 计算最近的码本
    distances = torch.cdist(grid_points, torch.FloatTensor(codebook))
    nearest_codebook = distances.argmin(dim=1).numpy().reshape(xx.shape)
    
    # 绘制Voronoi区域
    ax5.contourf(xx, yy, nearest_codebook, levels=len(codebook)-1, 
                 cmap='tab10', alpha=0.3)
    ax5.scatter(data_points[:, 0], data_points[:, 1], 
               c='gray', alpha=0.1, s=5)
    
    # 绘制码本
    for i, pos in enumerate(codebook):
        if usage_stats[i] > 0.01:
            ax5.scatter(pos[0], pos[1], c='red', s=200, 
                       marker='*', edgecolors='black', linewidth=2)
            ax5.annotate(f'{i}', pos, fontsize=10, ha='center', va='center', 
                        color='white', weight='bold')
    
    ax5.set_title('Voronoi Regions')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_xlim(-5, 5)
    ax5.set_ylim(-5, 5)
    
    # 6. 码本演化
    ax6 = plt.subplot(2, 3, 6)
    if len(train_history['codebook_positions']) > 1:
        # 显示码本轨迹
        for i in range(len(codebook)):
            trajectory = [pos[i] for pos in train_history['codebook_positions']]
            trajectory = np.array(trajectory)
            ax6.plot(trajectory[:, 0], trajectory[:, 1], '-', alpha=0.5, linewidth=1)
            ax6.scatter(trajectory[0, 0], trajectory[0, 1], c='blue', s=50, marker='o', alpha=0.5)
            ax6.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='*')
            ax6.annotate(f'{i}', trajectory[-1], fontsize=8, ha='center', va='center')
        
        ax6.set_title('Codebook Evolution')
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_xlim(-5, 5)
        ax6.set_ylim(-5, 5)
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vqvae_fixed_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to vqvae_fixed_visualization.png")
    
    return fig


def main():
    """主函数"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=== Fixed VQ-EMA 2D Clustering ===\n")
    
    # 1. 生成二维数据
    print("1. Generating 2D cluster data...")
    n_points = 700
    n_clusters = 7
    data, labels = generate_2d_clusters(n_points=n_points, n_clusters=n_clusters, noise=0.25)
    print(f"   Generated {n_points} points in {n_clusters} clusters")
    
    # 2. 训练VQ-EMA
    print("\n2. Training VQ-EMA...")
    num_embeddings = 10
    n_epochs = 200
    
    vq_layer, train_history = train_vq_direct(
        data,
        n_epochs=n_epochs,
        num_embeddings=num_embeddings,
        learning_rate=0
    )
    
    # 3. 可视化结果
    print("\n3. Visualizing results...")
    fig = visualize_results(data, labels, vq_layer, train_history)
    
    # 4. 分析
    print("\n=== Analysis ===")
    active_codes = (vq_layer._ema_usage > 0.01).sum().item()
    print(f"Final VQ Loss: {train_history['vq_loss'][-1]:.4f}")
    print(f"Final Perplexity: {train_history['perplexity'][-1]:.2f}")
    print(f"Active Codes: {active_codes}/{num_embeddings}")


if __name__ == "__main__":
    main()