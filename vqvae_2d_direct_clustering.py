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


def train_direct_vq(data, n_epochs=100, num_embeddings=8, learning_rate=1e-3):
    """直接使用VQ-EMA对二维数据进行聚类（无编码器/解码器）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建序列数据
    sequences = create_sequence_data(data, seq_length=10)
    sequences = sequences.to(device)
    batch_size, seq_length, dim = sequences.shape
    
    # 初始化VQ层
    vq_layer = VectorQuantizerEMA(
        num_embeddings=num_embeddings,
        embedding_dim=2,  # 二维嵌入
        commitment_cost=0.25,
        decay=0.99,
        device=device,
        epsilon=1e-5
    ).to(device)
    
    # 初始化码本位置在数据的范围内
    with torch.no_grad():
        # 计算数据的中心和范围
        data_mean = sequences.mean(dim=(0, 1))
        data_std = sequences.std(dim=(0, 1))
        # 在数据范围内初始化码本
        vq_layer._embedding.weight.data = torch.randn(num_embeddings, 2) * data_std + data_mean
        vq_layer._ema_w.data = vq_layer._embedding.weight.data.clone()
    
    # 由于没有编码器，我们直接优化输入到VQ层的映射
    # 使用一个简单的线性变换作为可学习的映射
    projection = nn.Linear(2, 2, bias=True).to(device)
    
    # 初始化projection为恒等变换
    with torch.no_grad():
        projection.weight.data = torch.eye(2)
        projection.bias.data = torch.zeros(2)
    
    optimizer = optim.Adam(projection.parameters(), lr=learning_rate)
    
    train_history = {
        'vq_loss': [],
        'perplexity': [],
        'codebook_positions': [],
        'input_positions': []
    }
    
    for epoch in range(n_epochs):
        vq_layer.train()
        projection.train()
        
        # 将数据转换为VQ期望的格式
        # 需要从 (batch, seq_length, dim) -> (dim, seq_length, batch)
        projected = projection(sequences)
        inputs_vq = projected.permute(2, 1, 0)
        
        # VQ前向传播
        vq_loss, quantized, perplexity, encodings, distances, encoding_indices, \
            vq_loss_dict, _, _, _, _ = vq_layer(inputs_vq)
        
        # 由于没有解码器，我们的损失就是VQ损失
        total_loss = vq_loss
        
        # 反向传播
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
            
            # 记录投影后的输入位置
            with torch.no_grad():
                proj_data = projection(sequences).cpu().numpy()
                train_history['input_positions'].append(proj_data.reshape(-1, 2))
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] "
                  f"VQ Loss: {vq_loss.item():.4f}, "
                  f"Perplexity: {perplexity.item():.2f}")
    
    return vq_layer, projection, train_history


def visualize_direct_clustering(data, labels, vq_layer, projection, train_history):
    """可视化直接VQ聚类结果"""
    device = next(vq_layer.parameters()).device
    
    # 准备数据
    sequences = create_sequence_data(data, seq_length=10)
    sequences = sequences.to(device)
    
    vq_layer.eval()
    projection.eval()
    
    with torch.no_grad():
        # 投影数据
        projected = projection(sequences)
        inputs_vq = projected.permute(2, 1, 0)
        
        # VQ推理
        _, quantized, _, _, _, encoding_indices, _, _, _, _, _ = vq_layer(
            inputs_vq, compute_distances_if_possible=False
        )
        
        # 获取结果
        projected_points = projected.reshape(-1, 2).cpu().numpy()
        quantized_points = quantized.permute(2, 1, 0).reshape(-1, 2).cpu().numpy()
        cluster_assignments = encoding_indices.squeeze().cpu().numpy().flatten()
        codebook = vq_layer.embedding.weight.data.cpu().numpy()
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    
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
    
    # 3. 原始数据和真实标签
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', 
                         alpha=0.6, s=20)
    ax3.set_title('Original Data (True Clusters)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(scatter, ax=ax3)
    
    # 4. 投影后的数据和VQ聚类结果
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(projected_points[:, 0], projected_points[:, 1], 
                         c=cluster_assignments, cmap='tab10', alpha=0.6, s=20)
    ax4.scatter(codebook[:, 0], codebook[:, 1], c='red', s=200, 
               marker='*', edgecolors='black', linewidth=2, label='Codebook')
    for i, pos in enumerate(codebook):
        ax4.annotate(f'{i}', pos, fontsize=10, ha='center', va='center', color='white')
    ax4.set_title('Projected Data & VQ Clustering')
    ax4.set_xlabel('Projected X')
    ax4.set_ylabel('Projected Y')
    ax4.legend()
    plt.colorbar(scatter, ax=ax4)
    
    # 5. Voronoi图
    ax5 = plt.subplot(2, 3, 5)
    
    # 创建网格点
    x_min, x_max = projected_points[:, 0].min() - 0.5, projected_points[:, 0].max() + 0.5
    y_min, y_max = projected_points[:, 1].min() - 0.5, projected_points[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    
    # 计算到码本的距离
    distances = torch.cdist(grid_points, torch.FloatTensor(codebook))
    nearest_codebook = distances.argmin(dim=1).numpy().reshape(xx.shape)
    
    # 绘制Voronoi区域
    ax5.contourf(xx, yy, nearest_codebook, levels=len(codebook)-1, 
                 cmap='tab10', alpha=0.3)
    ax5.scatter(projected_points[:, 0], projected_points[:, 1], 
               c='gray', alpha=0.2, s=10)
    ax5.scatter(codebook[:, 0], codebook[:, 1], c='red', s=200, 
               marker='*', edgecolors='black', linewidth=2)
    for i, pos in enumerate(codebook):
        ax5.annotate(f'{i}', pos, fontsize=10, ha='center', va='center', 
                    color='white', weight='bold')
    ax5.set_title('Voronoi Regions in Projected Space')
    ax5.set_xlabel('Projected X')
    ax5.set_ylabel('Projected Y')
    
    # 6. 码本使用统计
    ax6 = plt.subplot(2, 3, 6)
    unique, counts = np.unique(cluster_assignments, return_counts=True)
    bars = ax6.bar(range(len(codebook)), [0]*len(codebook), color='lightgray')
    colors = plt.cm.tab10(np.linspace(0, 1, len(codebook)))
    for u, c in zip(unique, counts):
        bars[u].set_height(c)
        bars[u].set_color(colors[u])
    ax6.set_xlabel('Codebook Index')
    ax6.set_ylabel('Usage Count')
    ax6.set_title('Codebook Usage Distribution')
    ax6.set_xticks(range(len(codebook)))
    ax6.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('vqvae_2d_direct_clustering.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to vqvae_2d_direct_clustering.png")
    
    return fig


def create_evolution_animation(train_history):
    """创建码本演化动画"""
    if len(train_history['codebook_positions']) < 2:
        print("Not enough codebook snapshots for animation")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 获取数据范围
    all_positions = np.vstack(train_history['input_positions'])
    x_min, x_max = all_positions[:, 0].min() - 1, all_positions[:, 0].max() + 1
    y_min, y_max = all_positions[:, 1].min() - 1, all_positions[:, 1].max() + 1
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_xlim(0, len(train_history['vq_loss']))
    ax2.set_ylim(0, max(train_history['vq_loss']) * 1.1)
    
    # 初始化元素
    data_scatter = ax1.scatter([], [], c='gray', alpha=0.2, s=10)
    codebook_scatter = ax1.scatter([], [], c='red', s=200, marker='*', 
                                  edgecolors='black', linewidth=2)
    loss_line, = ax2.plot([], [], 'b-', label='VQ Loss')
    
    ax1.set_title('Codebook Evolution')
    ax1.set_xlabel('Projected X')
    ax1.set_ylabel('Projected Y')
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('VQ Loss')
    ax2.legend()
    ax2.grid(True)
    
    def init():
        return data_scatter, codebook_scatter, loss_line
    
    def animate(frame):
        # 更新数据点
        input_pos = train_history['input_positions'][frame]
        data_scatter.set_offsets(input_pos)
        
        # 更新码本位置
        codebook_pos = train_history['codebook_positions'][frame]
        codebook_scatter.set_offsets(codebook_pos)
        
        # 更新损失曲线
        epoch = frame * 10  # 每10个epoch保存一次
        loss_line.set_data(range(epoch), train_history['vq_loss'][:epoch])
        
        ax1.set_title(f'Codebook Evolution (Epoch {epoch})')
        
        return data_scatter, codebook_scatter, loss_line
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(train_history['codebook_positions']),
                        interval=200, blit=True, repeat=True)
    
    anim.save('vqvae_direct_evolution.gif', writer='pillow', fps=5)
    print("Animation saved to vqvae_direct_evolution.gif")
    plt.close()
    
    return anim


def main():
    """主函数"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=== Direct VQ-EMA 2D Clustering (No Encoder/Decoder) ===\n")
    
    # 1. 生成二维数据
    print("1. Generating 2D cluster data...")
    n_points = 800
    n_clusters = 12
    data, labels = generate_2d_clusters(n_points=n_points, n_clusters=n_clusters, noise=0.3)
    print(f"   Generated {n_points} points in {n_clusters} clusters")
    
    # 2. 训练VQ-EMA
    print("\n2. Training VQ-EMA directly on 2D data...")
    num_embeddings = 7  # 码本大小
    n_epochs = 1000
    
    vq_layer, projection, train_history = train_direct_vq(
        data,
        n_epochs=n_epochs,
        num_embeddings=num_embeddings,
        learning_rate=1e-2
    )
    
    # 3. 可视化结果
    print("\n3. Visualizing results...")
    fig = visualize_direct_clustering(data, labels, vq_layer, projection, train_history)
    
    # 4. 创建演化动画
    print("\n4. Creating evolution animation...")
    anim = create_evolution_animation(train_history)
    
    # 5. 分析聚类质量
    print("\n=== Clustering Analysis ===")
    
    # 获取聚类结果
    sequences = create_sequence_data(data, seq_length=10)
    with torch.no_grad():
        device = next(vq_layer.parameters()).device
        sequences = sequences.to(device)
        projected = projection(sequences)
        inputs_vq = projected.permute(2, 1, 0)
        _, _, perplexity, _, _, encoding_indices, _, _, _, _, _ = vq_layer(
            inputs_vq, compute_distances_if_possible=False
        )
        cluster_assignments = encoding_indices.squeeze().cpu().numpy().flatten()
    
    # 计算聚类指标
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    
    data_subset = data[:len(cluster_assignments)].numpy()
    labels_subset = labels[:len(cluster_assignments)].numpy()
    
    sil_score = silhouette_score(data_subset, cluster_assignments)
    ch_score = calinski_harabasz_score(data_subset, cluster_assignments)
    
    print(f"Final VQ Loss: {train_history['vq_loss'][-1]:.4f}")
    print(f"Final Perplexity: {train_history['perplexity'][-1]:.2f}")
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Calinski-Harabasz Score: {ch_score:.1f}")
    
    unique_codes = len(np.unique(cluster_assignments))
    print(f"Active Codes: {unique_codes}/{num_embeddings}")
    
    print("\nResults saved:")
    print("  - vqvae_2d_direct_clustering.png")
    print("  - vqvae_direct_evolution.gif")


if __name__ == "__main__":
    main()