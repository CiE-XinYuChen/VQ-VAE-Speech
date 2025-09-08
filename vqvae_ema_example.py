#!/usr/bin/env python3
"""
Standalone VectorQuantizerEMA Training and Evaluation Example
Demonstrates training, evaluation, model structure visualization, and saving/loading
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

# Add src to path to import the models
sys.path.append('./src')

from models.vector_quantizer_ema import VectorQuantizerEMA
from models.convolutional_encoder import ConvolutionalEncoder
from models.deconvolutional_decoder import DeconvolutionalDecoder


class SimpleVQVAE(nn.Module):
    """Simplified VQ-VAE model with VectorQuantizerEMA"""
    
    def __init__(self, config, device):
        super(SimpleVQVAE, self).__init__()
        
        self.device = device
        
        # Encoder: Convert input to latent representation
        self.encoder = nn.Sequential(
            nn.Conv1d(config['input_dim'], config['hidden_dim'], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(config['hidden_dim'], config['hidden_dim'], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(config['hidden_dim'], config['embedding_dim'], kernel_size=3, padding=1)
        )
        
        # Vector Quantizer with EMA
        self.vq = VectorQuantizerEMA(
            num_embeddings=config['num_embeddings'],
            embedding_dim=config['embedding_dim'],
            commitment_cost=config['commitment_cost'],
            decay=config['decay'],
            device=device
        )
        
        # Decoder: Reconstruct from quantized representation
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(config['embedding_dim'], config['hidden_dim'], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(config['hidden_dim'], config['hidden_dim'], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(config['hidden_dim'], config['input_dim'], kernel_size=4, stride=2, padding=1)
        )
        
    def forward(self, x):
        # Encode
        z_e = self.encoder(x)
        
        # Quantize
        z_q, loss_vq, perplexity = self.vq(z_e)
        
        # Decode
        x_recon = self.decoder(z_q)
        
        return x_recon, loss_vq, perplexity
    
    def encode(self, x):
        """Encode input to indices"""
        z_e = self.encoder(x)
        z_q, indices = self.vq.straight_through(z_e)
        return indices
    
    def decode_from_indices(self, indices):
        """Decode from codebook indices"""
        z_q = self.vq.quantize_indices(indices)
        return self.decoder(z_q)


def print_model_structure(model, input_shape):
    """Print model structure and parameter count"""
    print("\n" + "="*80)
    print("MODEL STRUCTURE")
    print("="*80)
    
    # Print model architecture
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "-"*40)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Print input/output shapes
    print("\n" + "-"*40)
    print("Shape Analysis:")
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    with torch.no_grad():
        z_e = model.encoder(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Encoded shape: {z_e.shape}")
        print(f"Codebook size: {model.vq.num_embeddings} x {model.vq.embedding_dim}")
        
        x_recon, _, _ = model(dummy_input)
        print(f"Output shape: {x_recon.shape}")
    
    print("="*80 + "\n")


def train_epoch(model, data_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0
    
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        recon, vq_loss, perplexity = model(data)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, data)
        
        # Total loss
        loss = recon_loss + vq_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()
        
        if batch_idx % 10 == 0:
            print(f'  Batch [{batch_idx}/{len(data_loader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'Recon: {recon_loss.item():.4f} '
                  f'VQ: {vq_loss.item():.4f} '
                  f'Perplexity: {perplexity.item():.2f}')
    
    # Average metrics
    n_batches = len(data_loader)
    avg_loss = total_loss / n_batches
    avg_recon_loss = total_recon_loss / n_batches
    avg_vq_loss = total_vq_loss / n_batches
    avg_perplexity = total_perplexity / n_batches
    
    return avg_loss, avg_recon_loss, avg_vq_loss, avg_perplexity


def evaluate(model, data_loader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            
            # Forward pass
            recon, vq_loss, perplexity = model(data)
            
            # Losses
            recon_loss = F.mse_loss(recon, data)
            loss = recon_loss + vq_loss
            
            # Accumulate metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
    
    # Average metrics
    n_batches = len(data_loader)
    avg_loss = total_loss / n_batches
    avg_recon_loss = total_recon_loss / n_batches
    avg_vq_loss = total_vq_loss / n_batches
    avg_perplexity = total_perplexity / n_batches
    
    return avg_loss, avg_recon_loss, avg_vq_loss, avg_perplexity


def save_model(model, optimizer, epoch, config, metrics, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def load_model(model, optimizer, load_path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    config = checkpoint['config']
    metrics = checkpoint['metrics']
    print(f"Model loaded from {load_path} (epoch {epoch})")
    return epoch, config, metrics


def plot_metrics(train_metrics, eval_metrics, save_path):
    """Plot training and evaluation metrics"""
    epochs = range(1, len(train_metrics['loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total Loss
    axes[0, 0].plot(epochs, train_metrics['loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, eval_metrics['loss'], 'r-', label='Eval')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Reconstruction Loss
    axes[0, 1].plot(epochs, train_metrics['recon_loss'], 'b-', label='Train')
    axes[0, 1].plot(epochs, eval_metrics['recon_loss'], 'r-', label='Eval')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # VQ Loss
    axes[1, 0].plot(epochs, train_metrics['vq_loss'], 'b-', label='Train')
    axes[1, 0].plot(epochs, eval_metrics['vq_loss'], 'r-', label='Eval')
    axes[1, 0].set_title('VQ Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Perplexity
    axes[1, 1].plot(epochs, train_metrics['perplexity'], 'b-', label='Train')
    axes[1, 1].plot(epochs, eval_metrics['perplexity'], 'r-', label='Eval')
    axes[1, 1].set_title('Codebook Perplexity')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Perplexity')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Metrics plot saved to {save_path}")
    plt.close()


def visualize_reconstruction(model, data_loader, device, save_path, n_samples=3):
    """Visualize original vs reconstructed samples"""
    model.eval()
    
    with torch.no_grad():
        data = next(iter(data_loader))[:n_samples].to(device)
        recon, _, _ = model(data)
        
        fig, axes = plt.subplots(n_samples, 2, figsize=(12, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            # Original
            axes[i, 0].plot(data[i, 0].cpu().numpy())
            axes[i, 0].set_title(f'Original Sample {i+1}')
            axes[i, 0].set_xlabel('Time')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].grid(True)
            
            # Reconstruction
            axes[i, 1].plot(recon[i, 0].cpu().numpy())
            axes[i, 1].set_title(f'Reconstructed Sample {i+1}')
            axes[i, 1].set_xlabel('Time')
            axes[i, 1].set_ylabel('Amplitude')
            axes[i, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Reconstruction visualization saved to {save_path}")
        plt.close()


def create_random_dataloader(batch_size, input_dim, sequence_length, n_samples):
    """Create a DataLoader with random data for demonstration"""
    # Generate random data (e.g., simulating audio features)
    data = torch.randn(n_samples, input_dim, sequence_length)
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def main():
    parser = argparse.ArgumentParser(description='VectorQuantizerEMA Training Example')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'both'],
                        help='Mode: train, eval, or both')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./vqvae_ema_output',
                        help='Directory to save outputs')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to checkpoint to load')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'input_dim': 80,           # e.g., 80 mel-frequency bins
        'hidden_dim': 128,          # Hidden layer dimension
        'embedding_dim': 64,        # Embedding dimension
        'num_embeddings': 512,      # Codebook size
        'commitment_cost': 0.25,    # Commitment loss weight
        'decay': 0.99,              # EMA decay rate
        'sequence_length': 128,     # Sequence length
        'n_train_samples': 1000,    # Number of training samples
        'n_eval_samples': 200       # Number of evaluation samples
    }
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create model
    model = SimpleVQVAE(config, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Print model structure
    print_model_structure(model, (config['input_dim'], config['sequence_length']))
    
    # Create data loaders with random data
    print("\nCreating random data for demonstration...")
    train_loader = create_random_dataloader(
        args.batch_size, config['input_dim'], config['sequence_length'], config['n_train_samples']
    )
    eval_loader = create_random_dataloader(
        args.batch_size, config['input_dim'], config['sequence_length'], config['n_eval_samples']
    )
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.load_checkpoint:
        start_epoch, loaded_config, loaded_metrics = load_model(
            model, optimizer, args.load_checkpoint, device
        )
        config = loaded_config
    
    # Training mode
    if args.mode in ['train', 'both']:
        print("\n" + "="*80)
        print("TRAINING")
        print("="*80)
        
        # Metrics storage
        train_metrics = {'loss': [], 'recon_loss': [], 'vq_loss': [], 'perplexity': []}
        eval_metrics = {'loss': [], 'recon_loss': [], 'vq_loss': [], 'perplexity': []}
        
        for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
            print(f"\nEpoch {epoch}/{start_epoch + args.epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_recon, train_vq, train_perp = train_epoch(
                model, train_loader, optimizer, device, epoch
            )
            
            # Evaluate
            eval_loss, eval_recon, eval_vq, eval_perp = evaluate(
                model, eval_loader, device
            )
            
            # Store metrics
            train_metrics['loss'].append(train_loss)
            train_metrics['recon_loss'].append(train_recon)
            train_metrics['vq_loss'].append(train_vq)
            train_metrics['perplexity'].append(train_perp)
            
            eval_metrics['loss'].append(eval_loss)
            eval_metrics['recon_loss'].append(eval_recon)
            eval_metrics['vq_loss'].append(eval_vq)
            eval_metrics['perplexity'].append(eval_perp)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, "
                  f"VQ: {train_vq:.4f}, Perplexity: {train_perp:.2f}")
            print(f"  Eval  - Loss: {eval_loss:.4f}, Recon: {eval_recon:.4f}, "
                  f"VQ: {eval_vq:.4f}, Perplexity: {eval_perp:.2f}")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt')
                save_model(model, optimizer, epoch, config, 
                          {'train': train_metrics, 'eval': eval_metrics}, 
                          checkpoint_path)
        
        # Save final model
        final_path = os.path.join(args.save_dir, 'final_model.pt')
        save_model(model, optimizer, start_epoch + args.epochs, config,
                  {'train': train_metrics, 'eval': eval_metrics},
                  final_path)
        
        # Plot metrics
        plot_path = os.path.join(args.save_dir, 'training_metrics.png')
        plot_metrics(train_metrics, eval_metrics, plot_path)
        
        # Visualize reconstructions
        recon_path = os.path.join(args.save_dir, 'reconstructions.png')
        visualize_reconstruction(model, eval_loader, device, recon_path)
    
    # Evaluation mode
    if args.mode in ['eval', 'both']:
        print("\n" + "="*80)
        print("EVALUATION")
        print("="*80)
        
        model.eval()
        
        # Evaluate on test set
        eval_loss, eval_recon, eval_vq, eval_perp = evaluate(
            model, eval_loader, device
        )
        
        print("\nEvaluation Results:")
        print(f"  Total Loss: {eval_loss:.4f}")
        print(f"  Reconstruction Loss: {eval_recon:.4f}")
        print(f"  VQ Loss: {eval_vq:.4f}")
        print(f"  Codebook Perplexity: {eval_perp:.2f}")
        
        # Test encoding and decoding
        print("\nTesting encoding/decoding pipeline:")
        with torch.no_grad():
            test_data = next(iter(eval_loader))[:1].to(device)
            
            # Encode to indices
            indices = model.encode(test_data)
            print(f"  Encoded indices shape: {indices.shape}")
            print(f"  Unique indices used: {len(torch.unique(indices))}/{config['num_embeddings']}")
            
            # Decode from indices
            reconstructed = model.decode_from_indices(indices)
            recon_error = F.mse_loss(reconstructed, test_data).item()
            print(f"  Reconstruction error from indices: {recon_error:.4f}")
        
        # Codebook usage statistics
        print("\nCodebook Usage Statistics:")
        with torch.no_grad():
            all_indices = []
            for data in eval_loader:
                data = data.to(device)
                indices = model.encode(data)
                all_indices.append(indices.flatten())
            
            all_indices = torch.cat(all_indices)
            unique_indices = torch.unique(all_indices)
            usage_rate = len(unique_indices) / config['num_embeddings'] * 100
            
            print(f"  Active codes: {len(unique_indices)}/{config['num_embeddings']} ({usage_rate:.1f}%)")
            
            # Histogram of code usage
            hist = torch.histc(all_indices.float(), bins=config['num_embeddings'], 
                              min=0, max=config['num_embeddings']-1)
            most_used = torch.argmax(hist).item()
            least_used = torch.argmin(hist[hist > 0]).item() if (hist > 0).any() else -1
            
            print(f"  Most used code: {most_used} (used {int(hist[most_used])} times)")
            if least_used >= 0:
                print(f"  Least used code: {least_used} (used {int(hist[least_used])} times)")
    
    print("\n" + "="*80)
    print("EXAMPLE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()