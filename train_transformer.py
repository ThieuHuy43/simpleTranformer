import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time
from transformer_model import Transformer

class SimpleTranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data, src_vocab, tgt_vocab):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return {
            'src': torch.tensor(self.src_data[idx], dtype=torch.long),
            'tgt': torch.tensor(self.tgt_data[idx], dtype=torch.long)
        }

def create_sample_data():
    # Tạo dữ liệu mẫu đơn giản (số -> từ)
    src_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'one': 3, 'two': 4, 'three': 5, 'four': 6, 'five': 7}
    tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'mot': 3, 'hai': 4, 'ba': 5, 'bon': 6, 'nam': 7}
    
    # Dữ liệu mẫu mở rộng: English -> Vietnamese
    src_data = [
        [1, 3, 2, 0],  # <sos> one <eos> <pad>
        [1, 4, 2, 0],  # <sos> two <eos> <pad>
        [1, 5, 2, 0],  # <sos> three <eos> <pad>
        [1, 6, 2, 0],  # <sos> four <eos> <pad>
        [1, 7, 2, 0],  # <sos> five <eos> <pad>
        [1, 3, 2, 0],  # Duplicate for more data
        [1, 4, 2, 0],
        [1, 5, 2, 0],
    ]
    
    tgt_data = [
        [1, 3, 2, 0],  # <sos> mot <eos> <pad>
        [1, 4, 2, 0],  # <sos> hai <eos> <pad>
        [1, 5, 2, 0],  # <sos> ba <eos> <pad>
        [1, 6, 2, 0],  # <sos> bon <eos> <pad>
        [1, 7, 2, 0],  # <sos> nam <eos> <pad>
        [1, 3, 2, 0],
        [1, 4, 2, 0],
        [1, 5, 2, 0],
    ]
    
    return src_data, tgt_data, src_vocab, tgt_vocab

class LabelSmoothingLoss(nn.Module):
    """
    Hàm mất mát với Label Smoothing để tăng khả năng tổng quát
    """
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def calculate_accuracy(predictions, targets, ignore_index=0):
    """
    Tính accuracy, bỏ qua padding tokens
    """
    mask = targets != ignore_index
    correct = (predictions == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()

def calculate_bleu_score_simple(predictions, targets, vocab_size):
    """
    Tính BLEU score đơn giản (1-gram precision)
    """
    # Convert to numpy for easier processing
    pred_np = predictions.cpu().numpy()
    tgt_np = targets.cpu().numpy()
    
    total_matches = 0
    total_predictions = 0
    
    for pred_seq, tgt_seq in zip(pred_np, tgt_np):
        # Remove padding
        pred_clean = pred_seq[pred_seq != 0]
        tgt_clean = tgt_seq[tgt_seq != 0]
        
        matches = sum(1 for p in pred_clean if p in tgt_clean)
        total_matches += matches
        total_predictions += len(pred_clean)
    
    return total_matches / max(total_predictions, 1)

def plot_training_metrics(train_losses, train_accuracies, train_bleu_scores, epoch_times):
    """
    Trực quan hóa các metrics trong quá trình training
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Visualization', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot Loss
    axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_title('Training Loss Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')  # Log scale for better visualization
    
    # Plot Accuracy
    axes[0, 1].plot(epochs, train_accuracies, 'g-', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_title('Training Accuracy Over Time', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 100)
    
    # Plot BLEU Score
    axes[1, 0].plot(epochs, train_bleu_scores, 'r-', linewidth=2, marker='^', markersize=4)
    axes[1, 0].set_title('BLEU Score Over Time', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('BLEU Score')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Plot Training Time per Epoch
    axes[1, 1].bar(epochs, epoch_times, color='orange', alpha=0.7, width=0.8)
    axes[1, 1].set_title('Training Time per Epoch', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, vocab_dict, title='Confusion Matrix'):
    """
    Vẽ confusion matrix cho predictions
    """
    # Get label names (excluding special tokens for clarity)
    labels = [k for k, v in sorted(vocab_dict.items(), key=lambda x: x[1]) 
              if k not in ['<pad>', '<sos>', '<eos>']]
    label_indices = [vocab_dict[label] for label in labels]
    
    # Filter predictions and targets to only include non-special tokens
    mask = np.isin(y_true, label_indices) & np.isin(y_pred, label_indices)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    if len(y_true_filtered) == 0:
        print("No valid predictions for confusion matrix")
        return
    
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=label_indices)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_attention_weights(model, src_seq, tgt_seq, src_vocab, tgt_vocab):
    """
    Trực quan hóa attention weights (simplified version)
    """
    model.eval()
    with torch.no_grad():
        # Get attention weights from the last decoder layer
        src_mask, tgt_mask = model.generate_mask(src_seq.unsqueeze(0), tgt_seq.unsqueeze(0))
        
        # Forward pass to get attention weights
        # This is a simplified visualization
        src_tokens = [list(src_vocab.keys())[list(src_vocab.values()).index(i)] 
                     for i in src_seq if i != 0]
        tgt_tokens = [list(tgt_vocab.keys())[list(tgt_vocab.values()).index(i)] 
                     for i in tgt_seq if i != 0]
        
        print(f"Source: {' '.join(src_tokens)}")
        print(f"Target: {' '.join(tgt_tokens)}")

def evaluate_model(model, dataloader, criterion, tgt_vocab):
    """
    Đánh giá model trên test set
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src']
            tgt = batch['tgt']
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output = model(src, tgt_input)
            
            # Calculate loss
            output_flat = output.reshape(-1, output.size(-1))
            tgt_flat = tgt_output.reshape(-1)
            loss = criterion(output_flat, tgt_flat)
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(output, dim=-1)
            all_predictions.extend(predictions.reshape(-1).cpu().numpy())
            all_targets.extend(tgt_output.reshape(-1).cpu().numpy())
    
    # Calculate metrics
    accuracy = calculate_accuracy(torch.tensor(all_predictions), torch.tensor(all_targets))
    bleu_score = calculate_bleu_score_simple(torch.tensor(all_predictions), 
                                           torch.tensor(all_targets), len(tgt_vocab))
    
    return total_loss / len(dataloader), accuracy * 100, bleu_score

def train_transformer():
    # Hyperparameters
    d_model = 128
    num_heads = 8
    num_layers = 4
    d_ff = 512
    max_seq_length = 100
    dropout = 0.1
    num_epochs = 5
    
    # Tạo dữ liệu
    src_data, tgt_data, src_vocab, tgt_vocab = create_sample_data()
    
    # Split data into train/validation
    split_idx = int(0.8 * len(src_data))
    train_src, val_src = src_data[:split_idx], src_data[split_idx:]
    train_tgt, val_tgt = tgt_data[:split_idx], tgt_data[split_idx:]
    
    # Model
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function và optimizer
    criterion = LabelSmoothingLoss(classes=len(tgt_vocab), smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # Dataset và DataLoader
    train_dataset = SimpleTranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab)
    val_dataset = SimpleTranslationDataset(val_src, val_tgt, src_vocab, tgt_vocab)
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Tracking metrics
    train_losses = []
    train_accuracies = []
    train_bleu_scores = []
    val_losses = []
    val_accuracies = []
    val_bleu_scores = []
    epoch_times = []
    
    print("Starting training...")
    print("-" * 60)
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        total_train_loss = 0
        train_predictions = []
        train_targets = []
        
        for batch in train_dataloader:
            src = batch['src']
            tgt = batch['tgt']
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(src, tgt_input)
            
            # Calculate loss
            output_flat = output.reshape(-1, output.size(-1))
            tgt_flat = tgt_output.reshape(-1)
            loss = criterion(output_flat, tgt_flat)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Collect predictions for metrics
            predictions = torch.argmax(output, dim=-1)
            train_predictions.extend(predictions.reshape(-1).cpu().numpy())
            train_targets.extend(tgt_output.reshape(-1).cpu().numpy())
        
        # Calculate training metrics
        train_loss = total_train_loss / len(train_dataloader)
        train_acc = calculate_accuracy(torch.tensor(train_predictions), torch.tensor(train_targets)) * 100
        train_bleu = calculate_bleu_score_simple(torch.tensor(train_predictions), 
                                               torch.tensor(train_targets), len(tgt_vocab))
        
        # Validation phase
        val_loss, val_acc, val_bleu = evaluate_model(model, val_dataloader, criterion, tgt_vocab)
        
        # Record metrics
        epoch_time = time.time() - start_time
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_bleu_scores.append(train_bleu)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_bleu_scores.append(val_bleu)
        epoch_times.append(epoch_time)
        
        # Print progress
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch:3d}/{num_epochs} | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train BLEU: {train_bleu:.4f} | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val BLEU: {val_bleu:.4f} | '
                  f'Time: {epoch_time:.2f}s')
    
    print("-" * 60)
    print("Training completed!")
    
    # Final evaluation và visualization
    print("\nGenerating visualizations...")
    
    # Plot training metrics
    plot_training_metrics(train_losses, train_accuracies, train_bleu_scores, epoch_times)
    
    # Plot validation metrics comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train', linewidth=2)
    plt.plot(val_losses, label='Validation', linewidth=2)
    plt.title('Loss Comparison', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train', linewidth=2)
    plt.plot(val_accuracies, label='Validation', linewidth=2)
    plt.title('Accuracy Comparison', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(train_bleu_scores, label='Train', linewidth=2)
    plt.plot(val_bleu_scores, label='Validation', linewidth=2)
    plt.title('BLEU Score Comparison', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('train_val_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate confusion matrix
    print("Generating confusion matrix...")
    final_predictions = []
    final_targets = []
    
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            src = batch['src']
            tgt = batch['tgt']
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output = model(src, tgt_input)
            predictions = torch.argmax(output, dim=-1)
            
            final_predictions.extend(predictions.reshape(-1).cpu().numpy())
            final_targets.extend(tgt_output.reshape(-1).cpu().numpy())
    
    plot_confusion_matrix(np.array(final_targets), np.array(final_predictions), 
                         tgt_vocab, 'Final Model Confusion Matrix')
    
    # Final metrics summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS:")
    print(f"{'='*60}")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final Training BLEU: {train_bleu_scores[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")
    print(f"Final Validation BLEU: {val_bleu_scores[-1]:.4f}")
    print(f"Average Training Time per Epoch: {np.mean(epoch_times):.2f}s")
    print(f"{'='*60}")
    
    return model, src_vocab, tgt_vocab

if __name__ == "__main__":
    model, src_vocab, tgt_vocab = train_transformer()
    print("Training done")