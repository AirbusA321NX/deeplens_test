import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from dataset import LensDataset, get_transforms
from model import get_resnet18_model
from sklearn.model_selection import train_test_split

def train_model(root_dir, num_epochs=10, batch_size=32, lr=1e-4, device='cuda'):
    # Detect available directories
    available = os.listdir(root_dir)
    lenses_dir = next((d for d in available if 'train_lenses' in d and os.path.isdir(os.path.join(root_dir, d))), None)
    nonlenses_dir = next((d for d in available if 'train_nonlenses' in d and os.path.isdir(os.path.join(root_dir, d))), None)
    
    if not lenses_dir or not nonlenses_dir:
        # Try without 'train_' prefix
        lenses_dir = lenses_dir or next((d for d in available if 'lenses' in d and 'non' not in d and os.path.isdir(os.path.join(root_dir, d))), None)
        nonlenses_dir = nonlenses_dir or next((d for d in available if 'nonlenses' in d and os.path.isdir(os.path.join(root_dir, d))), None)

    if not lenses_dir or not nonlenses_dir:
        raise RuntimeError(f"Could not find 'lenses' and 'nonlenses' folders in {root_dir}. Found: {available}")

    print(f"Using Lenses: {lenses_dir}, Non-Lenses: {nonlenses_dir}")
    
    class_map = {lenses_dir: 1, nonlenses_dir: 0}
    full_dataset = LensDataset(root_dir, class_map, transform=get_transforms(train=True))
    
    # Split for validation if separate test folders are missing
    targets = [label for path, label in full_dataset.samples]
    train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=0.1, stratify=targets, random_state=42)
    
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    
    # Weighted Sampler for Imbalance
    train_labels = [full_dataset.samples[i][1] for i in train_idx]
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    class_counts = torch.bincount(train_labels_tensor)
    class_weights = (1. / class_counts.float())
    # Explicitly cast to tensor to satisfy IDE linter "float" false-positives
    weights_tensor = torch.as_tensor(class_weights)
    sample_weights = weights_tensor[train_labels_tensor.long()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model
    model = get_resnet18_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    log_file = "training_log.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "Time"])
        
    print(f"Starting training on {device}...")
    best_val_loss = float('inf')
    patience = 3
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
        train_loss = train_loss / len(train_dataset)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss = val_loss / len(val_dataset)
        
        epoch_time = time.time() - start_time
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, val_loss, epoch_time])
            
        # Checkpoint Management: Save latest and best
        current_model_path = f"lens_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), current_model_path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Remove previous best if it exists
            best_model_path = "best_lens_model.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! (Val Loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
            
        # Optional: Delete older epochs to save space (keep only latest and best)
        if epoch > 0:
            old_model = f"lens_model_epoch_{epoch}.pth"
            if os.path.exists(old_model):
                os.remove(old_model)

if __name__ == "__main__":
    root = "e:/test/lens-finding-test/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        train_model(root, num_epochs=10, batch_size=32, device=device)
    except Exception as e:
        print(f"Error during training: {e}")
