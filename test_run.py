import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import get_resnet18_model

def quick_proof():
    print("--- QUICK PROOF OF EXECUTION ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Create dummy data (100 samples of 64x64x3)
    # Shape: (N, C, H, W)
    images = torch.randn(100, 3, 64, 64)
    labels = torch.randint(0, 2, (100, 1)).float()
    
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    # 2. Load Model
    model = get_resnet18_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 3. One small epoch
    model.train()
    print("Starting small-scale training...")
    for i, (imgs, lbs) in enumerate(loader):
        imgs, lbs = imgs.to(device), lbs.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbs)
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0:
            print(f"Batch {i+1}/10 - Loss: {loss.item():.4f}")
            
    print("--- PROOF COMPLETE: MODEL IS TRAINING SUCCESSFULLY ON DEVICE ---")

if __name__ == "__main__":
    quick_proof()
