import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import LensDataset, get_transforms
from model import get_resnet18_model
from metrics import calculate_and_print_metrics, plot_evaluation_results, find_optimal_threshold

def evaluate_model(model_path, test_dir, batch_size=32, device='cuda'):
    # Define class mappings
    test_class_map = {'test_lenses': 1, 'test_nonlenses': 0}
    
    # Dataset and Loader
    test_dataset = LensDataset(test_dir, test_class_map, transform=get_transforms(train=False))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Load Model
    model = get_resnet18_model()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    print(f"Evaluating model {model_path} on {device}...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            
            # sigmoid probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.flatten())
            
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Automated threshold tuning to reduce false positives
    best_threshold = find_optimal_threshold(all_labels, all_probs)
    final_preds = (all_probs >= best_threshold).astype(int)
    
    _ = calculate_and_print_metrics(all_labels, final_preds, all_probs, threshold=best_threshold)
    plot_evaluation_results(all_labels, final_preds, all_probs, threshold=best_threshold, save_path='evaluation_results.png')

if __name__ == "__main__":
    # Example usage: Change to the actual model path after training
    root = "e:/test/lens-finding-test/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Find the latest model checkpoint if it exists
    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    if model_files:
        latest_model = sorted(model_files)[-1]
        try:
            evaluate_model(latest_model, root, device=device)
        except Exception as e:
            print(f"Error during evaluation: {e}")
    else:
        print("No model files (.pth) found in current directory. Run training first.")
