import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LensDataset(Dataset):
    def __init__(self, root_dir, class_map, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            class_map (dict): Mapping from directory name to class index (e.g., {'train_lenses': 1, 'train_nonlenses': 0}).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        for dir_name, label in class_map.items():
            # Check for direct path first, then join with root_dir
            if os.path.isabs(dir_name) and os.path.isdir(dir_name):
                dir_path = dir_name
            else:
                dir_path = os.path.join(root_dir, dir_name)
                
            if not os.path.isdir(dir_path):
                print(f"Warning: Directory {dir_path} not found. Skipping...")
                continue
                
            file_paths = glob.glob(os.path.join(dir_path, "*.npy"))
            for p in file_paths:
                self.samples.append((p, label))
                
        if len(self.samples) == 0:
            raise RuntimeError(f"No .npy files found in {root_dir} for classes {list(class_map.keys())}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image (64x64x3 float32)
        image = np.load(img_path)
        
        # Convert to float32 if not already
        image = image.astype(np.float32)
        
        # Transpose from (H, W, C) to (C, H, W) for PyTorch
        # Note: If the user's data is already (3, 64, 64), this will error if we assume (64, 64, 3).
        # Based on previous check, it was (64, 64, 3).
        if image.shape == (64, 64, 3):
            image = np.transpose(image, (2, 0, 1))
            
        # Convert to tensor
        image = torch.from_numpy(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # No normalization for now as we don't have stats yet, 
            # but we can add a simple 0-1 scaling if required.
        ])
    else:
        return None
