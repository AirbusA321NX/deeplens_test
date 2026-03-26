import torch.nn as nn
from torchvision import models

def get_resnet18_model(num_classes=1, pretrained=False):
    """
    Returns a ResNet-18 model modified for binary classification.
    Binary classification (num_classes=1) uses BCEWithLogitsLoss.
    """
    model = models.resnet18(pretrained=pretrained)
    
    # Modify the fully connected layer for binary classification (1 output)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model

if __name__ == "__main__":
    import torch
    # Quick test
    model = get_resnet18_model()
    dummy_input = torch.randn(2, 3, 64, 64)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be (2, 1)
