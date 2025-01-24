import torch
import numpy as np
import segmentation_models_pytorch as smp

from training import train_model, create_data_loaders
from test import test_model

# Load data
X_train = np.load('X_train.npy')
y_train = np.load('Y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('Y_val.npy')
X_test = np.load('X_test.npy')
y_test = np.load('Y_test.npy')

# Initialize model (example with TransUNet)
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)
model = model.cuda()

# Train model
trained_model = train_model(model, X_train, y_train, X_val, y_val)

# Create test data loader
_, _, test_loader = create_data_loaders(
    torch.from_numpy(X_test),
    torch.from_numpy(y_test),
    torch.from_numpy(X_test),
    torch.from_numpy(y_test),
    torch.from_numpy(X_test),
    torch.from_numpy(y_test),
    config
)

# Test model
dice_score, predictions, labels = test_model(trained_model, test_loader)
print(f"Final Dice Score: {dice_score}")