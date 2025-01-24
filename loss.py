import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryWeightedFocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1, epsilon=1e-7):
        super(BinaryWeightedFocalTverskyLoss, self).__init__()
        self.alpha = alpha  # False Positive weight
        self.beta = beta    # False Negative weight
        self.gamma = gamma  # Focal loss parameter
        self.epsilon = epsilon  # Small value to prevent division by zero

    def forward(self, inputs, targets):
        # Ensure inputs are probabilities
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Compute class imbalance weight
        num_positive = targets.sum()
        num_negative = targets.numel() - num_positive
        weight_positive = num_negative / (num_positive + self.epsilon)
        weight_negative = num_positive / (num_negative + self.epsilon)

        # Compute Tversky components
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        # Weighted Tversky index
        tversky_index = (TP + self.epsilon) / (
            TP +
            self.alpha * FP * weight_positive +
            self.beta * FN * weight_negative +
            self.epsilon
        )

        # Focal modification
        focal_tversky = (1 - tversky_index) ** self.gamma

        return focal_tversky