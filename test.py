import torch
import numpy as np
from tqdm import tqdm
import medpy.metric as med


def test_model(model, test_loader):
    model.eval()
    pred, labe = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            output = model(inputs)
            output = torch.sigmoid(output)
            output_binary = (output > 0.5).float()

            pred.extend(output_binary.cpu().detach().numpy())
            labe.extend(labels.cpu().detach().numpy())

    # Calculate Dice Score
    dice_score = med.binary.dc(np.array(pred), np.array(labe))
    return dice_score, pred, labe
