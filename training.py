import torch
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import medpy.metric as med

from config import get_config
from loss import BinaryWeightedFocalTverskyLoss
from utils import EarlyStopper, preprocess_data, create_data_loaders





def train_one_epoch(model, loader, optimizer, loss_fn, config):
    model.train(True)
    running_loss = 0.0

    for data in tqdm(loader):
        inputs, labels = data
        optimizer.zero_grad()

        output = model(inputs)

        # Compute combined loss
        loss1 = loss_fn['bce'](output, labels)
        loss2 = loss_fn['bin_weighted_focal_tversky'](output, labels)
        loss3 = loss_fn['dice'](output, labels)

        total_loss = (config.loss_weights['bce'] * loss1 +
                      config.loss_weights['bin_weighted_focal_tversky'] * loss2 +
                      config.loss_weights['dice'] * loss3)

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    return running_loss / len(loader)


def train_model(model, X_train, y_train, X_val, y_val):
    config = get_config()

    # Preprocess data
    X_train, y_train = preprocess_data(X_train, y_train)
    X_val, y_val = preprocess_data(X_val, y_val)

    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        X_train, y_train, X_val, y_val, X_val, y_val, config
    )

    # Loss functions
    loss_fn = {
        'bce': smp.losses.SoftBCEWithLogitsLoss(),
        'bin_weighted_focal_tversky': BinaryWeightedFocalTverskyLoss(),
        'dice': smp.losses.DiceLoss('binary', log_loss=True)
    }

    # Optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.85, verbose=True
    )

    # Early stopper
    early_stopper = EarlyStopper(
        patience=config.early_stopping['patience'],
        min_delta=config.early_stopping['min_delta']
    )

    # Training loop
    best_vloss = float('inf')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for epoch in range(config.epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, config)

        # Validation
        model.eval()
        with torch.no_grad():
            running_vloss = 0.0
            pred, labe = [], []

            for vinputs, vlabels in val_loader:
                voutput = model(vinputs)
                voutput = torch.sigmoid(voutput)
                voutput_binary = (voutput > 0.5).float()

                vloss1 = loss_fn['bce'](voutput_binary, vlabels)
                vloss2 = loss_fn['bin_weighted_focal_tversky'](voutput_binary, vlabels)
                vloss3 = loss_fn['dice'](voutput_binary, vlabels)

                total_vloss = (config.loss_weights['bce'] * vloss1 +
                               config.loss_weights['bin_weighted_focal_tversky'] * vloss2 +
                               config.loss_weights['dice'] * vloss3)

                running_vloss += total_vloss

                pred.extend(voutput_binary.cpu().detach().numpy())
                labe.extend(vlabels.cpu().detach().numpy())

            avg_vloss = running_vloss / len(val_loader)
            dice_score = med.binary.dc(np.array(pred), np.array(labe))

            print(f'Epoch {epoch + 1}: Train Loss {avg_loss}, Val Loss {avg_vloss}, Dice Score {dice_score}')

            # Save best model
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = f'./save/modelTransUnet/model_{timestamp}_epoch{epoch}'
                torch.save(model.state_dict(), model_path)

            # Learning rate scheduling
            scheduler.step()

            # Early stopping
            if early_stopper.early_stop(avg_vloss):
                print("Early stopping triggered")
                break

    return model