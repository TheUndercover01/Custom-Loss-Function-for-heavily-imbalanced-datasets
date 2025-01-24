import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Training Hyperparameters
    config.learning_rate = 8e-07
    config.batch_size = 16
    config.epochs = 200
    config.optimizer = 'SGD'

    # Loss Weights
    config.loss_weights = {
        'bce': 0.6,
        'bin_weighted_focal_tversky': 0.4,
        'dice': 0.6
    }

    # Early Stopping
    config.early_stopping = {
        'patience': 8,
        'min_delta': 0
    }

    # Augmentation Parameters
    config.augmentation = {
        'horizontal_flip': 0.8,
        'vertical_flip': 0.8,
        'rotate90': 0.8,
        'transpose': 0.5,
        'elastic_transform': {
            'alpha': 0.2,
            'sigma': 0.5
        }
    }

    return config