class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def preprocess_data(X, y):
    # Standardization
    stdx = X.std()
    meanx = X.mean()
    X = (X - meanx) / stdx

    # Convert to PyTorch tensors
    X = torch.tensor(np.moveaxis(X, -1, 1))
    y = torch.tensor(np.moveaxis(y / 255, -1, 1))

    return X, y


def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, config):
    train_loader = DataLoader(
        list(zip(X_train.float().cuda(), y_train.float().cuda())),
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        list(zip(X_val.float().cuda(), y_val.float().cuda())),
        batch_size=config.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        list(zip(X_test.float().cuda(), y_test.float().cuda())),
        batch_size=config.batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader

