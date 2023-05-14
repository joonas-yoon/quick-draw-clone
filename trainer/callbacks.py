import numpy as np


class EarlyStopper:
    def __init__(self, patience: int = 1, threshold: float = 0.0) -> None:
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.min_validation_loss = np.inf

    def check(self, validation_loss) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.threshold):
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
