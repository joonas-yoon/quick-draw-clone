import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl


class CNNModel(pl.LightningModule):
    def __init__(self, output_classes: int, dropout=0.2):
        super().__init__()
        self.conv_layer = nn.Sequential(
            # (3, 32, 32)
            nn.Conv2d(3, 32, 2),  # 32, 31, 31
            nn.ReLU(),
            nn.Conv2d(32, 64, 2),  # 64, 30, 30
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64, 15, 15

            nn.Conv2d(64, 128, 3),  # 128, 13, 13
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),  # 256, 11, 11
            nn.ReLU(),
            nn.MaxPool2d(3, 2),  # 256, 5, 5
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 5 * 5, output_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y)
        acc = self.accuracy_score(y_hat, y)
        self.log_dict({'train_loss': loss, 'train_acc': acc})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y)
        acc = self.accuracy_score(y_hat, y)
        self.log_dict({'valid_loss': loss, 'valid_acc': acc})
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y)
        acc = self.accuracy_score(y_hat, y)
        self.log_dict({'test_loss': loss, 'test_acc': acc})
        return loss

    def accuracy_score(self, probs, y):
        return torch.sum(torch.argmax(probs, dim=1) == y) / len(y)

    def configure_optimizers(self):
        lr = self.hparams.lr
        print("model.configure_optimizers.lr =", lr)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.85)
        return [optimizer], [scheduler]
