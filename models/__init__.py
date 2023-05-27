import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as L

from args import args as config


class LitLSTM(L.LightningModule):
    def __init__(self,
                 strokes: int,
                 out_classes: int,
                 dropout: float = 0.2):
        super().__init__()

        self.n_layers = 2
        self.h_n = strokes

        conv_layers = [
            nn.Conv1d(in_channels=strokes, out_channels=64, kernel_size=3),
            nn.Dropout(p=dropout),
            nn.Conv1d(in_channels=64, out_channels=96, kernel_size=1),
            nn.Dropout(p=dropout),
            nn.Conv1d(in_channels=96, out_channels=128, kernel_size=1),
            nn.Dropout(p=dropout),
        ]

        self.conv = nn.Sequential(*conv_layers)
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=self.h_n,
                             num_layers=self.n_layers, dropout=0.2, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.h_n, hidden_size=self.h_n,
                             num_layers=self.n_layers, dropout=0.2)
        self.fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=self.h_n, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=512, out_features=out_classes),
        )

        self.criterion = F.cross_entropy

    def forward(self, x):
        # Conv 1D layer
        x = self.conv(x)
        bs = x.size(0)
        x = x.view(bs, -1)

        # LSTM layer
        x, (h1, c1) = self.lstm1(x)
        x = F.dropout(x, p=0.1)
        x, (h2, c2) = self.lstm2(x, (h1, c1))
        x = F.dropout(x, p=0.1)
        x = self.fc(x)

        return x

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # implement your own
        log_probs: torch.Tensor = self(x)
        loss = self.criterion(log_probs, y)

        # calculate acc
        labels_hat = torch.argmax(log_probs, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # log the outputs!
        self.log_dict({'val_loss': loss, 'val_acc': val_acc})

    def configure_optimizers(self):
        try:
            lr = self.hparams.learning_rate or self.hparams.lr
        except AttributeError:
            lr = float(config.lr)
        print("model.configure_optimizers.lr =", lr)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.85)
        return [optimizer], [scheduler]
