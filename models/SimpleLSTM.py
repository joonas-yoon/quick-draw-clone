import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import numpy as np


class BaseLSTMModel(nn.Module):
    def __init__(self,
                 strokes: int,
                 out_classes: int,
                 dropout: float = 0.2,
                 batch_norm: bool = False,
                 device: str = 'cpu'):
        super().__init__()

        self.n_layers = 2
        self.h_n = strokes

        conv_layers = [
            nn.BatchNorm1d(num_features=strokes) if batch_norm else None,
            nn.Conv1d(in_channels=strokes, out_channels=64, kernel_size=3),
            nn.Dropout(p=dropout),
            nn.Conv1d(in_channels=64, out_channels=96, kernel_size=1),
            nn.Dropout(p=dropout),
            nn.Conv1d(in_channels=96, out_channels=128, kernel_size=1),
            nn.Dropout(p=dropout),
        ]
        conv_layers = list(filter(bool, conv_layers))

        self.conv = nn.Sequential(*conv_layers)
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=self.h_n,
                             num_layers=self.n_layers, dropout=0.2)
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

        self.device = device

    def forward(self, x):
        # Conv 1D layer
        x = self.conv(x)
        bs = x.size(0)
        x = x.view(bs, -1)

        hidden_state = (
            torch.zeros_like(torch.FloatTensor(
                self.n_layers, self.h_n)).to(self.device),
            torch.zeros_like(torch.FloatTensor(
                self.n_layers, self.h_n)).to(self.device),
        )

        # LSTM layer
        x, (h1, c1) = self.lstm1(x, hidden_state)
        x = F.dropout(x, p=0.1)
        x, (h2, c2) = self.lstm2(x, (h1, c1))
        x = F.dropout(x, p=0.1)
        x = self.fc(x)

#         # SoftMax if not using CrossEntropyLoss
#         x = F.softmax(x, dim=1)
        return x


class SimpleLSTM(BaseLSTMModel):
    def __init__(self,
                 strokes: int,
                 out_classes: int,
                 dropout: float = 0.2,
                 device: str = 'cpu'):
        super().__init__(
            strokes=strokes,
            out_classes=out_classes,
            dropout=dropout,
            batch_norm=False,
            device=device,
        )


class SimpleLSTMBn(BaseLSTMModel):
    def __init__(self,
                 strokes: int,
                 out_classes: int,
                 dropout: float = 0.2,
                 device: str = 'cpu'):
        super().__init__(
            strokes=strokes,
            out_classes=out_classes,
            dropout=dropout,
            batch_norm=True,
            device=device,
        )
