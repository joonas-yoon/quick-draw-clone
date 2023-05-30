import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
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

        self.init_weights()

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

    def init_weights(self):
        # 초기화 하는 방법
        # 모델의 모듈을 차례대로 불러옵니다.
        for m in self.modules():
            # 만약 그 모듈이 nn.Conv2d인 경우
            if isinstance(m, nn.Conv2d):
                '''
                # 작은 숫자로 초기화하는 방법
                # 가중치를 평균 0, 편차 0.02로 초기화합니다.
                # 편차를 0으로 초기화합니다.
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

                # Xavier Initialization
                # 모듈의 가중치를 xavier normal로 초기화합니다.
                # 편차를 0으로 초기화합니다.
                init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
                '''

                # Kaming Initialization
                # 모듈의 가중치를 kaming he normal로 초기화합니다.
                # 편차를 0으로 초기화합니다.
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

            # 만약 그 모듈이 nn.Linear인 경우
            elif isinstance(m, nn.Linear):
                '''
                # 작은 숫자로 초기화하는 방법
                # 가중치를 평균 0, 편차 0.02로 초기화합니다.
                # 편차를 0으로 초기화합니다.
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

                # Xavier Initialization
                # 모듈의 가중치를 xavier normal로 초기화합니다.
                # 편차를 0으로 초기화합니다.
                init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
                '''

                # Kaming Initialization
                # 모듈의 가중치를 kaming he normal로 초기화합니다.
                # 편차를 0으로 초기화합니다.
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)


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
