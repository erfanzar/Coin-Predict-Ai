import torch
import torch.nn as nn
import pytorch_lightning as pl


class Linear(pl.LightningModule):
    def __init__(self, b1, b2):
        super(Linear, self).__init__()
        self.linear = nn.Linear(b1, b2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.linear(x))


class LSTM(pl.LightningModule):
    def __init__(self, input_size: int, input_batch: int, num_layer: int = 2, hidden_size: int = 15):
        super(LSTM, self).__init__()
        self.layer = nn.LSTM(input_size=input_size, num_layers=num_layer, hidden_size=hidden_size)
        self.c0 = torch.zeros((input_size, input_batch, hidden_size))
        self.b0 = torch.zeros((input_size, input_batch, hidden_size))

    def forward(self, x):
        out, (self.c0, self.b0) = self.layer(x, (self.c0, self.b0))
        return out


class RepLinear(pl.LightningModule):
    def __int__(self, b1: int, b2: int, e: int = 2, n: int = 5):
        b_ = int(b2 * e)
        self.layer = nn.Sequential(*(Linear(b1 if i == 0 else b_, b_ if i != n else b2) for i in range(n)))

    def forward(self, x):
        return self.layer(x)


class ResidualLinear(pl.LightningModule):
    def __init__(self, b: int, n: int = 2):
        super(ResidualLinear, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n):
            self.layers.append(nn.Sequential(Linear(b, b), Linear(b, b)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return x
