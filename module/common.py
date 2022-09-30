import torch
import torch.nn as nn
import pytorch_lightning as pl

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.autograd.set_detect_anomaly(True)


class Linear(pl.LightningModule):
    def __init__(self, b1, b2):
        super(Linear, self).__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(b1, b2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.linear(x.float()))


class LSTM(pl.LightningModule):
    def __init__(self, input_size: int, input_batch: int, num_layer: int = 2, hidden_size: int = 15,
                 dv: str = 'cuda:0'):
        super(LSTM, self).__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.layer = nn.LSTM(input_size=input_size, num_layers=num_layer, hidden_size=hidden_size)

    def forward(self, x):

        x = torch.unsqueeze(x, 0) if len(x.shape) == 2 else x
        input_batch = x.shape[1]
        try:
            c0 = torch.zeros((self.num_layer, input_batch, self.hidden_size)).to(x.device)
            b0 = torch.zeros((self.num_layer, input_batch, self.hidden_size)).to(x.device)
            out, (c0, b0) = self.layer(x, (c0, b0))
            return out
        except IOError:
            print('except input DIM 3 but got {} DIM '.format(len(x.shape)))


class RepLinear(pl.LightningModule):
    def __init__(self, b1: int, b2: int, e: float = 2, n: int = 5, b1_prm: bool = False):
        super(RepLinear, self).__init__()
        self.save_hyperparameters()
        b_ = int(b1 * e) if b1_prm else int(b2 * e)
        self.layer = nn.Sequential(*(Linear(b1 if i == 0 else b_, b_ if i != n - 1 else b2) for i in range(n)))

    def forward(self, x):
        return self.layer(x)


class ResidualLinear(pl.LightningModule):
    def __init__(self, b: int, n: int = 2):
        super(ResidualLinear, self).__init__()
        self.save_hyperparameters()
        self.layers = nn.ModuleList()
        for i in range(n):
            self.layers.append(nn.Sequential(Linear(b, b), Linear(b, b)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return x
