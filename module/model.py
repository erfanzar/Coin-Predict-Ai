import torch
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, ModelSummary, ModelCheckpoint, EarlyStopping, \
    GradientAccumulationScheduler
from utils.dataset import DataLoaderLightning
from common import LSTM, Linear, RepLinear, ResidualLinear


class LS66V(pl.LightningModule):
    def __init__(self):
        super(LS66V, self).__init__()
        self.lstm_dd = LSTM(input_size=31, hidden_size=15, num_layer=1, input_batch=32)
        self.lstm_mm = LSTM(input_size=12, hidden_size=12, num_layer=1, input_batch=32)
        self.lstm_yy = LSTM(input_size=1, hidden_size=4, num_layer=1, input_batch=32)
        self.fc0 = Linear(1, 2)
        self.fc1 = Linear(1, 2)

    def forward(self, x):
        dd = self.lstm_dd(x['date_time'][2])
        mm = self.lstm_mm(x['date_time'][1])
        yy = self.lstm_yy(x['date_time'][0])

        mp = self.fc0(x['market_price'])
        tp = self.fc1(x['total_price'])

    def training_step(self, batch, index_batch):
        pass

    def validation_step(self, batch, index_batch):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        lr_lambda = lambda ep: 0.9 * ep
        lr_scheduler = optim.lr_scheduler.LambdaLR(lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]
