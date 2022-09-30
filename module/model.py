import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
# from numba import Optional
from torch import Tensor

# from torch.optim import Optimizer

from .common import LSTM, Linear, RepLinear, ResidualLinear

torch.autograd.set_detect_anomaly(True)


class LS66V(pl.LightningModule):
    def __init__(self, using_batch):
        super(LS66V, self).__init__()
        # self.automatic_optimization = False
        self.using_batch = using_batch
        self.lstm_dd = LSTM(input_size=31, hidden_size=15, num_layer=1, input_batch=using_batch)
        self.lstm_mm = LSTM(input_size=12, hidden_size=12, num_layer=1, input_batch=using_batch)
        self.lstm_yy = LSTM(input_size=1, hidden_size=4, num_layer=1, input_batch=using_batch)
        self.fc0 = Linear(1, 12)
        self.fc1 = Linear(1, 12)
        # concat
        self.fc2_cc = Linear(24, 48)
        self.rep_fc0 = RepLinear(b1=48, b2=68, e=2, n=6)
        self.rep_fc1 = RepLinear(b1=31, b2=68, e=2, n=3)
        # concat
        self.rep_fc2_b = RepLinear(b1=136, b2=68, e=0.8, n=3)
        self.fc2 = Linear(68, 34)
        self.residual_fc0 = ResidualLinear(b=34, n=5)

        self.rep_fc3 = RepLinear(b1=34, b2=1, e=5, n=5, )

        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        dd = self.lstm_dd(x['date_time'][2])
        mm = self.lstm_mm(x['date_time'][1])
        yy = self.lstm_yy(x['date_time'][0])

        mp = self.fc0(x['market_price'].view(x['market_price'].shape[0], -1))
        tp = self.fc1(x['total_price'].view(x['market_price'].shape[0], -1))
        vs = torch.cat([yy, mm, dd], dim=-1).view(x['market_price'].shape[0], -1)
        xt_ = self.rep_fc1(vs)
        vst = self.fc2_cc(torch.cat([mp, tp], dim=-1).view(x['market_price'].shape[0], -1))
        xmt_ = self.rep_fc0(vst)
        tc1 = torch.cat([xt_, xmt_], dim=-1)
        tc2 = self.rep_fc2_b(tc1)
        x_ = self.rep_fc3(self.residual_fc0(self.fc2(tc2)))
        x_ = x_.view(x['market_price'].shape[0])
        return x_

    # def training_step(self, batch, index_batch):
    #     opt = self.optimizers()
    #     opt = opt.optimizer
    #     opt.zero_grad()
    #     x, y = batch
    #     x_, y = [self(x), y['y_price']]
    #     lr_scheduler = self.lr_schedulers()
    #
    #     loss = self.criterion(x_, y)
    #
    #     self.manual_backward(loss)
    #     self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
    #     opt.step()
    #     return loss

    def training_step(self, batch, index_batch):
        x, y = batch
        x_, y = [self(x), y['y_price']]
        loss = self.criterion(x_.float(), y.float())
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, index_batch):
        x, y = batch
        x_, y = [self(x), y['y_price']]
        loss = self.criterion(x_.float(), y.float())
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        lr_lambda = lambda ep: 0.9 * ep
        lr_scheduler = optim.lr_scheduler.LambdaLR(lr_lambda=lr_lambda, optimizer=optimizer)
        return [optimizer], [lr_scheduler]

    # def backward(
    #         self, loss: Tensor, optimizer, optimizer_idx, *args, **kwargs
    # ) -> None:
    #     loss.backward(retain_graph=True)

    def manual_backward(self, loss: Tensor, *args, **kwargs) -> None:
        # loss.backward(retain_graph=True)
        loss.backward()
