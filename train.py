import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics as tm
import torch.optim as optim
from utils.dataset import DataLoaderTorch, DataLoaderLightning
from utils.test import dataloader_test
from module.model import LS66V
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelSummary, ModelCheckpoint, Checkpoint, \
    BaseFinetuning, Timer, BackboneFinetuning

from pytorch_lightning.utilities import warnings

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    batch = 24
    nw = 2
    c = DataLoaderLightning(path='data/bitcoin-cash.csv', batch_val=batch, batch_train=batch, nw_train=nw, nw_val=nw)
    train_data = c.train_dataloader()
    valid_data = c.val_dataloader()
    # dataloader_test(train_data)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(monitor='val_loss', )
    model_summery = ModelSummary(max_depth=5)
    model_ckpt = ModelCheckpoint(dirpath='model/save/LS66V/', monitor='val_loss')
    ckpt = Checkpoint()
    timer = Timer(duration='01:00:00:00')
    # multiplicative = lambda epoch: 1.2
    # backbone_finetuning = BackboneFinetuning(300, multiplicative)
    trainer = pl.Trainer(callbacks=[
        lr_monitor, early_stopping, model_ckpt, model_summery, timer
    ], max_epochs=5000, min_epochs=50, enable_checkpointing=True, log_every_n_steps=batch,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu', auto_lr_find=False)
    net = LS66V(using_batch=batch)
    trainer.fit(net, train_dataloaders=train_data, val_dataloaders=valid_data)
