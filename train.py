import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics as tm
import torch.optim as optim
from utils.dataset import DataLoaderTorch, DataLoaderLightning
from utils.test import dataloader_test

if __name__ == '__main__':
    c = DataLoaderTorch(path='data/bitcoin-cash.csv')
    train_data = c.train_dataloader()
    valid_data = c.val_dataloader()
    # dataloader_test(train_data)
