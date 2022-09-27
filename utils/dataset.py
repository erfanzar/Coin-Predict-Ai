import sys
from abc import ABC

import pandas as pd
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

pr = sys.stdout.write


class Ds(Dataset):
    def __init__(self, x: [list, dict], y: [list, dict], legacy: bool = False):
        super(Ds, self).__init__()
        self.x = x
        self.y = y
        self.legacy = legacy

    def __len__(self):
        return [len(self.x[v]) for v in self.x.keys()][0] if self.legacy else len(self.x)

    def __getitem__(self, item):
        if self.legacy:
            x = [{'date_time': dt, 'market_price': mp, 'total_price': tp} for dt, mp, tp in
                 zip(self.x['date_time'], self.x['market_price'], self.x['total_price'])]
            y = [{'y_price': yp} for yp in self.y['y_price']]
            # pr(f"{[x[item]['date_time'], x[item]['market_price'].item(), x[item]['total_price'].item()]}")
            x, y = [x[item]['date_time'], x[item]['market_price'].item(), x[item]['total_price'].item()], y
        else:
            x, y = self.x[item], self.y[item]
        return x, y


class DataLoaderLightning(LightningDataModule):
    def __init__(self, path: str, batch_train: int = 32, batch_val: int = 32):
        super(DataLoaderLightning, self).__init__()
        self.y_t = None
        self.x_t = None
        self.y_v = None
        self.x_v = None
        self.x = []
        self.y = []
        self.batch_train = batch_train
        self.batch_val = batch_val
        self.data = pd.read_csv(path)
        self.setup()
        self.limit = int(len(self.x) * 0.8)

    @staticmethod
    def time_to_tensor(input: str):
        yy, mm, dd = input[0:4], input[5:7], input[8:10]
        yl, ml, dl = torch.zeros(1), torch.zeros(12), torch.zeros(31)
        dl[int(dd) - 1], ml[int(mm) - 1], yl[0] = 1, 1, int(yy)
        return [yl, ml, dl]

    def setup(self, stage) -> None:
        d, p, t, m, _ = [self.data[f'{v}'].values for v in self.data.keys()]

        tp, yp, mp, dt = [], [], [], []
        pmx = max(p)
        tmx = max(t)
        mmx = max(m)
        for index, (da, pa, ta, ma) in enumerate(zip(d, p, t, m)):
            pr(f'\r Moving Data Around Pass : % {(index / len(d)) * 100:.1f}')
            da, pa, ta, ma = self.time_to_tensor(input=da), torch.tensor(pa / pmx), torch.tensor(
                ta / tmx), torch.tensor(ma / mmx)
            dt.append(da)
            yp.append(pa)
            tp.append(ta)
            mp.append(ma)
            self.x.append({'date_time': da, 'market_price': ma, 'total_price': ta})
            self.y.append({'y_price': pa})
        pr('\n')
        limit = int(len(d) / 1.5)
        self.x_t = {'date_time': dt[0:limit], 'market_price': mp[0:limit], 'total_price': tp[0:limit]}
        self.y_t = {'y_price': yp[0:limit], }
        self.x_v = {'date_time': dt[limit:], 'market_price': mp[limit:], 'total_price': tp[limit:]}
        self.y_v = {'y_price': yp[limit:], }

    def train_dataloader(self):
        data_l = Ds(x=self.x[:self.limit], y=self.y[:self.limit])
        return DataLoader(data_l, batch_size=self.batch_train)

    def val_dataloader(self):
        data_l = Ds(x=self.x[self.limit:], y=self.y[self.limit:])
        return DataLoader(data_l, batch_size=self.batch_val)


class DataLoaderTorch(Dataset, ABC):
    def __init__(self, path: str, batch_train: int = 32, batch_val: int = 32):
        super(DataLoaderTorch, self).__init__()
        self.y_t = None
        self.x_t = None
        self.y_v = None
        self.x_v = None
        self.x = []
        self.y = []
        self.batch_train = batch_train
        self.batch_val = batch_val
        self.data = pd.read_csv(path)
        self.setup()
        self.limit = int(len(self.x) * 0.8)

    @staticmethod
    def time_to_tensor(input: str):
        yy, mm, dd = input[0:4], input[5:7], input[8:10]
        yl, ml, dl = torch.zeros(1), torch.zeros(12), torch.zeros(31)
        dl[int(dd) - 1], ml[int(mm) - 1], yl[0] = 1, 1, int(yy)
        return [yl, ml, dl]

    def setup(self) -> None:
        d, p, t, m, _ = [self.data[f'{v}'].values for v in self.data.keys()]

        tp, yp, mp, dt = [], [], [], []
        pmx = max(p)
        tmx = max(t)
        mmx = max(m)
        for index, (da, pa, ta, ma) in enumerate(zip(d, p, t, m)):
            pr(f'\r Moving Data Around Pass : % {(index / len(d)) * 100:.1f}')
            da, pa, ta, ma = self.time_to_tensor(input=da), torch.tensor(pa / pmx), torch.tensor(
                ta / tmx), torch.tensor(ma / mmx)
            dt.append(da)
            yp.append(pa)
            tp.append(ta)
            mp.append(ma)
            self.x.append({'date_time': da, 'market_price': ma, 'total_price': ta})
            self.y.append({'y_price': pa})
        pr('\n')
        limit = int(len(d) / 1.5)
        self.x_t = {'date_time': dt[0:limit], 'market_price': mp[0:limit], 'total_price': tp[0:limit]}
        self.y_t = {'y_price': yp[0:limit], }
        self.x_v = {'date_time': dt[limit:], 'market_price': mp[limit:], 'total_price': tp[limit:]}
        self.y_v = {'y_price': yp[limit:], }

    def train_dataloader(self):
        data_l = Ds(x=self.x[:self.limit], y=self.y[:self.limit])
        return DataLoader(data_l, batch_size=self.batch_train)

    def val_dataloader(self):
        data_l = Ds(x=self.x[self.limit:], y=self.y[self.limit:])
        return DataLoader(data_l, batch_size=self.batch_val)
