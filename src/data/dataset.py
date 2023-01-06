import torch
import numpy as np
from scipy.stats import poisson
from torch.utils.data import Dataset

from params import N_IDS, NUM_CLASSES, CLASSES, CLS_TOKEN


class FeaturesDataset(Dataset):
    def __init__(self, df, target, features):
        self.df = df
        self.features = df[features].values
        self.targets = df[target].values if target is not None else np.zeros(len(df))

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def __len__(self):
        return len(self.df)


class OttoDataset(Dataset):
    def __init__(self, df, max_len=100, max_trunc=20, train=True, test=False):
        self.df = df
        self.test = test
        self.train = train
        self.max_len = max_len
        self.max_trunc = max_trunc
        self.paths = df["path"].values

        self.mu = 25
        self.ps = poisson.pmf(np.arange(self.mu, 1000), mu=self.mu)
        self.ps[0] *= 1.5  # More 1s
        self.ps[9:] += 0.01  # Higher tail

        self.tgt_dict = {}

    def pad(self, x):
        length = x.shape[0]
        if length > self.max_len:
            return x[: self.max_len]
        else:
            padded = np.zeros([self.max_len] + list(x.shape[1:]))
            padded[:length] = x
            return padded

    @staticmethod
    def add_cls_token(x):
        return np.concatenate([[CLS_TOKEN], x])

    def truncate(self, x, idx):
        if len(x) <= self.max_trunc:
            range_ = np.arange(1, len(x))
        elif len(x) > self.max_len:
            range_ = np.arange(self.max_len - self.max_trunc, self.max_len)
        else:
            range_ = np.arange(len(x) - self.max_trunc, len(x))

        ps = self.ps[: len(range_)].copy()
        ps /= ps.sum()

        if not self.train:  # deterministic
            np.random.seed(idx)
        trunc = np.random.choice(range_, p=ps)

        return x[:trunc], x[trunc:]

    def get_target(self, x, idx):
        y = np.zeros((N_IDS, NUM_CLASSES), dtype=np.uint8)

        y_dict = {"clicks": None, "carts": [], "orders": []}

        first = True
        for id_, _, c in x:
            if c > 1:
                y[id_, c] = 1
                y_dict[CLASSES[c]].append(id_)
            else:  # first clic only
                if first:
                    y[id_, 0] = 1
                    y_dict["clicks"] = id_
                    first = False

        if not self.train and not self.val:
            self.tgt_dict[idx] = y_dict

        return y

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])

        if not self.test:
            x, x_target = self.truncate(x, idx)
            y = self.get_target(x_target, idx)

        x = self.add_cls_token(x)
        x = self.pad(x)

        return {
            "ids": torch.tensor(x[:, 0], dtype=torch.long),
            "ts": torch.tensor(x[:, 1], dtype=torch.long),
            "token_type_ids": torch.tensor(x[:, 2], dtype=torch.long),
            "x_target": torch.tensor(x_target),
            "target": torch.from_numpy(y),
        }

    def __len__(self):
        return len(self.paths)
