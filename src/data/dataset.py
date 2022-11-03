import torch
import numpy as np
from scipy.stats import poisson
from torch.utils.data import Dataset

from params import N_IDS

class OttoTrainDataset(Dataset):
    def __init__(self, df, max_len=100, max_trunc=20, train=True, test=False):
        self.df = df
        self.test = test
        self.train = train
        self.max_len = max_len
        self.max_trunc = max_trunc
        self.paths = df['path'].values
        
        self.mu = 25
        self.ps = poisson.pmf(np.arange(self.mu, 1000), mu=self.mu)
        self.ps[0] *= 1.5  # More 1s
        self.ps[9:] += 0.01  # Higher tail

    def pad(self, x):
        length = x.shape[0]
        if length > self.max_len:
            return x[: self.max_len]
        else:
            padded = np.zeros([self.max_len] + list(x.shape[1:]))
            padded[:length] = x
            return padded
        
    def truncate(self, x):
        if len(x) <= self.max_trunc:
            range_ = np.arange(1, len(x))
        elif len(x) > self.max_len:
            range_ = np.arange(self.max_len - self.max_trunc, self.max_len)
        else:
            range_ = np.arange(len(x) - self.max_trunc, len(x))
        
        ps = self.ps[:len(range_)].copy()
        ps /= ps.sum()

        if not self.train:  # deterministic
            np.random.seed(2020)
        trunc = np.random.choice(range_, p=ps)

        return x[:trunc], x[trunc:]
        
    @staticmethod
    def get_target(x):
        y = np.zeros((N_IDS, NUM_CLASSES), dtype=np.uint8)
        y[x[:, 0], x[:, 2]] = 1
        return y.copy()

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])

        if not self.test:
            x, x_target = self.truncate(x)
            y = self.get_target(x_target)

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
