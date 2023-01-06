import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler, RandomSampler, WeightedRandomSampler

from utils.torch import worker_init_fn
from params import NUM_WORKERS


class LenMatchBatchSampler(BatchSampler):
    """
    Custom PyTorch Sampler that generate batches of similar length.
    Helps speed up training.
    """
    def __iter__(self):
        buckets = [[]] * 1000
        yielded = 0

        for idx in self.sampler:
            bucket_id = self.sampler.data_source[idx][0].size(-1) // 20
            if len(buckets[bucket_id]) == 0:
                buckets[bucket_id] = []

            buckets[bucket_id].append(idx)

            if len(buckets[bucket_id]) == self.batch_size:
                batch = list(buckets[bucket_id])
                yield batch
                yielded += 1
                buckets[bucket_id] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

        assert len(self) == yielded, f"Expected {len(self)}, but yielded {yielded} batches"


def custom_collate_fn(batch):

    y = torch.stack([b[1] for b in batch])
    y_aux = torch.stack([b[2] for b in batch])

#     print([b[0].size(-1) for b in batch])
    max_w = np.max([b[0].size(2) for b in batch])
    max_h = np.max([b[0].size(1) for b in batch])

    size = (len(batch), batch[0][0].size(0), max_h, max_w)
    x = torch.zeros(size, device=y.device)
    
    for i, b in enumerate(batch):
        mid = (max_w - b[0].size(-1)) // 2
        x[i, :, :b[0].size(1), mid : mid + b[0].size(2)] = b[0]

    return x, y, y_aux


class BalancedSampler(Sampler):
    """
    By Heng.
    """
    def __init__(self, dataset, batch_size, n_pos=1):
        self.r = batch_size - n_pos
        self.dataset = dataset
        self.pos_index = np.where(dataset.targets > 0)[0]
        self.neg_index = np.where(dataset.targets == 0)[0]

        self.length = self.r * int(np.floor(len(self.neg_index) / self.r))

    def __iter__(self):
        pos_index = self.pos_index.copy()
        neg_index = self.neg_index.copy()
        np.random.shuffle(pos_index)
        np.random.shuffle(neg_index)

        neg_index = neg_index[:self.length].reshape(-1, self.r)
        pos_index = np.random.choice(pos_index, self.length // self.r).reshape(-1, 1)

        index = np.concatenate([pos_index, neg_index], -1).reshape(-1)
        return iter(index)

    def __len__(self):
        return self.length


def define_loaders(
    train_dataset,
    val_dataset,
    batch_size=32,
    val_bs=32,
    use_weighted_sampler=False,
    use_len_sampler=False,
    use_balanced_sampler=False,
    use_custom_collate=False,
):
    """
    Builds data loaders.
    TODO
    Args:
        train_dataset (BCCDataset): Dataset to train with.
        val_dataset (BCCDataset): Dataset to validate with.
        batch_size (int, optional): Training batch size. Defaults to 32.
        val_bs (int, optional): Validation batch size. Defaults to 32.
    Returns:
       DataLoader: Train loader.
       DataLoader: Val loader.
    """
    collate_fn = custom_collate_fn if use_custom_collate else None

    sampler = None
    if use_weighted_sampler:
        sampler = WeightedRandomSampler(
            train_dataset.sample_weights,
            len(train_dataset),
            replacement=True,
        )
    elif use_balanced_sampler:
        sampler = BalancedSampler(train_dataset, batch_size, n_pos=1)

    if use_len_sampler:
        len_sampler = LenMatchBatchSampler(
            RandomSampler(train_dataset) if sampler is None else sampler,  # weighted sampler may not work
            batch_size=batch_size,
            drop_last=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=len_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
            persistent_workers=True,
        )

    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            drop_last=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
            persistent_workers=True,
        )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_bs,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True,
        )

    return train_loader, val_loader
