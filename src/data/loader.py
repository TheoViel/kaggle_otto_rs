import torch
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader

from utils.torch import worker_init_fn
from params import NUM_WORKERS


class LenMatchBatchSampler(BatchSampler):
    """
    Custom PyTorch Sampler that generate batches of similar length.
    Used alongside with trim_tensor, it helps speed up training.
    """
    def __init__(self, sampler, batch_size, drop_last, pad_token=0):
        super().__init__(sampler, batch_size, drop_last)
        self.pad_token = pad_token

    def __iter__(self):

        buckets = [[]] * 1000
        yielded = 0

        for idx in self.sampler:
            count_zeros = torch.sum(self.sampler.data_source[idx]['ids'] == self.pad_token)
            count_zeros = int(count_zeros / 10)
            if len(buckets[count_zeros]) == 0:
                buckets[count_zeros] = []

            buckets[count_zeros].append(idx)

            if len(buckets[count_zeros]) == self.batch_size:
                batch = list(buckets[count_zeros])
                yield batch
                yielded += 1
                buckets[count_zeros] = []

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


def define_loaders(
    train_dataset,
    val_dataset,
    batch_size=32,
    val_bs=32,
    use_len_sampler=False,
    pad_token=0
):
    """
    Builds data loaders.
    TODO

    Args:
        train_dataset (BCCDataset): Dataset to train with.
        val_dataset (BCCDataset): Dataset to validate with.
        samples_per_group (int, optional): Number of images to use per group. Defaults to 0.
        class_multipliers (dict, optional): Coefficients to increase class sampling. Defaults to {}.
        batch_size (int, optional): Training batch size. Defaults to 32.
        val_bs (int, optional): Validation batch size. Defaults to 32.

    Returns:
       DataLoader: Train loader.
       DataLoader: Val loader.
    """

    if use_len_sampler:
        len_sampler = LenMatchBatchSampler(
            RandomSampler(train_dataset), batch_size=batch_size, drop_last=True, pad_token=pad_token
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=len_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader
