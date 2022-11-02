import torch
import numpy as np
from torch.utils.data import Dataset


class PatientNoteDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, soft_labels=False):
        self.df = df
        self.max_len = max_len
        self.tokenizer = tokenizer

        self.texts = df['clean_text'].values
#         self.feature_text = df['feature_text'].values

        self.targets = df['target'].values

    def __getitem__(self, idx):
        text = self.texts[idx]
#         feature_text = self.feature_text[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
#             text,
            return_token_type_ids=True,
            return_offsets_mapping=False,
            return_attention_mask=False,
            truncation="only_first",
            max_length=self.max_len,
            padding='max_length',
        )
        # Targets
        token_target = self.targets[idx]

        return {
            "ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "token_type_ids": torch.tensor(encoding["token_type_ids"], dtype=torch.long),
            "target": torch.tensor(token_target, dtype=torch.float),
            "text": text,
        }

    def __len__(self):
        return len(self.texts)
