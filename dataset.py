import numpy as np
import torch
from torch.utils.data import Dataset

class SepsisDataset(torch.utils.data.Dataset):
    def __init__(self, texts, attention_masks, statics, ys):
        self.texts = texts
        self.attention_masks = attention_masks
        self.statics = statics
        self.ys = ys

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        attention_mask = self.attention_masks[idx]
        static_var = self.statics[idx]
        y = self.ys[idx]

        positive_idx = np.random.choice(np.where(self.ys == y)[0])
        positive_text = self.texts[positive_idx]
        positive_attention_mask = self.attention_masks[positive_idx]
        positive_static_var = self.statics[positive_idx]

        negative_idx = np.random.choice(np.where(self.ys != y)[0])
        negative_text = self.texts[negative_idx]
        negative_attention_mask = self.attention_masks[negative_idx]
        negative_static_var = self.statics[negative_idx]

        return (
        text, attention_mask, static_var,
        positive_text, positive_attention_mask, positive_static_var,
        negative_text, negative_attention_mask, negative_static_var,
        y
    )