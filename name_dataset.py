import torch
from torch.utils.data import Dataset
import os
import glob

from config import DATA_DIR, INPUT_SIZE
from data_processing import transform_text_to_tensor


class NameDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR):
        self.data_tensors = []
        self.labels = []
        self.labels_names = set()
        self.max_length = 0

        # First pass: collect label names
        for filename in glob.glob(os.path.join(data_dir, "*.txt")):
            label = os.path.splitext(os.path.basename(filename))[0]
            self.labels_names.add(label)

        # Label → index mapping
        self.label_to_idx = {
            label: idx for idx, label in enumerate(sorted(self.labels_names))
        }

        # Second pass: load data
        for filename in glob.glob(os.path.join(data_dir, "*.txt")):
            label_name = os.path.splitext(os.path.basename(filename))[0]
            label_idx = self.label_to_idx[label_name]

            lines = open(filename, encoding="utf-8").read().strip().split("\n")
            for name in lines:
                tensor = transform_text_to_tensor(name)
                self.data_tensors.append(tensor)
                self.labels.append(label_idx)
                self.max_length = max(self.max_length, tensor.size(0))

    def __len__(self):
        return len(self.data_tensors)

    def __getitem__(self, idx):
        x = self.data_tensors[idx]
        y = self.labels[idx]  # int

        # Padding
        pad_length = self.max_length - x.size(0)
        if pad_length > 0:
            padding = torch.zeros(
                pad_length, *x.shape[1:], dtype=x.dtype
            )
            x = torch.cat((x, padding), dim=0)

        return x, y
