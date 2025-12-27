from torch.utils.data import Dataset
import os
import glob

from config import DATA_DIR
from data_processing import transform_text_to_tensor
from utils import DEVICE


class NameDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.data = []
        self.labels = []
        self.labels_names = set()
        self.data_tensors = []

        for filename in glob.glob(os.path.join(data_dir, "*.txt")):
            label = os.path.basename(filename)
            label=os.path.splitext(label)[0]
            self.labels_names.add(label)
            # Read whole txt file, remove trailing white space and then split content line by line
            lines=open(filename,encoding="utf-8").read().strip().split("\n")
            for name in lines:
                self.data.append(name)
                self.data_tensors.append(transform_text_to_tensor(name))
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_tensor = self.data_tensors[idx].to(DEVICE)
        return   data_tensor,self.labels[idx]