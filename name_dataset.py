import torch
from torch.utils.data import Dataset
from torch import Tensor
from typing import List
import os
import glob


from config import DATA_DIR, INPUT_SIZE
from data_processing import transform_text_to_tensor
from utils import DEVICE


class NameDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.data = []
        self.labels = []
        self.labels_names = set()
        self.data_tensors:List[Tensor] = []
        self.label_tensors: List[Tensor] = []


        for filename in glob.glob(os.path.join(data_dir, "*.txt")):
            label = os.path.basename(filename)
            label=os.path.splitext(label)[0]
            self.labels_names.add(label)
            # Read whole txt file, remove trailing white space and then split content line by line
            lines=open(filename,encoding="utf-8").read().strip().split("\n")
            for name in lines:
                data_tensor=transform_text_to_tensor(name)
                self.data.append(name)
                self.data_tensors.append(data_tensor)
                self.labels.append(label)


        label_to_idx = {label: idx for idx, label in enumerate(sorted(self.labels_names))}
        for i,label in enumerate(self.labels):
            self.label_tensors.append(torch.tensor(label_to_idx[label],dtype=torch.long))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_tensor:Tensor = self.data_tensors[idx]
        data_tensor=data_tensor.to(DEVICE)
        label_tensor=self.label_tensors[idx].to(DEVICE)
        return   data_tensor,label_tensor,self.labels[idx],self.data[idx]

if __name__=="__main__":
    name_dataset=NameDataset()
    name=name_dataset[101]
    print(name.data_tensor.shape)
    print(name.label_tensor.s)
