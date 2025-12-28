import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from config import BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, EPOCHS
from model_architecture import HandCraftedRNN, BuiltINRnn
from name_dataset import NameDataset
from utils import DEVICE

dataset=NameDataset()
train_set,val_set=torch.utils.data.random_split(dataset,[0.80,0.20])

def dynamic_collate_fn(batch):
    data_tensors, label_tensors, labels, names = zip(*batch)

    lengths = torch.tensor(
        [t.size(0) for t in data_tensors],
        dtype=torch.long
    )

    padded_data = pad_sequence(
        data_tensors,
        batch_first=True,  # (batch, seq_len, input_size)
        padding_value=0.0
    )

    label_tensors = torch.stack(label_tensors)

    return padded_data, lengths, label_tensors, labels, names

train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True,collate_fn=dynamic_collate_fn)
val_loader=DataLoader(val_set,batch_size=BATCH_SIZE,shuffle=True,collate_fn=dynamic_collate_fn)
model=BuiltINRnn(input_size=INPUT_SIZE,hidden_size=HIDDEN_SIZE,num_classes=NUM_CLASSES)
optimizer=optim.Adam(model.parameters(),lr=0.0005)
criterion=nn.CrossEntropyLoss()


def train():
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for name_tensors, lengths, label_tensors, labels, names in train_loader:
        name_tensors = name_tensors.to(DEVICE)
        label_tensors = label_tensors.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(name_tensors, lengths)
        loss = criterion(predictions, label_tensors)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += predictions.argmax(dim=1).eq(label_tensors).sum().item()
        total += label_tensors.size(0)

    return total_loss / len(train_loader), correct / total


def validate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for name_tensors, lengths, label_tensors, labels, names in val_loader:
            name_tensors = name_tensors.to(DEVICE)
            label_tensors = label_tensors.to(DEVICE)

            predictions = model(name_tensors, lengths)

            correct += predictions.argmax(dim=1).eq(label_tensors).sum().item()
            total += label_tensors.size(0)

    return correct / total

if __name__=='__main__':
    print(len(dataset))
    for epoch in range(EPOCHS):
        loss,train_accuracy = train()
        val_accuracy=validate()
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss} |"
              f"Training Accuracy: {train_accuracy*100:.4f} | Validation accuracy: {val_accuracy*100:.4f}")
        if val_accuracy>.89:
            break

