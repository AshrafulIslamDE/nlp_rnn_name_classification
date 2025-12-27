import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, EPOCHS
from model_architecture import HandCraftedRNN
from name_dataset import NameDataset
from utils import DEVICE

from torch.nn.utils.rnn import pad_sequence




dataset=NameDataset()
train_set,val_set=torch.utils.data.random_split(dataset,[0.8,0.2])
train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
val_loader=DataLoader(val_set,batch_size=BATCH_SIZE,shuffle=True)
model=HandCraftedRNN(input_size=INPUT_SIZE,hidden_size=HIDDEN_SIZE,num_classes=NUM_CLASSES)
optimizer=optim.Adam(model.parameters())
criterion=nn.CrossEntropyLoss()
def train():
    model.train()
    total_loss=0
    correct=0
    total=0
    for names,labels in train_loader:
        labels=labels.to(DEVICE)
        optimizer.zero_grad()
        predictions:Tensor=model(names)
        loss=criterion(predictions,labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        correct+=predictions.argmax(dim=1).eq(labels).sum().item()
        total+=len(labels.size(0))

    return total_loss/len(train_loader),correct/total

def validate():
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for names,labels in val_loader:
            labels=labels.to(DEVICE)
            predictions=model(names)
            correct += predictions.eq(labels).sum().item()
            total += len(labels.size(0))
    return correct/total

if __name__=='__main__':
    print(len(dataset))
    for epoch in range(EPOCHS):
        loss,train_accuracy = train()
        val_accuracy=validate()
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss} |"
              f"Training Accuracy: {train_accuracy*100:.4f} | Validation accuracy: {val_accuracy*100:.4f}")

