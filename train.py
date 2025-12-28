import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, EPOCHS
from model_architecture import HandCraftedRNN, BuiltINRnn
from name_dataset import NameDataset

dataset=NameDataset()
train_set,val_set=torch.utils.data.random_split(dataset,[0.85,0.15])
train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
val_loader=DataLoader(val_set,batch_size=BATCH_SIZE,shuffle=True)
model=BuiltINRnn(input_size=INPUT_SIZE,hidden_size=HIDDEN_SIZE,num_classes=NUM_CLASSES)
optimizer=optim.Adam(model.parameters())
criterion=nn.CrossEntropyLoss()
def train():
    model.train()
    total_loss=0
    correct=0
    total=0
    for name_tensors,label_tensors,labels,names in train_loader:
        optimizer.zero_grad()
        predictions:Tensor=model(name_tensors)
        loss=criterion(predictions,label_tensors)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        correct+=predictions.argmax(dim=1).eq(label_tensors).sum().item()
        total+=len(labels)

    return total_loss/len(train_loader),correct/total

def validate():
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for name_tensors,label_tensors,labels,names in val_loader:
            predictions=model(name_tensors)
            correct += predictions.argmax(dim=1).eq(label_tensors).sum().item()
            total += len(labels)
    return correct/total

if __name__=='__main__':
    print(len(dataset))
    for epoch in range(EPOCHS):
        loss,train_accuracy = train()
        val_accuracy=validate()
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss} |"
              f"Training Accuracy: {train_accuracy*100:.4f} | Validation accuracy: {val_accuracy*100:.4f}")
        if val_accuracy>89:
            break

