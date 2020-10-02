#!/usr/bin/env python
# coding: utf-8

# # Fruit image classification

# ## Importing the modules

# In[ ]:


import torch
import numpy as np
import torchvision
import pandas as pd
import torch.nn as nn
from tqdm.notebook import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.utils import make_grid
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import transforms
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Preparing the data

# In[ ]:


transform_ds = transforms.Compose([
    transforms.RandomCrop(64, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize([.5, .5, .5], [.5, .5, .5])
])

ds = torchvision.datasets.ImageFolder(root="../input/fruits/fruits-360/Training", transform=transform_ds)


# In[ ]:


images, labels = ds[0]
print(images.shape)
plt.imshow(images.permute(1,2,0))
print(ds.classes[labels])


# In[ ]:


val_ds_size = 6769
train_ds_size = 60923

train_ds, val_ds = random_split(ds, [train_ds_size, val_ds_size])


# In[ ]:


batch_size = 128
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=3, pin_memory=True)


# In[ ]:


def show_batch(train_dl):
    for images, labels in train_dl:
        fig, ax = plt.subplots(figsize=(16,16))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1,2,0))
        break


# In[ ]:


show_batch(train_dl)


# In[ ]:


def accuracy(out, labels):
    _, preds = torch.max(out, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}
    
    def validation_epoch_end(self, outputs):
        batch_loss = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
            print("Epoch [{}], last_lr: {:.8f}, train_loss {:.4f}, val_loss {:.4f}, val_acc {:.4f}".format(
                epoch, result["lrs"][-1], result["train_loss"], result["val_loss"], result["val_acc"]))


# In[ ]:


def conv_block(input_channels, out_channels, pool=False):
    layers = [nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1, stride=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU()]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.conv1 = conv_block(input_channels, 64) #128, 64, 64, 64
        self.conv2 = conv_block(64, 128, pool=True) #128, 128, 32, 32
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128)) #128, 128, 32, 32
        
        self.conv3 = conv_block(128, 256, pool=True) #128, 256, 16, 16
        self.conv4 = conv_block(256, 512, pool=True) #128, 512, 8, 8
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512)) #128, 512, 8, 8
        
        self.classifier = nn.Sequential(nn.MaxPool2d(8), nn.Flatten(), nn.Linear(512, num_classes))
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
    
resnet_model = ResNet9(input_channels=3, num_classes=131)
resnet_model


# In[ ]:


class SiamoAllaFrutta(ImageClassificationBase):
    
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        
            nn.Conv2d(64, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 256, 16, 16
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 512, 8, 8
            
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 1024, 4, 4
            
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 2048, 2, 2
            
            nn.Flatten(),
            
            nn.Linear(8192, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 131))
        
    def forward(self, bx):
        return self.cnn(bx)


# In[ ]:


class FruitResnet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet18(pretrained=True)
        number_of_features = self.network.fc.in_features
        self.network.fc = nn.Linear(number_of_features, 131)
        
    def forward(self, xb):
        return self.network(xb)
    
    def freeze(self):
        for param in self.network.parameters():
            param.requires_grad = False
        for param in self.network.fc.parameters():
            param.requires_grad = True
            
    def unfreeze(self):
        for param in self.network.parameters():
            param.requires_grad = True


# In[ ]:


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def to_device(data, device):
    if isinstance (data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for x in self.dl:
            yield to_device(x, self.device)
    
    def __len__(self):
        return len(self.dl)


# In[ ]:


device = select_device()
device


# In[ ]:


train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)


# In[ ]:


for images, labels in train_dl:
    print(images.shape)
    print(labels)
    break


# In[ ]:


@torch.no_grad()
def evaluate(model, val_dl):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_dl]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    
def fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
                 weight_decay=0, grad_clip=None, opt_func=torch.optim.Adam):
    history = []
    
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr,
                                     epochs=epochs, steps_per_epoch=len(train_dl))
    
    for epoch in range(epochs):
        model.train()
        train_loss = []
        lrs = []
        for batch in tqdm(train_dl):
            loss = model.training_step(batch)
            train_loss.append(loss)
            loss.backward()
            
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            sched.step()
            
        result = evaluate(model, val_dl)
        result["train_loss"] = torch.stack(train_loss).mean().item()
        result["lrs"] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# ## Training

# In[ ]:


model = to_device(FruitResnet(), device)


# In[ ]:


history = [evaluate(model, val_dl)]
history


# In[ ]:


model.freeze()


# In[ ]:


epochs = 5
max_lr = 10e-4
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,\n                         weight_decay=weight_decay, grad_clip=grad_clip, \n                         opt_func=opt_func)\nhistory')


# In[ ]:


model.unfreeze()


# In[ ]:


epochs = 5
max_lr = 10e-5
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,\n                         weight_decay=weight_decay, grad_clip=grad_clip, \n                         opt_func=opt_func)\nhistory')


# ## Predictions

# In[ ]:


transform_test = transforms.Compose([transforms.ToTensor()])

test_ds = torchvision.datasets.ImageFolder(root="../input/fruits/fruits-360/Test", 
                                  transform=transform_test)


# In[ ]:


def prediction(images, model):
    xb = to_device(images.unsqueeze(0), device)
    out = model(xb)
    _, preds = torch.max(out, dim=1)
    return ds.classes[preds[0].item()]


# In[ ]:


image, label = test_ds[2000]
plt.imshow(image.permute(1,2,0))
print("Label:", test_ds.classes[label], "Prediction:", prediction(image,model))


# In[ ]:


image, label = test_ds[2900]
plt.imshow(image.permute(1,2,0))
print("Label:", test_ds.classes[label], "Prediction:", prediction(image,model))


# ## Model performance

# In[ ]:


def plot_accuracy(history):
    accuracy = [x["val_acc"] for x in history]
    plt.plot(accuracy, "-x")
    plt.title("Accuracy vs number of epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    
plot_accuracy(history)


# In[ ]:


def plot_losses(history):
    train_loss = [x.get("train_loss") for x in history]
    val_loss = [x["val_loss"] for x in history]
    plt.plot(train_loss, "-rx")
    plt.plot(val_loss, "-bx")
    plt.legend(["Training loss", "Validation loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    
plot_losses(history)


# In[ ]:




