#!/usr/bin/env python
# coding: utf-8

#  # Import Packages

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().system('pip install jcopdl #install jcopdl')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


import torch
from torch import nn, optim
from jcopdl.callback import Callback, set_config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# # Dataset dan Dataloader

# In[ ]:


from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# In[ ]:


bs = 64
crop_size = 224

data_train = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/"
data_test = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/"
data_val = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val/"

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(crop_size, scale=(0.7, 1)),
    transforms.ColorJitter(brightness=0.3,contrast=0.3),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(230),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = datasets.ImageFolder(data_train, transform=train_transform)
trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4)

test_set = datasets.ImageFolder(data_test, transform=test_transform)
testloader = DataLoader(test_set, batch_size=bs, shuffle=True)


# In[ ]:


feature, target = next(iter(trainloader))
feature.shape


# In[ ]:


label2cat = train_set.classes
label2cat


# # Arsitektur dan Config

# ## Model: MobilenetV2

# In[ ]:


from torchvision.models import mobilenet_v2

mnet = mobilenet_v2(pretrained=True) #Load pretrained model # pretrained = True | mendownload model berserta weightnya

for param in mnet.parameters(): # Freeze semua weight feature ekstraktornya
    param.requires_grad = False 


# ## Custom Classifier

# In[ ]:


mnet.classifier = nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
    
        nn.Linear(256,2),
        nn.LogSoftmax()
)


# In[ ]:


mnet


# ## Custom Class

# In[ ]:


class CustomMobilenetV2(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.mnet = mobilenet_v2(pretrained=True)
        self.freeze()
        self.mnet.classifier = nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
    
        nn.Linear(256,2),
        nn.LogSoftmax()
)
        
    def forward(self, x):
        return self.mnet(x)

    def freeze(self):
        for param in self.mnet.parameters():
            param.requires_grad = False
            
    def unfreeze(self):        
        for param in self.mnet.parameters():
            param.requires_grad = True


# ## Config

# In[ ]:


config = set_config({
    "output_size": len(train_set.classes),
    "batch_size": bs,
    "crop_size": crop_size
})


# # Fase 1: Adaptasi ( learning rate standard + patience kecil )

# In[ ]:


model = CustomMobilenetV2(config.output_size).to(device)
criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
callback = Callback(model, config, early_stop_patience=2, outdir="model")


# In[ ]:


from tqdm.auto import tqdm

def loop_fn(mode, dataset, dataloader, model, criterion, optimizer, device):
    if mode == "train":
        model.train()
    elif mode == "test":
        model.eval()
    cost = correct = 0
    for feature, target in tqdm(dataloader, desc=mode.title()):
        feature, target = feature.to(device), target.to(device)
        output = model(feature)
        loss = criterion(output, target)
        
        if mode == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        cost += loss.item() * feature.shape[0]
        correct += (output.argmax(1) == target).sum().item()
    cost = cost / len(dataset)
    acc = correct / len(dataset)
    return cost, acc


# In[ ]:


while True:
    train_cost, train_score = loop_fn("train", train_set, trainloader, model, criterion, optimizer, device)
    with torch.no_grad():
        test_cost, test_score = loop_fn("test", test_set, testloader, model, criterion, optimizer, device)
    
    # Logging
    callback.log(train_cost, test_cost, train_score, test_score)

    # Checkpoint
    callback.save_checkpoint()
        
    # Runtime Plotting
    callback.cost_runtime_plotting()
    callback.score_runtime_plotting()
    
    # Early Stopping
    if callback.early_stopping(model, monitor="test_score"):
        callback.plot_cost()
        callback.plot_score()
        break


# # Fase 2: Fine tuning ( learning rate dikecilkan + patience ditambah )

# In[ ]:


model.unfreeze()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

callback.reset_early_stop()
callback.early_stop_patience = 5


# In[ ]:


while True:
    train_cost, train_score = loop_fn("train", train_set, trainloader, model, criterion, optimizer, device)
    with torch.no_grad():
        test_cost, test_score = loop_fn("test", test_set, testloader, model, criterion, optimizer, device)
    
    # Logging
    callback.log(train_cost, test_cost, train_score, test_score)

    # Checkpoint
    callback.save_checkpoint()
        
    # Runtime Plotting
    callback.cost_runtime_plotting()
    callback.score_runtime_plotting()
    
    # Early Stopping
    if callback.early_stopping(model, monitor="test_score"):
        callback.plot_cost()
        callback.plot_score()
        break


# ## Prediksi

# In[ ]:


feature, target = next(iter(testloader))
feature, target = feature.to(device), target.to(device)


# In[ ]:


with torch.no_grad():
    model.eval()
    output = model(feature)
    preds = output.argmax(1)
preds

