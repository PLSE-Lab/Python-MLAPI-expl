#!/usr/bin/env python
# coding: utf-8

# # [](http://)Pneumonia
# 
# is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli, Typically symptoms include some combination of productive or dry cough, chest pain, fever, and trouble breathing. Severity is variable.
# 
# Pneumonia is usually caused by infection with viruses or bacteria and less commonly by other microorganisms, certain medications and conditions such as autoimmune diseases.  Risk factors include other lung diseases such as cystic fibrosis, COPD, and asthma, diabetes, heart failure, a history of smoking, a poor ability to cough such as following a stroke, or a weak immune system. Diagnosis is often based on the symptoms and physical examination. Chest X-ray, blood tests, and culture of the sputum may help confirm the diagnosis. The disease may be classified by where it was acquired with community, hospital, or health care associated pneumonia. **[See the source](https://en.wikipedia.org/wiki/Pneumonia)**

# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Chest_X-ray_in_influenza_and_Haemophilus_influenzae_-_annotated.jpg/300px-Chest_X-ray_in_influenza_and_Haemophilus_influenzae_-_annotated.jpg"> 

# In[ ]:


get_ipython().system('pip install jcopdl')
get_ipython().system('pip install jcopml')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# COMMON PACKAGES
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from jcopdl.callback import Callback, set_config

import helper

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# # Dataset & Dataloader

# In[ ]:


# DATASET & DATALOADER
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# In[ ]:


bs = 64
crop_size = 224 #standart MobileNet v2

train_transform = transforms.Compose([
    transforms.RandomRotation(10), #rotation 10%
    transforms.RandomResizedCrop(crop_size, scale=(0.7, 1)), #max zoom 70% from data
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #standart MobileNet v2
])

test_transform = transforms.Compose([ #rule  MobileNet v2
    transforms.Resize(230), #256 replaced 230 so that the size is not far from CenterCrop 224
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) #rule MobileNet

train_set = datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/train", transform=train_transform) 
trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4)

test_set = datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/val", transform=test_transform)
testloader = DataLoader(test_set, batch_size=bs, shuffle=True)


# In[ ]:


len(train_set.classes)


# In[ ]:


feature, target = next(iter(trainloader))
feature.shape


# In[ ]:


#label category
label2cat = train_set.classes
label2cat


# # Arsitektur & Config

# In[ ]:


#how to use Pretrained-Models
from torchvision.models import mobilenet_v2

mnet = mobilenet_v2(pretrained=True) #True: download model & weightnya 

#freze weight
for param in mnet.parameters():
    param.requires_grad = False


# In[ ]:


mnet


# In[ ]:


#replacing classifier sequential
mnet.classifier = nn.Sequential(
    nn.Linear(1280, 2), #2 total class
    nn.LogSoftmax() 
)


# In[ ]:


#custom arsitektur
class CustomMobilenetV2(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.mnet = mobilenet_v2(pretrained=True) #arsitektur
        self.freeze()
        self.mnet.classifier = nn.Sequential(  
            #linear_block(1280, 1, activation="lsoftmax")
            nn.Linear(1280, output_size),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        return self.mnet(x)

    def freeze(self):
        for param in self.mnet.parameters():
            param.requires_grad = False
            
    def unfreeze(self):        
        for param in self.mnet.parameters():
            param.requires_grad = True        


# In[ ]:


config = set_config({
    "output_size": len(train_set.classes),
    "batch_size": bs,
    "crop_size": crop_size
})


# ## Phase 1: Adaptation (lr standard + patience low)

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


# ## Phase 2: Fine-tuning (lr low, patience increases)

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


# In[ ]:


test_set = datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/test/",transform=test_transform)
testloader = DataLoader(test_set,batch_size = bs)

with torch.no_grad():
        test_cost, test_score = loop_fn("test", test_set, testloader, model, criterion, optimizer, device)
        print(f"Test accuracy:{test_score}")


# ## Save Model

# In[ ]:


from jcopml.utils import save_model


# In[ ]:


save_model(model, "xray_chest_pneumonia_mobilenet_v2.pkl")


# ## Predict

# In[ ]:


feature, target = next(iter(testloader))
feature, target = feature.to(device), target.to(device)


# In[ ]:


with torch.no_grad():
    model.eval()
    output = model(feature)
    preds = output.argmax(1)
preds


# ## Visualization

# In[ ]:


fig, axes = plt.subplots(6, 6, figsize=(24, 24))
for image, label, pred, ax in zip(feature, target, preds, axes.flatten()):
    ax.imshow(image.permute(1, 2, 0).cpu())
    font = {"color": 'r'} if label != pred else {"color": 'g'}        
    label, pred = label2cat[label.item()], label2cat[pred.item()]
    ax.set_title(f"L: {label} | P: {pred}", fontdict=font);
    ax.axis('off');

