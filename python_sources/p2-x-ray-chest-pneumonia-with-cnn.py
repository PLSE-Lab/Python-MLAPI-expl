#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install jcopdl')
get_ipython().system('pip install jcopml')


# In[ ]:


import jcopdl
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


import torch
from torch import nn, optim
from jcopdl.callback import Callback, set_config


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# # Dataset & Dataloader 

# In[ ]:


from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# In[ ]:


bs = 128
crop_size = 64

train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomSizedCrop(crop_size, scale=(0.6, 0.9)),
    transforms.ColorJitter(brightness=0.3),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(70),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor()
])

train_set = datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/train/", transform=train_transform)
trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4)

test_set = datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/val/", transform=test_transform)
testloader = DataLoader(test_set, batch_size=bs, shuffle=True)


# In[ ]:


feature, target = next(iter(trainloader))
feature.shape


# label category

# In[ ]:


label2cat = train_set.classes
label2cat


# # Architecture & Config

# In[ ]:


from jcopdl.layers import conv_block, linear_block


# In[ ]:


# conv_block = (
#     nn.Conv2d(3, 8, 3, 1, 1),
#     nn.ReLU(),
#     nn.MaxPool2d(2, 2)
# )


# In[ ]:


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            conv_block(3, 8),
            conv_block(8, 16),
            conv_block(16, 32),
            conv_block(32, 64), 
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            linear_block(1024, 256, dropout=0.1),
            linear_block(256, 2, activation="lsoftmax")
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# In[ ]:


config = set_config({
    "batch_size": bs,
    "crop_size": crop_size
})


# # Training Preparation --> MCOC 

# In[ ]:


model = CNN().to(device)
criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
callback = Callback(model, config, outdir="model")


# # Training

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


# # Predict

# In[ ]:


feature, target = next(iter(testloader))
feature, target = feature.to(device), target.to(device)


# In[ ]:


with torch.no_grad():
    model.eval()
    output = model(feature)
    preds = output.argmax(1)
preds


# # Visualization

# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


fig, axes = plt.subplots(6, 6, figsize=(24, 24))
for image, label, pred, ax in zip(feature, target, preds, axes.flatten()):
    ax.imshow(image.permute(1, 2, 0).cpu())
    font = {"color": 'r'} if label != pred else {"color": 'g'}        
    label, pred = label2cat[label.item()], label2cat[pred.item()]
    ax.set_title(f"L: {label} | P: {pred}", fontdict=font);
    ax.axis('off');


# # Save Model

# In[ ]:


from jcopml.utils import save_model


# In[ ]:


save_model(model, "xray_chest_pneumonia_v1.pkl")


# # Evaluation

# In[ ]:


test_set_final = datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/test", transform=test_transform)
testloader_final = DataLoader(test_set_final, batch_size=bs)

with torch.no_grad():
    test_cost_final, test_score_final = loop_fn("test", test_set_final, testloader_final, model, criterion, optimizer, device)
    print(f"Test Accuracy: {test_score_final}")    

