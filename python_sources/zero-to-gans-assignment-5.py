#!/usr/bin/env python
# coding: utf-8

# ### Import packages

# In[ ]:


get_ipython().system('pip install jovian -q')
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
get_ipython().run_line_magic('matplotlib', 'inline')


# ### data path and torch seeding

# In[ ]:


ROOT_PATH = '../input/jigsawpuzzle/data/'
TRAIN_PATH = ROOT_PATH + 'train/'
VALID_PATH = ROOT_PATH + 'valid/'
TEST_PATH = ROOT_PATH + 'test/'

TRAIN_CSV = ROOT_PATH + 'train.csv'
VALID_CSV = ROOT_PATH + 'valid.csv'
TEST_CSV = ROOT_PATH + 'test.csv'

BATCH_SIZE = 64
torch.manual_seed(12)


# ### convert label into 9x9 one hot tensor

# In[ ]:


def encode_label(label):
    target = torch.zeros(9, 9)
    for i in range(len(label)):
        target[i][int(label[i])] = 1.
    
    return target


# ### Class object for loading dataset

# In[ ]:


workers = cpu_count()
transform = transforms.Compose([transforms.ToTensor()])

class JigsawData(Dataset):
    def __init__(self, path, csv_file, transform=None):
        super().__init__()

        self.path = path
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_id, img_label = row['image'], row['label']
        img_fname = self.path + str(img_id)
        img = Image.open(img_fname)
        if self.transform:
            img = self.transform(img)

        return img, encode_label(img_label.split())


# ### Make train/valid/test dataset

# In[ ]:


train_dataset = JigsawData(TRAIN_PATH, TRAIN_CSV, transform)
valid_dataset = JigsawData(VALID_PATH, VALID_CSV, transform)
test_dataset = JigsawData(TEST_PATH, TEST_CSV, transform)


# ### Train/valid/test dataloader

# In[ ]:


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=workers, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, pin_memory=True, drop_last=True, num_workers=workers)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True, drop_last=True, num_workers=workers)


# ### Model for jigsaw puzzle

# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class Jigsaw(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout(0.2),
            
            nn.Flatten(),
            nn.Linear(28224, 81))

    def forward(self, x):
        out = self.model(x)
        return torch.reshape(out, (64, 9, 9))


# ### Training loop

# In[ ]:


model = Jigsaw().to(device)
optimizer = opt.Adam(model.parameters(), lr=0.01)
epochs = 10

train_loss_list = []
val_loss_list = []

def train(epoch):
    model.train()
    train_epoch_loss = 0

    for i, batch in enumerate(train_dataloader):
        imgs, labels = batch[0].to(device), batch[1].to(device)
        output = model(imgs)
        train_loss = F.binary_cross_entropy_with_logits(output, labels)
        train_epoch_loss += train_loss.item()
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 200 == 0:
            print(f'Epoch({epoch}): Step - {i} | Loss - {train_loss.item():.4f}')


    train_loss_list.append(train_epoch_loss/len(train_dataloader))

    print(f'==> Epoch {epoch} completed with avg loss - {(train_epoch_loss/len(train_dataloader)):.4f}')


def valid():
    model.eval()
    with torch.no_grad():
        val_epoch_loss = 0
        
        for i, batch in enumerate(valid_dataloader):
            imgs, labels = batch[0].to(device), batch[1].to(device)
            output = model(imgs)
            val_loss = F.binary_cross_entropy_with_logits(output, labels)
            val_epoch_loss += val_loss.item()

        val_loss_list.append(val_epoch_loss/len(valid_dataloader))
        print(f'==> Val loss - {(val_epoch_loss/len(valid_dataloader)):.4f}')


# ### Let's train!

# In[ ]:


print(f"==> Training Step: {len(train_dataloader)} | Validation Step: {len(valid_dataloader)}")
for epoch in range(1, epochs+1):
    train(epoch)
    valid()


# ### Let's plot!

# In[ ]:


plt.style.use('ggplot')
plt.plot(train_loss_list, label='train-loss')
plt.plot(val_loss_list, label='val-loss')
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title('Epochs VS Losses')
plt.legend()
plt.savefig('loss.png')
plt.show()


# ### Let's test

# In[ ]:


for i, batch in enumerate(test_dataloader):
    img, label = batch[0].to(device), batch[1].to(device)
    output = model(img)
    loss = F.binary_cross_entropy_with_logits(output, label)
    print(f'Test loss: {loss.item():.4f}')


# In[ ]:


import jovian
jovian.commit(project='zero-to-gans-assignment-5')


# In[ ]:




