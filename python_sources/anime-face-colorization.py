#!/usr/bin/env python
# coding: utf-8

# ### imports

# In[ ]:


import torch
import torchvision.datasets  as dsets
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder

import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


# ### dataset

# In[ ]:


batch_size = 32
image_size = 56
torch.cuda.set_device("cuda:0")


# In[ ]:


class ColorizationDataset(Dataset):
    def __init__(self, transform_x, transform_y):
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self) -> int:
        return 21511

    def __getitem__(self, idx: int):
        with Image.open(f'/kaggle/input/anime-faces/data/data/{str(idx+1)}.png') as image:
            img = image.copy()
        Y = self.transform_y(img)
        X = self.transform_x(Y)
        return X, Y


# In[ ]:


transform_all = transforms.Compose([
    transforms.RandomResizedCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def to_grayscale(x):
    return (x[0] * 0.299 + x[1] * 0.587 + x[2] * 0.114).view(1, image_size, image_size)

dset = ColorizationDataset(to_grayscale, transform_all)
# cut the size of dataset
dataset, _ = torch.utils.data.random_split(dset, [int(len(dset)/3.3), len(dset)-int(len(dset)/3.3)])
del _
dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)


# In[ ]:


real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:32], padding=2, normalize=True).cpu(),(1,2,0)))


# ### network

# In[ ]:


class Colorizer(nn.Module):
    def __init__(self):
        super(Colorizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 2, 1)
        self.conv2 = nn.Conv2d(16, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 256, 3, 2, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 2, dilation=2)
        self.conv5 = nn.Conv2d(256, 256, 3, 1, 2, dilation=2)
        self.conv6 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.conv7 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.conv8 = nn.ConvTranspose2d(64, 3, 3, 2, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn7 = nn.BatchNorm2d(64)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        y = self.bn1(self.leakyrelu(self.conv1(x)))
        y = self.bn2(self.leakyrelu(self.conv2(y)))
        y = self.bn3(self.leakyrelu(self.conv3(y)))
        y = self.bn4(self.leakyrelu(self.conv4(y)))
        y = self.bn5(self.leakyrelu(self.conv5(y)))
        y = self.bn6(self.leakyrelu(self.conv6(y)))
        y = self.bn7(self.leakyrelu(self.conv7(y)))
        y = self.tanh(self.conv8(y))
        return y


# In[ ]:


num_epochs = 4
lr = 1e-3

model = Colorizer()
if os.path.isfile("/kaggle/input/anime-face-colorization/colorizer.pth"):
    model.load_state_dict(torch.load("/kaggle/input/anime-face-colorization/colorizer.pth"))

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.L1Loss()


# In[ ]:


history = []
for epoch in range(num_epochs):
    for x, y in tqdm(dataloader):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        history.append(loss)
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), "colorizer.pth")


# ### visualization

# In[ ]:


def to_numpy_image(img):
    return img.detach().cpu().view(3, image_size, image_size).transpose(0, 1).transpose(1, 2).numpy()

for t in range(3):
    img_gray, img_true = dataset[t]
    img_pred = model(img_gray.view(1, 1, image_size, image_size))
    img_pred = to_numpy_image(img_pred)
    plt.figure(figsize=(10,10))
    
    plt.subplot(141)
    plt.axis('off')
    plt.set_cmap('Greys')
    plt.imshow(1-img_gray.reshape((image_size, image_size)))

    plt.subplot(142)
    plt.axis('off')
    plt.imshow(img_pred.reshape((image_size, image_size, 3)))

    plt.subplot(143)
    plt.axis('off')
    plt.imshow(to_numpy_image(img_true))
    
    plt.show()


# In[ ]:




