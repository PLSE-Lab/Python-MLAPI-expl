#!/usr/bin/env python
# coding: utf-8

# # OpenCV+albumentations is faster than PIL+torchvision
# You can speed up pytorch.utils.data.DataLoader by using cv2.imread + albumentations, instead of PIL.Image.open + torchvision.transforms.

# Based on an official report [here](https://github.com/albu/albumentations#benchmarking-results), this kernel shows some speed comparisons between cv2 method and pil method. Most of cases, cv2 method is faster than pil one.

# The work is now in progress. Feel free to post comments or suggestions. Thanks!

# # Some Settings

# In[ ]:


import os
import gc
import time

import cv2
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensor
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


INPUT_DIR = '../input/all-dogs/all-dogs'


# In[ ]:


image_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)]
print('total image files: {}'.format(len(image_files)))


# In[ ]:


def test(f, n_trials=5):
    elapsed_times = []
    for i in range(n_trials):
        t1 = time.time()
        f()
        t2 = time.time()
        elapsed_times.append(t2-t1)
    print('Mean: {:.3f}s - Std: {:.3f}s - Max: {:.3f}s - Min: {:.3f}s'.format(
        np.mean(elapsed_times),
        np.std(elapsed_times),
        np.max(elapsed_times),
        np.min(elapsed_times)
    ))


# # Simple Comparisons
# Due to memory constraint, using only 1,000 images.

# In[ ]:


image_files_1000 = image_files[:1000]


# ## Loading Speed
# - cv2: cv2.imread
# - PIL: PIL.Image.open

# In[ ]:


def cv2_imread(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

f = lambda: [cv2_imread(f) for f in image_files_1000]


# In[ ]:


test(f, n_trials=5)


# In[ ]:


f = lambda: [Image.open(f) for f in image_files_1000]


# In[ ]:


test(f, n_trials=5)


# In[ ]:


# use them following test

cv2_img_1000 = [cv2_imread(f) for f in image_files_1000]
pil_img_1000 = [Image.open(f) for f in image_files_1000]


# ## Resize Speed
# resize images to 64x64 by bilinear-interpolation.
# - cv2: albumentations.Resize
# - PIL: torchvision.transforms.Resize

# In[ ]:


cv2_transform = A.Compose([A.Resize(64, 64, interpolation=cv2.INTER_LINEAR)])

f = lambda: [cv2_transform(image=img)['image'] for img in cv2_img_1000]


# In[ ]:


test(f, n_trials=5)


# In[ ]:


pil_transform = transforms.Compose([transforms.Resize((64, 64), interpolation=2)])

f = lambda: [pil_transform(img) for img in pil_img_1000]


# In[ ]:


test(f, n_trials=5)


# In[ ]:


# use them following test

cv2_img_1000 = [cv2_transform(image=img)['image'] for img in cv2_img_1000]
pil_img_1000 = [pil_transform(img) for img in pil_img_1000]


# ## ToTensor Speed
# - cv2: albumentations.pytorch.ToTensor
# - PIL: torchvision.ToTensor

# In[ ]:


cv2_transform = A.Compose([ToTensor()])

f = lambda: [cv2_transform(image=img)['image'] for img in cv2_img_1000]


# In[ ]:


test(f, n_trials=5)


# In[ ]:


pil_transform = transforms.Compose([transforms.ToTensor()])

f = lambda: [pil_transform(img) for img in pil_img_1000]


# In[ ]:


test(f, n_trials=5)


# # DataLoader Comparisons
# Due to time constraints, using only 1,000 images.

# In[ ]:


image_files_1000 = image_files[:1000]


# ## Assumptions
# - Using pytorch. You can get image data as torch.Tensor through torch.utils.data.DataLoader.
# - Limited memory size. You cannot load all image data on memory so use load method.
# - Image size differs. MUST Resize.

# In[ ]:


class BaseDataset(Dataset):
    def __init__(self, files, transform=None):
        super().__init__()
        self.files = files
        self.transform = transform
    
    def __len__(self):
        return len(self.files)


class PILDataset(BaseDataset):
    def __getitem__(self, idx):
        file = self.files[idx]
        img = Image.open(file)
        if self.transform is not None:
            img = self.transform(img)
            
        return img

    
class CV2Dataset(BaseDataset):
    def __getitem__(self, idx):
        file = self.files[idx]
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)['image']
            
        return img


# In[ ]:


def dataloader_test(files, transform, test_type='cv2', batch_size=64, n_trials=5):
    assert test_type in ['cv2', 'pil']
    
    if test_type == 'cv2':
        test_dataset = CV2Dataset(files, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    else:
        test_dataset = PILDataset(files, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    def f():
        for batch in test_dataloader:
            pass
    
    test(f, n_trials=n_trials)


# ## Resize-ToTensor

# In[ ]:


cv2_transform = A.Compose([
    A.Resize(64, 64, interpolation=cv2.INTER_LINEAR),
    ToTensor()
])


# In[ ]:


dataloader_test(image_files_1000, cv2_transform, test_type='cv2', batch_size=64, n_trials=5)


# In[ ]:


pil_transform = transforms.Compose([
    transforms.Resize((64, 64), interpolation=2),
    transforms.ToTensor()
])


# In[ ]:


dataloader_test(image_files_1000, pil_transform, test_type='pil', batch_size=64, n_trials=5)


# ## Resize-CenterCrop-ToTensor

# In[ ]:


cv2_transform = A.Compose([
    A.SmallestMaxSize(64, interpolation=cv2.INTER_LINEAR),
    A.CenterCrop(64, 64),
    ToTensor()
])


# In[ ]:


dataloader_test(image_files_1000, cv2_transform, test_type='cv2', batch_size=64, n_trials=5)


# In[ ]:


pil_transform = transforms.Compose([
    transforms.Resize(64, interpolation=2),
    transforms.CenterCrop(64),
    transforms.ToTensor()
])


# In[ ]:


dataloader_test(image_files_1000, pil_transform, test_type='pil', batch_size=64, n_trials=5)


# ## Resize-HorizontalFlip-ToTensor

# In[ ]:


cv2_transform = A.Compose([
    A.Resize(64, 64, interpolation=cv2.INTER_LINEAR),
    A.HorizontalFlip(p=0.5),
    ToTensor()
])


# In[ ]:


dataloader_test(image_files_1000, cv2_transform, test_type='cv2', batch_size=64, n_trials=5)


# In[ ]:


pil_transform = transforms.Compose([
    transforms.Resize((64, 64), interpolation=2),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])


# In[ ]:


dataloader_test(image_files_1000, pil_transform, test_type='pil', batch_size=64, n_trials=5)


# ## Resize-RandomCrop-ToTensor

# In[ ]:


cv2_transform = A.Compose([
    A.Resize(96, 96, interpolation=cv2.INTER_LINEAR),
    A.RandomCrop(64, 64),
    ToTensor()
])


# In[ ]:


dataloader_test(image_files_1000, cv2_transform, test_type='cv2', batch_size=64, n_trials=5)


# In[ ]:


pil_transform = transforms.Compose([
    transforms.Resize((96, 96), interpolation=2),
    transforms.RandomCrop(64),
    transforms.ToTensor()
])


# In[ ]:


dataloader_test(image_files_1000, pil_transform, test_type='pil', batch_size=64, n_trials=5)


# In[ ]:




