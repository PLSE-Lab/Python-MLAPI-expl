#!/usr/bin/env python
# coding: utf-8

#  # Testing and Creating a DataLoader
# With new GPU limits, every single second saved is important. Dataloading take a lot of time per epoch so here in this notebook I have decided to test them out.
# In this notebook I will try various formats of dataloaders and try to find out which is the fastest. In particular, I will compare the time taken to run a certain number of epcohs for the following four kinds of dataloader:
# 1. Load Original Images and Resize + Create masks on spot
# 2. Load Resized Images + Create masks on spot
# 3. Load Resized Images in numpy format + Load masks in numpy format
# 4. Load both images and maps from RAM (unfortunately, with the resources I have and the resources kaggle kernels provide, it's not possible to load both images and masks in RAM unless you reduce the size a lot)
# 
# There are four ways to load an image:
# 1. Load large image then resize
# 2. Load resized image
# 3. Load resized image saved as numpy array
# 4. Load resized image from RAM
# 
# There are three ways to load the mask:
# 1. Create mask on spot them resize
# 2. Load mask from numpy array and don't resize
# 3. Load mask from the RAM
# 
# By various combinations of the above, twelve dataloaders are possible. But, I am just testing four dataloaders which I believe are the most important.

# # Prepare Data
# First, we need to resize and save images as both .jpg and .npy array. We also create and save the mask on disk as .npy array. 
# We have limited disk space and RAM (for RAM loader) so let's just work with 1k images. 

# In[ ]:


import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as albu

def get_img(image_path):
    """Load image from disk"""
    img = cv2.imread(image_path)
    return img

def rle_decode(mask_rle: str = "", shape: tuple = (1400, 2100)):
    """Source: https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools"""
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")

def make_mask(df: pd.DataFrame, image_name: str='img.jpg', shape: tuple = (1400, 2100)):
    """Source: https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools"""
    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask         
    return masks


#Load train.csv to make mask
train = pd.read_csv(f"../input/understanding_cloud_organization/train.csv")
train["label"] = train["Image_Label"].apply(lambda x: x.split("_")[1])
train["im_id"] = train["Image_Label"].apply(lambda x: x.split("_")[0])


N = 1000 #number of images
dir_ip = '../input/understanding_cloud_organization/train_images'
dir_op = 'img_resized'
dir_op_mask = 'mask'
dir_op_npy = 'img_resized_npy'

for d in [dir_op, dir_op_mask, dir_op_npy]:
    if not os.path.exists(d):
        os.makedirs(d)

tfms = albu.Compose([albu.Resize(320, 640)]) #To resize
bar = tqdm(os.listdir(dir_ip)[:N], postfix={"file":"none"})

for file in bar:
    bar.set_postfix(ordered_dict={"file":file})    
    path = os.path.join(dir_ip, file)
    img = get_img(path)    
    mask = make_mask(train, file) 
    tfmed = tfms(image=img, mask=mask)
    img = tfmed['image']
    mask = tfmed['mask']
    cv2.imwrite(os.path.join(dir_op, file), img)
    np.save(os.path.join(dir_op_mask, file), mask)
    np.save(os.path.join(dir_op_npy, file), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))     


# # Make DataLoaders

# In[ ]:


# Some utility functions

import matplotlib.pyplot as plt

def visualize(image, mask, original_image=None, original_mask=None, gray=True):
    """Source: https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools"""
    fontsize = 14
    class_dict = {0: "Fish", 1: "Flower", 2: "Gravel", 3: "Sugar"}    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(1, 5, figsize=(24, 24))

        if gray:
            ax[0].imshow(image, cmap='gray')
        else:    
            ax[0].imshow(image)
        for i in range(4):
            ax[i + 1].imshow(mask[:, :, i])
            ax[i + 1].set_title(f"Mask {class_dict[i]}", fontsize=fontsize)
    else:
        f, ax = plt.subplots(2, 5, figsize=(24, 12))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title("Original image", fontsize=fontsize)

        for i in range(4):
            ax[0, i + 1].imshow(original_mask[:, :, i])
            ax[0, i + 1].set_title(f"Original mask {class_dict[i]}", fontsize=fontsize)

        ax[1, 0].imshow(image)
        ax[1, 0].set_title("Transformed image", fontsize=fontsize)

        for i in range(4):
            ax[1, i + 1].imshow(mask[:, :, i])
            ax[1, i + 1].set_title(
                f"Transformed mask {class_dict[i]}", fontsize=fontsize
            )

            
def get_img(name, image_dir='dir_op_npy', npy=False):
    if npy:
        return np.load(os.path.join(image_dir, name+'.npy'))
    img = cv2.imread(os.path.join(image_dir, name))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        

def make_mask(image_name, mask_dir=False, df=False, shape: tuple = (1400, 2100)):
    if mask_dir: #Load numpy mask
        return np.load(os.path.join(mask_dir, image_name+'.npy'))  
    #Make mask
    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask         
    return masks


# In[ ]:


from torch.utils.data import TensorDataset, DataLoader,Dataset

class CloudDataset(Dataset):
    def __init__(
        self,
        img_ids = None,
        transforms = None,
        df = None,
        mask_dir = None,
        image_dir = None,
        npy = False, #Images in numpy format or not
        ram = False
    ):
        self.img_ids = img_ids
        self.transforms = transforms
        self.df = df
        self.mask_dir = mask_dir
        self.image_dir = image_dir
        self.npy = npy
        self.ram = ram
        if self.ram:
            self.samples = self.load_samples()
            
    def load_samples(self):
        samples = []
        print("Loading images...")
        for image_name in tqdm(self.img_ids):
            mask = make_mask(image_name, self.mask_dir, self.df)
            img = get_img(image_name, self.image_dir, self.npy)
            augmented = self.transforms(image=img, mask=mask)
            samples.append(augmented)
        return samples    
            
    
    def __getitem__(self, idx):
        if self.ram:
            sample = self.samples[idx]
            return sample["image"], sample["mask"]
        
        image_name = self.img_ids[idx]
        
        mask = make_mask(image_name, self.mask_dir, self.df)
        img = get_img(image_name, self.image_dir, self.npy)
        
        augmented = self.transforms(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]
        return img, mask

    def __len__(self):
        return len(self.img_ids)


# In[ ]:


imgs_ids = os.listdir(dir_ip)[:N]

tfms1 = albu.Compose([albu.Resize(320, 640),
                      albu.HorizontalFlip(p=0.5),
                    ])
tfms2 = albu.Compose([albu.HorizontalFlip(p=0.5)])

num_workers = 0
bs = 16
df = train

#Loader1: Load large images then resize + Create masks
dataset1 = CloudDataset(imgs_ids, tfms1, df, 
                             False, dir_ip, npy=False)
loader1 = DataLoader(dataset1, batch_size=bs, shuffle=False, num_workers=num_workers)

# Loader2: Load resized images + Create masks
dataset2 = CloudDataset(imgs_ids, tfms2, df, 
                             False, dir_op, npy=False)
loader2 = DataLoader(dataset2, batch_size=bs, shuffle=False, num_workers=num_workers)

# Loader3: Load resized images + Load masks
dataset3 = CloudDataset(imgs_ids, tfms2, False, 
                             dir_op_mask, dir_op, npy=False)
loader3 = DataLoader(dataset3, batch_size=bs, shuffle=False, num_workers=num_workers)

# Loader4: Load resized images in numpy + Load masks
dataset4 = CloudDataset(imgs_ids, tfms2, False, 
                             dir_op_mask, dir_op_npy, npy=True)
loader4 = DataLoader(dataset4, batch_size=bs, shuffle=False, num_workers=num_workers)

# Loader5: Ram loader
dataset5 = CloudDataset(imgs_ids, tfms2, False, 
                             dir_op_mask, dir_op_npy, npy=True, ram=True)
loader5 = DataLoader(dataset5, batch_size=bs, shuffle=False, num_workers=num_workers)


# # Visualize loaders

# In[ ]:


loaders = [loader1, loader2, loader3, loader4, loader5]

for i, loader in enumerate(loaders):
    print(f"Testing loader{i+1}")
    for batch in loader:
        images, masks = batch
        visualize(images[0], masks[0])
        break        


# # Test runtimes

# In[ ]:


import time

def test_bench(loader, epochs=5):
    runtimes = []
    for epoch in tqdm(range(epochs)):
        start = time.time()
        for batch in loader:
            images, masks = batch
        end = time.time()
        runtimes.append(end - start)
    return runtimes

runtimes = {}
for i, loader in enumerate(loaders):
    print(f"Testing loader{i+1}...")
    runtimes[f'loader{i+1}'] = test_bench(loader)


# In[ ]:


runtimes = pd.DataFrame(runtimes)
about = {
    'loader1': 'Load image and resize | Create mask',
    'loader2': 'Load resized image | Create mask',
    'loader3': 'Load resized image.npy | Create mask',
    'loader4': 'Load resized image.npy | Load mask.npy',
    'loader5': 'Image RAMLoader | Mask RAMLoader',
}
runtimes = runtimes.rename(columns=about)


# In[ ]:


plt.figure(figsize=(15,5))
runtimes.max().plot(kind='barh', title='Max-time (s)')
plt.show()
plt.figure(figsize=(15,5))
runtimes.min().plot(kind='barh', title='Min-time (s)')
plt.show()
plt.figure(figsize=(15,5))
runtimes.mean().plot(kind='barh', title='Average-time (s)')
plt.show()


# In[ ]:


runtimes.head(5)


# # Conclusion
# Clearly, as expected, RAM loader is the fastest. However, in this competition we can not load all the images and masks in RAM unless we reduce the size a lot. Loading one of them in RAM might work and increase the data loading speed though. 

# In[ ]:


get_ipython().system('rm {dir_op} {dir_op_mask} {dir_op_npy} -r')


# Thanks for reading!
