#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import fastai
from fastai.vision import *
print(os.listdir("../input"))


# In[ ]:


trn = pd.read_csv('../input/train.csv')


# In[ ]:


trn.head()


# In[ ]:


def open_mask_rle(mask_rle:str, shape:Tuple[int, int])->ImageSegment:
    "Return `ImageSegment` object create from run-length encoded string in `mask_lre` with size in `shape`."
    x = FloatTensor(rle_decode(str(mask_rle), shape).astype(np.uint8))
    x = x.view(shape[1], shape[0], -1)
    return ImageSegment(x.permute(2,1,0))


# In[ ]:


path = Path('../input')


# In[ ]:


zzip = list(zip(trn['ImageId'].values, trn['Height'].values, trn['Width'].values))


# In[ ]:


msk = list(trn['EncodedPixels'].values)


# In[ ]:


m1 = open_mask_rle(msk[0], (zzip[0][1], zzip[0][2]))


# In[ ]:


img = open_image('../input/train/'+ zzip[0][0])
_,axs = plt.subplots(1,3, figsize=(16,8))
img.show(ax=axs[0], title='no mask')
img.show(ax=axs[1], y=m1, title='masked')
m1.show(ax=axs[2], title='mask only', alpha=1.)


# In[ ]:


df_train_value_counts = trn.ClassId.value_counts().reset_index()
df_train_value_counts.columns=['ClassId', 'Count']
df_sample_unique = trn.groupby('ClassId', group_keys=False).apply(lambda df: df.sample(1))
df_sample_unique_counts = df_sample_unique.merge(df_train_value_counts, on='ClassId')
df_sample_unique_counts.sort_values(by='Count', ascending=False, inplace=True)


# In[ ]:


class RleSegList(ImageList):
    def __init__(self, items, itemsB=None, **kwargs):
        super().__init__(items, **kwargs)
        self.itemsB = itemsB
        self.copy_new.append('itemsB')
    def get(self, i):
        img1 = super().get(i)
        fn = self.itemsB[random.randint(0, len(self.itemsB)-1)]
        return ImageTuple(img1, open_image(fn))
    @classmethod
    def from_df(cls, path, folderA, folderB, **kwargs):
        itemsB = ImageList.from_folder(path/folderB).items
        res = super().from_folder(path/folderA, itemsB=itemsB, **kwargs)
        res.path = path
        return res


# In[ ]:


zls = list(zip(trn['EncodedPixels'].values, trn['Height'].values, trn['Width'].values))


# In[ ]:


epls,hls,wls = list(trn['EncodedPixels']), list(trn['Height']), list(trn['Width'])


# In[ ]:


qid = list(set(trn['ImageId']))


# In[ ]:


ImageId = qid[6] #'00000663ed1ff0c4e0132b9b9ac53f6e.jpg'

img = plt.imread('../input/train/' + ImageId)
img_masks = trn.loc[trn['ImageId'] == ImageId, 'EncodedPixels'].tolist()
shape = (list(trn.loc[trn['ImageId'] == ImageId, 'Height'])[0], 
         list(trn.loc[trn['ImageId'] == ImageId, 'Width'])[0])


all_masks = np.zeros((shape))
for mask in img_masks:
    amasks = open_mask_rle(mask, (shape))
    all_masks += amasks.px.numpy().squeeze()

fig, axarr = plt.subplots(1, 3, figsize=(20, 45))
axarr[0].axis('off')
axarr[1].axis('off')
axarr[2].axis('off')
axarr[0].imshow(img)
axarr[1].imshow(all_masks)
axarr[2].imshow(img)
axarr[2].imshow(all_masks, alpha=0.4)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()


# In[ ]:


import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# In[ ]:


trfm = transforms.Compose([
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #normalize,
        ])


# In[ ]:


tfms = albumentations.Compose([
    transforms.ToTensor()
    ])
data_transforms_test = albumentations.Compose([
    albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    transforms.ToTensor()
    ])


# In[ ]:


tes = pd.read_csv('../input/sample_submission.csv')
tes.head()


# In[ ]:


class Imat(Dataset):
    def __init__(self, df, datafolder, datatype='train', 
                 transform = transforms.Compose([transforms.CenterCrop(32),transforms.ToTensor()])):
        self.datafolder = datafolder
        self.df = df
        self.datatype = datatype
        self.image_files_list = list(self.df['ImageId'])
        if self.datatype == 'train':
            self.mask_list = list(self.df['EncodedPixels'])
            self.h = list(self.df['Height'])
            self.w = list(self.df['Width'])
        self.transform = transform
        
        if self.datatype == 'train':
            self.labels = [np.float32(i) for i in list(self.df['ClassId'])]
        else:
            self.labels = [np.float32(0.0) for _ in range(len(self.image_files_list))]

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.datafolder, self.image_files_list[idx])
        img = open_image(img_name)
        if self.datatype == 'train':
            mask = open_mask_rle(self.mask_list[idx], (self.h[idx], self.w[idx]))
            image = img.px
            label = self.labels[idx]
            return image, mask.px, label
        else:
            img_name = os.path.join(self.datafolder, self.image_files_list[idx])
            img = open_image(img_name)
            image = img.px
            label = [np.float32(0.0) for _ in range(len(self.image_files_list))]
            label = label[idx]
            return image, label


# In[ ]:


vlx = trn.sample(frac=0.2).index


# In[ ]:


tr = trn.loc[~trn.index.isin(vlx)].reset_index(drop=True)


# In[ ]:


vl = trn.loc[trn.index.isin(vlx)].reset_index(drop=True)


# In[ ]:





# In[ ]:


trds = Imat(tr, datafolder='../input/train', datatype='train', transform=tfms)


# In[ ]:


vrds = Imat(vl, datafolder='../input/train', datatype='train', transform=tfms)


# In[ ]:


teds = Imat(tes, datafolder='../input/test', datatype='test', transform=tfms)


# In[ ]:


next(iter(trds))[1].shape


# In[ ]:


data = DataBunch.create(trds, vrds, teds, bs=8, num_workers=0)


# In[ ]:


next(iter(data.train_dl))


# In[ ]:





# In[ ]:


#https://github.com/lmoroney/dlaicourse/blob/master/Exercises/Exercise%205%20-%20Real%20World%20Scenarios/Exercise%205%20-%20Answer.ipynb
#https://cmdlinetips.com/2018/12/how-to-loop-through-pandas-rows-or-how-to-iterate-over-pandas-rows/
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 12
ncols = 4

pic_index = 0 # Index for iterating over images
# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

ctr=0
for index, row in df_sample_unique_counts.head(n=24).iterrows():
     #print(index, row)
          # Set up subplot; subplot indices start at 1
    img_path = row.ImageId
    classId = row.ClassId
    rle = row.EncodedPixels
    h = row.Height
    w = row.Width
    mask = open_mask_rle(rle, (h, w))
    mk = np.array(mask.px).reshape(mask.shape[1], mask.shape[2], -1).astype(np.uint8)
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr)
    sp.axis('Off') # Don't show axes (or gridlines)
    img_path = '../input/train/' + img_path
    img = mpimg.imread(img_path)
    dimg = np.dstack((img, mk))
    plt.imshow(img)
    plt.imshow(dimg[..., -1])
    plt.title(classId)
    
for index, row in df_sample_unique_counts.head(n=24).iterrows():
     #print(index, row)
          # Set up subplot; subplot indices start at 1
    img_path = row.ImageId
    classId = row.ClassId
    rle = row.EncodedPixels
    h = row.Height
    w = row.Width
    mask = open_mask_rle(rle, (h, w))
    mk = np.array(mask.px).reshape(mask.shape[1], mask.shape[2], -1).astype(np.uint8)
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr)
    sp.axis('Off') # Don't show axes (or gridlines)
    img_path = '../input/train/' + img_path
    img = mpimg.imread(img_path)
    dimg = np.dstack((img, mk))
    plt.imshow(img)
    plt.title(classId)  


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




