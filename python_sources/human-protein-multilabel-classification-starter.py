#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder


# In[ ]:


from skmultilearn.model_selection import IterativeStratification


# In[ ]:


project_name = 'protein_classifier'


# # Exploring Data

# In[ ]:


DATA_DIR = '../input/jovian-pytorch-z2g/Human protein atlas'
TRAIN_DIR = DATA_DIR + '/train'
TEST_DIR = DATA_DIR + '/test'

TRAIN_CSV = DATA_DIR + '/train.csv'
TEST_CSV = '../input/jovian-pytorch-z2g/submission.csv'


# In[ ]:


get_ipython().system('head "{TRAIN_CSV}"')


# This is a multi-label classification. Some images can have one or a list of labels.
# train.csv contains image id and its label(s) of the train dataset. Similarly, the submission.csv contains the image id and default label for the images in test dataset.

# In[ ]:


get_ipython().system("head '{TEST_CSV}'")


# In[ ]:


train_df = pd.read_csv(TRAIN_CSV)
train_df.head()


# The textual labels are consolidated in a dictionary.

# In[ ]:


labels = {
    0: 'Mitochondria',
    1: 'Nuclear bodies',
    2: 'Nucleoli',
    3: 'Golgi apparatus',
    4: 'Nucleoplasm',
    5: 'Nucleoli fibrillar center',
    6: 'Cytosol',
    7: 'Plasma membrane',
    8: 'Centrosome',
    9: 'Nuclear speckles'
}


# In[ ]:


def encode_label(label):
    """Encodes the multi labels into a vector(tensor)."""
    target = torch.zeros(10)
    for l in str(label).split(' '):
        target[int(l)] = 1.
    return target

def decode_target(target, text_labels=False, threshold=0.5):
    """Decodes a tensor into a sequence of labels."""
    result = []
    for i, x in enumerate(target):
        if (x >= threshold):
            if text_labels:
                result.append(labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return ' '.join(result)


# Lets test the encode and decode methods.

# In[ ]:


encode_label('2 4 5')


# In[ ]:


decode_target(torch.tensor([0., 0., 1., 0., 1., 1., 0., 0., 0., 0.]))


# In[ ]:


decode_target(torch.tensor([0, 0, 1, 0, 1, 1, 0, 0, 0, 0.]), text_labels=True)


# # Creating Datasets & Data Loaders

# In[ ]:


class HumanProteinDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.df)    
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_id, img_label = row['Image'], row['Label']
        img_fname = self.root_dir + "/" + str(img_id) + ".png"
        img = Image.open(img_fname)
        if self.transform:
            img = self.transform(img)
        return img, encode_label(img_label)


# In[ ]:


transform = transforms.Compose([transforms.ToTensor()])
dataset = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, transform=transform)


# In[ ]:


len(dataset)


# In[ ]:


def show_sample(img, target, invert=True):
    if invert:
        plt.imshow(1 - img.permute((1, 2, 0)))
    else:
        plt.imshow(img.permute(1, 2, 0))
    print('Labels:', decode_target(target, text_labels=True))


# In[ ]:


show_sample(*dataset[0], invert=False)


# In[ ]:


show_sample(*dataset[0], invert=True)


# ## Training & Validation sets

# In[ ]:


torch.manual_seed(10)


# In[ ]:


val_pct = 0.1
val_size = int(val_pct * len(dataset))
train_size = len(dataset) - val_size


# In[ ]:


train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)


# ## DataLoader

# In[ ]:


batch_size = 64


# In[ ]:


train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=2, pin_memory=True)


# In[ ]:


def show_batch(dl, invert=True):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xticks([]); ax.set_yticks([])
        data = 1-images if invert else images
        ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))
        break


# In[ ]:


show_batch(train_dl, invert=False)


# # Model

# In[ ]:


train_df.head()


# # Distribution of combinations

# In[ ]:


train_df['Label'].value_counts().head()


# In[ ]:


train_df['Label'].value_counts().tail()


# In[ ]:


df = train_df.copy()


# In[ ]:


df = df.set_index('Image').sort_index()


# In[ ]:


df['Label'] = df['Label'].apply(lambda x: x.split(' '))


# In[ ]:


df.head()


# In[ ]:


df = df.explode('Label')
df.head()


# ## Distribution of Individual Classes

# In[ ]:


df['Label'].value_counts()


# In[ ]:


df = pd.get_dummies(df);df


# In[ ]:


df = df.groupby(df.index).sum(); df.head()


# In[ ]:


df.columns = labels.keys() ; df.head()


# In[ ]:


X, y = df.index.values, df.values


# In[ ]:


k_fold = IterativeStratification(n_splits = 5, order=2)

splits = list(k_fold.split(X, y))


# In[ ]:


splits[0][0].shape , splits[0][1].shape


# In[ ]:


df.tail(), len(df)


# In[ ]:


splits[0][0], splits[0][1]


# In[ ]:


fold_splits = np.zeros(df.shape[0]).astype(int)

for i in range(5):
    fold_splits[splits[i][1]] = i # Note the validation fold set#

df['Split'] = fold_splits


# In[ ]:


df.tail(10)


# In[ ]:


train_df = df[df['Split'] != 0]
valid_df = df[df['Split'] == 0]


# In[ ]:


train_df.head()


# In[ ]:


valid_df.head()


# In[ ]:


from pathlib import Path
from tqdm.notebook import tqdm
import cv2

train_set = set(Path(TRAIN_DIR).iterdir())
test_set = set(Path(TEST_DIR).iterdir())
whole_set = train_set.union(test_set)

x_tot, x2_tot = [], []
for file in tqdm(whole_set):
   img = cv2.imread(str(file), cv2.COLOR_RGB2BGR)
   img = img/255.0
   x_tot.append(img.reshape(-1, 3).mean(0))
   x2_tot.append((img**2).reshape(-1, 3).mean(0))


# In[ ]:


#image stats
img_avr =  np.array(x_tot).mean(0)
img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
print('mean:',img_avr, ', std:', np.sqrt(img_std))
mean = torch.as_tensor(x_tot)
std =torch.as_tensor(x2_tot)


# In[ ]:




