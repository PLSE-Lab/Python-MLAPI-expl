#!/usr/bin/env python
# coding: utf-8

# ## NN detection with Pytorch

# In this kernel I'll try to assemble NN architecture for detection task. Firstly, import necessary libraries:

# In[ ]:


import os
import io
import math
import gc
import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


# ### Import data and some preprocessing:

# In[ ]:


train = pd.read_csv('/kaggle/input/understanding_cloud_organization/train.csv')


# In[ ]:


train_new = pd.DataFrame()
train_new['img'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
train_new['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
train_new['EncodedPixels'] = train['EncodedPixels']


# In[ ]:


train_new.head()


# Histogram with existing labels distibution:

# In[ ]:


pd.value_counts(train_new[['label','EncodedPixels']].dropna()['label']).plot('barh');


# Distribution, how many figures we can find on one photo:

# In[ ]:


train_new.dropna()[['img','label']].groupby(['img']).count().reset_index()['label'].hist();


# ### Visualize detection area and photos:

# In[ ]:


def detect(train_new, img):
    image = plt.imread("../input/understanding_cloud_organization/train_images/" + img)
    rle_string = train_new[(train_new['img']==img)]['EncodedPixels'].iloc[0]
    rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
    rle_pairs = np.array(rle_numbers).reshape(-1,2)
    img = np.zeros(1400*2100, dtype=np.uint8)
    for index, length in rle_pairs:
        index -= 1
        img[index:index+length] = 100
    img = img.reshape(2100,1400)
    np_mask = img.T
    np_mask = np.clip(np_mask, 0, 1)
    return np_mask


# In[ ]:


fig = plt.figure(figsize=(20,25))
data_vis = train_new[train_new['label']=='Fish'].dropna()
for i in range(1,13):
    fig.add_subplot(4,3,i)
    mask = detect(data_vis , data_vis.iloc[i]['img'])
    image = plt.imread("../input/understanding_cloud_organization/train_images/" + data_vis.iloc[i]['img'])
    plt.imshow(image);
    plt.imshow(mask, alpha=0.4);


# ### I want to make Neural Network with 4 outputs:
# 1. Probability of an object appearing in the picture <br>
# Subsequent paragraphs if first = 1:
# 2. X coordinate of detection mask center 
# 3. Y coordinate of detection mask center
# 4. Height of detection mask
# 5. Width of detection mask

# Create function that return height, width, x_center and y_center

# In[ ]:


def center_grad(label, np_mask):
    """This function return h, w, x_c, y_c of our mask"""
    height = np.where(np_mask[:,:]==1)[0][-1]-np.where(np_mask[:,:]==1)[0][0]
    width = np.where(np_mask[:,:]==1)[1][-1]-np.where(np_mask[:,:]==1)[1][0]
    x_cen, y_cen = np.where(np_mask[:,:]==1)[0][0] + height//2, np.where(np_mask[:,:]==1)[1][0] + width//2
    return label, x_cen, y_cen, height, width


# Create special 'Fish' dataset for our network

# In[ ]:


fish_data = train_new[train_new['label']=='Fish']
fish_data.set_index(np.arange(fish_data.shape[0]), inplace=True)
fish_data['Label'] = fish_data['EncodedPixels'].apply(lambda x: 0 if pd.isnull(x) else 1)


# In[ ]:


fish_data.head()


# In[ ]:


height, width = 1400, 2100
def masks(train_new, name_image):
    rle_string = train_new[train_new['img']==name_image]['EncodedPixels'].values[0]
    if pd.isnull(rle_string):
        return pd.DataFrame([])
    else:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1,2)
        img = np.zeros(height*width, dtype=np.uint8)
        for index, length in rle_pairs:
            index -= 1
            img[index:index+length] = 100
        img = img.reshape(height,width)
        img = img.T

        np_mask = img
        np_mask = np.clip(np_mask, 0, 1)
        return np_mask


# Create DataLoader for this problem:

# In[ ]:


class CloudDataset(Dataset):
    def __init__(self, df: pd.DataFrame = train_new, datatype: str = 'train', img_ids: np.array = None,
                 transforms = transforms.ToTensor(),
#                 transforms = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()]),
                preprocessing=None):
        self.df = df
        if datatype != 'test':
            self.data_folder = f"../input/understanding_cloud_organization/train_images"
        else:
            self.data_folder = f"{path}/test_images"
        self.transforms = transforms

    def __getitem__(self, idx):
        image_name = self.df['img'][idx]
        mask = masks(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        
        image = Image.open(image_path)
        image = self.transforms(image)
        
        if mask.shape != (0,0):
            label = center_grad(self.df.iloc[idx]['Label'],mask)
            if label[0] == 1:
                label = (label[0], label[1]/height, label[2]/width,
                            math.log(abs(label[3]+0.0001)), math.log(abs(label[4]+0.0001)) )
        else: 
            label = (0,0,0,0,0)
        return image, label
    
    def __len__(self):
        return self.df.shape[0]


# Classes distribution in "Fish dataset"

# In[ ]:


fish_data[:100]['Label'].hist();


# In[ ]:


train_dataset = CloudDataset(df=fish_data[:2000], datatype='train')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)


# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
#         self.fc1 = nn.Linear(1*5*350*525, 5)
        self.fc1 = nn.Linear(543402,5)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=3, kernel_size=5)
        
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1)
        x = self.fc1(x)
        x = [F.sigmoid(x[0]),F.sigmoid(x[1]),F.sigmoid(x[2]),x[3],x[4]]
        return x


# Create the main function for learning with pytorch. I'll learn only 10 epochs to save time.

# In[ ]:


losses = []

# define model
model = Net()
model = model.cuda()
crit_mse = nn.MSELoss()
crit_bce = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-5)

for epoch in range(1, 11):
    print('epoch = ', epoch)
    for batch_idx, (data, label) in enumerate(train_loader):
            # get output
            data = data.cuda()
            for i in range(len(label)):
                label[i] = label[i].cuda()
            out = model(data)
            
            # transform output to our system
            output = out
            
            # define complex LOSS function
            if label[0].item() == 1:
                loss = crit_bce(output[0],torch.Tensor([label[0].item()]).cuda() ) +                     1*(crit_bce(output[1],torch.Tensor([label[1].item()]).cuda() ) +  
                       crit_bce(output[2],torch.Tensor([label[2].item()]).cuda() ) + \
                       crit_mse(output[3],torch.Tensor([label[3].item()]).cuda() ) + \
                       crit_mse(output[4],torch.Tensor([label[4].item()]).cuda() ) )
            else:
                loss = crit_bce(output[0],torch.Tensor([label[0].item()]).cuda() )
                
            if batch_idx % 500 == 0:
                print('Loss :{:.4f} Epoch - {}/{}'.format(loss.item(), epoch, 10))
            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()
    del data
    del label


# Loss plot in train dataset:

# In[ ]:


plt.plot(np.arange(len(losses)), losses);


# ### So, compare model prediction with target mask:

# In[ ]:


mask = detect(data_vis , data_vis.iloc[0]['img'])
image = plt.imread("../input/understanding_cloud_organization/train_images/" + data_vis.iloc[0]['img'])
plt.imshow(image);
plt.imshow(mask, alpha=0.4);


# In[ ]:


ss = center_grad(1, mask)
print(ss[0], ss[1]/height, ss[2]/width, math.log(ss[3]), math.log(ss[4]))


# In[ ]:


img = torch.Tensor(image.reshape(1,3,1400,2100)).cuda()
model(img)


# With same approach we can detect another categories. <br>
# To be continued...

# In[ ]:




