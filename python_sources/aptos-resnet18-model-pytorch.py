#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print(len(filenames))

# Any results you write to the current directory are saved as output.


# **Import Packages and Libraries**

# In[ ]:


import os
import pandas as pd
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


# **CSV Files Pasing and Visualisation-**

# In[ ]:


file_path1 ='../input/test.csv'
file_path2 ='../input/train.csv'
file_path3 ='../input/sample_submission.csv'


# In[ ]:


test = pd.read_csv(file_path1)
train = pd.read_csv(file_path2)
sample_submission=pd.read_csv(file_path3)


# In[ ]:


print(train.shape[0], test.shape[0]) 


# In[ ]:


test.head()


# In[ ]:


train.head()


# In[ ]:


sample_submission.head()


# **Data Visualisation**

# In[ ]:


from PIL import Image
import matplotlib.pyplot as plt
with open('/kaggle/input/test_images/b16787f65d49.png', 'rb') as file:
    img=Image.open(file)
    plt.axis('off')
    plt.imshow(img)
    #print(img.size)

#print(img.format)


# **Dataset Class**

# In[ ]:


class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform = None, train = True):
        self.label_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
    def __len__(self):
        return len(self.label_frame)
    
    def __getitem__(self, indx):
        img_name = os.path.join(self.root_dir, self.label_frame.iloc[indx, 0] + '.png')
        img = Image.open(img_name)
        if self.transform:
            img = self.transform(img)
            
        if self.train == True:
            label = self.label_frame.iloc[indx, 1]
            label = np.array([label])
            return img, label
        else:
            return img, img_name           
            


# **Data Preprocessing and Loading**

# In[ ]:


transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.Grayscale(3),
                                transforms.ToTensor(), 
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])                                               


# In[ ]:


train_data = ImageDataset("../input/train.csv", "../input/train_images", transform = transform, train = True)
test_data = ImageDataset("../input/test.csv", "../input/test_images", transform = transform, train = False)


# In[ ]:


train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)


# In[ ]:


len(train_loader), len(test_loader)


# In[ ]:


for data in train_loader:
    img, lab = data
    print(lab[0].shape)
    print(img[0].shape)
    break


# In[ ]:


import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm_notebook


# In[ ]:


model = models.resnet18(pretrained=True)


# In[ ]:


for param in model.parameters():
    param.requires_grad = False


# In[ ]:


model.fc = nn.Linear(512, 5)


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available else 'cpu')


# In[ ]:


model.to(device)
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters())


# In[ ]:


print(lab[0].shape)
print(lab[9,:])


# In[ ]:


for i in tqdm_notebook(range(5)):
    for data in tqdm_notebook(train_loader):
        image, label = data
        label = label.squeeze(1)
        image, label = image.to(device), label.to(device)
        
        opt.zero_grad()
        out = model(image)
        loss = loss_fn(out, label)
        loss.backward()
        opt.step()
        torch.save(model.state_dict(), "best_model.pth")
        del image, label, out
        torch.cuda.empty_cache()
        
        


# In[ ]:


model.load_state_dict(torch.load("best_model.pth"))
model.to(device)
model.eval()


# In[ ]:


outputs = []
for test_img, test_filename in tqdm_notebook(test_loader):
        test_img = test_img.to(device)
        output = model(test_img)
        num, ind = torch.max(output, 1)
        output =  ind.squeeze().cpu().numpy()
        outputs.extend(output)


# In[ ]:


submission = test
submission['diagnosis'] = outputs
submission.head()


# In[ ]:


submission.to_csv( 'submission.csv')

