#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *

import torchvision.datasets as dsets
from torchvision import transforms


# In[ ]:


bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart


# In[ ]:


import os
os.listdir("../input/fashionmnist/")


# In[ ]:


labels =['Tshirt/top',"Trouser","Pullover","Dress","Coat","Sandal","Shirt",'Sneaker',"Bag","Ankle boot"]


# In[ ]:


tfms = get_transforms(do_flip = False)


# In[ ]:


train_data = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
test_data = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


train_data['label']


# In[ ]:


train_images = train_data.drop('label', axis = 1)


# In[ ]:


img = train_images.iloc[2,:].values.reshape(28, 28)
plt.imshow(img, cmap = 'gray')


# In[ ]:


class FashionMNIST(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        fashion_df = pd.read_csv(path)
        self.labels = fashion_df.label.values
        self.images = fashion_df.iloc[:, 1:].values.astype('uint8').reshape(-1, 28, 28)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = Image.fromarray(self.images[idx])
        
        if self.transform:
            img = self.transform(img)

        return img, label


# In[ ]:





# In[ ]:


train_ds = FashionMNIST('../input/fashionmnist/fashion-mnist_train.csv')
test_ds = FashionMNIST('../input/fashionmnist/fashion-mnist_test.csv')


# In[ ]:


from PIL import Image

test_ds[0][0]


# In[ ]:


train_data_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)
#data = DataBunch.create(train_ds = train_ds, valid_ds = test_ds)
data = DataBunch.create(train_data_loader, test_data_loader)


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=[accuracy])

