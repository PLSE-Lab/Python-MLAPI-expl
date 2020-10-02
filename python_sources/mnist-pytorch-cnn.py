#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
GPU_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import time
import math
from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn,optim
import torch.nn.functional as F

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


USE_GPU = torch.cuda.is_available()
if USE_GPU:
    print('Use GPU')
else:
    print('Use CPU')


# ## Function and class definitions

# In[ ]:


def show_mnist_batch(sample_batched):
    """Show image for a batch of samples."""
    images_batch, labels_batch =             sample_batched['image'], sample_batched['label']

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


# In[ ]:


def cross_entropy(y,yp):
    # y is the ground truch
    # yp is the prediction
    yp[yp>0.99999] = 0.99999
    yp[yp<1e-5] = 1e-5
    return np.mean(-np.log(yp[range(yp.shape[0]),y.astype(int)]))

def accuracy(y,yp):
    return (y==np.argmax(yp,axis=1)).mean()

def softmax(score):
    score = np.asarray(score, dtype=float)
    score = np.exp(score-np.max(score))
    score = score/(np.sum(score, axis=1).reshape([score.shape[0],1]))#[:,np.newaxis]
    return score


# In[ ]:


class MnistDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform
        if 'label' in df.columns:
            self.labels = df['label'].values
            self.images = df.drop('label',axis=1).values
        else:
            self.labels = np.zeros(df.shape[0])
            self.images = df.values
        self.images = (self.images/255.0).astype(np.float32).reshape(df.shape[0],28,28)
        
    
    def head(self):
        return self.df.head()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        label = np.array(self.labels[idx])
        image = self.images[idx]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


# In[ ]:


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # torch image: [C, H, W]
        return {'image': torch.from_numpy(image).unsqueeze(0),
                'label': torch.from_numpy(label)}


# ### A model is a subclass of nn.Module which defines a computing graph

# In[ ]:


class Logistic_Model(nn.Module):
    def __init__(self,num_fea,num_class):
        super().__init__()
        #nn.Linear(input_dim, output_dim)
        self.lin = nn.Linear(num_fea,num_class)

    def forward(self, xb):
        B = xb.size()[0]
        if len(xb.size())>2:
            xb = xb.view(B,-1) # 4D tensor of B,C,H,W -> 2D tensor B,CxHxW
        return self.lin(xb)


# In[ ]:


class SimpleCNN(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self,h,w,c,num_class):
        super(SimpleCNN, self).__init__()
        
        #Input channels = 3, output channels = 18
        self.h = h
        self.w = w
        self.c = c
        self.num_class = num_class
        
        self.conv1 = torch.nn.Conv2d(c, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(18 * h//2 * w//2, 64)
        
        #64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(64, 10)
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))
        
        #Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 18 * self.w//2 * self.h//2)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return(x)


# ### A learner has functions fit() and predict(), like the sklearn model

# In[ ]:


class Learner(object):
    
    def __init__(self,model,**params): 
        self.model = model
        if USE_GPU:
            self.model.cuda()
        self.params = params
        
    def predict(self,test_dl):
        yps = []
        for batch in tqdm(test_dl):
            xb, yb = batch['image'],batch['label']
            if USE_GPU:
                xb, yb = xb.cuda(),yb.cuda()
            pred = self.model(xb)
            if USE_GPU:
                yps.append(pred.cpu().detach().numpy())
            else:
                yps.append(pred.detach().numpy())
        yps = np.vstack(yps)
        yps = softmax(yps)
        return yps
        
    def fit(self,train_dl,valid_dl=None,
            epochs=10,lr=0.001,wd=0.1):
        opt_type = self.params.get('opt','SGD')
        if opt_type == 'SGD':
            opt = optim.SGD(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            train_loss = 0
            for batch in tqdm(train_dl):
                xb, yb = batch['image'],batch['label']
                if USE_GPU:
                    xb, yb = xb.cuda(),yb.cuda()
                pred = self.model(xb)
                loss = F.cross_entropy(pred, yb)
                if USE_GPU:
                    train_loss += loss.cpu().detach().numpy()
                else:
                    train_loss += loss.detach().numpy()
                loss.backward()
                opt.step()
                opt.zero_grad()
            if valid_dl is None:
                print('Epoch %d Training Loss:%.4f'%(epoch,
                            train_loss/len(train_dl)))
                continue
            yps = []
            yrs = []
            for batch in tqdm(valid_dl):
                xb, yb = batch['image'],batch['label']
                if USE_GPU:
                    xb, yb = xb.cuda(),yb.cuda()
                pred = self.model(xb)
                if USE_GPU:
                    yps.append(pred.cpu().detach().numpy())
                    yrs.append(yb.cpu().detach().numpy())
                else:
                    yps.append(pred.detach().numpy())
                    yrs.append(yb.detach().numpy())
            yps = np.vstack(yps)
            yps = softmax(yps)
            yrs = np.concatenate(yrs)
            ce = cross_entropy(yrs,yps)
            acc = accuracy(yrs,yps)
            print('Epoch %d Training Loss:%.4f Valid ACC: %.4f Cross Entropy:%4f'%(epoch,
                            train_loss/len(train_dl),acc,ce))
            
        self.opt = opt


# ## Inspect datasets

# ### Read csv

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_df = pd.read_csv('../input/train.csv')\ntest_df = pd.read_csv('../input/test.csv')\nprint(train_df.shape, test_df.shape)")


# In[ ]:


train_df.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nval_pct = 0.2 # use 20% train data as local validation\nis_valid = np.random.rand(train_df.shape[0])<val_pct\ntrain_df, valid_df = train_df.loc[~is_valid], train_df.loc[is_valid]\nprint(train_df.shape, valid_df.shape)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_dataset = MnistDataset(df=train_df,\n                            transform=transforms.Compose([\n                                               ToTensor()\n                                           ]))\nvalid_dataset = MnistDataset(df=valid_df,\n                            transform=transforms.Compose([\n                                               ToTensor()\n                                           ]))\ntest_dataset = MnistDataset(df=test_df,\n                            transform=transforms.Compose([\n                                               ToTensor()\n                                           ]))')


# In[ ]:


fig = plt.figure(figsize=(20,8))

for i in range(len(train_dataset)):
    sample = train_dataset[i]

    print(i, sample['image'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{} Label {}'.format(i,sample['label']), fontsize=30)
    ax.axis('off')
    plt.imshow(sample['image'].numpy()[0],cmap='gray')

    if i == 3:
        plt.show()
        break


# ### Data loader generates batch of samples with multi-thread functions.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nbatch_size = 128\ncpu_workers = 8\n\ntrain_dataloader = DataLoader(train_dataset, batch_size=batch_size,\n                        shuffle=True, num_workers=cpu_workers,\n                        drop_last=True)\n\nvalid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,\n                        shuffle=False, num_workers=cpu_workers,\n                        drop_last=False)\n\ntest_dataloader = DataLoader(test_dataset, batch_size=batch_size,\n                        shuffle=False, num_workers=cpu_workers,\n                        drop_last=False)')


# ### Illustrate the first batch

# In[ ]:


for i_batch, sample_batched in enumerate(train_dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['label'].size())

    plt.figure(figsize=(10,10))
    show_mnist_batch(sample_batched)
    plt.axis('off')
    plt.ioff()
    plt.show()
    break


# ## Training

# In[ ]:


model = SimpleCNN(h=28,w=28,c=1,num_class=10)
learn = Learner(model)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'learn.fit(train_dl=train_dataloader,\n          valid_dl=valid_dataloader,\n          lr=0.01,\n          epochs=50)')


# ### Predict and write submission

# In[ ]:


get_ipython().run_cell_magic('time', '', 'yp = learn.predict(valid_dataloader)')


# In[ ]:


get_ipython().run_cell_magic('time', '', "acc = accuracy(valid_df.label.values,yp)\nce = cross_entropy(valid_df.label.values,yp)\nprint('Valid ACC: %.4f Cross Entropy:%4f'%(acc,ce))")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'yp = learn.predict(test_dataloader)')


# In[ ]:


sub = pd.DataFrame()
sub['ImageId'] = np.arange(yp.shape[0])+1
sub['Label'] = np.argmax(yp,axis=1)
sub.head()


# In[ ]:


from datetime import datetime
clock = "{}".format(datetime.now()).replace(' ','-').replace(':','-').split('.')[0]
out = 'pytorch_%s_acc_%.4f_ce_%.4f.csv'%(clock,acc,ce)
print(out)
sub.to_csv(out,index=False)

