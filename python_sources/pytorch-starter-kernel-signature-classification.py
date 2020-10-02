#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from torch.utils.data import DataLoader,Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


train_dir = '../input/signature-verification-dataset/sign_data/train'
test_dir = './input/signature-verification-dataset/sign_data/test'


# In[ ]:


df_train = pd.read_csv('../input/signature-verification-dataset/sign_data/train_data.csv')


# In[ ]:


# first column is actually a header
df_train


# In[ ]:


import imageio


# In[ ]:


img_arr = imageio.imread('/kaggle/input/signature-verification-dataset/sign_data/sign_data/test/056_forg/01_0105056.PNG')
img_arr.shape


# In[ ]:


df_train.describe()


# ## Preprocessing and loading data

# In[ ]:


get_ipython().system(' python -m pip install image')


# In[ ]:



class SignatureDataprocessing():
    def __init__(self, train_csv=None, train_dir=None, transform=None):
        self.train_df = train_csv
        #label =>0 or 1 means forged or unforged
        self.train_df.columns = ["img1","img2","label"]
        self.train_dir = train_dir
        self.transform = transform
        
    def __getitem__(self, index):
        img1_path = os.path.join(self.train_dir, self.train_df.iat[index,0])
        img2_path = os.path.join(self.train_dir, self.train_df.iat[index,1])
        # convert images and use grey level conversion
        im0 = Image.open(img1_path)
        im1 = Image.open(img2_path)
        im0 = im0.convert('L')
        im1 = im1.convert('L')
        
        if self.transform is not None:
            im0 = self.transform(im0)
            im1 = self.transform(im1)
       # return im0, im1, torch.from_numpy(np.array([int(self.train_df.iat(index,2))]), dtype=np.float32)
        return im0, im1, torch.from_numpy(np.array([int(self.train_df.iat[index,2])],dtype=np.float32))
    
    def __len__(self):
        return len(self.train_df)


# In[ ]:


from torchvision import transforms

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])

Siamesedataset = SignatureDataprocessing(df_train,train_dir, transform)


# In[ ]:


import os
from PIL import Image
import PIL.ImageOps 



# Viewing the sample of images and to check whether its loading properly
vis_dataloader = DataLoader(Siamesedataset,
                        shuffle=True,
                        batch_size=8)
dataiter = iter(vis_dataloader)


example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
# imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())

# what is the output


# In[ ]:


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            
            nn.Conv2d(1, 96, kernel_size=11,stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

        )
        
        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128,2))
        
  
  
    def forward_once(self, x):
        # Forward pass 
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2


# In[ ]:


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


# In[ ]:


train_dataloader = DataLoader(Siamesedataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=32)


# In[ ]:


# Check whether you have GPU is loaded or not
if torch.cuda.is_available():
    print('Yes')


# In[ ]:


# from torch.optim import lr_scheduler
import torch.optim
net = SiameseNetwork().cuda()
# Decalre Loss Function
criterion = ContrastiveLoss()
# Declare Optimizer
optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)


# In[ ]:


def train():
    counter = []
    loss_history = [] 
    iteration_number= 0
    
    for epoch in range(0,5):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            if i %50 == 0 :
                print("Epoch number {} Current loss {}".format(epoch,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    return net


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Train the model
model = train()
torch.save(model.state_dict(), "model.pt")
print("Model Saved Successfully")


# In[ ]:


model


# In[ ]:


m2 = torch.load("model.pt")
m2 = train()


# In[ ]:


state_dict = model.state_dict()

checkpoint = torch.load("model.pt")
for key in checkpoint.keys():
    if key not in state_dict.keys():
        continue
    if checkpoint[key].size() != state_dict[key].size():
        continue
    state_dict[key] = checkpoint[key]
model.load_state_dict(state_dict)


# In[ ]:


model = train()


# In[ ]:




