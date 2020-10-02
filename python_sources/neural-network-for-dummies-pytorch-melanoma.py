#!/usr/bin/env python
# coding: utf-8

# [Thanks for this source for getting reference on building pipeline](https://www.kaggle.com/nroman/melanoma-pytorch-starter-efficientnet?scriptVersionId=36599817)

# # **What are we going to build  ?**
# ### We will be building a neural network which consumes an image to predict the probability of being cancer 

# # Setup Block

# In[ ]:


get_ipython().system('pip install efficientnet_pytorch torchtoolbox')


# **Why do we need these ?**
# > These are some of the libraries we need to import for running the network

# In[ ]:


import pandas as pd
import numpy as np
import cv2
from efficientnet_pytorch import EfficientNet
import torchtoolbox.transform as transforms
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torch.utils import data

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().system('pip install torchsummary')
from torchsummary import summary


# **What is this MelanomaDataset class definition useful for ?**
# > In Pytorch we design our dataset as per torch.utils.data.Dataset which helps in training the model.<br>
# We need to implement 3 classes
# -  \_\_init__ : The method which is useful for setting up the variables
# - \_\_getitem__ : The method needed to access each item in the dataset
# - \_\_len__ : The method needed to get the length of dataset

# In[ ]:


from torch.utils.data import Dataset, DataLoader
class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, imfolder: str, transforms_object= None):
        self.data_frame = df
        self.path_to_folder = imfolder
        self.transforms_object = transforms_object
        
    def __getitem__(self, index):
        image_name_at_index = self.data_frame.loc[index,'image_name']
        load_path = self.path_to_folder +image_name_at_index+'.jpg'
        image_data = cv2.imread(load_path)
        if self.transforms_object:
            image_data = self.transforms_object(image_data)
        if 'target' in self.data_frame.columns.values:
            y = self.data_frame.loc[index,'target']
        else :
            y = 1
        return image_data,y
        
    def __len__(self):
        return self.data_frame.shape[0]


# **Is this the actual Neural network Architecture  ?**
# > Yes. The variable 'arch' is actually a pretrained neural network architecture called Efficientnet-B1 <br>
# > We are build layers over this "network" > 128 > 16 > 1 <br>
# > The last layer gives a number whose transformation will give us the probability <br>
# 
# 

# In[ ]:




class deeper_network(nn.Module):
    def __init__(self,arch):
        super(deeper_network,self).__init__()
        self.arch = arch
        self.arch._fc = nn.Linear(in_features=1280,out_features=512, bias=True)
        self.fin_net = nn.Sequential(self.arch,
                                     nn.Linear(512,128),
                                     nn.LeakyReLU(),
                                     nn.Linear(128,16),
                                     nn.LeakyReLU(),
                                     nn.Linear(16,1))
    def forward(self,inputs):
        output = self.fin_net(inputs)
        return output


# # Training Block

# **So what's the next step ?**
# > We will now prepare the data for training with transformation to suit our pipeline<br>
# > The data will be taken in via a dataframe<br>
# 

# In[ ]:


train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test_df = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')


# In[ ]:


train_transform = transforms.Compose([
                                    transforms.RandomResizedCrop(size=256,scale=(0.7,1)), # Take 70 - 100 % of the area and scale the image to 256 x 256 size
                                    transforms.RandomHorizontalFlip(),# Take the image and flip it horizontally or not 50% chance
                                    transforms.RandomVerticalFlip(), # Take the image and flip it vertically or not 50% chance
                                    transforms.ToTensor(), # Convert to tensor
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) # Adjust the values of image to standardise


# In[ ]:


model_eff = EfficientNet.from_pretrained('efficientnet-b1')
# print(summary(model_eff,(3, 256, 256)))


# **What's a dataloader and why do we need it ?**
# > Data loader is a part of torch.utils package and it gives us an iterable object.<br>
# > For training the model we need data in batches with some level of sophistication such as shuffling, batch size etc<br>
# > Data loader can provide all these functionalities for us
# 

# In[ ]:


path_for_jpeg= '../input/siim-isic-melanoma-classification/jpeg/train/'
train_dataset = MelanomaDataset(train_df,path_for_jpeg,transforms_object=train_transform)
train_loader_args = dict(shuffle=True, batch_size=64)
train_loader = data.DataLoader(train_dataset, **train_loader_args)


# # Model Setup

# **How we actually proceed now for training ?**
# > To train the network we need only 3 things <br>
# - THE NETWORK : Yes. Pretty obvious. We need the main network "deep_net" mentioned below
# - THE CRITERION : Criterion is a evaluation method which helps the model evaluate the predictions and gives error values on each prediction. Here its 'BCEWithLogitsLoss'
# - THE OPTIMIZER : Optimizer is kind of a correction module. It corrects the model itself by updating values which pleases the criterion. Here we are using 'Adam'

# In[ ]:


arch = EfficientNet.from_pretrained('efficientnet-b1') 

#setup
deep_net = deeper_network(arch)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(deep_net.parameters())

# Sometimes, I just use my previously trained network and load it just before training again
deep_net.load_state_dict(torch.load('../input/effnet-v2/effnet_v2',map_location=torch.device('cpu'))) 


# **What is cuda ?**
# > Ahh.. It is a command which enables the use of GPU <br>
# 
# **But what's GPU ?**
# > Its fullform is Graphics processing Unit. Inshort it can process things **much** faster CPU <br>
# > Its one of those things that gave Deep learning wings to fly

# In[ ]:


torch.cuda.is_available()


# In[ ]:


use_cuda = True
if use_cuda and torch.cuda.is_available():
    deep_net.cuda() # converting the model into GPU enabled variable


# # Model Training

# In[ ]:


model = deep_net
import time


# **Whats this piece of code ?**
# > Read the comments. This is the crux of all the drama we have been doing. <br>
# > This is how we train the model. If you dont understand. Its ok !

# In[ ]:


model.train()
for e in range(2,4):
    
    # variables to log results
    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0
    start_time = time.time()
    
    #loop for running the training in batches
    for batch_idx, (image_data_array, target) in enumerate(train_loader):
        
        #setting up batch data 
        optimizer.zero_grad()   # .backward() accumulates gradients
        image_data_array = image_data_array.float().cuda()
        target = target.long().cuda() # all data & model on same device
        
        #Prediction
        outputs = model(image_data_array)

        #Measuring the Error
        loss = criterion(outputs, target.reshape(-1,1).float())
        
        #Logging Error
        predictions = torch.round(torch.sigmoid(outputs))
        total_predictions += target.size(0)
        correct_predictions += (target.cpu() ==predictions.squeeze().cpu()).sum().item()
        running_loss += loss.item()
        
        #Correcting the model to reduce the error
        loss.backward()
        optimizer.step()
    
    acc = (correct_predictions/total_predictions)*100.0
    end_time = time.time()

    running_loss /= len(train_loader)
    print('Training Loss: ', round(running_loss,3), 'Time: ',round(end_time - start_time,3), 's')
    print('Training Accuracy: ', round(acc,3), '%')
    


# **What are we missing ?**
# - We have not at all validated the model. Whether its overfitting or underfitting 
# - We havent fine tuned the model to improve accuracy is on validation set
# - We have not played with criterion or optimizers<br>
# 
# **But why havent we done all this ^^ ?**
# - I am too lazy to do all of that and explain. But in a forked notebook of this version, I will do all of it and some sophisticated techniques but wont explain.

# In[ ]:


torch.save(model.state_dict(), 'effnet_v1')


# # Prediction Block

# **So now we need to predict on test data ?**
# > Yes. Get the test data, pass it through same transformation but DONT SHUFFLE (please)

# In[ ]:


path_for_jpeg= '../input/siim-isic-melanoma-classification/jpeg/test/'
test_dataset = MelanomaDataset(test_df,path_for_jpeg,transforms_object=train_transform)
test_loader_args = dict(shuffle=False, batch_size=10) # DONT SHUFFLE
test_loader = data.DataLoader(test_dataset, **test_loader_args)
model.eval()


# **Why is there a sigmoid function in the script?**
# > The last output of nueral network is float point number with dimenion 1 which is also unrestricted. While evaluating we use sigmoid inside BCEWithLogitsLoss. Thus while predicting we need to use sigmoid to convert that boundless number into probability.<br>
# 
# **What's sigmoid ?**
# > sigmoid(x) = 1/(1 +e^(-x))
# 

# In[ ]:


fin_temp=np.empty((0,))
for batch_idx, (image_data_array, target) in enumerate(test_loader):
    image_data_array = image_data_array.float()#.cuda()
    target = target.long()#.cuda() # all data & model on same device
    outputs = model(image_data_array)
    temp = torch.sigmoid(outputs).cpu().detach().numpy().squeeze()
    fin_temp = np.concatenate([fin_temp,temp])


# In[ ]:


Y_submission = test_df[['image_name']].copy()
Y_submission['target'] = fin_temp


# **Are we done ? Can I go home ?**
# > Yeah. Its the end. Upload the csv file and get your ranking on leaderboard.

# In[ ]:


Y_submission.to_csv('/kaggle/working/image_v3.csv',index=False)


# In[ ]:




