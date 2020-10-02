#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install torchsummary')


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
import torchvision
from torchvision import transforms
import math
from PIL import Image
from torchsummary import summary 
from tqdm import trange 

get_ipython().run_line_magic('matplotlib', 'inline')

h, w = 96, 96 #better to start off with lower values, like 32 or 64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True #speeds up training in case of constant input size

print(device)


# In[ ]:


#we resize images while saving their aspect ratio and pad them with 255 (white background) to h*w size
def resize (img):
    im = Image.fromarray(img)
    im.thumbnail((h, w), Image.ANTIALIAS)
    im = np.array(im).astype('float32')
    return np.expand_dims(np.pad(im, (((h - im.shape[0]) // 2, (h - im.shape[0] + 1) // 2), ((w - im.shape[1]) // 2, (w - im.shape[1] + 1) // 2)), 'constant', constant_values=255), axis = -1)


# In[ ]:


data1 = np.load('../input/train-1.npy')
data2 = np.load('../input/train-2.npy')
data3 = np.load('../input/train-3.npy')
data4 = np.load('../input/train-4.npy')
train = np.concatenate((data1, data2, data3, data4), axis=0)


# In[ ]:


#data visualisation
plt.figure(figsize=(16, 20))
for n, (image, tag) in enumerate(train, 1):
    if n > 64:
        break
    plt.subplot(8, 8, n)
    plt.title(tag)
    plt.imshow(image, cmap='gray')


# In[ ]:


#switching from Unicode to 0-999
dct = dict(zip(range(1000), np.unique(train[:, 1])))
dct = {v:k for k,v in dct.items()}

train[:, 1] = [dct[code] for code in train[:, 1]] 


# In[ ]:


#validation split (random_state for better reproducibility of results)
train, test = train_test_split(train, test_size=0.2, random_state=113)

train_x_np, train_y_np = train[:, 0], train[:, 1].astype('int32')
test_x_np, test_y_np = test[:, 0], test[:, 1].astype('int32')


# In[ ]:


#garbage collector (frees some RAM)
import gc
gc.collect() 


# In[ ]:


#custom dataset is a good practice (helps with data preprocessing, augmentation, etc.)
class Hieroglyph_data(Dataset):
    def __init__(self, tX, tY = None,
                 transform = transforms.Compose(
                     [transforms.ToPILImage(), 
                      transforms.ToTensor(), 
                      transforms.Normalize(mean=(0.5,), std=(0.5,))]), train=True): 
                                           #normalizing in range [-1; 1] speeds up training, better than [0; 1]
        
        if train == True:
            self.X = tX
            self.y = torch.Tensor(tY).type(torch.LongTensor)
            self.transform = transform
            self.train = True
            
        else:
            self.X = tX
            self.y = None
            self.transform = transform
            self.train = False
            
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        #preprocessing on demand trades computations for memory
        if self.train == True:
            return self.transform(resize(self.X[idx])), self.y[idx] 
        else:
            return self.transform(resize(self.X[idx]))


# In[ ]:


#data augmentation significantly boosts accuracy and prevents overfitting (RandomAffine does the augmentation)
train_dataset = Hieroglyph_data(train_x_np, train_y_np, transform = transforms.Compose(
                                [transforms.ToPILImage(), 
                                 transforms.RandomAffine(
                                 degrees=(-10, 10), 
                                 translate=(.1, .1), 
                                 scale=(0.9, 1.1),
                                 shear=(-10, 10)),
                                 transforms.ToTensor(), 
                                 transforms.Normalize(mean=(0.5,), std=(0.5,))]))

test_dataset = Hieroglyph_data(test_x_np, test_y_np)


# In[ ]:


num_classes = 1000
batch_size = 512
num_epochs = 20

#pin_memory speeds up data transfer from CPU to GPU
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, pin_memory = True) 
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, pin_memory = True)


# In[ ]:


class PabloNet(nn.Module):    
    def __init__(self):
        super(PabloNet, self).__init__()
          
        self.layers_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            #nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            #nn.Dropout(0.25),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            #nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            #nn.Dropout(0.25),
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            
            nn.Conv2d(512, 1024, kernel_size=3, stride=1),
            nn.BatchNorm2d(1024),
            nn.PReLU(),
            #nn.AvgPool2d(kernel_size=2),
            #nn.Dropout(0.25),
        )
        
        self.layers_dense = nn.Sequential(
            nn.Linear(9216, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 1000)
        )
        
    def forward(self, x):
        x = self.layers_conv(x)
        x = x.view(x.size(0), -1)
        x = self.layers_dense(x)
        return x     
    
model = PabloNet()

model = model.to(device)


#summary from torchsummary library is a helpful feature that provides useful Keras-style model summary that helps
#you to evaluate model size, check if everything is ok, and roughly estimate batchsize.
#Install it with pip install torchsummary

summary(model, (1, h, w))


# In[ ]:


error = nn.CrossEntropyLoss()

learning_rate = 0.002

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
#lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
#this scheduler leads to more smooth convergence yet it seems to me it works better on bigger models that require 
#more epochs to train, and performs worse otherwise.


# In[ ]:


def train(epoch):
    model.train() #don't forget to switch between train and eval!
    
    running_loss = 0.0 #more accurate representation of current loss than loss.item()

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = error(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if (i + 1)% 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (i+ 1) * len(images), len(train_loader.dataset),
                    100. * (i + 1) / len(train_loader), running_loss / 50))
            
            running_loss = 0.0
            
def evaluate(data_loader):
    model.eval() 
    loss = 0
    correct = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss += F.cross_entropy(outputs, labels, reduction = 'sum').item()

            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
        
    loss /= len(data_loader.dataset)
        
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))


# In[ ]:


for epoch in range(num_epochs): 
    lr_scheduler.step()
    train(epoch)
    evaluate(test_loader)


# In[ ]:


#loading train data
pred = np.load('../input/test.npy')
pred_dataset = Hieroglyph_data(pred, train = False)
pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size = batch_size, shuffle = False)


# In[ ]:


def prediciton(data_loader):
    model.eval()
    test_pred = torch.LongTensor()
    
    with torch.no_grad():
        for i, images in enumerate(data_loader):
            images = images.to(device)

            outputs = model(images)

            pred = outputs.cpu().data.max(1, keepdim=True)[1]

            test_pred = torch.cat((test_pred, pred), dim=0)
        
    return test_pred


# In[ ]:


#preparing the predictions
pred_y = prediciton(pred_loader)
pred_y = pred_y.cpu().numpy().flatten()

dct = {v:k for k,v in dct.items()} #reversing keys and values leads to easy inversion from 0-999 back to Unicode 

pred_y = [dct[code] for code in pred_y]


# In[ ]:


#creating a submission csv using pandas
import pandas as pd
results = pd.Series(pred_y, name="Category", dtype = 'object')

submission = pd.concat([pd.Series(range(1,83248), name = "Id", dtype='object'), results],axis = 1)

submission.to_csv("pablo_submit.csv",index=False) 
#providing key information in submisson name is actually very important as you may need to reuse these submissions
#for blending, for example (it's a practice used in Kaggle competitions; you basically fit several models with 
#different architectures and then generate an "average" submission (it's mostly used in classification tasks, so
#it's basically voting, usually weighted voting: the right label for the sample is chosen as a label with most 
#'votes' - weighted entries of every submission file. Weights are often chosen in accordance with performance on
#public leaderboards. The more advanced technique of weight choice is automated search: we can use sklearn function
#that finds optimum of a given function with n variables. We can use validation holdout, fit various models and 
#generate submissions for this validation holdout. As we know correct labels for the holdout, we can calculate
#accuracy (or any other metrics used for submission evaluation) for blended submissions with given weights for 
#each of csvs. Then we optimize this function using sklearn and use found weights for the actual submission. Given
#that the validation holdout is representative enough, this will give us a significant boost. The main problem 
#here is the need for validation holdout to be the same in every model, so it's used not as frequent as it should
#be. Rough estimations based on LB score, or even equal weights will still give you a small boost in LB, and, more
#importantly, it increase the robustness of your submission; you are less likely to lose places because of the 
#overfit on public LB.

