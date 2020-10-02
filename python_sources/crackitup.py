#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import numpy as np
import copy
import pandas as pd
import cv2
  
import pandas as pd
train=pd.read_csv("/kaggle/input/severstal-steel-defect-detection/train.csv")
train['imageId']=train['ImageId_ClassId'].map(lambda x : x[0:-2])
train['isFailurePresent']=train['EncodedPixels'].map(lambda x : 0 if pd.isnull(x) else 1)
train['defectType']=train['ImageId_ClassId'].map(lambda x : x[-1])

def getMask(encoding):
    mask=np.zeros(256*1600,dtype=np.uint8)
    encoding=encoding.split(' ')
    for curMaskPoint in list(zip([x for i,x in enumerate(encoding) if i%2==0],[x for i,x in enumerate(encoding) if i%2==1])):
        position=int(curMaskPoint[0])
        length=int(curMaskPoint[1])
        mask[position-1:position+length-1] = 1
    mask=mask.reshape(256,1600,order='F')
    return(mask)

train['mask']=train.apply(lambda row : getMask(row['EncodedPixels']) if row['isFailurePresent']==1 else 0,axis=1)
train.head(10)
print("Cell Execution Complete")


# In[ ]:


########################################################
# Exploring certain images to check the defects
########################################################



colourDifferentDefects = {'1':(249, 192, 12),'2':(0, 185, 241),'3':(114, 0, 218),'4':(249,50,12)}

imageList=train[train['isFailurePresent']==1]['ImageId_ClassId'].values[0:4]
for curImageIndex in imageList:
    imageVal,mask,defectType=train[train['ImageId_ClassId']==curImageIndex][['imageId','mask','defectType']].values[0]
    img = np.array(mpimg.imread("/kaggle/input/severstal-steel-defect-detection/train_images/{0}".format(imageVal)) )
    contours, _ = cv2.findContours(mask[:, :], cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        cv2.polylines(img, contours[i], True, colourDifferentDefects[defectType], 2)
    plt.figure(figsize=(10,5))
    plt.imshow(img)
print("Cell Execution Complete")


# In[ ]:


#################################################
#### Distribution of Defects
#################################################
train.groupby(['isFailurePresent','defectType'])['imageId'].nunique()
#isFailurePresent  defectType
# 0                 1             11671
#                   2             12321
#                   3              7418
#                   4             11767
# 1                 1               897
#                   2               247
#                   3              5150
#                   4               801


# In[ ]:


######################################################################################
# Breaking the steel image into smaller batches and use that for label classification
# For now we are focusing only on IS_DEFECT / IS_NO_DEFECT classification ( binary )
######################################################################################

# Working only on the first type of defects

# 256, 1600
# The breakDown in batches of 32, 40

import cv2

# train[train['isFailurePresent']==1].shape[0]
# 7095

trainingX=[]
trainingY=[]
testX=[]
testY=[]
for curRowIndex in range(5000):
    imageId,isFailure,defect,mask=train[['imageId','isFailurePresent','defectType','mask']].values[curRowIndex]
    img = np.array(mpimg.imread("/kaggle/input/severstal-steel-defect-detection/train_images/{0}".format(imageId)) )
    if(isFailure==1):
        for height in range(8):
            for width in range(40):
                numDefectPoints=np.sum(mask[height*32:(height+1)*32,width*40:(width+1)*40])
                if(numDefectPoints > 100):
                    trainingY.append(1)
                else:
                    trainingY.append(0)
                trainingX.append(img[height*32:(height+1)*32,width*40:(width+1)*40])
      
for curRowIndex in range(5000,7000):
    imageId,isFailure,defect,mask=train[['imageId','isFailurePresent','defectType','mask']].values[curRowIndex]
    img = np.array(mpimg.imread("/kaggle/input/severstal-steel-defect-detection/train_images/{0}".format(imageId)) )
    if(isFailure==1):
        for height in range(8):
            for width in range(40):
                numDefectPoints=np.sum(mask[height*32:(height+1)*32,width*40:(width+1)*40])
                if(numDefectPoints > 100):
                    testY.append(1)
                else:
                    testY.append(0)
                testX.append(img[height*32:(height+1)*32,width*40:(width+1)*40])

#for curRowIndex in range(int(train[train['isFailurePresent']==0].shape[0]/5)):
#    imageId,isFailure,defect,mask=train[['imageId','isFailurePresent','defectType','mask']].values[curRowIndex]
#    img = np.array(mpimg.imread("/kaggle/input/severstal-steel-defect-detection/train_images/{0}".format(imageId)) )
#    for height in range(8):
#        for width in range(40):
#            trainingY.append(0)
#            trainingX.append(img[height*32:(height+1)*32,width*40:(width+1)*40])

# Write the trainingX and trainingY to disk
trainingX=np.array(trainingX)
trainingY=np.array(trainingY)
testX=np.array(testX)
testY=np.array(testY)

del(train)

print("Cell Execution Complete")


# In[ ]:


import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=8,kernel_size=4,stride=1,padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=1,kernel_size=4,stride=1,padding=0)
        self.fc1 = nn.Linear(5*7*1, 8)
        self.fc2 = nn.Linear(8, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        #print("Forward Function Entry Size : {0}".format(x.size()))
        #x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv1(x)))
        #print("Forward Function Conv1 and Pool : {0}".format(x.size()))
        x = self.pool(F.relu(self.conv2(x)))
        #print("Forward Function Conv2 and Pool : {0}".format(x.size()))
        x = x.view(-1, 5 * 7 * 1)
        x = F.relu(self.fc1(x))
        #print("Forward Function fc1 : {0}".format(x.size()))
        x = self.fc2(x)
        # Applying Sigmoid
        x = self.sig(x)
        return x

model=BinaryClassification()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
epochRange=100000
miniBatches=10
batchSize=256

# Reshaping the data for pytorch purposes
trainingX=trainingX.reshape(-1,3,32,40).astype(np.float32)
trainingY=trainingY.reshape(-1,1).astype(np.float32)
testX=testX.reshape(-1,3,32,40).astype(np.float32)
testY=testY.reshape(-1,1).astype(np.float32)

print("Cell Execution Completed")


# In[ ]:


totalRows=trainingX.shape[0]
totalTestRows=testX.shape[0]

testBatchSize=2000

for epoch in range(epochRange):
    totalLoss=0
    optimizer.zero_grad()
    for curBatch in range(miniBatches):
        choiceList=np.random.choice(totalRows,batchSize)
        inputs=Variable(torch.from_numpy(trainingX[choiceList,:,:,:]))
        #print("Size of inputs is {0}".format(inputs.size()))
        labels=Variable(torch.from_numpy(trainingY[choiceList]))
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        #print("Before entering the loss criterion outputs={0} and labels={1}".format(outputs.size(),labels.size()))
        loss = criterion(outputs, labels)
        totalLoss=totalLoss + loss
    loss.backward()
    optimizer.step()
    if(epoch % 100==0):
        # We will do the testing
        totalTestLoss=0
        #for testIndex in range(int(totalTestRows/batchSize)):
        for testIndex in range(40):
            testInput=Variable(torch.from_numpy(testX[testBatchSize*testIndex:testBatchSize*(testIndex + 1),:,:,:]))
            testLabels=Variable(torch.from_numpy(testY[testBatchSize*testIndex:testBatchSize*(testIndex + 1)]))
            totalTestLoss=totalTestLoss + criterion(model(testInput), testLabels).detach().numpy().tolist()
        print("Epoch : {0} Train Loss : {1} Test Loss : {2}".format(epoch,totalLoss,totalTestLoss))

