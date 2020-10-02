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


imageList=train['imageId'].unique()
imageListLen=len(imageList)
for curImageBatchCount in range(20):
    trainingX=[]
    trainingY=[]
    for curImageIndex in range(curImageBatchCount*100,(curImageBatchCount+1)*100):
        curImageY=[]
        subData=train[train['imageId']==imageList[curImageIndex]]
        img = np.array(mpimg.imread("../input/severstal-steel-defect-detection/train_images/{0}".format(imageList[curImageIndex])) )
        for defectType in ['1','2','3','4']:
            curMask=subData[subData['defectType']==defectType]['mask'].values
            if(len(curMask)==1):
                curMask=np.zeros((256,1600))
            curImageY.append(curMask)
        trainingX.append(img)
        trainingY.append(curImageY)
    trainingX=np.array(trainingX).reshape(-1,3,256,1600).astype(np.float32)
    trainingY=np.array(trainingY).reshape(-1,4,256,1600).astype(np.float32)
    np.save("./../trainingX_FullImage_{0}".format(curImageBatchCount),trainingX)
    np.save("./../trainingY_FullImage_{0}".format(curImageBatchCount),trainingY)
    print("Completed for {0} ".format(curImageBatchCount))
            
print("Cell Execution Completed")


# In[ ]:


# Using a pretrained model
from torchvision import models
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict 
import numpy as np

# ALEXNET MODEL
alexnet = models.alexnet(pretrained=True)

class Flatten(nn.Module):
    def forward(self, input):
        #print("Size of the input is {0}".format(input.size()))
        return input.view(-1, 4 * 256 * 100)

def resize2d(img, size):
    return (F.adaptive_avg_pool2d(img, size))
    
class Transfer(nn.Module):
    def __init__(self):
        super(Transfer, self).__init__()
        self.transferModelFeatures=models.alexnet(pretrained=True).features
        self.conv4=nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=2,stride=1,padding=0)
        self.relu4=nn.ReLU()
        #self.conv5=nn.ConvTranspose2d(in_channels=2048,out_channels=4096,kernel_size=2,stride=1,padding=0)
        #self.relu5=nn.ReLU()
        #self.conv6=nn.ConvTranspose2d(in_channels=1024,out_channels=2048,kernel_size=2,stride=1,padding=0)
        #self.relu6=nn.ReLU()
        self.flatten=Flatten()
        self.smax=nn.Softmax(dim=1)
        
    def forward(self,x):
        #print("Size is {0}".format(x.size()))
        x=self.transferModelFeatures(x)
        #print("Size is {0}".format(x.size()))
        x=F.relu(self.conv4(x))
        #print("Size is {0}".format(x.size()))
        x=self.smax(self.flatten(x))
        #print("Size is {0}".format(x.view(-1,4,128,100).size()))
        return(x.view(-1,4,256,100))
        

model=Transfer()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
epochRange=10
miniBatches=8
batchSize=16

for epoch in range(epochRange):
    for trainingCount in range(20):
        trainingX=np.load("./../trainingX_FullImage_{0}.npy".format(trainingCount))
        trainingY=np.load("./../trainingY_FullImage_{0}.npy".format(trainingCount))
        totalLoss=0
        for curMiniBatch in range(miniBatches):
            choiceList=np.random.choice(trainingX.shape[0],batchSize)
            inputs=Variable(torch.from_numpy(trainingX[choiceList,:,:,:]))
            #print("Size of inputs is {0}".format(inputs.size()))
            labels=Variable(torch.from_numpy(trainingY[[choiceList]]))
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(resize2d(outputs, (256,1600)).view(-1,1), labels.view(-1,1))
            totalLoss=totalLoss + loss
        totalLoss.backward()
        optimizer.step()
        if(epoch % 1==0):
            print("Epoch : {0} trainingCount : {1} Total Loss : {2}".format(epoch,trainingCount,totalLoss))    
print("Cell Execution Complete")


# In[ ]:


# Predictions

# Convert to run length encoding
def getRLE(imgMask):
    rle=''
    x=0
    while(x < imgMask.shape[0]):
        if(imgMask[x]==1):
            rle=rle + ' ' + str(x)  
            while(imgMask[x]==1):
                x=x+1
            rle=rle + ' ' + str(x) 
        x=x+1
    return(rle)

# Read the test Data
test=pd.read_csv("/kaggle/input/severstal-steel-defect-detection/sample_submission.csv")
test['imageId']=test['ImageId_ClassId'].map(lambda x : x[0:-2])
test['defectType']=test['ImageId_ClassId'].map(lambda x : x[-1])
for curRowIndex,curRow in test.iterrows():
    curImage=curRow['imageId']
    curDefect=curRow['defectType']
    curImageClassId=curRow['ImageId_ClassId']
    img = np.array(mpimg.imread("../input/severstal-steel-defect-detection/test_images/{0}".format(curImage)) )
    imgMask=np.zeros((4,img.shape[0],img.shape[1]))
    # Get Predictions
    predictions=model(Variable(torch.from_numpy(img.reshape(1,3,256,1600).astype(np.float32))))
    predictions=predictions.detach().numpy()
    predictions=cv2.resize(predictions.reshape(4,64,400),(256,1600))
    #cv2.resize(model(Variable(torch.from_numpy(img.reshape(1,3,256,1600).astype(np.float32)))).detach().numpy().reshape(4,256,100),(4,256,1600))
    predictions[predictions > 0.8]=1
    predictions[predictions <= 0.8]=0
    imgMask=predictions
    imgMask=imgMask.reshape(-1)
    for defectType in range(1,5):
        if(np.sum(imgMask==1)>0):
            curRLE=getRLE(imgMask[defectType-1])
        else:
            curRLE=''
        test.loc[(test['imageId']==curImage) & (test['defectType']==str(defectType)),'EncodedPixels']=curRLE

test[['ImageId_ClassId','EncodedPixels']].to_csv('./../Submission1.csv')


# In[ ]:


#import torch
#import random
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from torch.autograd import Variable
#from collections import OrderedDict 
#import numpy as np
#
#class Flatten(nn.Module):
#    def forward(self, input):
#        #print("Size of the input is {0}".format(input.size()))
#        return input.view(-1, 256 * 7 * 49)
#
#class Transfer(nn.Module):
#    def __init__(self):
#        super(Transfer, self).__init__()
#        self.SequentialModel1 = nn.Sequential(OrderedDict([
#        ('transferModelFeatures' , models.alexnet(pretrained=True).features),
#        ('flatten' , Flatten()),
#        ('fc1' , nn.Linear(256*7*49,512)),
#        ('relu1', nn.ReLU()),
#        ('fc2' , nn.Linear(512,5)),
#        ('smax',nn.Softmax(dim=1))
#        ]))
#        #self.transferModelFeatures=models.alexnet(pretrained=True).features
#        #256, 7, 49
#        #self.flatten=Flatten()
#        #self.fc1=nn.Linear(256*7*49,5)
#        #self.smax=nn.Softmax(dim=1)
#        
#    def forward(self,x):
#        x=self.SequentialModel1(x)
#        #x=self.transferModelFeatures(x)
#        #print("Size of x is {0}".format(x.size()))
#        #x=self.flatten(x)
#        #print("Size of x is {0}".format(x.size()))
#        #x=self.fc1(x)
#        #print("Size of x is {0}".format(x.size()))
#        #x=self.smax(x)
#        #print("Size of x is {0}".format(x.size()))
#        return(x)
#        
#    
#model=Transfer()
#optimizer = optim.Adam(model.parameters(), lr=0.01)
#criterion = nn.CrossEntropyLoss()
#epochRange=100
#miniBatches=16
#batchSize=32
#
#import random
#trainingX=np.load("./../trainingX_{0}.npy".format('0'))
#trainingY=np.load("./../trainingY_{0}.npy".format('0'))
#for epoch in range(epochRange):
#    for trainingCount in range(1):
#        #trainingX=np.load("./../trainingX_{0}.npy".format(trainingCount))
#        #trainingY=np.load("./../trainingY_{0}.npy".format(trainingCount))
#        totalLoss=0
#        for curMiniBatch in range(miniBatches):
#            choiceList=np.random.choice(trainingX.shape[0],batchSize)
#            inputs=Variable(torch.from_numpy(trainingX[choiceList,:,:,:].astype(np.float32)))
#            #print("Size of inputs is {0}".format(inputs.size()))
#            labels=Variable(torch.from_numpy(trainingY[[choiceList]])).type(torch.long)
#            # zero the parameter gradients
#            optimizer.zero_grad()
#            # forward + backward + optimize
#            outputs = model(inputs)
#            #print("Before entering the loss criterion outputs={0} and labels={1}".format(outputs.size(),labels.size()))
#            loss = criterion(outputs, labels)
#            totalLoss=totalLoss + loss
#        totalLoss.backward()
#        optimizer.step()
#        if(epoch % 1==0):
#            print("Epoch : {0} trainingCount : {1} Total Loss : {2}".format(epoch,trainingCount,totalLoss))    
#            
#gradCamTestData=trainingX[0].reshape(1,3,256,1600).astype(np.float32)
#gradCamTestDataTorch=Variable(torch.from_numpy(gradCamTestData))
#print("Shape of the transformed data")
#print(alexnet.features(gradCamTestDataTorch).shape)            


# In[ ]:


#################
### GRAD CAM
#################

#import cv2
#import matplotlib.pyplot as plt
#%matplotlib inline

#class GetGradients():
#    def __init__(self, model, gradLayer, stopLayer):
#        self.model = model
#        self.gradLayer = gradLayer
#        self.stopLayer = stopLayer
#        self.gradient=None

#    def captureGradient(self, grad):
#        self.gradient=grad
        
#    def getObserveLayerGradient(self):
#        return(self.gradient)

#    def __call__(self, x):
#        layerOutput=None
#        for name, module in model._modules['SequentialModel1']._modules.items():
#            x = module(x)
#            if name == self.gradLayer:
#                x.register_hook(self.captureGradient)
#                layerOutput=x
#            if name == self.stopLayer:
#                break
#        return layerOutput, x.view(x.size()[0],-1)

#class GradCam:
#    def __init__(self, model,gradLayer,stopLayer):
#        self.model = model
#        self.model.eval()
#        self.gradLayer=gradLayer
#        self.stopLayer=stopLayer
#        self.getGradients = GetGradients(self.model,self.gradLayer,self.stopLayer)

#    def forward(self, input):
#        return self.model(input) 

#    def __call__(self, input):
#        gradLayerActivations,stopLayerOutput = self.getGradients(input)
#        print("Grad Layer Activations")
#        print(gradLayerActivations)
#        print("stopLayerOutput")
#        print(stopLayerOutput)
#        print("Size of stopLayerOutput is {0}".format(stopLayerOutput.size()))
#        # Getting the index which saw the maximum activation
#        index = np.argmax(stopLayerOutput.detach().numpy())
#        print("The index is {0}".format(index))
#        # Recreate the Last Layer with only that index active
#        one_hot = np.zeros((1, stopLayerOutput.size()[-1]), dtype = np.float32)
#        one_hot[0][index] = 1
#        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
#        one_hot = torch.sum(one_hot * stopLayerOutput)
#        # Releasing all the accumulated Grads
#        self.model.zero_grad()
#        # Backward propagation from the layer
#        one_hot.backward(retain_graph=True)
#        gradLayerGradients = self.getGradients.getObserveLayerGradient().detach().numpy()
#        # We will be doing it only for a single image hence 0
#        gradLayerActivations = gradLayerActivations.detach().numpy()[0,:]
#        # For each channel and image, we will be summing up the activations, hence summing up width and height a.k.a 2 and 3
#        # We add an index of 0 because we are doing it only for a single image
#        weights = np.mean(gradLayerGradients, axis = (2, 3))[0, :]
#        # The dimensions of the image are the height and width only of the activations at that time
#        gradCam = np.zeros(gradLayerActivations.shape[1 : ], dtype = np.float32)
#        for channelIndex, channelWeight in enumerate(weights):
#            gradCam += channelWeight * gradLayerActivations[channelIndex, :, :]
#        # This is a rough RELU truncation
#        gradCam = np.maximum(gradCam, 0)
#        # Resize it to the original image
#        gradCam = cv2.resize(gradCam, (256, 1600))
#        print("Size of gradCam is {0}".format(gradCam.shape))
#        print(gradCam)
#        # Scale the GradCam Mask
#        gradCam = gradCam - np.min(gradCam)
#        gradCam = gradCam / np.max(gradCam)
#        print("Size of gradCam is {0}".format(gradCam.shape))
#        return(gradCam)
    
#gradCamTestData=trainingX[0].reshape(1,3,256,1600).astype(np.float32)
#gradCamTestDataTorch=Variable(torch.from_numpy(gradCamTestData))

#grad_cam = GradCam(model,"transferModelFeatures","fc1")
#mask = grad_cam(gradCamTestDataTorch)
    
#plt.imshow(gradCamTestData.reshape(256,1600,3).astype(int))
#plt.show()
#plt.imshow(mask.reshape(256,1600))
#plt.show()

