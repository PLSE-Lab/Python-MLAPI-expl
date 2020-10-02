#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
import pathlib
from torch.utils.data import DataLoader
from torchvision import *


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


transformtrain = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])
transformtest = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])


# In[ ]:


trainds = datasets.ImageFolder('../input/seg_train/seg_train', transform=transformtrain)
testds = datasets.ImageFolder('../input/seg_test/seg_test', transform=transformtest)


# In[ ]:


trainloader = DataLoader(trainds, batch_size=256, shuffle=True)
testloader = DataLoader(testds, batch_size=64, shuffle=False)


# In[ ]:


root = pathlib.Path('../input/seg_train/seg_train')
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])


# In[ ]:


model = models.vgg19(pretrained=True).to(device)
for param in model.features.parameters():
    param.requires_grad = False


# In[ ]:


model


# In[ ]:


model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.00001)


# In[ ]:


trainlosses = []
testlosses = []
for e in range(50):
    trainloss = 0
    traintotal = 0
    trainsuccessful = 0
    for traininput, trainlabel in trainloader:
        traininputs, trainlabels = traininput.to(device), trainlabel.to(device)
        optimizer.zero_grad()
        trainpredictions = model(traininputs)
        _, trainpredict = torch.max(trainpredictions.data, 1)
        loss = criterion(trainpredictions, trainlabels)
        loss.backward()
        optimizer.step()
        trainloss += loss.item()
        traintotal += trainlabels.size(0)
        trainsuccessful += (trainpredict == trainlabels).sum().item()
    else:
        testloss = 0
        testtotal = 0
        testsuccessful = 0
        with torch.no_grad():
            for testinput, testlabel in testloader:
                testinputs, testlabels = testinput.to(device), testlabel.to(device)
                testpredictions = model(testinputs)
                _, testpredict = torch.max(testpredictions.data, 1)
                tloss = criterion(testpredictions, testlabels)
                testloss += tloss.item()
                testtotal += testlabels.size(0)
                testsuccessful += (testpredict == testlabels).sum().item()
        trainlosses.append(trainloss/len(trainloader))
        testlosses.append(testloss/len(testloader))
        print('Train Accuracy %{:.2f}'.format(100*trainsuccessful/traintotal))
        print('Test Accuracy %{:.2f}'.format(100*testsuccessful/testtotal))


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.plot(trainlosses, label='Training loss', color='green')
plt.plot(testlosses, label='Validation loss', color='black')
plt.legend(frameon=False)
plt.show()


# Seems there is overfitting. You could use L2 regularization or dropout to get better test accuracy if you wish to give it a try.

# In[ ]:


get_ipython().system('ls ../input/seg_pred/seg_pred/3966.jpg')


# __Let's predict__

# In[ ]:


from PIL import Image
import numpy as np


# In[ ]:


img = Image.open('../input/seg_pred/seg_pred/3966.jpg')


# In[ ]:


nimg = np.array(img)


# In[ ]:


plt.imshow(nimg)


# In[ ]:


pimg = transformtest(img).unsqueeze(0).to(device)


# In[ ]:


pimg.shape


# In[ ]:


prediction = model(pimg)


# In[ ]:


_, tpredict = torch.max(prediction.data, 1)


# In[ ]:


classes[tpredict[0].item()]


# It is amazing, isn't it?
