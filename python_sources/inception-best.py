#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# All the required imports


import pandas as pd
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch import optim
from skimage import io, transform
import torchvision.models as models
from PIL import Image
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Exploring train.csv file
df = pd.read_csv('../input/train.csv')
df.head()


# In[ ]:


#Dataset class

class ImageDataset(Dataset):
    

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame['Id'][idx])         # getting path of image
        image = Image.open(img_name).convert('RGB')                                # reading image and converting to rgb if it is grayscale
        label = np.array(self.data_frame['Category'][idx])                         # reading label of the image
        
        if self.transform:            
            image = self.transform(image)                                          # applying transforms, if any
        
        sample = (image, label)        
        return sample


# In[ ]:


# Transforms to be applied to each image (you can add more transforms), resizing every image to 3 x 224 x 224 size and converting to Tensor
transform = transforms.Compose([transforms.RandomResizedCrop(299),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()  ,
                                transforms.Normalize((0.5,), (0.5,))
                               
                                ])

trainset = ImageDataset(csv_file = '../input/train.csv', root_dir = '../input/data/data/', transform=transform)     #Training Dataset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)                     #Train loader, can change the batch_size to your own choice


# In[ ]:


#Checking training sample size and label
for i in range(len(trainset)):
    sample = trainset[i]
    print(i, sample[0].size(), " | Label: ", sample[1])
    if i == 9:
        break


# In[ ]:


# Visualizing some sample data
# obtain one batch of training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(16):                                             #Change the range according to your batch-size
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))


# In[ ]:


get_ipython().run_line_magic('xmode', 'plain')
def train(learningrate=0.0001):
    print("FOR LEARNING RATE LR=",learningrate)
    model = models.inception_v3(pretrained=True)
    model
    # Define your CNN model here, shift it to GPU if needed
    ct = 0
    for child in model.children():
        ct += 1
        if ct < 12:#original 14
            for param in child.parameters():
                param.requires_grad = False

    
    model.fc=nn.Linear(2048,67)
    model.AuxLogits.fc = nn.Linear(768, 67)


    criterion = nn.CrossEntropyLoss()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)


    epoch= 20
    for epochs in range(epoch):
        print(epochs)
        for images,labels in trainloader:
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
            optimizer.zero_grad()
            outputs, aux_outputs = model(images)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4*loss2
            #loss=criterion(outputs,labels) #originally present
            loss.backward()
            optimizer.step()
        print("Loss:",float(loss.data))
    return model

lr=0.0001
model=train(lr)
            


# In[ ]:


model


# In[ ]:



#Exit training mode and set model to evaluation mode
model.eval() # eval mode


# In[ ]:


# Reading sample_submission file to get the test image names
submission = pd.read_csv('../input/sample_sub.csv')
submission.head()


# In[ ]:


#Loading test data to make predictions

testset = ImageDataset(csv_file = '../input/sample_sub.csv', root_dir = '../input/data/data/', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=0)


# In[ ]:


# iterate over test data to make predictions
predictions=[]
for data, target in testloader:
    # move tensors to GPU if CUDA is available
    data, target = data.cuda(), target.cuda()
    output = model(data)
    _, pred = torch.max(output, 1)
    for i in range(len(pred)):
        predictions.append(int(pred[i]))
        

submission['Category'] = predictions             #Attaching predictions to submission file


# In[ ]:


get_ipython().system('pip install kaggle')
import os
submission.to_csv('submission.csv', index=False, encoding='utf-8')
os.environ['KAGGLE_USERNAME'] = "blackrosedragon2" # username from the json file
os.environ['KAGGLE_KEY'] = "f6b2d34630e576848cb1ef4cb1b6501a" # key afrom the json file
get_ipython().system('kaggle competitions submit -c qstp-deep-learning-2019 -f submission.csv -m "Message"')
