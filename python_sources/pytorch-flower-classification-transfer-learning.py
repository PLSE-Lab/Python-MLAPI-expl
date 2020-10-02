#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torchvision import datasets, transforms
from torch.autograd import Variable

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Image augmentation and normalization** 
# 
# - Transforms can be chained together using Compose
# - In image augmentation we randomly flip images, so that our model can detect wrongly oriented images too
# - All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. 
# - Normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
# - We first Resize the image to 256 then crop it to 224, so that it doesnt cut important features

# In[ ]:


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

test_transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])


# In[ ]:


data_dir = "../input/flowers_/flowers_/"


# A call to ImageFolder(Path, Transform) applies our transformations to all the images in the specified directory.
# We will create a dictorionary called img_dataset for train and test folder**

# In[ ]:


img_datasets ={}


# In[ ]:


# That's how easily you can for images folders in Pytorch for further operations
img_datasets['train']= datasets.ImageFolder(data_dir + '/train', train_transform)
img_datasets['test']= datasets.ImageFolder(data_dir + '/test', test_transform)


# Classes Present

# In[ ]:


# these gets extracted from the folder name
class_names = img_datasets['train'].classes
class_names


# In[ ]:


# these gets extracted from the folder name - class label mapping
class_idx = img_datasets['train'].class_to_idx
class_idx


# Creating Train & Test DataLoaders

# In[ ]:


train_loader = torch.utils.data.DataLoader(img_datasets['train'],
                                                   batch_size=10,
                                                   shuffle=True,
                                                   num_workers=4)

test_loader = torch.utils.data.DataLoader(img_datasets['test'],
                                                   batch_size=10,
                                                   shuffle=True,
                                                   num_workers=4)


# Let's examing a Batch of training Data

# In[ ]:


images , labels = next(iter(train_loader))
images.shape


# - 10 - number of images in a single batch
# - 3 - number channels 
# - 224 - width & height of the image

# In[ ]:


# lets look at the labels
labels


# All of the pretrained models are present inside torchvision , in this tutorial we will use vgg16 pretrained layer.
# PS: In Kaggle to download the pretrained model , you need to set Internet to On in settings.

# In[ ]:


import torchvision.models as models

model = models.vgg16(pretrained=True)


# **Freezing model's layers:**
# 
# We will freeze all the layers in the network except the final layer.
# requires_grad == False will freeze the parameters so that the gradients are not computed in backward() i.e. weights of these layers won't be trained

# In[ ]:


for param in model.parameters():
    param.required_grad = False


# In[ ]:


# Now let's check the model archietecture
model


# If you remember we have five classes i.e. five class image classification , in the above print out if you look closely the (classifier)
# section - this is doing something else. We need to change the classifier to make it a 5 class classifier.
# 
# we need to feed the no of input features to the linear layer (classifier[0]) to our newly created linear layer and output would be 5.

# In[ ]:


num_of_inputs = model.classifier[0].in_features
num_of_inputs


# In[ ]:


# restructaring the classifier
import torch.nn as nn
model.classifier = nn.Sequential(
                      nn.Linear(num_of_inputs, 5),
                        nn.LogSoftmax(dim=1))


# In[ ]:


# Now let's check the model archietecture again to see the changes 
model


# Hope you can see the changes in the classifier layer

# In[ ]:


# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()


# In[ ]:


# loss function and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)


# In[ ]:


# number of epochs to train the model
n_epochs = 10


for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    train_accuracy = 0
    
    ###################
    # train the model #
    ###################
    model.train() # prep model for training
    for data, target in train_loader:
        if train_on_gpu:
            data, target = Variable(data.cuda()), Variable(target.cuda())
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)
        #calculate accuracy
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == target.view(*top_class.shape)
        train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
# calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch+1, 
            train_loss
            ))
    print(f"Train accuracy: {train_accuracy/len(train_loader):.3f}")


# In[ ]:


# Checking Test Performence
test_accuracy = 0
model.eval() # prep model for evaluation
for data, target in test_loader:
    if train_on_gpu:
        data, target = Variable(data.cuda()), Variable(target.cuda())
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    #calculate accuracy
    ps = torch.exp(output)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == target.view(*top_class.shape)
    test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Test accuracy: {test_accuracy/len(test_loader):.3f}")


# Accuracy can be improved by changing the classifer archietecture !! 
