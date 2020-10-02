#!/usr/bin/env python
# coding: utf-8

# **Image classification using Transfer Learning technique**
# > Objective: Building complex model on Pytorch using Transfer Learning that is able to classify complex images.
# 
# What is Transfer Learning?
# 
# Transfer learning (TL) is a research problem in machine learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks.
# 
# In this project I have used Transfer Learning technique to classify complex images.
# 
# The model I used here is award wining [vgg16](http:/https://neurohive.io/en/popular-networks/vgg16//) model with 13 convolutional layers for feature extraction and 3 fully connected layers for classification. The accuracy this model yield is around 91% which is amazing considering that the dataset I used is small, by increasing the size of dataset, much better accuracy can be achieved.
# I would recommend to use your own dataset and just use my model as a reference.
# 
# requirements:-
# 
# pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
# 

# In[ ]:


import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms, models # this models contains all the highly pretrained models.


# Initialize GPU

# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Clone the git repo for getting our dataset

# In[ ]:


get_ipython().system('git clone https://github.com/jaddoescad/ants_and_bees.git')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('ls ants_and_bees')


# In[ ]:


get_ipython().system('ls ants_and_bees/train')


# Apply transform and augmentation on our images

# In[ ]:


transform_train = transforms.Compose([transforms.Resize((224,224)), #We resize our data as vgg16 is trained on 224*224 size images
                                      transforms.RandomHorizontalFlip(), # flips the image at horizontal axis
                                      transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                                      transforms.ColorJitter(brightness=1, contrast=1, saturation=1), # We can change it according to the accuracy and requirements
                                      transforms.ToTensor(), # convert the image to tensor to make it work with torch
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])


transform = transforms.Compose([transforms.Resize((224,224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])

training_dataset = datasets.ImageFolder('ants_and_bees/train', transform=transform_train)
validation_dataset = datasets.ImageFolder('ants_and_bees/val', transform=transform)

training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=20, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = 20, shuffle=False)


# In[ ]:


print(len(training_dataset))
print(len(validation_dataset))


# Convert the images in order to plot them using plt

# In[ ]:


def im_convert(tensor):
  image = tensor.cpu().clone().detach().numpy()
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  image = image.clip(0, 1)
  return image


# In[ ]:


classes = ('ant', 'bee') # since our dataset has 2 classes


# Iter our data to plot our images. 

# In[ ]:


dataiter = iter(training_loader) # converting our train folder to iterable so that it can be iterated through a batch of 20.
images, labels = dataiter.next() # We get our inputs for our model here.
fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
  ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
  plt.imshow(im_convert(images[idx]))
  ax.set_title(classes[labels[idx].item()])


# We define our model here. We are using vgg16 here which has 16 layers

# In[ ]:


model = models.vgg16(pretrained=True) 


# In[ ]:


model


# In[ ]:


# Here we are freezing the feature parameters and using same as in vgg16. These parameters are very good as this model is trained on 14 million images for 2 weeks with many GPUs
for param in model.features.parameters():
  param.requires_grad = False # the feature layers do not require any gradient.


# In[ ]:


#We take the last fully connected layer from vgg16 and replace it with our layer as we need to classify only 2 different categories
import torch.nn as nn

n_inputs = model.classifier[6].in_features # the i/p features of our classifier model
last_layer = nn.Linear(n_inputs, len(classes))  # new layer that we want to put in there.
model.classifier[6] = last_layer  # replacing last layer of vgg16 with our new layer
model.to(device)  # Put the model in device for higher processing power
print(model.classifier[6].out_features)


# In[ ]:


criterion = nn.CrossEntropyLoss() # same as categorical_crossentropy from keras
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001) #lr is very low as we have a very samll dataset


# # Fitting our model with our inputs and displaying the progress.

# In[ ]:


epochs = 5 # lesser epochs as our dataset is small
running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []

for e in range(epochs):
  
  running_loss = 0.0
  running_corrects = 0.0
  val_running_loss = 0.0
  val_running_corrects = 0.0
  
  for inputs, labels in training_loader: #taking inputs and put it in our model which is inside device
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    _, preds = torch.max(outputs, 1)
    running_loss += loss.item()
    running_corrects += torch.sum(preds == labels.data)

  else:
    with torch.no_grad():
      for val_inputs, val_labels in validation_loader:
        val_inputs = val_inputs.to(device)
        val_labels = val_labels.to(device)
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs, val_labels)

        _, val_preds = torch.max(val_outputs, 1)
        val_running_loss += val_loss.item()
        val_running_corrects += torch.sum(val_preds == val_labels.data)
      
    epoch_loss = running_loss/len(training_loader.dataset)
    epoch_acc = running_corrects.float()/ len(training_loader.dataset) # We are now dividing the total loss of one epoch with the enitre length of dataset to get the probability between 1 & 0
    running_loss_history.append(epoch_loss)
    running_corrects_history.append(epoch_acc)
    
    val_epoch_loss = val_running_loss/len(validation_loader.dataset)
    val_epoch_acc = val_running_corrects.float()/ len(validation_loader.dataset)
    val_running_loss_history.append(val_epoch_loss)
    val_running_corrects_history.append(val_epoch_acc)
    print('epoch :', (e+1))
    print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
    print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))


# In[ ]:


plt.style.use("ggplot")
plt.plot(running_loss_history, label='training loss')
plt.plot(val_running_loss_history, label='validation loss')
plt.legend()


# In[ ]:


plt.style.use("ggplot")
plt.plot(running_corrects_history, label='training accuracy')
plt.plot(val_running_corrects_history, label='validation accuracy')
plt.legend()


# # We now plot new images from web and try to predict them using our model

# In[ ]:


import PIL.ImageOps  # # from python imaging library we take this so we can preform operations on our image


# In[ ]:


import requests
from PIL import Image

url = 'http://cdn.sci-news.com/images/enlarge5/image_6425e-Giant-Red-Bull-Ant.jpg'
response = requests.get(url, stream = True)
img = Image.open(response.raw)
plt.imshow(img)


# In[ ]:


img = transform(img)  # transform before i/p to our model.
plt.imshow(im_convert(img)) #converting the image to plot using plt


# Prediction:

# In[ ]:


image = img.to(device).unsqueeze(0) # unsqueeze to add one dimension
output = model(image)
_, pred = torch.max(output, 1)
print(classes[pred.item()])


# Testing my model using random images from the validation dataset.

# In[ ]:


dataiter = iter(validation_loader)
images, labels = dataiter.next()
images = images.to(device)
labels = labels.to(device)
output = model(images)
_, preds = torch.max(output, 1)

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
  ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
  plt.imshow(im_convert(images[idx]))
  ax.set_title("{} ({})".format(str(classes[preds[idx].item()]), str(classes[labels[idx].item()])), color=("green" if preds[idx]==labels[idx] else "red"))


# In[ ]:


# The model has performed really well using transfer learning


# In[ ]:




