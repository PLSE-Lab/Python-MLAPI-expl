#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import os
print(os.listdir("../input"))

# import module we'll need to import our custom module
from shutil import copyfile

# copy our file into the working directory (make sure it has .py suffix)
copyfile(src = "../input/view-class/view_helper.py", dst = "../working/view_helper.py")

# import all our functions
import view_helper as helper

# Any results you write to the current directory are saved as output.


# In[ ]:


print(os.listdir("../input/dogs-vs-cats-for-pytorch/cat_dog_data/Cat_Dog_data"))


# In[ ]:


data_dir = '../input/dogs-vs-cats-for-pytorch/cat_dog_data/Cat_Dog_data'

# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)


# In[ ]:


# change this to the trainloader or testloader 
data_iter = iter(trainloader)

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,10), ncols=4)
for ii in range(4):
    ax = axes[ii]
    helper.imshow(images[ii], ax=ax, normalize=False)


# In[ ]:


# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

model.to(device);


# In[ ]:


torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


# epochs = 1
# steps = 0
# running_loss = 0
# print_every = 5
# for epoch in range(epochs):
#     for inputs, labels in trainloader:
#         steps += 1
#         # Move input and label tensors to the default device
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         optimizer.zero_grad()
        
#         logps = model.forward(inputs)
#         loss = criterion(logps, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
        
#         if steps % print_every == 0:
#             test_loss = 0
#             accuracy = 0
#             model.eval()
#             with torch.no_grad():
#                 for inputs, labels in testloader:
#                     inputs, labels = inputs.to(device), labels.to(device)
#                     logps = model.forward(inputs)
#                     batch_loss = criterion(logps, labels)
                    
#                     test_loss += batch_loss.item()
                    
#                     # Calculate accuracy
#                     ps = torch.exp(logps)
#                     top_p, top_class = ps.topk(1, dim=1)
#                     equals = top_class == labels.view(*top_class.shape)
#                     accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
#             print(f"Epoch {epoch+1}/{epochs}.. "
#                   f"Train loss: {running_loss/print_every:.3f}.. "
#                   f"Test loss: {test_loss/len(testloader):.3f}.. "
#                   f"Test accuracy: {accuracy/len(testloader):.3f}")
#             running_loss = 0
#             model.train()


# In[ ]:


print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())


# In[ ]:


#torch.save(model.state_dict(), 'cat_dog_dense121.pth')


# In[ ]:


print(os.listdir("../input/dogs-vs-cats-for-pytorch/cat_dog_data/Cat_Dog_data"))
print(os.listdir("../input/cat-dog-dense121-local"))


# In[ ]:


ls


# In[ ]:


state_dict = torch.load('../input/cat-dog-dense121-local/cat_dog_dense121_local.pth')
print(state_dict.keys())


# In[ ]:


model.load_state_dict(state_dict)
model


# In[ ]:


# change this to the trainloader or testloader 
data_iter = iter(testloader)

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,10), ncols=4)
for ii in range(4):
    ax = axes[ii]
    helper.imshow(images[ii], ax=ax, normalize=False)


# ## Test the Network

# In[ ]:


model.to('cpu')
model.eval()

data_iter = iter(testloader)

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,10), ncols=4)

for ii in range(4):
    ax = axes[ii]
    helper.imshow(images[ii], ax=ax, normalize=False)


# In[ ]:


with torch.no_grad():
    output = model.forward(images)

ps = torch.exp(output)


# In[ ]:


random_img = np.random.randint(64, size=1)[0]
random_img


# In[ ]:


# get the probability
probability = ps[random_img].data.numpy().squeeze()
probability


# In[ ]:


helper.imshow(images[random_img], normalize=False)


# In[ ]:


ind = np.arange(2)
labels = ['Cat', 'Dog',]
width = 0.35
locations = ind

class_probability = plt.barh(ind, probability, width, alpha=.7, label='Cats vs Dogs')

plt.yticks(np.arange(10))
plt.title('Class Probability')
plt.yticks(locations, labels)

#legend
plt.legend()
plt.ylim(top=3)
plt.ylim(bottom=-2)
plt.show();


# In[ ]:




