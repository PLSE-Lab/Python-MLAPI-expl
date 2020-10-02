#!/usr/bin/env python
# coding: utf-8

# # Introduction:
# 
# **Pneumonia is an infection in one or both lungs. Bacteria, viruses, and fungi cause it. The infection causes inflammation in the air sacs in your lungs, which are called alveoli. The alveoli fill with fluid or pus, making it difficult to breathe. Read on to learn more about pneumonia and how to treat it**
# 
# **Includes Diseases: Community-acquired pneumonia**
# 
# **Symptoms: Inflammation**
# 
# ![download.jpg](attachment:download.jpg)

# # Importing modules

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
import copy


# ## Getting Data

# In[ ]:


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])

transform_val = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor()
    ])

transform_test = transforms.Compose([
    transforms.RandomResizedCrop(224), 
    transforms.ToTensor(),
    ])
train_set = datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/train", transform = transform_train)
val_set = datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/val", transform = transform_val)
test_set= datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/test",transform=transform_test)


# ## Data Visualisation

# In[ ]:


num_classes = 2
batch_size = 8
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(val_set,batch_size=batch_size,shuffle=True)
testloader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True)


# In[ ]:


dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)

print(images[1].shape)
print(labels[1].item())

print("Classes in training set ")
print(train_set.classes)
print("Classes in validation set ")
print(val_set.classes)


# In[ ]:


def imshow(img, title):
    npimg = img.numpy()
    plt.figure(figsize=(15, 7))
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

def show_batch_images(dataloader):
    images, labels = next(iter(dataloader))
    img = torchvision.utils.make_grid(images,padding=25)
    imshow(img, title=["NORMAL" if x==0  else "PNEUMONIA" for x in labels])

for i in range(4):
    show_batch_images(trainloader)


# # Transfer learning and training VGG16

# In[ ]:


def evaluation(dataloader, model):
    total, correct = 0, 0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return (100 * correct / total)


# In[ ]:


vgg = models.vgg16_bn(pretrained=True)
for param in vgg.parameters():
    param.requires_grad = False

final_in_features = vgg.classifier[6].in_features
vgg.classifier[6] = nn.Linear(final_in_features, num_classes)
for param in vgg.parameters():
    if param.requires_grad:
        print(param.shape)


# In[ ]:


vgg = vgg.to(device)
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(vgg.parameters(),lr=0.01)


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nloss_epoch_arr = []\nmax_epochs = 20\n\nmin_loss = 10000000\n\nn_iters = np.ceil(5300/batch_size)\n\nfor epoch in range(max_epochs):\n\n    for i, data in enumerate(trainloader, 0):\n\n        inputs, labels = data\n        inputs, labels = inputs.to(device), labels.to(device)\n\n        opt.zero_grad()\n\n        outputs = vgg(inputs)\n        loss = loss_fn(outputs, labels)\n        loss.backward()\n        opt.step()\n        \n        if min_loss > loss.item():\n            min_loss = loss.item()\n            best_model = copy.deepcopy(vgg.state_dict())\n            print('Min loss %0.2f' % min_loss)\n        \n        if i % 100 == 0:\n            print('Iteration: %d/%d, Loss: %0.2f' % (i, n_iters, loss.item()))\n            \n        del inputs, labels, outputs\n        torch.cuda.empty_cache()\n        \n    loss_epoch_arr.append(loss.item())\n\nvgg.load_state_dict(best_model)\nprint(evaluation(trainloader, vgg), evaluation(valloader, vgg))")


# In[ ]:


print("Acc on training set is {} and validation set is {}".format(evaluation(trainloader, vgg),evaluation(valloader, vgg)))


# In[ ]:


plt.plot(loss_epoch_arr)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("loss vs epochs")
plt.show()

