#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch.autograd import Variable
from tqdm import tqdm_notebook

from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from glob import glob
from random import choice


# The Test and and Training folder have the structure:
# 
#     Test/Nectarine/62_100.jpg
#     Test/Nectarine/68_100.jpg
#     ...
#     Test/Kaki/254_100.jpg
#     Test/Kaki/265_100.jpg
#     ...
#     
#  This is the way [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) expects, so we can quickly build the dataset:

# In[ ]:


transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor()
])

train_dataset = ImageFolder('../input/fruits-360_dataset/fruits-360/Training/', transform=transform)
test_dataset = ImageFolder('../input/fruits-360_dataset/fruits-360/Test/', transform=transform)


# (224, 224) is the input shape used in the pretrained models. Here's a sample image:

# In[ ]:


transforms.ToPILImage()(train_dataset[0][0])


# From the `train_dataset` we can extract the number of classes:

# In[ ]:


n_classes = len(train_dataset.classes)


# Now we can load the pretrained model and change the last layer. Notice that the last layer is linear, without softmax. The softmax operation is computed insided the loss function.

# In[ ]:


model = models.resnet18(pretrained=True)


# In[ ]:


n_filters = model.fc.in_features
model.fc = nn.Linear(n_filters, n_classes)


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


# In[ ]:


model.cuda()


# In[ ]:


train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# Let's train the model for 2 epochs only:

# In[ ]:


hist = {'loss': [], 'val_loss': [], 'val_acc': []}
num_epochs = 2
for epoch in range(num_epochs):
    print('Starting epoch {}/{}'.format(epoch+1, num_epochs))
    # train
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.data[0]
    
    train_loss = running_loss / len(train_loader)
    
    # evalute
    model.eval()
    val_running_loss = 0.0
    correct = 0
    for images, labels in test_loader:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        val_running_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.data).sum()
    
    
    val_loss = val_running_loss / len(test_loader)
    val_acc = correct / len(test_dataset)
    
    hist['loss'].append(train_loss)
    hist['val_loss'].append(val_loss)
    hist['val_acc'].append(val_acc)
    
    print('loss: {:.4f}  val_loss: {:.4f} val_acc: {:4.4f}\n'.format(train_loss, val_loss, val_acc))


# ## Results

# In[ ]:


model.eval()
y_test = []
y_pred = []
for images, labels in test_loader:
    images = Variable(images.cuda())
    labels = Variable(labels.cuda())
    outputs = model(images)
    _, predictions = outputs.max(1)
    
    y_test.append(labels.data.cpu().numpy())
    y_pred.append(predictions.data.cpu().numpy())
    
y_test = np.concatenate(y_test)
y_pred = np.concatenate(y_pred)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


sns.heatmap(confusion_matrix(y_test, y_pred))


# In[ ]:


idx_to_class = {idx: key for (key, idx) in train_dataset.class_to_idx.items()}


# In[ ]:


fruits = glob('../input/fruits-360_dataset/fruits-360/Test/*/*.jpg', recursive=True)


# In[ ]:


def what_fruit_is_this(fruit_path):
    img = Image.open(fruit_path)
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])
    x = Variable(transform(img)).cuda().unsqueeze(0)
    output = model(x)
    _, prediction = output.max(1)
    prediction = prediction.data[0]
    plt.imshow(img)
    plt.axis('off')
    plt.title('I think this fruit is a... {}!'.format(idx_to_class[prediction]))


# In[ ]:


choice(fruits)


# In[ ]:


what_fruit_is_this(choice(fruits))


# In[ ]:


what_fruit_is_this(choice(fruits))


# In[ ]:


what_fruit_is_this(choice(fruits))

