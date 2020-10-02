#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from tqdm import tqdm # progress bar

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Import torch modules
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils import data as torch_data


# # 1. Preprocessing

# ## a. We load data and reshape it to normal image shape: 28x28 px

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.head(2)


# ## b. We check if any data is missing

# In[ ]:


print('Train data: {}'.format(df_train.isnull().any().describe()))
print('Test data: {}'.format(df_test.isnull().any().describe()))


# ### It turned out that everything is allright, so let us now check classes balance in train data

# In[ ]:


pixels_train = df_train.drop(['label'], axis = 1)
# Scale to 0..1
# pixels_train /= 1.0
labels_train = df_train['label']
sns.countplot(labels_train)


# ### More or less we may deem the classes are balanced, so it's time to reshape data. The shape is B x W x H x C where B is a size of batch, W is an image width, H is an image height, and C states for number of channels. Since we deal with gray scaled images, number of channels is 1. The number -1 for B dimensions sigifies that its value has to be inferred.

# In[ ]:


imgs_train = pixels_train.values.reshape(-1, 28, 28, 1)
imgs_test = (df_test.values).reshape(-1, 28, 28, 1)

# Let's plot an arbitrary image to see if reshaping works
plt.imshow(imgs_test[10,:,:,0])


# ### It seems like it works fine

# # 2. Data preparation

# ## a. Data augmentation

# Prior to training we will define data augmentation. It is a strategy to prevent overfiting. It relies on random transformation (like rotation, color changes, etc.) in order to force a model to generalize better. In order to use data augmentation, we gonna use PyTorch Compose method so as to concatenate several transformation we would need later. These are:
# * change a NumPy array to PIL image
# * resize an input image to given
# * random horizontal flipping  <- data augmentation
# * affine transformation       <- data augmentation
# * color jittering             <- data augmentation
# * changing to pytorch Tensor
# * and normalization of data
# 
# Note that we do not use data augmentation for validation set.
# If we have data normalization, we'd like to define also unnormalizing function to preview data. 

# In[ ]:


transform_train = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((28, 28)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(35),
                                      transforms.RandomAffine(0, shear=15, scale=(0.5,1.5)),
                                      transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))
                               ])

transform_val = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                               ])

def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
#     image *= 255.0
    return image


# ## b. Dataset definition

# Let's define a Dataset class which takes into account data augmentation for our digits...

# In[ ]:


class DigitsDataset(torch_data.Dataset):
    def __init__(self, x, y, transform = None):
        super().__init__()
        self.x = x
        self.y = y
        self.transform = transform
    def __len__(self,):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(self.x[index]), self.y[index]
        return self.x[index], self.y[index]


# ## c. Prepare train validation split

# Train- validation split is preformed to track how a model is trained and to identify if overfitting occures. The model would be overfitted when validation set accuracy starts to fall while train set accuracy is further being increased.
# 
# Let us set a size of validation split to 10%. It's widely used value. Using parameter 'stratify' we ensure that portion of classes in both train and validation split is kept (see Fig below)

# In[ ]:


train_x, val_x, train_y, val_y = train_test_split(imgs_train, labels_train, test_size = 0.1, stratify = labels_train)

# Change type to int32
train_x =  np.uint8(train_x)
val_x =  np.uint8(val_x)
train_y =  np.int64(train_y)
val_y =  np.int64(val_y)

fig = plt.figure(figsize=(25, 4))
fig.add_subplot(1,2, 1,  xticks=[], yticks=[])
sns.countplot(train_y)

fig.add_subplot(1, 2, 2,  xticks=[], yticks=[])
sns.countplot(train_y)


# ## d. Datasets initialization

# Having defined Dataset class and having obtained train-validation split, let the true datasets and corresponding DataLoader be defined. As batch_size we opted 128

# In[ ]:


train_dataset = DigitsDataset(train_x, train_y, transform_train)
val_dataset = DigitsDataset(val_x, val_y, transform_val)

training_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True)
validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 128, shuffle = False)


# Let's check if data loader work

# In[ ]:


images, labels = next(iter(validation_loader))
fig = plt.figure(figsize=(25, 4))

for idx in np.arange(16):
    ax = fig.add_subplot(2, 8, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title(labels[idx].item())


# # 3. Model learning

# ## a. CNN class definition

# Below the class representing our CNN classification network is depicted. We decided, firstly, to use convolutional layers with bigger kernel size = 5 to make a CNN to extract more abstract features. We applied also twice MaxPool layers as well as BatchNormalization. This is followed by two fully connected layers

# In[ ]:


class MNISTClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride = 2, padding = 2)
        self.conv2 = nn.Conv2d(32, 64, 5, stride = 2, padding = 2)
        self.mx1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 5, padding = 3)
        self.conv4 = nn.Conv2d(128, 256, 5, padding = 3)
        self.mx2 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, 3, stride = 2, padding = 2)
        self.conv6 = nn.Conv2d(512, 64, 3, stride = 2, padding = 4)
        self.mx3 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(2 * 2 * 64, 500)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 1024)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(1024, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.mx1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.mx2(x)
        x = self.bn1(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        x = self.mx3(x)
        x = self.bn2(x)   
        x = x.view(-1, 2 * 2 * 64)
        
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
model = MNISTClassification()


# In[ ]:


print(model)


# ## b. Loss & optimizer definition

# As a loss function, categorial cross entropy was used. To process optimization, SGD optimizer was selected. 

# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)


# ## c.GPU check

# Now, we check if GPU is available for us. If so, we gonna use it to accelerate training

# In[ ]:


print('Checking GPU...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('GPU available: {}'.format(torch.cuda.device_count()))


# In either case (GPU or CPU) we store a device to a variable 'device' and we transfer the model to it.

# In[ ]:


model = model.to(device)


# ## d. Training

#  Now, the model is about to be trained. There are four values we gonna track: accuracies of training and validation sets, and losses for training and validation sets. The model will be trained through 500 epochs.

# In[ ]:


loss_history = []
corrects_history = []
val_loss_history = []
val_corrects_history = []

EPOCHS = 900
for e in range(EPOCHS):
    print('============ EPOCH: {} =============='.format(e))
    train_loss = 0.0
    train_corrects = 0.0
    val_loss = 0.0
    val_corrects = 0.0
  
    # In each epoch create new train loader and shuffle it
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True)
    for inputs, labels in tqdm(training_loader):
        labels = labels.long() 
        inputs = inputs.to(device) # Similarily to model, we have to transfer input and labels to 'device' to make it accessible for that device
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad() # We don't want to accumulate gradients throughout epochs, hence we make it zero before calculation
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        train_loss += loss.item()
        train_corrects += torch.sum(preds == labels.data)
 
    # For validation set we won't caclulate gradients, so we use statement with torch.no_grad() to save some memory and computations
    with torch.no_grad(): 
        for val_inputs, val_labels in validation_loader:
            val_labels = val_labels.long()
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_labels)

            _, val_preds = torch.max(val_outputs, 1)
            val_loss += val_loss.item()
            val_corrects += torch.sum(val_preds == val_labels.data)
      
    epoch_loss = train_loss/len(training_loader.dataset)
    epoch_acc = train_corrects.float()/ len(training_loader.dataset)
    loss_history.append(epoch_loss)
    corrects_history.append(epoch_acc)
    
    val_epoch_loss = val_loss/len(validation_loader.dataset)
    val_epoch_acc = val_corrects.float()/ len(validation_loader.dataset)
    val_loss_history.append(val_epoch_loss)
    val_corrects_history.append(val_epoch_acc)
    print('Epoch {}:'.format(e+1))
    print('Training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
    print('Validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))


# ## e. Plot results

# After training, let us plot losses and accuracies for training and validation sets

# In[ ]:


plt.plot(loss_history, label='Training loss')
plt.plot(val_loss_history, label='Validation loss')
plt.yscale('log')
plt.legend()


# In[ ]:


plt.plot(corrects_history, label='Training accuracy')
plt.plot(val_corrects_history, label='Validation accuracy')
plt.legend()


# # 4. Prepare submission file

# ## a. Preprocess test data

# In[ ]:


test_x =  np.uint8(imgs_test)


# ## b. Make predictions

# In[ ]:


outputs = []
with torch.no_grad():
    for x in tqdm(test_x):
        x = transform_val(x).unsqueeze(0)
        x = x.to(device)
        output = model(x)
        outputs.append(torch.argmax(output).cpu().item())


# ## c. Save results to CSV

# In[ ]:


results = pd.Series(np.array(outputs, dtype = np.int32),name="Label")
submission = pd.concat([pd.Series(range(1,len(outputs)+1), dtype=np.int32, name = "ImageId"),results],axis = 1)
submission = submission.astype(np.int32)
submission.to_csv("mnist_test_preds.csv",index=False)
submission.head()


# ## d. Finally, show some test samples with predictions

# In[ ]:


idxs = np.random.randint(0, test_x.shape[0], 16)
test_samples = test_x[idxs]
predictions = np.array(outputs)[idxs]

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(16):
    ax = fig.add_subplot(2, 8, idx+1, xticks=[], yticks=[])
    ax.imshow(test_samples[idx,:,:,0])
    ax.set_title(predictions[idx].item())

