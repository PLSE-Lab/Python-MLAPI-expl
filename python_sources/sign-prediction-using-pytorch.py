#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **Image classification with sign langauage MNIST using pytorch**
# This is a part of course project conducted by jovian.ml with freecodecamp. In this project,I have used sign langauage MNIST dataset to predict sign language images using diffrent modals like loginstic regression, feed forword nn, convolution nn.

# In[ ]:


project_name = 'final-project-jovain.ml'


# # sign language MNIST dataset
# 

# # Downloading and exploring the data 

# first, I will import some libraries that i will throughout this project

# In[ ]:


get_ipython().system(' pip install jovian --upgrade -q')


# In[ ]:


import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from PIL import Image
import pandas as pd

from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import make_grid

import jovian


# In[ ]:


data_dir = "../input/sign-language-mnist/"


# In[ ]:


test_df = pd.read_csv(data_dir+'sign_mnist_test/sign_mnist_test.csv')
train_df = pd.read_csv(data_dir+'sign_mnist_train/sign_mnist_train.csv')


# Now, I have created a helper function to convert all dataframes into numpy array

# In[ ]:


def dataframe_to_nparray(train_df, test_df):
    train_df1 = train_df.copy(deep = True)
    test_df1 = test_df.copy(deep = True)
    train_images = train_df1.iloc[:, 1:].to_numpy(dtype = 'float32')
    test_images = test_df1.iloc[:, 1:].to_numpy(dtype = 'float32')
    return train_images,test_images


# In[ ]:


train_img, test_img = dataframe_to_nparray(train_df, test_df)
train_labels = train_df['label'].values
test_labels = test_df['label'].values


# In[ ]:


train_img.size


# In[ ]:


train_images_shaped = train_img.reshape(train_img.shape[0],1,28,28)
test_images_shaped = test_img.reshape(test_img.shape[0],1,28,28)


# Next step is to convert all numpy arrays into pytorch tensors

# In[ ]:


train_images_tensors = torch.from_numpy(train_images_shaped)
train_labels_tensors = torch.from_numpy(train_labels)

test_images_tensors = torch.from_numpy(test_images_shaped)
test_labels_tensors = torch.from_numpy(test_labels)


# In[ ]:


# pytorch dataset
train_ds_full = TensorDataset(train_images_tensors, train_labels_tensors) #this dataset will further devided into validation dataset and training dataset
test_ds = TensorDataset(test_images_tensors, test_labels_tensors)


# We can see that we converted each image in a 3-dimensions tensor (1, 28, 28). The first dimension is for the number of channels. The second and third dimensions are for the size of the image, in this case, 28px by 28px.

# In[ ]:


img, label = train_ds_full[0]
print(img.shape, label)
img


# Now we will define hyperparameters for our modal

# In[ ]:


# Hyperparmeters
batch_size = 64
learning_rate = 0.001

# Other constants
in_channels = 1
input_size = in_channels * 28 * 28
num_classes = 26


# # Training and validation dataset 
# Now we are going to use three datasets-
# <ol>
# <li>Training set - used to train the model (compute the loss and adjust the weights of the model using gradient descent).</li>
# <li>Validation set - used to evaluate the model while training, adjust hyperparameters (learning rate etc.) and pick the best version of the model.</li>
# <li>Test set - used to compare different models, or different types of modeling approaches, and report the final accuracy of the model.</li>
#     </ol>

# In[ ]:


random_seed = 11
torch.manual_seed(random_seed);


# In[ ]:


val_size = 7455
train_size = len(train_ds_full) - val_size

train_ds, val_ds = random_split(train_ds_full, [train_size, val_size,])
len(train_ds), len(val_ds), len(test_ds)


# Now we will load the training,validation and test dataset in batches 

# In[ ]:


train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size*2, num_workers=4, pin_memory=True)


# In[ ]:


for img, label in train_dl:
    print(img.size())
    break


# # Models for image classification
# We are going to create three different models for this project:
# 
# 1. Logistic Regression
# 1. Deep Neural Network
# 1. Convolutional Neural Network
# 

# # Logistic regression

# In[ ]:


class ASLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, in_channels*28*28)
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    
model = ASLModel()


# In[ ]:


for images, labels in test_dl:
    outputs = model(images)
    print(labels)
    print(accuracy(outputs, labels))
    
    break

print('outputs.shape : ', outputs.shape)
print('Sample outputs :\n', outputs[:2].data)


# In[ ]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


# In[ ]:


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# In[ ]:


result0 = evaluate(model, val_dl)
result0


# The initial accuracy is around 4%, which is what one might expect from a randomly intialized model (since it has a 1 in 10 chance of getting a label right by guessing randomly). Also note that we are using the .format method with the message string to print only the first four digits after the decimal point.
# 
# We are now ready to train the model. Let's train for 5 epochs and look at the results.

# In[ ]:


history1 = fit(10, 0.001, model, train_dl, val_dl)


# In[ ]:


history2 = fit(10, 0.0001, model, train_dl, val_dl)


# In[ ]:


history3 = fit(10, 0.00001, model, train_dl, val_dl)


# In[ ]:


history4 = fit(10, 0.000001, model, train_dl, val_dl)


# Now with 40 iteration we went from 4% acc to 94% accuracy.It's quite amazing

# In[ ]:


history = [result0] + history1 + history2 + history3 + history4
accuracies = [result['val_acc'] for result in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');


# In[ ]:


history = [result0] + history1 + history2 + history3 + history4
accuracies = [result['val_loss'] for result in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');


# In[ ]:


# evaluate on test dataset
result = evaluate(model, test_dl)
result


# # Prediction

# In[ ]:


def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()


# In[ ]:


img, label = test_ds[10]
plt.imshow(img.view(28,28), cmap='gray')
print('Label:', label.item(), ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = test_ds[200]
plt.imshow(img.view(28,28), cmap='gray')
print('Label:', label.item(), ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = test_ds[1000]
plt.imshow(img.view(28,28), cmap='gray')
print('Label:', label.item(), ', Predicted:', predict_image(img, model))


# # saving the modal

# In[ ]:


torch.save(model.state_dict(), 'ASL-logistic.pth')


# # Commit and upload the notebook
# As a final step, we can save and commit our work using the jovian library. Along with the notebook, we can also attach the weights of our trained model, so that we can use it later.

# In[ ]:


jovian.commit(project= project_name, enviornment= None)


# # Deep Neural Network

# ## Definging the model

# In[ ]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# In[ ]:


class ASLModel2(nn.Module):
    """Feedfoward neural network with 2 hidden layer"""
    def __init__(self, in_size, out_size):
        super().__init__()
        # hidden layer 1
        self.linear1 = nn.Linear(in_size, 512)
        # hidden layer 2
        self.linear2 = nn.Linear(512, 256)
        # hidden layer 3
        self.linear3 = nn.Linear(256, 128)
        # output layer  
        self.linear4 = nn.Linear(128, out_size)
        
    def forward(self, xb):
        # Flatten the image tensors
        out = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer 1
        out = self.linear1(out)
        # Apply activation function
        out = F.relu(out)
        # Get intermediate outputs using hidden layer 2
        out = self.linear2(out)
        # Apply activation function
        out = F.relu(out)
        # Get inermediate outputs using hidden layer 3
        out = self.linear3(out)
        # Apply a activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear4(out)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


# # Using a GPU 
# To work with GPU's we have to take help of some utility functions, so let's define couple utility functions 

# In[ ]:


torch.cuda.is_available()


# In[ ]:


def get_default_device():
    if torch.cuda.is_available() == True:
        return torch.device('cuda')
    else: 
        return torch.device('cpu')


# In[ ]:


device = get_default_device()
device


# In[ ]:


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[ ]:


train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
test_dl = DeviceDataLoader(test_dl, device)


# In[ ]:


print(train_dl.device)
print(test_dl.device)
print(val_dl.device)


# # Training the Modal

# In[ ]:


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# In[ ]:


input_size, num_classes


# In[ ]:


model = ASLModel2(input_size, out_size = num_classes)


# In[ ]:


# for loading our model into GPU
model = to_device(model, device)


# In[ ]:


model


# In[ ]:


history = [evaluate(model, val_dl)]
history


# so initially, this modal has very small accuracy of almost 3% that is vary low.
# so to improve this, we will iterate the process upto some epochs

# In[ ]:


history += fit(10, .001, model, train_dl, val_dl)


# In[ ]:


history


# In[ ]:


losses = [x['val_loss'] for x in history]
plt.plot(losses)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('epoch vs loss')


# In[ ]:


acc = [x['val_acc'] for x in history]
plt.plot(acc)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('epoch vs accuracy')


# In[ ]:


result = evaluate(model, test_dl)


# In[ ]:


result


# # predictions

# In[ ]:


def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()


# In[ ]:


img, label = test_ds[229]
plt.imshow(img.view(28,28), cmap='gray')
print('Label:', label.item(), ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = test_ds[6767]
plt.imshow(img.view(28,28), cmap='gray')
print('Label:', label.item(), ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = test_ds[7171]
plt.imshow(img.view(28,28), cmap='gray')
print('Label:', label.item(), ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = test_ds[6762]
plt.imshow(img.view(28,28), cmap='gray')
print('Label:', label.item(), ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = test_ds[55]
plt.imshow(img.view(28,28), cmap='gray')
print('Label:', label.item(), ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = test_ds[6766]
plt.imshow(img.view(28,28), cmap='gray')
print('Label:', label.item(), ', Predicted:', predict_image(img, model))


# # saving the Model

# In[ ]:


torch.save(model.state_dict(), 'ASL-dnn.pth')


# # commiting the notebook 

# In[ ]:


jovian.commit(project=project_name, enviornment=True)


# In[ ]:




