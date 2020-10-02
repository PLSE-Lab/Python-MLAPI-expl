#!/usr/bin/env python
# coding: utf-8

# # MNIST - Looking for a nice FCNN #
# We are going to look for a neural network using only fully connected layers.

# In[ ]:


# Some variables that change the values when commited
# Before commiting this variable must be setted to True
COMMIT = False

if COMMIT:
    # show some information during execution
    verbose = 1
    # use a nice number of epochs to train models when commiting
    training_epochs = 30
else:
    # no commiting (so direct work on jupyter notebook)
    # show more information during execution
    verbose = 2
    # decrease the number of epochs when we are in front of t
    training_epochs = 10
    
# Global batch_size that will be used all over this jupyter notebook
batch_size = 100


# # Import libraries #

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import time
import seaborn as sn

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Load Training and Test Data #

# In[ ]:


train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train.head()


# ## Prepare training data for training and testing ##

# In[ ]:


# Original training labels
Y_train = train["label"].values
# Original training images (784 pixels values per row)
X_train = train.drop(labels = ["label"], axis = 1).values
# Original testing images (784 pixels values per row)
X_test = test.values

# Split original training data for cross validation
test_sample = 0.2
split_idx = int((1 - test_sample) * len(Y_train))
x_train, x_test = X_train[0:split_idx], X_train[split_idx:]
y_train, y_test = Y_train[0:split_idx], Y_train[split_idx:]


# In[ ]:


print(x_train.shape)
print(x_test.shape)


# In[ ]:


# training data are row images of 784 pixel integer values in range 0-255
print(sorted(pd.unique(x_train.flatten())))


# In[ ]:


# labels are integer values in range 0-9
print(sorted(pd.unique(y_train.flatten())))


# ## 1D to 2D ##
# 
# We want to use torchvision.transforms on the data because it will help in the future to do data augmentation.
# This will help when working with CNN too!

# In[ ]:


# torchvision.transforms expect data in PIL format (2D arrays with values in range 0-255)
x_train = x_train.astype('uint8').reshape(-1, 28, 28)
x_test = x_test.astype('uint8').reshape(-1, 28, 28)
X_test = X_test.astype('uint8').reshape(-1, 28, 28)

print(x_train.shape)
print(x_test.shape)
print(X_test.shape)


# ## CustomDataset class ##
# 
# We create a class that will be useful to transform original data. This will allow us to do data augmentation. A CustomDataset instance receive the original training data and then it transforms the data when requested.
# 

# In[ ]:


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        # data: tuple (X, Y)
        if len(data) not in [1,2]:
            raise "Not expected data shape"
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):

        x = self.data[0][idx]
        
        if len(self.data) == 2:
            y = self.data[1][idx]
        else:
            y = torch.zeros(len(x))


        if self.transform:
            x = self.transform(x)
        
        x = x.double()

        return x, y


# ## Dataset creator ##

# In[ ]:


def create_sets(augmented=False, degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.0)):
    # Rotate, translate and scale original data
    # There is only one line that makes the difference (RandomAffine)
    # If we comment this line we will eliminate data augmentation
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=degrees, translate=translate, scale=scale),
        transforms.ToTensor()
    ])

    # Augmentation must not be used on test images
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    
    if augmented:
        train_set = CustomDataset(data=(x_train, y_train), transform=transform_train)
    else:
        # transform_test does not transform the original training data
        train_set = CustomDataset(data=(x_train, y_train), transform=transform_test)        
    test_set =  CustomDataset(data=(x_test, y_test), transform=transform_test)
    
    return train_set, test_set


# # Visualize #
# ## Augmented Data ##

# In[ ]:


train_set, test_set = create_sets(augmented=True)

W, H = 10, 10

fig, axes = plt.subplots(H, W, figsize=(10,10))
# show the first H different images in train set
for h in range(H):
    # show W times the same image transformed in and by CustomDataset
    for w in range(W):
        img, label = train_set[h]
        axes[h,w].imshow(img.numpy().squeeze(), cmap='gray')
        axes[h,w].axis('off')
plt.show()


# ## First 72 transformed images ##

# In[ ]:


W, H = 12, 6
fig, axes = plt.subplots(H, W, figsize=(16,8))
for h in range(H):
    for w in range(W):
        img, label = train_set[h*H+w]
        axes[h,w].imshow(img.numpy().squeeze(), cmap='gray')
        axes[h,w].axis('off')
        
plt.show()


# # Create dataset loaders #
# First we are going to work with no augmentation

# In[ ]:


train_set, test_set = create_sets(augmented=False)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


# # Fully Connected Neural Network #
# This class is used to create FCNN with differents configurations.

# In[ ]:


class FCN(nn.Module):
    def __init__(self, layers = 1, hidden_units = 200, name=''):
        super(FCN, self).__init__()
        if layers not in range(1,10):
            raise "Invalid number of layers"
            
        self.name = name
        
        self.layers = layers
        if layers == 1:
            self.fc1 = nn.Linear(in_features=784, out_features=10, bias=True)
        else:
            self.fc1 = nn.Linear(in_features=784, out_features=hidden_units, bias=True)
            
        for i in range(layers-2):
            self.add_module('fc_hidden'+str(i+2), nn.Linear(in_features=hidden_units, out_features=hidden_units, bias=True))
        
        if layers > 1:
            self.add_module('fc_last'+str(layers), nn.Linear(in_features=hidden_units, out_features=10, bias=True))            

    def forward(self, x):
        x = self.fc1(x)
        if self.layers > 1:
            x = F.relu(x)
        for name, fc in self.named_modules():
            if name.startswith('fch'):
                x = fc(x)
                x = F.relu(x)
            elif name.startswith('fc_last'):
                x = fc(x)

        x = F.softmax(x, dim=1)
        return x


# ## Use GPU if possible ##

# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available(): print('Thanks Kaggle! Guau!!!')
print(device)


# ## Training Procedure ##
# This method will be used to train a lot of models.  
# (It is already prepared to train CNN too (I hope))

# In[ ]:


def train(model, criterion, optimizer, scheduler, train_loader, test_loader, CNN=False, device='cpu', epochs=2, verbose=1):

    start = time.time()

    model.to(device)
    history = {}
    history['train_losses'] = []
    history['test_losses'] = []
    history['acc'] = []
    history['best_model'] = None
    history['best_epoch'] = -1
    history['best_acc'] = -1
    
    for epoch in range(epochs):
        model.train()
        train_losses = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Load images
            if CNN:
                images = images.view(-1, 28, 28)
            else:
                images = images.view(-1, 784)
                
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_losses += loss.item()
            loss.backward()
            optimizer.step()
        
        else:

            with torch.no_grad(): #Turning off gradients to speed up
                model.eval()
                correct = 0.0
                total = 0.0
                test_losses = 0.0
                for images, labels in test_loader:
                    if CNN:
                        images = images.view(-1, 28, 28)
                    else:
                        images = images.view(-1, 784)

                    images, labels = images.to(device), labels.to(device)
            
                    outputs = model(images)
                
                    loss = criterion(outputs, labels)
                    test_losses += loss.item()

                    _, predicted = torch.max(outputs.data, 1)

                    total += labels.size(0)
                    # Total correct predictions
                    correct += (predicted == labels).sum()

            accuracy = float(correct) / total
            train_losses = train_losses/len(train_loader)
            test_losses = test_losses/len(test_loader)
            if verbose > 1:
                # Print Loss
                print('Epoch: {}/{}. LR: {}. Train Loss: {:.4f}. Test Loss: {:.4f}. Accuracy: {:.4f}'.format(
                    epoch+1, epochs, model.scheduler.get_lr(), train_losses, test_losses, accuracy))
                        
            history['train_losses'].append(train_losses)
            history['test_losses'].append(test_losses)
            history['acc'].append(accuracy)

            if accuracy > history['best_acc']:
                history['best_acc'] = accuracy
                history['best_model'] = copy.copy(model)
                history['best_epoch'] = epoch
            
        scheduler.step()
    
    total_time = time.time() - start
    
    if verbose > 0:
        print('Best model found at epoch {} with accuracy {}'.format(
                history['best_epoch'],
                history['best_acc']
        ))
        n_params = sum(p.numel() for p in model.parameters())
        print('Training time required: {:.2f} secs'.format(total_time))
        print('Training time required per epoch: {:0.2f} secs'.format(total_time/epochs))
        print('Training time per model parameter: {:0.2f} msecs'.format(total_time*1000/n_params))
    
    return history['best_model'], history


# # First models to try #
# Generate the first 15 models with different combinations of layers and hidden unints

# In[ ]:


# Learning Rate
learning_rate = 0.003
# Every 'lr_step_size' the learning_rate will be modified to 'lr_decrement' proportion
lr_step_size, lr_decrement = 10, 0.5

# Test performance for different number of layers and hidden_units
layers = [1,2,3,4,5]
hidden_units = [16,32,64]

# Create the models
models = []
for l in layers:
    for hu in hidden_units:
        name = 'FCN{}_{}'.format(l, hu)
        model = FCN(layers=l, hidden_units=hu, name=name).double()
        # use the model as some kind of toolsbox
        model.criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
        model.optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        model.scheduler = optim.lr_scheduler.StepLR(model.optimizer, step_size=lr_step_size, gamma=lr_decrement)
        models.append(model)


# In[ ]:


def show_params(models):
    for model in models:  
        print('{} has {} parameters.'.format(model.name, sum(p.numel() for p in model.parameters())))

# Show the number of trainable parameters of every model
show_params(models)


# In[ ]:


# This method will be helpful to train many models
def train_all(models, epochs=10, device=device, verbose=2):
    history = {}
    n = len(models)
    for i, model in enumerate(models):
        print('Training model {}/{}: {} ...'.format(i+1, n, model.name))
        best_model, h = train(model, criterion=model.criterion, optimizer=model.optimizer,
                              scheduler=model.scheduler, train_loader=train_loader,
                              test_loader=test_loader, CNN=False,
                              epochs=epochs, device=device, verbose=verbose)
        history[best_model.name] = h
    return history


# In[ ]:


# plot every model training and testing losses
def show_losses(history, COLS=None):
    n = len(history)
    if COLS is None:
        COLS = 2
        if n >= 9: COLS = 3
    ROWS = int(np.ceil(n/COLS))
    fig, axes = plt.subplots(ROWS, COLS, figsize=(16, ROWS*4))
    i = 1
    for name, h in history.items():
        ax = plt.subplot(ROWS, COLS, i)        
        ax.set_title(name + ' losses')
        ax.plot(h['train_losses'], label='train')
        ax.plot(h['test_losses'], label='test')
        ax.legend()
        i+=1
    plt.show()

# show all models accuracies in one plot
def show_accuracy(history):
    fig = plt.figure(figsize=(16,8))
    plt.title('Accuracy')
    accs = [(name, h['best_acc']) for name, h in history.items()]
    accs = sorted(accs, key=lambda item: item[1], reverse=True)
    for name, acc in accs:
        print('{}: {:0.3f}'.format(name, acc))
    for name, h in history.items():
        plt.plot(h['acc'], label=name)

    plt.legend()    
    plt.show()


# ## First training ##

# In[ ]:


history = train_all(models, epochs=training_epochs, device=device, verbose=verbose)


# In[ ]:


show_losses(history, COLS=3)


# In[ ]:


show_accuracy(history)


# # Try: 2 #
# 3, 4 and 5 layers are the best.
# 64 hidden units per layers works well. We can test more hidden units.

# In[ ]:


# Test performance for different number of layers (and hidden_units)
layers = [3, 4, 5]
hidden_units = [64, 128, 254]

models = []
for l in layers:
    for hu in hidden_units:
        name = 'FCN{}_{}'.format(l, hu)
        model = FCN(layers=l, hidden_units=hu, name=name).double()
        # use the model as some kind of box (for tools)
        model.criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
        model.optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        model.scheduler = optim.lr_scheduler.StepLR(model.optimizer, step_size=lr_step_size, gamma=lr_decrement)
        models.append(model)


# In[ ]:


show_params(models)


# In[ ]:


history_2 = train_all(models, epochs=training_epochs, device=device, verbose=verbose)


# In[ ]:


show_losses(history_2, COLS=3)


# In[ ]:


show_accuracy(history_2)


# # Try: 3 #

# In[ ]:


# Test performance for different number of layers (and hidden_units)
layers = [3, 4]
hidden_units = [254, 512, 1024]

models = []
for l in layers:
    for hu in hidden_units:
        name = 'FCN{}_{}'.format(l, hu)
        model = FCN(layers=l, hidden_units=hu, name=name).double()
        # use the model as some kind of box (for tools)
        model.criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
        model.optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        model.scheduler = optim.lr_scheduler.StepLR(model.optimizer, step_size=lr_step_size, gamma=lr_decrement)
        models.append(model)


# In[ ]:


show_params(models)


# In[ ]:


history_3 = train_all(models, epochs=training_epochs, device=device, verbose=verbose)


# In[ ]:


show_losses(history_3, COLS=3)


# In[ ]:


show_accuracy(history_3)


# # Try: 4 #

# In[ ]:


# 1024 hidden units usually is the best, but it requires too many parameters.
# fcnn with 3 and 4 layers looks good as well as 254 and 512 hidden units
# Lets use the one that require less number of parameters to be trained
# Test performance for different leating rates
layers = 3
hidden_units = 254
learning_rates = [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]

models = []
for lr in learning_rates:
    name = 'FCN{}_{}_lr_{}'.format(layers, hidden_units, lr)
    model = FCN(layers=layers, hidden_units=hidden_units, name=name).double()
    # use the model as some kind of box (for tools)
    model.criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    model.optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # set a big step size to avoid decreasing it (I want to check LR performance)
    model.scheduler = optim.lr_scheduler.StepLR(model.optimizer, step_size=100, gamma=lr_decrement)
    models.append(model)


# In[ ]:


show_params(models)


# In[ ]:


history_4 = train_all(models, epochs=training_epochs, device=device, verbose=verbose)


# In[ ]:


show_losses(history_4, COLS=2)


# In[ ]:


show_accuracy(history_4)


# # Try: 5 #

# In[ ]:


# learning rates 0.001 and 0.003 works well.
# Now lets see if a StepLR scheduler can improve results for learning rate 0.003
layers = 3
hidden_units = 254
lr = 0.003
step_sizes = [7, 10, 13]
gammas = [0.1, 0.3, 0.5]
models = []
for step_size in step_sizes:
    for gamma in gammas:
        name = 'FCN{}_{}_step_size_{}_gamma_{}'.format(layers, hidden_units, step_size, gamma)
        model = FCN(layers=layers, hidden_units=hidden_units, name=name).double()
        # use the model as some kind of box (for tools)
        model.criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
        model.optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        model.scheduler = optim.lr_scheduler.StepLR(model.optimizer, step_size=step_size, gamma=gamma)
        models.append(model)        


# In[ ]:


show_params(models)


# In[ ]:


history_5 = train_all(models, epochs=training_epochs, device=device, verbose=verbose)


# In[ ]:


show_losses(history_5, COLS=3)


# In[ ]:


show_accuracy(history_5)


# # Try: 6 #
# Build 10 identical models to ensemble our network

# In[ ]:


# StepLR scheduler works well with step_size 13 and gamma 0.3
# Lets see how it does work using augmented data
# To create an ensembled model
layers = 3
hidden_units = 254
lr = 0.003
step_size = 13
gamma = 0.3

NETS = 10
models = []

for i in range(NETS):
    name = 'FCN{}_{}_{}_{}_data'.format(layers, hidden_units, i, 'augmented')
    model = FCN(layers=layers, hidden_units=hidden_units, name=name).double()
    # use the model as some kind of box (for tools)
    model.criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    model.optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    model.scheduler = optim.lr_scheduler.StepLR(model.optimizer, step_size=step_size, gamma=gamma)
    models.append(model)

### AUGMENTED DATA ###
train_set, test_set = create_sets(augmented=True)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)


# In[ ]:


show_params(models)


# In[ ]:


history_6 = train_all(models, epochs=training_epochs, device=device, verbose=verbose)


# In[ ]:


show_losses(history_6, COLS=2)


# In[ ]:


show_accuracy(history_6)


# # Confusion matrix #
# Show the confusion matrix of the ensembled model

# In[ ]:


CNN = False
results = []
confusion_matrix = torch.zeros(10, 10)
with torch.no_grad():
    for images, labels in test_loader:
        if CNN:
            images = images.view(-1, 28, 28)
        else:
            images = images.view(-1, 784)

        images, labels = images.to(device), labels.to(device)
        r = torch.zeros((len(labels), 10)).double().to(device)
        # sum 10 predictions
        for model in models:
            model.to(device)
            outputs = model(images)
            r += outputs
        
        # select top 1 prediction ()
        _, preds = torch.max(r, 1)
        # save prediction
        results.append(preds)
        
        for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    results = np.array(results).flatten()

print('Confusion Matrix')
print(confusion_matrix)
print('Accuracy per class')
print(confusion_matrix.diag()/confusion_matrix.sum(1))


# In[ ]:


fig = plt.figure(figsize=(10,8))
#plt.imshow(confusion_matrix, cmap='binary');
sn.heatmap(confusion_matrix, annot=True);


# # Test Prediction #

# ## Prepare test data ##

# In[ ]:


# prepare original test to make predictions
no_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

testset =  CustomDataset(data=(X_test, np.zeros(len(test))), transform=no_transform)
testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)


# ## Make prediction ##

# In[ ]:


CNN = False
results = []

with torch.no_grad():
    for images, labels in testloader:
        if CNN:
            images = images.view(-1, 28, 28)
        else:
            images = images.view(-1, 784)

        images, labels = images.to(device), labels.to(device)
        r = torch.zeros((len(labels), 10)).double().to(device)
        for model in models:
            model.to(device)
            outputs = model(images)
            r += outputs
        results.append(torch.argmax(r, 1).to('cpu').numpy())

    results = np.array(results).flatten()


# ## Submit prediction ##

# In[ ]:


results = [int(i) for i in results]
results = pd.Series(results, name="Label")
submission = pd.concat([pd.Series(range(1, len(results)+1), name = "ImageId"), results], axis = 1)
submission.to_csv("submission.csv", index=False)

