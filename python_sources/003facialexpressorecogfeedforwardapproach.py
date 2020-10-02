#!/usr/bin/env python
# coding: utf-8

# - **User**: [@manishshah120](https://www.kaggle.com/manishshah120)
# - **LinkedIn**: https://www.linkedin.com/in/manishshah120/
# - **GitHub**: https://github.com/ManishShah120
# - **Twitter**: https://twitter.com/ManishShah120
# 
# > This Notebook was created while working on project for a course "**Deep Learning with PyTorch: Zero to GANs**" from "*jovian.ml*" in collaboratoin with "*freecodecamp.org*"

# # Facial Expressoin Recognition with Feed Forward Neural Network

# ## Imports

# In[ ]:


import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


project_name = '003facialexpressorecogfeedforwardapproach'


# ## Dataset

# In[ ]:


data_dir = '../input/facial-expression-recog-image-ver-of-fercdataset/Dataset'
classes = os.listdir(data_dir + '/train')


# No. of training images of each class in training set:-

# In[ ]:


for i in classes:
    var_files = os.listdir(data_dir + '/train/' + i)
    print(i,': ',len(var_files))


# No. of training images of each class in test set:-

# In[ ]:


for i in classes:
    var_files = os.listdir(data_dir + '/test/' + i)
    print(i,': ',len(var_files))


# Creating the `dataset` variable

# In[ ]:


dataset = ImageFolder(
    data_dir + '/train', 
    transform = ToTensor()
                     )


# In[ ]:


dataset


# Lets have a look to the tensors and the labels

# In[ ]:


img, label = dataset[0]
print(img.shape, label)
img


# In[ ]:


print(dataset.classes)


# In[ ]:


def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))


# In[ ]:


show_example(*dataset[9050])


# In[ ]:


val_size = int(0.1*32298) # 10% of the dataset as validatoin set
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset,[train_size, val_size])
test_ds = ImageFolder(data_dir + '/test', transform = ToTensor())


# In[ ]:


print(train_ds)
print(val_ds)
print(test_ds)


# ## Data Loader

# In[ ]:


batch_size = 64


# In[ ]:


train_loader = DataLoader(
                          train_ds, 
                          batch_size, 
                          shuffle=True, 
                          num_workers=4, 
                          pin_memory=True
                         )

val_loader = DataLoader(
                        val_ds, 
                        batch_size*2, 
                        num_workers=4, 
                        pin_memory=True
                       )


# In[ ]:


for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break


# ## Model

# In[ ]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# So to improve upon Logistic regression we will implement Neural netwrok.
# 
# And this is where neural network comes into play after this our model becomes a neural network with `no. of layer` hidden layer.

# In[ ]:


class Facial_Recog_Model(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""
    def __init__(self, in_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, 1024)
        # hidden layer 2
        self.linear2 = nn.Linear(1024, 512)
        # hidden layer 3
        self.linear3 = nn.Linear(512,256)
        # hidden Layer 4
        self.linear4 = nn.Linear(256, 128)
        # hidden Layer 5
        self.linear5 = nn.Linear(128, 64)
        # output layer
        self.linear6 = nn.Linear(64, out_size)
        
    def forward(self, xb):
        # Flatten the image tensors
        out = xb.view(xb.size(0), -1)

        # Get intermediate outputs using hidden layer
        out = self.linear1(out)
        # Apply activation function
        out = F.relu(out)

        # Get intermediate outputs using hidden layer 2
        out = self.linear2(out)
        # Apply activation function
        out = F.relu(out)

        # Get intermediate outputs using hidde layer 3
        out = self.linear3(out)
        # Apply activation function
        out = F.relu(out)

        # Get intermediate outputs using hidde layer 4
        out = self.linear4(out)
        # Apply activation function
        out = F.relu(out)
        
        # Get intermediate outputs using hidde layer 5
        out = self.linear5(out)
        # Apply activation function
        out = F.relu(out)

        # Get predictions using output layer
        out = self.linear6(out)
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


# In[ ]:


input_size = 48*48*3
num_classes = 7


# In[ ]:


model = Facial_Recog_Model(input_size, out_size = num_classes)


# In[ ]:


for images, labels in train_loader:
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    print('Loss:', loss.item())
    break

print('outputs.shape : ', outputs.shape)
print('Sample outputs :\n', outputs[:2].data)


# ## Using a GPU

# In[ ]:


torch.cuda.is_available()


# In[ ]:


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


# In[ ]:


device = get_default_device()
device


# Moving all our data and model to a choosen devise

# In[ ]:


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# In[ ]:


for images, labels in train_loader:
    print(images.shape)
    images = to_device(images, device)
    print(images.device)
    break


# In[ ]:


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


train_loader = DeviceDataLoader(
                                train_loader, 
                                device
                               )
val_loader = DeviceDataLoader(
                              val_loader, 
                              device
                             )


# In[ ]:


for xb, yb in val_loader:
    print('xb.device:', xb.device)
    print('yb:', yb)
    break


# ## Defining the training function

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


# Model (on GPU)
model = Facial_Recog_Model(input_size, out_size = num_classes)
to_device(model, device)


# In[ ]:


for t in model.parameters():
    print(t.shape)


# In[ ]:


history = [evaluate(model, val_loader)]
history


# ## Training the Model

# In[ ]:


history += fit(10, 0.05, model, train_loader, val_loader)


# In[ ]:


history += fit(15, 0.03, model, train_loader, val_loader)


# In[ ]:


history += fit(20, 0.01, model, train_loader, val_loader)


# In[ ]:


history += fit(20, 0.001, model, train_loader, val_loader)


# ## Plotting functions

# In[ ]:


def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs');
    
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x', color='red')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');


# In[ ]:


plot_losses(history)


# In[ ]:


plot_accuracies(history)


# ### Final Evaluation

# In[ ]:


evaluate(model, val_loader)


# In[ ]:


Stop here


# ## Hyper params ad other details

# In[ ]:


val_loss = history[-1]['val_loss']
val_acc = history[-1]['val_acc']
num_epochs = [10, 15, 20, 20]
lr = [0.05, 0.03, 0.01, 0.001]
arch = "7 layers (1024, 512, 256, 128, 64, 7)"


# ## Commiting to Jovian

# In[ ]:


get_ipython().system('pip install jovian --upgrade -q')


# In[ ]:


import jovian


# In[ ]:


#jovian.commit(project=project_name, environment = None)


# In[ ]:


jovian.log_dataset(dataset_url='https://www.kaggle.com/manishshah120/facial-expression-recog-image-ver-of-fercdataset', val_size=val_size, train_size = train_size)


# In[ ]:


jovian.log_hyperparams({
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'lr': lr,
    'arch': arch
})


# In[ ]:


jovian.log_metrics(val_loss=val_loss, val_acc=val_acc)


# In[ ]:


jovian.commit(project = project_name, environment=None)


# In[ ]:




