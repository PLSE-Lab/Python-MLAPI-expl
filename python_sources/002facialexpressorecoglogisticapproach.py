#!/usr/bin/env python
# coding: utf-8

# - **User**: [@manishshah120](https://www.kaggle.com/manishshah120)
# - **LinkedIn**: https://www.linkedin.com/in/manishshah120/
# - **GitHub**: https://github.com/ManishShah120
# - **Twitter**: https://twitter.com/ManishShah120
# 
# > This Notebook was created while working on project for a course "**Deep Learning with PyTorch: Zero to GANs**" from "*jovian.ml*" in collaboratoin with "*freecodecamp.org*"

# # Facial Expression Recogniton with Logistic Regression

# In[ ]:


# Imports
import os
import torch
import torchvision as tv
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as tt
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Lets Specify a preoject name to be able to commit file on jovian

# In[ ]:


project_name = '002facialexpressorecoglogisticapproach'


# ## Understanding the file structure

# In[ ]:


data_dir = '../input/facial-expression-recog-image-ver-of-fercdataset/Dataset'

print(os.listdir(data_dir))
classes = os.listdir(data_dir + '/train')
print(classes)


# Lets see the number of images belonging to each classes in the training set

# In[ ]:


anger_files = os.listdir(data_dir + '/train/anger')
print('Total no. of images for training anager class: ',len(anger_files))


# No. of training images of each class:-

# In[ ]:


for i in classes:
    var_files = os.listdir(data_dir + '/train/' + i)
    print(i,': ',len(var_files))


# ## Creating the `dataset` variable

# Lets convert all the images into tensor with the help of `ToTensor()`

# In[ ]:


dataset = ImageFolder(data_dir + '/train', transform = ToTensor())


# In[ ]:


print(dataset)


# Lets have a look to the tensors and the labels

# In[ ]:


img, label = dataset[0]
print(img.shape, label)
img


# Our image is of 48*48 pixels and has 3 channels RGB, but though our image is B/W i don't know why its showing channel as 3, instead it should have shown 1.

# Owooooooooo.....The dataset I created is working just perfectly.

# In[ ]:


print(dataset.classes)


# Lets have a look at some images.

# In[ ]:


def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))


# In[ ]:


show_example(*dataset[172])


# Yeah, Thats how an angry young man look like.

# Now it's time to create our training and validation dataset

# In[ ]:


len(dataset)


# so our training dataset contains `32298` of toal images

# In[ ]:


val_size = int(0.1*32298)
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset,[train_size, val_size])

test_ds = ImageFolder(data_dir + '/test', transform = ToTensor())


# In[ ]:


print(train_ds)
print(val_ds)
print(test_ds)


# ## Data Loaders

# In[ ]:


# Hyperparmeters
batch_size = 64

# Other constants
input_size = 3*48*48
num_classes = 7


# In[ ]:


train_loader = DataLoader(
    train_ds, 
    batch_size, 
    shuffle=True         )

val_loader = DataLoader(
    val_ds, 
    batch_size*2       )

test_loader = DataLoader(
    test_ds, 
    batch_size*2        )


# In[ ]:


show_example(*train_ds[1])


# Now, lets have a look at batch of images

# In[ ]:


for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break


# ## Model
# It's time to define our model

# In[ ]:


class FacialExprRecog(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, xb):
        xb = xb.view(xb.size(0), -1)#else xb.reshape(-1, 3*48*48)
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
    
model = FacialExprRecog()


# ## Training

# In[ ]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


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


evaluate(model, val_loader)


# Lets start the training process

# In[ ]:


history = fit(45, 0.01, model, train_loader, val_loader)


# In[ ]:


history += fit(40, 0.001, model, train_loader, val_loader)


# In[ ]:


history += fit(40, 0.0001, model, train_loader, val_loader)


# ## PLotting Functions are necessary too..

# In[ ]:


# Lets define a function for plotting graphs
def plot_accuracies(history):
    accuracies = [r['val_acc'] for r in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')

def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x', color='red')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs No. of Epoch')


# In[ ]:


plot_accuracies(history)


# In[ ]:


plot_losses(history)


# In[ ]:


result = evaluate(model, test_loader)


# In[ ]:


result


# In[ ]:


num_epochs = [45, 40, 40]
lr = [0.01, 0.001, 0.0001]


# ## Predictions
# Now comes the final part

# In[ ]:


def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()


# In[ ]:


img, label = test_ds[1322]
plt.imshow(img[0], cmap='gray')
print('Label:', test_ds.classes[label], ', Predicted:', dataset.classes[predict_image(img, model)])


# Let's just see random 10 predictions

# In[ ]:


img, label = test_ds[1392]
plt.imshow(img[0], cmap='gray')
print('Label:', test_ds.classes[label], ', Predicted:', dataset.classes[predict_image(img, model)])


# # ---------The End----------

# Comitting our work to Jovian with all the hyper params.

# In[ ]:


get_ipython().system('pip install jovian --upgrade -q')


# In[ ]:


import jovian
jovian.commit(project=project_name)


# Let's log all our hyperparams

# In[ ]:


jovian.log_dataset(dataset_url='https://www.kaggle.com/manishshah120/facial-expression-recog-image-ver-of-fercdataset', val_size=val_size)


# In[ ]:


jovian.log_metrics(
    val_loss = result['val_loss'],
    val_acc = result['val_acc']
                  )


# In[ ]:


jovian.log_hyperparams({
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'lr': lr,
})


# In[ ]:


jovian.commit(project=project_name, is_cli =True,environment=None)


# In[ ]:


Done


# In[ ]:




