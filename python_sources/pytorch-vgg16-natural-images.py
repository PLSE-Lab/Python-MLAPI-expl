#!/usr/bin/env python
# coding: utf-8

# In this notebook, I'll be training a model on the natural image dataset available on Kaggle using transfer learning techniques to extract features from a pre-trained model to achieve high accuracy classification of this dataset. This is my first kernel on Kaggle, so any feedback on areas of improvement and better practices that I should adopt is definitely welcome.

# In[ ]:


from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler, random_split
import torch.nn as nn

from PIL import Image
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ![](http://)We'll first import the data from the relevant path and do some simple visualisations

# In[ ]:


print(os.listdir('../input/data/natural_images'))


# In[ ]:


#Visualising Data
classes = []
img_classes = []
n_image = []
height = []
width = []
dim = []


# Using folder names to identify classes
for folder in os.listdir('../input/data/natural_images'):
    classes.append(folder)
    
    # Number of each image
    images = os.listdir('../input/data/natural_images/'+folder)
    n_image.append(len(images))
      
    for i in images:
        img_classes.append(folder)
        img = np.array(Image.open('../input/data/natural_images/'+folder+'/'+i))
        height.append(img.shape[0])
        width.append(img.shape[1])
    dim.append(img.shape[2])
    
df = pd.DataFrame({
    'classes': classes,
    'number': n_image,
    "dim": dim
})
print("Random heights:" + str(height[10]), str(height[123]))
print("Random Widths:" + str(width[10]), str(width[123]))
df


# In[ ]:


image_df = pd.DataFrame({
    "classes": img_classes,
    "height": height,
    "width": width
})
img_df = image_df.groupby("classes").describe()
print(img_df)


# From Above, we can see the number of images each class contains, and that the height and width of images are not of a standard size. The only constant is the dimension, 3, representing that all are RGB colored images.
# 
# When using images in the pre-trained network, we'll have to reshape them to 224 x 224. This is the size of Imagenet images and is therefore what the model expects. The images that are larger than this will be truncated while the smaller images will be interpolated. Data Augmentation and Resizing will be used later to modify the images

# In[ ]:


#Display random image
Image.open('../input/data/natural_images/cat/cat_0007.jpg')


# We'll define the transformations that will be done for the 3 different sets of data (training, validation, test - which will be split later as well). Only the training data will be augmented.

# In[ ]:


image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.95, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    'val':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# Visualising and demonstrating how transformations will look like by writing a function below to plot examples of how images can be randomly transformed
# 

# In[ ]:


def imshow_tensor(image, ax=None, title=None):

    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')

    return ax, image


# In[ ]:


img = Image.open('../input/data/natural_images/flower/flower_0124.jpg')
img


# In[ ]:


transform = image_transforms['train']
plt.figure(figsize=(24, 24))

for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    _ = imshow_tensor(transform(img), ax=ax)

plt.tight_layout()


# Data is now split into training, validation and test sets and then loaded into the data loaders

# In[ ]:


batch_size = 128

all_data = datasets.ImageFolder(root='../input/data/natural_images')
train_data_len = int(len(all_data)*0.8)
valid_data_len = int((len(all_data) - train_data_len)/2)
test_data_len = int(len(all_data) - train_data_len - valid_data_len)
train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
train_data.dataset.transform = image_transforms['train']
val_data.dataset.transform = image_transforms['val']
test_data.dataset.transform = image_transforms['test']
print(len(train_data), len(val_data), len(test_data))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# In[ ]:


trainiter = iter(train_loader)
features, labels = next(trainiter)
print(features.shape, labels.shape)


# Now that the data is prepared, We'll start defining the model used, which in this case is a pre-trained model

# In[ ]:


model = models.vgg16(pretrained=True)
model


# In[ ]:


# Freeze early layers
for param in model.parameters():
    param.requires_grad = False


# After freezing the pre-trained layers of the network (which will take extremely long if we were to retrain it due to the sheer amount of layers), we now have to define the classifier layer which we will train to suit our dataset and use case

# In[ ]:


n_classes = 8
n_inputs = model.classifier[6].in_features
# n_inputs will be 4096 for this case
# Add on classifier
model.classifier[6] = nn.Sequential(
    nn.Linear(n_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, n_classes),
    nn.LogSoftmax(dim=1))

model.classifier


# In[ ]:


# Show the summary of our model and the training params
# Can use the below code if torchsummary is available (which it is not on Kaggle)
#summary(model, input_size=(3, 224, 224), batch_size=batch_size, device='cuda')


# The loss function and optimizer is defined below and the model will use GPU to train

# In[ ]:


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.cuda()


# In[ ]:


model.class_to_idx = all_data.class_to_idx
model.idx_to_class = {
    idx: class_
    for class_, idx in model.class_to_idx.items()
}

list(model.idx_to_class.items())


# In[ ]:


def train(model,
         criterion,
         optimizer,
         train_loader,
         val_loader,
         save_location,
         early_stop=3,
         n_epochs=20,
         print_every=2):
   
#Initializing some variables
  valid_loss_min = np.Inf
  stop_count = 0
  valid_max_acc = 0
  history = []
  model.epochs = 0
  
  #Loop starts here
  for epoch in range(n_epochs):
    
    train_loss = 0
    valid_loss = 0
    
    train_acc = 0
    valid_acc = 0
    
    model.train()
    ii = 0
    
    for data, label in train_loader:
      ii += 1
      data, label = data.cuda(), label.cuda()
      optimizer.zero_grad()
      output = model(data)
      
      loss = criterion(output, label)
      loss.backward()
      optimizer.step()
      
      # Track train loss by multiplying average loss by number of examples in batch
      train_loss += loss.item() * data.size(0)
      
      # Calculate accuracy by finding max log probability
      _, pred = torch.max(output, dim=1) # first output gives the max value in the row(not what we want), second output gives index of the highest val
      correct_tensor = pred.eq(label.data.view_as(pred)) # using the index of the predicted outcome above, torch.eq() will check prediction index against label index to see if prediction is correct(returns 1 if correct, 0 if not)
      accuracy = torch.mean(correct_tensor.type(torch.FloatTensor)) #tensor must be float to calc average
      train_acc += accuracy.item() * data.size(0)
      if ii%15 == 0:
        print(f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete.')
      
    model.epochs += 1
    with torch.no_grad():
      model.eval()
      
      for data, label in val_loader:
        data, label = data.cuda(), label.cuda()
        
        output = model(data)
        loss = criterion(output, label)
        valid_loss += loss.item() * data.size(0)
        
        _, pred = torch.max(output, dim=1)
        correct_tensor = pred.eq(label.data.view_as(pred))
        accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
        valid_acc += accuracy.item() * data.size(0)
        
      train_loss = train_loss / len(train_loader.dataset)
      valid_loss = valid_loss / len(val_loader.dataset)
      
      train_acc = train_acc / len(train_loader.dataset)
      valid_acc = valid_acc / len(val_loader.dataset)
      
      history.append([train_loss, valid_loss, train_acc, valid_acc])
      
      if (epoch + 1) % print_every == 0:
        print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}')
        print(f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%')
        
      if valid_loss < valid_loss_min:
        torch.save(model.state_dict(), save_location)
        stop_count = 0
        valid_loss_min = valid_loss
        valid_best_acc = valid_acc
        best_epoch = epoch
        
      else:
        stop_count += 1
        
        # Below is the case where we handle the early stop case
        if stop_count >= early_stop:
          print(f'\nEarly Stopping Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
          model.load_state_dict(torch.load(save_location))
          model.optimizer = optimizer
          history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc','valid_acc'])
          return model, history
        
  model.optimizer = optimizer
  print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
  
  history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
  return model, history


# In[ ]:


model, history = train(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    save_location='./natural_images.pt',
    early_stop=5,
    n_epochs=30,
    print_every=2)


# In[ ]:


history


# After training over a number of epochs, we can see that the model is able to pick up the features of the images very quickly and classify them accordingly, therefore the accuracy is also so high for the first few epochs. This is because the model was trained on ImageNet, which contained millions of data and over 1,000 classes, therefore the layers (which we froze) were able to pick out the distinguishing features of the images, making it straightforward for the classifier which we were training to predict accurately

# In[ ]:


def accuracy(model, test_loader, criterion):
  with torch.no_grad():
    model.eval()
    test_acc = 0
    for data, label in test_loader:
      data, label = data.cuda(), label.cuda()
      
      output = model(data)
      
      _, pred = torch.max(output, dim=1)
      correct_tensor = pred.eq(label.data.view_as(pred))
      accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
      test_acc += accuracy.item() * data.size(0)
      
    test_acc = test_acc / len(test_loader.dataset)
    return test_acc


# I've defined the function for testing the model on the last set of data that we prepared earlier, and the model has never seen this set of data before.

# In[ ]:


model.load_state_dict(torch.load('./natural_images.pt'))
test_acc = accuracy(model.cuda(), test_loader, criterion)
print(f'The model has achieved an accuracy of {100 * test_acc:.2f}% on the test dataset')


# To see which class the model made the mistake on, the function below picks it out individually

# In[ ]:


def evaluate(model, test_loader, criterion):
  
  classes = []
  acc_results = np.zeros(len(test_loader.dataset))
  i = 0

  model.eval()
  with torch.no_grad():
    for data, labels in test_loader:
      data, labels = data.cuda(), labels.cuda()
      output = model(data)
      
      for pred, true in zip(output, labels):
        _, pred = pred.unsqueeze(0).topk(1)
        correct = pred.eq(true.unsqueeze(0))
        acc_results[i] = correct.cpu()
        classes.append(model.idx_to_class[true.item()])
        i+=1
  
  results = pd.DataFrame({
      'class': classes,
      'results': acc_results    
  })
  results = results.groupby(classes).mean()

  return results


# In[ ]:


evaluate(model, test_loader, criterion)


# As seen in the results, this model was probably an overkill for this dataset, however, this is just a walkthrough of how transfer learning from pre-trained models can be used for image classification purposes. Moving forward, for more complex and larger image datasets, the earlier layers of the model can be unfrozen such that the model can pick up features specific to the dataset it is learning on.
