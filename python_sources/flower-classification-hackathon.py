#!/usr/bin/env python
# coding: utf-8

# ***Project - virtual hackathon - Hackathon Blossom (Flower Classification)***

# **Model architechture and intuition behind it**
# * Firstly i analyzed the dataset that it is a multi class classification problem with 102 classes of flowers to be predicted accurately.
# * I have already experience with pretrained models like resnet models trained on imagenet dataset and therefore , i choose resnet152 as it contains complexities for non - linear boundaries needed to classify the dataset with so many classes and has already been proven in many competitions round the world as one of the best architechtures.
# * Also i firstly simply trained the model with just changing the last layer 'fc' and keeping other layers freezed.
# * In the second mode , i tried to unfreeze the last two layers - 'layer3', 'layer4' and got accuracy on validation set about 95 % using lr scheduler
# * In the third mode , i tried to run by lowering the learn rate more and applying lr scheduler with more lower gamma and unfrozen 'layer1' and 'layer2' keeping others freezed as they are already trained.
# **Hyperparameteres**
# * For first run : i set 
# * -> epochs : 20
# * -> optimizer : adam with learning rate 0.001
# * -> classifier : one hidden layer containing 512 neurons and last final layer containing 102 neurons 
# * -> used ReLU activations for inner layers and softmax for final layer
# 
# * For second run : i set
# * -> epochs : 20
# * -> learning rate : 0.0001
# * -> learning scheduler : stepLR with step_size = 6 and gamma = 0.1
# 
# keeping other things same as in above first run
# 
# * For third run : i set
# * -> epochs : 10
# * -> lr : 0.0001
# * -> lr scheduler : stepLR with step_size = 5 and gamma 0.01
# 
# keeping other things same as in second run

# In[ ]:


import torch
from torch import optim, nn
import torchvision
from torchvision import datasets, models, transforms
import numpy as np


# In[ ]:


get_ipython().system('wget -c https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/master/intro-to-pytorch/helper.py')


# In[ ]:


train_dir = '../input/hackathon-blossom-flower-classification/flower_data/flower_data/train'
valid_dir = '../input/hackathon-blossom-flower-classification/flower_data/flower_data/valid'


# In[ ]:


train_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


# In[ ]:


train_data = datasets.ImageFolder(train_dir , transform=train_transforms)
test_data = datasets.ImageFolder(valid_dir, transform=test_transforms)


# In[ ]:


trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)


# In[ ]:


print(len(train_data)/64)
print(len(test_data)/64)


# In[ ]:


import json

with open('../input/hackathon-blossom-flower-classification/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# In[ ]:


# This is the contents of helper.py 
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable


def test_network(net, trainloader):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Create Variables for the inputs and targets
    inputs = Variable(images)
    targets = Variable(images)

    # Clear the gradients from all Variables
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = net.forward(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    return True
def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax
def view_recon(img, recon):
    ''' Function for displaying an image (as a PyTorch Tensor) and its
        reconstruction also a PyTorch Tensor
    '''

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].imshow(img.numpy().squeeze())
    axes[1].imshow(recon.data.numpy().squeeze())
    for ax in axes:
        ax.axis('off')
        ax.set_adjustable('box-forced')  


# In[ ]:


class_names = train_data.classes


# In[ ]:


images, labels = next(iter(testloader))


# In[ ]:


import torchvision
grid = torchvision.utils.make_grid(images, nrow = 20, padding = 2)
plt.figure(figsize = (15, 15))  
plt.imshow(np.transpose(grid, (1, 2, 0)))   
print('labels:', labels)


# In[ ]:


imshow(images[63])
labels[63].item()


# In[ ]:


train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


# In[ ]:


get_ipython().system('ls -la')


# In[ ]:


from torch.optim import lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet152(pretrained=True)

for name,child in model.named_children():
  if name in ['layer3','layer4']:
    print(name + 'is unfrozen')
    for param in child.parameters():
      param.requires_grad = True
  else:
    print(name + 'is frozen')
    for param in child.parameters():
      param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),nn.ReLU(),nn.Linear(512,102),nn.LogSoftmax(dim=1))    

criterion = nn.NLLLoss()


optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()) , lr = 0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)


model.to(device);


# In[ ]:


def train_and_test():
    epochs = 10
    train_losses , test_losses = [] , []
    valid_loss_min = np.Inf 
    model.train()
    for epoch in range(epochs):
      running_loss = 0
      batch = 0
      scheduler.step()
      for images , labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        batch += 1
        if batch % 10 == 0:
            print(f" epoch {epoch} batch {batch} completed")
      test_loss = 0
      accuracy = 0
      with torch.no_grad():
        model.eval() 
        for images , labels in testloader:
          images, labels = images.to(device), labels.to(device)
          logps = model(images) 
          test_loss += criterion(logps,labels) 
          ps = torch.exp(logps)
          top_p , top_class = ps.topk(1,dim=1)
          equals = top_class == labels.view(*top_class.shape)
          accuracy += torch.mean(equals.type(torch.FloatTensor))
      train_losses.append(running_loss/len(trainloader))
      test_losses.append(test_loss/len(testloader))
      print("Epoch: {}/{}.. ".format(epoch+1, epochs),"Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),"Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
        "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
      model.train() 
      if test_loss/len(testloader) <= valid_loss_min:
        print('test loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,test_loss/len(testloader))) 
        torch.save(model.state_dict(), 'checkpoint.pth')
        valid_loss_min = test_loss/len(testloader)


# In[ ]:


def load_model():
    state_dict = torch.load('checkpoint.pth')
    print(state_dict.keys())
    model.load_state_dict(state_dict)
    


# In[ ]:


for name,child in model.named_children():
  if name in ['layer1','layer2']:
    print(name + 'is unfrozen')
    for param in child.parameters():
      param.requires_grad = True
  else:
    print(name + 'is frozen')
    for param in child.parameters():
      param.requires_grad = False


# In[ ]:


load_model()


# In[ ]:


optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()) , lr = 0.00001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)


# In[ ]:


model.class_idx_mapping = train_data.class_to_idx
class_idx_mapping = train_data.class_to_idx
idx_class_mapping = {v: k for k, v in class_idx_mapping.items()}


# In[ ]:


data_dir = "../input/hackathon-blossom-flower-classification/test set"
valid_data = datasets.ImageFolder(data_dir, transform=test_transforms)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)


# In[ ]:


import pandas as pd
import os
def predict(validloader, model_checkpoint, topk=1, device="cuda", idx_class_mapping=idx_class_mapping):
    model.to(device)
    model.eval()
    
    labels=[]
    
    with torch.no_grad():
        for images, _ in validloader:
            images = images.to(device)
            output = model(images)
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1) 
            for i in top_class:
                
                labels.append(idx_class_mapping[i.item()] )
                
                      
               
    return labels


# In[ ]:


class_label=predict(validloader,"checkpoint.pth", idx_class_mapping=idx_class_mapping)


# In[ ]:


class_label


# In[ ]:


image_col=[]
for img_filename in os.listdir('../input/hackathon-blossom-flower-classification/test set/test set'):
    image_col.append(img_filename)


# In[ ]:


category_map = sorted(cat_to_name.items(), key=lambda x: int(x[0]))
plant_name=[]
for label in class_label:
    name=cat_to_name[label]
    
    plant_name.append(name)
    


# In[ ]:


submission = pd.DataFrame({'image_test': image_col, 'pred_class': class_label,'species': plant_name})
submission.sort_values('image_test')


# In[ ]:


print(submission.head())


# In[ ]:


submission.to_csv('my_predictions_test.csv')

