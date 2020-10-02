#!/usr/bin/env python
# coding: utf-8

# # **10 Monkey Species Classification using Convolutional Neural Networks in PyTorch** #

# In[ ]:


import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from tqdm.notebook import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


DATA_DIR = '../input/10-monkey-species'

TRAIN_DIR = DATA_DIR + '/training/training'                           # Contains training images  
VAL_DIR = DATA_DIR + '/validation/validation'
TEST_DIR='../input/testing-dataset-for-10-species-of-monkey/testing'


# In[ ]:


labels_name= ['mantled_howler',
              'patas_monkey',
            'bald_uakari',
            'japanese_macaque',
            'pygmy_marmoset',
            'white_headed_capuchin',
            'silvery_marmoset',
            'common_squirrel_monkey',
            'black_headed_night_monkey',
            'nilgiri_langur']


# In[ ]:


labels_name['n0']


# In[ ]:


transform = transforms.Compose ([ transforms.Resize(size=(256,256) , interpolation=2),transforms.ToTensor(),])


# In[ ]:


train_dataset = ImageFolder ( TRAIN_DIR , transform=transform )
val_dataset = ImageFolder ( VAL_DIR , transform=transform ) 
test_dataset= ImageFolder ( TEST_DIR , transform=transform ) 


# In[ ]:


len(train_dataset)


# In[ ]:


len(test_dataset)


# In[ ]:


len(val_dataset)


# In[ ]:


print(train_dataset.classes)


# In[ ]:


train_dataset[576]


# In[ ]:


def show_example(img, label):
    print('Label: ', train_dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))


# In[ ]:


show_example(*train_dataset[0])


# In[ ]:


show_example(*test_dataset[0])


# In[ ]:


random_seed = 10
torch.manual_seed(random_seed);


# In[ ]:


batch_size=32


# In[ ]:


train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size, num_workers=4, pin_memory=True)


# In[ ]:


for images,labels in train_loader:
    print('images.shape:', images.shape)
    fig, ax = plt.subplots(figsize=(32, 16))
    plt.axis('on')
    ax.set_xticks([]); ax.set_yticks([])
    plt.imshow(make_grid(images, nrow=8).permute((1, 2, 0)))
    break


# In[ ]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class MonkeyClassificationBase(nn.Module):
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
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# In[ ]:


class MonkeyCnnModel(MonkeyClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
             nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 32 x 128 x 128
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
             nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 64 x 64

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
             nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 32 x 32
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # output: 128 x 16 x 16

            nn.Flatten(), 
            nn.Linear(128*16*16, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10))
        
    def forward(self, xb):
        return self.network(xb)


# In[ ]:


class MonkeyCnnModel2(MonkeyCnnModel):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 10)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


# In[ ]:


model = MonkeyCnnModel2()
model


# In[ ]:


for images, labels in train_loader:
    print('images.shape:', images.shape)
    out = model(images)
    print('out.shape:', out.shape)
    print('out[0]:', out[0])
    break


# In[ ]:


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
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


device = get_default_device()
device


# In[ ]:


train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)
to_device(model, device);


# In[ ]:


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# In[ ]:


model = to_device(MonkeyCnnModel2(), device)


# In[ ]:


evaluate(model, val_loader)


# In[ ]:


num_epochs = 2
opt_func = torch.optim.Adam
lr = 0.001


# In[ ]:


history = fit(num_epochs, lr, model, train_loader, val_loader, opt_func)


# In[ ]:


history += fit(3, lr, model, train_loader, val_loader, opt_func)


# In[ ]:


history += fit(3, 0.0001, model, train_loader, val_loader, opt_func)


# In[ ]:


history += fit(2, 0.00001, model, train_loader, val_loader, opt_func)


# In[ ]:


get_ipython().system('pip install jovian --upgrade -q')


# In[ ]:


import jovian


# In[ ]:


jovian.commit(project='Classification_of_Monkeys_Project_Zero_to_Gans')


# In[ ]:


jovian.log_metrics(train_loss=history[-1]['train_loss'], 
                   val_loss=history[-1]['val_loss'], 
                   val_acc=history[-1]['val_acc'])


# In[ ]:


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');


# In[ ]:


plot_accuracies(history)


# In[ ]:


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');


# In[ ]:


plot_losses(history)


# In[ ]:


def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return labels_name[preds[0].item()]


# In[ ]:


img, label = test_dataset[50]
plt.imshow(img.permute(1, 2, 0))
print('Label:', labels_name[label], ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = test_dataset[0]
plt.imshow(img.permute(1, 2, 0))
print('Label:', labels_name[label], ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = test_dataset[99]
plt.imshow(img.permute(1, 2, 0))
print('Label:', labels_name[label], ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = test_dataset[11]
plt.imshow(img.permute(1, 2, 0))
print('Label:', labels_name[label], ', Predicted:', predict_image(img, model))


# test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)
# result = evaluate(model, test_loader)
# result

# In[ ]:


test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)
result = evaluate(model, test_loader)
result


# In[ ]:


jovian.log_metrics(test_loss=result['val_loss'], test_acc=result['val_acc'])


# In[ ]:


torch.save(model.state_dict(), 'monkeycnn.pth')


# In[ ]:


model2 = to_device(MonkeyCnnModel2(), device)


# In[ ]:


model2.load_state_dict(torch.load('monkeycnn.pth'))


# In[ ]:


evaluate(model2, val_loader)


# In[ ]:


jovian.commit(project='Classification_of_Monkeys_Project_Zero_to_Gans')

