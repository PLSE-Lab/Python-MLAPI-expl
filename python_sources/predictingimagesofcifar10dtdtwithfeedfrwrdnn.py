#!/usr/bin/env python
# coding: utf-8

# # **Image Classification Using Feed Forward Neural Network in PyTorch with CIFAR-10 Data Set**

# In[ ]:


import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


project_name='PredictingImagesOfCifar10DtDtWithFeedfrwrdNN'


# ## Preparing the Data

# In[ ]:


# Downloading dataset
dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())


# In[ ]:


dataset


# In[ ]:


test_dataset


# In[ ]:


dataset.classes


# In[ ]:


val_size = 10000
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)


# In[ ]:


batch_size=128


# ## Data Loaders

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
test_loader = DataLoader(
                         test_dataset, 
                         batch_size*2, 
                         num_workers=4, 
                         pin_memory=True
                        )


# ## Lets View a set of Data

# In[ ]:


for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break


# ## **Model**

# In[ ]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# In[ ]:


class CIFAR10Model(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""
    def __init__(self, in_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, 64)
        # hidden layer 2
        self.linear2 = nn.Linear(64, 128)
        # hidden layer 3
        self.linear3 = nn.Linear(128,256)
        # hidden Layer 4
        self.linear4 = nn.Linear(256, 512)
        # output layer
        self.linear5 = nn.Linear(512, out_size)
        
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

        # Get intermediate outputs using hidde layer 3
        out = self.linear3(out)
        # Apply activation function
        out = F.relu(out)

        # Get intermediate outputs using hidde layer 4
        out = self.linear4(out)
        # Apply activation function
        out = F.relu(out)

        # Get predictions using output layer
        out = self.linear5(out)
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


# ## **Using A GPU**

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


# In[ ]:


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Lets define a function to move all our data and the model from athe cpu to GPU

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


train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)
test_loader = DeviceDataLoader(test_loader, device)


# ## **Training the Model**

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


input_size = 3*32*32
num_classes = 10


# In[ ]:


model = CIFAR10Model(input_size, out_size=num_classes)
to_device(model, device)


# In[ ]:


history = [evaluate(model, val_loader)]
history


# In[ ]:


history += fit(50, 0.01, model, train_loader, val_loader)


# In[ ]:


history += fit(10, 0.001, model, train_loader, val_loader)


# In[ ]:


#history += fit(20, 0.0001, model, train_loader, val_loader)


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


# Evaluate on test dataset
result = evaluate(model, test_loader)
result


# ## **Predictions**

# In[ ]:


def predict_image(img, model):
    xb = to_device(img.unsqueeze(0),device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()


# In[ ]:


img, label = test_dataset[10000-4143]
plt.imshow(img[0])
print('Label:', test_dataset.classes[label], ', Predicted:', dataset.classes[predict_image(img, model)])


# In[ ]:


for i in range(10):
    img, label = test_dataset[i]
    print('Label:', test_dataset.classes[label], ', Predicted:', dataset.classes[predict_image(img, model)])


# In[ ]:


evaluate(model, test_loader)


# In[ ]:


Stop here


# ## Lets record our parameters and results to jovian.ml

# In[ ]:


# E.g. "3 layers (16,32,10)" (16, 32 and 10 represent output sizes of each layer)
arch = "5 layers (64, 128, 256, 512, 10)"
# the list of learning rates used while training.
lrs = [0.01, 0.001]
# the list of no. of epochs used while training.
epochs = [50, 10]
# the final test accuracy & test loss?
test_acc = 0.4920898377895355
test_loss = 1.4203943014144897


# In[ ]:


# let's save the trained model weights to disk, so we can use this model later.
torch.save(model.state_dict(), 'Mcifar10-feedforwardNN.pth')


# In[ ]:


get_ipython().system('pip install jovian --upgrade --quiet')
import jovian
# Clear previously recorded hyperparams & metrics
#jovian.reset()


# In[ ]:


jovian.log_hyperparams(arch=arch, 
                       lrs=lrs, 
                       epochs=epochs)


# In[ ]:


jovian.log_metrics(test_loss=test_loss, test_acc=test_acc)


# Finally, we can commit the notebook to Jovian, attaching the hypeparameters, metrics and the trained model weights.

# In[ ]:


jovian.commit(project=project_name, outputs=['Mcifar10-feedforwardNN.pth'], environment=None, message='4th commit achieved 49% accu', is_cli=True)


# # **THE END**
