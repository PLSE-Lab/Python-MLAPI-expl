#!/usr/bin/env python
# coding: utf-8

# - **User**: [@manishshah120](https://www.kaggle.com/manishshah120)
# - **LinkedIn**: https://www.linkedin.com/in/manishshah120/
# - **GitHub**: https://github.com/ManishShah120
# - **Twitter**: https://twitter.com/ManishShah120
# 
# > This Notebook was created while working on project for a course "**Deep Learning with PyTorch: Zero to GANs**" from "*jovian.ml*" in collaboratoin with "*freecodecamp.org*"

# # Facial Expressoin Recognition with Convolutional Neural Network

# ## Imports

# In[ ]:


import os
import tarfile
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


project_name = '004FacialExpressoRecogCNNApproach'


# ## Info of the Dataset

# In[ ]:


data_dir = '../input/facial-expression-recog-image-ver-of-fercdataset/Dataset'
classes = os.listdir(data_dir + '/train')


# No. of training images of each class in training set:-

# In[ ]:


for i in classes:
    var_files = os.listdir(data_dir + '/train/' + i)
    print(i, ':', len(var_files))


# No. of images of each class in test set:-

# In[ ]:


for i in classes:
    var_files = os.listdir(data_dir + '/test/' + i)
    print(i,': ',len(var_files))


# creating the `dataset` variable

# In[ ]:


dataset = ImageFolder(
    data_dir + '/train', 
    transform = ToTensor()
                     )


# In[ ]:


dataset


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


show_example(*dataset[5300])


# ## **Training and Validation Datasets**

# In[ ]:


random_seed = 42
torch.manual_seed(random_seed)


# In[ ]:


val_size = int(0.1*len(dataset))
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)


# In[ ]:


batch_size = 64


# ## Data Loaders

# In[ ]:


train_dl = DataLoader(
                        train_ds, 
                        batch_size, 
                        shuffle=True, 
                        num_workers=4,
                        pin_memory=True   
                     )

val_dl = DataLoader(
                        val_ds, 
                        batch_size*2, 
                        num_workers=4, 
                        pin_memory=True
                    )


# In[ ]:


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break


# In[ ]:


show_batch(train_dl)


# ## Defining the Model(CNN)

# In[ ]:


def apply_kernel(image, kernel):
    ri, ci = image.shape       # image dimensions
    rk, ck = kernel.shape      # kernel dimensions
    ro, co = ri-rk+1, ci-ck+1  # output dimensions
    output = torch.zeros([ro, co])
    for i in range(ro): 
        for j in range(co):
            output[i,j] = torch.sum(image[i:i+rk,j:j+ck] * kernel)
    return output


# In[ ]:


# simple_model = nn.Sequential(
#     nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
#     nn.MaxPool2d(2, 2)
# )


# In[ ]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
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


class FacialExpressRecogCnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 96 x 24 x 24

            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 192 x 12 x 12

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 384 x 6 x 6

            nn.Flatten(), 
            nn.Linear(384*6*6, 2304),
            nn.ReLU(),
            nn.Linear(2304, 1152),
            nn.ReLU(),
            nn.Linear(1152, 576),
            nn.ReLU(),
            nn.Linear(576, 7)
                                    )
        
    def forward(self, xb):
        return self.network(xb)


# In[ ]:


model = FacialExpressRecogCnnModel()
model


# In[ ]:


for images, labels in train_dl:
    print('images.shape:', images.shape)
    out = model(images)
    print('out.shape:', out.shape)
    print('out[0]:', out[0])
    break


# ## Using a GPU

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


train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device)


# ## Functions to train the Model

# In[ ]:


@torch.no_grad()    # This is to say that PyTorch to stop tracking of grad
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
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


model = to_device(FacialExpressRecogCnnModel(), device)


# In[ ]:


evaluate(model, val_dl)


# In[ ]:


num_epochs = 10
opt_func = torch.optim.Adam
lr = 0.001


# In[ ]:


history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)


# In[ ]:


#num_epochs = 20
#opt_func = torch.optim.Adam
#lr = 0.001


# In[ ]:


#history += fit(num_epochs, lr, model, train_dl, val_dl, opt_func)


# In[ ]:


#num_epochs = 30
#opt_func = torch.optim.Adam
#lr = 0.001


# In[ ]:


#history += fit(num_epochs, lr, model, train_dl, val_dl, opt_func)


# In[ ]:


evaluate(model, test_loader)


# ## Plotting Functions

# In[ ]:


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')


# In[ ]:


plot_accuracies(history)


# In[ ]:


plot_losses(history)


# ## Function to Predict

# In[ ]:


def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]


# ## Test Dataset
# Lets store the test images in the `test_dataset` variable

# In[ ]:


test_dataset = ImageFolder(data_dir+'/test', transform=ToTensor())


# ### Predictions

# In[ ]:


img, label = test_dataset[1]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = test_dataset[1034]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = test_dataset[2315]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))


# In[ ]:


stop here


# ## Commit to Jovian

# In[ ]:


get_ipython().system('pip install jovian')


# In[ ]:


import jovian
jovian.commit(project=project_name)


# In[ ]:


test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)
result = evaluate(model, test_loader)
result


# In[ ]:


jovian.log_metrics(
                    test_loss=result['val_loss'], 
                    test_acc=result['val_acc'],
                    train_loss=history[-1]['train_loss'], 
                    val_loss=history[-1]['val_loss'], 
                    val_acc=history[-1]['val_acc']
                   )


# In[ ]:


num_epochs = [10]
opt_func = torch.optim.Adam
lr = [0.001]


# In[ ]:


jovian.log_hyperparams({
    'num_epochs': num_epochs,
    'opt_func': opt_func.__name__,
    'batch_size': batch_size,
    'lr': lr,
})


# In[ ]:



jovian.commit(project=project_name, environment=None)


# In[ ]:




