#!/usr/bin/env python
# coding: utf-8

# # Architure Style Classification Project
# This kernel is about different architure style found across globe. This dataset is available on kaggle and the link is https://www.kaggle.com/wwymak/architecture-dataset.

# In[ ]:


get_ipython().system('pip install jovian --upgrade --quiet')
import jovian
jovian.commit(project='architecture-style-classification-project')


# In[ ]:


get_ipython().system('pip install git+https://github.com/ufoym/imbalanced-dataset-sampler.git # GITHUB REPO FOR IMBALANCED DATASET SAMPLER ')
import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import Dataset, random_split, Subset
from torchsampler import ImbalancedDatasetSampler
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# I have downloaded the data from above mentioned link and performed some pre-processing offline. 
# I split made 2 seperate folder i.e train and test dataset. I chose images at random for test and train set and renamed them zipped the file a .zip and uploaded the dataset on Kaggle (.zip files doesn't require to be 'unzipped' and extract the entire dataset was extracted to input folder without unzipping it)
# You can chose to use my processed dataset at https://www.kaggle.com/aarshibhatt112/archiset. Or if you want to do you own preprocessing on the main dataset you use the code below.
# 
# **NOTE**: The input directory is read-only you can't make changes there also if you make a directory at same level as input you can view the directory structure on this right side tab. That's why processed the dataset on my pc. 

# In[ ]:


# TO SPLIT THE DATA INTO TRAIN AND TEST SET

# source = "../Downloads/Dataset/train"
# dest = "../Downloads/Dataset/test"
# files = os.listdir(source)
# import shutil
# import random 
# for file in files:
#     for imgs in os.listdir(source + '/' + file):
#         if random.random() < 0.17:
#             shutil.move(source + '/'+ file + '/' + imgs, dest + '/' + file)
# print('done')

# TO RENAME THE FILES 

# for file in files:
#     for count, imgs in enumerate(os.listdir(source + '/' + file)):
#         dst ="img_" + str(count) + ".jpg"
#         src =source + '/' + file +'/' + imgs   #CURRENTLY POINT TO TRAIN SET 
#         dst = source + '/' + file +'/'+ dst
#         os.rename(src, dst) 
# print('done')

#TO GET THE DIRECTORY STRUCTURE ON KAGGLE THE PATH TO DATASET 

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# In[ ]:


data_dir = '/kaggle/input/Dataset/'

print(os.listdir(data_dir))
classes = os.listdir(data_dir + "/train")
print(classes)


# As you can above there are in total 24 classes (the original one has 25, dropped on class)
# The classes are as :
# * 'Byzantine architecture' ,
# * 'Queen Anne architecture', 
# * 'Georgian architecture',
# * 'American Foursquare architecture', 
# * 'Bauhaus architecture',
# * 'Postmodern architecture', 
# * 'Baroque architecture', 
# * 'Deconstructivism',
# * 'Tudor Revival architecture', 
# * 'Edwardian architecture', 
# * 'Chicago school architecture',
# * 'Romanesque architecture',
# * 'International style',
# * 'American craftsman style',
# * 'Russian Revival architecture',
# * 'Novelty architecture',
# * 'Palladian architecture',
# * 'Greek Revival architecture',
# * 'Art Deco architecture',
# * 'Art Nouveau architecture',
# * 'Achaemenid architecture',
# * 'Colonial architecture',
# * 'Beaux-Arts architecture',
# * 'Ancient Egyptian architecture'
# 
# 
# Let' look at one such class

# In[ ]:


decon_files = os.listdir(data_dir + "/train/Deconstructivism")
print('No. of training examples for Deconstructivism Style :', len(decon_files))
print(decon_files[:5])


# Split dataset train dataset onto 2 part validation set and training set. you can chose any ratio but I prefer 3:7.

# In[ ]:


dataset = ImageFolder(data_dir+'/train')

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
    
    
    
lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]
train_subset, val_subset = random_split(dataset, lengths)

transforms1 = tt.Compose([
    tt.RandomHorizontalFlip(),
    tt.Resize((512, 512)),
    tt.ToTensor()
])

transforms2 = tt.Compose([
    tt.Resize((512, 512)),
    tt.ToTensor()
])
    
train_dataset = DatasetFromSubset(
    train_subset, transform=transforms1
)
val_dataset = DatasetFromSubset(
    val_subset, transform=transforms2
)
# len(train_dataset), len(val_dataset)

img, label = train_dataset[0]
print(img.shape, label)


# In[ ]:


img, label = train_dataset[14]
print(img.shape, label, dataset.classes[label])


# In[ ]:


print(dataset.classes)


# In[ ]:


import matplotlib.pyplot as plt

def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))


# In[ ]:


show_example(*train_dataset[14])


# In[ ]:


random_seed = 10
torch.manual_seed(random_seed);
batch_size=10


# In[ ]:


train_dl = DataLoader(train_dataset, batch_size, sampler=ImbalancedDatasetSampler(train_subset), num_workers=4, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size*2, num_workers=4, pin_memory=True)


# In[ ]:


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break


# In[ ]:


show_batch(train_dl)


# In[ ]:


ans = dict()
for _, label in train_dataset:
    if label in ans.keys():
        ans[label] += 1
    else: 
        ans[label] = 1
# ans.items()
xlabel = classes
plt.bar(classes, ans.values())
plt.xticks(rotation=90)


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


class ArchitectureResnet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet101(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 24)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
    
#     def freeze(self):
#         # To freeze the residual layers
#         for param in self.network.parameters():
#             param.require_grad = False
#         for param in self.network.fc.parameters():
#             param.require_grad = True
    
#     def unfreeze(self):
#         # Unfreeze all layers
#         for param in self.network.parameters():
#             param.require_grad = True


# In[ ]:


model = ArchitectureResnet()
model


# In[ ]:


# for images, labels in train_dl:
#     print('images.shape:', images.shape)
#     out = model2(images)
#     print('out.shape:', out.shape)
#     print('out[0]:', out[0])
#     break


# if high usage switch

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
to_device(model, device);


# In[ ]:


# for images, labels in train_dl:
#     print('images.shape:', images.shape)
#     out = simple_model(images)
#     print('out.shape:', out.shape)
#     break


# In[ ]:


@torch.no_grad()
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


model = to_device(ArchitectureResnet(), device)


# In[ ]:


history = [evaluate(model, val_dl)]
history


# In[ ]:


num_epochs = 20
# opt_func = torch.optim.Adam
lr = 0.01


# In[ ]:


history += fit(num_epochs, lr, model, train_dl, val_dl)


# In[ ]:


num_epochs = 20
opt_func = torch.optim.Adam
lr = 0.01
history += fit(num_epochs, lr, model, train_dl, val_dl)


# In[ ]:


num_epochs = 30
opt_func = torch.optim.Adam
lr = 0.001
history += fit(num_epochs, lr, model, train_dl, val_dl)


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


transforms = tt.Compose([
    tt.Resize((512, 512)),
    tt.ToTensor()
])
test_dataset = ImageFolder(data_dir+'/test', transform=transforms)
img, label = test_dataset[197]
print(img.shape, label)


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


# In[ ]:


def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))


# In[ ]:


show_example(*test_dataset[185])


# In[ ]:


img, label = test_dataset[185]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = test_dataset[197]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))


# In[ ]:


test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)
result = evaluate(model, test_loader)
result


# In[ ]:


get_ipython().system('pip install jovian --upgrade --quiet')
import jovian
jovian.commit(project='architecture-style-classification-project')


# In[ ]:




