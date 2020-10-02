#!/usr/bin/env python
# coding: utf-8

# # Flower Recognition
# ## Recognizing the type of flower from its image
# 
# ### About the dataset:
# - This dataset has been derived from the [Kaggle Flowers Recognition Dataset](https://www.kaggle.com/alxmamaev/flowers-recognition). I have further cleaned the dataset by removing irrevelant files and images contained in the different folders corresponding to each class of flower (More on this later)
# - This dataset contains approx. 4138 images of flowers, the distribution of images for each type of flower is not equal
# - The data collection is based on the data from **flicr**, **google images**, **yandex images**
# 
# ### Dataset Contents:
# - The images are divided into five classes: 
#     1. *daisy*
#     2. *dandelion*
#     3. *rose*
#     4. *sunflower*
#     5. *tulip*
# - For each class of flowers there are about 800 photos each
# - The images are not high resolution, each of them are about `320x240` pixels
# - The images are not reduced to a single size, they are of different proportions
# 
# ### Acknowledgements
# - The data collection is based on scraped data from **flicr**, **google images**, **yandex images**

# In[ ]:


import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torchvision.transforms as T

import random
import math

from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Preparing the data for training
# We'll use a validation set with about 510 images (12% of the dataset). To ensure we get the same train, validation and test set each time, we'll set the random number generator to a seed value of 43. This is done so that the results of running this notebook can be reproduced by others as well.

# In[ ]:


np.random.seed(43)


# ### Also before starting work on the project, lets also set the name of out project

# In[ ]:


PROJECT_NAME='final-course-project'


# Now let's install jovain in our environment and perform an inital commit

# In[ ]:


get_ipython().system('pip install jovian --upgrade -q')


# In[ ]:


import jovian


# In[ ]:


dir(jovian)


# In[ ]:


jovian.commit(project=PROJECT_NAME, environment=None)


# ### In the original [Kaggle Flowers Recognition Dataset](https://www.kaggle.com/alxmamaev/flowers-recognition), a few of the image folders had `.py` and `.pyc` files in them, due to which the model while loading the data from the folders would load these files as well thinking that they are images. To correct this issue, I downloaded the dataset and removed these files from the various image folders and re-uploaded the corrected dataset onto my Kaggle account.
# ### The below cell of code just checks all the image folders to make sure that there are no such files left in the folders, which are not in `.jpg` or `.jpeg` or `.png` format. Just in case there are such files which are not of `.jpg` or `.jpeg` or `.png` format then the program prints their names out, so that I can remove those files.
# #### On Kaggle the dataset folder is read-only so if any corrections are required then I need to first download the dataset and then make the needed corrections and finally re-upload it onto Kaggle.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

bad_file = False

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if not filename.endswith('.jpg') and not filename.endswith('.jpeg') and not filename.endswith('.png'):  # If it does not end with .jpg or .jpeg or .png extension
            bad_file = True
            print(os.path.join(dirname, filename))  # Show the file that does not have a .jpg or .jpeg extension
            print('-'*80)  # Print a line just under the file that does not end with .jpg or .jpeg or .png extension

if bad_file == False:
    print("All files in the dataset are of .jpg or .jpeg or .png format only.")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


DATA_DIR = os.path.join('..', 'input', 'newFlowers')
print(DATA_DIR)
print(type(DATA_DIR))


# ### Now lets take a look at the structure of the dataset folder

# In[ ]:


os.listdir(DATA_DIR)


# ### Also let's get the absolute path to the current directory

# In[ ]:


os.getcwd()


# ## Creating a DataLoadingPipeline class to accumulate all the dataloading operations in one place.
# ### In order to bring all the dataloading operations to one place, I have created a class that will load the data and apply all the required transforms to the it as well.
# 
# 
# - The object of this class contains the train, validation and test data loaders as well
# - The other perculiar feature of the dataset is that there is no separate train, validation and test set, there is just one dataset of images that needs to be manually split into train, validation and test sets. 
#     * Rather than restructuring the whole dataset, which can be time consuming and unnecessary, we will use a SubsetRandomSampler which will only pick those images whose respective indices we provide. 
#     * This way eventhough the train_data, validation_data and test_data attributes of the object, all have the same images; we can create one `SubsetRandomSampler` object for each set and provide each *one different and mutually exclusive indices*. This way the images in the train_loader, validation_loader and test_loader attributes of the object will all be different.

# In[ ]:


class DataLoadingPipeline:
    def __init__(self, data_dir: str, validation_fraction:float=0.15,
                 test_fraction:float=0.2, train_transforms:list=[T.ToTensor(),],
                 valid_and_test_transforms:list=[T.ToTensor(),], batch_size:int=64):
        """
            This function sets up the basic data loader for the dataset. The data loaded is not moved to GPU by this method.
            data_dir: {str}  ==> This is the absolute path to the dataset folder
            validation_fraction: {float} ==> This is the fraction of train set that will be used for validation
            test_fraction: {float} ==> This is the fraction of dataset that will be used for testing, rest will be used for training (and validation of course)
            train_transforms: {list} ==> This is the list of transforms which will be applied to the train set
            valid_and_test_transforms: {list} ==> This is the list of transforms which will be applied to the validation and test set
        """
        # Creating the transforms pipeline for each of the fractions of the dataset
        self.train_transforms = T.Compose(train_transforms)
        self.valid_transforms = T.Compose(valid_and_test_transforms)
        self.test_transforms  = T.Compose(valid_and_test_transforms)
        
        # Reading the folder containing the images for creating the initial train, validation and test datasets
        self.train_data = ImageFolder(data_dir, transform=self.train_transforms)
        self.validation_data = ImageFolder(data_dir, transform=self.valid_transforms)
        self.test_data = ImageFolder(data_dir, transform=self.test_transforms)
        self.train_validation_data = ImageFolder(data_dir, transform=self.train_transforms)
        
        # Creating a dictionary for storing the conversion from the lable number to the flower name
        self.classes = {}
        # Print out the classes in the dataset along with their corresponding index
        print("The classes in this dataset are:")
        for ctr, i in enumerate(self.train_data.classes):
            print(f"{ctr}: {i.capitalize()}")
            self.classes[ctr] = i
        print()  # Print a newline for better output formatting

        self.count = num_train = len(self.train_data)     # Get the total number of images in the dataset
        print(f"The dataset has {num_train} images")
        indices = list(range(num_train))  # Create a list of indices for the all images in the dataset

        test_split = int(np.floor(test_fraction * num_train))  # Getting the number of images in the test set
        train_validation_split = num_train - test_split        # Getting the number of images in the train and validation set
        validation_split = int(np.floor(validation_fraction * train_validation_split))  # Getting the number of images in the validation set
        train_split = train_validation_split - validation_split  # Getting the number of images in the train set

        # Construct a new Generator with the default BitGenerator (PCG64), this will hellp us shuffle the indices randomly
        rng = np.random.default_rng()
        rng.shuffle(indices)  # Shuffling the indices so that every set gets approximately equal number of images for each class

        # Splitting the indices list into train_validation and test indices lists
        train_validation_idx, test_idx = indices[test_split:], indices[:test_split]
        
        # Reshuffling the train_validation indices list; preparing it for another split
        rng = np.random.default_rng()
        rng.shuffle(train_validation_idx)
        
        # Further split the train_validation indices list into train indices list and validation indices list
        train_idx, validation_idx = train_validation_idx[validation_split:], train_validation_idx[:validation_split]

        # Just for a sanity check, lets check if the train, validation and test sets have all got unique indices and none have overlapped
        # as that would mean that the corresponding image is in both the sets its indice occurs in
        if not set(train_idx).intersection(set(validation_idx)) and not set(validation_idx).intersection(set(test_idx)) and            not set(train_idx).intersection(set(test_idx)):
            print("[PASS] The splits are mutually exculsive of each other!")
        else:
            print("[FAIL] The splits are not mutually exculsive of each other!")
            
        
        
        # We now create random samplers to take random samples of indices from the indices lists we created.
        # This will make sure that the train set only accesses the images referred to by the train images indices list
        # and the same applies to the validation and test sets and their validation and test images indices lists.
        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(validation_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        
        # This will be used to make sure that the model gets an opportunity to train on more data.
        # As we cannot use the test set for this (for obvious reasons), we instead use the validation set, so hence the below dataloader
        # for using the train and validation set together for training, this will be used towards the end of the training process
        train_validation_sampler = SubsetRandomSampler(train_idx + validation_idx)  # This is a combination of the train and validation sets
        
        # We now create the dataloaders for the train, validation and test sets
        self.train_loader = DataLoader(self.train_data, sampler=train_sampler, batch_size=batch_size, num_workers=3, pin_memory=True)
        self.validation_loader = DataLoader(self.validation_data, sampler=validation_sampler, batch_size=batch_size, num_workers=3, pin_memory=True)
        self.test_loader = DataLoader(self.test_data, sampler=test_sampler, batch_size=batch_size, num_workers=3, pin_memory=True)
        self.train_validation_loader = DataLoader(self.train_validation_data, sampler=train_validation_sampler, batch_size=batch_size, num_workers=3, pin_memory=True)
       


# ## Now let's set the basic parameters for the dataset before we create a `DataLoaderPipeline object` for it

# In[ ]:


RESIZE_DIM = 300
IMG_DIM = 256     # We want all images to be of dimension 128x128
BATCH_SIZE = 128  # 64
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


TRAIN_TRANSFORMS = [
                    T.Resize(RESIZE_DIM, interpolation=Image.BICUBIC),
                    T.CenterCrop(IMG_DIM),
                    T.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1),
                    T.RandomHorizontalFlip(), 
                    # T.RandomCrop(IMG_DIM, padding=8, padding_mode='reflect'),
                    # T.RandomRotation(10),  #  Did not give any improvements, some images lost important details as they go cut off
                    T.ToTensor(), 
                    # T.Normalize(*imagenet_stats,inplace=True),  #  Did not give good results, converted some images into just white squares
                    T.RandomErasing(inplace=True, scale=(0.01, 0.23)),
                   ]

VALIDATION_and_TEST_TRANSFORMS = [
                                  T.Resize(RESIZE_DIM, interpolation=Image.BICUBIC),
                                  T.CenterCrop(IMG_DIM),
                                  T.ToTensor(), 
                                  # T.Normalize(*imagenet_stats)  #  Did not give good results, converted some images into just white squares
                                 ]


# In[ ]:


VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.2


flowers_data_loader = DataLoadingPipeline(
                                          data_dir=DATA_DIR,
                                          validation_fraction=VALIDATION_FRACTION,
                                          test_fraction=TEST_FRACTION,
                                          train_transforms=TRAIN_TRANSFORMS,
                                          valid_and_test_transforms=VALIDATION_and_TEST_TRANSFORMS,
                                          batch_size=BATCH_SIZE
                                         )


# In[ ]:


help(jovian.log_dataset)


# ## Now lets see the number of batches in each of the dataloaders

# In[ ]:


train_loader_batches_count, test_loader_batches_count, validation_loader_batches_count = len(flowers_data_loader.train_loader),                                                                                         len(flowers_data_loader.test_loader),                                                                                         len(flowers_data_loader.validation_loader)

# Let's see the batch sizes for the different sets of data
print(f"{'The number of training batches are': <40} {train_loader_batches_count:^4}, each of size of {BATCH_SIZE: ^4}")
print(f"{'The number of testing batches are': <40} {test_loader_batches_count: ^4}, each of size of {BATCH_SIZE: ^4}")
print(f"{'The number of validation batches are': <40} {validation_loader_batches_count: ^4}, each of size of {BATCH_SIZE: ^4}")

# Also lets show the data in the form of a simple tuple representation
print(f"({train_loader_batches_count}, {test_loader_batches_count}, {validation_loader_batches_count})")


# In[ ]:


jovian.log_dataset(dataset_url='https://www.kaggle.com/alxmamaev/flowers-recognition',
                   val_fraction=VALIDATION_FRACTION,
                   test_fraction=TEST_FRACTION,
                   train_batches=train_loader_batches_count,
                   test_batches=test_loader_batches_count,
                   validation_batches=validation_loader_batches_count)


# ## Now we are checking the train, validation and test data loaders to see if they all got images from the different classes in the dataset

# In[ ]:


flowers_data_loader.train_loader.dataset.classes == flowers_data_loader.test_loader.dataset.classes
flowers_data_loader.test_loader.dataset.classes == flowers_data_loader.validation_loader.dataset.classes


# Looks like all the dataloaders got at least some images from each class in the dataset

# In[ ]:


for images, labels in flowers_data_loader.train_loader:
    print(images)
    print(labels)
    break


# In[ ]:


for images, labels in flowers_data_loader.train_validation_loader:
    print(images)
    print(labels)
    break


# In[ ]:


for images, labels in flowers_data_loader.test_loader:
    print(images)
    print(labels)
    break


# In[ ]:


def show_sample(data_item_obj, classes:dict, invert:bool=False):
    print("The tensor representing the image and the target image", data_item_obj)
    img, target = data_item_obj  # This is a particular data item from the data set having its own image and label
    if invert:
        plt.imshow(1 - img.permute((1, 2, 0)))
    else:
        plt.imshow(img.permute(1, 2, 0))
    plt.title(classes[target])
    print('Labels:', classes[target])


# ## The function below generates random integers in the range of: *0 to (number_of_elements_in_the_data_set - 1)*
# ### The reason for this upper and lower limit is that we need to generate random integers representing the indices of data points in the dataset

# In[ ]:


def random_no_gen(no_of_elements:int, lower_limit:int=0) -> int:
    return lower_limit + math.floor(random.random() * no_of_elements)


# ## Taking a Glance at the Data
# ### Now lets take a look at some of the images in the dataset, along with their respective labels

# In[ ]:


show_sample(flowers_data_loader.train_data[random_no_gen(flowers_data_loader.count)], flowers_data_loader.classes)


# In[ ]:


show_sample(flowers_data_loader.train_data[random_no_gen(flowers_data_loader.count)], flowers_data_loader.classes)


# In[ ]:


show_sample(flowers_data_loader.train_data[random_no_gen(flowers_data_loader.count)], flowers_data_loader.classes)


# In[ ]:


show_sample(flowers_data_loader.train_data[random_no_gen(flowers_data_loader.count)], flowers_data_loader.classes)


# In[ ]:


show_sample(flowers_data_loader.train_data[random_no_gen(flowers_data_loader.count)], flowers_data_loader.classes)


# In[ ]:


def show_batch(dl, invert:bool=False):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(8, 16))
        ax.set_xticks([]); ax.set_yticks([])
        data = 1-images if invert else images
        ax.imshow(make_grid(data, nrow=8).permute(1, 2, 0))
        break


# In[ ]:


show_batch(flowers_data_loader.train_loader, invert=True)


# In[ ]:


show_batch(flowers_data_loader.train_loader)


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


flowers_data_loader.train_loader = DeviceDataLoader(flowers_data_loader.train_loader, device)
flowers_data_loader.validation_loader = DeviceDataLoader(flowers_data_loader.validation_loader, device)
flowers_data_loader.test_loader = DeviceDataLoader(flowers_data_loader.test_loader, device)
flowers_data_loader.train_validation_loader = DeviceDataLoader(flowers_data_loader.train_validation_loader, device)


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
    
    @torch.no_grad()
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    @torch.no_grad()
    def evaluate_test_set(self, test_dataset):
        result = evaluate(self, test_dataset)
        print("The results are: test_loss: {:.4f}, test_acc: {:.4f}".format(result['val_loss'], result['val_acc']))
        return {'test_loss': result['val_loss'], 'test_acc': result['val_acc']}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


# In[ ]:


COMMON_IN = 2048  # The input to the classifier layer is 2048 from the rest of ResNet50, so this value is fixed for ResNet50
NUM_CLASSES = 5   # There are 5 classes of flowers and for each we need to return a prediction probability, in the end

CLASSIFIER_ARCHITECTURES = {
                            "Simple With Dropout": nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, NUM_CLASSES)),
                            "Simple Without Dropout": nn.Sequential(nn.Linear(2048, 128),nn.ReLU(), nn.Linear(128, NUM_CLASSES)),
                            "Medium With Dropout": nn.Sequential(nn.Linear(2048, 256), nn.Dropout(0.1), nn.ReLU(), nn.Linear(256, 64), nn.Dropout(0.01), nn.ReLU(), nn.Linear(64, NUM_CLASSES)),
                            "Medium Without Dropout": nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, NUM_CLASSES))
                            }

class ResNet50(ImageClassificationBase):
    def __init__(self, num_classes:int):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        # Freeze parameters so we don't backprop through them
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = CLASSIFIER_ARCHITECTURES["Medium With Dropout"]
        
        for param in self.model.fc.parameters():
            param.require_grad = True
        
    def forward(self, xb):
        return self.model(xb)

    def switch_on_gradients(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def switch_off_gradients_except_classifier(self):
        # We first switch off the requires_grad parameter for all the layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # We then only switch on the requires_grad parameter for the layers of the (fc) classifer layer
        for param in self.model.fc.parameters():
            param.require_grad = True


# In[ ]:


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    history = []
    
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader),
                                                cycle_momentum=True)
    
    for epoch in range(epochs):
        # Training Phase 
        model.train() # Switches on training mode
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad() # reset the gradients
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# In[ ]:


model = to_device(ResNet50(num_classes=5), device)
model


# In[ ]:


jovian.commit(project=PROJECT_NAME, environment=None)


# ## Before starting the various training steps, let's clear the GPU cache first
# ### This is done so that the GPU does not run out of memory during the training steps

# In[ ]:


torch.cuda.empty_cache()


# In[ ]:


lrs = []
epochs_list = []
train_times = []


# ## Let's try evaluating the model with the validation set without any prior training
# 

# In[ ]:


history = [evaluate(model, flowers_data_loader.validation_loader)]
history


# Namely its quite interesting that eventhough the classification layer is randomly initiated, it is still able to get a **accuracy of 15%**. This may be because the *rest of the model* is already `pretrained`.

# ## Training the model
# 
# Before we train the model, we're going to make a bunch of small but important improvements to our `fit` function:
# 
# * **Learning rate scheduling**: Instead of using a fixed learning rate, we will use a learning rate scheduler, which will change the learning rate after every batch of training. There are many strategies for varying the learning rate during training, and the one we'll use is called the **"One Cycle Learning Rate Policy"**, which involves starting with a low learning rate, gradually increasing it batch-by-batch to a high learning rate for about 30% of epochs, then gradually decreasing it to a very low value for the remaining epochs. Learn more: https://sgugger.github.io/the-1cycle-policy.html
# 
# * **Weight decay**: We also use weight decay, which is yet another regularization technique which prevents the weights from becoming too large by adding an additional term to the loss function.Learn more: https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab
# 
# * **Gradient clipping**: Apart from the layer weights and outputs, it also helpful to limit the values of gradients to a small range to prevent undesirable changes in parameters due to large gradient values. This simple yet effective technique is called gradient clipping. Learn more: https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48
# 
# 
# Let's define a `fit_one_cycle` function to incorporate these changes. We'll also record the learning rate used for each batch.

# In[ ]:


epochs = 10
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-5
opt_func = torch.optim.Adam

# Logging the hyper-parameters
lrs.append(max_lr)
epochs_list.append(epochs)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history += fit_one_cycle(epochs, max_lr, model, flowers_data_loader.train_loader, flowers_data_loader.validation_loader, \n                         grad_clip=grad_clip, \n                         weight_decay=weight_decay, \n                         opt_func=opt_func)')


# In[ ]:


train_times.append('5min 36s')


# In[ ]:


model.switch_on_gradients()
model


# In[ ]:


epochs = 20
max_lr = 0.001
grad_clip = 0.05
weight_decay = 1e-4
opt_func = torch.optim.Adam

# Logging the hyper-parameters
lrs.append(max_lr)
epochs_list.append(epochs)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history += fit_one_cycle(epochs, max_lr, model, flowers_data_loader.train_loader, flowers_data_loader.validation_loader, \n                         grad_clip=grad_clip, \n                         weight_decay=weight_decay, \n                         opt_func=opt_func)')


# In[ ]:


train_times.append('4min 4s')


# ---

# In[ ]:


def save_clear_reload_model(model):
    torch.save(model.state_dict(), 'flowers_cnn.pth')
    torch.cuda.empty_cache()
    model = to_device(ResNet50(num_classes=5), device)
    model.load_state_dict(torch.load('flowers_cnn.pth'))


# In[ ]:


save_clear_reload_model(model)
model.switch_on_gradients()
model


# ## We now train the model on the train and validation sets together so that we can give our model one final boost, to get the best results during the final run on the test set

# In[ ]:


epochs = 10
max_lr = 0.001
grad_clip = 0.015
weight_decay = 1e-4
opt_func = torch.optim.Adam

# Logging the hyper-parameters
lrs.append(max_lr)
epochs_list.append(epochs)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history += fit_one_cycle(epochs, max_lr, model, flowers_data_loader.train_validation_loader,flowers_data_loader.test_loader, \n                                           grad_clip=grad_clip, \n                                           weight_decay=weight_decay, \n                                           opt_func=opt_func)')


# In[ ]:


train_times.append('2min 19s')


# In[ ]:


save_clear_reload_model(model)


# In[ ]:


epochs = 10
max_lr = 0.0001
grad_clip = 0.005
weight_decay = 1e-5
opt_func = torch.optim.Adam

# Logging the hyper-parameters
lrs.append(max_lr)
epochs_list.append(epochs)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history += fit_one_cycle(epochs, max_lr, model, flowers_data_loader.train_validation_loader, flowers_data_loader.test_loader,\n                                           grad_clip=grad_clip, \n                                           weight_decay=weight_decay, \n                                           opt_func=opt_func)')


# In[ ]:


train_times.append('2min 20s')


# In[ ]:


T_MODEL = "RESNET-50-Pretrained"
CLASSIFER_LAYER_1 = 2048
CLASSIFER_LAYER_2 = 256
CLASSIFER_LAYER_3 = 64
CLASSIFER_LAYER_4 = 5


# In[ ]:


jovian.log_hyperparams(arch=f"{T_MODEL} --> Classifer layers: ({CLASSIFER_LAYER_1}, {CLASSIFER_LAYER_2}, {CLASSIFER_LAYER_3}, {CLASSIFER_LAYER_4})", 
                       lrs=lrs, 
                       epochs=epochs_list,
                       times=train_times,
                       img_dimensions=IMG_DIM,
                       batch_size=BATCH_SIZE,
                       validation_fraction=VALIDATION_FRACTION,
                       test_fraction=TEST_FRACTION
                       )


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


def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');


# In[ ]:


plot_lrs(history)


# ---

# ## Now lets finally run the model on the test set and see the results

# In[ ]:


history.append(model.evaluate_test_set(flowers_data_loader.test_loader))


# In[ ]:


history[-1]


# In[ ]:


def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    print("The prediction tensor is:")
    for i in range(len(flowers_data_loader.classes)):
        print(f"{flowers_data_loader.classes[i]:^10}"," : ",F.softmax(yb, dim=1)[0][i].item())
    # Retrieve the class label
    return flowers_data_loader.classes[preds[0].item()]


# In[ ]:


for imgs, labels in flowers_data_loader.test_loader:
    for img, label in zip(imgs[:1], labels[:1]):
        plt.imshow(img.cpu().permute(1, 2, 0))
        print('Label:', flowers_data_loader.classes[label.item()], ', Predicted:', predict_image(img.cpu(), model))
    break


# In[ ]:


for imgs, labels in flowers_data_loader.test_loader:
    for img, label in zip(imgs[:1], labels[:1]):
        plt.imshow(img.cpu().permute(1, 2, 0))
        print('Label:', flowers_data_loader.classes[label.item()], ', Predicted:', predict_image(img.cpu(), model))
    break


# In[ ]:


for imgs, labels in flowers_data_loader.test_loader:
    for img, label in zip(imgs[:1], labels[:1]):
        plt.imshow(img.cpu().permute(1, 2, 0))
        print('Label:', flowers_data_loader.classes[label.item()], ', Predicted:', predict_image(img.cpu(), model))
    break


# In[ ]:


for imgs, labels in flowers_data_loader.test_loader:
    for img, label in zip(imgs[:1], labels[:1]):
        plt.imshow(img.cpu().permute(1, 2, 0))
        print('Label:', flowers_data_loader.classes[label.item()], ', Predicted:', predict_image(img.cpu(), model))
    break


# In[ ]:


for imgs, labels in flowers_data_loader.test_loader:
    for img, label in zip(imgs[:1], labels[:1]):
        plt.imshow(img.cpu().permute(1, 2, 0))
        print('Label:', flowers_data_loader.classes[label.item()], ', Predicted:', predict_image(img.cpu(), model))
    break


# In[ ]:


for imgs, labels in flowers_data_loader.test_loader:
    for img, label in zip(imgs[:1], labels[:1]):
        plt.imshow(img.cpu().permute(1, 2, 0))
        print('Label:', flowers_data_loader.classes[label.item()], ', Predicted:', predict_image(img.cpu(), model))
    break


# In[ ]:


for imgs, labels in flowers_data_loader.test_loader:
    for img, label in zip(imgs[:1], labels[:1]):
        plt.imshow(img.cpu().permute(1, 2, 0))
        print('Label:', flowers_data_loader.classes[label.item()], ', Predicted:', predict_image(img.cpu(), model))
    break


# In[ ]:


for imgs, labels in flowers_data_loader.test_loader:
    for img, label in zip(imgs[:1], labels[:1]):
        plt.imshow(img.cpu().permute(1, 2, 0))
        print('Label:', flowers_data_loader.classes[label.item()], ', Predicted:', predict_image(img.cpu(), model))
    break


# ## Now lets try the model on some images not in this dataset

# In[ ]:


other_imgs_path = os.path.join('..', 'input', 'other_imgs')
other_imgs_path


# In[ ]:


other_images = ImageFolder(other_imgs_path, transform=T.Compose(VALIDATION_and_TEST_TRANSFORMS))


# In[ ]:


LEN_OF_OTHER_IMAGES = len(other_images)
LEN_OF_OTHER_IMAGES


# In[ ]:


other_images.classes


# In[ ]:


def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    print("The prediction tensor is:")
    for i in range(len(flowers_data_loader.classes)):
        print(f"{flowers_data_loader.classes[i]:^10}"," : ",F.softmax(yb, dim=1)[0][i].item())
    # Retrieve the class label
    return flowers_data_loader.classes[preds[0].item()]


# In[ ]:


img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', other_images.classes[label], ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', flowers_data_loader.classes[label], ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', flowers_data_loader.classes[label], ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', flowers_data_loader.classes[label], ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', flowers_data_loader.classes[label], ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', flowers_data_loader.classes[label], ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', flowers_data_loader.classes[label], ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', flowers_data_loader.classes[label], ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', flowers_data_loader.classes[label], ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = other_images[random_no_gen(LEN_OF_OTHER_IMAGES)]
plt.imshow(img.permute(1, 2, 0))
print('Label:', flowers_data_loader.classes[label], ', Predicted:', predict_image(img, model))


# In[ ]:


jovian.log_metrics(test_loss=history[-1]['test_loss'],
                   test_acc=history[-1]['test_acc'])


# In[ ]:


jovian.commit(project=PROJECT_NAME, environment=None)


# ## Saving and Loading the model for later:
# 
# Since we've trained our model for a long time and achieved a resonable accuracy, it would be a good idea to save the weights of the model to disk, so that we can reuse the model later and avoid retraining from scratch.

# In[ ]:


torch.save(model.state_dict(), 'flowers_cnn.pth')


# In case you forget, the `.state_dict` method returns an **OrderedDict** containing all the weights and bias matrices mapped to the right attributes of the model. To load the model weights, we can redefine the model with the same structure, and use the `.load_state_dict` method.

# In[ ]:


sanity_check_model = to_device(ResNet50(num_classes=5), device)


# In[ ]:


sanity_check_model.load_state_dict(torch.load('flowers_cnn.pth'))


# Just as a sanity check, let's verify that this model has the same loss and accuracy on the test set as before.

# In[ ]:


sanity_check_model.evaluate_test_set(flowers_data_loader.test_loader)


# Let's make one final commit using jovian, just to be sure everything is committed.

# In[ ]:


jovian.commit(project=PROJECT_NAME, outputs=['flowers_cnn.pth'], environment=None)


# Check out the **Files tab** on the project page to view or download the trained model weights. You can also download all the files together using the *Download Zip* option in the *Clone* dropdown.

# # Summary & Future Work
# 

# So at the end of this notebook, we can see that we have quite good results on a dataset that is not very clean and whose images are not regular in size, as was visible by the large black borders around many of the images. This shows the power of CNNs and transfer learning as well as picking a good classifier layer for the task.
# 
# In terms of future work, I would love to scrape and collect more images of the flowers in this dataset as well as images of flowers not it this dataset from various websites and other online sources, in an attempt to make the model more generalizable.
# 
# The uses of a such a model are quite a lot, it can be used by people visiting botanical garderns and flower shows alike, as many at times people may not know what flower they are looking at. Rather than doing pointless Google searches, they can instead use their camera which can send the video output to the model which can then figure out which flower is being presented to it and show the result text on top of the image itself as an overlay on the user's screen.
