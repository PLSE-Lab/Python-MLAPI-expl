#!/usr/bin/env python
# coding: utf-8

# # Monkey Species Classification Using CNN & Resnet34

# As part of the Zero to Gans course by Jovian.ml and freecodecamp.org, I worked on a image classification project.
# 
# In this project, I am to build a model that can classify monkeys into the 10 different species.
# 
#             0 : 'mantled_howler',
#             1 : 'patas_monkey',
#             2 : 'bald_uakari',
#             3 : 'japanese_macaque',
#             4 : 'pygmy_marmoset',
#             5 : 'white_headed_capuchin',
#             6 : 'silvery_marmoset',
#             7 : 'common_squirrel_monkey',
#             8 : 'black_headed_night_monkey',
#             9 : 'nilgiri_langur'
#             
#  I used PyTorch, CNN and Resnet34 pretrained model for the classification
#  
#  I obtain over 99% accuracy in around 5 minutes of training! This is the power of using pretrained model and using imagenet normalization.
# 
# *As I felt the model had overfitted as the accuracy was too good. I used a "holdout" set. I pulled 5 images  from the the internet and checked if they matched the predections from the actuals
# even here the accuracy was 100% proving the model is good and can predict images not known to it as well!*

# # Import libraries

# In the start always import the libraries that you feel you may use or need. Over time, build a list of libraries that you use and use it in all the notebooks you are working. You will naturally strat using some of the libraries that you are comfortable with and this will help.

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import cv2
import random
from random import randint
import time


import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image
from scipy import ndimage

import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder

from tqdm.notebook import tqdm

from sklearn.metrics import f1_score


# ## Preparing the Data

# Let us load the required directories and files

# In[ ]:


DATA_DIR = '../input/10-monkey-species'

TRAIN_DIR = DATA_DIR + '/training/training/'                           
VAL_DIR = DATA_DIR + '/validation/validation/'                             
LABEL_FIL=DATA_DIR +"/monkey_labels.txt"


# Create the labels dictionary using the common names of the 10 monkey species

# In[ ]:


labels ={   0 : 'mantled_howler',
            1 : 'patas_monkey',
            2 : 'bald_uakari',
            3 : 'japanese_macaque',
            4 : 'pygmy_marmoset',
            5 : 'white_headed_capuchin',
            6 : 'silvery_marmoset',
            7 : 'common_squirrel_monkey',
            8 : 'black_headed_night_monkey',
            9 : 'nilgiri_langur' }
target_labels=labels


# In[ ]:


# Load the paths to the images in a directory

def load_images_from_folder(folder,only_path = False, label = ""):
    if only_path == False:
        images = []
        file_name=[]
        for filename in os.listdir(folder):
            img = plt.imread(os.path.join(folder,filename))
            
            if img is not None:
                end=filename.find(".")
                file_name.append(file[0:end])
                images.append(img)
                
        return images, file_name
    else:
        path = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder,filename)
            if img_path is not None:
                path.append([label,img_path])
        return path


# In[ ]:


# Load the paths on the images
images = []
path = TRAIN_DIR
for f in os.listdir(path):
    if "jpg" in os.listdir(path+f)[0]:
        images += load_images_from_folder(path+f,True,label = f)
      
    else: 
        for d in os.listdir(path+f):
            images += load_images_from_folder(path+f+"/"+d,True,label = f)
            
                        
# Create a dataframe with the paths and the label for each monkey species
train_df = pd.DataFrame(images, columns = ["monkey_id", "path_img"])
train_len=len(train_df["path_img"])


monkey_label=[]
monkey_name=[]
for i in range(train_len):
    temp=train_df.monkey_id[i][1]
    temp=int(temp)
    
    monkey_label.append(temp)
    monkey_name.append(labels[temp])


train_df['monkey_label'] = monkey_label
train_df['monkey_name'] =monkey_name


train_df.head()




# In[ ]:


# Load the paths on the images for the validation set
images = []
path = VAL_DIR
for f in os.listdir(path):
    if "jpg" in os.listdir(path+f)[0]:
        images += load_images_from_folder(path+f,True,label = f)
      
    else: 
        for d in os.listdir(path+f):
            images += load_images_from_folder(path+f+"/"+d,True,label = f)
            
                        
# Create a dataframe with the paths and the label for each monkey species
val_df = pd.DataFrame(images, columns = ["monkey_id", "path_img"])
val_len=len(val_df["path_img"])


monkey_label=[]
monkey_name=[]
for i in range(val_len):
    temp=val_df.monkey_id[i][1]
    temp=int(temp)
    
    monkey_label.append(temp)
    monkey_name.append(labels[temp])


val_df['monkey_label'] = monkey_label
val_df['monkey_name'] =monkey_name


val_df.head()


# In[ ]:


train_len
print('Number of images in Training file:', train_len)
val_len
print('Number of images in Validation file:', val_len)


no_labels=len(train_df["monkey_id"].unique())
print('Number of Monkey species:', no_labels)


# # Check the distribution of the 10 monkey species

# In[ ]:


bar = train_df["monkey_name"].value_counts(ascending=True).plot.bar(figsize = (30,5))
plt.title("Distribution of the Monkeys", fontsize = 20)
bar.tick_params(labelsize=16)
plt.show()


# In[ ]:


train_df["monkey_name"].value_counts(ascending=False)


# They are not equal but in range between 105 and 122 images for each species.

# # Plot images of 20 monkeys 

# In[ ]:


fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15, 15),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    j=randint(50*(i),50*(i+1))
    ax.imshow(plt.imread(train_df.path_img[j]))
    ax.set_title(train_df.monkey_name[j])
plt.tight_layout()
plt.show()


# We observe they are of different sizes- we will need to resize them to same size
# 
# Backrounds vary for each monkey; some have full bodies and some have part of the bodies
# 
# Also, some have one monkey; others have 2-3 monkeys but of the same species
# 

# # Define classes and functions

# In[ ]:


imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tfms = T.Compose([
    T.Resize((512,512)),        #change size to 512 X 512 
#    T.CenterCrop(256),
#    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#    T.RandomCrop(32, padding=4, padding_mode='reflect'),
    T.RandomHorizontalFlip(), # flip  random images
    T.RandomRotation(10),     # rotate random  images by 10 degrees
    T.ToTensor(),
    T.Normalize(*imagenet_stats,inplace=True),  #normalise to imagaenet stats 
#    T.RandomErasing(inplace=True)
])

valid_tfms = T.Compose([
    T.Resize((512,512)),
#    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize(*imagenet_stats)
])


# In[ ]:


train_ds = ImageFolder ( TRAIN_DIR , transform=train_tfms )
val_ds = ImageFolder ( VAL_DIR , transform=valid_tfms ) 
len(train_ds), len(val_ds)


# In[ ]:


def show_sample(img, target, invert=True):
    if invert:
        plt.imshow(1 - img.permute((1, 2, 0)))
    else:
        plt.imshow(img.permute(1, 2, 0))
  
    print('Labels:',target, target_labels[target])


# We display some images that are transformed- 
# these are resized to a uniform size (512 X 512), random images are flipped or rotated by 10 degrees, they are normalised to imagenet stats as well
# 
# Lets see some images- one inverted and one in original colour

# In[ ]:


show_sample(*train_ds[83])


# In[ ]:


show_sample(*train_ds[228], invert=False)


# # Data holders

# In[ ]:


np.random.seed(42)


# In[ ]:


batch_size = 16


# In[ ]:


train_dl = DataLoader(train_ds, batch_size, shuffle=True, 
                          num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, 
                        num_workers=4, pin_memory=True)


# In[ ]:


def show_batch(dl, invert=True):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(32, 16))
        ax.set_xticks([]); ax.set_yticks([])
        data = 1-images if invert else images
        ax.imshow(make_grid(data, nrow=8).permute(1, 2, 0))
        break


# We see a batch of images - again in both inverted colours and in original colors

# In[ ]:


show_batch(train_dl, invert=True)


# In[ ]:


show_batch(train_dl, invert=False)


# # Model - Transfer Learning

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


class MonkeyResnet34(MonkeyCnnModel):
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


model = MonkeyResnet34()
model


# In[ ]:


for images, labels in train_dl:
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


train_loader = DeviceDataLoader(train_dl, device)
val_loader = DeviceDataLoader(val_dl, device)
to_device(model, device);


# # Training

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


model = to_device(MonkeyResnet34(), device)


# In[ ]:


evaluate(model, val_loader)


# As there are 10 monkey species, a default accuracy of around 10% is expected 
# The evaluate model also shaows an intial "val_acc" of 10% or 0.1

# In[ ]:


num_epochs = 2
lr = 0.0001
opt_func = torch.optim.Adam


# In[ ]:


get_ipython().run_cell_magic('time', '', 'starttime= time.time()\nhistory = fit(num_epochs, lr, model, train_loader, val_loader, opt_func)')


# In[ ]:


lr=lr/10
history += fit(num_epochs, lr, model, train_loader, val_loader, opt_func)


# In[ ]:


#lr=lr/10
#history += fit(num_epochs, lr, model, train_loader, val_loader, opt_func)


# In[ ]:


endtime=time.time()

duration=endtime-starttime
train_time=time.strftime('%M:%S', time.gmtime(duration))


# In[ ]:


train_time


# In[ ]:


def plot_scores(history):
    scores = [x['val_acc'] for x in history]
    plt.plot(scores, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy vs. No. of epochs');


# In[ ]:


plot_scores(history)


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


# # Test HoldOut Set

# As I felt the model had overfitted as the accuracy was too good. I used a "holdout" set. I pulled 5 images  from the the internet and checked if they matched the predections from the actual labels of images that the model has not seen yet
# 

# In[ ]:


image1= "https://jackdrawsanything.com/img/posts/2011/06/27924935-bald-uakari.jpg"#'bald_uakari',
image2= "https://farm4.staticflickr.com/3371/3559083674_a49f4f9992_z.jpg" #'common_squirrel_monkey',
image3= "https://farm4.staticflickr.com/3640/3701492988_366c8f120a_z.jpg" #'nilgiri_langur'
image4= "https://www.activewild.com/wp-content/uploads/2015/11/White-Headed-Capuchin.jpg" #'white_headed_capuchin',
image5= "https://news.janegoodall.org/wp-content/uploads/2016/09/pygmy-marmoset-1440482961tgM-1024x683.jpg" #'pygmy_marmoset',

holdout_labels= {0: "bald_uakari", 1: "common_squirrel_monkey", 2: "nilgiri_langur",
                3:  "white_headed_capuchin", 4: "pygmy_marmoset"}


# In[ ]:


holdout_path = "../input/monkey-holdout/"
holdout_ds = ImageFolder( holdout_path , transform=valid_tfms ) 
len(holdout_ds)


# In[ ]:


img, target = holdout_ds[0]
img.shape


# In[ ]:


def show_predict(img, target):
    plt.imshow(img.permute((1, 2, 0)))
    print('Predicted:',target, target_labels[target])


# In[ ]:


def predict_single(image):
    xb = image.unsqueeze(0)
    xb = to_device(xb, device)
    preds = model(xb)
    prediction = preds[0]
    _,preds = torch.max(prediction, dim=-1)
    preds= torch.IntTensor.item(preds)
    show_predict(image, preds)


# In[ ]:


for i in range(5):
    print("Holdout image", i)
    print("Actual:", holdout_labels[i])
    predict_single(holdout_ds[i][0])


# Even here the accuracy was 100% (5 out of 5) of the images are accurately predicted.
# This proves the model is good and can predict images not known to it as well!

# # Save and Commit

# In[ ]:


weights_fname = 'Monkey-resnet.pth'
torch.save(model.state_dict(), weights_fname)


# In[ ]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[ ]:


import jovian


# In[ ]:


jovian.reset()
jovian.log_hyperparams(arch='resnet34', 
                       epochs=2*num_epochs, 
                       lr=lr*10, 
                       opt=opt_func.__name__)


# In[ ]:


jovian.log_metrics(val_loss=history[-1]['val_loss'], 
                   val_score=history[-1]['val_acc'],
                   train_loss=history[-1]['train_loss'],
                   time=train_time)


# In[ ]:


project_name='Monkey-classification'


# In[ ]:


jovian.commit(project=project_name, environment=None, outputs=[weights_fname])

