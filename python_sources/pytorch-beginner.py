#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
os.listdir('../input/histopathologic-cancer-detection')
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


path2csv='../input/histopathologic-cancer-detection/train_labels.csv'
labels_df=pd.read_csv(path2csv)


# In[ ]:


labels_df.head(5)


# In[ ]:


print(labels_df['label'].value_counts())


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
labels_df['label'].hist();


# In[ ]:


import matplotlib.pylab as plt
from PIL import Image, ImageDraw
import numpy as np
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# get ids for malignant images
malignantIds = labels_df.loc[labels_df['label']==1]['id'].values


# In[ ]:


path2train="../input/histopathologic-cancer-detection/train"


# In[ ]:


color=False


# In[ ]:


plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.subplots_adjust(wspace=0, hspace=0)
nrows,ncols=3,3


# In[ ]:


for i,id_ in enumerate(malignantIds[:nrows*ncols]):
    full_filenames = os.path.join(path2train , id_ +'.tif')
    # load image
    img = Image.open(full_filenames)
    # draw a 32*32 rectangle
    draw = ImageDraw.Draw(img)
    draw.rectangle(((32, 32), (64, 64)),outline="green")
    plt.subplot(nrows, ncols, i+1)
    if color is True:
        plt.imshow(np.array(img))
    else:
        plt.imshow(np.array(img)[:,:,0],cmap="gray")
    plt.axis('off')


# In[ ]:


print("image shape:", np.array(img).shape)
print("pixel values range from %s to %s" %(np.min(img),
np.max(img)))


# We will define a class for the custom dataset, define the transformation function, and then load an image from the dataset using the Dataset class. 

# In[ ]:


from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
import os


# In[ ]:


# fix torch random seed
torch.manual_seed(0)


# In[ ]:


class histoCancerDataset(Dataset):
    def __init__(self, data_dir, transform,data_type="train"):
        # path to images
        path2data = os.path.join(data_dir,data_type)
        # get a list of images
        filenames = os.listdir(path2data)
        # get the full path to images
        self.full_filenames = [os.path.join(path2data, f) for f in filenames]
        # labels are in a csv file named train_labels.csv
        csv_filename=data_type+"_labels.csv"
        path2csvLabels=os.path.join(data_dir,csv_filename)
        labels_df=pd.read_csv(path2csvLabels)
        # set data frame index to id
        labels_df.set_index("id", inplace=True)
        # obtain labels from data frame
        self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in filenames]
        self.transform = transform
        
        
    def __len__(self):
        # return size of dataset
        return len(self.full_filenames)
    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx]) # PIL image
        image = self.transform(image)
        return image, self.labels[idx]


# In[ ]:


import torchvision.transforms as transforms
data_transformer = transforms.Compose([transforms.ToTensor()])


# In[ ]:


data_dir = "../input/histopathologic-cancer-detection"
histo_dataset = histoCancerDataset(data_dir, data_transformer,"train")
print(len(histo_dataset))


# Next, we will load an image using the custom dataset:

# In[ ]:


# load an image
img,label=histo_dataset[256]
print(img.shape,torch.min(img),torch.max(img))


# We need to provide a validation dataset to track the model's performance during training. We use 20% of histo_dataset as the validation dataset and use the rest as the training dataset.

# In[ ]:


from torch.utils.data import random_split
len_histo=len(histo_dataset)
len_train=int(0.8*len_histo)
len_val=len_histo-len_train
train_ds,val_ds=random_split(histo_dataset,[len_train,len_val])
print("train dataset length:", len(train_ds))
print("validation dataset length:", len(val_ds))


# In[ ]:


for x,y in train_ds:
    print(x.shape,y)
    break
for x,y in val_ds:
    print(x.shape,y)
    break


# Let's display a few samples from train_ds.

# In[ ]:


#Import the required packages:
from torchvision import utils
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(0)


# In[ ]:


#helper function
def show(img,y,color=True):
    # convert tensor to numpy array
    npimg = img.numpy()
    # Convert to H*W*C shape
    npimg_tr=np.transpose(npimg, (1,2,0))
    if color==False:
        npimg_tr=npimg_tr[:,:,0]
        plt.imshow(npimg_tr,interpolation='nearest',cmap="gray")
    else:
        # display images
        plt.imshow(npimg_tr,interpolation='nearest')
    plt.title("label: "+str(y))


# In[ ]:


#making grid 
grid_size=4
rnd_inds=np.random.randint(0,len(train_ds),grid_size)
print("image indices:",rnd_inds)
x_grid_train=[train_ds[i][0] for i in rnd_inds]
y_grid_train=[train_ds[i][1] for i in rnd_inds]
x_grid_train=utils.make_grid(x_grid_train, nrow=4, padding=2)
print(x_grid_train.shape)
plt.rcParams['figure.figsize'] = (10.0, 5)
show(x_grid_train,y_grid_train)


# Augmentation of data to trainset and other transformation 

# In[ ]:


train_transformer = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(45),
    transforms.RandomResizedCrop(96,scale=(0.8,1.0),ratio=(1.0,1.0)),
    transforms.ToTensor()])
#we do not distort validation dataset except for convetring it to tensor
val_transformer = transforms.Compose([transforms.ToTensor()])


# In[ ]:


#overwrite the transform functions
train_ds.transform=train_transformer
val_ds.transform=val_transformer


#  creating dataloaders

# In[ ]:


from torch.utils.data import DataLoader
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)


# In[ ]:


# extract a batch from training data
for x, y in train_dl:
    print(x.shape)
    print(y.shape)
    break
for x, y in val_dl:
    print(x.shape)
    print(y.shape)
    break


# Building the CLASSIFICATION MODEL

# First, let's create dumb baselines for the validation dataset.

# In[ ]:


# get labels for validation dataset
y_val=[y for _,y in val_ds]


# In[ ]:


def accuracy(labels, out):
    return np.sum(out==labels)/float(len(labels))


# In[ ]:


# accuracy all zero predictions
acc_all_zeros=accuracy(y_val,np.zeros_like(y_val))
print("accuracy all zero prediction: %.2f" %acc_all_zeros)


# In[ ]:


# accuracy all ones predictions
acc_all_ones=accuracy(y_val,np.ones_like(y_val))
print("accuracy all one prediction: %.2f" %acc_all_ones)


# In[ ]:


# accuracy random predictions
acc_random=accuracy(y_val,np.random.randint(2,size=len(y_val)))
print("accuracy random prediction: %.2f" %acc_random)


# In[ ]:


#now the full cnn model
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# In[ ]:


#helper function to give output size
def findConv2dOutShape(H_in,W_in,conv,pool=2):
    # get conv arguments
    kernel_size=conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation
    H_out=np.floor((H_in+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    W_out=np.floor((W_in+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)
    if pool:
        H_out/=pool
        W_out/=pool
    return int(H_out),int(W_out)


# In[ ]:


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        C_in,H_in,W_in=params["input_shape"]
        init_f=params["initial_filters"]
        num_fc1=params["num_fc1"]
        num_classes=params["num_classes"]
        self.dropout_rate=params["dropout_rate"]
        self.conv1 = nn.Conv2d(C_in, init_f, kernel_size=3)
        h,w=findConv2dOutShape(H_in,W_in,self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv2)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv3)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv4)
        # compute the flatten size
        self.num_flatten=h*w*8*init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.num_flatten)
        x = F.relu(self.fc1(x))
        x=F.dropout(x, self.dropout_rate, training= self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    


# In[ ]:


# dict to define model parameters
params_model={
    "input_shape": (3,96,96),
    "initial_filters": 8,
    "num_fc1": 100,
    "dropout_rate": 0.25,
    "num_classes": 2,
    }


# In[ ]:


# create model
cnn_model = Net(params_model)


# In[ ]:


if torch.cuda.is_available():
    device=torch.device("cuda")
    cnn_mdel = cnn_model.to(device) 


# In[ ]:


print(cnn_model)


# In[ ]:


pip install torchsummary


# In[ ]:


from torchsummary import summary
summary(cnn_model, input_size=(3, 96, 96),device=device.type)


# In[ ]:


loss_func = nn.NLLLoss(reduction="sum")


# In[ ]:


from torch import optim
opt = optim.Adam(cnn_model.parameters(), lr=3e-4)


# In[ ]:


# get learning rate
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']
current_lr=get_lr(opt)
print('current lr={}'.format(current_lr))


# In[ ]:


from torch.optim.lr_scheduler import ReduceLROnPlateau
# define learning rate scheduler
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5,
patience=20,verbose=1)
for i in range(100):
    lr_scheduler.step(1)


# # **Training and evaluation of the model**

# In[ ]:


def metrics_batch(output, target):
    # get output class
    pred = output.argmax(dim=1, keepdim=True)
    # compare output class with target class
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects


# In[ ]:


#loss value per batch
def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    with torch.no_grad():
        metric_b = metrics_batch(output,target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b


# In[ ]:


#loss per epoch
def loss_epoch(model,loss_func,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    running_metric=0.0
    len_data=len(dataset_dl.dataset)
    
    for xb, yb in dataset_dl:
        # move batch to device
        xb=xb.to(device)
        yb=yb.to(device)
        # get model output
        output=model(xb)
        # get loss per batch
        loss_b,metric_b=loss_batch(loss_func, output, yb, opt)
        
        # update running loss
        running_loss+=loss_b
        # update running metric
        if metric_b is not None:
            running_metric+=metric_b
        # break the loop in case of sanity check
        if sanity_check is True:
            break
            
    # average loss value
    loss=running_loss/float(len_data)
    # average metric value
    metric=running_metric/float(len_data)
    return loss, metric


# In[ ]:


def train_val(model, params):
    # extract model parameters
    num_epochs=params["num_epochs"]
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]
    
    
    # history of loss values in each epoch
    loss_history={
    "train": [],
    "val": [],
        }
    # history of metric values in each epoch
    metric_history={
    "train": [],
    "val": [],
        }
    # a deep copy of weights for the best performing model
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # initialize best loss to a large value
    best_loss=float('inf')
    
    #we will define a loop that will calculate the training loss over an epoch:
    # main loop
    for epoch in range(num_epochs):
        # get current learning rate
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs- 1, current_lr))
        # train model on training dataset
        model.train()
        train_loss,train_metric=loss_epoch(model,loss_func,train_dl,sanity_check,opt)
        # collect loss and metric for training dataset
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        
        # evaluate model on validation dataset
        model.eval()
        with torch.no_grad():
            val_loss,val_metric=loss_epoch(model,loss_func,val_dl,sanity_check)
    
        # collect loss and metric for validation dataset
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
    
    
        # store best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            # store weights into a local file
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")
        # learning rate schedule
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)
        
            
        print("train loss: %.6f, dev loss: %.6f, accuracy: %.2f"%(train_loss,val_loss,100*val_metric))
        print("-"*10)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history
        


# In[ ]:


import copy
loss_func = nn.NLLLoss(reduction="sum")
opt = optim.Adam(cnn_model.parameters(), lr=3e-4)
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5,
patience=20,verbose=1)


# In[ ]:


params_train = {
    "num_epochs": 100,
    "optimizer": opt,
    "loss_func": loss_func,
    "train_dl": train_dl,
    "val_dl": val_dl,
    "sanity_check": True,
    "lr_scheduler": lr_scheduler,
    "path2weights": "weights.pt",
    
} 


# In[ ]:


# train and validate the model
cnn_model,loss_hist,metric_hist=train_val(cnn_model,params_train)


# In[ ]:


# Train-Validation Progress
num_epochs=params_train["num_epochs"]
# plot loss progress
plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()
# plot accuracy progress
plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),metric_hist["val"],label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.grid()
plt.show()


# # **Deploying the model**

# In[ ]:


#First, we'll create an object of the Net class and load the stored weights into the model
# model parameters
params_model={
    "input_shape": (3,96,96),
    "initial_filters": 8,
    "num_fc1": 100,
    "dropout_rate": 0.25,
    "num_classes": 2,
}


# In[ ]:


# initialize model
cnn_model = Net(params_model)


# In[ ]:


# load state_dict into model
# load state_dict into model
path2weights="weights.pt"
cnn_model.load_state_dict(torch.load(path2weights))


# In[ ]:


cnn_model.eval()


# In[ ]:


# move model to cuda/gpu device
if torch.cuda.is_available():
    device = torch.device("cuda")
    cnn_model=cnn_model.to(device)


# In[ ]:


import time
def deploy_model(model,dataset,device,num_classes=2,sanity_check=False):
    len_data=len(dataset)
    # initialize output tensor on CPU: due to GPU memory limits
    y_out=torch.zeros(len_data,num_classes)
    # initialize ground truth on CPU: due to GPU memory limits
    y_gt=np.zeros((len_data),dtype="uint8")
    # move model to device
    model=model.to(device)
    elapsed_times=[]
    with torch.no_grad():
        for i in range(len_data):
            x,y=dataset[i]
            y_gt[i]=y
            start=time.time()
            y_out[i]=model(x.unsqueeze(0).to(device))
            elapsed=time.time()-start
            elapsed_times.append(elapsed)
            if sanity_check is True:
                break
    inference_time=np.mean(elapsed_times)*1000
    print("average inference time per image on %s: %.2f ms "%(device,inference_time))
    return y_out.numpy(),y_gt
            


# In[ ]:


# deploy model
y_out,y_gt=deploy_model(cnn_model,val_ds,device=device,sanity_check=False)
print(y_out.shape,y_gt.shape)


# In[ ]:


from sklearn.metrics import accuracy_score
# get predictions
y_pred = np.argmax(y_out,axis=1)
print(y_pred.shape,y_gt.shape)
# compute accuracy
acc=accuracy_score(y_pred,y_gt)
print("accuracy: %.2f" %acc)


# # **Applying on ths test set**

# In[ ]:


path2csv="./../input/histopathologic-cancer-detection/sample_submission.csv"
labels_df=pd.read_csv(path2csv)
labels_df.head()


# In[ ]:


class histoCancerDataset_test(Dataset):
    def __init__(self, data_dir, transform,data_type="train"):
        # path to images
        path2data = os.path.join(data_dir,data_type)
        # get a list of images
        filenames = os.listdir(path2data)
        # get the full path to images
        self.full_filenames = [os.path.join(path2data, f) for f in filenames]
        # labels are in a csv file named train_labels.csv
        csv_filename="sample_submission.csv"
        path2csvLabels=os.path.join(data_dir,csv_filename)
        labels_df=pd.read_csv(path2csvLabels)
        # set data frame index to id
        labels_df.set_index("id", inplace=True)
        # obtain labels from data frame
        self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in filenames]
        self.transform = transform
        
        
    def __len__(self):
        # return size of dataset
        return len(self.full_filenames)
    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx]) # PIL image
        image = self.transform(image)
        return image, self.labels[idx]


# In[ ]:


histo_test = histoCancerDataset_test(data_dir,val_transformer,data_type="test")
print(len(histo_test))


# In[ ]:


y_test_out,_=deploy_model(cnn_model,histo_test, device,sanity_check=False)



# In[ ]:


y_test_pred=np.argmax(y_test_out,axis=1)
print(y_test_pred.shape)


# In[ ]:


grid_size=4
rnd_inds=np.random.randint(0,len(histo_test),grid_size)
print("image indices:",rnd_inds)
x_grid_test=[histo_test[i][0] for i in range(grid_size)]
y_grid_test=[y_test_pred[i] for i in range(grid_size)]
x_grid_test=utils.make_grid(x_grid_test, nrow=4, padding=2)
print(x_grid_test.shape)
plt.rcParams['figure.figsize'] = (10.0, 5)
show(x_grid_test,y_grid_test)


# In[ ]:


print(y_test_out.shape)
cancer_preds = np.exp(y_test_out[:, 1])
print(cancer_preds.shape)


# In[ ]:




