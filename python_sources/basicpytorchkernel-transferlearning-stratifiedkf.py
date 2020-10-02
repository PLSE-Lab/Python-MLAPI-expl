#!/usr/bin/env python
# coding: utf-8

# # This is a basic kernel designed on giving a head start for those who are begineer in Pytorch.
# **Please give a upvote so that I could get a motivation to contribute more.By the way thanks for reading this.**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join
from PIL import Image
import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
from sklearn.model_selection import StratifiedKFold
from torch import nn,optim
import random
import os
get_ipython().system('pip install barbar')
from barbar import Bar
get_ipython().system('pip install albumentations')
import albumentations as aug


# **Seeding is very necessary to produce the same result.This code seeds *everything* .So no need to explicitly define seed on and on.**

# In[ ]:


#Seed everything at ones for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(17)


# In[ ]:


Input_Sizes = 224
Num_Classes = 4
data_dir = '../input/plant-pathology-2020-fgvc7/'
labels = pd.read_csv(join(data_dir, 'train.csv'))


# ****When we have a small amount of data,It requires a wise decision to split our data in train and validation set.
# If we use sklearn's train-test-split,It will randomly split the dataset ,but it will not train on the validation set later which implies that
# we have reduced our dataset even more.Moreover it doesnt care for even distribution of Dataset.To overcome this I have used Stratified k fold.****

# In[ ]:


#Now we will perform StratifiedKFold with splits=5
skf=StratifiedKFold(n_splits=5,random_state=17,shuffle=True)


# **Data Augmentation is a primary task which helps our model generalize better.A wise augmentation can even boost your test accuracy.**

# In[ ]:


def get_transform(phase):
    list_transforms =[]
    if phase =='train':
        list_transforms.extend([
            aug.Flip(),
            aug.Resize(256,256),
            aug.RandomCrop(Input_Sizes,Input_Sizes),
            aug.OneOf([
                aug.RandomContrast(),
                aug.RandomGamma(),
                aug.RandomBrightness()],p=1),
            aug.ShiftScaleRotate(rotate_limit=90),
            aug.OneOf([
                aug.GaussNoise(p=0.35),
                aug.IAASharpen()],p=0.5)])
    list_transforms.extend([
        aug.Resize(Input_Sizes,Input_Sizes),
        aug.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225],p=1)
        ])
    list_trfms=aug.Compose(list_transforms)
    return list_trfms


# In[ ]:


#Now lets make a Dataset Class
class Dats(Dataset):
    def __init__(self,labels,root_dir,phase):
        self.labels=labels
        self.root_dir=root_dir
        self.phase=phase
        self.transform=get_transform(phase)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,idx):
        img_name = '{}.jpg'.format(self.labels.iloc[idx, 0])
        fullname = join(self.root_dir, img_name)
        image = Image.open(fullname)
        labels = self.labels.iloc[idx, 1:].to_numpy().astype('float')
        labels = np.argmax(labels)
        img=self.transform(image=np.array(image))
        img=img['image']
        img=np.transpose(img,(2,0,1)).astype(np.float32)
        img=torch.tensor(img,dtype=torch.float)
        return (img,labels)


# ****Stratified K fold requires X,y pair but splits on the basis of y(predictions) so as to maintain uniform 
# distribution****

# In[ ]:


maximas=np.argmax(np.array(labels.drop('image_id',axis=1)),axis=1)
maximas


# In[ ]:


#Now lets make split of labels according to Stratified k fold:
Dict={}
for i,(train_idx,val_idx) in enumerate(skf.split(labels,maximas)):
    Dict['train'+str(i+1)]=labels.iloc[train_idx].reset_index().drop('index',axis=1)
    Dict['val'+str(i+1)]=labels.iloc[val_idx].reset_index().drop('index',axis=1)
    Dict['train'+str(i+1)].to_csv('train'+str(i+1)+'.csv')
    Dict['val'+str(i+1)].to_csv('val'+str(i+1)+'.csv')


# **Now lets make Dataset class object.**

# In[ ]:


dset={}
for i in range(5):
    dset['train'+str(i+1)]=Dats(Dict['train'+str(i+1)],data_dir+'images/',phase='train')
    dset['val'+str(i+1)]=Dats(Dict['val'+str(i+1)],data_dir+'images/',phase='val')


# **Dataloader for loading Data**

# In[ ]:


Dloader={}
for i in range(5):
    Dloader['d'+str(i+1)]={'train':torch.utils.data.DataLoader(dset['train1'],batch_size=24,shuffle=True,num_workers=4,pin_memory=True),
                          'val':torch.utils.data.DataLoader(dset['val1'],batch_size=24,shuffle=True,num_workers=4,pin_memory=True)}


# In[ ]:


#Now lets view our image:
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# In[ ]:


load=Dloader['d1']['val']
image,label=next(iter(load))
out = torchvision.utils.make_grid(image)
imshow(out)


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# **Use of pretrained model.**

# In[ ]:


resnet = models.resnext101_32x8d(pretrained=True).to(device)


# **Replace the output layer according to our classes**

# In[ ]:


#Now replace the outer layer:
fc_inputs=resnet.fc.in_features
for param in resnet.parameters():
    param.requires_grad=False
resnet.fc=nn.Linear(fc_inputs,Num_Classes).to(device)


# In[ ]:


criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam([parameters for parameters in resnet.parameters() if parameters.requires_grad],lr=0.002)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# In[ ]:


def train_model(dataloaders_dict, model, criterion, optimizer, scheduler, num_epochs=15):
    since =time.time()
    best_acc=0.0
    c=0
    #scheduler
    schedule=scheduler
    for dc in dataloaders_dict.values():
            print('Running dataloader part: {} '.format(c))
            c+=1
            dataset_sizes = {'train': len(dc['train'].dataset), 
                     'val': len(dc['val'].dataset)}
            scheduler=schedule #Since after every iteration lr is decreasing
            for epoch in range(num_epochs):
                for phase in ['train','val']:
                    if phase=='train':
                        scheduler.step()
                        model.train(True)
                    else:
                        model.train(False)
                    running_loss=0.0
                    running_correct=0
                    for i,(inputs,labels) in enumerate(Bar(dc[phase])):
                        
                        
                        inputs,labels=Variable(inputs.to(device)),Variable(labels.to(device))

                        optimizer.zero_grad()

                        outputs=model(inputs)
                        _,preds=torch.max(outputs.data,1)
                        loss=criterion(outputs,labels)

                        if phase=='train':
                            loss.backward()
                            optimizer.step()
                        running_loss+=(loss.data).item()
                        running_correct+=torch.sum(preds==labels.data)
                    if phase=='train':
                        train_epoch_loss=float(running_loss)/dataset_sizes[phase]
                        train_epoch_acc=float(running_correct)/dataset_sizes[phase]
                    else:
                        valid_epoch_loss=float(running_loss)/dataset_sizes[phase]
                        valid_epoch_acc=float(running_correct)/dataset_sizes[phase]
            
                print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} ' 
                  'valid loss: {:.4f} acc: {:.4f}'.format(
                    epoch, num_epochs - 1,
                    train_epoch_loss, train_epoch_acc, 
                    valid_epoch_loss, valid_epoch_acc))
    return model


# In[ ]:


start_time = time.time()
model = train_model(Dloader,resnet,criterion,optimizer,exp_lr_scheduler,num_epochs=10)
print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))


# In[ ]:


submission_df = pd.read_csv(join(data_dir, 'sample_submission.csv'))
output_df = pd.DataFrame(index=submission_df.index, columns=submission_df.keys() )
output_df['image_id'] = submission_df['image_id']
submission_ds = Dats(submission_df,data_dir+'images/',phase='test')

sub_loader = DataLoader(submission_ds, batch_size=24,
                        shuffle=False, num_workers=4)
def test_sub(model):
    since = time.time()
    sub_outputs = []
    model.train(False)  # Set model to evaluate mode
    # Iterate over data.
    for data in sub_loader:
        # get the inputs
        inputs, labels = data

        inputs = Variable(inputs.to(device))
        labels = Variable(labels.to(device))

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        sub_outputs.append(outputs.data.cpu().numpy())

    sub_outputs = np.concatenate(sub_outputs)
    for idx,row in enumerate(sub_outputs.astype(float)):
        sub_outputs[idx] = np.exp(row)/np.sum(np.exp(row))
        
    output_df.loc[:,1:] = sub_outputs
        
    print()
    time_elapsed = time.time() - since
    print('Run complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return output_df


# In[ ]:


odf = test_sub(model)
odf.to_csv("plants_resnext1018d.csv", index=False)


# In[ ]:




