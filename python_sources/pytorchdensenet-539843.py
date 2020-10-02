#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("/kaggle/input/test"))
get_ipython().system('pip install efficientnet_pytorch')
get_ipython().system('pip install torchsummary')
from efficientnet_pytorch import EfficientNet
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
import torch.optim as optim
import copy
import os
import torch
from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install googledrivedownloader')


# In[ ]:


'''from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1DSTHrFxJF4Xq7Jyu1ZdTZMs1NuRLelQ2',
                                    dest_path='/kaggle/working/cactuseff_2.h5')'''


# In[ ]:


from google_drive_downloader import GoogleDriveDownloader as gdd
gdd.download_file_from_google_drive(file_id='1_O-7ypeXY381lP6vaSQBeGDr6x5Y6VCN',
                                    dest_path='/kaggle/working/cactusdense_3.h5')


# In[ ]:


#train_csv = pd.read_csv('/kaggle/input/train.csv')
#train_csv.head(10)


# In[ ]:


'''from sklearn.model_selection import train_test_split
class cactus_dataset(Dataset):
  def __init__(self,image_dir,train_csv,transform = None):
    self.img_dir = image_dir
    self.transform = transform
    self.id = train_csv.iloc[:,0]
    self.classes =  train_csv.iloc[:,1]
  def __len__(self):
    return len(self.id)
  def __getitem__(self,idx):
    img_name = os.path.join(self.img_dir, self.id[idx])
    image = cv2.imread(img_name)
    if self.transform:
        image = self.transform(image)
    label = self.classes[idx]
    return image,label
'''


# In[ ]:


'''batch_size = 8
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                    
                                        transforms.RandomResizedCrop(224),                                    
                                        transforms.RandomHorizontalFlip(),
                                        #transforms.RandomRotation(30),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

#inverse normalization for image plot
train_data = cactus_dataset('/kaggle/input/train/train',train_csv,transform = train_transforms)
#val_data = cactus_dataset('/kaggle/input/train/train',val_df,transform = test_transforms)
train_loader = DataLoader(train_data, batch_size=8,
                        shuffle=True, num_workers=0)

#val_loader = DataLoader(val_data, batch_size=4,shuffle=True, num_workers=0)
dataloaders = {'train':train_loader}
'''


# In[ ]:


import torchvision
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
#from torchsummary import summary
import torch.optim as optim
import copy
import os
import torch
from tqdm.autonotebook import tqdm

import matplotlib.pyplot as plt

class classifie(nn.Module):
    def __init__(self):
        super(classifie, self).__init__()
        model = models.densenet201(pretrained = True)
        model = model.features
        #model = EfficientNet.from_pretrained('efficientnet-b3')
        #model =  nn.Sequential(*list(model.children())[:-3])
        
        self.model = model
        self.linear = nn.Linear(3840, 512)
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.elu = nn.ELU()
        self.out = nn.Linear(512, 2)
        self.bn1 = nn.BatchNorm1d(3840)
        self.dropout2 = nn.Dropout(0.2)
    def forward(self, x):
        out = self.model(x)
        avg_pool = nn.functional.adaptive_avg_pool2d(out, output_size = 1)
        max_pool = nn.functional.adaptive_max_pool2d(out, output_size = 1)
        out = torch.cat((avg_pool,max_pool),1)
        batch = out.shape[0]
        out = out.view(batch, -1)
        conc = self.linear(self.dropout2(self.bn1(out)))
        conc = self.elu(conc)
        conc = self.bn(conc)
        conc = self.dropout(conc)
        res = self.out(conc)
        return out


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = classifie().to(device)


# In[ ]:


model.load_state_dict(torch.load('/kaggle/working/cactusdense_3.h5'))


# In[ ]:


'''import torch.optim as optim
import matplotlib.pyplot as plt
import random
from torch.autograd import Variable
import numpy as np
import torch
from torch import nn
import sys
def train(model,dataloaders,device,num_epochs,lr,batch_size,patience):
    phase1 = dataloaders.keys()
    losses = list()
    criterion = nn.CrossEntropyLoss()
    acc = list()
    for epoch in range(num_epochs):
        print('Epoch:',epoch)
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay = 1e-6)
        lr = lr*0.9
        for phase in phase1:
            epoch_metrics = {"loss": [], "acc": []}
            if phase == ' train':
                model.train()
            else:
                model.eval()
            for  batch_idx, (data, target) in enumerate(dataloaders[phase]):
                data, target = Variable(data), Variable(target)
                data = data.type(torch.FloatTensor).to(device)
                target = target.type(torch.LongTensor).to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                target = target.type(torch.LongTensor).to(device)

                acc = 100 * (output.detach().argmax(1) == target).cpu().numpy().mean()
                epoch_metrics["loss"].append(loss.item())
                epoch_metrics["acc"].append(acc)
                if(phase =='train'):
                    loss.backward()
                    optimizer.step()
                sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)]"
                % (
                    epoch,
                    num_epochs,
                    batch_idx,
                    len(dataloaders[phase]),
                    loss.item(),
                    np.mean(epoch_metrics["loss"]),
                    acc,
                    np.mean(epoch_metrics["acc"]),
                    )
                )
               
            epoch_acc = np.mean(epoch_metrics["acc"])
            epoch_loss = np.mean(epoch_metrics["loss"])
        print('')  
        print('{} Accuracy: {}'.format(phase,epoch_acc.item()))
    return losses,acc

def train_model(model,dataloaders,encoder,lr_scheduler = None,inv_normalize = None,num_epochs=10,lr=0.0001,batch_size=8,patience = None,classes = None):
    dataloader_train = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = list()
    accuracy = list()
    key = dataloaders.keys()
    perform_test = False
    for phase in key:
        if(phase == 'test'):
            perform_test = True
        else:
            dataloader_train.update([(phase,dataloaders[phase])])
    losses,accuracy = train(model,dataloader_train,device,num_epochs,lr,batch_size,patience)'''


# In[ ]:


#import cv2
#lr = 0.001
#train_model(classifier,dataloaders,encoder,inv_normalize = None,num_epochs=4,lr = lr,batch_size = batch_size,patience = None,classes = classes)


# In[ ]:


class cactus_dataset_test(Dataset):
  def __init__(self,image_dir,transform = None):
    self.img_dir = image_dir
    self.transform = transform
    self.id = os.listdir(image_dir)
  def __len__(self):
    return len(self.id)
  def __getitem__(self,idx):
    img_name = os.path.join(self.img_dir, self.id[idx])
    image = cv2.imread(img_name)
    if self.transform:
        image = self.transform(image)
    return (self.id[idx],image)


# In[ ]:


import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
test_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


# In[ ]:


test1 = cactus_dataset_test('/kaggle/input/test/test',test_transforms)


# In[ ]:


test_loader = DataLoader(test1, batch_size =32, shuffle = True)


# In[ ]:


def test(model,dataloader,device,batch_size):
    running_corrects = 0
    running_loss=0
    pred = []
    id = list()
    sm = nn.Softmax(dim = 1)
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (id_1,data) in enumerate(dataloader):
        data = Variable(data)
        data = data.type(torch.FloatTensor).to(device)
        model.eval()
        output = model(data)
        #output = sm(output)
        _, preds = torch.max(output, 1)
        preds = preds.cpu().numpy()
        preds = np.reshape(preds,(len(preds),1))
        
        for i in range(len(preds)):
            pred.append(preds[i])
            id.append(id_1[i])
    return id,pred


# In[ ]:


import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import torch
from torch import nn
id,pred = test(model,test_loader,'cuda',32)


# In[ ]:


a = list()
for i in range(len(pred)):
    a.append(pred[i][0])


# In[ ]:


a = np.asarray(a)


# In[ ]:


a = np.reshape(a,(-1,1))


# In[ ]:


b = np.asarray(id)


# In[ ]:


b = np.reshape(b,(-1,1))


# In[ ]:


sub = np.concatenate((b,a),axis = 1)


# In[ ]:


sub_df = pd.DataFrame(sub)


# In[ ]:


sub_df.columns = ['id','has_cactus']


# In[ ]:


sub_df.head(10)


# In[ ]:


sub_df.to_csv("/kaggle/working/submission.csv", index=False)


# In[ ]:




