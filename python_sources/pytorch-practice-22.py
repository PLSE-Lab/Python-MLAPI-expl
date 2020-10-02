#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import display,HTML
def dhtml(str):
    display(HTML("""<style>
    @import 'https://fonts.googleapis.com/css?family=Smokum&effect=3d';      
    </style><h1 class='font-effect-3d' 
    style='font-family:Smokum; color:#ff5511; font-size:35px;'>
    %s</h1>"""%str))


# [Google Colaboratory Version](https://colab.research.google.com/drive/1PxqVoIvUkv-bYDMTGtCNYNji3-ObqNWz)

# In[ ]:


dhtml('Code Modules, Functions, & Classes')


# In[ ]:


import numpy as np,pandas as pd,pylab as pl
import h5py,torch,time,copy,urllib,zipfile
from torchvision.datasets import CIFAR10 as tcifar10
from torchvision import transforms,utils,models
from torch.utils.data import DataLoader as tdl
from torch.utils.data import Dataset as tds
from torch.utils.data.dataset import Subset
import torch.nn.functional as tnnf
import torch.nn as tnn
import tensorflow.image as timage
from IPython.core.magic import register_line_magic
dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


class TData(tds):
    def __init__(self,x,y):   
        self.x=torch.tensor(x,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.int32)
    def __getitem__(self,index):
        img,lbl=self.x[index],self.y[index]
        return img,lbl
    def __len__(self):
        return self.y.shape[0]


# In[ ]:


@register_line_magic
def display_examples(d):
    if d=='1': loaders=dataloaders
    if d=='2': loaders=dataloaders2
    for images,labels in loaders['valid']:  
        print('Image dimensions: %s'%str(images.shape))
        print('Label dimensions: %s'%str(labels.shape))
        n=np.random.randint(1,50)
        images=np.transpose(images,(0,2,3,1))/2.+.5
        fig=pl.figure(figsize=(10,4))
        for i in range(n,n+5):
            ax=fig.add_subplot(1,5,i-n+1,            xticks=[],yticks=[],title=labels[i].item())
            ax.imshow((images[i]).reshape(img_size,img_size,3))
        break


# In[ ]:


def model_acc(model,data_loader):
    correct_preds,num_examples=0,0    
    for features,targets in data_loader:
        features=features.to(dev)
        targets=targets.to(dev).long()
        logits=model(features)
        _,pred_labels=torch.max(logits,1)
        num_examples+=targets.size(0)
        correct_preds+=(pred_labels==targets).sum()        
    return correct_preds.float()/num_examples*100
def epoch_loss(model,data_loader):
    model.eval()
    curr_loss,num_examples=0.,0
    with torch.no_grad():
        for features,targets in data_loader:
            features=features.to(dev)
            targets=targets.to(dev).long()
            logits=model(features)
            loss=tnnf.cross_entropy(logits,targets,
                                    reduction='sum')
            num_examples+=targets.size(0)
            curr_loss+=loss
        return curr_loss/num_examples


# In[ ]:


dhtml('Data')


# In[ ]:


img_size=64
classes=('plane','car','bird','cat','deer',
          'dog','frog','horse','ship','truck')
random_seed=12; batch_size=128
train_ids=torch.arange(0,44000)
valid_ids=torch.arange(44000,50000)
tr0=(.5,.5,.5)
trans=transforms.Compose([transforms.Resize((img_size,img_size)),
          transforms.ToTensor(),
          transforms.Normalize(tr0,tr0)])
train_valid=tcifar10(root='data',train=True,
                     download=True,
                     transform=trans)
train=Subset(train_valid,train_ids)
valid=Subset(train_valid,valid_ids)
test=tcifar10(root='data',train=False, 
              transform=trans)
dataloaders={'train':tdl(dataset=train,shuffle=True, 
                         batch_size=batch_size), 
             'valid':tdl(dataset=valid,shuffle=True, 
                         batch_size=batch_size),
             'test':tdl(dataset=test,shuffle=True, 
                        batch_size=batch_size)}


# In[ ]:


get_ipython().run_line_magic('display_examples', '1')


# In[ ]:


fpath='https://olgabelitskaya.github.io/' # from my website 
zf='LetterColorImages_123.h5.zip'
input_file=urllib.request.urlopen(fpath+zf)
output_file=open(zf,'wb'); 
output_file.write(input_file.read())
output_file.close(); input_file.close()
zipf=zipfile.ZipFile(zf,'r')
zipf.extractall(''); zipf.close()
f=h5py.File(zf[:-4],'r')
keys=list(f.keys()); print(keys)
x=np.array(f[keys[1]],dtype='float32')
x=timage.resize(x,[img_size,img_size])/255
x=2*np.transpose(x.numpy(),(0,3,1,2))-1
print(x.mean(),x.std())
y=np.array(f[keys[2]],dtype='int32')-1
N=len(y); n=int(.1*N)
shuffle_ids=np.arange(N)
np.random.RandomState(23).shuffle(shuffle_ids)
x,y=x[shuffle_ids],y[shuffle_ids]
x_test,x_valid,x_train=x[:n],x[n:2*n],x[2*n:]
y_test,y_valid,y_train=y[:n],y[n:2*n],y[2*n:]
random_seed=23
train2=TData(x_train,y_train)
valid2=TData(x_valid,y_valid)
test2=TData(x_test,y_test)
dataloaders2={'train':tdl(dataset=train2,shuffle=True, 
                          batch_size=batch_size), 
              'valid':tdl(dataset=valid2,shuffle=True, 
                          batch_size=batch_size),
              'test':tdl(dataset=test2,shuffle=True, 
                         batch_size=batch_size)}


# In[ ]:


get_ipython().run_line_magic('display_examples', '2')


# In[ ]:


dhtml('VGG16')


# In[ ]:


model=models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad=False
model.classifier[3].requires_grad=True
model


# In[ ]:


dhtml('Training')


# In[ ]:


@register_line_magic
def train_run(pars):
    [epochs,n]=pars.split()
    epochs=int(epochs); n=int(n)
    if n==1: loaders=dataloaders
    if n==2: loaders=dataloaders2
    for epoch in range(epochs):
        model.train()
        for batch_ids,(features,targets)         in enumerate(loaders['train']):        
            features=features.to(dev)
            targets=targets.to(dev).long()
            logits=model(features)
            cost=tnnf.cross_entropy(logits,targets)
            optimizer.zero_grad(); cost.backward()
            optimizer.step()
            if not batch_ids%100:
                print ('Epoch: %03d/%03d | Batch: %03d/%03d | Cost: %.4f' 
                       %(epoch+1,epochs,batch_ids,
                         len(loaders['train']),cost))
        model.eval()
        with torch.set_grad_enabled(False):
            print('Epoch: %03d/%03d'%(epoch+1,epochs))
            print('train acc/loss: %.2f%%/%.2f valid acc/loss: %.2f%%/%.2f'%                  (model_acc(model,loaders['train']),
                   epoch_loss(model,loaders['train']),
                   model_acc(model,loaders['valid']),
                   epoch_loss(model,loaders['valid'])))


# In[ ]:


num_classes=10
model.classifier[6]=tnn.Sequential(
    tnn.Linear(4096,512),tnn.ReLU(),
    tnn.Dropout(.5),tnn.Linear(512,num_classes))
model=model.to(dev)
optimizer=torch.optim.Adam(model.parameters())


# In[ ]:


get_ipython().run_line_magic('train_run', '15 1')


# In[ ]:


#model=models.vgg16(pretrained=True)
#for param in model.parameters():
#    param.requires_grad=False#model.classifier[3].requires_grad=True
#num_classes=33
#model.classifier[6]=tnn.Sequential(
#    tnn.Linear(4096,256),tnn.ReLU(),
#    tnn.Dropout(.5),tnn.Linear(256,num_classes))
#model=model.to(dev)
#optimizer=torch.optim.Adam(model.parameters())


# In[ ]:


#%train_run 3 2

