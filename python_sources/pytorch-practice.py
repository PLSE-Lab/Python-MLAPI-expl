#!/usr/bin/env python
# coding: utf-8

# Reading classics [Deep Learning Models](https://github.com/rasbt/deeplearning-models)
# ## Basic Examples

# In[ ]:


import numpy as np,pandas as pd,pylab as pl
import h5py,torch
from torchvision.datasets import MNIST as tmnist
from torchvision import transforms
from torch.utils.data import DataLoader as tdl
from torch.utils.data import Dataset as tds
import torch.nn.functional as tnnf
from sklearn.datasets import make_classification
dev=torch.device("cuda:0" if torch.cuda.is_available() 
                 else "cpu")


# In[ ]:


# artificial data
N=500; n=int(.2*N)
X,y=make_classification(n_samples=N,n_features=2,
                        n_redundant=0,n_informative=2)
mu,std=np.mean(X,axis=0),np.std(X,axis=0)
X=(X-mu)/std
X,y=X.astype('float32'),y.astype('int32')
pl.figure(figsize=(11,3)); pl.grid()
pl.scatter(X[:,0],X[:,1],marker='o',
           s=10,c=y,cmap='cool');


# In[ ]:


# shuffling & splitting
shuffle_ids=np.arange(N)
np.random.RandomState(23).shuffle(shuffle_ids)
X,y=X[shuffle_ids],y[shuffle_ids]
X_test,X_train=X[:n],X[n:]
y_test,y_train=y[:n],y[n:]
pl.figure(figsize=(11,3)); pl.grid()
pl.scatter(X_test[:,0],X_test[:,1],marker='o',
           s=10,c=y_test,cmap='cool');


# In[ ]:


class Perceptron():
    def __init__(self,num_features):
        self.num_features=num_features
        self.weights=torch.zeros(num_features,1, 
                                 dtype=torch.float32,device=dev)
        self.bias=torch.zeros(1,dtype=torch.float32,device=dev)
    def forward(self,x):
        values=torch.add(torch.mm(x,self.weights),self.bias)
        a,b=torch.ones(values.size()[0],1),torch.zeros(values.size()[0],1)
        predictions=torch.where(values>0.,a,b).float()
        return predictions        
    def backward(self,x,y):  
        predictions=self.forward(x)
        errors=y-predictions
        return errors        
    def train(self,x,y,epochs):
        for e in range(epochs):            
            for i in range(y.size()[0]):
                errors=self.backward(x[i].view(1,self.num_features),
                                     y[i]).view(-1)
                self.weights+=(errors*x[i]).view(self.num_features,1)
                self.bias+=errors                
    def acc(self,x,y):
        predictions=self.forward(x).view(-1)
        accuracy=torch.sum(predictions==y).float()/y.size()[0]
        return accuracy


# In[ ]:


model=Perceptron(num_features=2)
tX_train=torch.tensor(X_train,dtype=torch.float32,
                      device=dev)
ty_train=torch.tensor(y_train,dtype=torch.float32,
                      device=dev)
model.train(tX_train,ty_train,epochs=5)
print('Weights: %s'%model.weights)
print('Bias: %s'%model.bias)


# In[ ]:


# evaluating
tX_test=torch.tensor(X_test,dtype=torch.float32,
                     device=dev)
ty_test=torch.tensor(y_test,dtype=torch.float32,
                     device=dev)
acc_test=model.acc(tX_test,ty_test)
print('Test accuracy: %.2f%%'%(acc_test*100))


# In[ ]:


W,b=model.weights,model.bias
x_min=-2; x_max=2
y_min=((-(W[0]*x_min)-b[0])/W[1])
y_max=((-(W[0]*x_max)-b[0])/W[1])
fig,ax=pl.subplots(1,2,sharex=True,figsize=(11,3))
ax[0].plot([x_min,x_max],[y_min,y_max],c='red')
ax[1].plot([x_min,x_max],[y_min,y_max],c='red')
ax[0].scatter(X_train[:,0],X_train[:,1],
              c=y_train,s=10,cmap=pl.cm.cool)
ax[1].scatter(X_test[:,0], X_test[:,1],
              c=y_test,s=10,cmap=pl.cm.cool)
ax[0].grid(); ax[1].grid()


# In[ ]:


class LogisticRegression():
    def __init__(self,num_features):
        self.num_features=num_features
        self.weights=torch.zeros(num_features,1, 
                                dtype=torch.float32,device=dev)
        self.bias=torch.zeros(1,dtype=torch.float32,device=dev)
    def forward(self,x):
        values=torch.add(torch.mm(x,self.weights),self.bias)
        probs=self._sigmoid(values)
        return probs       
    def backward(self,probs,y):  
        errors=y-probs.view(-1)
        return errors            
    def predict_labels(self,x):
        probs=self.forward(x)
        a=torch.ones(probs.size()[0],1)
        b=torch.zeros(probs.size()[0],1)
        labels=torch.where(probs>=.5,a,b)
        return labels                
    def acc(self,x,y):
        labels=self.predict_labels(x).float()
        accuracy=torch.sum(labels.view(-1)==y).float()/y.size()[0]
        return accuracy    
    def _sigmoid(self,z):
        return 1./(1.+torch.exp(-z))    
    def _logit_cost(self,y,prob):
        tmp1=torch.mm(-y.view(1,-1),torch.log(prob))
        tmp2=torch.mm((1-y).view(1,-1),torch.log(1-prob))
        return tmp1-tmp2
    def train(self,x,y,epochs,learning_rate=.01):
        for e in range(epochs):
            probs=self.forward(x)
            errors=self.backward(probs,y)
            neg_grad=torch.mm(x.transpose(0,1),errors.view(-1,1))
            self.weights+=learning_rate*neg_grad
            self.bias+=learning_rate*torch.sum(errors)
            print('Epoch: %03d'%(e+1),end="")
            print(' | Train accuracy: %.3f'%self.acc(x,y),end="")
            print(' | Cost: %.3f'%self._logit_cost(y,self.forward(x)))


# In[ ]:


model=LogisticRegression(num_features=2)
model.train(tX_train,ty_train,epochs=10,learning_rate=.02)
print('Weights: %s'%model.weights)
print('Bias: %s'%model.bias)


# In[ ]:


# evaluating
acc_test=model.acc(tX_test,ty_test)
print('Test accuracy: %.2f%%'%(acc_test*100))


# In[ ]:


W,b=model.weights,model.bias
x_min=-2; x_max=2
y_min=((-(W[0]*x_min)-b[0])/W[1])
y_max=((-(W[0]*x_max)-b[0])/W[1])
fig,ax=pl.subplots(1,2,sharex=True,figsize=(11,3))
ax[0].plot([x_min,x_max],[y_min,y_max],c='red')
ax[1].plot([x_min,x_max],[y_min,y_max],c='red')
ax[0].scatter(X_train[:,0],X_train[:,1],
              c=y_train,s=10,cmap=pl.cm.cool)
ax[1].scatter(X_test[:,0], X_test[:,1],
              c=y_test,s=10,cmap=pl.cm.cool)
ax[0].grid(); ax[1].grid()


# ## Softmax Regression

# In[ ]:


random_seed=23; batch_size=128
train=tmnist(root='data',train=True,download=True,
            transform=transforms.ToTensor())
test=tmnist(root='data',train=False, 
            transform=transforms.ToTensor())
train_loader=tdl(dataset=train,shuffle=True, 
                 batch_size=batch_size)
test_loader=tdl(dataset=test,shuffle=False, 
                batch_size=batch_size)
for images,labels in train_loader:  
    print('Image dimensions: %s'%str(images.shape))
    print('Label dimensions: %s'%str(labels.shape))
    break


# In[ ]:


learning_rate=.1; epochs=15
num_features=784; num_classes=10
class SoftmaxRegression(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        super(SoftmaxRegression,self).__init__()
        self.linear=torch.nn.Linear(num_features,num_classes)        
        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()     
    def forward(self,x):
        logits=self.linear(x)
        probs=tnnf.softmax(logits,dim=1)
        return logits,probs
model=SoftmaxRegression(num_features=num_features,
                        num_classes=num_classes)
model.to(dev)
optimizer=torch.optim.SGD(model.parameters(),
                          lr=learning_rate) 


# In[ ]:


def model_acc(model,data_loader,num_features):
    correct_preds,num_examples=0,0    
    for features,targets in data_loader:
        features=features.view(-1,num_features).to(dev)
        targets=targets.to(dev)
        logits,probs=model(features)
        _,pred_labels=torch.max(probs,1)
        num_examples+=targets.size(0)
        correct_preds+=(pred_labels==targets).sum()        
    return correct_preds.float()/num_examples*100


# In[ ]:


for epoch in range(epochs):
    for batch_ids,(features,targets) in enumerate(train_loader):        
        features=features.view(-1,num_features).to(dev)
        targets=targets.to(dev)
        logits,probs=model(features)
        cost=tnnf.cross_entropy(logits,targets)
        optimizer.zero_grad(); cost.backward()
        optimizer.step()
        if not batch_ids%200:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1,epochs,batch_ids, 
                     len(train)//batch_size,cost))           
    with torch.set_grad_enabled(False):
        print('Epoch: %03d/%03d train accuracy: %.2f%%'%              (epoch+1,epochs,model_acc(model,train_loader,num_features)))


# In[ ]:


print('Test accuracy: %.2f%%'%(model_acc(model,test_loader,num_features)))


# ## Applying to Color Images

# In[ ]:


fpath='../input/flower-color-images/'
f=h5py.File(fpath+'FlowerColorImages.h5','r')
keys=list(f.keys()); print(keys)
X=np.array(f[keys[0]],dtype='float32')/255
y=np.array(f[keys[1]],dtype='int32')
N=len(y); n=int(.2*N); batch_size=16
shuffle_ids=np.arange(N)
np.random.RandomState(23).shuffle(shuffle_ids)
X,y=X[shuffle_ids],y[shuffle_ids]
X_test,X_train=X[:n],X[n:]
y_test,y_train=y[:n],y[n:]
X_train.shape,y_train.shape


# In[ ]:


class TData(tds):
    def __init__(self,X,y):   
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.int32)
    def __getitem__(self,index):
        train_img,train_lbl=self.X[index],self.y[index]
        return train_img,train_lbl
    def __len__(self):
        return self.y.shape[0]
train=TData(X_train,y_train)
test=TData(X_test,y_test)
train_loader=tdl(dataset=train,batch_size=batch_size,shuffle=True)
test_loader=tdl(dataset=test,batch_size=batch_size,shuffle=False)
for images,labels in train_loader:  
    print('Image dimensions: %s'%str(images.shape))
    print('Label dimensions: %s'%str(labels.shape))
    break


# In[ ]:


learning_rate=.01; epochs=25
num_features=49152; num_classes=10
torch.manual_seed(random_seed)
model=SoftmaxRegression(num_features=num_features,
                         num_classes=num_classes)

model.to(dev)
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)


# In[ ]:


for epoch in range(epochs):
    for batch_ids,(features,targets) in enumerate(train_loader):        
        features=features.view(-1,num_features).to(dev)
        targets=targets.to(dev)
        logits,probs=model(features)
        cost=tnnf.cross_entropy(logits,targets.long())
        optimizer.zero_grad(); cost.backward()
        optimizer.step()
        if not batch_ids%10:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1,epochs,batch_ids, 
                     len(train)//batch_size,cost))           
    with torch.set_grad_enabled(False):
        print('Epoch: %03d/%03d train accuracy: %.2f%%'%              (epoch+1,epochs,model_acc(model,train_loader,num_features)))


# In[ ]:


print('Test accuracy: %.2f%%'%(model_acc(model,test_loader,num_features)))

