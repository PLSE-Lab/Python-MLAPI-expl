#!/usr/bin/env python
# coding: utf-8

# # What's Cooking with bidirectional GRU in Pytorch

# > This code uses only Pytorch, but is heavily inspired by fast.ai.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.utils.data
import collections
from scipy.spatial import KDTree
from tqdm import tqdm_notebook
import pdb

from pathlib import Path
path = Path("../input")
list(path.iterdir())


# In[ ]:


#Make sure GPU is on 
torch.cuda.device_count()


# In[ ]:


#Read in training data
fullDF = pd.read_json(path/"train.json")


# In[ ]:


#Create a list of all unique ingredients
vocab = list({s for l in fullDF.ingredients for s in l})
stoi = collections.defaultdict(lambda: len(vocab),{s:i for i,s in enumerate(vocab)})
padIdx = len(vocab)+1


# In[ ]:


#Set cuisine type as a categorical column
fullDF.cuisine = fullDF.cuisine.astype("category")

#Create new column, to be passed to the model, which contains an array of ingredients by position in the vocab array
fullDF["x"] = fullDF.ingredients.apply(lambda l: np.array([stoi[s] for s in l]))


# In[ ]:


nCuisines = len(fullDF.cuisine.cat.categories)
#Dictionary which converts cuisine index to string value
itosCuisine = {i:c for i,c in enumerate(fullDF.cuisine.cat.categories)}


# In[ ]:


#Our processed dataframe
fullDF.head()


# In[ ]:


#Extremely basic dataset class
class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self,x,y):
        self.x, self.y = x,y
        
    def __len__(self): return len(self.x)
    
    def __getitem__(self,idx): return self.x[idx], self.y[idx]


# In[ ]:


#Split train/valid sets
np.random.seed(5342)
valDF = fullDF.sample(frac=0.15,replace=False)
trainDF = fullDF[~fullDF.index.isin(valDF.index)]
assert len(valDF) + len(trainDF) == len(fullDF)


# In[ ]:


#Create three different datasets. fullDS contains all rows in training data
fullDS = RecipeDataset(fullDF.x.values,fullDF.cuisine.cat.codes.values)
trainDS = RecipeDataset(trainDF.x.values,trainDF.cuisine.cat.codes.values)
valDS = RecipeDataset(valDF.x.values,valDF.cuisine.cat.codes.values)


# In[ ]:


#Custom collate function which takes a batch of samples and embeds them in a tensor (sequence length,batch size), padded out to the max ingredient list length of the batch
def collate(samples):
    bs = len(samples)
    maxLen = max(len(s[0]) for s in samples)
    out = torch.zeros(maxLen,bs,dtype=torch.long) + padIdx
    for i,s in enumerate(samples):
        out[:len(s[0]),i] = torch.tensor(s[0],dtype=torch.long)
    return out.cuda(), torch.tensor([s[1] for s in samples],dtype=torch.long).cuda()

#Create the dataloaders
bs = 64
trainDL = torch.utils.data.DataLoader(trainDS,bs,shuffle=True,collate_fn=collate)
valDL = torch.utils.data.DataLoader(valDS,bs,collate_fn=collate)
fullDL = torch.utils.data.DataLoader(fullDS,bs,collate_fn=collate,shuffle=True)


# In[ ]:


#copied from https://github.com/fastai/fastai/blob/master/fastai/layers.py#L116, implements He initialization for the embedding layer
def trunc_normal_(x:torch.tensor, mean:float=0., std:float=1.) -> torch.tensor:
    "Truncated normal initialization."
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


# This is the class for the model. The first layer is an embedding matrix, which maps each ingredient to a vector. Next is the GRU, which encodes the list of ingredients. The last timestep of output from the GRU is then put through a linear layer, which outputs a vector of size nCuisines. Dropout is added at each layer. 
# 
# The approach is to use a large number of parameters, while also utilizing large amounts of regularization, specifically dropout and weight decay. 

# In[ ]:


class CuisineNet(torch.nn.Module):
    def __init__(self,nIngred,embSize,hiddenSize,nCuisines):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.ingredEmb = torch.nn.Embedding(nIngred,embSize,padding_idx = padIdx)
        self.embDropout = torch.nn.Dropout(0.5)
        with torch.no_grad(): trunc_normal_(self.ingredEmb.weight, std=0.01) # Use He initilization on the embedding layer
        self.ingredEnc = torch.nn.GRU(embSize,hiddenSize,2,dropout=0.90,bidirectional=True)
        self.encDropout = torch.nn.Dropout(0.5)
        self.out = torch.nn.Linear(hiddenSize*2,nCuisines)
        
    def forward(self,inp):
        sl, bs = inp.size()
        inp = self.embDropout(self.ingredEmb(inp))
        enc,h = self.ingredEnc(inp,torch.zeros(4,bs,self.hiddenSize).cuda())
        #Since we are using a bidrectional GRU, we need to concat the forward state to the backward state, then pass it to the output layer
        return self.out(self.encDropout(torch.cat([h[-2],h[-1]],dim=1)))


# In[ ]:


#initialize the model. The number of embeddings it two larger than the vocab size, since we need embeddings for padding and unknown. The embedding dimension is 100, and the 
#hidden size of the GRU is 400 (since it is bidrectional, we end up with an output of size 800).
model = CuisineNet(len(vocab)+2,100,400,nCuisines).cuda()


# In[ ]:


#Grab a batch from the dataloader, and pass it through the model to make sure the output shape is correct
x,y = next(iter(trainDL))
model(x)


# In[ ]:


#function to calculate the average accuracy of a batch
def batchAccuracy(preds,target):
    preds = torch.softmax(preds,dim=1)
    preds = torch.argmax(preds,dim=1)
    o = (preds == target).sum().item()
    return o / len(preds)


# In[ ]:


#fit function
def learn(model,epochs,lr,trainDL,valDL=None):
    lossFn = torch.nn.functional.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,amsgrad=True,weight_decay=5e-4)

    for e in tqdm_notebook(range(epochs)):
        model.train()
        with tqdm_notebook(iter(trainDL),leave=False) as t:
            bloss, n = 0.0,0
            for x,y in t:
                pred = model(x)
                loss = lossFn(pred,y)
                bloss += loss.item()
                n += 1
                t.set_postfix({"loss": bloss / n})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {e+1} Training Set Loss: {bloss / n}")
        if valDL is not None:
            model.eval()
            with torch.no_grad():
                loss,accuracy,n =0.0,0.0,0
                for x,y in tqdm_notebook(iter(valDL),leave=False):
                    pred = model(x)
                    loss += lossFn(pred,y)
                    accuracy += batchAccuracy(pred,y)
                    n += 1
                print(f"Validation Set Loss: {loss / n}, Accuracy: {accuracy / n}")


# In[ ]:


#For some reason tqdm progress bars don't display correctly in the kernel (bar is missing).

#The model is trained for 18 epochs, lowering the learning rate every 6 epochs
learn(model,6,1e-3,fullDL)


# In[ ]:


learn(model,6,1e-4,fullDL)


# In[ ]:


learn(model,6,1e-5,fullDL)


# In[ ]:


torch.save(model.state_dict(),"model.pth")


# In[ ]:


testDF = pd.read_json(path/"test.json")
testDF["x"] = testDF.ingredients.apply(lambda l: np.array([stoi[s] for s in l]))


# In[ ]:


testDS = RecipeDataset(testDF.x.values,testDF.id.values)
testDL = torch.utils.data.DataLoader(testDS,bs,collate_fn=collate)


# In[ ]:


def testModel():
    o = []
    model.eval()
    with torch.no_grad():
        for x,ids in tqdm_notebook(iter(testDL),leave=False):
            preds = model(x)
            preds = torch.softmax(preds,dim=1)
            preds = torch.argmax(preds,dim=1)
            for c,id in zip(preds,ids): o.append([id.item(),itosCuisine[c.item()]])
    return pd.DataFrame(o,columns=["id","cuisine"])


# In[ ]:


t = testModel()
t.head()


# In[ ]:


t.to_csv("out.csv",index=False)

