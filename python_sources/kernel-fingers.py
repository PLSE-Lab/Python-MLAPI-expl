#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import torch
import shap
from torchvision import transforms as trans
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder


# In[51]:


from torchvision.models import inception_v3
import cv2 


# In[52]:


dataset = ImageFolder('../input/ml3taskfingers', transform=
                      trans.Compose([
                                                                          trans.ToTensor(),
                                                                          trans.ToPILImage(), 
                                                                          trans.RandomRotation(15), 
                                                                          trans.RandomHorizontalFlip(), 
                                                                          trans.ColorJitter(0.5,0.5,0.4), 
                                                                          trans.ToTensor()])
                     )
dataset,valid = torch.utils.data.random_split(dataset, (len(dataset)-250, 250))
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
dataloadervalid = DataLoader(valid, batch_size=16, shuffle=False)


# In[53]:


i , a = next(iter(dataloader))
i2 , a = next(iter(dataloadervalid))
plt.imshow(i[0][2].numpy())


# In[54]:


i.size()


# In[55]:


model = inception_v3(pretrained=True)


# In[56]:


model.aux_logits = False
for child in list(model.children())[:4]:
    for param in child.parameters():
        param.requires_grad = False
model.fc = torch.nn.Linear(2048, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
loss_fn = torch.nn.CrossEntropyLoss()


# In[57]:


def acc(ans, pred):
    pred = torch.argmax(pred, dim=1)
    accuracy = torch.mean((ans == pred).float())
    return accuracy.item() 


# In[ ]:


model = model.cuda()


# In[ ]:



epoches = 40
for epoch in range(epoches):
    ans, preds = [], []  
#     ansv, predsv = [], []

    model.train()
    for batch in dataloader:
        X, y = batch
        X = X.cuda()
        y = y.cuda()
        optimizer.zero_grad() 
        output = model(X)
        target = y
        preds.append(output)
        ans.append(target)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
#     model.eval()
#     for batch in dataloadervalid:
#         Xv, yv = batch
#         Xv = Xv.cuda()
#         yv = yv.cuda()
#         outputv = model(Xv)
#         targetv = yv
#         predsv.append(outputv)
#         ansv.append(targetv)
        
    ans = torch.cat(ans)
    preds = torch.cat(preds)
    train_accuracy = acc(ans, preds)
    
#     ansv = torch.cat(ansv)
#     predsv = torch.cat(predsv)
#     valid_accuracy = acc(ansv, predsv)
    print(f'Epoch {epoch}: train acc: {train_accuracy:.4f}  ')    
#     print(f'Epoch {epoch}: val acc: {valid_accuracy:.4f}  ') 


# In[ ]:


# torch.save(model)


# In[ ]:


model.cpu()
model.eval()
ans, preds, f = [], [], []

for batch in dataloadervalid:
    X, y = batch
    X = X
    y = y
    output = model(X)
    target = y
    preds.append(torch.argmax(output, dim=1))
    ans.append(target)
    f.append(X)


# In[ ]:


ans = torch.cat(ans)
preds = torch.cat(preds)


# In[ ]:


(ans==preds).sum().item() / len(ans)


# In[ ]:


mask = (ans!=preds)
eans, epreds = ans[mask], preds[mask]
feat = torch.cat(f)
ef = feat[mask]


# In[ ]:


(preds>5).sum()


# In[ ]:


f, axes = plt.subplots(5,5, figsize=(30, 30))

for x, a, feature, ax in zip(eans, epreds, ef, axes.ravel()):
  ax.imshow(feature[0])
  ax.set_title(f'Predicted as {a}, true label is {x}')


# In[ ]:


# select a set of background examples to take an expectation over
# background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
shap.
# explain predictions of the model on four images
e = shap.DeepExplainer(model.cpu(), i.cpu())
# ...or pass tensors directly
# e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), i)
shap_values = e.shap_values(i2[1:5].cpu())

# plot the feature attributions
shap.image_plot(shap_values, -i2[1:5])

