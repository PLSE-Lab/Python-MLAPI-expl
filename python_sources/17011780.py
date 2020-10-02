#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip uninstall kaggle')
get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install kaggle==1.5.6')

get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle')
get_ipython().system('ls -lha kaggle.json')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# In[ ]:


get_ipython().system('kaggle competitions download -c sejongaiclasspredicteq ')


# In[ ]:


get_ipython().system('unzip sejongaiclasspredicteq.zip')


# In[ ]:


import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np


# In[ ]:


data=pd.read_excel('train.xlsx')
data=np.array(data)
data=data[1:,]
scale=data[:,2]

lat=[]
for i in range(1019):
  try:
    lat.append(float(data[:,4][i][:5]))
  except:
    lat.append(float(0))

log=[]
for i in range(1019):
  try:
    log.append(float(data[:,5][i][:5]))
  except:
    log.append(float(0))


# In[ ]:


import matplotlib.pyplot as plt

plt.scatter(log,lat)

plt.xlim([122,132])
plt.ylim([32,43])
plt.show()


# In[ ]:


s=[]
for i in range(1019):
  if(lat[i]>=34.5 and lat[i]<=40 and log[i]>=124.5 and log[i]<=129.5):
    s.append(scale[i])

len(s)
s=np.array(s)

cnt=[]
for i in range(26):
  cnt.append([(25+i),0])

for i in range(26):
  k=s>=(2.5+i*0.1)
  cnt[i][1]=k.sum()


# In[ ]:


x=[]
y=[]
for i in range(26):
  x.append(2.5+0.1*i)
  y.append(cnt[i][1])
X=np.array(x)
Y=np.array(y)
X=torch.FloatTensor(X)
Y=torch.FloatTensor(Y)


# In[ ]:


X


# In[ ]:


Y


# In[ ]:


import torch
import torch.nn.functional as F
import torch.optim as optim


# In[ ]:


nb_epochs=10000
a=torch.ones(1,requires_grad=True)
b=torch.ones(1,requires_grad=True)
optimizer=optim.Adam([a,b],lr=0.001, betas=(0, 0))
for epoch in range(nb_epochs):
  h=10**(a-b*X)
  
  
  cost=torch.mean((h-Y)**2)

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch%100==0:
    print(epoch ,cost)


# In[ ]:


ans=[]
for i in range(20):
  ans.append([i,0])
t=[2.0,2.1,2.2,2.3,2.4]
t=np.array(t)
t=torch.FloatTensor(t)

ans=10**(a-b*t)


# In[ ]:


ar=[]
for i in range(5):
  ar.append(int(ans[i]))
ar=np.array(ar)


# In[ ]:


sol=pd.DataFrame(ar)
sol=pd.DataFrame({"id":[0,1,2,3,4],"expected":ar})


# In[ ]:


sol.to_csv("yw2.csv",header=True,index=False)


# In[ ]:


get_ipython().system('kaggle competitions submit -c sejongaiclasspredicteq -f yw2.csv -m "Message"')


# In[ ]:




