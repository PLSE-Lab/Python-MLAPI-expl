#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[ ]:


data=pd.read_csv('../input/train.csv')
data.head(7)


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


plt.boxplot(data['x'])


# In[ ]:


plt.boxplot(data['y'])


# In[ ]:


b1=0
b2=0
l=0.01
#o=b1+b2*x


# In[ ]:


lx=list(data.x)
lx[216]
ly=list(data.y)


# In[ ]:


for i in range(260):
    o=b1+b2*lx[i]
    er=o-ly[i]
    b1=round(b1-l*er,2)
    b2=b2-l*er*lx[i]
    print(b1,b2)
    


# In[ ]:


data2=pd.read_csv('../input/test.csv')
data2.head()


# In[ ]:


testx=list(data2.x)
result=[]
index=[]
testy=list(data2.y)


# In[ ]:


for i in range(len(testx)-1):
    o=b1+b2*testx[i]
    result.append(o)
    index.append(i)


# In[ ]:


predict=list(zip(result,testy))


# In[ ]:


predictFrame=pd.DataFrame(predict,columns=['Y','y'])
predictFrame.head()


# In[ ]:


plt.scatter(x=predictFrame['Y'],y=predictFrame['y'],data=predictFrame)

