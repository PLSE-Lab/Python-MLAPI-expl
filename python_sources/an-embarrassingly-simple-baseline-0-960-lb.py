#!/usr/bin/env python
# coding: utf-8
Interesting! Of course we chose a really short and pure test to try and make it relatively achievable, but I am surprised by this one!
Let's look and see how it performs with the traing data itself; treating THAT as test data if you get me
# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import math


# In[ ]:


df = pd.read_csv("../input/liverpool-ion-switching/train.csv")
data=df.values


# In[ ]:


n_groups = 100
df["group"] = 0
for i in range(n_groups):
    ids = np.arange(i*50000, (i+1)*50000)
    df.loc[ids,"group"] = i


# In[ ]:


for i in range(n_groups):
    sub = df[df.group == i]
    signals = sub.signal.values
    imax, imin = math.floor(np.max(signals)), math.ceil(np.min(signals))
    signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals))
    signals = signals*(imax-imin)
    df.loc[sub.index,"open_channels"] = [0,] + list(np.array(signals[:-1],np.int))


# We are not going to bother with sample submission, let's just test this smoothing method against the labels in the training data. Should be fine.

# In[ ]:


print(data[:5,1])
print(data[:5,2])


# In[ ]:


prediction = np.array(df.open_channels, np.int)
print(prediction[:5])


# Looks good!
# So get the metrics, Kappa, Quadratic Kappa and Accuracy;

# In[ ]:


#To check I am working the metrics right ;-)
gd=[1,2,3,4,5,6,7,8,9,0]
pr=[1,2,3,4,5,6,7,8,8,0]

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
print("Regular Cohen's Kappa", cohen_kappa_score(np.asarray(data[:,2],np.int),np.array(df.open_channels, np.int),weights="quadratic"))
print("Quadratic Cohen's Kappa", cohen_kappa_score(np.asarray(data[:,2],np.int),np.array(df.open_channels, np.int)))
print("Accuracy", accuracy_score(data[:,2],np.array(df.open_channels, np.int)))
print("test Accuracy", accuracy_score(gd,pr))


# OK, I guess I messed up, perhaps you could help with this?

# In[ ]:




