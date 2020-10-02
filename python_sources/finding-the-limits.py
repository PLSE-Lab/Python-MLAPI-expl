#!/usr/bin/env python
# coding: utf-8

# Let see what are the range of the data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16,5)


# In[ ]:


train = pd.read_csv('../input/train.csv', index_col='row_id')
train.describe()


# - x and y are in the range of [0,10]. 
# - accuracy is in the range of [1, 1033].
# - time is in the range of [1, 786239]

# In[ ]:


test = pd.read_csv('../input/test.csv', index_col='row_id')
test['place_id'] = -1
#test.head()
test.describe()


# The same thing is for the test set, except:
# - time is in the range of [786242, 1006589]
# The test and train are time splited.

# In[ ]:


df = pd.concat([train, test])
idx_test = (df.place_id == -1)
print(df.head())
print(df.tail())
df.describe()


# In[ ]:


df.time.hist(bins=100);


# In[ ]:


checkins, bins = np.histogram(df.time, bins=range(0,df.time.max()+60,60))
fft = np.fft.fft(checkins-checkins.mean())


# In[ ]:


plt.xlim(100,1000)#len(fft)/2)
plt.ylim(10**3,10**7)
X = np.array([100, 200, 499, 599, 699, 799, 899, 999,1298, 2097])
for x in X:
    plt.axvline(x,color='red')
for x in [300, 400]:
    plt.axvline(x,color='green')
plt.loglog(np.sqrt(fft * fft.conj()).real);


# In[ ]:


import datetime as dt


# In[ ]:


rng = pd.date_range('1/1/2013','1/1/2015',freq='H')
rng.shape


# In[ ]:


checkin_sim = pd.DataFrame(index=rng)


# In[ ]:


checkin_sim['open'] = 0


# In[ ]:


checkin_sim['dayofweek'] = rng.dayofweek
checkin_sim['month'] = rng.month
checkin_sim['day'] = rng.day
checkin_sim['hour'] = rng.hour


checkin_sim.ix[(checkin_sim.hour>8) & (checkin_sim.hour<17),'open']=1
#checkin_sim.ix[(checkin_sim.hour==1),'open']=0
checkin_sim.ix[checkin_sim.dayofweek>4,'open']=0
checkin_sim.ix[(checkin_sim.month == 8) & (checkin_sim.month<15),'open']=0
checkin_sim.ix[(checkin_sim.month == 1) & (checkin_sim.month<7),'open']=0
checkin_sim.ix[(checkin_sim.month == 12) & (checkin_sim.month>24),'open']=0


# In[ ]:


ds = checkin_sim.open
fft = np.fft.fft(ds-ds.mean())


# In[ ]:


plt.rcParams['figure.figsize'] = (16,5)
plt.xlim(1000,2000)#len(fft)/2)
plt.ylim(0.01,10**5)
plt.loglog(np.sqrt(fft * fft.conj()).real*np.arange(len(fft))**0.0);
X = np.array([52, 52*2, 52*3, 52*4, 52*5, 52*6, 365, 356*2+18])#, 259, 365, 365-52, 365*2])
for x in 2*X:
    plt.axvline(x,color='red')
for x in [116, 299]:
    plt.axvline(x,color='green')


# In[ ]:


365-52*2


# In[ ]:


116*2


# In[ ]:


df.time.max()/(700*24*60)


# In[ ]:




