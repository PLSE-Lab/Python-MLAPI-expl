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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


ecg_train=pd.read_csv('../input/mitbih_train.csv',header=None)


# In[ ]:


ecg_train.head(5)


# In[ ]:


ecg_train.shape


# In[ ]:


y_train=ecg_train.iloc[:,187]   # 187 th coloumn index location
y_train.head(5)


# In[ ]:


x_train=ecg_train.iloc[:,0:187]
x_train.head(5)


# In[57]:


from matplotlib import pyplot as plt #import matplotlib.pyplot as plt
plt.plot(x_train.iloc[5,:])


# In[ ]:


cata=y_train.value_counts()
cata


# In[ ]:


from scipy.signal import find_peaks
pk=find_peaks(x_train.iloc[0,:])
print(pk)


# In[ ]:


from scipy.signal import find_peaks,peak_widths,peak_prominences,argrelmin,argrelmax,argrelextrema,spectrogram
peaks=[]
height=[]
width=[]
prominance=[]
arg_min=[]
arg_max=[]
for i in range(len(x_train)):
    peak,hei=find_peaks(x_train.iloc[i,:],height=0)
    peaks.append(peak)
    height.append(hei['peak_heights'])
    wid=peak_widths(x_train.iloc[i,:],peak)[0]
    width.append(wid)
    pro=peak_prominences(x_train.iloc[i,:],peak)[0]
    prominance.append(pro)
    amin=argrelmin(np.array(x_train.iloc[i,:]))[0]
    arg_min.append(amin)
    amax=argrelmax(np.array(x_train.iloc[i,:]))[0]
    arg_max.append(amax)


# In[ ]:


print(arg_max)
type(width)


# In[ ]:


type(width)
len(width[3])


# In[ ]:


f1_len=max([len(i) for i in peaks])
f2_len=max([len(i) for i in height])
f3_len=max([len(i) for i in width])
f4_len=max([len(i) for i in prominance])
f5_len=max([len(i) for i in arg_min])
f6_len=max([len(i) for i in arg_max])


# In[45]:


f1_len


# In[46]:


f2_len


# In[47]:


F1_peaks=[]
F2_height=[]
F3_width=[]
F4_prominance=[]
F5_argmin=[]
F6_argmax=[]
for i in range(len(peaks)):
    pa1=np.pad(peaks[i],(0,f1_len-len(peaks[i])),'constant')
    F1_peaks.append(pa1)
    pa2=np.pad(height[i],(0,f2_len-len(height[i])),'constant')
    F2_height.append(pa2)
    pa3=np.pad(width[i],(0,f3_len-len(width[i])),'constant')
    F3_width.append(pa3)
    pa4=np.pad(prominance[i],(0,f4_len-len(prominance[i])),'constant')
    F4_prominance.append(pa4)
    pa5=np.pad(arg_min[i],(0,f5_len-len(arg_min[i])),'constant')
    F5_argmin.append(pa5)
    pa6=np.pad(arg_max[i],(0,f6_len-len(arg_max[i])),'constant')
    F6_argmax.append(pa6)


# In[48]:


pa1


# In[50]:


ecg_feat=np.concatenate((F1_peaks,F2_height,F3_width,F4_prominance,F5_argmin,F6_argmax),axis=1)
ecg_feat


# In[51]:


ecg_feat.shape


# In[52]:


from sklearn.model_selection import train_test_split
[xtrain,xtest,ytrain,ytest]=train_test_split(ecg_feat,y_train,test_size=0.2,random_state=48)                     
xtrain.shape


# In[53]:


ytest.shape


# In[54]:


from sklearn.ensemble import RandomForestClassifier
ecg_mod=RandomForestClassifier()
ecg_mod.fit(xtrain,ytrain)
ypred=ecg_mod.predict(xtest)


# In[55]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(ytest,ypred)
print("Accuracy of the model is",acc*100)
cm=confusion_matrix(ytest,ypred)
print(cm)


# In[ ]:




