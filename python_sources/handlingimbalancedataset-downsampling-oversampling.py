#!/usr/bin/env python
# coding: utf-8

# # **Down Sampling:
# #    If the dataset is imbalance like 20-80 or 10-90 like that the model will be biased.In order to avoid that we have to balance the dataset.Down sampling is a technique of reduced the highest no of records so that the both the classes in the data are balanced.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]


# In[ ]:


df=pd.read_csv("../input/creditcardfraud/creditcard.csv")


# In[ ]:


df.head()


# **Create Independent and dependent features**

# In[ ]:


x=df.drop('Class',axis=1)
y=df['Class']


# **Create the random state**

# In[ ]:


state=np.random.RandomState(42)
xoutlier=state.uniform(low=0,high=1,size=(x.shape[0],x.shape[1]))


# In[ ]:


sns.countplot(y)


# ***From the above figure we can say that there are more no of values in normal(0) class and no values in 1 class,so we can clearly say that the dataset is imbalanced***

# In[ ]:


normal=df[df['Class']==0]
fraud=df[df['Class']==1]


# In[ ]:


print('No of fraud records of data: {}'.format(fraud.shape[0]))
print('No of normal records of data: {}'.format(normal.shape[0]))


# **Imblearn is the library used for handling the imbalanced dataset which contains methods to handle upsampling,downsampling,etc.,**

# In[ ]:


from imblearn.under_sampling import NearMiss


# In[ ]:


nm=NearMiss()
xsam,ysam=nm.fit_sample(x,y)


# In[ ]:


fraud=ysam[ysam==1]
normal=ysam[ysam==0]


# In[ ]:


print('No of fraud records of data: {}'.format(fraud.shape[0]))
print('No of normal records of data: {}'.format(normal.shape[0]))


# **Now we can able to see that the fraud and normal no of records are balanced**

# In[ ]:


from collections import Counter


# In[ ]:


print('Shape of original dataset {}'.format(Counter(y)))
print('Shape of resampled dataset {}'.format(Counter(ysam)))


# **Thus we successfully down sampled the imbalanced dataset**

# # Oversampling: ****
# Oversampling makes the class with less number of data to balance with the other class by increasing the no of records,so that the dataset will be balanced.Oversampling is better than downsampling because it gives more no of data whereas in downsampling we lose some amount of data.

# In[ ]:


from imblearn.over_sampling import RandomOverSampler


# In[ ]:


ros=RandomOverSampler()
xosam,yosam=ros.fit_sample(x,y)


# In[ ]:


print('Shape of original dataset {}'.format(Counter(y)))
print('Shape of resampled dataset {}'.format(Counter(yosam)))


# # Combination: Combination of Oversampling and Downsampling****

# In[ ]:


from imblearn.combine import SMOTETomek


# In[ ]:


com=SMOTETomek(random_state=42)
xcom,ycom=com.fit_sample(x,y)


# In[ ]:


print('Shape of original dataset {}'.format(Counter(y)))
print('Shape of resampled dataset {}'.format(Counter(ycom)))


# In[ ]:


plt.figure(figsize=(7,7))
plt.tight_layout()
plt.subplot(2,2,1)
plt.title('Original Data')
sns.countplot(y)


plt.tight_layout()
plt.subplot(2,2,2)
plt.title('Downsampled data')
sns.countplot(ysam)

plt.tight_layout()
plt.subplot(2,2,3)
plt.title('Oversampled data')
sns.countplot(yosam)

plt.tight_layout()
plt.subplot(2,2,4)
plt.title('Smotet data')
sns.countplot(ycom)

plt.show()


# In[ ]:




