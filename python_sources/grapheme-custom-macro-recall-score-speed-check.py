#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
'''
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''
# Any results you write to the current directory are saved as output.
from sklearn.metrics import recall_score
import time
import matplotlib.pyplot as plt


# In[ ]:


from numba import jit,autojit
@jit('float32(float32[:],float32[:])')
def score(label:np.array,target:np.array) -> float:
    score = 0
    labels = np.concatenate((label,target))
    labels = np.unique(labels)
    for i in labels:
        index = np.where(target==i)
        TP = (label[index]==i).sum()
        FN = (label[index]!=i).sum()
        score += (TP / (TP+FN+1e-20))
    return score/(len(labels))
def MacroRecall(label:np.array,target:np.array) -> float:
    return score(label.astype(np.float32),target.astype(np.float32))


# #### Record scores & excecution times roughly loop x 100000

# In[ ]:


num_trial=100000
times1=[]
records1=[]
times2=[]
records2=[]
for _ in range(num_trial):
    num_labels = np.random.randint(168)
    num_size = np.random.randint(500)
    label = np.random.randint(num_labels+1,size=num_size+1)
    target = np.random.randint(num_labels+1,size=num_size+1)
    start_time = time.monotonic()
    records1.append(MacroRecall(label,target))
    times1.append(time.monotonic()-start_time)
    start_time = time.monotonic()
    records2.append(recall_score(target,label,average='macro'))
    times2.append(time.monotonic()-start_time)


# #### Check discrepancies threshold 1e-7 and 1e-5

# In[ ]:


diff_scores = np.subtract(records1,records2)


# In[ ]:


print('Mismatch counts : {}'.format(sum(abs(diff_scores)>=1e-7)))


# In[ ]:


print('Mismatch counts : {}'.format(sum(abs(diff_scores)>=1e-5)))


# In[ ]:


custom_mean = np.mean(times1)
sklearn_mean = np.mean(times2)
custom_std = np.std(times1)
sklearn_std = np.std(times2)
mean = np.mean(np.hstack([times1,times2]))
std = np.std(np.hstack([times1,times2]))


# #### check execution time

# In[ ]:


plt.hist(np.log(times1),bins=2000,label='custom')
plt.hist(np.log(times2),bins=2000,label='sklearn-macro recall')
plt.legend()
plt.title('log transform of excution time')
plt.show()


# In[ ]:


print(f'the percentage of outlier custom :{100*sum(times1>=mean+10*std)/len(times1)}%')
print(f'the percentage of outlier sklearn macro recall :{100*sum(times2>=mean+10*std)/len(times2)}%')


# In[ ]:


plt.bar(['custom', 'sklearn-macro recall'],np.hstack([custom_mean,sklearn_mean]),yerr=np.hstack([custom_std,sklearn_std]))
plt.ylim((0,max(custom_mean,sklearn_mean)+2*0.002))
plt.title('mean execution time with std')
plt.show()


# In[ ]:


check_records = pd.DataFrame([records1,times1,records2,times2,diff_scores]).T.rename(columns={0:'custom'
                                                                                              ,1:'custom_exec'
                                                                                             ,2:'sklearn'
                                                                                              ,3:'sklearn_exec'
                                                                                             ,4:'diff_score'})


# In[ ]:


check_records.to_csv('check_records.csv')


# In[ ]:


check_records.sample(20)

