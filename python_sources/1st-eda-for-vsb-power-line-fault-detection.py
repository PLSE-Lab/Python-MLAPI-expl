#!/usr/bin/env python
# coding: utf-8

# # Overview
# #### electric transmission lines --> fault --> Partial discharge
# 
# #### each signal --> 800,000 measurements of a power line's voltage, taken over 20 milliseconds
# 
# #### underlying electric grid operates at 50Hz
# 
# #### 3-phase power scheme. / 3-phases are measured simultaneously
# 
# 
# 
# # data description in homepage
# ## Metadata_[train/test].csv
# #### id_measurement: ID code for a trio of signals recorded at the same time
# #### signal_id: foreign key for the signal data. (unique across both train and test)
# #### phase: the phase ID code within the signal trio. *** The phases may or may not all be impacted by a fault on the line.
# #### target: 0 (undamaged)/ 1 (fault)
# 
# ## parquet
# #### each column contains one signal: 800,000 int8 measurments.
# #### (Note that this is different than our usual data orientation of one row per observation)

# In[ ]:


import os
import gc
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
from numba import jit, int32
plt.style.use("fast")

import warnings
warnings.filterwarnings("ignore") 

plt.style.use('seaborn')
sns.set(font_scale=1)


# ## read parquet file as df_train

# In[ ]:


df_train = pq.read_pandas('../input/train.parquet', columns=[str(i) for i in range(3000)]).to_pandas()
# df_train = pq.read_pandas('../input/train.parquet').to_pandas()


# ## Read test data (test.parquet)
# #### We cannot read test parquet in the same way with train parquet.
# #### The first ID in train is '0' but the first ID in test is '8712'.
# 
# #### cannot use below
# ###### df_test = pq.read_pandas(os.path.join(cwd, 'test.parquet'), columns=[str(i) for i in range(1000)]).to_pandas()
# ###### df_test = pq.read_pandas(os.path.join(cwd, 'test.parquet')).to_pandas()
# 
# #### We should read test.parquet like below line
# ###### df_test = pq.read_pandas('../input/train.parquet', columns=[str(i+8712) for i in range(1000)]).to_pandas()

# #### train parquet data check

# In[ ]:


print(df_train.info())
df_train.head()


# ### null check

# In[ ]:


df_train.isnull().sum().sum()


# ## read Metadata_[train/test].csv as meta_train & meta_test

# In[ ]:


meta_train = pd.read_csv('../input/metadata_train.csv')
meta_test = pd.read_csv('../input/metadata_test.csv')


# #### Metadata_train

# In[ ]:


meta_train.head()


# In[ ]:


meta_train.describe()


# In[ ]:


meta_train.shape


# #### Metadata_test

# In[ ]:


meta_test.head()


# In[ ]:


meta_test.describe()


# In[ ]:


meta_test.shape


# In[ ]:


meta_train['target'].unique()


# ### null check

# In[ ]:


print(meta_train.isnull().sum())
print(meta_test.isnull().sum())


# ## Number & ratio of normal and fault samples

# In[ ]:


meta_train['target'].value_counts()


# In[ ]:


print('Normal sample number: {:d}'.format(meta_train['target'].value_counts()[0]))
print('Fault sample number: {:d}'.format(meta_train['target'].value_counts()[1]))


# In[ ]:


print('Normal sample ratio: {:.3f} %'.format(meta_train['target'].value_counts()[0]/len(meta_train)*100))
print('Fault sample ratio: {:.3f} %'.format(meta_train['target'].value_counts()[1]/len(meta_train)*100))


# In[ ]:


f, ax = plt.subplots(1, 3, figsize=(24, 8))

ax[0].pie(meta_train['target'].value_counts(), explode=[0, 0.1], autopct='%1.3f%%', shadow=True)
ax[0].set_title('Pie plot - Fault')
ax[0].set_ylabel('')

sns.countplot('target', data=meta_train, ax=ax[1])
ax[1].set_title('Count plot - Fault')

sns.countplot('target', data=meta_train, hue ='phase', ax=ax[2])


# ## --> very small number of fault data: imbalanced dataset

# In[ ]:


meta_train.head()


# In[ ]:


diff = meta_train.groupby(['id_measurement']).sum().query("target != 3 & target != 0")
print('not all fault or normal data number: {:d}'.format(diff.shape[0]))
diff


# In[ ]:


pd.crosstab(meta_train['phase'], meta_train['target'], margins=True)


# In[ ]:


meta_train[['phase', 'target']].groupby(['phase'], as_index=True).mean()


# In[ ]:


meta_train[['phase', 'target']].groupby(['phase'], as_index=True).mean().sort_values(by='target', ascending=False).plot.bar()


# In[ ]:


targets = meta_train.groupby('id_measurement', as_index=True)[['target','id_measurement']].mean()
targets.iloc[67,:]


# In[ ]:


targets.head()


# In[ ]:


sns.countplot(x='target',data=round(targets, 2))


# ### There is little difference in the number of faults (or fault rate) with respect to the phases
# ### Data with the same id is not always all fault or normal.
# 
# ### as data description, "phase: the phase ID code within the signal trio. The phases may or may not all be impacted by a fault on the line."

# In[ ]:


meta_train['phase'].value_counts()


# In[ ]:


meta_train['phase'].value_counts().plot.bar()


# In[ ]:


meta_train.groupby('phase').count().plot.bar()


# In[ ]:


meta_train.corr()


# In[ ]:


plt.figure(figsize = (10,10))

colormap = plt.cm.summer_r
sns.heatmap(meta_train.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})


# # Normal and Fault data visualization

# In[ ]:


#np.unique(meta_train.loc[meta_train['target']==1, 'id_measurement'].values)
# Fault measurement ID
fault_mid = meta_train.loc[meta_train['target']==1, 'id_measurement'].unique()
# Normal measurement ID
normal_mid = meta_train.loc[meta_train['target']==0, 'id_measurement'].unique()


# In[ ]:


print(fault_mid.shape)
print(normal_mid.shape)


# In[ ]:


print('8712//3={:d}'.format(8712//3))
print('2748+194-38={:d} (38 overlapped samples (some phases are normal, the others are fault))'.format(2748+194-38))


# ## There are 38 overlapping data. That is, for the 38 measure IDs, the normal phase and the fault phase exist simultaneously.

# In[ ]:


print(meta_train.signal_id.unique())
print(meta_train.id_measurement.unique())


# In[ ]:


meta_train.head(10)


# In[ ]:


fault_mid[1]


# ## N/F signal ID_sid

# In[ ]:


fault_sid = meta_train.loc[meta_train.id_measurement == fault_mid[0], 'signal_id']
normal_sid = meta_train.loc[meta_train.id_measurement == normal_mid[0], 'signal_id']


# In[ ]:


print(fault_sid)
print(normal_sid)


# In[ ]:


fault_sample = df_train.iloc[:, fault_sid]
normal_sample = df_train.iloc[:, normal_sid]


# # Plotting the normal and fault signals

# ## Normal sample

# In[ ]:


plt.figure(figsize=(24, 8))
plt.plot(normal_sample, alpha=0.7);
plt.ylim([-100, 100])


# ## Fault sample

# In[ ]:


plt.figure(figsize=(24, 8))
plt.plot(fault_sample, alpha=0.8);
plt.ylim([-100, 100])


# # Faltiron
# #### reference: https://www.kaggle.com/miklgr500/flatiron

# In[ ]:


@jit('float32(float32[:,:], int32, int32)')
def flatiron(x, alpha=100., beta=1):
    new_x = np.zeros_like(x)
    zero = x[0]
    for i in range(1, len(x)):
        zero = zero*(alpha-beta)/alpha + beta*x[i]/alpha
        new_x[i] =  x[i] - zero
    return new_x


# In[ ]:


fault_sample_filt = flatiron(fault_sample.values)
normal_sample_filt = flatiron(normal_sample.values)


# In[ ]:


f, ax = plt.subplots(2, 2, figsize=(24, 16))

ax[0, 0].plot(fault_sample, alpha=0.8)
ax[0, 0].set_title('fault signal')
ax[0, 0].set_ylim([-100, 100])

ax[0, 1].plot(fault_sample_filt, alpha=0.5)
ax[0, 1].set_title('filtered fault signal')
ax[0, 1].set_ylim([-100, 100])

ax[1, 0].plot(normal_sample, alpha=0.7)
ax[1, 0].set_title('normal signal')
ax[1, 0].set_ylim([-100, 100])

ax[1, 1].plot(normal_sample_filt, alpha=0.5)
ax[1, 1].set_title('filtered normal signal')
ax[1, 1].set_ylim([-100, 100])


# In[ ]:




