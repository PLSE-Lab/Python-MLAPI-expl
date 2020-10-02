#!/usr/bin/env python
# coding: utf-8

# In[ ]:


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.rcParams.update(params)
import h5py
import random
from tqdm.notebook import tqdm
import nibabel as nib
from glob import glob
from scipy.stats import ks_2samp


# ## Reading using h5py

# In[ ]:


f = h5py.File('/kaggle/input/trends-assessment-prediction/fMRI_test/20305.mat','r')


# ## Checking its keys

# In[ ]:


f.keys()


# ## Getting data and checking shape

# In[ ]:


data = f['SM_feature']
#array = data.value   
array = data[()]
array.shape   # it is 4D tensor


# ## Checking some statistics

# In[ ]:


stats = pd.DataFrame({"Min" : array.min() , "Max" : array.max() , 'Std': array.std() , 'Mean' : array.mean()}, index = ['Values'])
print(stats)


# ## Plotting few random planes 

# In[ ]:


f, ax = plt.subplots(1,4)
f.set_size_inches(25, 10)
for i in range(4):
    ax[i].imshow(array[i*5, :, 10, :], cmap = 'gray')


# In[ ]:


# Highlighting the values 

f, ax = plt.subplots(1,4)
f.set_size_inches(25, 10)
for i in range(4):
    Temp = array[i*10, :, 10, :] !=0  
    ax[i].imshow(Temp)


# In[ ]:


list_train_path = glob("/kaggle/input/trends-assessment-prediction/fMRI_train/*")
list_test_path = glob("/kaggle/input/trends-assessment-prediction/fMRI_test/*")


# In[ ]:


print(f"Number of Train .mat files - {len(list_train_path)}" )
print(f"Number of Test .mat files  - {len(list_test_path)}" )


# ## Plotting Distribution of Values :: Flattening into 1D array

# In[ ]:


f, ax = plt.subplots(1,3)
f.set_size_inches(25, 10)

for counter, i in enumerate(random.choices(list_train_path, k = 3)):
    array_iter = h5py.File(i,'r')['SM_feature'][()]
    ax[counter].hist(array_iter.flatten(), bins = 20, 
                     color = 'orange', edgecolor = 'black')


# In[ ]:


f, ax = plt.subplots(1,3)
f.set_size_inches(25, 10)

for counter, i in enumerate(random.choices(list_test_path, k = 3)):
    array_iter = h5py.File(i,'r')['SM_feature'][()]
    ax[counter].hist(array_iter.flatten(), bins = 20, 
                     color = 'orange', edgecolor = 'black')


# ## Exploring CSV Files

# In[ ]:


df_loading = pd.read_csv("/kaggle/input/trends-assessment-prediction/loading.csv")  # This data consist of both train as well test information
df_fnc = pd.read_csv("/kaggle/input/trends-assessment-prediction/fnc.csv")            # This data consist of both train as well test information
df_train_score = pd.read_csv("/kaggle/input/trends-assessment-prediction/train_scores.csv")   # This data consist of only train information


# In[ ]:


df_train_score.head()


# In[ ]:


df_loading.head()


# In[ ]:


df_fnc.head()


# ## Splitting the DataFrame into Train and Test

# In[ ]:


df_loading_train = df_loading[df_loading.Id.isin(df_train_score.Id)]
df_loading_test = df_loading[~df_loading.Id.isin(df_train_score.Id)]


# In[ ]:


print(df_loading_train.shape)
df_loading_train.head()


# In[ ]:


print(df_loading_test.shape)
df_loading_test.head()


# In[ ]:


df_fnc_train = df_fnc[df_fnc.Id.isin(df_train_score.Id)]
df_fnc_test = df_fnc[~df_fnc.Id.isin(df_train_score.Id)]


# In[ ]:


print(df_fnc_train.shape)
df_fnc_train.head()


# In[ ]:


print(df_fnc_test.shape)
df_fnc_test.head()


# ## Merging Train and Test : Loading and FNC

# In[ ]:


df_train_loading_fcn = pd.merge(df_loading_train, df_fnc_train, on = 'Id', how = 'inner')
df_test_loading_fcn =   pd.merge(df_loading_test, df_fnc_test, on = 'Id', how = 'inner')

print(df_train_loading_fcn.shape)
print(df_test_loading_fcn.shape)


# In[ ]:


columns = df_train_loading_fcn.columns
df_test_loading_fcn = df_test_loading_fcn[columns]


# ## K-S Test to check the similarity of train and test distribution

# In[ ]:


p_value = .000005   # you can change the p-value and experiment
list_p_value =[]

for i in tqdm(columns[1:]):
    list_p_value.append(ks_2samp(df_test_loading_fcn[i] , df_train_loading_fcn[i])[1])

Se = pd.Series(list_p_value, index = columns[1:]).sort_values() 
list_dissimilar = list(Se[Se < p_value].index)


# In[ ]:


len(list_dissimilar)


# ## Plotting Distribution Train (red) vs Test (blue)

# In[ ]:


for i in tqdm(range(len(list_dissimilar))[::2]):
    f, ax = plt.subplots(1,2)
    f.set_size_inches(20, 8)

    try:
        ax[0].hist(df_train_loading_fcn[list_dissimilar[i]], bins = 20, edgecolor = 'black', color = 'red')
        ax[0].hist(df_test_loading_fcn[list_dissimilar[i]], bins = 20, edgecolor = 'black', alpha = .5, color = 'blue')
        ax[0].title.set_text(list_dissimilar[i])

        ax[1].hist(df_train_loading_fcn[list_dissimilar[i+1]], bins = 20, edgecolor = 'black', color = 'red')
        ax[1].hist(df_test_loading_fcn[list_dissimilar[i+1]], bins = 20, edgecolor = 'black', alpha = .5, color = 'blue')
        ax[1].title.set_text(list_dissimilar[i+1])
        
#         ax[2].hist(df_train_loading_fcn[list_dissimilar[i+2]], bins = 20, edgecolor = 'black', color = 'red')
#         ax[2].hist(df_test_loading_fcn[list_dissimilar[i+2]], bins = 20, edgecolor = 'black', alpha = .5, color = 'blue')
#         ax[2].title.set_text(list_dissimilar[i+2])
    
    except:
        pass

 


# In[ ]:


img = nib.load("/kaggle/input/trends-assessment-prediction/fMRI_mask.nii")


# In[ ]:


img.shape


# In[ ]:


img.get_data_dtype()


# In[ ]:


hdr = img.header


# In[ ]:


hdr


# In[ ]:


hdr.get_xyzt_units()


# In[ ]:




