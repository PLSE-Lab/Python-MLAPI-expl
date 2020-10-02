#!/usr/bin/env python
# coding: utf-8

# # TReNDS Neuroimaging - Data exploration

# In[ ]:


# Importing dependencies

from sklearn.model_selection import KFold, train_test_split
#import cudf
#from cuml import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#import joypy

from tqdm.notebook import tqdm
from glob import glob
import gc

import nilearn as nl
import nilearn.plotting as nlplt
import nibabel as nib

import h5py

import lightgbm as lgb

from scipy.stats import skew, kurtosis

import os, random

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from tensorflow.keras.utils import Sequence


# In[ ]:


# Loading train scores

MAIN_DATA_PATH = '/kaggle/input/trends-assessment-prediction/'

train_scores_df = pd.read_csv(MAIN_DATA_PATH + 'train_scores.csv')
icn_numbers_df = pd.read_csv(MAIN_DATA_PATH + 'ICN_numbers.csv')
loading_df = pd.read_csv(MAIN_DATA_PATH + 'loading.csv')
fnc_df = pd.read_csv(MAIN_DATA_PATH + 'fnc.csv')
print('Train Scores')
print(train_scores_df.head())
print('ICN Numbers')
print(icn_numbers_df.head())
print('Loading')
print(loading_df.head())
print('FNC')
print(fnc_df.head())


# ### <a href='#2-1'>Target distributions</a>

# The train_scores.csv file contains the targets that we need to predict.

# In[ ]:


# Plot the distribution of the target variables

fig, ax = plt.subplots(1, 5, figsize=(20, 5))

sns.distplot(train_scores_df['age'], ax=ax[0])
ax[0].set_title('Age')

sns.distplot(train_scores_df['domain1_var1'], ax=ax[1])
ax[1].set_title('Domain 1 - Var 1')

sns.distplot(train_scores_df['domain1_var2'], ax=ax[2])
ax[2].set_title('Domain 1 - Var 2')

sns.distplot(train_scores_df['domain2_var1'], ax=ax[3])
ax[3].set_title('Domain 2 - Var 1')

sns.distplot(train_scores_df['domain2_var2'], ax=ax[4])
ax[4].set_title('Domain 2 - Var 2')

fig.suptitle('Target distributions', fontsize=14)


# The distributions are bell-shaped. Age and domain 2 variables seems to have a slight skew. Furthermore, the kurtosis is small, meaning that there is not much weight in the tails.

# In[ ]:


# Compute statistics

print("Kurtosis (Fisher's definition)")
train_scores_df.kurtosis()


# Indeed, the target variables are normally-distributed. Note that Fisher's definition means that the kurtosis of a Gaussian distribution is 0.

# In[ ]:


round(train_scores_df.isna().sum() / len(train_scores_df) * 100, 2)


# In[ ]:


train_scores_df.fillna(train_scores_df.median(), inplace=True)
train_scores_df.isna().sum()


# ### <a href='#2-2'>SBM loadings</a>

# In[ ]:


loading_df.head()


# In[ ]:


targets = loading_df.columns[1:]

plt.figure(figsize=(16,10), dpi= 80)
fig, axes = joypy.joyplot(loading_df, column=list(targets), ylim='own', figsize=(14,10))

# Decoration
plt.title('Source-based morphometry loadings distribution', fontsize=22)
plt.show()


# Let's know build our training set composed of multiple sets of features.

# In[ ]:


features_df = pd.merge(train_scores_df, loading_df, on=['Id'], how='left')
features_df.head()


# Since we still have relatively few features, let's investigate the correlation between target variables and features.

# In[ ]:


fig, ax = plt.subplots(figsize=(20, 20))
cols = features_df.columns[1:]
sns.heatmap(features_df[cols].corr(), annot=True, cmap='RdYlGn', ax=ax)


# ### <a href='#2-3'>FNC correlation</a>

# This dataframe contains static FNC correlation features for both train and test samples.
# 
# As given in the competition data summary,
# 
# > The second set are static functional network connectivity (FNC) matrices. These are the subject-level cross-correlation values among 53 component timecourses estimated from group inform guided ICA of resting state functional MRI.

# #### What are functional network connectivity matrices?
# 
# Functional connectivity is the **connectivity between brain regions that share functional properties**. More specifically, it can be defined as the temporal correlation between spatially remote neurophysiological events, expressed as deviation from statistical independence across these events in distributed neuronal groups and areas. This applies to both resting state and task-state studies. While functional connectivity can refer to correlations across subjects, runs, blocks, trials, or individual time points, resting state functional connectivity focuses on connectivity assessed across individual BOLD time points during resting conditions. Functional connectivity has also been evaluated using the perfusion time series sampled with arterial spin labeled perfusion fMRI.
# 
# Source: https://en.wikipedia.org/wiki/Resting_state_fMRI#Functional

# In[ ]:


fnc_df.head()


# In[ ]:


# No NaN values in the DataFrame

fnc_df.isna().sum().sum()


# In[ ]:


features_df = pd.merge(features_df, fnc_df, how='left', on='Id')
features_df.head()


# ### <a href='#2-4'>Visualizing 3D spatial maps</a>

# Credits to: https://www.kaggle.com/soham1024/visualization-using-nilearn

# In[ ]:


get_ipython().system('wget https://github.com/Chaogan-Yan/DPABI/raw/master/Templates/ch2better.nii')


# In[ ]:


mask_filename = '../input/trends-assessment-prediction/fMRI_mask.nii'
smri_filename = 'ch2better.nii'

mask_niimg = nl.image.load_img(mask_filename)

def load_subject(filename, mask_niimg):
    subject_data = None
    
    with h5py.File(filename, 'r') as f:
        subject_data = f['SM_feature'][()]
        
    subject_data = np.moveaxis(subject_data, [0, 1, 2, 3], [3, 2, 1, 0])
    subject_niimg = nl.image.new_img_like(mask_niimg, subject_data, affine=mask_niimg.affine, copy_header=True)
    
    return subject_niimg 


# In[ ]:


fMRI_train_data_path = '../input/trends-assessment-prediction/fMRI_train/'
filenames = random.choices(os.listdir(fMRI_train_data_path), k=4)


# Using soham1024's function, we change the dimensions of the 4d array that is in the Matlab file. We flip the axes such that the first two axes are the width and height.

# In[ ]:


for filename in filenames:
    subject_filename = os.path.join(fMRI_train_data_path, filename)
    subject_niimg = load_subject(subject_filename, mask_niimg)

    print("Image shape is %s" % (str(subject_niimg.shape)))
    num_components = subject_niimg.shape[-1]
    print("Detected {num_components} spatial maps".format(num_components=num_components))

    nlplt.plot_prob_atlas(subject_niimg, 
                          bg_img=smri_filename,
                          view_type='filled_contours',
                          draw_cross=False,
                          title='All %d spatial maps' % num_components,
                          threshold='auto')


# #### Visualize independent components (IC)

# In[ ]:


filename = random.choice(os.listdir(fMRI_train_data_path))
subject_filename = os.path.join(fMRI_train_data_path, filename)
subject_niimg = load_subject(subject_filename, mask_niimg)

grid_size = int(np.ceil(np.sqrt(53)))
fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*10, grid_size*10))
[axi.set_axis_off() for axi in axes.ravel()]
row = -1 

for i, cur_img in enumerate(nl.image.iter_img(subject_niimg)):
    col = i % grid_size
    if col == 0:
        row += 1
    
    nlplt.plot_stat_map(cur_img,
                        bg_img=smri_filename,
                        title='IC %d' % i,
                        axes=axes[row, col],
                        threshold=3,
                        colorbar=False)


# ## <a href='#3'>Basic modelling</a>

# For a basic modelling using SVM, I hhave used RAPIDS SVM as shown in Ahmet Erdem's kernel: https://www.kaggle.com/aerdem4/rapids-svm-on-trends-neuroimaging and used feature engineering from https://www.kaggle.com/jafarib/trends-eda-fe-submission

# ### <a href='#3-1'> Preparing data</a>

# For this model, we will only be using basic features, and not the actual MRI image. As you'll see, the model works decently.

# In[ ]:


# Loading 
fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")
loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")
labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")

fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")
labels_df["is_train"] = True

df = df.merge(labels_df, on="Id", how="left")

target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']

test_df = df[df["is_train"] != True].copy()
train_df = df[df["is_train"] == True].copy()


# In[ ]:


y_train_df = train_df[target_cols]
train_df = train_df.drop(target_cols + ['is_train'], axis=1)
test_df = test_df.drop(target_cols + ['is_train'], axis=1)

FNC_SCALE = 1/500
test_df[fnc_features] *= FNC_SCALE
train_df[fnc_features] *= FNC_SCALE


# **We know that there is no data leakage between patients since each patient has a unique id.**

# ### <a href='#3-2'> Training</a>

# In[ ]:


def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))


# In[ ]:


param = {'objective':'regression',
        'metric':'rmse',
        'bossting_type':'gbdt',
        'learning_rate':0.005,
        'max_depth':-1}

output = pd.DataFrame()

for target in ['age','domain1_var1','domain1_var2','domain2_var1','domain2_var2']:
    
    X_train, X_val, y_train, y_val = train_test_split(train_df.iloc[:,1:], y_train_df[target], test_size=0.25, 
                                                      shuffle=True, random_state=20)
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(param, 
                      train_data, 
                      50000, 
                      #nfold=10,
                      early_stopping_rounds=500, 
                      valid_sets=[val_data], 
                      verbose_eval=50)
    
    temp = pd.DataFrame(test_df['Id'].apply(lambda x:str(x)+ '_'+ target))
    temp['Predicted'] = model.predict(test_df.iloc[:,1:])
    output = pd.concat([output,temp])    


# In[ ]:


sample_submission = pd.read_csv("/kaggle/input/trends-assessment-prediction/sample_submission.csv")
output = sample_submission.drop('Predicted',axis=1).merge(output,on='Id',how='left')


# In[ ]:


final_sub = pd.DataFrame(data={
    'id':sample_submission['Id'],
    'Predicted': 1*output['Predicted'] #+ 0.1*output_svm['Predicted']
})
final_sub.to_csv("submission.csv", index=False)

