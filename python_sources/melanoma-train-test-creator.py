#!/usr/bin/env python
# coding: utf-8

# The purpose of this kernel is just to create a few datasets that can be used for further exploration and modeling in other kernels. 

# It turns out that just the image metadata and and the image size contains a lot of useful information We'll start by creating feature-engineered datasets with just that information. The approach here follows the one in this notebook: https://www.kaggle.com/zzy990106/lgb-meta-data-image-size

# In[ ]:


import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import roc_auc_score

import lightgbm as lgb


# In[ ]:


train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
sample = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')


# In[ ]:


train['sex'] = train['sex'].fillna('na')
train['age_approx'] = train['age_approx'].fillna(0)
train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('na')

test['sex'] = test['sex'].fillna('na')
test['age_approx'] = test['age_approx'].fillna(0)
test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('na')


# In[ ]:


trn_images = train['image_name'].values
trn_sizes = np.zeros((trn_images.shape[0],2))
for i, img_path in enumerate(tqdm(trn_images)):
    img = Image.open(os.path.join('../input/siim-isic-melanoma-classification/jpeg/train/', f'{img_path}.jpg'))
    trn_sizes[i] = np.array([img.size[0],img.size[1]])


# In[ ]:


test_images = test['image_name'].values
test_sizes = np.zeros((test_images.shape[0],2))
for i, img_path in enumerate(tqdm(test_images)):
    img = Image.open(os.path.join('../input/siim-isic-melanoma-classification/jpeg/test/', f'{img_path}.jpg'))
    test_sizes[i] = np.array([img.size[0],img.size[1]])


# In[ ]:


train['w'] = trn_sizes[:,0]
train['h'] = trn_sizes[:,1]
test['w'] = test_sizes[:,0]
test['h'] = test_sizes[:,1]


# In[ ]:


le = preprocessing.LabelEncoder()

train.sex = le.fit_transform(train.sex)
train.anatom_site_general_challenge = le.fit_transform(train.anatom_site_general_challenge)
test.sex = le.fit_transform(test.sex)
test.anatom_site_general_challenge = le.fit_transform(test.anatom_site_general_challenge)


# In[ ]:


feature_names = ['sex','age_approx','anatom_site_general_challenge','w','h']
ycol = ['target']


# In[ ]:


train[feature_names + ycol].to_csv('train_meta_size.csv', index=False)
test[feature_names ].to_csv('test_meta_size.csv', index=False)


# The problem with the above approach is that we have very different distribution of missing values in train and test sets, so any algorithms that are sensitive to those discrepancies will lead to difference between the local CV and LB. We'll try to do somethign a bit more sophisticated now. 

# In[ ]:


train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')


# In[ ]:


test.head()


# In[ ]:


train.head()


# In[ ]:


np.unique(train.diagnosis.values, return_counts=True)


# In[ ]:


cols = ['sex', 'age_approx', 'anatom_site_general_challenge']

train_test = train[cols].append(test[cols])


# In[ ]:


train_test.shape


# In[ ]:


train_test['age_approx'].mean()


# In[ ]:


train_test['age_approx'] = train_test['age_approx'].fillna(train_test['age_approx'].mean())#float
train_test['sex'] = train_test['sex'].fillna(train_test['sex'].value_counts().index[0])
train_test['anatom_site_general_challenge'] = train_test['anatom_site_general_challenge'].fillna(train_test['anatom_site_general_challenge'].value_counts().index[0])


# In[ ]:


train[cols] = train_test[:train.shape[0]][cols].values
test[cols] = train_test[train.shape[0]:][cols].values


# In[ ]:


test.head()


# In[ ]:


trn_images = train['image_name'].values
trn_sizes = np.zeros((trn_images.shape[0],2))
for i, img_path in enumerate(tqdm(trn_images)):
    img = Image.open(os.path.join('../input/siim-isic-melanoma-classification/jpeg/train/', f'{img_path}.jpg'))
    trn_sizes[i] = np.array([img.size[0],img.size[1]])
    
    
test_images = test['image_name'].values
test_sizes = np.zeros((test_images.shape[0],2))
for i, img_path in enumerate(tqdm(test_images)):
    img = Image.open(os.path.join('../input/siim-isic-melanoma-classification/jpeg/test/', f'{img_path}.jpg'))
    test_sizes[i] = np.array([img.size[0],img.size[1]])


# In[ ]:


train['w'] = trn_sizes[:,0]
train['h'] = trn_sizes[:,1]
test['w'] = test_sizes[:,0]
test['h'] = test_sizes[:,1]


# In[ ]:


le = preprocessing.LabelEncoder()

le.fit(train_test.sex)

train.sex = le.transform(train.sex)
test.sex = le.transform(test.sex)

le = preprocessing.LabelEncoder()

le.fit(train_test.anatom_site_general_challenge)

train.anatom_site_general_challenge = le.transform(train.anatom_site_general_challenge)
test.anatom_site_general_challenge = le.transform(test.anatom_site_general_challenge)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train[feature_names + ycol].to_csv('train_meta_size_2.csv', index=False)
test[feature_names ].to_csv('test_meta_size_2.csv', index=False)


# In[ ]:


ycol


# We'll now add metafeatures from Chris Deotte's TF kernel:

# In[ ]:


oof_c = pd.read_csv('../input/triple-stratified-kfold-with-tfrecords/oof.csv')
submission_c = pd.read_csv('../input/triple-stratified-kfold-with-tfrecords/submission.csv')
oof_c.head()


# In[ ]:


del oof_c['target']
oof_c.head()


# In[ ]:


oof_c.shape


# In[ ]:


train.shape


# In[ ]:


train_2 = train[train['image_name'].isin(oof_c['image_name'].values)]


# In[ ]:


train_2 = train_2.merge(oof_c, on='image_name')


# In[ ]:


feature_names.append('pred')


# In[ ]:


submission_c.head()


# In[ ]:


test['pred'] = submission_c['target']
test.head()


# In[ ]:


ycol


# In[ ]:


train_2[feature_names + ['fold'] + ycol].to_csv('train_meta_size_3.csv', index=False)
test[feature_names ].to_csv('test_meta_size_3.csv', index=False)


# In[ ]:


train_2[feature_names + ['fold'] + ycol].head()


# In[ ]:


test[feature_names]


# In[ ]:


train_32 = np.load('../input/siimisic-melanoma-resized-images/x_train_32.npy')/255
test_32 = np.load('../input/siimisic-melanoma-resized-images/x_test_32.npy')/255


# In[ ]:


train_32 = train_32.reshape((train_32.shape[0], 32*32*3))
test_32 = test_32.reshape((test_32.shape[0], 32*32*3))


# In[ ]:


columns = [f'c_{i}' for i in range(3072)]


# In[ ]:


train_32 = pd.DataFrame(data = train_32, columns=columns)
test_32 = pd.DataFrame(data = test_32, columns=columns)


# In[ ]:


train_32['target'] = train['target']


# In[ ]:


train_32.to_csv('train_32.csv', index=False)
test_32.to_csv('test_32.csv', index=False)
np.save('columns_32', columns)


# In[ ]:




