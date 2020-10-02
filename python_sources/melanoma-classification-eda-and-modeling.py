#!/usr/bin/env python
# coding: utf-8

# ## Overview
# 
# The purpose of this kernel is to take a look at the data, come up with some insights, and attempt to create a predictive model or two. This notebook is still **very** raw. I will work on it as my very limited time permits, and hope to expend it in the upcoming days and weeks.
# 
# 
# ## Packages
# 
# First, let's load a few useful Python packages. This section will keep growing in subsequent versions of this EDA.

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import json
import math
import cv2
import PIL
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from sklearn.decomposition import PCA
import os
import imagesize

get_ipython().run_line_magic('matplotlib', 'inline')


# Let's see what files we have in the input directory:

# In[ ]:


import os
print(os.listdir("../input/siim-isic-melanoma-classification"))


# In[ ]:


#Loading Train and Test Data
train = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
test = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
print("{} images in train set.".format(train.shape[0]))
print("{} images in test set.".format(test.shape[0]))


# In[ ]:


train.head()


# In[ ]:


test.head()


# Let's look at the distribution of teh target:

# In[ ]:


np.mean(train.target)


# So this is a binary classification problem with highly imbalanced data.

# Let's now look at the distributions of various "features"

# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(train['age_approx'].values, bins=200)
plt.title('Histogram age_approx counts in train')
plt.xlabel('Value')
plt.ylabel('Count')
plt.show()


# Let's take a look at a few images.

# In[ ]:


images = []
for i, image_id in enumerate(tqdm(train['image_name'].head(10))):
    im = Image.open(f'../input/siim-isic-melanoma-classification/jpeg/train/{image_id}.jpg')
    im = im.resize((128, )*2, resample=Image.LANCZOS)
    images.append(im)
    


# In[ ]:


images[0]


# In[ ]:


images[1]


# In[ ]:


images[3]


# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(test['age_approx'].values, bins=200)
plt.title('Histogram age_approx counts in test')
plt.xlabel('Value')
plt.ylabel('Count')
plt.show()


# Now we will load some of the resized images (32x32 for now) and try to build some simple models. 

# In[ ]:


x_train_32 = np.load('../input/siimisic-melanoma-resized-images/x_train_32.npy')
x_test_32 = np.load('../input/siimisic-melanoma-resized-images/x_test_32.npy')


# In[ ]:


x_train_32.shape


# In[ ]:


x_train_32 = x_train_32.reshape((x_train_32.shape[0], 32*32*3))
x_train_32.shape


# In[ ]:


x_test_32 = x_test_32.reshape((x_test_32.shape[0], 32*32*3))
x_test_32.shape


# In[ ]:


y = train.target.values


# In[ ]:


train_oof = np.zeros((x_train_32.shape[0], ))
test_preds = 0
train_oof.shape


# In[ ]:


n_splits = 5
kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)

for jj, (train_index, val_index) in enumerate(kf.split(x_train_32)):
    print("Fitting fold", jj+1)
    train_features = x_train_32[train_index]
    train_target = y[train_index]
    
    val_features = x_train_32[val_index]
    val_target = y[val_index]
    
    model = LogisticRegression(C=1, solver='lbfgs', multi_class='multinomial', max_iter=60)
    model.fit(train_features, train_target)
    val_pred = model.predict_proba(val_features)[:,1]
    train_oof[val_index] = val_pred
    print("Fold AUC:", roc_auc_score(val_target, val_pred))
    test_preds += model.predict_proba(x_test_32)[:,1]/n_splits
    del train_features, train_target, val_features, val_target
    gc.collect()


# In[ ]:


print(roc_auc_score(y, train_oof))


# In[ ]:


train_oof_0_2 = np.zeros((x_train_32.shape[0], ))
test_preds_0_2 = 0

n_splits = 5
kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)

for jj, (train_index, val_index) in enumerate(kf.split(x_train_32)):
    print("Fitting fold", jj+1)
    train_features = x_train_32[train_index]
    train_target = y[train_index]
    
    val_features = x_train_32[val_index]
    val_target = y[val_index]
    
    model = LogisticRegression(C=5, solver='lbfgs', multi_class='multinomial', max_iter=80)
    model.fit(train_features, train_target)
    val_pred = model.predict_proba(val_features)[:,1]
    train_oof_0_2[val_index] = val_pred
    print("Fold AUC:", roc_auc_score(val_target, val_pred))
    test_preds_0_2 += model.predict_proba(x_test_32)[:,1]/n_splits
    del train_features, train_target, val_features, val_target
    gc.collect()
    
print(roc_auc_score(y, train_oof_0_2))


# In[ ]:


print(roc_auc_score(y, 0.95*train_oof+0.05*train_oof_0_2))


# In[ ]:


train['age_approx'].unique()


# Wow, so we get an 0.82 AUC with just unravelled resized images and a simple Logistic Regression!

# Let's now add some non-image features. We can start with sex, and one-hot encode it.

# In[ ]:


train['sex'] = (train['sex'].values == 'male')*1
test['sex'] = (test['sex'].values == 'male')*1
train.head()


# In[ ]:


test.head()


# In[ ]:


train['sex'].mean()


# In[ ]:


test['sex'].mean()


# In[ ]:


train['age_approx'].mean()


# In[ ]:


test['age_approx'].mean()


# In[ ]:


train['age_approx'] = train['age_approx'].fillna(train['age_approx'].mean())
test['age_approx'] = test['age_approx'].fillna(test['age_approx'].mean())


# In[ ]:


x_train_32 = np.hstack([x_train_32, train['sex'].values.reshape(-1,1), train['age_approx'].values.reshape(-1,1)])
x_test_32 = np.hstack([x_test_32, test['sex'].values.reshape(-1,1), test['age_approx'].values.reshape(-1,1)])


# In[ ]:


train_oof_2 = np.zeros((x_train_32.shape[0], ))
test_preds_2 = 0


n_splits = 5
kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)

for jj, (train_index, val_index) in enumerate(kf.split(x_train_32)):
    print("Fitting fold", jj+1)
    train_features = x_train_32[train_index]
    train_target = y[train_index]
    
    val_features = x_train_32[val_index]
    val_target = y[val_index]
    
    model = LogisticRegression(C=1, solver='lbfgs', multi_class='multinomial', max_iter=50)
    model.fit(train_features, train_target)
    val_pred = model.predict_proba(val_features)[:,1]
    train_oof_2[val_index] = val_pred
    print("Fold AUC:", roc_auc_score(val_target, val_pred))
    test_preds_2 += model.predict_proba(x_test_32)[:,1]/n_splits
    del train_features, train_target, val_features, val_target
    gc.collect()


# In[ ]:


print(roc_auc_score(y, train_oof_2))


# In[ ]:


print(roc_auc_score(y, 0.8*train_oof_2+0.2*train_oof))


# In[ ]:


print(roc_auc_score(y, 0.5*train_oof_2+0.5*train_oof))


# In[ ]:


test_preds.max()


# In[ ]:


test_preds_2.max()


# In[ ]:


train_oof_2_2 = np.zeros((x_train_32.shape[0], ))
test_preds_2_2 = 0


n_splits = 5
kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)

for jj, (train_index, val_index) in enumerate(kf.split(x_train_32)):
    print("Fitting fold", jj+1)
    train_features = x_train_32[train_index]
    train_target = y[train_index]
    
    val_features = x_train_32[val_index]
    val_target = y[val_index]
    
    model = LogisticRegression(C=5, solver='lbfgs', multi_class='multinomial', max_iter=80)
    model.fit(train_features, train_target)
    val_pred = model.predict_proba(val_features)[:,1]
    train_oof_2_2[val_index] = val_pred
    print("Fold AUC:", roc_auc_score(val_target, val_pred))
    test_preds_2_2 += model.predict_proba(x_test_32)[:,1]/n_splits
    del train_features, train_target, val_features, val_target
    gc.collect()


# In[ ]:


print(roc_auc_score(y, train_oof_2_2))


# In[ ]:


train['anatom_site_general_challenge'].unique()


# In[ ]:


test['anatom_site_general_challenge'].unique()


# In[ ]:


train['anatom_site_general_challenge'].mode()


# In[ ]:


test['anatom_site_general_challenge'].mode()


# In[ ]:


train['anatom_site_general_challenge'].fillna(train['anatom_site_general_challenge'].mode(), inplace=True)
test['anatom_site_general_challenge'].fillna(test['anatom_site_general_challenge'].mode(), inplace=True)


# In[ ]:


train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].astype(str)
test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].astype(str)


# In[ ]:


test['anatom_site_general_challenge'].isnull().sum()


# In[ ]:


x_train_32 = np.hstack([x_train_32, pd.get_dummies(train['anatom_site_general_challenge']).values])
x_test_32 = np.hstack([x_test_32, pd.get_dummies(test['anatom_site_general_challenge']).values])


# In[ ]:


train_oof_3 = np.zeros((x_train_32.shape[0], ))
test_preds_3 = 0


n_splits = 5
kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)

for jj, (train_index, val_index) in enumerate(kf.split(x_train_32)):
    print("Fitting fold", jj+1)
    train_features = x_train_32[train_index]
    train_target = y[train_index]
    
    val_features = x_train_32[val_index]
    val_target = y[val_index]
    
    model = LogisticRegression(C=1, solver='lbfgs', multi_class='multinomial', max_iter=60)
    model.fit(train_features, train_target)
    val_pred = model.predict_proba(val_features)[:,1]
    train_oof_3[val_index] = val_pred
    print("Fold AUC:", roc_auc_score(val_target, val_pred))
    test_preds_3 += model.predict_proba(x_test_32)[:,1]/n_splits
    del train_features, train_target, val_features, val_target
    gc.collect()


# In[ ]:


print(roc_auc_score(y, train_oof_3))


# In[ ]:


train_oof_4 = np.zeros((x_train_32.shape[0], ))
test_preds_4 = 0


n_splits = 5
kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)

for jj, (train_index, val_index) in enumerate(kf.split(x_train_32)):
    print("Fitting fold", jj+1)
    train_features = x_train_32[train_index]
    train_target = y[train_index]
    
    val_features = x_train_32[val_index]
    val_target = y[val_index]
    
    model = LogisticRegression(C=5, max_iter=80)
    model.fit(train_features, train_target)
    val_pred = model.predict_proba(val_features)[:,1]
    train_oof_4[val_index] = val_pred
    print("Fold AUC:", roc_auc_score(val_target, val_pred))
    test_preds_4 += model.predict_proba(x_test_32)[:,1]/n_splits
    del train_features, train_target, val_features, val_target
    gc.collect()


# In[ ]:


print(roc_auc_score(y, train_oof_4))


# In[ ]:


print(roc_auc_score(y, 0.25*train_oof_2+0.25*train_oof+0.25*train_oof_3+0.25*train_oof_4))


# Taken from raddar's notebook: https://www.kaggle.com/raddar/simple-baseline-revamped

# In[ ]:


1


# In[ ]:


im_shape_test = []
im_shape_train = []

for i in range(train.shape[0]):
    im_shape_train.append(imagesize.get('../input/siim-isic-melanoma-classification/jpeg/train/'+train['image_name'][i]+'.jpg'))
for i in range(test.shape[0]):
    im_shape_test.append(imagesize.get('../input/siim-isic-melanoma-classification/jpeg/test/'+test['image_name'][i]+'.jpg'))
    

train['dim'] = im_shape_train
test['dim'] = im_shape_test


# In[ ]:


train['dim'] == (6000,4000)


# In[ ]:


train['dim'] == (1872,1053)


# In[ ]:


(train['dim'] != (6000,4000)) & (train['dim'] != (1872,1053))


# In[ ]:


train['dim_1'] = (train['dim'] == (6000,4000))
train['dim_1'] = train['dim_1'].values*1
train['dim_2'] = (train['dim'] == (1872,1053))
train['dim_2'] = train['dim_2'].values*1
train['dim_3'] = (train['dim'] != (6000,4000)) & (train['dim'] != (1872,1053))
train['dim_3'] = train['dim_3'].values*1
train['dim_3']


# In[ ]:


test['dim_1'] = (test['dim'] == (6000,4000))
test['dim_1'] = test['dim_1'].values*1
test['dim_2'] = (test['dim'] == (1872,1053))
test['dim_2'] = test['dim_2'].values*1
test['dim_3'] = (test['dim'] != (6000,4000)) & (test['dim'] != (1872,1053))
test['dim_3'] = test['dim_3'].values*1
test['dim_3']


# In[ ]:


x_train_32 = np.hstack([x_train_32, train['dim_1'].values.reshape(-1,1), train['dim_2'].values.reshape(-1,1), train['dim_3'].values.reshape(-1,1)])
x_test_32 = np.hstack([x_test_32, test['dim_1'].values.reshape(-1,1), test['dim_2'].values.reshape(-1,1), test['dim_3'].values.reshape(-1,1)])


# In[ ]:


train_oof_4_2 = np.zeros((x_train_32.shape[0], ))
test_preds_4_2 = 0


n_splits = 5
kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)

for jj, (train_index, val_index) in enumerate(kf.split(x_train_32)):
    print("Fitting fold", jj+1)
    train_features = x_train_32[train_index]
    train_target = y[train_index]
    
    val_features = x_train_32[val_index]
    val_target = y[val_index]
    
    model = LogisticRegression(C=0.9, max_iter=50)
    model.fit(train_features, train_target)
    val_pred = model.predict_proba(val_features)[:,1]
    train_oof_4_2[val_index] = val_pred
    print("Fold AUC:", roc_auc_score(val_target, val_pred))
    test_preds_4_2 += model.predict_proba(x_test_32)[:,1]/n_splits
    del train_features, train_target, val_features, val_target
    gc.collect()
    
print(roc_auc_score(y, train_oof_4_2))


# In[ ]:


0.8262080489449671


# In[ ]:


get_ipython().run_cell_magic('time', '', 'pca = PCA(n_components=0.99)\npca.fit(x_train_32)')


# In[ ]:


pca.n_components_


# In[ ]:


x_train_32 = pca.transform(x_train_32)
x_test_32 = pca.transform(x_test_32)


# In[ ]:


train_oof_5 = np.zeros((x_train_32.shape[0], ))
test_preds_5 = 0


n_splits = 5
kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)

for jj, (train_index, val_index) in enumerate(kf.split(x_train_32)):
    print("Fitting fold", jj+1)
    train_features = x_train_32[train_index]
    train_target = y[train_index]
    
    val_features = x_train_32[val_index]
    val_target = y[val_index]
    
    model = LogisticRegression(C=0.1, max_iter=6)
    model.fit(train_features, train_target)
    val_pred = model.predict_proba(val_features)[:,1]
    train_oof_5[val_index] = val_pred
    print("Fold AUC:", roc_auc_score(val_target, val_pred))
    test_preds_5 += model.predict_proba(x_test_32)[:,1]/n_splits
    del train_features, train_target, val_features, val_target
    gc.collect()
    
print(roc_auc_score(y, train_oof_5))


# In[ ]:


0.7910534268464863


# In[ ]:


print(roc_auc_score(y, 0.988*(0.27*train_oof_2+0.27*train_oof+0.27*train_oof_3+0.19*train_oof_4)+0.012*train_oof_5))


# In[ ]:


print(roc_auc_score(y, 1.082*(0.99*(0.25*train_oof_2+0.25*train_oof+0.25*train_oof_3+0.25*train_oof_4)+0.01*train_oof_5)-0.082*(train_oof_0_2+train_oof_2_2)/2))


# How about 64x64 images?

# In[ ]:


x_train_64 = np.load('../input/siimisic-melanoma-resized-images/x_train_64.npy')
x_test_64 = np.load('../input/siimisic-melanoma-resized-images/x_test_64.npy')


# In[ ]:


x_train_64 = x_train_64.reshape((x_train_64.shape[0], 64*64*3))
x_train_64.shape


# In[ ]:


x_test_64 = x_test_64.reshape((x_test_64.shape[0], 64*64*3))
x_test_64.shape


# In[ ]:


train_oof_6 = np.zeros((x_train_64.shape[0], ))
test_preds_6 = 0


n_splits = 5
kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)

for jj, (train_index, val_index) in enumerate(kf.split(x_train_64)):
    print("Fitting fold", jj+1)
    train_features = x_train_64[train_index]
    train_target = y[train_index]
    
    val_features = x_train_64[val_index]
    val_target = y[val_index]
    
    model = LogisticRegression(C=0.1, max_iter=45)
    model.fit(train_features, train_target)
    val_pred = model.predict_proba(val_features)[:,1]
    train_oof_6[val_index] = val_pred
    print("Fold AUC:", roc_auc_score(val_target, val_pred))
    test_preds_6 += model.predict_proba(x_test_64)[:,1]/n_splits
    del train_features, train_target, val_features, val_target
    gc.collect()
    
print(roc_auc_score(y, train_oof_6))


# In[ ]:


0.8213209504598062


# In[ ]:


print(roc_auc_score(y, 0.73*(1.1*(0.99*(0.25*train_oof_2+0.25*train_oof+0.25*train_oof_3+0.25*train_oof_4)+0.01*train_oof_5)-0.1*(train_oof_0_2+train_oof_2_2)/2)+0.27*train_oof_6))


# In[ ]:


print(roc_auc_score(y, 0.9*(0.73*(1.1*(0.99*(0.25*train_oof_2+0.25*train_oof+0.25*train_oof_3+0.25*train_oof_4)+0.01*train_oof_5)-0.1*(train_oof_0_2+train_oof_2_2)/2)+0.27*train_oof_6)+
                   0.1*train_oof_4_2))


# In[ ]:


0.8285058171399994


# Now let's make a submission.

# In[ ]:


sample_submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
sample_submission.head()


# In[ ]:


sample_submission['target'] = 0.9*(0.73*(1.1*(0.99*(0.25*test_preds+0.25*test_preds_2+0.25*test_preds_3+0.25*test_preds_4)+0.015*test_preds_5)- 0.1*(0.5*test_preds_0_2+0.5*test_preds_2_2))+0.27*test_preds_6)+0.1*test_preds_4_2
sample_submission.to_csv('submission_32x32_64x64_lr.csv', index=False)


# In[ ]:


sample_submission['target'].max()


# In[ ]:


sample_submission['target'].min()


# In[ ]:




