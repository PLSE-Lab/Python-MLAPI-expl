#!/usr/bin/env python
# coding: utf-8

# This kernel is a fork from Jonne's kernel (https://www.kaggle.com/jonnedtc/cnn-segmentation-connected-components). Jonne creates a submission using a convolutional neural network. However, Jonne does not use any DICOM data for the prediction. I am creating this kernel to improve on Jonne's predictions by using the DICOM data. The model I am using is LightGBM, since it is fast, often accurate, and reliable.

# This kernel is also a fork from jtlowery's kernel (https://www.kaggle.com/jtlowery/intro-eda-with-dicom-metadata). Jtlowery's kernel has functions I can copy to read in the DICOM data.[](http://)

# In[ ]:


from functools import partial
from collections import defaultdict
import pydicom
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

np.warnings.filterwarnings('ignore')


# In[ ]:


labels = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_1_train_labels.csv')
details = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_1_detailed_class_info.csv')
# duplicates in details just have the same class so can be safely dropped
details = details.drop_duplicates('patientId').reset_index(drop=True)
labels_w_class = labels.merge(details, how='inner', on='patientId')


# In[ ]:


# get lists of all train/test dicom filepaths
train_dcm_fps = glob.glob('../input/rsna-pneumonia-detection-challenge/stage_1_train_images/*.dcm')
test_dcm_fps = glob.glob('../input/rsna-pneumonia-detection-challenge/stage_1_test_images/*.dcm')

train_dcms = [pydicom.read_file(x, stop_before_pixels=True) for x in train_dcm_fps]
test_dcms = [pydicom.read_file(x, stop_before_pixels=True) for x in test_dcm_fps]


# In[ ]:


def parse_dcm_metadata(dcm):
    unpacked_data = {}
    group_elem_to_keywords = {}
    # iterating here to force conversion from lazy RawDataElement to DataElement
    for d in dcm:
        pass
    # keys are pydicom.tag.BaseTag, values are pydicom.dataelem.DataElement
    for tag, elem in dcm.items():
        tag_group = tag.group
        tag_elem = tag.elem
        keyword = elem.keyword
        group_elem_to_keywords[(tag_group, tag_elem)] = keyword
        value = elem.value
        unpacked_data[keyword] = value
    return unpacked_data, group_elem_to_keywords

train_meta_dicts, tag_to_keyword_train = zip(*[parse_dcm_metadata(x) for x in train_dcms])
test_meta_dicts, tag_to_keyword_test = zip(*[parse_dcm_metadata(x) for x in test_dcms])


# In[ ]:


# join all the dicts
unified_tag_to_key_train = {k:v for dict_ in tag_to_keyword_train for k,v in dict_.items()}
unified_tag_to_key_test = {k:v for dict_ in tag_to_keyword_test for k,v in dict_.items()}

# quick check to make sure there are no different keys between test/train
assert len(set(unified_tag_to_key_test.keys()).symmetric_difference(set(unified_tag_to_key_train.keys()))) == 0

tag_to_key = {**unified_tag_to_key_test, **unified_tag_to_key_train}
tag_to_key


# In[ ]:


# using from_records here since some values in the dicts will be iterables and some are constants
train_df = pd.DataFrame.from_records(data=train_meta_dicts)
test_df = pd.DataFrame.from_records(data=test_meta_dicts)
train_df['dataset'] = 'train'
test_df['dataset'] = 'test'
df = pd.concat([train_df, test_df])


# In[ ]:


df.head(1)


# In[ ]:


# separating PixelSpacing list to single values
df['PixelSpacing_x'] = df['PixelSpacing'].apply(lambda x: x[0])
df['PixelSpacing_y'] = df['PixelSpacing'].apply(lambda x: x[1])
df = df.drop(['PixelSpacing'], axis='columns')

# x and y are always the same
assert sum(df['PixelSpacing_x'] != df['PixelSpacing_y']) == 0


# In[ ]:


# ReferringPhysicianName appears to just be empty strings
assert sum(df['ReferringPhysicianName'] != '') == 0

# SeriesDescription appears to be 'view: {}'.format(ViewPosition)
set(df['SeriesDescription'].unique())

# so these two columns don't have any useful info and can be safely dropped


# In[ ]:


nunique_all = df.aggregate('nunique')
nunique_all


# In[ ]:


# drop constant cols and other two from above
#ReferringPhysicianName is all ''
#PatientName is the same as PatientID
#PixelSpacing_y is the same as PixelSpacing_x
#The series and SOP UID's are just random numbers / id's, so I'm deleting them too
df = df.drop(nunique_all[nunique_all == 1].index.tolist() + ['SeriesDescription', 'ReferringPhysicianName', 'PatientName', 'PixelSpacing_y', 'SOPInstanceUID','SeriesInstanceUID','StudyInstanceUID'], axis='columns')

# now that we have a clean metadata dataframe we can merge back to our initial tabular data with target and class info
df = df.merge(labels_w_class, how='left', left_on='PatientID', right_on='patientId')

df['PatientAge'] = df['PatientAge'].astype(int)


# In[ ]:


# df now has multiple rows for some patients (those with multiple bounding boxes in label_w_class)
# so creating one with no duplicates for patients
df_deduped = df.drop_duplicates('PatientID', keep='first')


# In[ ]:


df_deduped.head()


# In[ ]:


#Correct ages that are mistyped
df_deduped.loc[df_deduped['PatientAge'] > 140, 'PatientAge'] = df_deduped.loc[df_deduped['PatientAge'] > 140, 'PatientAge'] - 100


# In[ ]:


#Convert binary features from categorical to 0/1
# Categorical features with Binary encode (0 or 1; two categories)
for bin_feature in ['PatientSex', 'ViewPosition']:
    df_deduped[bin_feature], uniques = pd.factorize(df_deduped[bin_feature])


# In[ ]:


#Drop the duplicated column patientID
del df_deduped['patientId']

#Drop columns that are going to be repetitive
del df_deduped['dataset']


# In[ ]:


df_deduped.head()


# Now that we have a data frame that links PatientID to DICOM data, let's merge this with train and the submission file.

# In[ ]:


jonneoofs = pd.read_csv("../input/jonneoofs/jonne_oofs.csv")
jonneoofs = jonneoofs.sort_values('patientID').reset_index(drop=True)
andyharless_sub = pd.read_csv("../input/andyharless/submission (7).csv")


# In[ ]:


labels.head() #The real train


# In[ ]:


jonneoofs.head() #The oofs from Jonne's kernel


# In[ ]:


andyharless_sub.head() # The submission from Andy Harless, which is a fork from Jonne


# In[ ]:


jonneoofs['i_am_train'] = 1
andyharless_sub['i_am_train'] = 0
tr_te = jonneoofs.append(andyharless_sub)


# In[ ]:


del tr_te['confidence'] #Not used in grading


# In[ ]:


tr_te.columns = ['PatientID','x_guess','y_guess','width_guess','height_guess','i_am_train']
tr_te.head()


# In[ ]:


df_deduped.head()


# In[ ]:


merged_df = tr_te.merge(df_deduped, how='left', on='PatientID')
merged_df.head()


# In[ ]:


filledmerged_df = merged_df.fillna(-1) #Fill in missings


# # Predict for x

# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold

train_df = filledmerged_df[filledmerged_df['i_am_train']==1]
test_df = filledmerged_df[filledmerged_df['i_am_train']==0]
             
#Cross validate with K Fold, 5 splits
folds = KFold(n_splits= 5, shuffle=True, random_state=2222)

# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
             
feats = [f for f in train_df.columns if f not in ['PatientID', 'i_am_train', 'x','y','width','height','Target','class']]

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats])):
    dtrain = lgb.Dataset(data=train_df[feats].iloc[train_idx], 
                         label=train_df['x'].iloc[train_idx], 
                         free_raw_data=False, silent=True)
    dvalid = lgb.Dataset(data=train_df[feats].iloc[valid_idx], 
                         label=train_df['x'].iloc[valid_idx], 
                         free_raw_data=False, silent=True)

    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'nthread': 4,
        'learning_rate': 0.10, 
        'max_depth': 2,
        #'reg_alpha': 0,
        #'reg_lambda': 0,
        #'min_split_gain': 0.0222415,
        'seed': 15000,
        'verbose': 50,
        'metric': 'l2',
    }

    clf = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dvalid],
        early_stopping_rounds=50,
        verbose_eval=True
    )

    oof_preds[valid_idx] = clf.predict(dvalid.data)
    sub_preds += clf.predict(test_df[feats]) / folds.n_splits


# In[ ]:


xpreds_oof = oof_preds.copy()
xpreds_sub = sub_preds.copy()


# # Predict for y

# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold

train_df = filledmerged_df[filledmerged_df['i_am_train']==1]
test_df = filledmerged_df[filledmerged_df['i_am_train']==0]
             
#Cross validate with K Fold, 5 splits
folds = KFold(n_splits= 5, shuffle=True, random_state=2222)

# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
             
feats = [f for f in train_df.columns if f not in ['PatientID', 'i_am_train', 'x','y','width','height','Target','class']]

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats])):
    dtrain = lgb.Dataset(data=train_df[feats].iloc[train_idx], 
                         label=train_df['y'].iloc[train_idx], 
                         free_raw_data=False, silent=True)
    dvalid = lgb.Dataset(data=train_df[feats].iloc[valid_idx], 
                         label=train_df['y'].iloc[valid_idx], 
                         free_raw_data=False, silent=True)

    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'nthread': 4,
        'learning_rate': 0.10, 
        'max_depth': 2,
        #'reg_alpha': 0,
        #'reg_lambda': 0,
        #'min_split_gain': 0.0222415,
        'seed': 15000,
        'verbose': 50,
        'metric': 'l2',
    }

    clf = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dvalid],
        early_stopping_rounds=50,
        verbose_eval=True
    )

    oof_preds[valid_idx] = clf.predict(dvalid.data)
    sub_preds += clf.predict(test_df[feats]) / folds.n_splits


# In[ ]:


ypreds_oof = oof_preds.copy()
ypreds_sub = sub_preds.copy()


# # Predict for width

# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold

train_df = filledmerged_df[filledmerged_df['i_am_train']==1]
test_df = filledmerged_df[filledmerged_df['i_am_train']==0]
             
#Cross validate with K Fold, 5 splits
folds = KFold(n_splits= 5, shuffle=True, random_state=2222)

# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
             
feats = [f for f in train_df.columns if f not in ['PatientID', 'i_am_train', 'x','y','width','height','Target','class']]

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats])):
    dtrain = lgb.Dataset(data=train_df[feats].iloc[train_idx], 
                         label=train_df['width'].iloc[train_idx], 
                         free_raw_data=False, silent=True)
    dvalid = lgb.Dataset(data=train_df[feats].iloc[valid_idx], 
                         label=train_df['width'].iloc[valid_idx], 
                         free_raw_data=False, silent=True)

    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'nthread': 4,
        'learning_rate': 0.10, 
        'max_depth': 2,
        #'reg_alpha': 0,
        #'reg_lambda': 0,
        #'min_split_gain': 0.0222415,
        'seed': 15000,
        'verbose': 50,
        'metric': 'l2',
    }

    clf = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dvalid],
        early_stopping_rounds=50,
        verbose_eval=True
    )

    oof_preds[valid_idx] = clf.predict(dvalid.data)
    sub_preds += clf.predict(test_df[feats]) / folds.n_splits


# In[ ]:


widthpreds_oof = oof_preds.copy()
widthpreds_sub = sub_preds.copy()


# # Predict for height

# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold

train_df = filledmerged_df[filledmerged_df['i_am_train']==1]
test_df = filledmerged_df[filledmerged_df['i_am_train']==0]
             
#Cross validate with K Fold, 5 splits
folds = KFold(n_splits= 5, shuffle=True, random_state=2222)

# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
             
feats = [f for f in train_df.columns if f not in ['PatientID', 'i_am_train', 'x','y','width','height','Target','class']]

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats])):
    dtrain = lgb.Dataset(data=train_df[feats].iloc[train_idx], 
                         label=train_df['height'].iloc[train_idx], 
                         free_raw_data=False, silent=True)
    dvalid = lgb.Dataset(data=train_df[feats].iloc[valid_idx], 
                         label=train_df['height'].iloc[valid_idx], 
                         free_raw_data=False, silent=True)

    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'nthread': 4,
        'learning_rate': 0.10, 
        'max_depth': 2,
        #'reg_alpha': 0,
        #'reg_lambda': 0,
        #'min_split_gain': 0.0222415,
        'seed': 15000,
        'verbose': 50,
        'metric': 'l2',
    }

    clf = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dvalid],
        early_stopping_rounds=50,
        verbose_eval=True
    )

    oof_preds[valid_idx] = clf.predict(dvalid.data)
    sub_preds += clf.predict(test_df[feats]) / folds.n_splits


# In[ ]:


heightpreds_oof = oof_preds.copy()
heightpreds_sub = sub_preds.copy()


# # Remove any boxes below a threshold

# In[ ]:


# What is the number of rows where we have a box?
train_df.loc[train_df['x'] > -1]['x'].shape[0] / train_df.shape[0]


# 0.22 rows have a box, so now let's cull our predictions until only there is 0.22

# In[ ]:


train_df['xpredsoof'] = xpreds_oof
train_df['ypredsoof'] = ypreds_oof
train_df['widthpredsoof'] = widthpreds_oof
train_df['heightpredsoof'] = heightpreds_oof


# In[ ]:


train_df.loc[train_df['widthpredsoof'] <= 100]


# In[ ]:


#train_df.loc[(train_df['xpredsoof'] > 130) & (train_df['ypredsoof'] > 134)].shape[0] / train_df.shape[0]
train_df.loc[(train_df['widthpredsoof'] > 100)].shape[0] / train_df.shape[0]


# In[ ]:


andyharless_sub['xpred'] = xpreds_sub
andyharless_sub['ypred'] = ypreds_sub
andyharless_sub['widthpred'] = widthpreds_sub
andyharless_sub['heightpred'] = heightpreds_sub

andyharless_sub['xpred'] = andyharless_sub['xpred'].round()
andyharless_sub['ypred'] = andyharless_sub['ypred'].round()
andyharless_sub['widthpred'] = andyharless_sub['widthpred'].round()
andyharless_sub['heightpred'] = andyharless_sub['heightpred'].round()


# In[ ]:


#andyharless_sub.loc[andyharless_sub['widthpred'] <= 100, 'xpred'] = ''
#andyharless_sub.loc[andyharless_sub['widthpred'] <= 100, 'ypred'] = ''
#andyharless_sub.loc[andyharless_sub['widthpred'] <= 100, 'heightpred'] = ''
#andyharless_sub.loc[andyharless_sub['widthpred'] <= 100, 'widthpred'] = ''
andyharless_sub['confidence'] = '1'


# In[ ]:


andyharless_sub.head()


# In[ ]:


#del andyharless_sub['x']
#del andyharless_sub['y']
#del andyharless_sub['width']
#del andyharless_sub['height']
#del andyharless_sub['i_am_train']


# In[ ]:


andyharless_sub['PredictionString'] = andyharless_sub['confidence'].map(str)+' '+andyharless_sub['xpred'].map(str)+' '+andyharless_sub['ypred'].map(str)+' '+andyharless_sub['widthpred'].map(str)+' '+andyharless_sub['heightpred'].map(str)


# In[ ]:


andyharless_sub.loc[andyharless_sub['PredictionString']=='1    ', 'PredictionString'] = '' #Correct empties


# In[ ]:


andyharless_sub.loc[andyharless_sub['x'].isnull(), 'PredictionString'] = '' #Remove boxes if we predicted there were none


# In[ ]:


andyharless_sub[['patientID','PredictionString']].to_csv('dicom_corrections.csv', index=False)


# In[ ]:




