#!/usr/bin/env python
# coding: utf-8

# # Practical EDA on numerical data

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


import pydicom


# In[ ]:


import gc
import warnings
warnings.simplefilter(action = 'ignore')


# In[ ]:


from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.model_selection import KFold


# ## Analisys of base information

# ### Loading data

# In[ ]:


detailed_class_info = pd.read_csv('../input/stage_1_detailed_class_info.csv')
train_labels = pd.read_csv('../input/stage_1_train_labels.csv')

df = pd.merge(left = detailed_class_info, right = train_labels, how = 'left', on = 'patientId')

del detailed_class_info, train_labels
gc.collect()

df.info(null_counts = True)


# In[ ]:


df.head()


# In[ ]:


df = df.drop_duplicates()
df.info()


# ### Rows per patientID

# In[ ]:


df['patientId'].value_counts().head(10)


# In[ ]:


df[df['patientId'] == '32408669-c137-4e8d-bd62-fe8345b40e73']


# Count of rows per patientID has 4 values:

# In[ ]:


df['patientId'].value_counts().value_counts()


# Each of patients without pneumonia has only one row in dataset:

# In[ ]:


df[df['Target'] == 0]['patientId'].value_counts().value_counts()


# ### Distribution of `class`

# In[ ]:


sns.countplot(x = 'class', hue = 'Target', data = df);


# In[ ]:


df[df['class'] == 'Lung Opacity']['Target'].value_counts(dropna = False)


# In[ ]:


df[df['class'] == 'No Lung Opacity / Not Normal']['Target'].value_counts(dropna = False)


# In[ ]:


df[df['class'] == 'Normal']['Target'].value_counts(dropna = False)


# Only class `Lung Opacity` has pneumonia on the train set.

# In[ ]:


print('Patients can have {} different classes'.format(df.groupby('patientId')['class'].nunique().nunique()))


# That is, "class == Lung Opacity" is equivalent to "Target == 1" or "image has pneumonia areas".

# ### Spatial features: x, y, width, height

# In[ ]:


df_areas = df.dropna()[['x', 'y', 'width', 'height']].copy()
df_areas['x_2'] = df_areas['x'] + df_areas['width']
df_areas['y_2'] = df_areas['y'] + df_areas['height']
df_areas['x_center'] = df_areas['x'] + df_areas['width'] / 2
df_areas['y_center'] = df_areas['y'] + df_areas['height'] / 2
df_areas['area'] = df_areas['width'] * df_areas['height']

df_areas.head()


# In[ ]:


sns.jointplot(x = 'x', y = 'y', data = df_areas, kind = 'hex', gridsize = 20);


# In[ ]:


sns.jointplot(x = 'x_center', y = 'y_center', data = df_areas, kind = 'hex', gridsize = 20);


# In[ ]:


sns.jointplot(x = 'x_2', y = 'y_2', data = df_areas, kind = 'hex', gridsize = 20);


# Centers and opposite corners have density more (variance less), than main corners (x, y). The centers have a high density and small correlation. 
# 
# There is no reason to replace (x, y) with (x_center, y_center) or (x_2, y_2).

# In[ ]:


sns.jointplot(x = 'width', y = 'height', data = df_areas, kind = 'hex', gridsize = 20);


# Widths and heights have a very hight correlation.

# In[ ]:


n_columns = 3
n_rows = 3
_, axes = plt.subplots(n_rows, n_columns, figsize=(8 * n_columns, 5 * n_rows))
for i, c in enumerate(df_areas.columns):
    sns.boxplot(y = c, data = df_areas, ax = axes[i // n_columns, i % n_columns])
plt.tight_layout()
plt.show()


# There are some outliers, especially for 'width' and 'height' features.

# In[ ]:


df_areas[df_areas['width'] > 500]


# In[ ]:


pid_width = list(df[df['width'] > 500]['patientId'].values)
df[df['patientId'].isin(pid_width)]


# One patient. Row can be dropped.

# In[ ]:


df_areas[df_areas['height'] > 900].shape[0]


# In[ ]:


pid_height = list(df[df['height'] > 900]['patientId'].values)
df[df['patientId'].isin(pid_height)]


# Two patients. All rows must be dropped together.

# In[ ]:


df = df[~df['patientId'].isin(pid_width + pid_height)]
df.shape


# ## Analisys of meta information

# In[ ]:


df_meta = df.drop('class', axis = 1).copy()


# In[ ]:


dcm_columns = None

for n, pid in enumerate(df_meta['patientId'].unique()):
    dcm_file = '../input/stage_1_train_images/%s.dcm' % pid
    dcm_data = pydicom.read_file(dcm_file)
    
    if not dcm_columns:
        dcm_columns = dcm_data.dir()
        dcm_columns.remove('PixelSpacing')
        dcm_columns.remove('PixelData')
    
    for col in dcm_columns:
        if not (col in df_meta.columns):
            df_meta[col] = np.nan
        index = df_meta[df_meta['patientId'] == pid].index
        df_meta.loc[index, col] = dcm_data.data_element(col).value
        
    del dcm_data
    
gc.collect()

df_meta.head()


# In[ ]:


to_drop = df_meta.nunique()
to_drop = to_drop[(to_drop <= 1) | (to_drop == to_drop['patientId'])].index
to_drop = to_drop.drop('patientId')
to_drop


# In[ ]:


df_meta.drop(to_drop, axis = 1, inplace = True)
df_meta.head()


# In[ ]:


print('Dropped {} useless features'.format(len(to_drop)))


# In[ ]:


df_meta.nunique()


# In[ ]:


sum(df_meta['ReferringPhysicianName'].unique() != '')


# In[ ]:


df_meta.drop('ReferringPhysicianName', axis = 1, inplace = True)


# Dropped one more useless feature

# In[ ]:


df_meta['PatientAge'] = df_meta['PatientAge'].astype(int)
df_meta['SeriesDescription'] = df_meta['SeriesDescription'].map({'view: AP': 'AP', 'view: PA': 'PA'})
df_meta.head()


# In[ ]:


print('There are {} equal elements between SeriesDescription and ViewPosition from {}.'       .format(sum(df_meta['SeriesDescription'] == df_meta['ViewPosition']), df_meta.shape[0]))


# In[ ]:


df_meta.drop('SeriesDescription', axis = 1, inplace = True)


# Dropped one feature wich is equal another.

# In[ ]:


plt.figure(figsize = (25, 5))
sns.countplot(x = 'PatientAge', hue = 'Target', data = df_meta);


# In[ ]:


sns.countplot(x = 'PatientSex', hue = 'Target', data = df_meta);


# In[ ]:


sns.countplot(x = 'ViewPosition', hue = 'Target', data = df_meta);


# In[ ]:


df_meta['PatientSex'] = df_meta['PatientSex'].map({'F': 0, 'M': 1})
df_meta['ViewPosition'] = df_meta['ViewPosition'].map({'PA': 0, 'AP': 1})
df_meta.head()


# In[ ]:


df_meta.corr()


# 'ViewPosition' have a high correlation with 'Target' and 'height' features. It can be useful...

# ## Attempt of forecasting of target and spatial variables according to meta information

# In[ ]:


def fast_lgbm_cv_scores(df, target, task, rs = 0):
    warnings.simplefilter('ignore')
    
    if task == 'classification':
        clf = LGBMClassifier(n_estimators = 10000, nthread = 4, random_state = rs)
        metric = 'auc'
    else:
        clf = LGBMRegressor(n_estimators = 10000, nthread = 4, random_state = rs)
        metric = 'mean_absolute_error'

    # Cross validation model
    folds = KFold(n_splits = 2, shuffle = True, random_state = rs)
        
    # Create arrays and dataframes to store results
    pred = np.zeros(df.shape[0])
    
    feats = df.columns.drop(target)
    
    feature_importance_df = pd.DataFrame(index = feats)
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df[feats], df[target])):
        train_x, train_y = df[feats].iloc[train_idx], df[target].iloc[train_idx]
        valid_x, valid_y = df[feats].iloc[valid_idx], df[target].iloc[valid_idx]

        clf.fit(train_x, train_y, 
                eval_set = [(valid_x, valid_y)], eval_metric = metric, 
                verbose = -1, early_stopping_rounds = 100)

        if task == 'classification':
            pred[valid_idx] = clf.predict_proba(valid_x, num_iteration = clf.best_iteration_)[:, 1]
        else:
            pred[valid_idx] = clf.predict(valid_x, num_iteration = clf.best_iteration_)
        
        feature_importance_df[n_fold] = pd.Series(clf.feature_importances_, index = feats)
        
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    if task == 'classification':    
        return feature_importance_df, pred, roc_auc_score(df[target], pred)
    else:
        return feature_importance_df, pred, mean_absolute_error(df[target], pred)


# In[ ]:


f_imp, _, score = fast_lgbm_cv_scores(df_meta.drop(['patientId', 'x', 'y', 'width', 'height'], axis = 1), 
                                      target = 'Target', task = 'classification')
print('ROC-AUC for Target = {}'.format(score))


# Score of prediction is rather high

# In[ ]:


f_imp


# In[ ]:


for c in ['x', 'y', 'width', 'height']:
    df_meta[c] = df_meta[c].fillna(-1)
df_meta.head()


# In[ ]:


f_imp, pred, score = fast_lgbm_cv_scores(df_meta[['x', 'PatientAge', 'PatientSex', 'ViewPosition']], 
                                   target = 'x', task = 'regression')
print('MAE for x = {}'.format(score))


# In[ ]:


val = df_meta[['x']]
val['pred'] = pred
val['error'] = abs(val['x'] - val['pred'])
val[['pred', 'error', 'x']].sort_values('x').reset_index(drop = True).plot();


# In[ ]:


f_imp


# In[ ]:


f_imp, pred, score = fast_lgbm_cv_scores(df_meta[['y', 'PatientAge', 'PatientSex', 'ViewPosition']], 
                                   target = 'y', task = 'regression')
print('MAE for y = {}'.format(score))


# In[ ]:


val = df_meta[['y']]
val['pred'] = pred
val['error'] = abs(val['y'] - val['pred'])
val[['pred', 'error', 'y']].sort_values('y').reset_index(drop = True).plot();


# In[ ]:


f_imp


# In[ ]:


f_imp, pred, score = fast_lgbm_cv_scores(df_meta[['width', 'PatientAge', 'PatientSex', 'ViewPosition']], 
                                   target = 'width', task = 'regression')
print('MAE for width = {}'.format(score))


# In[ ]:


val = df_meta[['width']]
val['pred'] = pred
val['error'] = abs(val['width'] - val['pred'])
val[['pred', 'error', 'width']].sort_values('width').reset_index(drop = True).plot();


# In[ ]:


f_imp


# In[ ]:


f_imp, pred, score = fast_lgbm_cv_scores(df_meta[['height', 'PatientAge', 'PatientSex', 'ViewPosition']], 
                                   target = 'height', task = 'regression')
print('MAE for height = {}'.format(score))


# In[ ]:


val = df_meta[['height']]
val['pred'] = pred
val['error'] = abs(val['height'] - val['pred'])
val[['pred', 'error', 'height']].sort_values('height').reset_index(drop = True).plot();


# In[ ]:


f_imp


# It can be useful to predict Target for selecting images with pneumonia.
# 
# There is no useful information in meta data for prediction spatial features directly.

# # Conclusion

# -  "class == Lung Opacity" is equivalent to "Target == 1" or "image has pneumonia areas". But the advantage of it is doubtful, because test images have not such information.
# 
# - 5 rows (3 patients) have been dropped as obvious outliers.
# 
# - It can be useful to predict 'Target' directly with meta information ('PatientAge', 'PatientSex', 'ViewPosition') for preliminary selecting images with pneumonia from the test set.
# 
# - There is no useful information in meta data for directly prediction spatial features.

# In[ ]:




