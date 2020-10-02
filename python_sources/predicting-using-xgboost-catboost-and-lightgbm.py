#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


# In[ ]:


sc = StandardScaler()
le = LabelEncoder()
onehot = OneHotEncoder(sparse=False)


# In[ ]:


# Reading train.csv
train_df = pd.read_csv('../input/train.csv')
train_df.head()


# In[ ]:


train_df.shape


# In[ ]:


# Getting the description of train.csv file
train_df.describe()


# In[ ]:


# Distribution of target classes. As, we can see the target classes are highly imbalanced, espeicially class 4 taking majority of samples
sns.countplot(train_df['Target'])
plt.show()


# In[ ]:


# Reading test.csv file
test_df = pd.read_csv('../input/test.csv')
test_df.head()


# In[ ]:


test_df.shape


# In[ ]:


# Combining both train and test file into one. This will help us preprocessing both files simultaneously, and after we are done with that, we can seperate both.
all_df = train_df.append(test_df, sort=False)
all_df.shape


# In[ ]:


# Checking fraction of null values in each feature column (ignore the Target variable as its null for the test.csv file)
missing_vals = (all_df.isnull().sum() / len(all_df)).sort_values(ascending=False)
missing_vals = missing_vals[missing_vals > 0]
missing_vals = missing_vals.to_frame()
missing_vals.columns = ['count']
missing_vals.index.names = ['Name']
missing_vals['Name'] = missing_vals.index

sns.set(style="whitegrid", color_codes=True)
sns.barplot(x = 'Name', y = 'count', data=missing_vals)
plt.xticks(rotation = 90)
plt.show()


# In[ ]:


# Dropping columns having too much null values. Filling null values by interpolating in such feature columns can lead to misleading data, so its better to drop them.
# Filling feature columns with too little null values with their median values, as the Target classes are imbalanced, its a good idea to replace null values with
# median values rather than mean values
all_df.drop(['rez_esc', 'v18q1', 'v2a1'], axis=1, inplace=True)
all_df.fillna({'SQBmeaned': all_df['SQBmeaned'].median(), 'meaneduc': all_df['meaneduc'].median()}, inplace=True)


# In[ ]:


# dividing feature columns according to their dtypes, so that we can visualize them further
float_cols = [col for col in all_df.columns if all_df[col].dtype=='float64']
int_cols = [col for col in all_df.columns if all_df[col].dtype=='int64']
object_cols = [col for col in all_df.columns if all_df[col].dtype=='object']


# In[ ]:


# Removing Target feature from list and visualizing float feature columns
del(float_cols[-1])
float_flat = pd.melt(all_df, value_vars=float_cols)
g = sns.FacetGrid(float_flat, col='variable', col_wrap=5, sharex=False, sharey=False)
g = g.map(sns.distplot, 'value')
plt.show()


# In[ ]:


log_meaneduc = np.log1p(all_df['meaneduc'])
log_overcrowding = np.log1p(all_df['overcrowding'])
log_SQBovercrowding = np.log1p(all_df['SQBovercrowding'])
log_SQBdependency = np.log1p(all_df['SQBdependency'])
log_SQBmeaned = np.log1p(all_df['SQBmeaned'])

temp_df = pd.DataFrame({'log_meaneduc': log_meaneduc, 'log_overcrowding': log_overcrowding, 'log_SQBovercrowding': log_SQBovercrowding, 'log_SQBdependency': log_SQBdependency, 'log_SQBmeaned': log_SQBmeaned})
temp_df.head()


# In[ ]:


temp_flat = pd.melt(temp_df)
g = sns.FacetGrid(temp_flat, col='variable', col_wrap=5, sharex=False, sharey=False)
g = g.map(sns.distplot, 'value')
plt.show()


# In[ ]:


#temp_df['log_meaneduc'] = pd.cut(temp_df['log_meaneduc'], [0.0, 1.945910, 2.268684, 2.525729, 3.637586], labels=[1, 2, 3, 4], include_lowest=True)
#temp_df['log_overcrowding'] = pd.cut(temp_df['log_overcrowding'], [0.133531, 0.693147, 0.916291, 1.098612, 2.639057], labels=[1, 2, 3, 4], include_lowest=True)
#temp_df['log_SQBovercrowding'] = pd.cut(temp_df['log_SQBovercrowding'], [0.020203, 0.693147, 1.178655, 1.609438, 5.135798], labels=[1, 2, 3, 4], include_lowest=True)
#temp_df['log_SQBdependency'] = pd.cut(temp_df['log_SQBdependency'], [0.0, 0.105361, 0.367725, 1.021651, 4.174387], labels=[1, 2, 3, 4], include_lowest=True)
#temp_df['log_SQBmeaned'] = pd.cut(temp_df['log_SQBmeaned'], [0.0, 3.610918, 4.332194, 4.892227, 7.222566], labels=[1, 2, 3, 4], include_lowest=True)


# In[ ]:


#temp_df.fillna({'log_meaneduc': 2, 'log_overcrowding': 2, 'log_SQBovercrowding': 2, 'log_SQBdependency': 1, 'log_SQBmeaned': 4}, inplace=True)
all_df[['meaneduc', 'overcrowding', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned']] = temp_df[['log_meaneduc', 'log_overcrowding', 'log_SQBovercrowding', 'log_SQBdependency', 'log_SQBmeaned']]


# In[ ]:


# Visualizing integer feature columns
int_flat = pd.melt(all_df, value_vars=int_cols)
g = sns.FacetGrid(int_flat, col='variable', col_wrap=6, sharex=False, sharey=False)
g = g.map(sns.countplot, 'value')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# Removing Id feature from list and visualizing object feature columns
del(object_cols[0])
object_flat = pd.melt(all_df, value_vars=object_cols)
g = sns.FacetGrid(object_flat, col='variable', col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.countplot, 'value')
plt.show()


# In[ ]:


# Encoding feature columns of 'object' dtype
le = LabelEncoder()
for col in object_cols:
    all_df[col] = le.fit_transform(all_df[col].values)


# In[ ]:


dup_cols = [col for col in all_df.columns if col[:3] == 'SQB']
all_df.drop(dup_cols, axis=1, inplace=True)
all_df.shape


# In[ ]:


all_df.drop('agesq', axis=1, inplace=True)
all_df['edjefe'] = (all_df['edjefe'] == all_df['edjefe'].max()) * 1
all_df['edjefa'] = (all_df['edjefa'] == all_df['edjefa'].max()) * 1
all_df.shape


# In[ ]:


all_df['age'] = sc.fit_transform(all_df['age'].values.reshape((-1, 1)))
all_df['idhogar'] = sc.fit_transform(all_df['idhogar'].values.reshape((-1, 1)))


# In[ ]:


cat_cols = [col for col in int_cols if col not in [col for col in int_cols if col[:3]=='SQB']]
cat_cols.remove('agesq')
cat_cols.remove('age')


# In[ ]:


onehot_cols = [col for col in cat_cols if len(all_df[col].unique()) > 2]
onehot_arr = onehot.fit_transform(all_df[onehot_cols].values)
onehot_arr.shape


# In[ ]:


all_df.drop(onehot_cols, axis=1, inplace=True)
all_df.shape


# In[ ]:


# Dividing the whole dataframe into train and test dataframes
train_df = all_df[all_df['Target'].notnull()]
test_df = all_df[all_df['Target'].isnull()]
print (train_df.shape, test_df.shape)


# In[ ]:


# We have to reduce the target value of each class by 1, otherwise XgBoost thinks its training on 5 classes, since highest class is 4. We will undo this change
# after prediction
train_df['Target'] = train_df['Target'].apply(lambda x: x-1)


# In[ ]:


# Splitting train dataframe into train and validation dataset
tr_cols = [col for col in train_df.columns if col not in ['Id', 'Target']]
X_train = train_df[tr_cols].values
X_train = np.concatenate((X_train, onehot_arr[:9557, :]), axis=1)
y_train = train_df['Target'].values
skf = StratifiedKFold(n_splits=5, shuffle=True)


# In[ ]:


# Declaring class weights as the 4 classes are imbalanced, all some common constant parameters across three models
class_weights = compute_class_weight('balanced', np.sort(train_df['Target'].unique()), train_df['Target'].values)
n_rounds = 1000
learning_rate = 0.2
max_depth = 5
l2_reg = 2
n_classes = 4
early_stopping_rounds = 20


# In[ ]:


test_arr = test_df[[col for col in test_df.columns if col not in ['Id', 'Target']]].values
test_arr = np.concatenate((test_arr, onehot_arr[9557:, :]), axis=1)


# In[ ]:


# Defining XgBoost parameters and training the model. I have used EarlyStopping using the validation error
xgb_params = [('eta', learning_rate), ('max_depth', max_depth), ('colsample_bytree', 0.8), ('lambda', l2_reg), ('objective', 'multi:softprob'), ('num_class', n_classes), ('eval_metric', 'mlogloss'), ('silent', 1)]

for tr_idx, val_idx in skf.split(X_train, y_train):
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    dtrain = xgb.DMatrix(X_tr, y_tr, weight=[class_weights[int(y_tr[i])] for i in range(y_tr.shape[0])])
    dval = xgb.DMatrix(X_val, y_val, weight=[class_weights[int(y_val[i])] for i in range(y_val.shape[0])])
    eval_list = [(dtrain, 'train'), (dval, 'validation')]
    bst = xgb.train(xgb_params, dtrain, n_rounds, eval_list, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)


# In[ ]:


# Predicting the class labels on test dataset
dtest = xgb.DMatrix(test_arr, weight=class_weights)
xgb_preds = bst.predict(dtest)
xgb_sample_subm = pd.read_csv('../input/sample_submission.csv')
xgb_sample_subm['Target'] = np.argmax(xgb_preds, axis=1).astype(int)
xgb_sample_subm['Target'] = xgb_sample_subm['Target'].apply(lambda x: x+1)
xgb_sample_subm.to_csv('xgb_preds.csv', index=False)
sns.countplot(xgb_sample_subm['Target'])
plt.show()


# In[ ]:


# Defining LightGBM parameters and training the model, using earlystopping on validation error
lgb_params = {'objective': 'multiclass', 'num_class': n_classes, 'learning_rate': learning_rate, 'num_leaves': 31, 'num_thread': 4, 'max_depth': max_depth, 'feature_fraction': 0.8, 'lambda_l2': l2_reg}

for tr_idx, val_idx in skf.split(X_train, y_train):
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    dtrain = lgb.Dataset(X_tr, y_tr, weight=[class_weights[int(y_tr[i])] for i in range(y_tr.shape[0])])
    dval = lgb.Dataset(X_val, y_val, weight=[class_weights[int(y_val[i])] for i in range(y_val.shape[0])])
    bst = lgb.train(lgb_params, dtrain, n_rounds, valid_sets = [dval, dtrain], valid_names = ['validation', 'train'], early_stopping_rounds = early_stopping_rounds, verbose_eval=False)


# In[ ]:


# Predicting on test set using LightGBM model
lgb_preds = bst.predict(test_arr)
lgb_sample_subm = pd.read_csv('../input/sample_submission.csv')
lgb_sample_subm['Target'] = np.argmax(lgb_preds, axis=1).astype(int)
lgb_sample_subm['Target'] = lgb_sample_subm['Target'].apply(lambda x: x+1)
lgb_sample_subm.to_csv('lgb_preds.csv', index=False)
sns.countplot(lgb_sample_subm['Target'])
plt.show()


# In[ ]:


# Defining CatBoost parameters and training the model, using earlystopping on validation error
cat_model = CatBoostClassifier(iterations=n_rounds, learning_rate=learning_rate, depth=max_depth, loss_function='MultiClass', classes_count=n_classes, logging_level='Silent', l2_leaf_reg=l2_reg, thread_count=4, class_weights=class_weights)
for tr_idx, val_idx in skf.split(X_train, y_train):
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    cat_model.fit(X_tr, y_tr, use_best_model=True, eval_set=(X_val, y_val), early_stopping_rounds=early_stopping_rounds)


# In[ ]:


# Predicting on test set using CatBoost model
cat_preds = cat_model.predict_proba(test_arr)
cat_sample_subm = pd.read_csv('../input/sample_submission.csv')
cat_sample_subm['Target'] = np.argmax(cat_preds, axis=1).astype(int)
cat_sample_subm['Target'] = cat_sample_subm['Target'].apply(lambda x: x+1)
cat_sample_subm.to_csv('cat_preds.csv', index=False)
sns.countplot(cat_sample_subm['Target'])
plt.show()


# In[ ]:


# Averaging all three predictions using arithmetic mean and weighing models according to their individual performances
avg_preds = (xgb_preds + cat_preds + lgb_preds) / 3.0
avg_sample_subm = pd.read_csv('../input/sample_submission.csv')
avg_sample_subm['Target'] = np.argmax(avg_preds, axis=1).astype(int)
avg_sample_subm['Target'] = avg_sample_subm['Target'].apply(lambda x: x+1)
avg_sample_subm.to_csv('avg_preds.csv', index=False)
sns.countplot(avg_sample_subm['Target'])
plt.show()


# In[ ]:


# Geometric weighing three predictions and weighing models according to their individual performances
#geo_preds = (xgb_preds**0.15) * (cat_preds**0.6) * (lgb_preds**0.25)
#geo_sample_subm = pd.read_csv('../input/sample_submission.csv')
#geo_sample_subm['Target'] = np.argmax(geo_preds, axis=1).astype(int)
#geo_sample_subm['Target'] = geo_sample_subm['Target'].apply(lambda x: x+1)
#sns.countplot(geo_sample_subm['Target'])
#plt.show()

