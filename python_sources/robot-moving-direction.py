#!/usr/bin/env python
# coding: utf-8

# In[133]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import gc

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[134]:


train=pd.read_csv("../input/X_train.csv")
y=pd.read_csv("../input/y_train.csv")
test=pd.read_csv("../input/X_test.csv")


# In[135]:


train.head()


# In[136]:


y.head()


# Data exploation

# In[137]:


train.shape,test.shape


# numericla features in train data

# In[138]:


train.describe()


# now our target variable  analysis 

# In[139]:


y.head(),y.shape,y.columns


# In[140]:


target=y['surface'].value_counts().reset_index().rename(columns={"index":"target"})


# In[141]:


target.head()


# In[142]:


target.plot.bar()


# In[143]:


y['surface'].value_counts().plot.bar()


# 
# ***DATA PreProcessing****

# In[144]:


train.isnull().sum()


# No missing values on the train data

# In[145]:


train['is_duplicate'] = train.duplicated()
train['is_duplicate'].value_counts()


# No duplicates on the train data

# In[146]:


train = train.drop(['is_duplicate'], axis = 1)


# In[147]:


train_sort_value=train.sort_values(by=['series_id','measurement_number'],ascending=True)


# In[148]:


train_sort_value.head()


# Min and max values of each feature of the train data 

# In[149]:


train.max(),train.min()


# 1. Correlations of features Train data

# In[150]:


f,ax=plt.subplots(1,1 ,figsize=(8,8))
sns.heatmap(train.iloc[:,3:].corr(),annot=True,linewidths=.5,cmap="YlGnBu",ax=ax)


#  Correlations of features Test data

# In[151]:


f,ax=plt.subplots(1,1 ,figsize=(8,8))
sns.heatmap(test.iloc[:,3:].corr(),annot=True,linewidths=.5,cmap="YlGnBu",ax=ax)


# 
# well , some fetures strongly  Correlated to each other
# * Angular_velocity_Y is related to Angular_velocity_Z
# * Linear_Acceleration_Y is related to the Linear_Acceleration_Z

# In[152]:


def feature_distribution_plot(train, test, label1, label2, features,a=2,b=5):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(a,b,figsize=(17,9))

    for feature in features:
        i += 1
        plt.subplot(a,b,i)
        sns.kdeplot(train[feature], bw=0.5,label=label1)
        sns.kdeplot(test[feature], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show(); 


# In[153]:


features = train.columns.values[3:]
feature_distribution_plot(train, test, 'train', 'test', features)


# * Godd news, our basic features have the same distribution (Normal) on test and training. There are some differences between orientation_X , orientation_Y and linear_acceleration_Y.
#  * I willl try StandardScaler to fix this, and remember: orientation , angular velocity and linear acceleration are measured with different units, scaling might be a good choice.

# In[154]:


plt.figure(figsize=(26, 16))
for i, col in enumerate(train.columns[3:]):
    ax = plt.subplot(3, 4, i + 1)
    sns.distplot(train[col], bins=100, label='train')
    sns.distplot(test[col], bins=100, label='test')
    ax.legend()   


# linear_accelaration are normally distributed/symmetrical distribution but average value is slightly negative for linear_accelaration_Z
# X,Y,Z,W orientation data are not symmetrical or bell shaped distributed.
# X,Y orientation data are distributed un-even between 1 to -1.
# Z,W orientation data are distributed un-even between 1.5 to -1.5
# Since orientation data is not linearly distributed, taking log of the orientation data may improve the results.
# 
# 

# In[155]:


data=train.merge(y,on="series_id",how='inner')
targets=(y['surface'].value_counts()).index


# In[156]:


data.head()


# In[157]:


plt.figure(figsize=(26, 16))
for i,col in enumerate(data.columns[3:13]):
    ax = plt.subplot(3,4,i+1)
    ax = plt.title(col)
    for surface in targets:
        surface_feature = data[data['surface'] == surface]
        sns.kdeplot(surface_feature[col], label = surface)


# **Feature Engineering **
# 
# 
# 
# Euler angles
# The Euler angles are three angles introduced by Leonhard Euler to describe the orientation of a rigid body with respect to a fixed coordinate system.

# In[158]:


def feature_eng1(df):
    df['norm_q']=df['orientation_X']**2 + df['orientation_Y']**2 + df['orientation_Z']**2 +df['orientation_W'] **2
    df['mor_q']=df['norm_q']**0.5
    df['norm_X']=df['orientation_X']/df['norm_q']
    df['norm_Y']=df['orientation_Y']/df['norm_q']
    df['norm_Z']=df['orientation_Z']/df['norm_q']
    df['norm_W']=df['orientation_W']/df['norm_q']
    
    return df


# In[159]:


df=feature_eng1(train)
test=feature_eng1(test)
print(df.shape,test.shape)
df.head()


# In[160]:



def feat_eng2(df):
    df['totl_anglr_vel'] = (df['angular_velocity_X']**2 + df['angular_velocity_Y']**2 + df['angular_velocity_Z']**2)** 0.5
    df['totl_linr_acc'] = (df['linear_acceleration_X']**2 + df['linear_acceleration_Y']**2 + df['linear_acceleration_Z']**2)**0.5
    df['totl_orientation'] = (df['orientation_X']**2 + df['orientation_Y']**2 + df['orientation_Z']**2)**0.5
    df['acc_vs_vel'] = df['totl_linr_acc'] / df['totl_anglr_vel']
    return df


# In[161]:


data = feat_eng2(train)
test = feat_eng2(test)
print(data.shape, test.shape)
data.head()


# In[162]:


def feat_eng3(df):
    data = pd.DataFrame()
    for col in df.columns:
        if col in ['row_id','series_id','measurement_number']:
            continue
        data[col + '_mean'] = df.groupby(['series_id'])[col].mean()
        data[col + '_median'] = df.groupby(['series_id'])[col].median()
        data[col + '_max'] = df.groupby(['series_id'])[col].max()
        data[col + '_min'] = df.groupby(['series_id'])[col].min()
        data[col + '_std'] = df.groupby(['series_id'])[col].std()
        data[col + '_range'] = data[col + '_max'] - data[col + '_min']
        data[col + '_maxtoMin'] = data[col + '_max'] / data[col + '_min']
        #in statistics, the median absolute deviation (MAD) is a robust measure of the variablility of a univariate sample of quantitative data.
        data[col + '_mad'] = df.groupby(['series_id'])[col].apply(lambda x: np.median(np.abs(np.diff(x))))
        data[col + '_abs_max'] = df.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))
        data[col + '_abs_min'] = df.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))
        data[col + '_abs_avg'] = (data[col + '_abs_min'] + data[col + '_abs_max'])/2
    return data


# In[163]:


get_ipython().run_cell_magic('time', '', 'data = feat_eng3(train)\ntest = feat_eng3(test)\nprint(data.shape, test.shape)\ndata.head()')


# In[164]:


data.fillna(0, inplace = True)
data.replace(-np.inf, 0, inplace = True)
data.replace(np.inf, 0, inplace = True)
test.fillna(0, inplace = True)
test.replace(-np.inf, 0, inplace = True)
test.replace(np.inf, 0, inplace = True)


# In[165]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data_y = le.fit_transform(y['surface'])


# In[166]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_y= le.fit_transform(y['surface'])


# **Model Builiding **

# In[167]:


folds = StratifiedKFold(n_splits=12, shuffle=True, random_state=60)
predicted = np.zeros((test.shape[0],9))
measured= np.zeros((data.shape[0]))
score = 0


# In[168]:


for times, (trn_idx, val_idx) in enumerate(folds.split(data.values,data_y)):
    model = RandomForestClassifier(n_estimators=500, n_jobs = -1)
    #model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=5, n_jobs=-1)
    model.fit(data.iloc[trn_idx],data_y[trn_idx])
    measured[val_idx] = model.predict(data.iloc[val_idx])
    predicted += model.predict_proba(test)/folds.n_splits
    score += model.score(data.iloc[val_idx],data_y[val_idx])
    print("Fold: {} score: {}".format(times,model.score(data.iloc[val_idx],data_y[val_idx])))
    
    gc.collect()


# Understanding about important features will help us fine tuning feature enginnering as well accuracy improvement.
# 
# 

# In[169]:


importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis = 0)
indices = np.argsort(importances)[::-1]


# In[170]:


feature_imp = pd.DataFrame(importances, index = data.columns, columns = ['importance'])
feature_imp.sort_values('importance', ascending = False)
feature_imp.head()


# In[171]:


less_important_features = feature_imp.loc[feature_imp['importance'] < 0.0025]
print('There are {0} features their importance value is less then 0.0025'.format(less_important_features.shape[0]))


# In[172]:


submission = pd.read_csv('../input/sample_submission.csv')
submission['surface'] = le.inverse_transform(predicted.argmax(axis=1))
submission.to_csv('rs_surface_submission6.csv', index=False)
submission.head()

