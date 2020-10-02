#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from IPython.display import HTML
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from category_encoders import  LeaveOneOutEncoder, BinaryEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from catboost import CatBoost
from catboost import Pool
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import time
import logging


sample_submission = pd.read_csv("../input/cat-in-the-dat-ii/sample_submission.csv")
test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")
train = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")


# In[ ]:


test.head()


# In[ ]:


train.head()


# In[ ]:


train[['nom_5','nom_6','nom_7','nom_8','nom_9','target']][60:70]


# In[ ]:


#create new features   nom_x by nom_y
train["nom_0by1"]=train.nom_0.str.cat(train['nom_1'])
train["nom_0by2"]=train.nom_0.str.cat(train['nom_2'])
train["nom_0by3"]=train.nom_0.str.cat(train['nom_3'])
train["nom_0by4"]=train.nom_0.str.cat(train['nom_4'])
test["nom_0by1"]=test.nom_0.str.cat(test['nom_1'])
test["nom_0by2"]=test.nom_0.str.cat(test['nom_2'])
test["nom_0by3"]=test.nom_0.str.cat(test['nom_3'])
test["nom_0by4"]=test.nom_0.str.cat(test['nom_4'])

train["nom_1by2"]=train.nom_1.str.cat(train['nom_2'])
train["nom_1by3"]=train.nom_1.str.cat(train['nom_3'])
train["nom_1by4"]=train.nom_1.str.cat(train['nom_4'])
test["nom_1by2"]=test.nom_1.str.cat(test['nom_2'])
test["nom_1by3"]=test.nom_1.str.cat(test['nom_3'])
test["nom_1by4"]=test.nom_1.str.cat(test['nom_4'])


train["nom_2by3"]=train.nom_2.str.cat(train['nom_3'])
train["nom_2by4"]=train.nom_2.str.cat(train['nom_4'])
test["nom_2by3"]=test.nom_2.str.cat(test['nom_3'])
test["nom_2by4"]=test.nom_2.str.cat(test['nom_4'])

train["nom_3by4"]=train.nom_3.str.cat(train['nom_4'])
test["nom_3by4"]=test.nom_3.str.cat(test['nom_4'])

#create new features   bin_x by nom_y
train["bin_3by4"]=train.bin_3.str.cat(train['bin_4'])
test["bin_3by4"]=test.bin_3.str.cat(test['bin_4'])


train["bin_3bynom_0"]=train.bin_3.str.cat(train['nom_0'])
train["bin_3bynom_1"]=train.bin_3.str.cat(train['nom_1'])
train["bin_3bynom_2"]=train.bin_3.str.cat(train['nom_2'])
train["bin_3bynom_3"]=train.bin_3.str.cat(train['nom_3'])
train["bin_3bynom_4"]=train.bin_3.str.cat(train['nom_4'])
test["bin_3bynom_0"]=test.bin_3.str.cat(test['nom_0'])
test["bin_3bynom_1"]=test.bin_3.str.cat(test['nom_1'])
test["bin_3bynom_2"]=test.bin_3.str.cat(test['nom_2'])
test["bin_3bynom_3"]=test.bin_3.str.cat(test['nom_3'])
test["bin_3bynom_4"]=test.bin_3.str.cat(test['nom_4'])


train["bin_4bynom_0"]=train.bin_4.str.cat(train['nom_0'])
train["bin_4bynom_1"]=train.bin_4.str.cat(train['nom_1'])
train["bin_4bynom_2"]=train.bin_4.str.cat(train['nom_2'])
train["bin_4bynom_3"]=train.bin_4.str.cat(train['nom_3'])
train["bin_4bynom_4"]=train.bin_4.str.cat(train['nom_4'])
test["bin_4bynom_0"]=test.bin_4.str.cat(test['nom_0'])
test["bin_4bynom_1"]=test.bin_4.str.cat(test['nom_1'])
test["bin_4bynom_2"]=test.bin_4.str.cat(test['nom_2'])
test["bin_4bynom_3"]=test.bin_4.str.cat(test['nom_3'])
test["bin_4bynom_4"]=test.bin_4.str.cat(test['nom_4'])

train["nom_5bynom_8"]=train.nom_5.str.cat(train['nom_8'])
test["nom_5bynom_8"]=test.nom_5.str.cat(test['nom_8'])

train["nom_6bynom_9"]=train.nom_6.str.cat(train['nom_9'])
test["nom_6bynom_9"]=test.nom_6.str.cat(test['nom_9'])


# In[ ]:


train.nom_0by1


# In[ ]:


#create new features   day as string
train['day_obj']=train.day.astype('str')
test['day_obj']=test.day.astype('str')

train['month_obj']=train.month.astype('str')
test['month_obj']=test.month.astype('str')

#create new features   ord_x as object
train['ord_0_obj']=train.ord_0
train['ord_1_obj']=train.ord_1
train['ord_2_obj']=train.ord_2
train['ord_3_obj']=train.ord_3
train['ord_4_obj']=train.ord_4
train['ord_5_obj']=train.ord_5
test['ord_0_obj']=test.ord_0
test['ord_1_obj']=test.ord_1
test['ord_2_obj']=test.ord_2
test['ord_3_obj']=test.ord_3
test['ord_4_obj']=test.ord_4
test['ord_5_obj']=test.ord_5

train['nom_0_obj']=train.nom_0
train['nom_1_obj']=train.nom_1
train['nom_2_obj']=train.nom_2
train['nom_3_obj']=train.nom_3
train['nom_4_obj']=train.nom_4
test['nom_0_obj']=test.nom_0
test['nom_1_obj']=test.nom_1
test['nom_2_obj']=test.nom_2
test['nom_3_obj']=test.nom_3
test['nom_4_obj']=test.nom_4

train['bin_0_obj']=train.bin_0
train['bin_1_obj']=train.bin_1
train['bin_2_obj']=train.bin_2
train['bin_3_obj']=train.bin_3
train['bin_4_obj']=train.bin_4
test['bin_0_obj']=test.bin_0
test['bin_1_obj']=test.bin_1
test['bin_2_obj']=test.bin_2
test['bin_3_obj']=test.bin_3
test['bin_4_obj']=test.bin_4


# In[ ]:


#create new features   month by day
train["daybymonth"]=train.month_obj.str.cat(train['day_obj'])
test["daybymonth"]=test.month_obj.str.cat(test['day_obj'])


# In[ ]:


train.ord_1_obj


# In[ ]:


train.columns


# In[ ]:


#Categorical features

#cat_cols=['nom_5','nom_6','nom_7','nom_8','nom_9','day_obj','month_obj','ord_0_obj','ord_1_obj','ord_2_obj','ord_3_obj','ord_4_obj',
#          'ord_5_obj','nom_0_obj','nom_1_obj','nom_2_obj','nom_3_obj','nom_4_obj','bin_0_obj','bin_1_obj','bin_2_obj','bin_3_obj','bin_4_obj']

#Try new features and select only good effectors

cat_cols=['nom_5','nom_6','nom_7','nom_8','nom_9','day_obj','month_obj','ord_0_obj','ord_1_obj','ord_2_obj','ord_3_obj','ord_4_obj',
          'ord_5_obj',
          'nom_0_obj','nom_1_obj','nom_2_obj','nom_3_obj','nom_4_obj',
          'bin_0_obj','bin_1_obj','bin_2_obj','bin_3_obj','bin_4_obj',
          "nom_3by4","bin_4bynom_0"]

#Remain bad effector
#"nom_0by1""nom_0by2""nom_0by3""nom_0by4""nom_1by2""nom_1by3""nom_1by4""nom_2by3""nom_2by4","bin_3by4","bin_3bynom_0","bin_3bynom_1"
#,"bin_3bynom_3","bin_3bynom_2","bin_3bynom_4","bin_4bynom_1","bin_4bynom_2""bin_4bynom_3","bin_4bynom_4"


#Target encorder
for c in cat_cols:
    data_tmp=pd.DataFrame({c:train[c],'target':train.target})
    target_mean=data_tmp.groupby(c)['target'].mean()
    test[c]=test[c].map(target_mean)
    
    tmp=np.repeat(np.nan, train.shape[0])
    
    kf=KFold(n_splits=4, shuffle=True, random_state=72)
    for idx_1,idx_2 in kf.split(train):
        target_mean=data_tmp.iloc[idx_1].groupby(c)['target'].mean()
        tmp[idx_2]=train[c].iloc[idx_2].map(target_mean)
        
        
    train[c]=tmp


# In[ ]:


#Create new features
train['nom_mean']=(train.nom_5+train.nom_6+train.nom_7)/3
test['nom_mean']=(test.nom_5+test.nom_6+test.nom_7)/3


# In[ ]:


#Numerical features

cols_to_use = ['ord_1', 'ord_2', 'ord_3', 'ord_4','ord_5','nom_0','nom_1','nom_2','nom_3','nom_4','bin_3','bin_4']  #cat_columns
#cols_to_use = ['ord_1', 'ord_2', 'ord_3', 'ord_4','ord_5','nom_2']
#cols_to_use = ['nom_0','nom_1','nom_2','nom_3','nom_4','ord_5']

#Try new features and select only good effectors

#num_cols_to_use = ['bin_0','bin_1','bin_2','ord_0']
#num_cols_to_use = ['ord_0','bin_0','bin_1','bin_2','day','month','nom_5','nom_6','nom_7','nom_8','nom_9','day_obj','month_obj',
#                   'ord_0_obj','ord_1_obj','ord_2_obj','ord_3_obj','ord_4_obj','ord_5_obj',
#                   'nom_0_obj','nom_1_obj','nom_2_obj','nom_3_obj','nom_4_obj',
#                   'bin_0_obj','bin_1_obj','bin_2_obj','bin_3_obj','bin_4_obj']
num_cols_to_use = ['ord_0','bin_0','bin_1','bin_2','day','month','nom_5','nom_6','nom_7','nom_8','nom_9','day_obj','month_obj',
                   'ord_0_obj','ord_1_obj','ord_2_obj','ord_3_obj','ord_4_obj','ord_5_obj',
                   'nom_0_obj','nom_1_obj','nom_2_obj','nom_3_obj','nom_4_obj',
                   'bin_0_obj','bin_1_obj','bin_2_obj','bin_3_obj','bin_4_obj',
                   "nom_3by4","bin_4bynom_0"
                   ]


#num_cols_to_use = ['day_obj','month_obj','ord_0_obj','ord_1_obj','ord_2_obj','ord_3_obj','ord_4_obj','ord_5_obj',
#                   'nom_0_obj','nom_1_obj','nom_2_obj','nom_3_obj','nom_4_obj','bin_0_obj','bin_1_obj','bin_2_obj',
#                   'bin_3_obj','bin_4_obj']

#bad effect
#"nom_0by1""nom_0by2""nom_0by3""nom_0by4""nom_1by2""nom_1by3""nom_1by4""nom_2by3""nom_2by4","bin_3by4","bin_3bynom_0","bin_3bynom_1"
#,"bin_3bynom_3","bin_3bynom_2","bin_3bynom_4","bin_4bynom_1","bin_4bynom_2","bin_4bynom_3","bin_4bynom_4"
cols_sum = cols_to_use + num_cols_to_use

X = train[cols_sum]
y = train.target

#One Hot Encoder

from sklearn.model_selection import train_test_split
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state = 0)

X_val = test[cols_sum]

numerical_transformer = SimpleImputer(strategy='mean')

categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='constant')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols_to_use),
        ('cat',categorical_transformer, cols_to_use)
    ])

#Try some models and select only good result

#params={'objective':'binary:logistic','eta':0.05,'gamma':0.0,'lambda':1.0,'min_child_weight':1,'max_depth':10,'colsample_bytree':0.8,'random_state':71}
#my_model = XGBRegressor(**params,num_round=5000, early_stopping_rounds=10,n_estimators=500)
#my_model = XGBRegressor(n_estimators=800, learning_rate=0.2,objective ='binary:logistic')
#my_model = XGBClassifier(n_estimators=50, learning_rate=0.01, max_depth=5)
my_model = CatBoost({'loss_function':'Logloss','num_boost_round':2500,'learning_rate':0.03,'early_stopping_rounds': 10,'depth':2})
#my_model = CatBoostClassifier(custom_loss=['Accuracy'],random_seed=42,iterations=1800,learning_rate=0.7,early_stopping_rounds=10,depth=10)
#my_model = LGBMClassifier(n_estimators=1500, learning_rate=0.5, max_depth=10,num_leaves=50,max_bin=500)
#my_model = RandomForestClassifier(n_estimators=50, random_state=0)

my_pipeline = Pipeline(steps=[('preprocessor',preprocessor),
                      ('model',my_model)
                     ])

my_pipeline.fit(train_X,train_y)

prediction = my_pipeline.predict(X_val)


# In[ ]:


from sklearn.metrics import roc_auc_score
val_predictions = my_pipeline.predict(val_X)
print(roc_auc_score(val_y,val_predictions))


# In[ ]:


train.ord_0_obj


# In[ ]:


ss=sample_submission
ss.target = prediction
ss.to_csv("submission.csv", index=False)


# In[ ]:


prediction

