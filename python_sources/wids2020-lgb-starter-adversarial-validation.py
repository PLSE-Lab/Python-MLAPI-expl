#!/usr/bin/env python
# coding: utf-8

# * starter LGB with no feature engering /
# 
# * added some viz about null values
# 
# * added adversarial validation 

# In[ ]:


from IPython.core.display import display, HTML
display(HTML('<style>.container {width:98% !important;}</style>'))


# In[ ]:


import numpy as np
import pandas as pd

import random
random.seed(28)
np.random.seed(28)


import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
import os
import copy
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15
from collections import defaultdict
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
import time
from collections import Counter
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold
from sklearn import metrics
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from bayes_opt import BayesianOptimization
import eli5
import shap
from IPython.display import HTML
import json

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import time
import datetime
import gc
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
pd.set_option('max_rows', 500)
import re

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd.set_option('display.max_columns', 1000)
np.random.seed(566)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:20,.2f}'.format)
pd.set_option('display.max_colwidth', -1)


# # APACHE III Scoring Card :
# Here is the way apache score (hence the apache probas when rescale to 0-1) is computed :
# 
# <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F273819%2F407c8f1acd66feba1e4ddafb1c8f3a12%2FAPACHEIII_scorecard.png?generation=1579776903551461&alt=media" alt="APACHE III Scoring Card" />

# # Read the data

# In[ ]:


train = pd.read_csv("/kaggle/input/widsdatathon2020/training_v2.csv")
samplesubmission = pd.read_csv("/kaggle/input/widsdatathon2020/samplesubmission.csv")
test = pd.read_csv("/kaggle/input/widsdatathon2020/unlabeled.csv")
dictionary = pd.read_csv("/kaggle/input/widsdatathon2020/WiDS Datathon 2020 Dictionary.csv")
solution_template = pd.read_csv("/kaggle/input/widsdatathon2020/solution_template.csv")

print('train ' , train.shape)
print('test ' , test.shape)
print('samplesubmission ' , samplesubmission.shape)
print('solution_template ' , solution_template.shape)
print('dictionary ' , dictionary.shape)


# # OverView of the dataset

# In[ ]:


dico=pd.DataFrame(dictionary.T.head(6))
dico.columns=list(dico.loc[dico.index == 'Variable Name'].unstack())
dico = dico.loc[dico.index != 'Variable Name']
dico.columns
train_stat = pd.DataFrame(train.describe())
train_stat2 = pd.concat([dico,train_stat],axis=0)
train_stat2.head(20)


# In[ ]:


train_stat2.T.head(200)


# # Have a look at missing value

# In[ ]:


import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Matrix 
# 
# Using this matrix you can very quickly find the pattern of missingness in the dataset.
# 
# --> lots of data are missing, for some of them you cannot do anything so just impute with a fixed value , for those that are missing with no specific reason you can try fixex imputation , mean/median, or to predict it with other value (just try with CV to see what is the best choice for each feature / group of feature ) .
# 

# In[ ]:


msno.matrix(train.sample(1000),figsize=(35, 60), width_ratios=(10, 1), color=(.0, 0.5, 0.5),           fontsize=16)


# ## Due to the amout of features we will split the dataset by category 

# In[ ]:


for color, variable in enumerate(dictionary['Category'].unique()) :
  if variable not in ['GOSSIS example prediction','identifier']  :
    print(variable)
    column_list = list(dictionary[dictionary['Category']==variable]['Variable Name'].values)
    column_list = [f for f in column_list if f in train.columns]
    if len(column_list) > 0:
        msno.matrix(train[column_list].sample(1000),figsize=(30, 10), labels=True, color=(color/10, 1/(color+1), 0.5),  fontsize=16)
        msno.heatmap(train[column_list],figsize=(10, 10)     ,  labels=False,    fontsize=14)
        plt.show()


# ## Heatmap showing the correlation of missingness between every 2 columns
# 
# A value near -1 means if one variable appears then the other variable is very likely to be missing.
# 
# A value near 0 means there is no dependence between the occurrence of missing values of two variables.
# 
# A value near 1 means if one variable appears then the other variable is very likely to be present.
# 

# In[ ]:


msno.heatmap(train,figsize=(35, 40)     ,  labels=False,    fontsize=10)


# # dendrogram visualization 
# It is based on hierachical clustering of missing values, so it shows a tree representing groupings of columns that have strong nullity correlations. so it identifies groups that are correlated, rather than simple pairs (as in the heatmap)

# In[ ]:


msno.dendrogram(train,fontsize=14)


# # Bar chart
# 
# This bar chart gives you an idea about how many missing values are there in each column. 

# In[ ]:


msno.bar(train.sample(10000))


# In[ ]:


# function to evaluate the score of our model
def eval_auc(pred,real):
    false_positive_rate, recall, thresholds = roc_curve(real, pred)
    roc_auc = auc(false_positive_rate, recall)
    return roc_auc    


# In[ ]:


# a wrapper class  that we can have the same ouput whatever the model we choose
class Base_Model(object):
    
    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=10, verbose=True,ps={}):
        self.train_df = train_df
        self.test_df = test_df
        self.features = features
        self.n_splits = n_splits
        self.categoricals = categoricals
        self.target = 'hospital_death'
        self.cv = self.get_cv()
        self.verbose = verbose
#         self.params = self.get_params()
        self.params = self.set_params(ps)
        self.y_pred, self.score, self.model , self.oof_pred = self.fit()
        
    def train_model(self, train_set, val_set):
        raise NotImplementedError
        
    def get_cv(self):
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        return cv.split(self.train_df, self.train_df[self.target])
    
    def get_params(self):
        raise NotImplementedError
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError
        
    def convert_x(self, x):
        return x
        
    def fit(self):
        oof_pred = np.zeros((len(self.train_df), ))
        y_pred = np.zeros((len(self.test_df), ))
        for fold, (train_idx, val_idx) in enumerate(self.cv):
            x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]
            y_train, y_val = self.train_df[self.target][train_idx], self.train_df[self.target][val_idx]
            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
            model = self.train_model(train_set, val_set)
            conv_x_val = self.convert_x(x_val)
            oof_pred[val_idx] = model.predict(conv_x_val).reshape(oof_pred[val_idx].shape)
            x_test = self.convert_x(self.test_df[self.features])
            y_pred += model.predict(x_test).reshape(y_pred.shape) / self.n_splits

            print('Partial score of fold {} is: {}'.format(fold,eval_auc(oof_pred[val_idx],y_val) ))
        #print(oof_pred, self.train_df[self.target].values)
        loss_score = eval_auc(oof_pred,self.train_df[self.target].values) 
        if self.verbose:
            print('Our oof AUC score is: ', loss_score)
        return y_pred, loss_score, model , oof_pred


# In[ ]:


#we choose to try a LightGbM using the Base_Model class
class Lgb_Model(Base_Model):
    
    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        return lgb.train(self.params, train_set, valid_sets=[train_set, val_set], verbose_eval=verbosity)
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)
        val_set   = lgb.Dataset(x_val,    y_val,  categorical_feature=self.categoricals)
        return train_set, val_set
        
    def get_params(self):
        params = {'n_estimators':50000,
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'metric': 'auc',
                    'subsample': 0.75,
                    'subsample_freq': 1,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.9,
                    'max_depth': 15,
                    'lambda_l1': 1,  
                    'lambda_l2': 1,
                    'early_stopping_rounds': 100,
                    #'is_unbalance' : True ,
                    'scale_pos_weight' : 3
                  
                    }
        return params
    def set_params(self,ps={}):
        params = self.get_params()
        if 'subsample_freq' in ps:
            params['subsample_freq']=int(ps['subsample_freq'])
            params['learning_rate']=ps['learning_rate']
            params['feature_fraction']=ps['feature_fraction']
            params['lambda_l1']=ps['lambda_l1']
            params['lambda_l2']=ps['lambda_l2']
            params['scale_pos_weight']=ps['scale_pos_weight']
            params['max_depth']=int(ps['max_depth'])
        
        return params  


# In[ ]:


train['apache_3j_diagnosis_split0'] = np.where(train['apache_3j_diagnosis'].isna() , np.nan , train['apache_3j_diagnosis'].astype('str').str.split('.',n=1,expand=True)[0]  )
test['apache_3j_diagnosis_split0']   = np.where(test['apache_3j_diagnosis'].isna() , np.nan , test['apache_3j_diagnosis'].astype('str').str.split('.',n=1,expand=True)[0]  )


# In[ ]:


#we are going to drop these columns because we dont want our ML model to be bias toward these consideration
#(we also remove the target and the ids.)
to_drop = ['gender','ethnicity' ,'encounter_id', 'patient_id',  'hospital_death']

# this is a list of features that look like to be categorical
categoricals_features = ['hospital_id','ethnicity','gender','hospital_admit_source','icu_admit_source','icu_stay_type','icu_type','apache_3j_bodysystem','apache_2_bodysystem','apache_3j_diagnosis_split0']
categoricals_features = [col for col in categoricals_features if col not in to_drop]

# this is the list of all input feature we would like our model to use 
features = [col for col in train.columns if col not in to_drop ]
print('numerber of features ' , len(features))
print('shape of train / test ', train.shape , test.shape)


# ### categorical feature need to be transform to numeric for mathematical purpose.
# 
# different technics of categorical encoding exists here we will rely on our model to deal with categorical data,
# still we need to encode each categorical value to an id , for this purpose we use LabelEncoder
# 

# In[ ]:


# categorical feature need to be transform to numeric for mathematical purpose.
# different technics of categorical encoding exists here we will rely on our model API to deal with categorical
# still we need to encode each categorical value to an id , for this purpose we use LabelEncoder

print('Transform all String features to category.\n')
for usecol in categoricals_features:
    train[usecol] = train[usecol].astype('str')
    test[usecol] = test[usecol].astype('str')
    
    #Fit LabelEncoder
    le = LabelEncoder().fit(
            np.unique(train[usecol].unique().tolist()+
                      test[usecol].unique().tolist()))

    #At the end 0 will be used for null values so we start at 1 
    train[usecol] = le.transform(train[usecol])+1
    test[usecol]  = le.transform(test[usecol])+1
    
    train[usecol] = train[usecol].replace(np.nan, 0).astype('int').astype('category')
    test[usecol]  = test[usecol].replace(np.nan, 0).astype('int').astype('category')


# # Modele

# In[ ]:


# percentage of death , hopefully it s a bit unbalanced
train['hospital_death'].sum()/train['hospital_death'].count()


# # Hyper parameter tuning

# In[ ]:


def LGB_Beyes(subsample_freq,
                    learning_rate,
                    feature_fraction,
                    max_depth,
                    lambda_l1,
                    lambda_l2,
                    scale_pos_weight):
    params={}
    params['subsample_freq']=subsample_freq
    params['learning_rate']=learning_rate
    params['feature_fraction']=feature_fraction
    params['lambda_l1']=lambda_l1
    params['lambda_l2']=lambda_l2
    params['max_depth']=max_depth
    params['scale_pos_weight']=scale_pos_weight
    
    lgb_model= Lgb_Model(train, test, features, categoricals=categoricals_features,ps=params)
    print('auc: ',lgb_model.score)
    return lgb_model.score

bounds_LGB = {
    'subsample_freq': (1, 10),
    'learning_rate': (0.005, 0.02),
    'feature_fraction': (0.5, 1),
    'lambda_l1': (0, 5),
    'lambda_l2': (0, 5),
    'max_depth': (13, 17),
    'scale_pos_weight': (1, 10),
}

# ACTIVATE it if you want to search for better parameter
if 0 : 
    LGB_BO = BayesianOptimization(LGB_Beyes, bounds_LGB, random_state=1029)
    import warnings
    init_points = 16
    n_iter = 16
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')    
        LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)


# In[ ]:


params = {'feature_fraction': 0.8,
 'lambda_l1': 1,
 'lambda_l2': 1,
 'learning_rate': 0.001,
 'max_depth': 13,
 'subsample_freq': 1,
 'scale_pos_weight':1}


# In[ ]:


if 0: # ACTIVATE it if you want to search for better parameter
    lgb_model = Lgb_Model(train,test, features, categoricals=categoricals_features, ps= LGB_BO.max['params']  )
else :
    lgb_model = Lgb_Model(train,test, features, categoricals=categoricals_features, ps=params)


# Feature Importance from the lightgbm model (gain)

# In[ ]:


imp_df = pd.DataFrame()
imp_df['feature'] = features
imp_df['gain']  = lgb_model.model.feature_importance(importance_type='gain')
imp_df['split'] = lgb_model.model.feature_importance(importance_type='split')


# In[ ]:


def plot_importances(importances_):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    plt.figure(figsize=(18, 44))
    data_imp = importances_.sort_values('mean_gain', ascending=False)
    sns.barplot(x='gain', y='feature', data=data_imp[:300])
    plt.tight_layout()
    plt.savefig('importances.png')
    plt.show()


# In[ ]:


plot_importances(imp_df)


# # Feature Importance by permutation importance algo

# In[ ]:


import shap
explainer   =  shap.TreeExplainer(lgb_model.model)
shap_values = explainer.shap_values(train[features].iloc[:1000,:])
shap.summary_plot(shap_values, train[features].iloc[:1000,:])


# # Some univariate plot of the best feature

# In[ ]:





# In[ ]:


import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=UserWarning)
i=0
for index, row in imp_df.sort_values(by=['gain'],ascending=False).iterrows():  
    column=row['feature']
    if i< 50:
            print(column,i,"gain :",row['gain'])
            df1      = train.loc[train['hospital_death']==0]
            df2      = train.loc[train['hospital_death']==1]

            fig = plt.figure(figsize=(20,4))
            sns.distplot(df1[column].dropna(),  color='green', label='hospital_death 0', kde=True); 
            sns.distplot(df2[column].dropna(),  color='red', label='hospital_death 1', kde=True); 
            fig=plt.legend(loc='best')
            plt.xlabel(column, fontsize=12);
            plt.show()
            i=i+1


# # Adversarial Validation
# 
# The main idea of adversarial validation is to detect shift/drift in the different features between 2 datasets. 
# 
# We usually train a model on past data to forecast future data so it can happened that these futures datas have a distribution that is no longer in line with the data we used for training, or maybe we train on some hospital datas and apply our model on other hospital ?
# 
# You can detect drift by statistical test (like t-test) but here we will do it by training a machine learning model and check if the model can figure out if the data is from the train or test set. If it can, this means that the test data comes from another distribution compare to the train data and then you have to check the distribution of the most important features that are likely to be different between train and test.

# In[ ]:


def adversarial_validation(train, test, features):
    
    tr_data   = train.copy()
    tst_data = test.copy()
    tr_data['target']  = 0 
    tst_data['target'] = 1
    av_data = pd.concat([tr_data, tst_data], axis = 0)
    av_data.reset_index(drop = True)        
    params = {
            'learning_rate': 0.1, 
            'seed': 50,
            'objective':'binary',
            'boosting_type':'gbdt',
            'metric': 'auc',
        }    
    # define a KFold strategy
    kf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=42)
    target = 'target'
    oof_pred = np.zeros(len(av_data))
    important_features = pd.DataFrame()
    fold_auc = []    
    
    for fold, (tr_ind, val_ind) in enumerate(kf.split(av_data, av_data[target])) :
        print('Fold {}'.format(fold + 1))
        x_train, x_val = av_data[features].iloc[tr_ind], av_data[features].iloc[val_ind]
        y_train, y_val = av_data[target].iloc[tr_ind], av_data[target].iloc[val_ind]
        train_set = lgb.Dataset(x_train, y_train)
        val_set   = lgb.Dataset(x_val, y_val)
        
        model = lgb.train(params, train_set, num_boost_round = 1000, early_stopping_rounds = 20, valid_sets = [train_set, val_set], verbose_eval = 100)
        
        fold_importance = pd.DataFrame()
        fold_importance['feature'] = features
        fold_importance['gain'] = model.feature_importance()
        important_features = pd.concat([important_features, fold_importance], axis = 0)
        
        oof_pred[val_ind] = model.predict(x_val)
        fold_auc.append(metrics.roc_auc_score(y_train, model.predict(x_train)))
        
    print('Our mean train roc auc score is :', np.mean(fold_auc))
    print('Our oof roc auc score is :', metrics.roc_auc_score(av_data[target], oof_pred))
    return important_features


# run the adversatial model with all the feature we used :

# In[ ]:


adversarial_features = adversarial_validation(train, test, features)


# AUC is almost perfect so we can expect that **some feature are perfectly different between train / test**

# In[ ]:


adversarial_features = adversarial_features[['gain', 'feature']].groupby('feature').mean().reset_index()
adversarial_features= adversarial_features.sort_values('gain', ascending=False)
plot_importances(adversarial_features)


# So icu_id columns seems to be the feature that dominate the feature importance for the adversarial validation model, so it is likely to be totally different between train and test, lets check the distribution of the top features :

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=UserWarning)
i=0
for index, row in adversarial_features.sort_values(by=['gain'],ascending=False).iterrows():  
    column=row['feature']
    if i< 3:
            print(column,i,"gain :",row['gain'])
            df1      = train.copy()
            df2      = test.copy()

            fig = plt.figure(figsize=(20,4))
            sns.distplot(df1[column].dropna(),  color='yellow', label='train', kde=True); 
            sns.distplot(df2[column].dropna(),  color='violet', label='test', kde=True); 
            fig=plt.legend(loc='best')
            plt.xlabel(column, fontsize=12);
            plt.show()
            i=i+1


# it is ....
# Let's remove **icu_id** and see the results ..

# In[ ]:


adversarial_features2 = adversarial_validation(train, test, [ f for f in features if f not in ['icu_id'] ])


# In[ ]:


adversarial_features2 = adversarial_features2[['gain', 'feature']].groupby('feature').mean().reset_index()
adversarial_features2= adversarial_features2.sort_values('gain', ascending=False)
plot_importances(adversarial_features2)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=UserWarning)
i=0
for index, row in adversarial_features2.sort_values(by=['gain'],ascending=False).iterrows():  
    column=row['feature']
    if i< 3:
            print(column,i,"gain :",row['gain'])
            df1      = train.copy()
            df2      = test.copy()

            fig = plt.figure(figsize=(20,4))
            sns.distplot(df1[column].dropna(),  color='yellow', label='train', kde=True); 
            sns.distplot(df2[column].dropna(),  color='violet', label='test', kde=True); 
            fig=plt.legend(loc='best')
            plt.xlabel(column, fontsize=12);
            plt.show()
            i=i+1


# **hospital_id** seems to be also from a different distribution.
# We can check it directly, obviously only few hospital are common to both dataset ..

# In[ ]:


common_id  = list([id for id in train['hospital_id'].unique() if id in test['hospital_id'].unique() ])
id_only_in_train  = [id for id in train['hospital_id'].unique() if id not in test['hospital_id'].unique() ]
id_only_in_test   = [id for id in test['hospital_id'].unique()  if id not in train['hospital_id'].unique() ]
count_common_train = train.loc[train['hospital_id'].isin(common_id)].shape[0]
count_common_test  = test.loc[test['hospital_id'].isin(common_id)].shape[0]

count_train = train.loc[train['hospital_id'].isin(id_only_in_train)].shape[0]
count_test  = test.loc[test['hospital_id'].isin(id_only_in_test)].shape[0]
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
 
fig = plt.figure(figsize=(20,6))
venn2(subsets = (count_train,  count_test, count_common_train+count_common_test), set_labels = ('Hospital only in train', 'Hospital only in test'),set_colors=('purple', 'yellow'), alpha = 0.7);
plt.show()


# Let's do an ultimate try **without 'icu_id','hospital_id'**

# In[ ]:


adversarial_features3 = adversarial_validation(train, test, [ f for f in features if f not in ['icu_id','hospital_id'] ])


# I leave it to you to see what you can do with other features..

# In[ ]:


adversarial_features3 = adversarial_features3[['gain', 'feature']].groupby('feature').mean().reset_index()
adversarial_features3= adversarial_features3.sort_values('gain', ascending=False)
plot_importances(adversarial_features3)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=UserWarning)
i=0
for index, row in adversarial_features3.sort_values(by=['gain'],ascending=False).iterrows():  
    column=row['feature']
    if i< 5:
            print(column,i,"gain :",row['gain'])
            df1      = train.copy()
            df2      = test.copy()

            fig = plt.figure(figsize=(20,4))
            sns.distplot(df1[column].dropna(),  color='yellow', label='train', kde=True); 
            sns.distplot(df2[column].dropna(),  color='violet', label='test', kde=True); 
            fig=plt.legend(loc='best')
            plt.xlabel(column, fontsize=12);
            plt.show()
            i=i+1


# In[ ]:


more_drop = ['hospital_id','icu_id','apache4ahospitaldeathprob', 'apache4aicudeath_prob']
features = [col for col in features if col not in more_drop] 
categoricals_features = [col for col in categoricals_features if col not in more_drop] 
lgb_model1 = Lgb_Model(train,test, features, categoricals=categoricals_features, ps=params)


# In[ ]:


#OHE
print('Transform all String features to OHE.\n')
for usecol in categoricals_features:
    train[usecol] = train[usecol].astype('str')
    test[usecol] = test[usecol].astype('str')
    
    train=pd.concat([train,pd.get_dummies(train[usecol],drop_first=True, prefix=usecol)],axis=1)
    test =pd.concat([test ,pd.get_dummies(test[usecol],drop_first=True, prefix=usecol)],axis=1)
    del train[usecol], test[usecol]


# In[ ]:


features = [col for col in train.columns if col not in to_drop and col in test.columns ]


# In[ ]:


lgb_model2 = Lgb_Model(train,test, features, categoricals=[], ps=params)


# In[ ]:


test["hospital_death"] = lgb_model1.y_pred * 0.7 + lgb_model2.y_pred * 0.3 
test[["encounter_id","hospital_death"]].to_csv("submission.csv",index=False)


# In[ ]:


test["hospital_death"] = lgb_model1.y_pred 
test[["encounter_id","hospital_death"]].to_csv("submission1.csv",index=False)


# In[ ]:


test["hospital_death"] = lgb_model2.y_pred 
test[["encounter_id","hospital_death"]].to_csv("submission2.csv",index=False)

