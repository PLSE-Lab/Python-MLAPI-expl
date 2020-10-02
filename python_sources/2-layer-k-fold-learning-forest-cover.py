#!/usr/bin/env python
# coding: utf-8

# This notebook follows my notebooks:
# * https://www.kaggle.com/arateris/xgb-rf-with-gridsearch-for-forest-classifier/ that looked for hyper-paramters (although it was without all the FE so the parameters may not be ideal anymore)
# * https://www.kaggle.com/arateris/stacked-classifiers-for-forest-cover where I was playing with various FE and orinal stacking
# * https://www.kaggle.com/arateris/probing-stats/ for the Public Test set probing
# 
# There was a lot of inspirations from other notebooks which I will mention on the way as much as I remember.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import gc


from time import time

from collections import Counter
from itertools import combinations

from sklearn.model_selection import cross_val_score,cross_validate, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from mlxtend.classifier import StackingCVClassifier, StackingClassifier
# import lightgbm
from lightgbm import LGBMClassifier, plot_importance
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


from tqdm import tqdm


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# #Import datasets

# In[ ]:


X = pd.read_csv("../input/learn-together/train.csv", index_col='Id')
X_test = pd.read_csv("../input/learn-together/test.csv", index_col='Id')

y = X['Cover_Type'] # this is the target
X = X.drop('Cover_Type', axis = 1)

test_index = X_test.index
num_train = len(X)

print('Train set shape : ', X.shape)
print('Test set shape : ', X_test.shape)

#renaming to avoid long names in the feature engineering
X.rename({'Horizontal_Distance_To_Roadways':'HDR',
              'Horizontal_Distance_To_Hydrology':'HDH',
              'Horizontal_Distance_To_Fire_Points':'HDF',
              'Vertical_Distance_To_Hydrology':'VDH'}, axis="columns", inplace=True)
X_test.rename({'Horizontal_Distance_To_Roadways':'HDR',
              'Horizontal_Distance_To_Hydrology':'HDH',
              'Horizontal_Distance_To_Fire_Points':'HDF',
              'Vertical_Distance_To_Hydrology':'VDH'}, axis="columns", inplace=True)


columns = X.columns
categorial_feat = [] 
X.head()


# Note : large difference between train and test size. Will need to check input distributions.

# In[ ]:


# Helper function to generate submission files.
def to_submission(preds, file_name):
    output = pd.DataFrame({'Id': test_index,
                           'Cover_Type': preds})
    output.to_csv(file_name+'.csv', index=False)


# # Test label distribution
# https://www.kaggle.com/arateris/probing-stats/
# This is to know the distribution of the labels in the (public) test set. This allows to get a better accuracy check and validation during tuning/training phase.
# Notes: 
# * this may not be a good idea if the private/public distribution are different. In this competition, as the distribution is too flat in the train, I somehow hope the private is similar to the the public LB. 
# * I have seen the full/private distributions but prefer not to use it to keep with the competition spirit (use only available data, not the full test answers)
# 
# EDIT : v5 : not using this anymore.
# 

# In[ ]:


# count = { 1: 0.37062,
#  2: 0.49657,
#  3: 0.05947,
#  4: 0.00106,
#  5: 0.01287, 
#  6: 0.02698, 
#  7: 0.03238} 
# weight = [count[x]/(sum(count.values())) for x in range(1,7+1)]
# class_weight_lgbm = {i: v for i, v in enumerate(weight)}  #LGB uses a different way of counting..


# In[ ]:


# checking score with the public test distribution https://www.kaggle.com/nadare/eda-feature-engineering-and-modeling-4th-359

# def imbalanced_accuracy_score(y_true, y_pred):
#     return accuracy_score(y_true, y_pred, sample_weight=[weight[x] for x in y_true-1])

# imbalanced_accuracy_scorer = make_scorer(imbalanced_accuracy_score, greater_is_better=True)

# def imbalanced_cross_val_score(clf, X, y, cfg_args={}, fit_params={}, cv=5):
#     return cross_val_score(clf, X, y, scoring= imbalanced_accuracy_scorer, cv=cv, n_jobs=-1, fit_params=fit_params )


# In[ ]:


X.head()


# In[ ]:


X_test.head()


# # Feature engineering,
# TODO : data cleaning

# In[ ]:


print('Missing Label? ', y.isnull().any())
print('Missing train data? ', X.isnull().any().any())
print('Missing test data? ', X_test.isnull().any().any())


# --> No missing data.

# In[ ]:


print (X.dtypes.value_counts())
print (X_test.dtypes.value_counts())


# --> Everything in numeric. 
# 
# Soil_type and Wilderness_area are categorial data already put as one hot encoded.

# ## Categories vs OHE
# Originally transformed the binary classes to categorial features but maybe not a good idea so in this version I won't keep it for training at the end. 
# I keep it here to do frequency encoding and may try to use it again later.

# In[ ]:


#transform Soil_Type into categorial
def categorify(df, col_string_search, remove_original=False):
    for key_str in col_string_search:
        new_col_name = key_str+'_cat'
        df[new_col_name]=0
        for col in columns:
            if ~str(col).find(key_str):
                df[new_col_name]= df[new_col_name]+int(str(col).lstrip(key_str))*df[col]
                if remove_original:
                    df.drop(col, axis=1, inplace=True)
#         df[new_col_name] = df[new_col_name].astype('category')
    return df


# In[ ]:


cols_to_categorify = ['Soil_Type', 'Wilderness_Area']
X = categorify(X, cols_to_categorify, remove_original=False)
X_test = categorify(X_test, cols_to_categorify, remove_original=False)

#keeping track of the categorial features
categorial_feat.append('Soil_Type_cat')
categorial_feat.append('Wilderness_Area_cat')

X_test.head()


# In[ ]:


X.describe()


# ## Fixing Hillshade_3pm
# Ploting all the histograms I saw one weird thing in the Hillshade_3pm

# In[ ]:


# for col in X.columns:
plt.figure(figsize=(15,5))
sns.distplot(X.Hillshade_3pm)
plt.show()


# In[ ]:


print(X.Hillshade_3pm[(X.Hillshade_3pm<130).to_numpy() &  (X.Hillshade_3pm>120).to_numpy()].value_counts())
print((X.Hillshade_3pm==0).sum())
print((X_test.Hillshade_3pm==0).sum())


# Hill_shade_3pm is missing ~30 values at 126. 
# Hill_shade_3pm has 88 values equal to 0 which probably should not. 1250 zeros in the test set. --> this is not so many values, but still prefer fixing it to avoid spreading bad data in the feature engineering and training.

# In[ ]:


# checking which features correlates with Hillshade_3pm
corr = X[X.Hillshade_3pm!=0].corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr,annot=True)


# In[ ]:


#replacing the zeros for better guess, mainly to avoid zeros in the feature engineering and fake outliers. 
all_data = X.append(X_test)

cols_for_HS = ['Aspect','Slope', 'Hillshade_9am','Hillshade_Noon']
HS_zero = all_data[all_data.Hillshade_3pm==0]
HS_zero.shape

HS_train = all_data[all_data.Hillshade_3pm!=0]
# res = cross_val_score(RandomForestRegressor(n_estimators=100), HS_train.drop('Hillshade_3pm',axis=1), HS_train.Hillshade_3pm, n_jobs=-1, verbose=True)
# print(res) --> Output:  #[0.9996774  0.99989463 0.9999186 ]
##actually, the CV is so close to perfect that there is actually no new information here..keeping it for .. sanity ?

rf_hs = RandomForestRegressor(n_estimators=100).fit(HS_train[cols_for_HS], HS_train.Hillshade_3pm)
out = rf_hs.predict(HS_zero[cols_for_HS]).astype(int)
all_data.loc[HS_zero.index,'Hillshade_3pm'] = out

X['Hillshade_3pm']= all_data.loc[:num_train,'Hillshade_3pm']
X_test['Hillshade_3pm']= all_data.loc[num_train:,'Hillshade_3pm']


# In[ ]:


# Add PCA features

t = time()

pca = PCA(n_components=0.99).fit(all_data)
trans = pca.transform(all_data)
print(trans.shape)

for i in range(trans.shape[1]):
    col_name= 'pca'+str(i+1)
    X[col_name] = trans[:num_train, i]
    X_test[col_name] = trans[num_train:, i]

print('duration: '+ str(time()-t))


# In[ ]:


# Adding Gaussian Mixture features to perform some unsupervised learning hints from the full data
# https://www.kaggle.com/stevegreenau/stacking-multiple-classifiers-clustering

t = time()
components = 10 # TODO check other numbers.  with 10 labels there are a few ones with 0 importances. 
gmix = GaussianMixture(n_components=components) 
gaussian = gmix.fit_predict(StandardScaler().fit_transform(all_data))

X['GM'] = gaussian[:num_train]
X_test['GM'] = gaussian[num_train:]

categorial_feat.append('GM')

for i in range(components):
    X['GM'+str(i)] = gaussian[:num_train]==i  
    X_test['GM'+str(i)] = gaussian[num_train:]==i 

print('duration: '+ str(time()-t))


# In[ ]:


X.head()


# In[ ]:


del all_data
gc.collect()


# In[ ]:


# Helper function to generate some basic FE
def quick_fe(df, cols, operations, max_combination=2):
    
    if max_combination>=2:
        for col1, col2 in combinations(cols, 2):
            for ope in operations:
                if ope=='add': df[col1 + "_add_" + col2] = df[col1]+df[col2]
                elif ope=='minus': df[col1 + "_minus_" + col2] = df[col1]-df[col2]
                elif ope=='minabs': df[col1 + "_minabs_" + col2] = abs(df[col1]-df[col2])
                elif ope=='time': df[col1 + "_time_" + col2] = df[col1]*df[col2]
    if max_combination>=3:
        for col1, col2, col3 in combinations(cols, 3):
            for ope in operations:
                if ope=='add': df[col1 + "_add_" + col2 + "_add_" + col3] = df[col1]+df[col2]+df[col3]
                elif ope=='time': df[col1 + "_time_" + col2+ "_time_" + col3] = df[col1]*df[col2]*df[col3]
    return df

X.head()


# In[ ]:



# group all the FE features
def feature_eng(dataset):
    # https://www.kaggle.com/nadare/eda-feature-engineering-and-modeling-4th-359#nadare's-kernel
    #https://www.kaggle.com/lukeimurfather/adversarial-validation-train-vs-test-distribution
    #https://www.kaggle.com/evimarp/top-6-roosevelt-national-forest-competition
    
    dataset['Distance_hyd'] = (dataset['HDH']**2+dataset['VDH']**2)**0.5

    cols_to_combine = ['HDH', 'HDF', 'HDR']
    dataset = quick_fe(dataset, cols_to_combine, ['add','time','minus', 'minabs'], max_combination=3)

    cols_to_combine = ['Elevation', 'VDH']
    dataset = quick_fe(dataset, cols_to_combine, ['add','time','minus', 'minabs'], max_combination=2)

    dataset['Mean_Distance']=(dataset.HDF + 
                               dataset.Distance_hyd + 
                               dataset.HDR) / 3 
    dataset['Elevation_Adj_distanceH'] = dataset['Elevation'] - 0.25*dataset['Distance_hyd']
    dataset['Elevation_Adj_distanceV'] = dataset['Elevation'] - 0.19*dataset['HDH']

    
    # Hillshade
    hillshade_col = ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    dataset = quick_fe(dataset,hillshade_col, ['add','minus'], max_combination=3)

    dataset["Hillshade_std"] = dataset[hillshade_col].std(axis=1)
    dataset["Hillshade_max"] = dataset[hillshade_col].max(axis=1)
    dataset["Hillshade_min"] = dataset[hillshade_col].min(axis=1)
   
    #Aspect
    dataset['Aspect'] = dataset['Aspect'].astype(int) % 360
    
    from bisect import bisect
    cardinals = [i for i in range(45, 361, 90)]
    points = ['N', 'E', 'S', 'W']
    dataset['Cardinal'] = dataset.Aspect.apply(lambda x: points[bisect(cardinals, x) % 4])
    dataset.loc[:,'North']= dataset['Cardinal']=='N'
    dataset.loc[:,'East']= dataset['Cardinal']=='E'
    dataset.loc[:,'West']= dataset['Cardinal']=='W'
    dataset.loc[:,'South']= dataset['Cardinal']=='S'
    
    dataset['Sin_Aspect'] = np.sin(np.radians(dataset['Aspect'])) # not important feature at all
    dataset['Cos_Aspect'] = np.cos(np.radians(dataset['Aspect']))
    
    dataset['Slope_hyd'] = np.arctan(dataset['VDH']/(dataset['HDH']+0.001))
    dataset.Slope_hyd=dataset.Slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any
    
    dataset['Sin_Slope_hyd'] = np.sin(np.radians(dataset['Slope_hyd']))
    dataset['Cos_Slope_hyd'] = np.cos(np.radians(dataset['Slope_hyd']))

    dataset['Sin_Slope'] = np.sin(np.radians(dataset['Slope'])) # not important feature at all
    dataset['Cos_Slope'] = np.cos(np.radians(dataset['Slope']))
    
    # extremely stony = 4, very stony = 3, stony = 2, rubbly = 1, None = 0
    Soil_to_stony = [4, 3, 1, 1, 1, 2, 0, 0, 3, 1,
                1, 2, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 4, 4, 4, 4, 4, 3, 4, 4, 4, 
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    dataset['Stonyness'] = [Soil_to_stony[x] for x in (dataset['Soil_Type_cat'].astype(int)-1)]
    dataset.loc[:,'Extremely_Stony']= dataset['Stonyness']==4
    dataset.loc[:,'Very_Stony']= dataset['Stonyness']==3
    dataset.loc[:,'Stony']= dataset['Stonyness']==2
    dataset.loc[:,'Rubbly']= dataset['Stonyness']==1
    dataset.loc[:,'Stony_NA']= dataset['Stonyness']==0
    
    return dataset

categorial_feat.append('Stonyness')
categorial_feat.append('Cardinal')

X = feature_eng(X)
X_test = feature_eng(X_test)
columns = X.columns


# In[ ]:


# Frequency encoding  <- Heard it helps the LightGBM.
#https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575#latest-628340
def freq_encoding(df_train, df_test, cols_to_encode):
    df = pd.concat([df_train[cols_to_encode], df_test[cols_to_encode]],axis=0)
    for col in cols_to_encode:
        new_name = col+'_counts'
        temp = df[col].value_counts().to_dict()
        df[new_name] = df[col].map(temp)
        df_train[new_name] = df.loc[:len(df_train),new_name]
        df_test[new_name] = df.loc[len(df_train):,new_name]
    return df_train, df_test


# In[ ]:


selected_cols = categorial_feat #['Soil_Type_cat', 'Wilderness_Area_cat', 'Stonyness', 'Cardinal']
X, X_test = freq_encoding(X, X_test, selected_cols)


# In[ ]:


droping_list = categorial_feat# [col for col in X.columns if ~str(col).find('Soil_Type')]

X.drop(droping_list, axis=1, inplace = True)
X_test.drop(droping_list, axis=1, inplace = True)

columns = X.columns
X_test.head()


# In[ ]:


def mem_reduce(df):
    for col in df.columns:
        if df[col].dtype=='float64': 
            df[col] = df[col].astype('float32')
        if df[col].dtype=='int64': 
            if df[col].max()<1: df[col] = df[col].astype(bool)
            elif df[col].max()<128: df[col] = df[col].astype('int8')
            elif df[col].max()<32768: df[col] = df[col].astype('int16')
            else: df[col] = df[col].astype('int32')
    return df

X= mem_reduce(X)
X_test=mem_reduce(X_test)
gc.collect()


# scaler = StandardScaler()  #--> Standard Scaler ?
# X.loc[:,:] = scaler.fit_transform(X)
# X_test.loc[:,:] = scaler.transform(X_test)
# 

# In[ ]:


X.dtypes


# # Feature removal
# this is a bit of a "cleaning after the party" thing but to speed up training and avoid noise data I try to check the relevant features and remove useless ones. 
# 
# a better approach would be to check the feature one by one (or small groups) when generating them. one reason being: sometime adding a feature won't improve the model but they may share their importance and it gets hard to see if it was actually useful or not.
# 

# In[ ]:


def get_LGBC():
    return LGBMClassifier(n_estimators=500,  
                     learning_rate= 0.1,
                     objective= 'multiclass', 
                     num_class=7,
                     random_state= 2019,
#                      class_weight=class_weight_lgbm,
                     n_jobs=-1)
lgbc= get_LGBC()
lgbc.fit(X,y)

plot_importance(lgbc, ignore_zero=False, figsize=(8,40))


# In[ ]:


#checking how many features we can cut without performance loss
# print(np.mean(cross_val_score(get_LGBC(), X, y, cv=5)))
# print(np.mean(cross_val_score(get_LGBC(), X.drop(X.columns[lgbc.feature_importances_<100], axis=1), y, cv=5)))
# print(np.mean(cross_val_score(get_LGBC(), X.drop(X.columns[lgbc.feature_importances_<50], axis=1), y, cv=5)))
# print(np.mean(cross_val_score(get_LGBC(), X.drop(X.columns[lgbc.feature_importances_<10], axis=1), y, cv=5)))
# print(np.mean(cross_val_score(get_LGBC(), X.drop(X.columns[lgbc.feature_importances_<0], axis=1), y, cv=5)))

#output : 
# 0.8001322751322751
# 0.7950396825396825
# 0.7952380952380953
# 0.7956349206349207
# 0.8001322751322751


# In[ ]:


# just to be safe I remove only the zero importance features.. if the model gets too slow we can cut more.
zero_importance = X.columns[lgbc.feature_importances_==0]
X.drop(zero_importance, axis=1, inplace=True)
X_test.drop(zero_importance, axis=1, inplace=True)


# # Model generation

# (Now quite old) List of classifiers and hyper-parameters
# - XGBClassifier
# -- Params: {'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 10} ?
# -- n_estimators = 719, max_depth = 464 https://www.kaggle.com/phsheth/forestml-part-6-stacking-eval-selected-fets-2  https://www.kaggle.com/joshofg/pure-random-forest-hyperparameter-tuning
# - RFClassifier
# -- {'max_depth': 100, 'max_features': 0.3, 'n_estimators': 2000}  https://www.kaggle.com/arateris/xgb-rf-with-gridsearch-for-forest-classifier/
# -- Params: {n_estimators = 719, max_features = 0.3, max_depth = 464, min_samples_split = 2, min_samples_leaf = 1, bootstrap = False} https://www.kaggle.com/joshofg/pure-random-forest-hyperparameter-tuning
# - ExtraTrees
# -- Params : n_estimators = 750, max_features = 0.3, max_depth = None,  https://www.kaggle.com/arateris/xgb-rf-with-gridsearch-for-forest-classifier/
# - LGBM 
# -- Params : n_estimators=400,  num_leaves=100  ?  https://www.kaggle.com/stevegreenau/stacking-multiple-classifiers-clustering
# -- {'learning_rate': 0.5, 'max_depth': 25, 'n_estimators': 500}  https://www.kaggle.com/arateris/xgb-rf-with-gridsearch-for-forest-classifier/
# - ADABoost 
# -- Params : {max_depth  = 464, min_samples_split = 2, min_samples_leaf = 1,}  https://www.kaggle.com/phsheth/forestml-part-6-stacking-eval-selected-fets-2
# 
# 

# In[ ]:


#prepare df to store pred proba
Id_train=X.index
Id_test=X_test.index

X_train_L1=pd.DataFrame(Id_train)
X_test_L1=pd.DataFrame(Id_test)


# ## L1 training
# inspired by https://www.kaggle.com/nadare/eda-feature-engineering-and-modeling-4th-359#ykskks's-kernel and many blending kernels in other competitions
# 
# I feel, using this 2-layer model is good for the LGB and XGB that can use the early_stopping and comparison with some validation set to improve the training. I don't know if it is possible in the StackingCVClassifier.

# In[ ]:


# L1 training will store the probability predictions of the train and test set both. for the train set, to avoid leakage, we use a K-fold approach (each fold predicts a part of the training set)
def L1_Training(clf, clf_name, X_train, X_test, cv=5, early_stop=False):
    scores = []
    clf_cul=[str(clf_name)+str(i+1) for i in range(7)]
    for i in clf_cul:
        X_train_L1.loc[:, i]=0
        X_test_L1.loc[:, i]=0

    clf_proba = np.zeros((X_test.shape[0], 7))
    for train, val in tqdm(StratifiedKFold(n_splits=cv, shuffle=True, random_state=9999).split(X_train, y)): 
        X_train_loc = X_train.iloc[train,:]
        X_val_loc = X_train.iloc[val,:]
        y_train_loc = y.iloc[train]
        y_val_loc = y.iloc[val]
        if early_stop:
            # fit the model  ##Do we need to reset the model in between loops??
            clf.fit(X_train_loc, 
                    y_train_loc, 
                    verbose=False,
                    eval_set=[(X_train_loc, y_train_loc), (X_val_loc, y_val_loc)], 
                    early_stopping_rounds=50)
            # use this fitted model to predict Test set.
            clf_pred_proba_test = clf.predict_proba(X_test)
            X_test_L1.loc[:, clf_cul] +=  clf_pred_proba_test/ cv  #average over the CV rounds
        else :
            # when no early stoping the prediction of the Test set will be done once for all after (better use the full training set)
            clf.fit(X_train_loc, y_train_loc)
            
        #checking validation
        clf_pred_proba_val = clf.predict_proba(X_val_loc)
        X_train_L1.loc[val, clf_cul]= clf_pred_proba_val
        y_pred = clf.predict(X_val_loc)
        scores.append(accuracy_score(y_pred,y_val_loc))
#         scores.append(imbalanced_accuracy_score(y_pred,y_val_loc))
        
    if ~early_stop:
        #retrain on full data
        clf.fit(X_train,y)
        clf_pred_proba_test = clf.predict_proba(X_test)
        X_test_L1.loc[:, clf_cul] = clf_pred_proba_test
        
    clf_pred_test = X_test_L1.loc[:,clf_cul].to_numpy().argmax(axis=1)+1
    return scores, clf_pred_test


# In[ ]:


# some constant to be low for kernel editing/ code checking and higher for commit/real training.

MODEL_FACTOR = 10
CV = 6


# In[ ]:


def get_XGB():
    return XGBClassifier( n_estimator= 50*MODEL_FACTOR, 
                    learning_rate= 0.1, 
                    max_depth= 50,  
                    objective= 'binary:logistic',
                    random_state= 2019,
#                     sample_weight=count,
                    n_jobs=-1)
def get_LGBM():
    return LGBMClassifier(n_estimators=50*MODEL_FACTOR,  
                     learning_rate= 0.1,
                     objective= 'multiclass', 
                     num_class=7,
                     random_state= 2019,
#                      class_weight=class_weight_lgbm,
                     n_jobs=-1)
def get_RF():
    return RandomForestClassifier(n_estimators = 100*MODEL_FACTOR, 
                            max_features = 0.3, 
                            max_depth = 100, 
                            min_samples_split = 2, 
                            min_samples_leaf = 1,
                            bootstrap = False,
#                             class_weight=count,
                            random_state=2019)
def get_EXT():
    return ExtraTreesClassifier(n_estimators = 75*MODEL_FACTOR, 
                            max_features = 0.3, 
                            max_depth = None, 
                            min_samples_split = 2, 
                            min_samples_leaf = 1,
                            bootstrap = False, 
#                             class_weight=count,
                            random_state=2019)


# In[ ]:


selected_features = X.columns

X_train_select = X[selected_features]
X_test_select = X_test[selected_features]

clf_list = [(get_LGBM(), 'lgbc', True),
            (get_XGB(), 'xgb', True),
            (get_EXT(), 'xtc', False),
            (get_RF(), 'rf', False)]

for clf, clf_name, early_stop in clf_list:
    print ('Fitting L1 : '+ clf_name)
    score, preds = L1_Training(clf, clf_name, X_train_select, X_test_select, cv=CV, early_stop=early_stop) 
    print(str(np.mean(score)) + ' ( ' + str(np.var(score)) + ')')
    to_submission(preds, 'L1_sub_'+clf_name)


# In[ ]:



# lgbc_score, lgb_preds = L1_Training(lgbc, 'lgbc', X_train_select, X_test_select, cv=CV, early_stop=True) 
# to_submission(lgb_preds, 'lgb_sub')
# print(np.mean(lgbc_score))


# In[ ]:



# xgb_score, xgb_preds = L1_Training(xgb, 'xgb', X_train_select, X_test_select, cv=CV, early_stop=True) 
# to_submission(xgb_preds, 'xgb_sub')
# print(np.mean(xgb_score))


# In[ ]:



# rf_score, rf_preds = L1_Training(rf, 'rf', X_train_select, X_test_select, cv=CV, early_stop=False) 
# to_submission(rf_preds, 'rf_sub')
# print(np.mean(rf_score))


# In[ ]:



# xtc_score, xtc_preds = L1_Training( xtc, 'xtc', X_train_select, X_test_select, cv=CV, early_stop=False) 
# to_submission(xtc_preds, 'xtc_sub')
# print(np.mean(xtc_score))


# In[ ]:


X_train_L1.drop('Id', axis=1, inplace=True)
X_test_L1.drop('Id', axis=1, inplace=True)


# ## L2 training
# this time, we train using (X_train_L1, y) (X_test_L1) as input. 
# NOTE: at this stage, this only use the prediction probabilities as input, it could be good to add the original data as well.

# In[ ]:


X_test_L2 = pd.DataFrame(Id_test)

def L2_Training(clf, clf_name, cv=5, early_stop=False):
    scores = []
    clf_proba = np.zeros((X_test.shape[0], 7))
    
    clf_cul=[str(clf_name)+str(i+1) for i in range(7)]
    for i in clf_cul:
        X_test_L2.loc[:, i]=0
        
    for train, val in tqdm(StratifiedKFold(n_splits=cv, shuffle=True, random_state=9999).split(X_train_L1, y)): 
        X_train_loc = X_train_L1.iloc[train,:]
        X_val_loc = X_train_L1.iloc[val,:]
        y_train_loc = y.iloc[train]
        y_val_loc = y.iloc[val]
        if early_stop:
            # fit the model  ##Do we need to reset the model in between loops??
            clf.fit(X_train_loc, y_train_loc, 
                verbose=False,
                eval_set=[(X_train_loc, y_train_loc), (X_val_loc, y_val_loc)], 
                early_stopping_rounds=50)
            # use this fitted model to predict Test set.
            clf_pred_proba_test = clf.predict_proba(X_test_L1)
            X_test_L2.loc[:, clf_cul] +=  clf_pred_proba_test/ cv  #average over the CV rounds
        else :
            # when no early stoping the prediction of the Test set will be done once for all after (better use the full training set)
            clf.fit(X_train_loc, y_train_loc)
            
        #checking validation
        y_pred = clf.predict(X_val_loc)
        scores.append(accuracy_score(y_pred,y_val_loc))
#         scores.append(imbalanced_accuracy_score(y_pred,y_val_loc))
        
    if ~early_stop:
        #retrain on full data
        clf.fit(X_train_L1,y)
        clf_pred_proba_test = clf.predict_proba(X_test_L1)
        X_test_L2.loc[:, clf_cul] = clf_pred_proba_test
        
    clf_pred_test = X_test_L2.loc[:,clf_cul].to_numpy().argmax(axis=1)+1
    return scores, clf_pred_test


# In[ ]:


# Use a "simple" Logistic regression for mixing probabilities and learning the finale version. 
# TODO: try other classifiers.
lr= LogisticRegression(max_iter=1000,#not checked at all hyper-param
                       n_jobs=-1,
                       solver= 'lbfgs',
#                        class_weight=count,
                       multi_class = 'multinomial')

l2_lr_score, l2_lr_preds = L2_Training(lr, 'lr', cv=CV, early_stop=False)
print(l2_lr_score)  
# this shows something surprisingly low (0.6~0.7) yet achieve 0.84 on public LB.. not sure what's happening

to_submission(l2_lr_preds, 'lr_stack_preds_sub')

lgbm= LGBMClassifier(n_estimators=500,  #not checked at all hyper-param
                     learning_rate= 0.1,#not checked at all hyper-param
                     objective= 'multiclass', 
                     num_class=7,
                     random_state= 2019,
#                      class_weight=class_weight_lgbm,
                     n_jobs=-1)

l2_lgbm_score, l2_lgbm_preds = L2_Training(lgbm, 'lgbm', cv=CV, early_stop=True)
print(l2_lgbm_score)
to_submission(l2_lgbm_preds, 'lgbm_stack_preds_sub')

#do we need a third layer ?  :D  This sounds like deep forest learning 


# In[ ]:


plot_importance(lgbm)


# In[ ]:




