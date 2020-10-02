#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib as mlp
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
import seaborn as sns
import itertools
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool, freeze_support
pd.set_option('display.max_rows', 500)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


### MAIL

# gabriele.maroni@unibg.it


# In[ ]:


def mean_encoding_cv(train, test, categorical, cv=5, drop_categorical=False):
    skf = StratifiedKFold(n_splits=5)
    train_new = train.copy()
    global_mean = train_new[target].mean()
    categorical_enc = []

    for col in categorical:
        print('Encoding variable: ' + col)
        train_new[f'{col}_target_enc'] = np.nan
        categorical_enc.append(f'{col}_target_enc')
        for train_index, valid_index in skf.split(train, train[target]):
            train_fold = train.iloc[train_index].copy()
            valid_fold = train.iloc[valid_index].copy()
            valid_fold[f'{col}_target_enc'] = valid_fold[col].map(train_fold.groupby(col)[target].mean())
            train_new.loc[valid_index, f'{col}_target_enc'] = valid_fold[f'{col}_target_enc']
            train_new[f'{col}_target_enc'].fillna(global_mean, inplace=True) 

    for col in categorical:
        test[f'{col}_target_enc'] = test[col].map(train_new.groupby(col)[f'{col}_target_enc'].mean())
        test[f'{col}_target_enc'].fillna(global_mean, inplace=True) 


    comb = pd.concat([train_new, test])
    if drop_categorical:
        comb.drop(categorical, axis=1, inplace=True)
    return comb, categorical_enc


# In[ ]:


def cross_val_score(train, features, target, N_boot=10, N_bag=10, N_fold=5, max_features=0.7, bootstrap=False, plot=False):   
   N_boot = N_boot
   N_bag = N_bag
   N_fold = N_fold
   max_features = max_features
   cv_preds = []
   scores_list = []

   for j in range(N_boot):
       train2 = train.sample(frac=1, replace=bootstrap, random_state=2*j) # replace=False-> bootstrap off
       for i in range(N_bag): 
           model = DecisionTreeClassifier(random_state=i*42, max_features=max_features,
                                                             criterion='gini',
                                                             splitter='best',
                                                             max_depth=7,
                                                             min_samples_split=580, 
                                                             min_samples_leaf=30)  

           scores = cross_validate(model, train2[features], train2[target], scoring='neg_log_loss', n_jobs=10, verbose=0, cv=N_fold, return_train_score=True)
           #print('Log-Loss oof: ', -np.mean(scores['test_score']), '+-' , np.std(scores['test_score']))
           #print('Log-Loss train: ', -np.mean(scores['train_score']), '+-' , np.std(scores['train_score']))
           scores_list.append(scores)
   oof_scores = pd.DataFrame(np.ravel([-scores_list[i]['test_score'] for i in range(N_boot*N_bag)]), columns=['oof_score'])
   train_scores = pd.DataFrame(np.ravel([-scores_list[i]['train_score'] for i in range(N_boot*N_bag)]), columns=['train_score'])
   print('Mean Log-Loss oof: ', oof_scores.mean().values[0], '+-' , oof_scores.std().values[0])
   print('Mean Log-Loss train: ', train_scores.mean().values[0], '+-' , train_scores.std().values[0])
   if plot:
       sns.boxenplot(x='type', y='score', 
                     data=pd.concat([oof_scores.stack(), train_scores.stack()]).reset_index(1).rename({'level_1':'type', 0:'score'}, axis=1))


# In[ ]:


class NearestNeighborsFeats(BaseEstimator, ClassifierMixin):
    
    def __init__(self, n_jobs, k_list, metric, n_classes=None, n_neighbors=None, eps=1e-6):
        self.n_jobs = n_jobs
        self.k_list = k_list
        self.metric = metric       
        if n_neighbors is None:
            self.n_neighbors = max(k_list) 
        else:
            self.n_neighbors = n_neighbors            
        self.eps = eps        
        self.n_classes_ = n_classes
    
    def fit(self, X, y):    
        self.NN = NearestNeighbors(n_neighbors=max(self.k_list), 
                                      metric=self.metric, 
                                      n_jobs=1, 
                                      algorithm='brute' if self.metric=='cosine' else 'auto')
        self.NN.fit(X)  
        self.y_train = y      
        self.n_classes = np.unique(y).shape[0] if self.n_classes_ is None else self.n_classes_
        
        
    def predict(self, X):       

        if self.n_jobs == 1:
            test_feats = []
            for i in range(X.shape[0]):
                test_feats.append(self.get_features_for_one(X[i:i+1]))
        else:
            test_feats = Pool(self.n_jobs).map(self.get_features_for_one, (X[i:i+1] for i in range(X.shape[0])))            
        return np.vstack(test_feats)
        
        
    def get_features_for_one(self, x):
        NN_output = self.NN.kneighbors(x)
        neighs = NN_output[1][0]
        neighs_dist = NN_output[0][0] 
        neighs_y = self.y_train[neighs] 
        neighs=neighs.astype(int)
        neighs_dist=neighs_dist.astype(float)
        neighs_y=neighs_y.astype(int)
        return_list = [] 
        feats_names = []
        
        for k in self.k_list:
            means_kn = np.mean(neighs_y[:k])
            return_list += [means_kn]
        knn_feats = np.hstack(return_list)
        
        return knn_feats


# # Baseline

# In[ ]:


train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')
test['target'] = np.nan


# In[ ]:


categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features = numerical+categorical
target = 'target'


# In[ ]:


comb, categorical_enc = mean_encoding_cv(train, test, categorical)


# In[ ]:


train = comb[:len(train)].copy()
test = comb[len(train):].copy()
features = numerical + categorical_enc


# In[ ]:


cross_val_score(train, features, target, plot=True)


# # Feature engineering with knn
# ## numerical

# In[ ]:


train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')
test['target'] = np.nan
comb = pd.concat([train, test])
comb[numerical] = (comb[numerical]-comb[numerical].mean())/comb[numerical].std()
train = comb[:len(train)].copy()
test = comb[len(train):].copy()


# In[ ]:


comb, categorical_enc = mean_encoding_cv(train, test, categorical)
train = comb[:len(train)].copy()
test = comb[len(train):].copy()


# In[ ]:


# i compute only the best features choosen with respect to feature selection
k_list=[90] # tested [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
metrics = ['manhattan'] # testes ['minkowski', 'manhattan', 'chebyshev']


# In[ ]:


train_knn_feats_list=[]
for metric in metrics:
    print (metric) 
    skf = StratifiedKFold(n_splits=5)#, shuffle=True, random_state=42)
    NNF = NearestNeighborsFeats(n_jobs=10, k_list=k_list, metric=metric)
    train_knn_feats = cross_val_predict(NNF, train[numerical].values, train[target].values, cv=skf)
    train_knn_feats_list.append(train_knn_feats)


# In[ ]:


test_knn_feats_list=[]
for metric in metrics:
    NNF = NearestNeighborsFeats(n_jobs=10, k_list=k_list, metric=metric)
    NNF.fit(train[numerical].values, train[target].values)
    test_knn_feats = NNF.predict(test[numerical].values)
    test_knn_feats_list.append(test_knn_feats)


# In[ ]:


knn_feature_names = []
for metric in metrics:
    for k in k_list:
        knn_feature_names.append(f'knn_{k}_{metric}_target_mean')


# In[ ]:


train_knn_feats_df = pd.concat([pd.DataFrame(train_knn_feats_list[i], columns=knn_feature_names) for i in range(len(train_knn_feats_list))], axis=1)
test_knn_feats_df = pd.concat([pd.DataFrame(test_knn_feats_list[i], columns=knn_feature_names) for i in range(len(test_knn_feats_list))], axis=1)


# In[ ]:


train_knn = pd.concat([train, train_knn_feats_df], axis=1)
test_knn = pd.concat([test, test_knn_feats_df], axis=1)
features = numerical + categorical_enc + knn_feature_names
features


# In[ ]:


cross_val_score(train_knn, features, target, plot=True)


# In[ ]:


knn_numerical = pd.concat([ train_knn['knn_90_manhattan_target_mean'], test_knn['knn_90_manhattan_target_mean'] ])
knn_numerical.to_csv('/kaggle/working/knn_numerical.csv')


# # Feature engineering with knn
# ## categorical

# In[ ]:


train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')
test['target'] = np.nan


# In[ ]:


categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features = numerical+categorical
target = 'target'


# In[ ]:


comb, categorical_enc = mean_encoding_cv(train, test, categorical, drop_categorical=False)
train = comb[:len(train)].copy()
test = comb[len(train):].copy()


# In[ ]:


le = OrdinalEncoder()
le.fit(train[categorical])
train_enc = le.transform(train[categorical])
test_enc = le.transform(test[categorical])
train[categorical] = train_enc
test[categorical] = test_enc


# In[ ]:


# i compute only the best features choosen with respect to feature selection
k_list=[90] # tested [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
metrics = ['hamming'] # tested ['hamming', 'canberra', 'braycurtis']


# In[ ]:


train_knn_feats_list=[]
if __name__ == '__main__': 
    freeze_support()
    for metric in metrics:
        print (metric) 
        skf = StratifiedKFold(n_splits=5)#, shuffle=True, random_state=42)
        NNF = NearestNeighborsFeats(n_jobs=10, k_list=k_list, metric=metric)
        train_knn_feats = cross_val_predict(NNF, train[categorical].values, train[target].values, cv=skf)
        train_knn_feats_list.append(train_knn_feats)


# In[ ]:


test_knn_feats_list=[]
if __name__ == '__main__': 
    freeze_support()
    for metric in metrics:
        NNF = NearestNeighborsFeats(n_jobs=10, k_list=k_list, metric=metric)
        NNF.fit(train[categorical].values, train[target].values)
        test_knn_feats = NNF.predict(test[categorical].values)
        test_knn_feats_list.append(test_knn_feats)


# In[ ]:


knn_feature_names = []
for metric in metrics:
    for k in k_list:
        knn_feature_names.append(f'knn_{k}_{metric}_target_mean')


# In[ ]:


train_knn_feats_df = pd.concat([pd.DataFrame(train_knn_feats_list[i], columns=knn_feature_names) for i in range(len(train_knn_feats_list))], axis=1)
test_knn_feats_df = pd.concat([pd.DataFrame(test_knn_feats_list[i], columns=knn_feature_names) for i in range(len(test_knn_feats_list))], axis=1)


# In[ ]:


train_knn = pd.concat([train, train_knn_feats_df], axis=1)
test_knn = pd.concat([test, test_knn_feats_df], axis=1)
features = numerical + categorical_enc + knn_feature_names
features


# In[ ]:


cross_val_score(train_knn, features, target, plot=True)


# In[ ]:


knn_categorical = pd.concat([train_knn['knn_90_hamming_target_mean'], test_knn['knn_90_hamming_target_mean']])
knn_categorical.to_csv('/kaggle/working/knn_categorical.csv')


# # Add numerical interactions

# In[ ]:


train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')
knn_num = pd.read_csv('/kaggle/working/knn_numerical.csv')
knn_cat = pd.read_csv('/kaggle/working/knn_categorical.csv')
test['target'] = np.nan
comb = pd.concat([train, test])
comb['knn_90_manhattan_target_mean'] = knn_num['knn_90_manhattan_target_mean'].values
comb['knn_90_hamming_target_mean'] = knn_cat['knn_90_hamming_target_mean'].values


# In[ ]:


categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'knn_90_manhattan_target_mean', 'knn_90_hamming_target_mean']
features = numerical+categorical
target = 'target'


# In[ ]:


interaction_features = []
for combination in list(itertools.combinations(numerical, 2)):
    i,j = combination[0],combination[1]
    comb[i+'_multiplied_by_'+j] = comb[i]*comb[j]
    comb[i+'_sum_by_'+j] = comb[i]+comb[j]
    comb[i+'_diff_by_'+j] = comb[i]-comb[j]
    interaction_features.append(i+'_multiplied_by_'+j)
    interaction_features.append(i+'_sum_by_'+j)
    interaction_features.append(i+'_diff_by_'+j)
    try:
        comb[i+'_divided_by_'+j] = comb[i]/(comb[j]+1e-10)
        interaction_features.append(i+'_divided_by_'+j)
    except:
        pass


# In[ ]:


# selected by feature importance
interaction_features=['knn_90_manhattan_target_mean_sum_by_knn_90_hamming_target_mean',
       'knn_90_manhattan_target_mean_multiplied_by_knn_90_hamming_target_mean',
       'capital-gain_sum_by_knn_90_hamming_target_mean',
       'capital-gain_diff_by_capital-loss',
       'capital-gain_sum_by_knn_90_manhattan_target_mean',
       'capital-loss_diff_by_hours-per-week',
       'age_multiplied_by_knn_90_hamming_target_mean',
       'capital-gain_sum_by_capital-loss',
       'capital-gain_sum_by_hours-per-week',
       'capital-gain_diff_by_knn_90_manhattan_target_mean',
       'capital-gain_divided_by_capital-loss', 'age_diff_by_capital-gain']


# In[ ]:


train = comb[:len(train)].copy()
test = comb[len(train):].copy()


# In[ ]:


numerical_tree = []
for n in numerical+interaction_features:    
    param_grid = {'max_depth': [6,7,8,9,10,11,12]}
    model = GridSearchCV(DecisionTreeClassifier(random_state=0), cv = 5, scoring = 'roc_auc', param_grid = param_grid)
    model.fit(train[n].to_frame(), train[target])
    train[f'{n}_tree'] = model.predict_proba(train[n].to_frame())[:,1]
    test[f'{n}_tree'] = model.predict_proba(test[n].to_frame())[:,1]
    numerical_tree.append(f'{n}_tree')


# In[ ]:


comb, categorical_enc = mean_encoding_cv(train, test, categorical+numerical_tree)
train = comb[:len(train)].copy()
test = comb[len(train):].copy()


# In[ ]:


features = categorical_enc
features


# In[ ]:


cross_val_score(train, features, target, plot=True)


# # Add grouping features

# In[ ]:


train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')
knn_num = pd.read_csv('/kaggle/working/knn_numerical.csv')
knn_cat = pd.read_csv('/kaggle/working/knn_categorical.csv')
test['target'] = np.nan
comb = pd.concat([train, test])
comb['knn_90_manhattan_target_mean'] = knn_num['knn_90_manhattan_target_mean'].values
comb['knn_90_hamming_target_mean'] = knn_cat['knn_90_hamming_target_mean'].values


# In[ ]:


categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'knn_90_manhattan_target_mean', 'knn_90_hamming_target_mean']
features = numerical+categorical
target = 'target'


# In[ ]:


interaction_features = []
for combination in list(itertools.combinations(numerical, 2)):
    i,j = combination[0],combination[1]
    comb[i+'_multiplied_by_'+j] = comb[i]*comb[j]
    comb[i+'_sum_by_'+j] = comb[i]+comb[j]
    comb[i+'_diff_by_'+j] = comb[i]-comb[j]
    interaction_features.append(i+'_multiplied_by_'+j)
    interaction_features.append(i+'_sum_by_'+j)
    interaction_features.append(i+'_diff_by_'+j)
    try:
        comb[i+'_divided_by_'+j] = comb[i]/(comb[j]+1e-10)
        interaction_features.append(i+'_divided_by_'+j)
    except:
        pass


# In[ ]:


# selected by feature importance
interaction_features=['knn_90_manhattan_target_mean_sum_by_knn_90_hamming_target_mean',
       'knn_90_manhattan_target_mean_multiplied_by_knn_90_hamming_target_mean',
       'capital-gain_sum_by_knn_90_hamming_target_mean',
       'capital-gain_diff_by_capital-loss',
       'capital-gain_sum_by_knn_90_manhattan_target_mean',
       'capital-loss_diff_by_hours-per-week',
       'age_multiplied_by_knn_90_hamming_target_mean',
       'capital-gain_sum_by_capital-loss',
       'capital-gain_sum_by_hours-per-week',
       'capital-gain_diff_by_knn_90_manhattan_target_mean',
       'capital-gain_divided_by_capital-loss', 'age_diff_by_capital-gain']


# In[ ]:


import itertools
grouped_features=[]
for cat, num in list(itertools.product(categorical, numerical)):
    comb_tmp = pd.DataFrame(index=comb.groupby(cat).size().index)
    comb_tmp[num +'_mean_gropby_'+cat] = comb.groupby(cat)[num].mean().values
    comb = pd.merge(comb, comb_tmp, how='left', on=[cat])
    comb[num+'_minus_'+num +'_mean_gropby_'+cat] = comb[num].values - comb[num +'_mean_gropby_'+cat]
    grouped_features.append(num +'_mean_gropby_'+cat)
    grouped_features.append(num+'_minus_'+num +'_mean_gropby_'+cat)


# In[ ]:


grouped_features = ['knn_90_hamming_target_mean_minus_knn_90_hamming_target_mean_mean_gropby_race',
       'capital-gain_minus_capital-gain_mean_gropby_native-country',
       'knn_90_hamming_target_mean_minus_knn_90_hamming_target_mean_mean_gropby_native-country',
       'capital-gain_minus_capital-gain_mean_gropby_race',
       'knn_90_manhattan_target_mean_minus_knn_90_manhattan_target_mean_mean_gropby_sex',
       'capital-gain_minus_capital-gain_mean_gropby_relationship',
       'knn_90_manhattan_target_mean_minus_knn_90_manhattan_target_mean_mean_gropby_workclass',
       'capital-gain_minus_capital-gain_mean_gropby_marital-status',
       'knn_90_manhattan_target_mean_minus_knn_90_manhattan_target_mean_mean_gropby_native-country',
       'knn_90_hamming_target_mean_minus_knn_90_hamming_target_mean_mean_gropby_sex']


# In[ ]:


train = comb[:len(train)].copy()
test = comb[len(train):].copy()


# In[ ]:


numerical_tree = []
for n in numerical+interaction_features+grouped_features:    
    param_grid = {'max_depth': [6,7,8,9,10,11,12]}
    model = GridSearchCV(DecisionTreeClassifier(random_state=0),cv = 5, n_jobs=-1, scoring = 'roc_auc', param_grid = param_grid)
    model.fit(train[n].to_frame(), train[target])
    train[f'{n}_tree'] = model.predict_proba(train[n].to_frame())[:,1]
    test[f'{n}_tree'] = model.predict_proba(test[n].to_frame())[:,1]
    numerical_tree.append(f'{n}_tree')


# In[ ]:


comb, categorical_enc = mean_encoding_cv(train, test, categorical+numerical_tree)
train = comb[:len(train)].copy()
test = comb[len(train):].copy()


# In[ ]:


train = comb[:len(train)].copy()
test = comb[len(train):].copy()
features = categorical_enc
features


# In[ ]:


cross_val_score(train, features, target, plot=True)


# In[ ]:


comb = pd.concat([train, test])
comb.to_csv('/kaggle/working/comb_final.csv')


# # Bagging for prediction

# In[ ]:


N_boot = 50
N_bag = 5
N_fold = 5
max_features_list = [0.8]
preds_list = []
scores = []

for max_features in max_features_list:
    print('-'*200)
    for i in range(N_boot):
        print('Bootstrapping round: ', i)
        train2 = train.sample(frac=1, replace=True, random_state=31*i) # replace=True-> bootstrap on
        index = train2.index
        skf = StratifiedKFold(n_splits=N_fold, shuffle=False) # so im sure cross_val_predict does not shuffle dataset 
        for j in range(N_bag):
            preds_df = pd.DataFrame(index=range(len(train)), columns=['original_index', 'preds'])
            preds_df['original_index'] = index
            model = DecisionTreeClassifier(random_state=j*42, max_features=max_features,
                                                              criterion='gini',
                                                              splitter='best',
                                                              max_depth=7,
                                                              min_samples_split=480, 
                                                              min_samples_leaf=30)

            preds=cross_val_predict(model,train2[features],train2[target],n_jobs=10,verbose=0,cv=skf,method='predict_proba')
            preds_df['preds'] = preds[:,1]
            preds_list.append(preds_df)
    mean_preds = pd.concat(preds_list, ignore_index=True).groupby('original_index').mean()
    print(f'Log-loss with max_features {max_features}: ', log_loss(train[target], mean_preds))
    scores.append(log_loss(train[target], mean_preds))


# In[ ]:


N_boot = 50
N_bag = 5
max_features=0.8
predictions_test = []

for i in range(N_boot):
    print('Bootstrapping round: ', i)
    train2 = train.sample(frac=1, replace=True, random_state=31*i) # replace=True-> bootstrap on
    index = train2.index
    for j in range(N_bag):
        model = DecisionTreeClassifier(random_state=j*42, max_features=max_features,
                                                          criterion='gini',
                                                          splitter='best',
                                                          max_depth=7,
                                                          min_samples_split=480, 
                                                          min_samples_leaf=30)
        model.fit(train2[features], train2[target])
        p_test = model.predict_proba(test[features])[:, 1]
        predictions_test.append(p_test)


# In[ ]:


preds_test = pd.DataFrame(predictions_test).T.mean(axis=1)
preds_test.plot.hist(density=True, bins=100)


# In[ ]:


df_test_sub = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')
df_submit = pd.DataFrame({'uid': df_test_sub['uid'],'target': preds_test})
df_submit.to_csv('/kaggle/working/submission.csv', index=False)


# # Feature importance

# In[ ]:


N_bag = 20
N_fold = 5
max_features = 0.7
cv_preds = []
feat_imp_list = []

for i in range(N_bag): 
    print('Bag: ', i)
    model = DecisionTreeClassifier(random_state=i*42, max_features=max_features,
                                                      criterion='gini',
                                                      splitter='best',
                                                      max_depth=7,
                                                      min_samples_split=580, 
                                                      min_samples_leaf=30)
    
    skf = StratifiedKFold(n_splits=N_fold, random_state=i*123, shuffle=True)
    for train_index, valid_index in skf.split(train, train[target]):
        train_fold = train.iloc[train_index].copy()
        valid_fold = train.iloc[valid_index].copy()
        model.fit(train_fold[features], train_fold[target])
        result = permutation_importance(model, valid_fold[features], valid_fold[target], n_repeats=10,
                                        random_state=i*42, n_jobs=-1, scoring='neg_log_loss')
        feat_imp_list.append(result)


# In[ ]:


Feature_importance = pd.DataFrame(pd.DataFrame(feat_imp_list).importances_mean.explode().astype(float).values,
                                  index=(list(train[features].columns)*N_bag*N_fold), columns=['importance'])
Feature_importance.reset_index(drop=False, inplace=True)
Feature_importance.rename({'index': 'feature_name'}, axis=1, inplace=True)
means = Feature_importance.groupby('feature_name').mean()
means.sort_values('importance', ascending=False, inplace=True)
Feature_importance=Feature_importance.set_index('feature_name').loc[means.index]
Feature_importance.reset_index(drop=False, inplace=True)
plt.figure(figsize=(8,10))
sns.boxplot(y='feature_name', x='importance', data=Feature_importance)


# In[ ]:




