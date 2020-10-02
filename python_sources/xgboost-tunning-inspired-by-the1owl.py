# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 13:02:31 2017

@author: dolatae
inspired by the1owl: https://www.kaggle.com/the1owl
"""

##!/usr/bin/env python2
## -*- coding: utf-8 -*-
#"""
#Created on Fri Jul 28 11:04:20 2017
#
#@author: elhamdolatabadi
#"""
#
#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib.pyplot as plt
#import seaborn as sns
#from nltk.tokenize import RegexpTokenizer
#from nltk.stem.porter import PorterStemmer
#from stop_words import get_stop_words
from __future__ import print_function
from sklearn import preprocessing
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import model_selection


# read train and stage_1 test
train = pd.read_csv('../input/training_variants')
test_s1 = pd.read_csv('../input/test_variants')
trainx = pd.read_csv('../input/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx_s1 = pd.read_csv('../input/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

#read stage_2 test
test = pd.read_csv('../input/stage2_test_variants.csv')
testx = pd.read_csv('../input/stage2_test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values
test = test.drop(['ID'], axis=1)

# merge train text and varients
train = pd.merge(train, trainx, how='left', on='ID').fillna('')
y = train['Class'].values
train = train.drop(['Class'], axis=1)
train = train.drop(['ID'], axis=1)

test_s1 = pd.merge(test_s1, testx_s1, how='left', on='ID').fillna('')
# read stage1 solutions
df_labels_test = pd.read_csv('../input/stage1_solution_filtered.csv')
df_labels_test['Class'] = pd.to_numeric(df_labels_test.drop('ID', axis=1).idxmax(axis=1).str[5:])

# join with test_data on same indexes
test_s1 = test_s1.merge(df_labels_test[['ID', 'Class']], on='ID', how='left').drop('ID', axis=1)
test_s1 = test_s1[test_s1['Class'].notnull()]
y_test = test_s1['Class'].values
test_s1 = test_s1.drop(['Class'], axis=1)

# join train and test files
train = pd.concat([train, test_s1])
y_s2 = np.append(y,y_test)

df_all = pd.concat((train, test), axis=0, ignore_index=True)
df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)

gen_var_lst = sorted(list(train.Gene.unique()) + list(train.Variation.unique()))
gen_var_lst = [x for x in gen_var_lst if len(x.split(' '))==1]

for c in df_all.columns:
    if df_all[c].dtype == 'object':
        if c in ['Gene','Variation']:
            lbl = preprocessing.LabelEncoder()
            df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)  
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
        elif c != 'Text':
            lbl = preprocessing.LabelEncoder()
            df_all[c] = lbl.fit_transform(df_all[c].values)
        if c=='Text': 
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' '))) 
            
            
train = df_all.iloc[:len(train)]
test = df_all.iloc[len(train):]            

class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x.drop(['Gene', 'Variation','Text'],axis=1).values
        return x

class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.key].apply(str)

#Tune Pipeline parameters using GridSearch
pipes = Pipeline([
    ('union', FeatureUnion(
        n_jobs = -1,
        transformer_list = [
            ('standard', cust_regression_vals()),
            ('pi1', Pipeline([('Gene', cust_txt_col('Gene')), ('vect_Gene', CountVectorizer(analyzer=u'char')), ('tsvd1', TruncatedSVD(n_iter=25, random_state=12))])),
            ('pi2', Pipeline([('Variation', cust_txt_col('Variation')), ('vect_Variation', CountVectorizer(analyzer=u'char')), ('tsvd2', TruncatedSVD(n_iter=25, random_state=12))])),
            #commented for Kaggle Limits
            ('pi3', Pipeline([('Text', cust_txt_col('Text')), ('tfidf', TfidfVectorizer()), ('tsvd3', TruncatedSVD(n_iter=25, random_state=12))]))
        ])
    ),
    ('xgb', xgb.XGBClassifier())])

     
grid_parameters = {
    'union__pi1__vect_Gene__max_df': [0.5,0.75,1],
    'union__pi1__vect_Gene__ngram_range': ((1, 2),(1,4),(1,8)),  # unigrams or bigrams'vect_Gene__max_df': (0.5, 0.75, 1.0),
    'union__pi2__vect_Variation__max_df': (0.5,0.75,1),
    'union__pi1__tsvd1__n_components' : (5,10,20),
    'union__pi2__tsvd2__n_components' : (5,10,20),
    'union__pi3__tsvd3__n_components' : (10,20,30,40,50),
    'union__pi3__tfidf__ngram_range': ((1, 2),(1,4),(1,8)),
    'xgb__max_depth': [1,15,100,1000],
    'xgb__min_child_weight': [1,15,100,1000],
    'xgb__learning_rate': [0.01,0.1,1],
    'xgb__gamma': np.arange(0,5,.5),
    'xgb__n_estimators': [10,100]
}    
y_s2 = y_s2 - 1 #fix for zero bound array
clf = GridSearchCV(pipes, grid_parameters, scoring='log_loss', n_jobs=-1, verbose=2) 
clf.fit(train, y_s2)  
    
# choose best parameters to transform train and test 
pipe_ = Pipeline([
    ('union', FeatureUnion(
        n_jobs = -1,
        transformer_list = [
            ('standard', cust_regression_vals()),
            ('pi1', Pipeline([('Gene', cust_txt_col('Gene')), ('vect_Gene', CountVectorizer(analyzer=u'char')), ('tsvd1', TruncatedSVD(n_iter=25, random_state=12))])),
            ('pi2', Pipeline([('Variation', cust_txt_col('Variation')), ('vect_Variation', CountVectorizer(analyzer=u'char')), ('tsvd2', TruncatedSVD(n_iter=25, random_state=12))])),
            #commented for Kaggle Limits
            ('pi3', Pipeline([('Text', cust_txt_col('Text')), ('tfidf', TfidfVectorizer()), ('tsvd3', TruncatedSVD(n_iter=25, random_state=12))]))
        ])
    )])

best_parameters = {
    'union__pi1__vect_Gene__max_df': [clf.best_params_.get('union__pi1__vect_Gene__max_df')],
    'union__pi1__vect_Gene__ngram_range': [clf.best_params_.get('union__pi1__vect_Gene__ngram_range')],
    'union__pi2__vect_Variation__max_features': [clf.best_params_.get('union__pi2__vect_Variation__max_features')],    
    'union__pi2__vect_Variation__max_df': [clf.best_params_.get('union__pi2__vect_Variation__max_df')],
    'union__pi2__vect_Variation__ngram_range': [clf.best_params_.get('union__pi2__vect_Variation__ngram_range')],
    'union__pi1__tsvd1__n_components' : [clf.best_params_.get('union__pi1__tsvd1__n_components')],
    'union__pi2__tsvd2__n_components' : [clf.best_params_.get('union__pi2__tsvd2__n_component')],
    'union__pi3__tsvd3__n_components' : [clf.best_params_.get('union__pi3__tsvd3__n_components')],
    'union__pi3__tfidf__ngram_range': [clf.best_params_.get('union__pi3__tfidf__ngram_range')]
}    


train = pipe_.fit_transform(train,*best_parameters)
print(train.shape)
test = pipe_.transform(test)
print(test.shape)

denom = 0
iter_ = 1
counter = 0
for iter in range(iter_):
    xgb_params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'seed': iter,
        'silent': True,
        'max_depth':[clf.best_params_.get('xgb__max_depth')],
        'min_child_weight': [clf.best_params_.get('xgb__min_child_weight')],
        'learning_rate': [clf.best_params_.get('xgb__learning_rate')],
        'gamma': [clf.best_params_.get('xgb__gamma')],
        'n_estimators': [clf.best_params_.get('xgb__n_estimators')]
                         }
    x1, x2, y1, y2 = model_selection.train_test_split(train, y_s2, test_size=0.2, random_state=iter)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(xgb_params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
    score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), labels = list(range(9)))
    print(score1)
    if counter != 0:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
        preds += pred
    else:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
        preds = pred.copy()
    counter += 1
    submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = pid   
    submission.to_csv('submission___xgb_iter_'  + str(iter) + '.csv', index=False)
preds /= denom
submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('submission___xgb_.csv', index=False) 