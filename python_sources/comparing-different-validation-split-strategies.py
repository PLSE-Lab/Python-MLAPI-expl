#!/usr/bin/env python
# coding: utf-8

# The goal of machine learning excercises is to train an effective generaliser - that is to find some function that is effectively able to map some new samples of data to an unknown target label or value.
# 
# In order to assess the effectiveness of our models we need to use some form of test set. Kaggle provides a test set divided into two parts: public and private. 
# 
# However, we need to avoid over fitting - building models which are too sensitive to patterns in a subset of our data and so do not generalise as well to new data.
# 
# If we simply used the kaggle public test set for our test set we may end up overfitting to that data set and so see a big drop in our model accuracy on the private leaderboard.
# Using the Kaggle public leaderboard to evaluate our models and feature engineering is also slow and inefficient.
# 
# The solution to these problems is constructing a validation data set from the training data and using this for our models.
# This validation set needs to be sufficiently similar to the training data set.
# 
# This kernel seeks to investigate different validation data set construction approaches and is based on some of the lessons from Jeremy Howard's Fastai "Introduction to Machine Learning for Coders"
# 
# I'm used to working in R so a lot of these python libraries are new to me. Any suggestions for improvement are greatly appreciated.

# In[ ]:


#install the fastai library version 0.7 as used in the fastai course
get_ipython().system('pip install fastai==0.7.0 --no-deps')


# In[ ]:


#import libraries, this code follows the imports from the fastai course
import os
get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import  RandomForestClassifier, GradientBoostingClassifier
from IPython.display import display

from sklearn import metrics
np.random.seed(20190810)


# In[ ]:


#load data, we only need the transactions dataset for now
#train_id = pd.read_csv('/kaggle/input/train_identity.csv')
train_trans = pd.read_csv('/kaggle/input/train_transaction.csv')
#test_id = pd.read_csv('/kaggle/input/test_identity.csv')
test_trans = pd.read_csv('/kaggle/input/test_transaction.csv')
#sample_submission = pd.read_csv('/kaggle/input/sample_submission.csv')


# In[ ]:


#the proc_df installed in kaggle was not working properly so copied the definition from my pc
def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):

    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res


# One potential issue in our validation data set is data leakage due to a time component in the data set.
# Our data set has a time based column TransactionDT 

# In[ ]:


#testing for some sort of time order in the data set
experiment_set = train_trans.tail(100000)
last_20k = experiment_set.tail(20000)
first_80k = experiment_set.head(80000)
random_20k = first_80k.sample(20000)
experiment_train = first_80k[~first_80k.TransactionID.isin(random_20k.TransactionID)]


# In[ ]:


def print_score(m,X_train,y_train,X_valid,y_valid):
    res = {'auc_train' : metrics.roc_auc_score(y_train,m.predict_proba(X_train)[:,1]),
           'auc_valid' : metrics.roc_auc_score(y_valid,m.predict_proba(X_valid)[:,1])}
    if hasattr(m, 'oob_score_'): res["oob"] = m.oob_score_
    print(res)


# In[ ]:


train_cats(experiment_train)
apply_cats(df=random_20k, trn=experiment_train)

X, y , nas = proc_df(experiment_train, 'isFraud')
random_valid, random_valid_y, _ = proc_df(random_20k,'isFraud', na_dict=nas)


m = RandomForestClassifier(n_jobs=-1, n_estimators=10)

m.fit(X, y)

print_score(m,X,y,random_valid,random_valid_y)


# In[ ]:


train_cats(experiment_train)
apply_cats(df=last_20k, trn=experiment_train)

X, y , nas = proc_df(experiment_train, 'isFraud')

last_valid, last_valid_y, _= proc_df(last_20k,'isFraud', na_dict=nas)


m = RandomForestClassifier(n_jobs=-1, n_estimators=10)

m.fit(X, y)

print_score(m,X,y, last_valid, last_valid_y)


# We can see that the auc on the random validation set is significantly higher than the auc on the last rows of the dataset - this indicates data leakage.
# If our test set is on data later than the training set then we need to replicate this in our validation set.
# A simple plot will allow us to see whether this is the case.

# In[ ]:


#create a data frame from the TransactionDT columns of the train and test set and add an index to make plotting easy.
df1 = pd.DataFrame(train_trans.TransactionDT)
df1['datasource'] = "train"
df2 = pd.DataFrame(test_trans.TransactionDT)
df2['datasource'] = "test"
df = pd.concat([df1,df2])
df = df.reset_index()


# In[ ]:


#create a sampled scatterplot showing the distribution
sns.scatterplot(x='TransactionDT', y='index',hue='datasource', data=df.sample(100000),edgecolors='none',marker='.')
plt.axvline(df1.TransactionDT.max())
plt.axvline(df2.TransactionDT.min())
plt.title("Ranges of the TransactionDT values in the two data sets with lines showing max and min of the two datasets")


# In[ ]:


#need to delete objects from memory in order to free up memory for model training.
del[df1,df2,df,last_valid,last_valid_y,random_valid,random_valid_y,experiment_set,last_20k,first_80k,random_20k,experiment_train,m]


# We can see that the test set does not overlap the training set and is completely after it in time. 
# This visualisation and the clear differences in the results of our models mean that we should use the last rows of the train set as the validation dataset. 
# 
# Next step is to check that the prediction AUC is similar for an actual local validation data set and the public leaderboard.
# To do this we will take the last 90540 rows of the training set as a validation set.
# We will then train a number of different models on this training set and compare the performance of these models on this validations set and the public leaderboard score. If the validation set is good we would expect a consistent positive correlation.
# 
# NB. Our goal in building a validation set is to create a data set that will allow us to reliably identify improvements in our models. We do not need to know what the AUC on the test set will be as much as we need to know if a change we make to our model will result in an improvement to the AUC on the test set.

# In[ ]:


tt_train = train_trans.head(500000)
tt_valid = train_trans.tail(90540)

train_cats(tt_train)
apply_cats(df=tt_valid, trn=tt_train)

X, y , nas = proc_df(tt_train, 'isFraud')

valid, valid_y, _= proc_df(tt_valid,'isFraud', na_dict=nas)

apply_cats(df=test_trans, trn=tt_train)
X_test, _ , _= proc_df(test_trans,na_dict=nas)


# In[ ]:


del([test_trans,train_trans,tt_train,tt_valid])


# In[ ]:


mrf1 = RandomForestClassifier(n_jobs=-1, n_estimators=10)
mrf1.fit(X, y)

#output predictions and make submission files
mrf1_sub = pd.DataFrame()
mrf1_sub['TransactionID']=X_test['TransactionID']
mrf1_sub['isFraud']=mrf1.predict_proba(X_test)[:,1]
mrf1_sub.to_csv('mrf1_submission.csv', index=False)
print_score(mrf1,X,y, valid, valid_y)


# In[ ]:


del([mrf1,mrf1_sub])


# valid score: 0.8273
# 
# test lb score: 0.8506

# In[ ]:


mrf2 = RandomForestClassifier(n_jobs=-1, n_estimators=10,min_samples_leaf=1000)
mrf2.fit(X, y)

mrf2_sub = pd.DataFrame()
mrf2_sub['TransactionID']=X_test['TransactionID']
mrf2_sub['isFraud']=mrf2.predict_proba(X_test)[:,1]
mrf2_sub.to_csv('mrf2_submission.csv', index=False)
print_score(mrf2,X,y, valid, valid_y)


# In[ ]:


del([mrf2,mrf2_sub])


# valid score: 0.8599
# 
# test lb score: 0.8826

# In[ ]:


mgb1 = GradientBoostingClassifier(n_estimators=10)
mgb1.fit(X, y)

mgb1_sub = pd.DataFrame()
mgb1_sub['TransactionID']=X_test['TransactionID']
mgb1_sub['isFraud']=mgb1.predict_proba(X_test)[:,1]
mgb1_sub.to_csv('mgb1_submission.csv', index=False)
print_score(mgb1,X,y, valid, valid_y)


# valid score: 0.8044
# 
# test lb score: 0.8368

# In[ ]:


del([mgb1,mgb1_sub])


# In[ ]:


mgb2 = GradientBoostingClassifier(n_estimators=10,min_samples_leaf=1000)
mgb2.fit(X, y)

mgb2_sub = pd.DataFrame()
mgb2_sub['TransactionID']=X_test['TransactionID']
mgb2_sub['isFraud']=mgb2.predict_proba(X_test)[:,1]
mgb2_sub.to_csv('mgb2_submission.csv', index=False)
print_score(mgb2,X,y, valid, valid_y)


# valid score: 0.8105
# 
# test lb score: 8424

# In[ ]:


del([mgb2,mgb2_sub])


# In[ ]:


outcome = pd.DataFrame({'valid':[0.8273,0.8599,0.8044,0.8105],
          'public':[0.8506,0.8826, 0.8386,0.8424]})
outcome.plot.scatter(x='valid',y='public')
plt.title('Performance of different models on public leaderboard and validation set')


# We see that there is a strong correlation between the auc on the public lb and the valid dataset.
# This seems to be a reliable validation set for comparing different models.
