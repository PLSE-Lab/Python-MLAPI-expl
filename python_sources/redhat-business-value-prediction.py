#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge,Lasso
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import skew

print("ok")


# In[ ]:


train = pd.read_csv('../input/act_train.csv')
test = pd.read_csv('../input/act_test.csv')
print(train.head())


# In[ ]:


for i in train.columns:

    print (i,len(train[i]),train[i].notnull().sum(),(train[i].notnull().sum())/len(train[i]))
    


# In[ ]:



for i in train.columns:

    print(i,train[i].isnull().sum())
    


# In[ ]:



people = pd.read_csv('../input/people.csv')

print(people)


# In[ ]:



people.isnull().sum()


# In[ ]:



activity_id = test['activity_id']


# In[ ]:



people['people_id'] = people['people_id'].apply(lambda x : x.split('_')[1])


# In[ ]:



people['people_id'] = pd.to_numeric(people['people_id']).astype(int)


# In[ ]:



print(people['people_id'])


# In[ ]:



train['people_id'] = train['people_id'].apply(lambda x : x.split('_')[1])

train['people_id'] = pd.to_numeric(train['people_id']).astype(int)

test['people_id'] = test['people_id'].apply(lambda x : x.split('_')[1])
test['people_id'] = pd.to_numeric(test['people_id']).astype(int)


# In[ ]:


print(train['people_id'])


# In[ ]:


print (train.isnull().sum())


# In[ ]:


train = train.drop('activity_id',axis = 1)
train = train.drop('date',axis = 1)
test = test.drop('activity_id',axis = 1)
test = test.drop('date',axis = 1)


# In[ ]:


string_feature = train.select_dtypes(include = ['object'])
string_feature_test = test.select_dtypes(include = ['object'])


# In[ ]:


for i in string_feature.columns:
    string_feature[i] = string_feature[i].fillna("type 0")
    string_feature[i] =  string_feature[i].apply(lambda x :x.split(" ")[1])
    string_feature[i] = pd.to_numeric(string_feature[i])

for i in string_feature.columns:
    string_feature_test[i] = string_feature_test[i].fillna("type 0")
    string_feature_test[i] =  string_feature_test[i].apply(lambda x :x.split(" ")[1])
    string_feature_test[i] = pd.to_numeric(string_feature_test[i])


# In[ ]:


print(string_feature,string_feature_test)


# In[ ]:


train_new = string_feature
train_new['people_id'] = train['people_id']
y = train['outcome']
print (train_new.head())

test_new = string_feature_test
test_new['people_id'] = test['people_id']
print (test_new.head())


# In[ ]:


people= people.drop('date',axis = 1)
print (people)


# In[ ]:


string_feature_people = people.select_dtypes(include = ['object'])
bool_feature_people = people.select_dtypes(include = ['bool'])
print(string_feature_people)
print(bool_feature_people)


# In[ ]:


for i in string_feature_people.columns:
    string_feature_people[i] = string_feature_people[i].fillna("type 0")
    string_feature_people[i] = string_feature_people[i].apply(lambda x :x.split(" ")[1])
    string_feature_people[i] = pd.to_numeric(string_feature_people[i]).astype(int)


# In[ ]:


print (string_feature_people)


# In[ ]:


from sklearn.preprocessing import LabelEncoder


for i in bool_feature_people.columns :
    lb = LabelEncoder()
    lb.fit(list(bool_feature_people[i].values) )
    bool_feature_people[i] = lb.transform(list(bool_feature_people[i].values))
print (bool_feature_people)


# In[ ]:


people_new = (pd.concat([string_feature_people,bool_feature_people],axis = 1))
people_new['people_id'] = people['people_id']
people_new['char_38'] = people['char_38']

print (people_new.head())


# In[ ]:


total_train = train_new.merge(people_new,on = 'people_id' ,how = "left")
print (total_train.head(10))

total_test = test_new.merge(people_new,on = 'people_id' ,how = "left")
print(total_test)


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(total_train,y, test_size = 0.2,random_state = 42)
print (X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV

model =  RandomForestClassifier()

model.fit(X_train,Y_train)
predictions = model.predict(X_test)
probability = model.predict_proba(X_test)
print(predictions,probability)


# In[ ]:


import xgboost as xgb
model_xgb = xgb.XGBClassifier(n_estimators=360, max_depth=2)
model_xgb.fit(X_train, Y_train)


# In[ ]:


predictions = model.predict(X_test)
probability = model.predict_proba(X_test)


# In[ ]:


roc_score = roc_auc_score(Y_test,predictions)
print (roc_score)


# In[ ]:


prediction_test = model.predict(total_test)
probability_test = model.predict_proba(total_test)
test_preds = probability_test[:,1]


# In[ ]:


sub = pd.DataFrame()
sub['activity_id'] = activity_id
sub['outcome'] = test_preds

sub.to_csv('submission_redhat.csv',index=False)

