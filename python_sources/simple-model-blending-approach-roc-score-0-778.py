#!/usr/bin/env python
# coding: utf-8

# **Solution code for Analytics Vidhya Janata Hack 2020. This is a very basic approach using simple model blending technique. **
# 
# **But this can act as a pipeline for any machine learning competition for beginners. Upvote it if you like it.**

# In[ ]:


# Importing lib

import numpy as np 
import pandas as pd 


# In[ ]:


# Reading data
train= pd.read_csv('/kaggle/input/av-janata-hack-payment-default-prediction/train_20D8GL3.csv')
test= pd.read_csv('/kaggle/input/av-janata-hack-payment-default-prediction/test_O6kKpvt.csv')
sample= pd.read_csv('/kaggle/input/av-janata-hack-payment-default-prediction/sample_submission_gm6gE0l.csv')


# **Data cleaning steps**
# 1. Correcting: check for outliers or ambigous data and fix
# 2. Completing: deal with missing values
# 3. Creating: mainly feature engineering
# 4. Converting: Encoding for categorical data

# In[ ]:


#step1: correcting
#Check data description, these have kinda non-existent labels,so needed fix
#marriage col- should have 1,2,3 but it has an unlabelled 0 as well. replace 0 with 3(others category).
#education col- similary education has unlabelled 6 and 0.
#PAY_0 TO PAY_6 has ambiguity in form of -1 and -2 values.

all_data = [train, test]     #to perform ops on train+test both
for df in all_data:
    df['MARRIAGE'].replace({0 : 3},inplace = True)
    df["EDUCATION"].replace({6 : 5, 0 : 5}, inplace = True)
    df["PAY_0"].replace({-1 : 0, -2 : 0}, inplace = True)
    df["PAY_2"].replace({-1 : 0, -2 : 0}, inplace = True)
    df["PAY_3"].replace({-1 : 0, -2 : 0}, inplace = True)
    df["PAY_4"].replace({-1 : 0, -2 : 0}, inplace = True)
    df["PAY_5"].replace({-1 : 0, -2 : 0}, inplace = True)
    df["PAY_6"].replace({-1 : 0, -2 : 0}, inplace = True)


# In[ ]:


#step2: completing- dealing with null values
print(train.isnull().sum())
print('-----------------')
print(test.isnull().sum())


# #### No Null values in the dataset- step 2 done.

# In[ ]:


#step3: creating (feature engg)
#created Age bins out of Age but it didn't turned out to be an important feature, will update soon


# In[ ]:


#step4: converting (encoding) + feature scaling here if using models other than tree based.
cat_cols = ["SEX","MARRIAGE","EDUCATION"]
train = pd.get_dummies(train, columns = cat_cols, prefix=['SEX','MARRIAGE','EDUCATION'])
test = pd.get_dummies(test, columns = cat_cols, prefix=['SEX','MARRIAGE','EDUCATION'])


# #### All 4 steps of data cleaning process completed. 
# ### To do: Experiment with different encoding methods like target encoding and do feature engineering in later versions. 

# In[ ]:


#final check on the data
print(train.shape)
print(test.shape)
print('--------------------')
train.head()


# In[ ]:


#train.columns.tolist()


# In[ ]:


#remove ID and target field from features list
features= [
 'LIMIT_BAL',
 'AGE',
 'PAY_0',
 'PAY_2',
 'PAY_3',
 'PAY_4',
 'PAY_5',
 'PAY_6',
 'BILL_AMT1',
 'BILL_AMT2',
 'BILL_AMT3',
 'BILL_AMT4',
 'BILL_AMT5',
 'BILL_AMT6',
 'PAY_AMT1',
 'PAY_AMT2',
 'PAY_AMT3',
 'PAY_AMT4',
 'PAY_AMT5',
 'PAY_AMT6',
 'SEX_1',
 'SEX_2',
 'MARRIAGE_1',
 'MARRIAGE_2',
 'MARRIAGE_3',
 'EDUCATION_1',
 'EDUCATION_2',
 'EDUCATION_3',
 'EDUCATION_4',
 'EDUCATION_5']


# In[ ]:


# splitting: for local validation, later will train model on all of train set without split
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y= train_test_split(train[features], train.default_payment_next_month,test_size= 0.2, random_state=12)


# In[ ]:


print(train_x.shape)
print(train_y.shape)
print('---------------')
print(test_x.shape)
print(test_y.shape)


# In[ ]:


#modelling starts here
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import confusion_matrix, roc_auc_score


# In[ ]:


models = [
    #ensemble
    AdaBoostClassifier(),
    ExtraTreesClassifier(),
    GradientBoostingClassifier(),
    RandomForestClassifier(),
    
    #linear models
    LogisticRegression(),
          
    XGBClassifier(),
    LGBMClassifier(),
    CatBoostClassifier()
         ]


# In[ ]:


df_models = pd.DataFrame(columns=['Model_name','ROC'])

i=0
for model in models:
    model.fit(train_x,train_y)
    pred_y = model.predict(test_x)
    proba = model.predict_proba(test_x)[:,1]
    roc_score = roc_auc_score(test_y, proba)
    name = str(model)
    print(name[0:name.find("(")])
    df_models.loc[i,'Model_name']= name[0:name.find("(")]
 
    df_models.loc[i,'ROC']= roc_score
    print(confusion_matrix(test_y,pred_y))
    print("------------------------------------------------------------")
    i=i+1


# In[ ]:


df_models.sort_values('ROC', ascending=False)


# ### We have our top 3 models as Gradient boosting classifier, lgbm and catboost.
# #### TO DO: Hypertuning for all these models, pass categorical fields to catboost. Also, feature scaling could help logistic regression perform much better.

# In[ ]:


#for submission: 
# Model blend from all three models and train on all of training dataset this time.

model= GradientBoostingClassifier()
model.fit(train[features],train.default_payment_next_month)
pp1= model.predict_proba(test[features])

model2= LGBMClassifier()
model2.fit(train[features],train.default_payment_next_month)
pp2= model2.predict_proba(test[features])

model3= CatBoostClassifier()
model3.fit(train[features],train.default_payment_next_month)
pp3= model3.predict_proba(test[features])


# In[ ]:


#using very simple average blend
pp_blend= (pp1 +pp2+pp3)/3


# In[ ]:


pp_blend


# In[ ]:


#submission file
sub = pd.DataFrame({'ID':test['ID'],'default_payment_next_month':pp_blend[:,1]})
sub.to_csv('blend cat+gradient+lgbm.csv',index=False)


# ### Thank you. Here are few things which could help score better.
# 1. Use better validation strategy (stratified k fold)
# 2. Use model stacking technique
# 3. Use gridsearchcv or randomsearch for hypertuning of parameters
# 4. Better feature engineering
# 5. Scaling (Minmax scalar) could help linear models like logistic regression to perfom better
