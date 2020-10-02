#!/usr/bin/env python
# coding: utf-8

# ## Covariate shift 
# This data set is closely resembling reality, i.e. the distributions of the various features may vary greatly from one period to the next. This, in turn, can cause great degradation in the model's predictive performance between train and test.
# 
# The purpose of this kernel is to highlight these features in order to remove them or treat them somehow. 
# 
# If you find the kernel usefull, an upvote is alwyas welcome! Enjoy!

# In[ ]:


#  Libraries
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


#Read data
train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')
train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')


# In[ ]:


# Merge
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)


# In[ ]:


# Pre-process
train = train.fillna(-999)
test = test.fillna(-999)

label_y = train['isFraud']
del train['isFraud']

# Label Encoding
print('Label Encoding...')
for f in train.columns:
    if train[f].dtype=='object' or test[f].dtype=='object':
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values)+ list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))


# ### Explanation of next cell (or how to find features with covariate shift)
# - First I assign a new variable (lets say 'origin') to pinpoint each dataset. 
# - Then, I take a random sample of both sets for convenience.
# - In the next step I concatenate the two datasets into a single dataframe
# - Afterwards, I build a simple random forest and I try to predict the new variable ('origin') using only 1 feature at a time
# - I calculate the performance of the model using AUC 
# - Any given variable should not be able to predict the dummy 'origin' and should thus return an AUC score of around 0.5
# - If any variable succeeds in predicting the 'origin' variable (i.e. it is able to distinguish between train and test sets), it becomes a suspect of covariate shift

# In[ ]:


# Create new y label to detect shift covariance
train['origin'] = 0
test['origin'] = 1

# Create a random index to extract random train and test samples
training = train.sample(10000, random_state=12)
testing = test.sample(10000, random_state=11)

## Combine random samples
combi = training.append(testing)
y = combi['origin']
combi.drop('origin',axis=1,inplace=True)

## Modelling
model = RandomForestClassifier(n_estimators = 50, max_depth = 5,min_samples_leaf = 5)
all_scores = []
drop_list = []
score_list =[]
temp = -1
for i in combi.columns:
    temp +=1
    score = cross_val_score(model,pd.DataFrame(combi[i]),y,cv=2,scoring='roc_auc')
    if (np.mean(score) > 0.8):
        drop_list.append(i)
        score_list.append(np.mean(score))
    all_scores.append(np.mean(score))    
    print('Checking feature no {} out of {}'.format(temp, train.shape[1]))
    print(i,np.mean(score))


# In[ ]:


#Print Top 20 features with possible covariate shift
scores_df = pd.DataFrame({'feature':combi.columns, 
                          'score': all_scores})

scores_df = scores_df.sort_values(by = 'score', ascending = False)
scores_df.head(20)


# ### Conclusion 
# So, not only D15, but other features like id_31 seem to present a covariate shift between train and test. 
# I have personally removed both id_31 and D15 original features after including them in some feature engineering and I saw an increase in Public LB. 
# Treat at your own discretion though: these features may contain usefull information. 
# 
# Happy to hear your own thoughts! 
# 

# In[ ]:




