#!/usr/bin/env python
# coding: utf-8

# ## My Kernel for WebClub Recruitment Test 2019

# ### Importing required packages

# In[ ]:


import os
print((os.listdir('../input/')))


# In[ ]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np


# ### Reading the Train and Test Set

# In[ ]:


df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')
df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')
test_index=df_test['Unnamed: 0']


# ### Visualizing the Training Set

# In[ ]:


df_train.dtypes


# In[ ]:


df_train.V6.describe()


# In[ ]:


# Mean and standard deviation suggest outliers in the data which are not useful 
df = df_train[df_train.V6 < 6000]
df = df[df.V6 > -3000]


# In[ ]:


(df_train.Class == 1).sum()


# In[ ]:


(df_train['Class'] == 0).sum()


# Hence data is skewed

# In[ ]:


from sklearn.model_selection import GridSearchCV
params = {"max_depth" : (3,4), "n_estimators" : (100, 150, 200, 250, 300, 350, 400, 450), 'loss' : ('exponential', 'deviance')}


# In[ ]:


from sklearn.model_selection import train_test_split
train_X = df.loc[:, 'V1':'V16']

train_y = df.loc[:, 'Class']
#train_X.drop('V2', axis = 'columns', inplace = True)

x_train, x_test, y_train, y_test = train_test_split(train_X, train_y, test_size = 0.3)

# V2 has many categories for binary classification. However dropping it doesnt help much


# In[ ]:


train_X.columns


# One Hot Encoding avoided since it doesn't affect Tree based models.

# ### Initializing Classifier

# In[ ]:


gb = GradientBoostingClassifier(n_estimators = 350,random_state = 123, max_depth = 3, loss = 'exponential')
#gb= GradientBoostingClassifier()


# In[ ]:


#clf  = GridSearchCV(gb, params, cv = 5, scoring = 'roc_auc')
#clf.fit(train_X, train_y)


# In[ ]:


#(clf.best_params_)


# GridSearchCV gives best parameters as n_estimators = 150, max_depth = 3, loss = 'exponential'

# ### Training Classifier

# In[ ]:


gb.fit(x_train, y_train)


# ### Calculating predictions for the test set

# In[ ]:



pred = gb.predict(x_test)


# In[ ]:


print(f1_score(y_test, pred, average='macro')) 


# In[ ]:


print(roc_auc_score(y_test, pred))


# ### Now we use the entire dataset to train the Gradient Boosted Classifier

# In[ ]:


df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')
df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')
test_index=df_test['Unnamed: 0']


# In[ ]:


df_train = df_train[df_train.V6 < 6000]
df_train = df_train[df_train.V6 > -3000]


# #### Though GridSearchCV gives n_estimators = 100 to be the best parameter, n_estimators = 350 is giving a higher score on submission

# In[ ]:


gb = GradientBoostingClassifier(n_estimators=350, random_state=123, loss = 'exponential', max_depth = 3)
#df_train.drop("V2", axis = 'columns', inplace = True)
train_X = df_train.loc[:, 'V1':'V16']
train_y = df_train.loc[:, 'Class']



gb.fit(train_X, train_y)


# In[ ]:


df_test.head()


# In[ ]:


df_test.columns


# In[ ]:


df_test = df_test.loc[:, 'V1':'V16']
#df_test.drop("V2", axis = 'columns', inplace = True)
pred = gb.predict_proba(df_test)


# ### Writing the results to a file

# In[ ]:


result=pd.DataFrame()
result['Id'] = test_index
result['PredictedValue'] = pd.DataFrame(pred[:,1])
result.head()


# In[ ]:


result.to_csv('output1.csv', index=False)

