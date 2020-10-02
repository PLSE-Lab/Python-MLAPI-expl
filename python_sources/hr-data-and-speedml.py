#!/usr/bin/env python
# coding: utf-8

# This notebook is totally based on the SpeedML notebook written for the famous Titanic machine learning from disaster.

# In[1]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[2]:


df = pd.read_csv('../input/HR_comma_sep.csv')


# In[3]:


df.head()


# In[4]:


len(df)


# In[5]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.3)


# In[6]:


len(train)


# In[7]:


len(test)


# In[8]:


train.head()


# In[9]:


test.head()


# In[10]:


print(train.columns.values)


# In[11]:


train.info()


# In[12]:


train.describe()


# In[13]:


train.describe(include=['O'])


# In[14]:


from speedml import Speedml


# In[15]:


train.to_csv('train.csv')


# In[16]:


test.to_csv('test.csv')


# In[17]:


sml = Speedml('train.csv', 'test.csv', target = 'left')


# In[18]:


sml.train.head()


# In[19]:


sml.plot.correlate()


# In[20]:


sml.plot.distribute()


# In[21]:


sml.configure('overfit_threshold',sml.np.sqrt(sml.train.shape[0])/sml.train.shape[0])


# In[22]:


sml.feature.density('satisfaction_level')


# In[23]:


sml.train[['satisfaction_level', 'satisfaction_level_density']].head()


# In[24]:


sml.feature.density('last_evaluation')


# In[25]:


sml.train[['last_evaluation', 'last_evaluation_density']].head()


# In[26]:


sml.plot.crosstab('left', 'satisfaction_level')


# In[27]:


sml.plot.crosstab('left', 'last_evaluation')


# In[28]:


sml.plot.crosstab('left', 'salary')


# In[29]:


sml.plot.crosstab('left', 'sales')


# In[30]:


sml.feature.labels(['sales'])
sml.train.head()


# In[31]:


sml.feature.labels(['salary'])
sml.train.head()


# In[32]:


sml.eda()


# In[33]:


sml.plot.continuous('promotion_last_5years')


# In[34]:


sml.plot.crosstab('left', 'promotion_last_5years')


# In[35]:


sml.plot.importance()


# In[36]:


sml.feature.drop('Unnamed: 0')


# In[37]:


sml.plot.importance()


# In[38]:


sml.feature.outliers('promotion_last_5years', upper=97)


# In[39]:


sml.plot.continuous('promotion_last_5years')


# In[40]:


sml.plot.importance()


# In[41]:


sml.feature.drop('promotion_last_5years')


# In[42]:


sml.plot.correlate()


# In[43]:


sml.model.data()


# In[44]:


select_params = {'max_depth': [11,12,13], 'min_child_weight': [0,1,2]}
fixed_params = {'learning_rate': 0.1, 'subsample': 0.8, 
                'colsample_bytree': 0.8, 'seed':0, 
                'objective': 'binary:logistic'}

sml.xgb.hyper(select_params, fixed_params)


# In[45]:


select_params = {'learning_rate': [0.3, 0.1, 0.01], 'subsample': [0.7,0.8,0.9]}
fixed_params = {'max_depth': 12, 'min_child_weight': 0, 
                'colsample_bytree': 0.8, 'seed':0, 
                'objective': 'binary:logistic'}

sml.xgb.hyper(select_params, fixed_params)


# In[46]:



tuned_params = {'learning_rate': 0.1, 'subsample': 0.9, 
                'max_depth': 12, 'min_child_weight': 0,
                'seed':0, 'colsample_bytree': 0.8, 
                'objective': 'binary:logistic'}
sml.xgb.cv(tuned_params)


# In[47]:


sml.xgb.cv_results.tail(10)


# In[48]:


tuned_params['n_estimators'] = sml.xgb.cv_results.shape[0] - 1
sml.xgb.params(tuned_params)


# In[49]:



sml.xgb.classifier()


# In[50]:


sml.model.evaluate()


# In[51]:


sml.plot.model_ranks()


# In[52]:


sml.model.ranks()


# In[53]:


sml.xgb.fit()
sml.xgb.predict()


# In[55]:


sml.xgb.feature_selection()


# In[56]:


sml.xgb.sample_accuracy()


# In[ ]:





# In[ ]:





# In[ ]:




