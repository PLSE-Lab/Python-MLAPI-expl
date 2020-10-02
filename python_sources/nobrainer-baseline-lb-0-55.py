#!/usr/bin/env python
# coding: utf-8

# This notebook gives a nobrainer baseline

# In[ ]:


# Init
import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


# Get files
train= pd.read_csv('../input/train.csv')
test= pd.read_csv('../input/test.csv')


# Get the numeric columns

# In[ ]:


numeric = train.select_dtypes(include=[np.number])


# Change the categorical input into dummies

# In[ ]:


string_var = train.select_dtypes(include=['object'])
dummies =pd.get_dummies(string_var)


# Merge the numeric columns with the dummie-frame

# In[ ]:


train_new = numeric.join(dummies)


# Get all feature columns

# In[ ]:


features = [col for col in list(train_new) if col.startswith('X')]


# Split into train and test

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(train_new[features], train_new.y, train_size=0.7)


# In[ ]:


clf = RandomForestRegressor(max_depth=3, n_estimators=100)
clf.fit(x_train,y_train)
print('Train score')
print(clf.score(x_train, y_train))
print('Test score')
print(clf.score(x_test, y_test))


# Apply to test

# In[ ]:


numeric_test = test.select_dtypes(include=[np.number])

string_var_test = test.select_dtypes(include=['object'])
dummies_test =pd.get_dummies(string_var_test)

test_new = numeric_test.join(dummies_test)


# Check whether all trainfeatures  exist in testset.

# In[ ]:


not_in_test = list(set(train_new.columns).difference(test_new.columns))
print(not_in_test)


# Not all dummies present in the trainset exist in the testset. Lets add them as zero's for now.

# In[ ]:


add = pd.DataFrame(0, index=np.arange(len(test_new)), columns=not_in_test)
test_new = test_new.join(add)


# In[ ]:


test_new['y'] = clf.predict(test_new[features])


# In[ ]:


test_new[['ID', 'y']].to_csv('nobrainer_baseline.csv', index=False)


# **Conclusion**
# 
# Now we have a baseline, let's move on!
# 
# Apparently not all dummies are useful, since some don't even occur in the testset! 
