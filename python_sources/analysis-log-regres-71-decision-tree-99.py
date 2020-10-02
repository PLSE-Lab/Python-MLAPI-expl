#!/usr/bin/env python
# coding: utf-8

# ## Regression (Logistic Regression-71%, Decision Tree-99%)
# In this Kernel, let's perform Logistic Regression using the columns (first 50 columns which shows the answers of the questions column) and predict the country of the candidate.

# In[ ]:


import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data_set = pd.read_csv('../input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv', sep='\t')


# Removing the null values from the dataset.

# In[ ]:


data_set = data_set.dropna()


# Selecting the first 50 columns and the Country column from the dataset to perform the Logistic Regression and transforming the columns into Category Type.

# In[ ]:


answer_data = data_set.iloc[:,0:50]


# In[ ]:


answer_data['country'] = data_set['country']


# In[ ]:


for col in answer_data.columns:
    answer_data[col] = answer_data[col].astype('category').cat.codes


# Calculating the correlation of the columns with the target column ('country') and choosing the top 10 correlated columns and the bottom 5 correlated columns.

# In[ ]:


corr_data = pd.DataFrame(answer_data.corr()['country'][:])


# In[ ]:


corr_data = corr_data.reset_index()


# In[ ]:


top_correlation = corr_data.sort_values('country', ascending=False).head(10)['index'].to_list()


# In[ ]:


least_correlation = corr_data.sort_values('country', ascending=False).tail(5)['index'].to_list()


# In[ ]:


correlation_data = answer_data[top_correlation+least_correlation]


# In[ ]:


target_data = answer_data['country']


# Let's split the dataset into train and test data.

# In[ ]:


var_train, var_test, res_train, res_test = train_test_split(correlation_data, target_data, test_size = 0.3)


# ### Logistic Regression
# Initializing the Logistic Regression

# In[ ]:


logistic_reg = LogisticRegression(random_state=0).fit(var_train, res_train)


# In[ ]:


prediction = logistic_reg.predict(var_test)


# The accuracy score of the regression

# In[ ]:


accuracy_score(res_test, prediction)


# ### Decision Tree Classifier

# In[ ]:


decision_tree = tree.DecisionTreeClassifier()
decision_tree = decision_tree.fit(var_train, res_train)


# In[ ]:


decision_prediction = decision_tree.predict(var_test)


# In[ ]:


accuracy_score(res_test, decision_prediction)

