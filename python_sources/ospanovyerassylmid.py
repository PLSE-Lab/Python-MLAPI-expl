#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt #visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# In[ ]:


data_train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
data_test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
data_sample = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')


# **Let's see general our train data, test data and sample
# **

# In[ ]:


data_train


# In[ ]:


data_train.describe()


# In[ ]:


data_test.columns


# In[ ]:


data_test.describe()


# In[ ]:


data_sample


# In[ ]:


data_sample.describe()


# *Let's see for example distribution of first one var_1 and last one var_195 in train data*

# In[ ]:


num_bins = 10
plt.hist(data_train['var_1'], num_bins, density=1, facecolor='blue', alpha=0.5)
plt.show()


# In[ ]:


num_bins = 10
plt.hist(data_train['var_199'], num_bins, density=1, facecolor='blue', alpha=0.5)
plt.show()


# *Let's see for example distribution of first one var_1 and last one var_195 in test data*

# In[ ]:


num_bins = 10
plt.hist(data_test['var_0'], num_bins, density=1, facecolor='blue', alpha=0.5)
plt.show()


# In[ ]:


num_bins = 10
plt.hist(data_test['var_199'], num_bins, density=1, facecolor='blue', alpha=0.5)
plt.show()


# **So we see general viewer of data just for fun**

# ** for x i take all columns from train data except target, for y took target column and for x test took whole x test file**

# In[ ]:


x = data_train.iloc[:, 2:].values
y = data_train.target.values
x_test = data_test.iloc[:, 1:].values


# In[ ]:


#creating new useful variables
x_train = x
y_train = y


# In[ ]:


#Let's see x train data after calculation
x_test


# In[ ]:


y


# In[ ]:


#our shape of x test and y test
x_test.shape


# In[ ]:


x_train.shape


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


gauss = GaussianNB()
y_prediction = gauss.fit(x_train, y_train).predict(x_test)


# In[ ]:


print(classification_report(y_train,y_prediction))


# In[ ]:



sub_df = pd.DataFrame({'ID_code':data_test.ID_code.values})
sub_df['target'] = y_prediction
sub_df.to_csv('submission.csv', index=False)

