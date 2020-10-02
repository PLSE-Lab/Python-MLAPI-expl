#!/usr/bin/env python
# coding: utf-8

# In this notebook, we will use scikit-learn to perform a decision tree based classification of loan data. This research aimed at the case of customers default payments in Taiwan and compares the predictive accuracy of probability of default among six data mining methods. 
# 
# 

#  Import all libaries which are necessary 

# In[2]:


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# Reda the excel file and  skip the frist row as we have tow headers 

# In[3]:


data = pd.read_excel(open('../input/default of credit card clients.xls','rb'), sheetname='Data', skiprows=1)


# Attribute Information:
# 
# This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables: 
# X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit. 
# X2: Gender (1 = male; 2 = female). 
# X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others). 
# X4: Marital status (1 = married; 2 = single; 3 = others). 
# X5: Age (year). 
# X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above. 
# X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005. 
# X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005. 
# 
# Citation Request:
# 
# Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.
# 
# 

# In[4]:


# check the data of first 5 rows using head function.
data.head()


# In[5]:


data.shape


# Check the columns and its anmes 

# In[7]:


data.columns


# Check if there is any null values in the data frame

# In[8]:


data.isnull().values.any()


# No Null values and no action needed to take.

# In[10]:


# check if there is any null data 
data[data.isnull().any(axis=1)] 


# In[11]:


# add a clumn to insert row numbers for entire dateframe for easy smalping of observations


# In[13]:


data ['a'] = pd.DataFrame({'a':range(30001)})
    


# In[14]:


# check if the colun a is added 


# In[15]:


data.columns


# select every 10th row of the data to train the model

# In[16]:


sampled_df = data[(data['a'] % 10) == 0]
sampled_df.shape


# select remaing data for testing to check the results

# In[17]:


sampled_df_remaining = data[(data['a'] % 10) != 0]
sampled_df_remaining.shape


# In[18]:


y = sampled_df['default payment next month'].copy()


# In[19]:


loan_features = ['LIMIT_BAL','SEX','EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']


# In[24]:


x = sampled_df[loan_features].copy()


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=324)


# In[28]:


loan_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
loan_classifier.fit(X_train, y_train)


# In[29]:


predictions = loan_classifier.predict(X_test)


# In[30]:


predictions[:20]


# In[31]:


accuracy_score(y_true = y_test, y_pred = predictions)


# In[32]:


X1 = sampled_df_remaining[loan_features].copy()


# In[33]:


y1 = sampled_df_remaining['default payment next month'].copy()


# In[34]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.33, random_state=324)


# In[35]:


loan_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
loan_classifier.fit(X1_train, y1_train)


# In[43]:


predictions1 = loan_classifier.predict(X1_test)


# In[44]:


predictions1[:20]


# In[45]:


accuracy_score(y_true = y1_test, y_pred = predictions1)


# In[ ]:




