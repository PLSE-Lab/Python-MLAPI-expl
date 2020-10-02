#!/usr/bin/env python
# coding: utf-8

# **Load the data**
# 
# Load the data using **Pandas DataFrame**
# 
# **Data.shape** allows you to print the columns and rows

# In[38]:


import pandas

data = pandas.read_csv('../input/Internship Application  - Classification.csv')
print(data.shape)


# To print out an overview of the dataset

# In[39]:


data.head(10)


# **data.sample** picks the data set randomly and displays them

# In[40]:


data.sample(6)


# In[41]:


data.info()


# Columns can be renamed using **data.rename**
# 
# Also multiple columns can be name at a time. 
# Example:
# 
# data.rename(
#     columns={
#         "Experience Level": "Experience",
#         "Number of Applications": "Application Number"
#     })

# In[42]:


data = data.rename(columns={"Experience Level": "Experience"})


# To see the data types of the dataset in case any one needs a conversion, **data.dtypes is used**

# In[43]:


data.dtypes


# Converting the Data  types that are in objects to category in order for one label encoding to be easily done.

# In[44]:


data['Gender'] = data['Gender'].astype('category')
data['Experience'] = data['Experience'].astype('category')


# Checking to see the data types 

# In[45]:


data.dtypes


# **Label Encoding**: This allows you to change the data types of object to Numbers since Applied Machine Learning works best with numbers.
# 

# In[46]:


data['Gender'] = data['Gender'].cat.codes
data['Experience'] = data['Experience'].cat.codes


# Taking an overall view of how our dataset now looks like

# In[47]:


data.head()


# Using one **hot encoding** to separate values for good prediction

# In[48]:


pandas.get_dummies(data, columns=["Gender", "Experience"]).head()


# In[49]:


data = data.values


X = data[:,0:5]
Y = data[:,5]


# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1
                                                    ,random_state = 0)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.11
                                                   ,random_state = 0)


# We now see the shapes of all the inputs and outputs; the **train set** , the **validation set** and the **test set**

# In[51]:


X_train.shape


# In[52]:


X_val.shape


# In[53]:


X_test.shape


# In[54]:


Y_train.shape


# In[55]:


Y_val.shape


# In[56]:


Y_test.shape

