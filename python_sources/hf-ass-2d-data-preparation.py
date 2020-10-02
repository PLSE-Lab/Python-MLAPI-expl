#!/usr/bin/env python
# coding: utf-8

# **Load the data**
# 
# Load the data using **Pandas DataFrame**
# 
# **Data.shape** allows you to print the columns and rows

# In[ ]:


import pandas

data = pandas.read_csv('../input/Road Accidents-Regression.csv')

print(data.shape)


# To print out an overview of the dataset

# In[ ]:


data.head(10)


# In[ ]:


data.info()


# In[ ]:


data.sample()


# Columns can be renamed using **data.rename**
# 
# Also multiple columns can be name at a time. Example:
# 
# data.rename( columns={ "Layout of Township": "Layout", "Number of cars": " cars" })

# In[ ]:


data = data.rename(columns={"Layout of Township": "Layout"})


# Converting the Data types that are in objects to category in order for one label encoding to be easily done.

# In[ ]:


data['Place'] = data['Place'].astype('category')
data['Nature of roads'] = data['Nature of roads'].astype('category')
data['Layout'] = data['Layout'].astype('category')


# To see the data types of the dataset in case any one needs a conversion, **data.dtypes** is used

# In[ ]:


data.dtypes


# Taking an overall view of how our dataset now looks like

# In[ ]:


data.head()


# using **one hot encoding** for good prediction

# In[ ]:


pandas.get_dummies(data, columns=['Place', 'Nature of roads', 'Layout']).head()


# USing **slicing** to put the dataset into input and putputs

# In[ ]:


data = data.values


X = data[:,0:5]
Y = data[:,5]


# From **scikit learn**, we use **train_test_split** to allocate values to our input and output. It is best to also randomise your data.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1
                                                    ,random_state = 0)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.11
                                                   ,random_state = 0)


# We now see the shapes of all the inputs and outputs; the **train set** , the **validation set** and the **test set**

# In[ ]:


X_train.shape


# In[ ]:


X_val.shape


# In[ ]:


X_test.shape


# In[ ]:


Y_train.shape


# In[ ]:


Y_val.shape


# In[ ]:


Y_test.shape

