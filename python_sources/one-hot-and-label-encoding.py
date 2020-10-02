#!/usr/bin/env python
# coding: utf-8

# # Convert Categorical Data to Numerical data 
# The goal of this notebook is to help you understand two different methods of converting categorical data to numerical data.  
# First of all, we are going to understand how those technic work by implementing manually and then apply the existing functions 
# that can get the job done easily.  

# In[ ]:


import pandas as pd  
import matplotlib.pyplot as plt
from sklearn import preprocessing 


# In[ ]:


# Read the dataset 
tent_sales = pd.read_csv('../input/sales_data.csv')  
tent_sales.head()


# We can notice that our dataset has many categorical features: Gender, Marital Status Profession. Using this dataset involves predicting if the customer purchased a tent or not, given by the **IS_TENT** column.

# In[ ]:


tent_sales.shape


# In[ ]:


tent_sales.describe()


# The average age of customers is around 34 years and the standard deviation is around 10.  

# #### Some Visualization   
# We are going to perform some visualizations before we move forward for processing.

# In[ ]:


plt.figure(figsize=(10, 7))
pd.value_counts(tent_sales['IS_TENT']).plot.bar()
plt.show()


# We can notice that over 50.000 customers did not purchase a tent. 

# In[ ]:


plt.figure(figsize=(10, 7))
pd.value_counts(tent_sales['MARITAL_STATUS']).plot.bar()
plt.show()


# Most of the customers are married, many are single, and few have not specified their marrital status. 

# In[ ]:


plt.figure(figsize=(10, 7))
pd.value_counts(tent_sales['GENDER']).plot.bar()
plt.show()


# Our customers are distributed between males and females. Over 30000 of them are males and the others are female. 

# In[ ]:


plt.figure(figsize=(10, 7))
pd.value_counts(tent_sales['PROFESSION']).plot.bar()
plt.show()


# Most of our customers have characterised their profession as others. some are in Sales, some are Executives and few are retired.  

# Categorical features are not understood by machine learning models, we need to encode them into numeric form. So, we will encode all the categorical columns. The encoding technics will be performed using both **label and one hot** encoders.

# In[ ]:


# Gender Column 
gender = ['M','F'] 

label_encoding = preprocessing.LabelEncoder()

'''
This will generate a unique ID for Male and 
a unique ID for Female. 
'''
label_encoding = label_encoding.fit(gender)


# label encoder generate a numeric identifier starting with zero. 

# In[ ]:


tent_sales['GENDER'] = label_encoding.transform(tent_sales['GENDER'].astype(str))


# In[ ]:


# Shows the categories that have been encoded.
label_encoding.classes_


# * **F** will be represented using the numeric integer **0** 
# * **M** will be represented using the integer **1**.

# In[ ]:


tent_sales.sample(10)


# We can now see that the Gender column have been transformed. 
# When we are using more than 2 categories in our features, then after label encoding the sklearn estimators could understand that there are some specific orders. This is not a problem when we are using binary categories. But we must be careful while using more than 2 categories. 

# In[ ]:


# Marital Status Column 
tent_sales['MARITAL_STATUS'].unique()


# We see that the marital status has three categories. We also know that there is not a specific order in our categories (Married > Single > Unspecified). So, using a label encoder might not be a good solution. 
# The solution here is to use **One hot Encoding** 

# In[ ]:


one_hot_encoding = preprocessing.OneHotEncoder()
one_hot_encoding = one_hot_encoding.fit(tent_sales['MARITAL_STATUS'].values.reshape(-1, 1))


# In[ ]:


one_hot_encoding.categories_


# The encoded columns are Married, Single and Unspecified. 
# We need to get **MARITAL STATUS** in one hot encoding form by calling transform on the same column. The one hot encoder expects the input feature to be in two dimensions, so we need to reshape the values of that column to be two dimensional. 

# In[ ]:


one_hot_labels = one_hot_encoding.transform(
                tent_sales['MARITAL_STATUS'].values.reshape(-1,1)).toarray()
one_hot_labels


# Here, we can see that each category is represented using one hot encoding. 
# * The first column represents the Married status  
# * The second column represents the Single status  
# * The third column represents the Unspecified status  
# 
# The one hot encoded values are in the form of a numpy array, now we are going to assign these as columns in our dataframe.  

# In[ ]:


labels_df = pd.DataFrame()

labels_df['MARITAL_STATUS_Married'] = one_hot_labels[:,0]
labels_df['MARITAL_STATUS_Single'] = one_hot_labels[:,1]
labels_df['MARITAL_STATUS_Unspecified'] = one_hot_labels[:,2]

labels_df.head(10)


# Now, we can drop the origial categorical column MARITAL_STATUS. 

# In[ ]:


encoded_df = pd.concat([tent_sales, labels_df], axis=1)
encoded_df.drop('MARITAL_STATUS', axis=1, inplace=True)


# In[ ]:


encoded_df.sample(10)


# Our dataframe has MARITAL_STATUS and GENDER columns in the right encoded format. 

# In[ ]:


# Marital Status Column 
tent_sales['PROFESSION'].unique()


# We manually used this technic in order to understand the process. But the easiest way to one hot encode our categorical features is to use **pd.get_dummies** function by giving the corresponding column. 

# In[ ]:


tent_sales = pd.get_dummies(encoded_df, columns=['PROFESSION'])
tent_sales.sample(10)


# We can now see that the **PROFESSION** column has been automatically one hot encoded while removing the original column. 

# ### Let's restart using pd.getdummies()   
# We are going to re-read the dataset in order to apply the pd.getdummies() function. 

# In[ ]:


# Read the dataset 
tent_sales = pd.read_csv('../input/sales_data.csv')  
tent_sales.head()


# In[ ]:


'''
Encode all of the categorical features while removing 
all the original categorical features.
'''
tent_sales = pd.get_dummies(tent_sales)
tent_sales.sample(10)


# In[ ]:




