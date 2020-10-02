#!/usr/bin/env python
# coding: utf-8

# ## Dealing with categorical data 

# In[ ]:


#First import necessary libraries
import pandas as pd
import numpy as np


# Getting data regarding asking gf what she wanted to eat
# Gathered from years of experience of asking gf what she wanted to eat

# In[ ]:


#import data into df
df =pd.read_csv("../input/Decisiontree.csv")


# In[ ]:


#Look at initial df
df.head()


# As you can see most of data is categorical.  We need to deal with this.

# In[ ]:


#Get info regarding df
print(df.info())


# As you can see most info is objects or categorical data

# In[ ]:


#Seperating dtypes into own df
print(df.select_dtypes(include=['object']).head())


# 4 possible decisions

# In[ ]:


print(df['Fastfood'].value_counts())  


# 3 decisions for fast food and has a lower cardinality then the column asking What do you want to eat?
# We need to encode these categorical data into something we can use for any modeling we wish to do.
# I decided to show some different encoding techniques to deal with this categorical data

# In[ ]:


##Check categories for unique values and counts to find out cardinality 
print(df['Do_you_want_to_eat?'].value_counts())


# Seeing as there are only 2 unique values I could encode this the old fashion way using pandas.

# In[ ]:


label = {'y':1, 'n':0}
df['Do_you_want_to_eat?'] = df['Do_you_want_to_eat?'].map(label)
df.head()


# Category encoder is a set of scikit-learn-style transformers for encoding categorical variables into numeric with different techniques.  Easy to use and works well with modeling.  Will be showing classic encoders as examples.
# Would install using  
# ```pip install category_encoders```  
# or  
# ```conda install -c conda-forge category_encoders```

# In[ ]:


#Importing category encoder library
import category_encoders as ce


# Next column I will be checking out the column, What_do_you_want_to_eat?, for unique values and cardinality

# In[ ]:


##Check categories for unique values and counts to find out cardinality 
print(df['What_do_you_want_to_eat?'].value_counts())


# As you can see there are 4 unique values I will use one of the classic encoders called Onehot to encode this.
# One hot will give each value a column and it will either be 1 or a 0 depending if it is true or false.

# In[ ]:


encoder = ce.OneHotEncoder(cols=['What_do_you_want_to_eat?'])
df= encoder.fit_transform(df)
df.head()


# It made 5 new one hot encoded columns for the 4 options and a 5th for Null value if no option was chosen

# In[ ]:


##Check categories for unique values and counts to find out cardinality 
print(df['Fastfood'].value_counts())


# Another classic encoder that is used is binary encoder.  Binary encoder converts each integer to binary digits.  Each binary digit gets one column.   Some loss but fewer dimensions.

# In[ ]:


encoder = ce.BinaryEncoder(cols=['Fastfood'])
df = encoder.fit_transform(df)

df.head()


# We will deal with the Restuarant column and find unique values

# In[ ]:


##Check categories for unique values and counts to find out cardinality 
print(df['Restaurant'].value_counts())


# Another classic encoder that is used is Ordinal encoder. Ordinal encorder will use a single column of integers to represent classes. Classes are assumed to have not true order and selected at random unless indicated.

# In[ ]:


encoder = ce.OrdinalEncoder(cols = ['Restaurant'])
# ce_leave.fit(X3, y3['outcome'])        
# ce_leave.transform(X3, y3['outcome']) 
df = encoder.fit_transform(df)

df.head()


# Assigned an integer to the 7 different unique vales into a single column

# In[ ]:


##Check categories for unique values and counts to find out cardinality 
print(df['Choice'].value_counts())


# ## Will continue progress with working with this and building a model to hopefully get better predictions of what a hungry gf wants to eat but current findings is the final choice seems to be favoring IDLT with this small sample set.
# 
# IDLT means after all those decisions the result was I Don't like that.
# 
