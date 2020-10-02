#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
df = pd.read_csv('../input/pima-diabetes/pimaindians-diabetes.data.csv', header = None)


# In[ ]:


df.describe()


# 1: Plasma glucose concentration
# 
# 2: Diastolic blood pressure
# 
# 3: Triceps skinfold thickness
# 
# 4: 2-Hour serum insulin
# 
# 5: Body mass index

# In[ ]:


df.head(3)


# In[ ]:


print((df[[1,2, 3,4,5,6,7,8]] ==0).sum())


# In[ ]:


#marking 0 as NAN


# In[ ]:


df[[1,2,3,4,5]] = df[[1,2,3,4,5]].replace(0, np.NaN)
print(df.isnull().sum())


# ## Remove Rows With Missing Values

# In[ ]:


df.dropna(inplace=True)
print(df.shape)


# In[ ]:


# split dataset into inputs and outputs
values = df.values
X = values[:,0:8]
y = values[:,8]
# evaluate an LDA model on the dataset using k-fold cross validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold,cross_val_score
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=3, random_state=7)
result = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(result.mean())


# In[ ]:


type(values)


# ## Imputing Missing Values

# In[ ]:


df.fillna(df.mean(), inplace=True)

# count the number of NaN values in each column

print(df.isnull().sum())


# In[ ]:


values = df.values


# In[ ]:


X = values[:,0:8]
model = LinearDiscriminantAnalysis()

kfold = KFold(n_splits=3, random_state=7)

result = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(result.mean())


# The scikit-learn library provides the Imputer() pre-processing class that can be used to replace missing values.
# 
# It is a flexible class that allows you to specify the value to replace (it can be something other than NaN) and the technique used to replace it (such as mean, median, or mode). The Imputer class operates directly on the NumPy array instead of the DataFrame.
# 
# The example below uses the Imputer class to replace missing values with the mean of each column then prints the number of NaN values in the transformed matrix.

# In[ ]:


from sklearn.preprocessing import Imputer

# fill missing values with mean column values

values = df.values

imputer = Imputer()

transformed_values = imputer.fit_transform(values)

# count the number of NaN values in each column

print(np.isnan(transformed_values).sum())


# In[ ]:


# evaluate an LDA model on the dataset using k-fold cross validation

model = LinearDiscriminantAnalysis()

kfold = KFold(n_splits=3, random_state=7)

result = cross_val_score(model, transformed_values, y, cv=kfold, scoring='accuracy')

print(result.mean())


# ## Algo's that Support Missing Values

# There are algorithms that can be made robust to missing data, such as k-Nearest Neighbors that can ignore a column from a distance measure when a value is missing.
# 
# There are also algorithms that can use the missing value as a unique and different value when building the predictive model, such as classification and regression trees.

# In[ ]:




