#!/usr/bin/env python
# coding: utf-8

# This analysis report contains various steps involved in the classification analysis. We look at various classification algorithm.

# In[ ]:


# Importing the Required Libraries for data analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# **Step 1 - Exploratory Data Analysis** 

# In[ ]:


# Reading the dataset as a dataframe
file_name = '../input/data.csv'
data_df = pd.read_csv(file_name)


# In[ ]:


# A sneak peek into the dataframe
data_df.head()


# In[ ]:


data_df.describe()


# **Handling the Missing Data**
# 1. Find the count of Null Values in Each Column
# 2. Remove the column if all the values of a column is Null or missin
# 3. If only some of the values are missing replace them with mean or median

# In[ ]:


# Finding the count of Null Values in each column
pd.DataFrame(data_df.isna().sum())


# If we look at the dataframe above we can see that the all the column except the last column does not have any missing value. The last column `Unnamed: 32` has all the missing value it have to be removed.

# In[ ]:


# Dropping the Column that have all Nan/Null values only
data_df=data_df.set_index('id')
data_df.drop(columns=['Unnamed: 32'],axis=1,inplace=True)


# Graphs are very helpful tools for data analysis. It gives us a clear picture of how our data is distributed. Looking at the graph we can decide what are the tranformation steps required and what algorithm can be used

# In[ ]:


print(data_df.shape)
data_df.head()


# In[ ]:


sns.countplot(data_df.diagnosis,label='count')
B, M = data_df.diagnosis.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)
print('Percentage of Benign: ',B/(B+M)*100)
print('Number of Malignant : ',M/(B+M)*100)


# In[ ]:


encoder = LabelEncoder()
data_df.diagnosis = encoder.fit_transform(data_df.diagnosis)


# In[ ]:


data_df.head()


# In[ ]:


# Using Seaborn pair plot to take a look at the data graph
sns.pairplot(data_df,dropna=True)
plt


# **Step - 2:  Using Random Forest as the Classifier**

# In[ ]:


X = data_df.drop(columns=['diagnosis'],axis=1).values


# In[ ]:


y = data_df['diagnosis']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,
                                                 random_state=0)


# In[ ]:


classifier = RandomForestClassifier(n_estimators=100)


# In[ ]:


classifier.fit(X_train,y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:




