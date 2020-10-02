#!/usr/bin/env python
# coding: utf-8

# **Ghouls, Goblins, and Ghosts... Boo:** 
# Random Forest Classifier has been used in this project to identify between Ghouls, Goblins and Ghosts based on the given feature. 

# In[ ]:


# Importing the Required Libaries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
plt.style.use('seaborn')


# In[ ]:


# Reading the Train and Test files
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/sample_submission.csv")
submission["type"] = "Unknown"


# ### Checking if there are any Null values or not. If there are any we need to handle them.

# In[ ]:


print("Train Data Null Values Count \n",train_df.isna().sum())
print("Test Data Null Values Count \n",test_df.isna().sum())


# Since there are no missing values we can continue our work further. If are any missing values I have explained in detail what steps we should take in my [blog](http://https://thedatafreak.wordpress.com/2018/08/16/foreplay-before-doing-data-part-i-handling-missing-data/).

# ### Taking a look at the distribution
# If the features have normal distrubtion we can move forward, else we have to use some transformation for making them normal. 

# ## Train Data Distribution

# In[ ]:


for col in train_df.drop(columns=['id','color','type'],axis=1).columns:
    sns.distplot(train_df[col])
    plt.show()


# ## Test Data Distrubtion

# In[ ]:


for col in test_df.drop(columns=['id','color'],axis=1).columns:
    sns.distplot(test_df[col])
    plt.show()


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# **Since the Normal Distribution Rule only applies to the Numeric data. For the Categorical data we need to take a look at the bar graph**

# In[ ]:


train_df.color.value_counts().plot(kind='bar')


# In[ ]:


test_df.color.value_counts().plot(kind='bar')


# In[ ]:


train_data = train_df.drop(columns=['id'],axis=1)


# In[ ]:


train_data.head()


# In[ ]:


test_data = test_df.drop(columns=['id'])


# In[ ]:


test_data.head()


# In[ ]:


train_data.type.value_counts()


# **We assign a 1 to Ghoul, 2 to Goblin and 3 to Ghost, since we need numbers for calculation in the algorithm. We cannot use string in the Machine Learning Algorithm so we need to encode them with numbers**

# In[ ]:


le = LabelEncoder()
train_data['type'] = le.fit_transform(train_data['type'])
print(train_data.type.value_counts())


# In[ ]:


train_data_x = train_data.drop(columns=['type'],axis=1)
train_data_y = train_data['type'].values


# We also have to covert the categorical variable into dummy variable. While creating dummy we need to remove one column to escape the dummy variable trap.

# In[ ]:


train_data_x = pd.get_dummies(train_data_x,columns=['color'],drop_first=True).values


# In[ ]:


y_data = pd.get_dummies(test_data, columns=['color'], drop_first=True).values


# In[ ]:


rfclf = RandomForestClassifier(n_estimators=1000)


# In[ ]:


rfclf.fit(train_data_x,train_data_y)


# In[ ]:


y_pred = rfclf.predict(y_data)
submission['type'] = y_pred


# In[ ]:


submission['type'] = submission.type.map({0:"Ghost", 1:"Ghoul", 2:"Goblin"})


# In[ ]:


submission.to_csv('../working/submission.csv', index=False)


# In[ ]:




