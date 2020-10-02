#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))

import warnings 
warnings.filterwarnings('ignore')


# In[ ]:


train=pd.read_csv('../input/advertising.csv')
train.head()


# In[ ]:


train.shape


# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


#lets categorise the data

num_cols=[var for var in train.columns if train[var].dtypes !='O']
cat_cols=[var for var in train.columns if train[var].dtypes !='int64' and train[var].dtypes !='float64']

print('No of numerical columns: ',len(num_cols))
print('No of categorical columns: ',len(cat_cols))
print('Total no of columns: ',len(num_cols+cat_cols))


# In[ ]:


sns.countplot(train['Clicked on Ad'])


# In[ ]:


#Missing Data

sns.heatmap(train.isnull())


# In[ ]:


train.isnull().sum()


# In[ ]:


sns.set_style('whitegrid')
plt.hist(train['Age'],color='darkred',bins=30,)


# In[ ]:


plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.title('Avg time on website ')
sns.distplot(train['Area Income'])
plt.subplot(1,3,2)
sns.distplot(train['Age'])


# In[ ]:


sns.lineplot(train['Age'],train['Area Income'])


# In[ ]:


sns.jointplot(x='Age',y='Daily Time Spent on Site',data=train,color='red',kind='kde')


# ## Model Selection

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = train[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = train['Clicked on Ad']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# ## Predictions and Evaluation

# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,predictions))


# In[ ]:


final_df = pd.DataFrame({
        "Clicked on Ad": predictions
    })


# In[ ]:


# Save the dataframe to a csv file
final_df.to_csv('submission.csv',index=False)

