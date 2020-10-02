#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[ ]:


df=pd.read_csv('../input/winequality-red.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nwine_data=df.columns.drop('quality')\nfor wd in wine_data:\n    fig=plt.figure(figsize=(10,8))\n    sns.barplot(x=df['quality'], y=df[wd])")


# In[ ]:


bins = [2,4,6,9]
labels= ['bad','medium','good']
df['quality']=pd.cut(df['quality'],bins=bins, labels=labels)
df.head()


# In[ ]:


le = LabelEncoder()
df['quality'] = le.fit_transform(df.quality)
sns.countplot(df['quality'])


# In[ ]:


#Split data into training and testing datasets

X=df.drop(columns='quality')
Y=df['quality']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30, shuffle=True)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[ ]:


#Standardise data for better results

scalar=StandardScaler()
X_train=scalar.fit_transform(X_train)
X_test=scalar.fit_transform(X_test)


# ## Classifier: Logistic regression

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nregressor=LogisticRegression(solver='saga', multi_class='multinomial',max_iter=1000)\nregressor.fit(X_train,Y_train)\n\nscore=regressor.score(X_test,Y_test)\nprint('accuracy = '+str(score))")


# In[ ]:




