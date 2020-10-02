#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/mushroom-classification/mushrooms.csv')


# In[ ]:


df.head()


# # Let's see wheather we have any Null values present.

# In[ ]:


df.isnull().sum()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
Encoder=LabelEncoder()


# # Now Let's use LabelEncoder, to convert our categorical Values, in Numeric Labels, using a Simple "For" Loop!

# In[ ]:


for col in df.columns:
    df[col]=Encoder.fit_transform(df[col])


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# # Let's see how many values, does each class has!

# In[ ]:


plt.hist(df['class'])


# In[ ]:


X=df.iloc[:,1:23]
y=df[['class']]


# In[ ]:


X.head()


# In[ ]:


y.head()


# # Splitting the data into, Train and Test!

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # Now, Using StandardScaler, we'll scale our data, which will help us gain much better Accuracy!

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


X_train[0]


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier()
RF.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix
pred=RF.predict(X_test)

print(accuracy_score(pred,y_test))
print(confusion_matrix(pred,y_test))


# In[ ]:


pred


# In[ ]:




