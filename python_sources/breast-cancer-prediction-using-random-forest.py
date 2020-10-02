#!/usr/bin/env python
# coding: utf-8

# # Predicting Breast Cancer Using Random Forest Classifier
# 
# Breast Cancer is one of leading cancers amoung women. Intial diagnosis inlcudes checking for lumps. But not all breast lumps are cancerous. So in this project, we are going to attempt to predict the chances that the lump is malignant or benign.
# 
# Check out [this tutorial](https://medium.com/@enfageorge/predicting-breast-cancer-using-random-forest-classifier-d193c72de8a3?source=friends_link&sk=ab8c3240995bf97e7733a8dc2b2f55bc) on how to.
# 
# 

# In[ ]:


import pandas as pd


# In[ ]:


data= pd.read_csv('../input/data.csv')


# In[ ]:


data


# # Data Preprocessing

# In[ ]:


# Missing values
data.isnull().sum()


# In[ ]:


data.isna().sum() # No missing values


# # Data Exploration

# In[ ]:


data.head()


# In[ ]:


data.shape


# # Data Preparation

# In[ ]:


X = data.iloc[:,2:32].values
Y = data.iloc[:,1].values


# In[ ]:


Y


# In[ ]:


#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder #encode categorical features using a one-hot or ordinal encoding scheme
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# In[ ]:


Y


# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[ ]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)


# # Prediction 

# In[ ]:


Y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


# In[ ]:


cm


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test,Y_pred)


# In[ ]:


accuracy

