#!/usr/bin/env python
# coding: utf-8

# # Iris 

# ## Data Set Information:
# 
# This is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.
# 
# Predicted attribute: class of iris plant.
# 
# This is an exceedingly simple domain.

# ## Attribute Information:
# 
# 1. sepal length in cm
# 2. sepal width in cm
# 3. petal length in cm
# 4. petal width in cm
# 5. class:
# -- Iris Setosa
# -- Iris Versicolour
# -- Iris Virginica
# 
# 

# ## Import Libraries

# In[ ]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ## The Dataset

# In[ ]:


# Checking files in directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Column Headers
headers=['sepal length','sepal width','petal length','petal width','class']

# Read the data
df=pd.read_csv('/kaggle/input/iris-dataset/iris.data',names=headers)
df.head()


# In[ ]:


df.columns


# In[ ]:


df.describe()


# In[ ]:


# Label encoding the target variable

encode=LabelEncoder()
df['class']=encode.fit_transform(df['class'])
df.head()


# ## Preparing the Model

# In[ ]:


# Selecting target and features

X=df.drop(['class'],axis=1)
y=df['class']


# In[ ]:


# Train Test split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[ ]:


# Making the model

lr=LogisticRegression()
lr.fit(X_train,y_train)
yhat=lr.predict(X_test)


# In[ ]:


# Evaluating Model

print('Predicted Values on Test Data',encode.inverse_transform(yhat))

print("Accuracy Score : ",accuracy_score(yhat,y_test))


# ## Conclusion
# 
# Thus, we used logistic regression model to successfully evaluate the given Iris Data set and received a 100% accuracy score.

# In[ ]:




