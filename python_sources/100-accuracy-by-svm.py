#!/usr/bin/env python
# coding: utf-8

# # SVM based mushroom classification

# ### Importing libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading data from csv

# In[ ]:


df=pd.read_csv('../input/mushrooms.csv')


# ### Exploratory Data Analysis:

# In[ ]:


df.head()


# ### Function to map categorical features to integral values

# In[ ]:


def Encoder(val):
    if val in category:
        return category[val]
    else:
        category[val]=len(category)
    return category[val]


# In[ ]:


df.info()


# ### no missing values

# In[ ]:


df.shape


# #### Encoder at work

# In[ ]:


for i in range(df.shape[1]):
    category={}
    df.iloc[:,i]=df.iloc[:,i].apply(Encoder)


# In[ ]:


df.head()


# In[ ]:


sns.countplot(x='class',data=df)


# roughly equivalent distribution

# In[ ]:


correlation=df.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation,annot=True,cmap='coolwarm')


# #### why is veil-type showing such unusual behavior?

# In[ ]:


df['veil-type'].value_counts()


# #### it's 0 for the entire dataset and is thus useless for analysis
# #### Let's check for other such const cases by checking mean of each column,for const columns mean will be in whole number

# In[ ]:


df.describe()


# In[ ]:


X=df.drop(['class','veil-type'],axis=1)


# In[ ]:


y=df['class']


# In[ ]:


X.head()


# ### Training the model:

# In[ ]:


(X_train,X_test,Y_train,Y_test)=train_test_split(X,y,test_size=0.30)


# #### Using grid search with 10-fold cross validation to find the best set of parameters:

# In[ ]:


svc=SVC()
param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5],'C': [1, 10, 100]},
              {'kernel': ['linear'], 'C': [1, 10, 100]}]

grid=GridSearchCV(svc,param_grid,cv=10,scoring='accuracy')
print("Tuning hyper-parameters")
grid.fit(X_train,Y_train)
print(grid.best_params_)
print(np.round(grid.best_score_,3))


# #### Making classifier using above parameters

# In[ ]:


svc=SVC(C=100,gamma=0.001,kernel='rbf')


# In[ ]:


svc.fit(X_train,Y_train)


# In[ ]:


svc.score(X_test,Y_test)


# ## 100% accuracy achieved

# ### Confusion Matrix:

# In[ ]:


Ypreds=svc.predict(X_test)
cm = confusion_matrix(Y_test,Ypreds)
xy=np.array([0,1])
plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True,square=True,cmap='coolwarm',xticklabels=xy,yticklabels=xy)


# In[ ]:




