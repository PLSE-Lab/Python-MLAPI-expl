#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[4]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


train.describe()


# Combine train and test datasets for further work with features

# In[8]:


dataset = pd.concat((train, test))


# Check for null and missing values

# In[9]:


dataset = dataset.fillna(np.nan)
dataset.isnull().sum()


# Age and Cabin features have an important part of missing values. Missing values in Survived correspond to the join testing dataset(Survived column doesn't exist in test set ). 

# In[10]:


dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())


# In[11]:


dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].mean())


# In[12]:


dataset["Embarked"] = dataset["Embarked"].fillna("S")


# ## Feature engineering
# 

# Name

# The Name feature contains information on passenger's title.

# In[13]:


dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
dataset["Title"].value_counts()


# There is 17 titles in the dataset, most of them are very rare and we can group them in 4 categories

# In[14]:


dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)


# Women are more likely to survive than men

#  Family size

#  Family size feature which is the sum of SibSp , Parch and 1 (including the passenger). I decided to created 4 categories of family size.

# In[15]:


dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1
dataset['Single'] = dataset['FamilySize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
dataset['MedF'] = dataset['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['FamilySize'].map(lambda s: 1 if s >= 5 else 0)


# Cabin

# In[16]:


dataset['Cabin'].describe()


# The Cabin feature column contains 295 values and 10014 missing values. Replace the Cabin number by the type of cabin 'X' if not

# In[17]:


dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])


# Ticket

# I decided to replace the Ticket feature column by the ticket prefixe

# In[18]:


Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0])
    else:
        Ticket.append("X")
        
dataset["Ticket"] = Ticket
dataset["Ticket"].head()


# ### Encoding categorical variables

# In[19]:


dataset = pd.get_dummies(dataset, columns = ["Sex", "Title"])
dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")
dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")


# In[20]:


dataset.drop(labels = ["Name"], axis = 1, inplace = True)
dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)


# In[21]:


dataset.shape


# ## Splitting the training data

# In[22]:


X_train = dataset[:train.shape[0]]
X_test = dataset[train.shape[0]:]
y = train['Survived']


# In[23]:


X_train = X_train.drop(labels='Survived', axis=1)
X_test = X_test.drop(labels='Survived', axis=1)


# ## Feature Scaling

# In[24]:


from sklearn.preprocessing import StandardScaler


# In[25]:


headers_train = X_train.columns
headers_test = X_test.columns


# In[26]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Modeling

# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.model_selection import GridSearchCV


# In[28]:


cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25, random_state = 0 )
accuracies = cross_val_score(LogisticRegression(solver='liblinear'), X_train, y, cv  = cv)


# ### SVM

# In[ ]:


from sklearn.svm import SVC

C = [0.1, 1]
gammas = [0.01, 0.1]
kernels = ['rbf', 'poly']
param_grid = {'C': C, 'gamma' : gammas, 'kernel' : kernels}

cv = StratifiedShuffleSplit(n_splits=5, test_size=.25, random_state=8)

grid = GridSearchCV(SVC(probability=True), param_grid, cv=cv)
grid.fit(X_train,y)


# In[ ]:


svm_grid= grid.best_estimator_
svm_score = round(svm_grid.score(X_train,y), 4)
print('Accuracy for SVM: ', svm_score)


# ### KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


k_range = range(1,31)
weights_options=['uniform','distance']
param = {'n_neighbors':k_range, 'weights':weights_options}
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
grid = GridSearchCV(KNeighborsClassifier(), param,cv=cv,verbose = False, n_jobs=-1)

grid.fit(X_train,y)


# In[ ]:


knn_grid= grid.best_estimator_
knn_score = round(knn_grid.score(X_train,y), 4)
knn_score
print('Accuracy for KNN: ', knn_score)


# ### Logistic Regression

# In[ ]:


C_vals = [0.2,0.3,0.4,0.5,1,5,10]

penalties = ['l1','l2']

cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25)


param = {'penalty': penalties, 'C': C_vals}

logreg = LogisticRegression(solver='liblinear')
 
grid = GridSearchCV(estimator=LogisticRegression(), 
                           param_grid = param,
                           scoring = 'accuracy',
                            n_jobs =-1,
                           cv = cv
                          )

grid.fit(X_train, y)


# In[ ]:


logreg_grid = grid.best_estimator_
logreg_score = round(logreg_grid.score(X_train,y), 4)
print('Accuracy for Logistic Regression: ', logreg_score)


# I decided to use KNN model for prediction

# ## Prediction

# In[ ]:


predict = knn_grid.predict(X_test)
submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predict})
submission.to_csv('submission_knn.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




