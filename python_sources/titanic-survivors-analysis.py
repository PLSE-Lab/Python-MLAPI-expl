#!/usr/bin/env python
# coding: utf-8

# ## <a id="intro">Introduction</a>

# This is my first kernel on Kaggle. My objective here is to perform a detailed analysis of the dataset in order to understand which are the features that might have influenced if a passenger survived or not to the sinking of the RMS Titanic in 1912. 
# From that understanding of the dataset, I want to use several Maching Learning models in order to predict if a passenger might survive or not to the sinking and score the different models in order to choose the best solution.
# 
# It is very exciting for me to share my first kernel with the community and I wish to have your opinion about the kernel as you **leave a comment** !

# ## Table of contents

# <hr>
# <ol id="1">
#   <li><a>[Introduction](#intro)</li>
#   <li>[Libraries](#lib)</li>
#   <li>[Loading the data](#load)</li>
#   <li>[Data Preparation](#prep)</li>
#   <li>[Exploratory Data Analysis](#section1)</li>
#   <li>[Machine Learning models](#model)
#       <ol id="2">
#           <li>[Logistic Regression](#lr)</li>
#           <li>[Random Forest](#rf)</li>
#           <li>[Decision Tree](#lr)</li>
#           <li>[Neural Networks](#nn)</li>
#           <li>[Bayesian Networks](#bn)</li>      
#       </ol>
#   </li>
#   <li>[Conclusion](#concl)</li>
# </ol>
# <hr>
# 
# 

# ## <a id="lib">Libraries</a>

# In[24]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


import os
print(os.listdir("../input"))
get_ipython().run_line_magic('matplotlib', 'inline')


# ## <a id="load">Loading the data</a>

# In[25]:


train_df = pd.read_csv("/kaggle/input/train.csv")
test_df = pd.read_csv("/kaggle/input/test.csv")
survivor_df = pd.read_csv("/kaggle/input/gender_submission.csv")


# ## <a id="prep">Data Preparation</a>

# <hr>

# In[26]:


train_df.count()


# In[27]:


test_df.count()


# **Age**, **Cabin** and **Embarked** values are missing in the training dataset<br>
# **Age**, **Fare** and **Cabin** values are missing in the testing dataset
# 
# Let's fill these missing values.

# In[28]:


train_df["Sex"].replace(['male', 'female'], [0,1], inplace=True)
train_df["Embarked"].replace(['C', 'Q', 'S'], [0,1,2], inplace=True)
test_df["Sex"].replace(['male', 'female'], [0,1], inplace=True)
test_df["Embarked"].replace(['C', 'Q', 'S'], [0,1,2], inplace=True)


# In[29]:


train_df["Age"].replace(np.nan, train_df["Age"].median(), inplace=True)
test_df["Age"].replace(np.nan, test_df["Age"].median(), inplace=True)


# In[30]:


train_df["Embarked"].replace(np.nan, train_df["Embarked"].median(), inplace=True)


# In[31]:


train_df['Age'] = train_df['Age'].mask((train_df['Age'] >= 0) & (train_df['Age'] < 4), 0)
train_df['Age'] = train_df['Age'].mask((train_df['Age'] >= 4) & (train_df['Age'] < 18), 1)
train_df['Age'] = train_df['Age'].mask((train_df['Age'] >= 18) & (train_df['Age'] < 20), 2)
train_df['Age'] = train_df['Age'].mask((train_df['Age'] >= 20) & (train_df['Age'] < 29), 3)
train_df['Age'] = train_df['Age'].mask((train_df['Age'] >= 29) & (train_df['Age'] < 39), 4)
train_df['Age'] = train_df['Age'].mask((train_df['Age'] >= 39) & (train_df['Age'] < 49), 5)
train_df['Age'] = train_df['Age'].mask((train_df['Age'] >= 49) & (train_df['Age'] < 59), 6)
train_df['Age'] = train_df['Age'].mask((train_df['Age'] >= 59) & (train_df['Age'] < 69), 7)
train_df['Age'] = train_df['Age'].mask((train_df['Age'] >= 69) & (train_df['Age'] <= 80), 8)


# In[32]:


test_df['Age'] = test_df['Age'].mask((test_df['Age'] >= 0) & (test_df['Age'] < 4), 0)
test_df['Age'] = test_df['Age'].mask((test_df['Age'] >= 4) & (test_df['Age'] < 18), 1)
test_df['Age'] = test_df['Age'].mask((test_df['Age'] >= 18) & (test_df['Age'] < 20), 2)
test_df['Age'] = test_df['Age'].mask((test_df['Age'] >= 20) & (test_df['Age'] < 29), 3)
test_df['Age'] = test_df['Age'].mask((test_df['Age'] >= 29) & (test_df['Age'] < 39), 4)
test_df['Age'] = test_df['Age'].mask((test_df['Age'] >= 39) & (test_df['Age'] < 49), 5)
test_df['Age'] = test_df['Age'].mask((test_df['Age'] >= 49) & (test_df['Age'] < 59), 6)
test_df['Age'] = test_df['Age'].mask((test_df['Age'] >= 59) & (test_df['Age'] < 69), 7)
test_df['Age'] = test_df['Age'].mask((test_df['Age'] >= 69) & (test_df['Age'] <= 80), 8)


# > ## <a id="section1">Exploratory Data Analysis</a>

# In[33]:


train_df.info()


# In[34]:


train_df.describe()


# In[35]:


test_df.info()


# In[36]:


test_df.describe()


# <hr>

# * There are 891 passengers in **train.csv**
# * There are 418 passengers in **test.csv**

# <hr>

# In[37]:


fig, axes = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()
train_df.groupby("Sex")["PassengerId"].count().plot.pie(labels=["male", "female"], ax=axes[0], title="Repartition female/male", figsize=(15,20))
train_df[train_df.Survived == 1].groupby("Sex")["PassengerId"].count().plot.pie(labels=["male", "female"], ax=axes[1], title="Repartition female/male among survivors")


# In[38]:


fig, axes = plt.subplots(nrows=1, ncols=2)
fig.subplots_adjust(wspace=1)
train_df.groupby("Pclass")["PassengerId"].count().plot.bar(ax=axes[0], title="Repartition Pclass", figsize=(20,5))
train_df[train_df.Survived == 1].groupby("Pclass")["PassengerId"].count().plot.bar(ax=axes[1], title="Repartition Pclass among survivors")


# In[39]:


fig, axes = plt.subplots(nrows=1, ncols=2)
fig.subplots_adjust(wspace=0.5)
train_df.groupby("Age")["PassengerId"].count().plot.bar(ax=axes[0], title="Repartition Age", figsize=(40,10))
train_df[train_df.Survived == 1].groupby("Age")["PassengerId"].count().plot.bar(x=axes[1], title="Repartition Age among survivors")


# In[40]:


fig, axes = plt.subplots(nrows=1, ncols=2)
fig.subplots_adjust(wspace=1)
train_df.groupby("SibSp")["PassengerId"].count().plot.bar(ax=axes[0], title="Repartition SibSp", figsize=(20,5))
train_df[train_df.Survived == 1].groupby("SibSp")["PassengerId"].count().plot.bar(ax=axes[1], title="Repartition SibSp among survivors")


# In[41]:


fig, axes = plt.subplots(nrows=1, ncols=2)
fig.subplots_adjust(wspace=1)
train_df.groupby("Parch")["PassengerId"].count().plot.bar(ax=axes[0], title="Repartition Parch", figsize=(20,5))
train_df[train_df.Survived == 1].groupby("Parch")["PassengerId"].count().plot.bar(ax=axes[1], title="Repartition Parch among survivors")


# In[42]:


fig, axes = plt.subplots(nrows=1, ncols=2)
fig.subplots_adjust(wspace=1)
train_df.groupby("Embarked")["PassengerId"].count().plot.bar(ax=axes[0], title="Repartition Embarked", figsize=(20,5))
train_df[train_df.Survived == 1].groupby("Embarked")["PassengerId"].count().plot.bar(ax=axes[1], title="Repartition Embarked among survivors")


# <hr>

# **Features** that seem to have an influence on the survival chance :<br>
# <br>
# **Sex** - Females are more likely to survive than men <br>
# **Pclass** - 1st class passenger have more chance to survive<br>
# **Age**<br>
# **SibSp** - Passenger without siblings or spouse are more likely to survive <br>
# **Parch** - Passenger without children or parents are more likely to survive <br>
# **Embarked** - Passenger that embarked on S are more likely to survive <br>

# In[43]:


train_df["NbRelatives"] = train_df["Parch"] + train_df["SibSp"]
test_df["NbRelatives"] = test_df["Parch"] + test_df["SibSp"]


# In[44]:


train_df.drop(columns=["Cabin", "Ticket", "Parch", "SibSp"], inplace=True)
test_df.drop(columns=["Cabin", "Ticket", "Parch", "SibSp"], inplace=True)


# <hr>

# ## <a id="model">Machine Learning models</a>

# In[22]:


score_list = []


# ### <a id="lr">Logistic Regression</a>

# In[23]:


X_train = train_df[["NbRelatives", "Sex", "Pclass", "Embarked", "Age"]]
y_train = train_df["Survived"]

X_test = test_df[["NbRelatives", "Sex", "Pclass", "Embarked", "Age"]]


# In[ ]:


regressor = LogisticRegression()
regressor.fit(X_train, y_train)


# In[ ]:


score_list.append(regressor.score(X_train, y_train))


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


corr_df = pd.DataFrame({'features':X_train.columns})
corr_df["Corr"] = pd.Series(regressor.coef_[0])
corr_df


# ### <a id="rf">Random Forest</a>

# In[ ]:


clf = RandomForestClassifier(n_estimators=128, max_depth=2, random_state=0)
clf.fit(X_train, y_train)


# In[ ]:


score_list.append(clf.score(X_train, y_train))


# ### <a id="dt">Decision Tree</a>

# In[45]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
score_list.append(clf.score(X_train, y_train))


# In[47]:


prediction = clf.predict(X_test)


# In[48]:


submit_df = pd.DataFrame({'PassengerId': survivor_df["PassengerId"], 'Survived': prediction})


# ### <a id="nn">Neural Networks</a>

# In[ ]:


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)
score_list.append(clf.score(X_train, y_train))


# ### <a id="bn">Bayesian Networks</a>

# In[ ]:


gnb = GaussianNB()
gnb.fit(X_train, y_train)
score_list.append(gnb.score(X_train, y_train))


# ## <a id="concl">Conclusion</a>

# In[ ]:


models = ["Logistic Regression", "Random Forest", "Decision Tree", "Neural Networks", "Bayesian Networks"]


# In[ ]:


pd.DataFrame({"Models":models, "Score":score_list}).sort_values("Score", ascending=False)


# **Decision Tree** offers the best prediction with 86.4% of accuracy.

# In[55]:


submit_df.to_csv('submission.csv', index=False)


# In[54]:


ls


# In[ ]:




