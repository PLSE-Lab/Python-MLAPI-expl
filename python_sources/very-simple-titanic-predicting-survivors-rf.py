#!/usr/bin/env python
# coding: utf-8

# # Very Simple Predicting the Survival of Titanic Passengers with RandomForest

# ### In this challenge, we are asked to predict whether a passenger on the titanic would have been survived or not.

# ## Importing Libraries and Packages

# In[9]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# disable Pandas notification 
pd.options.mode.chained_assignment = None
#to write model in file
from sklearn.externals import joblib
# for plotting
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

#function for check missing data in Dataframes
def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    alldata = pd.concat([total, percent], axis=1, keys=['Total', 'in Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    alldata['Types'] = types
    return(np.transpose(alldata)) 


# ## Loading and Viewing Dataset

# In[10]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv('../input/test.csv')


# In[11]:


train_df.head()


# In[12]:


test_df.head()


# ### Viewing missing values

# As you remember, we wrote a small function to display NaN values in our datasets. Let's use it.

# In[13]:


missing_data(train_df)


# In[14]:


missing_data(test_df)


# Ok. There are NaN values in our data set in the Age, Fare and Embarked columns. We will fill them later. We can also drop the Cabin column as it contains a lot of NaN values.

# ## Feature Engineering

# Let's look at the Embarked column to determine better candidate to fill in the missing values.

# In[15]:


train_df['Embarked'].value_counts()


# So we use value *"S"*. For fill all Nan values in Age columns we will use median(), for fare mean() value. Also we add some extra field named FamSize (counting family).

# In[16]:


# Getting median for Age
median_age = train_df.Age.median()
# Getting mean value for Fare
mean_fare = train_df.Fare.mean()
# Feature Engineering
for line in [train_df, test_df]:    
    # filling 'Age' Nan values and converting to integer type
    line['Age'].fillna(median_age, inplace = True)
    line['Age'] = line['Age'].astype(int)    
    # Filling 'Fare' feature with mean value
    line['Fare'].fillna(mean_fare, inplace = True)    
    # Filling 'Embarked' nan with S
    line['Embarked'].fillna('S', inplace = True)    
    # Creating new feature "Family Size"
    line['FamSize'] = line.SibSp + line.Parch


# ### Converting Features

# In[17]:


# Creating dictionarys for mapping values
map_embarked = {'S': 1, 'C': 2, 'Q': 3}
map_sex = {'male': 1, 'female': 2}

for dataset in [train_df, test_df]:
    dataset['Embarked'] = dataset.Embarked.map(map_embarked)
    dataset['Sex'] = dataset.Sex.map(map_sex)


# In both datasets we drop 'PassengerId', 'Name', 'Ticket' and 'Cabin' features 

# In[18]:


train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)


# In[19]:


test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)


# Take a look at our data again

# In[20]:


train_df.head()


# In[21]:


test_df.head()


# In[22]:


train_df.info()


# In[23]:


test_df.info()


# Looks like everything is good. So we can start building machine learning model

# ## Building Machine Learning Model

# In[24]:


X_train = train_df.drop('Survived', axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop('PassengerId', axis=1).copy()


# In[25]:


# Random Forest with oob

random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)


# In[26]:


print('Accuracy:', acc_random_forest, '%')


# In[27]:


print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


# ### Feature Importance

# In[28]:


feature_imp = pd.DataFrame({'Feature':X_train.columns,'Importance':np.round(random_forest.feature_importances_,3)})
feature_imp = feature_imp.sort_values('Importance',ascending=False).set_index('Feature')
feature_imp.head(15)


# In[29]:


submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':Y_prediction})


# In[30]:


submission.head()


# In[31]:


# submission.to_csv('submission.csv', index=False)


# ## Taking a prediction on me.

# In[32]:


mean_fare


# We have a model, we need to try it. Let's imagine me on Tiatnic and try to predict whether I will survive or not. So, Im 32 years old (Age = 32) man (Sex = 1) traveling Second class (Pclass = 2) without family (SibSp = 0, Parch = 0, FamSize = 0). My Port of Embarkation is Southampton (Embarked = 1). And my Fare is mean_fare value (32.2). Array to prediction will be look like this '2, 1, 32, 0, 0, 32.2000, 1, 0'

# In[33]:


im = random_forest.predict([[2, 1, 32, 0, 0, 32.2000, 1, 0]])


# In[34]:


print ('Survived =', im[0])


# Unfortunately, the model predicts not that I want to see. But it's works.
