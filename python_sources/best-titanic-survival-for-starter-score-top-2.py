#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Competition for Starter...
# ### I am using here Random Forest Classifier Algorithm that will classify which Passenger is Survived or not. For each step, running Markdown cell for better undderstanding.

# ## This is my first Kernel here....
# # Please Upvote my kernel so that i will be thankful to you.

# **Importing Simple Libraries**

# In[ ]:


import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics


# **Read input train and test data**

# In[ ]:


df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# ### Now I am understanding training data and applying feature engineering.

# In[ ]:


df.head()


# In[ ]:


print(df.shape)


# In[ ]:


y_train = df['Survived']
y_train.shape


# In[ ]:


df.drop(['Survived'],axis=1,inplace=True)
df.info()


# **Calculating total number of blank data**

# In[ ]:


df.isnull().sum()


# **Visualising null data in each feature with heat map** 

# In[ ]:


sns.heatmap(df.isnull(), yticklabels=False, cbar=True)


# In[ ]:


df[['Pclass','Age']].groupby(['Pclass'],as_index=False).mean().sort_values(['Age'],ascending=False)


# **Define a function which replace age with most specific value**

# In[ ]:


def replace_nan_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if(pd.isnull(Age)):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 30
        else:
            return 25
    else: 
        return Age


# **Replace null data of age column according to Pclass column that is most correlated together**

# In[ ]:


df['Age'] = df[['Age','Pclass']].apply(replace_nan_age, axis=1)


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=True)


# **Drop Cabin column**

# In[ ]:


df.drop(columns='Cabin',axis=1,inplace=True)


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=True)


# **Drop Name Ticket column to get with numerical values**

# In[ ]:


df.drop(columns=['Name','Ticket'],axis=1,inplace=True)


# ## One Hot Encoding

# **Applying onehot encoding with Sex column**

# In[ ]:


Sex = pd.get_dummies(df['Sex'],drop_first=True)
df = pd.concat([df,Sex],axis=1)
df.drop(['Sex'],axis=1,inplace=True)


# **Considering Embarked column**

# In[ ]:


df['Embarked'].value_counts()


# **Applying onehot encoding on Embarked feature** 

# In[ ]:


Embarked=pd.get_dummies(df['Embarked'])
df=pd.concat([df,Embarked],axis=1)
df.drop(['Embarked'],axis=1,inplace=True)


# In[ ]:


df.info()


# **All features are converted in numerical values and now look our training data**

# In[ ]:


df.head()


# ## Similarly Applying feature engineering on test data

# In[ ]:


df_test.head()


# In[ ]:


df_test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)


# **Applying Onehot Encoding on Sex feature**

# In[ ]:


Sex = pd.get_dummies(df_test['Sex'],drop_first=True)
df_test = pd.concat([df_test,Sex],axis=1)
df_test.drop(['Sex'],axis=1,inplace=True)


# **Applying onehot encoding on Embarked feature**

# In[ ]:


Embarked=pd.get_dummies(df_test['Embarked'])
df_test=pd.concat([df_test,Embarked],axis=1)
df_test.drop(['Embarked'],axis=1,inplace=True)


# In[ ]:


df_test.info()


# In[ ]:


df_test.corr()


# **Here we see age column is more correlated with Pclass as compare to Sex(male) feature**

# In[ ]:


df_test[['Pclass','Age']].groupby(['Pclass'],as_index=False).mean().sort_values(['Age'],ascending=False)


# In[ ]:


df[['male','Age']].groupby(['male'],as_index=False).mean().sort_values(['Age'],ascending=False)


# **So we go through Pclass cloumn to replace age with its most specific values by defininig function**

# In[ ]:


def replace_nan_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if(pd.isnull(Age)):
        if Pclass == 1:
            return 41
        elif Pclass == 2:
            return 29
        else:
            return 24
    else: 
        return Age


# In[ ]:


df_test['Age'] = df_test[['Age','Pclass']].apply(replace_nan_age, axis=1)


# **Replace NaN value of Fare column with its mean value**

# In[ ]:


df_test.replace(np.nan,df_test['Fare'].mean(),inplace=True)


# **Now we see all values are filled and all are numerical values**

# In[ ]:


df_test.info()


# In[ ]:


print(df.shape)
print(df_test.shape)


# ## Modeling

# **Applying Random Forest Algorthm**

# **Now Spliting Data into training and validation data**

# In[ ]:


x_trn,x_valid,y_trn,y_valid=train_test_split(df,y_train,test_size=0.33,random_state=150)


# In[ ]:


model=RandomForestClassifier(n_estimators=200,random_state=200,max_features=0.5,min_samples_leaf=3,oob_score=True,n_jobs=-1)
model.fit(df,y_train)


# **Our Score on Validation Data**

# In[ ]:


model.score(x_valid,y_valid)


# **Prediction on Test Set**

# In[ ]:


predict_y = model.predict(df_test)


# **Our Score on Training data**

# In[ ]:


model.score(x_trn,y_trn)


# **Finally view our feature importance value of all features**

# In[ ]:


model.feature_importances_
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_trn.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(18,12))
plt.title("Feature importances")
plt.bar(range(x_trn.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(x_trn.shape[1]), indices)
plt.xlim([-1, x_trn.shape[1]])
plt.show()


# ## Prepare our Submission file

# In[ ]:


my_submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': predict_y })
my_submission.to_csv('submission.csv', index=False)

