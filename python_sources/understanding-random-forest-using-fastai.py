#!/usr/bin/env python
# coding: utf-8

# # Fastai Random Forest Algorithm

# In this competition task is find out how many passenger are survived on the ship. So, I am using random forest algorithm and fastai liabrary to find out how many passenger are survived.

# ## import data

# Here, I am using random forest to find passenger is survived or not and for this import necessary liabraries and titanic data

# In[ ]:


get_ipython().system('pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887')
get_ipython().system('apt update && apt install -y libsm6 libxext6')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.imports import *
from fastai.structured import *
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
from sklearn import metrics


# In[ ]:


import seaborn as sns
sns.set_style('whitegrid')


# In[ ]:


#path = 'titanic/'
get_ipython().system('ls ../input')


# In[ ]:


df = pd.read_csv('../input/titanic/train.csv')


# In[ ]:


df.head()


# 
# 
# Showing correlation between survived and embarked columns by using bargraph.

# In[ ]:


f,ax=plt.subplots(1,2, figsize=(18,8))
df[['Embarked','Survived']].groupby(['Embarked']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Embarked')
sns.countplot('Embarked',hue='Survived',data=df,ax=ax[1])
ax[1].set_title('Sex:Survived vs Embarked')
plt.show()


# Using bargraph showing how many people survived from ship. According to graph more than 50% people on the ship are died.

# In[ ]:


df.Survived.value_counts().plot(kind='bar',legend=True)


# According to graph, female are more than male present in the ship.

# In[ ]:


df.Sex.value_counts().plot(kind='bar')


# In Pclass columns there are 3 categories named as upper class(1), second class(2) and third class(3). I am showing how many upper class, second class and thirs class present in the ship using bar graph

# In[ ]:


_=df.Pclass.value_counts().plot(kind='bar')


# Graph showing the how many male and female was survived as well as showing correlation between sex and survived columns. According to the graph, female was survived more than male.

# In[ ]:


f,ax=plt.subplots(1,2, figsize=(18,5))
df[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=df,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()


# ## The data

# Convert all dataframe columns into categorical and separate prediction column into another variable and find accuraccy on training data.

# In[ ]:


df.head()


# In[ ]:


train_cats(df)


# In[ ]:


df,y,nas=proc_df(df, 'Survived')


# In[ ]:


df.head()


# In[ ]:


m=RandomForestClassifier(n_jobs=-1)
m.fit(df, y)
m.score(df,y)


# The given data is randomised for this I am writing one simple function to split validation data from training data. Here, Validation data is 50% of testing data. 50% means 209 rows seprate from training data. 

# In[ ]:


def split_vals(a, n): return a[:n].copy(), a[n:].copy()

n_valid = 209
n_trn = len(df) - n_valid
raw_train, raw_valid = split_vals(df, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape

Writting simple two functions to calculate RMSE(Root Mean Square Error) and accuracy of trainng and validation data. Here, print_score print the rmse and accuracy score of training and validation data. If we use oob_score parameter while fitting model then oob_score also print.
# In[ ]:


def rmse(x, y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res=[rmse(m.predict(X_train),y_train), rmse(m.predict(X_valid), y_valid),
         m.score(X_train, y_train), m.score(X_valid, y_valid)]
    
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


m = RandomForestClassifier(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


m = RandomForestClassifier(n_estimators=40, n_jobs=-1, max_depth=3, bootstrap=False)
m.fit(X_train, y_train)
print_score(m)


# Random Forest algorithm draw number of decision trees and aggregating them to find result. In previous random forest model creating 40 trees. I am showing one decision tree following, decision tree shows the feature name on which they classified, which algorithm is used for classification, how much sample used to draw tree and value. After drawing tree, i am trying to build model using some hyperparameter but sometime model is overfit and underfit.

# In[ ]:


m = RandomForestClassifier(n_jobs=-1, n_estimators=1, bootstrap=False)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


m =RandomForestClassifier(n_estimators=5, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


m = RandomForestClassifier(n_estimators=40, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


m = RandomForestClassifier(n_estimators=60, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# ## Out-of-bag ( OOB ) score

# Is our validation set worse than our training set because we're over-fitting, or because the validation set is for a different time period, or a bit of both? With the existing information we've shown, we can't tell. However, random forests have a very clever trick called out-of-bag (OOB) error which can handle this (and more!)
# 
# The idea is to calculate error on the training set, but only include the trees in the calculation of a row's error where that row was not included in training that tree. This allows us to see whether the model is over-fitting, without needing a separate validation set.
# 
# This also has the benefit of allowing us to see whether our model generalizes, even if we only have a small amount of data so want to avoid separating some out to create a validation set.
# 
# This is as simple as adding one more parameter to our model constructor. We print the OOB error last in our print_score function below.

# In[ ]:


m = RandomForestClassifier(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


m = RandomForestClassifier(n_estimators=200, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


X_train, X_valid = split_vals(df, n_trn)
m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# ## Reducing Over-fitting

# ### Tree building parameters

# In[ ]:


def dectree_max_depth(tree):
    children_left = tree.children_left
    children_right = tree.children_right

    def walk(node_id):
        if (children_left[node_id] != children_right[node_id]):
            left_max = 1 + walk(children_left[node_id])
            right_max = 1 + walk(children_right[node_id])
            return max(left_max, right_max)
        else: # leaf
            return 1

    root_node_id = 0
    return walk(root_node_id)


# In[ ]:


m = RandomForestClassifier(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


t = m.estimators_[0].tree_
dectree_max_depth(t)


# In[ ]:


m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


m = RandomForestClassifier(n_estimators=100, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


m = RandomForestClassifier(n_estimators=40, n_jobs=-1, min_samples_leaf=3, oob_score=True, max_features=0.5)
m.fit(X_train, y_train)
print_score(m)


# ## Feature Importance

# Find out importance of feature by giving model instance and dataframe to the rf_feat_importance method. After that, I decided which feature is important or not. According to my observation Name, Ticket, PassengerId, Embarked, Age_na, Fare are not important so that's why I am removing from dataframe and build model.

# In[ ]:


fi=rf_feat_importance(m,df); fi


# In[ ]:


feats=['Name','Ticket','PassengerId','Embarked','Age_na','Fare']


# In[ ]:


df.drop(feats, axis=1, inplace=True)


# In[ ]:


fi.plot('cols','imp',figsize=(5,6),legend=False)


# In[ ]:


X_train, X_valid = split_vals(df, n_trn)


# In[ ]:


df.head()


# In[ ]:


m = RandomForestClassifier(n_estimators=50, n_jobs=-1, min_samples_leaf=3,oob_score=True, random_state=1)
m.fit(X_train,y_train)
print_score(m)


# ## Final model

# In[ ]:


m = RandomForestClassifier(n_estimators=50, n_jobs=-1, min_samples_leaf=3, oob_score=True, 
                           random_state=1, max_features=None)
m.fit(X_train,y_train)
print_score(m)


# ## Testing model on test data

# When model behave good I am trying to tested on testing data. According to this model total 159 are survived from  testing data. In the last I am creating one submit.csv file according to the kaggle competition rule that contain only two columns named as PassengerId and Survived. 

# In[ ]:


df_test=pd.read_csv('../input/titanic/test.csv')
df_test.head()


# In[ ]:


train_cats(df_test)


# In[ ]:


df_test,y_name,nas=proc_df(df_test, 'Name')


# In[ ]:


df_test.head()


# In[ ]:


feats=['Ticket','PassengerId','Embarked','Age_na','Fare_na','Fare']


# In[ ]:


df_test.drop(feats, axis=1, inplace=True)


# In[ ]:


df_test.head()


# In[ ]:


df.head()


# In[ ]:


Survived=m.predict(df_test)
Survived.sum()


# In[ ]:


df_sample=pd.read_csv('../input/titanic/test.csv')


# In[ ]:


df_sample['Survived']=pd.Series(Survived)


# In[ ]:


df_sample.head()


# In[ ]:


df_sample.to_csv('../input/submit.csv',columns=['PassengerId','Survived'], index=False)


# In[ ]:


submit=pd.read_csv('submit.csv')


# In[ ]:




