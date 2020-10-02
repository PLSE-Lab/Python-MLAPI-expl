#!/usr/bin/env python
# coding: utf-8

# ## Advanced Ensemble Techniques Bagging
# this Kernel is built in base to this documents: https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/ , https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-5-ensembles-of-algorithms-and-random-forest-8e05246cbba7
# If I help you please upvote

# In[ ]:


import numpy as np 
import pandas as pd 

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import os
print(os.listdir("../input"))
np.random.seed(42)


# In[ ]:


def get_samples_bagging(data,target,size_col):
    features = np.random.choice(data.columns, size_col, replace=True)
    data = data[features]
    indices = list(set(np.random.randint(data.shape[0], size=data.shape[0])))
    data = data.iloc[indices,:]
    targ = target.iloc[indices,:]
    return data, targ,features


# This Kernel es a basic example of implementation about Stacking, this technique is very import for understand techniques more avanzed 

# Upload dataset

# In[ ]:


train = pd.read_csv("../input/train.csv",index_col='PassengerId')
test = pd.read_csv("../input/test.csv",index_col='PassengerId')#


# i count data missing and count dataset

# In[ ]:


train.shape,test.shape


# Show my Dataset

# In[ ]:


train.head()


# ## Engineer Features
# We work in the features Dataset, this example is about Stacking, and it's not important to deep in the feature

# In[ ]:


def replaceGen(sex):
    gen =0
    if sex=='male':
        gen=0
    elif sex=='female':
        gen=1
    return gen
    


# In[ ]:


train['Sex'] = train['Sex'].apply(replaceGen)
test['Sex'] = test['Sex'].apply(replaceGen)


# In[ ]:


train['Age'].hist(figsize=(10, 4));


# In[ ]:


train['Age'].fillna(train['Age'].mean(), inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)


# In[ ]:


test[test['Fare'].isna()]


# In[ ]:


Age_mean = train[(train['Pclass']==3) & (train['Embarked']=='S') & (train['Age']>55) & (train['Sex']==0)]['Fare'].mean()


# In[ ]:


test['Fare'].fillna(Age_mean, inplace=True)


# In[ ]:


X =train.drop(['Survived','Name','Ticket','Cabin','Embarked'],axis=1)
y =pd.DataFrame(train['Survived'])
test_f =test.drop(['Name','Ticket','Cabin','Embarked'],axis=1)


# In[ ]:


X.shape,y.shape


# In[ ]:


X.head()


# Bootstrap
# 
# Bootstrap is a method using in ensemble for create DataSet with features random, but each features could repeat with the same probability

# <img src="https://cdn-images-1.medium.com/max/1400/1*Ei3eNxEKrPm7qpcDXsW_MA.png" alt="Drawing" style="width: 600px;"/>

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


x_bag1,y_bag1,feat = get_samples_bagging(x_train,y_train,6)
print(x_bag1.head(5))
print(x_bag1.shape)
x_bag1,y_bag1,feat = get_samples_bagging(x_train,y_train,6)
print(x_bag1.head(5))
print(x_bag1.shape)
x_bag1,y_bag1,feat = get_samples_bagging(x_train,y_train,6)
print(x_bag1.head(5))
print(x_bag1.shape)


# We take each de the Dataset built,  get in our model and obtain combinate prediction

# <img src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/05/Screenshot-from-2018-05-08-13-11-49-768x580.png" alt="Drawing" style="width: 600px;"/>

# ## How does it work?, 
# 
# see this step for step
# 
# I Generated the first model and using predict_proba for get probability for column

# In[ ]:


np.random.seed(2)
x_bag1,y_bag1,feact = get_samples_bagging(x_train,y_train,6)
tmodel1 = DecisionTreeClassifier()
tmodel1.fit(x_bag1, y_bag1)
pd1 =tmodel1.predict_proba(x_test[feact])
pd.DataFrame(tmodel1.predict_proba(x_test[feact])).head(5)


# I Generated the seconf model and using predict_proba for get probability for column

# In[ ]:


np.random.seed(1767)
x_bag1,y_bag1,feact = get_samples_bagging(x_train,y_train,6)
tmodel2 = DecisionTreeClassifier()
tmodel2.fit(x_bag1, y_bag1)
pd2 =tmodel2.predict_proba(x_test[feact])
pd.DataFrame(tmodel2.predict_proba(x_test[feact])).head(5)


# Finally combination probabilities with mean

# In[ ]:


probs = []
probs.append(pd1)
probs.append(pd2)
meand_target = np.mean(probs, axis=0)
pd.DataFrame(np.mean(probs, axis=0)).head()


# We select the column with major probability

# In[ ]:


pd.DataFrame(np.argmax(meand_target, axis=1)).head()


# ----------------------------------------------------------------------

# Run with 10 samples

# In[ ]:


probs = []
model = []

for i in range(10):
    np.random.seed(19+i)
    x_bag1,y_bag1,feact = get_samples_bagging(x_train,y_train,6)
    model.append(DecisionTreeClassifier())
    model[i].fit(x_bag1, y_bag1)
    probs.append(model[i].predict_proba(x_test[feact]))

t = np.mean(probs, axis=0)
y_target =np.argmax(t, axis=1).reshape(-1,1)
accuracy_score(y_test, y_target)


# Run with 100 samples

# In[ ]:


probs = []
model = []
for i in range(100):
    np.random.seed(19+i)
    x_bag1,y_bag1,feact = get_samples_bagging(x_train,y_train,6)
    model.append(DecisionTreeClassifier())
    model[i].fit(x_bag1, y_bag1)
    probs.append(model[i].predict_proba(x_test[feact]))

t = np.mean(probs, axis=0)
y_target =np.argmax(t, axis=1).reshape(-1,1)
accuracy_score(y_test, y_target)


# Run with 1000 samples

# In[ ]:


probs = []
model = []
for i in range(1000):
    np.random.seed(19+i)
    x_bag1,y_bag1,feact = get_samples_bagging(x_train,y_train,6)
    model.append(DecisionTreeClassifier())
    model[i].fit(x_bag1, y_bag1)
    probs.append(model[i].predict_proba(x_test[feact]))

t = np.mean(probs, axis=0)
y_target =np.argmax(t, axis=1).reshape(-1,1)
accuracy_score(y_test, y_target)


# Run model with all Dataset

# In[ ]:


probs = []
model = []
for i in range(10):
    x_bag1,y_bag1,feact = get_samples_bagging(x_train,y_train,6)
    model.append(DecisionTreeClassifier())
    model[i].fit(x_bag1, y_bag1)
    probs.append(model[i].predict_proba(test_f[feact]))

t = np.mean(probs, axis=0)


# In[ ]:


pd.DataFrame(t).head()


# In[ ]:


y_target =np.argmax(t, axis=1).reshape(-1,1)


# In[ ]:


d_ytarget = pd.DataFrame(y_target)


# In[ ]:


test_f_salida = pd.DataFrame( { 'PassengerId': test_f.index , 'Survived': d_ytarget[0]} )


# In[ ]:


#Show Output
test_f_salida.head(20)


# In[ ]:


test_f_salida.to_csv( 'titanic_pred.csv' , index = False )

