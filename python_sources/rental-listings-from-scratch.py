#!/usr/bin/env python
# coding: utf-8

# ## Background
# I'm starting with some simple approaches and not using others' kernels or notebooks for this.  I've done that for other projects (housing regression and Titanic).  That's useful for learning new topics, but the real growth happens when you come up with the script yourself.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import linear_model
from collections import Counter


# In[ ]:


train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")


# In[ ]:


test.shape


# In[ ]:


train.shape


# In[ ]:


train.info()


# In[ ]:


train.head()


# In[ ]:


train['interest_level'].value_counts(normalize=True)


# ## Game Plan 1: The Layup
# Create a simple regression model with the following features:
# - bedrooms
# - bathrooms
# - price
# - features
# 
# Additional processing will be needed to extract features (they are currently in list form) and create dummies from them.

# In[ ]:


print(train.shape[0])
print(train.shape[0]/8)


# ### Feature Processing - Creating a list of all features from features columns
# This runs much faster in batches.  I needed the batches to be in whole numbers (with no decimals), and realized 8 batches/lots of 6169 will equal the train size of 49,352 rows
# 

# In[ ]:


x = train['features'].tolist()

feature_list = []

for i in range(0,len(x),6169):
    print(i+6169)
    feature_listadd = []
    for j in range(i,i+6169):#len(x)):
        feature_listadd = feature_listadd + x[j]
    feature_list = feature_list + feature_listadd
    
#lowercase is necessary, otherwise matches will be missed when there are differences in case
feature_list = [item.lower() for item in feature_list]
cl = Counter(feature_list)


# In[ ]:


#Keeping features that occur at least 1,000 times yields the following list

feature_list = []

for key in cl:
    if cl[key]>1000:
        feature_list.append(key)
        print(key)

feature_list


# ## Creating Dummy Variables for the Feature List

# In[ ]:


for i in feature_list:
    train[i] = train['features'][:train.shape[0]].apply         (lambda x: 1 if i in [y.lower() for y in x] else 0)


# In[ ]:


for i in feature_list:
    test[i] = test['features'][:test.shape[0]].apply         (lambda x: 1 if i in [y.lower() for y in x] else 0)


# In[ ]:


#Combining 'pre-war' & 'prewar'

train['prewar'] = train['prewar'] + train['pre-war']
del train ['pre-war']
test['prewar'] = test['prewar'] + test['pre-war']
del test ['pre-war']


# ### Choosing a model:
# This isn't a simple linear regression after all, but rather would require an ordinal model.  I still haven't looked to see what others are doing, but I'm planning to turn this into a simpler regression by changing interest levels: low=0, medium=1, high=2.  Thus, we will have a linear relationship.  And, if our model predict 1.6 for example, I would interpret that as 60% chance for high and 40% chance for medium. I'm sure this isn't the perfect model, but it makes sense, roughly.

# In[ ]:


mapping = {'low': 0, 'medium': 1, 'high':2}
train['interest_level'] = train['interest_level'].apply(lambda x: mapping.get(x))


# # Model Time

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


feature_list.remove('pre-war')
feature_list.append('bedrooms')
feature_list.append('bathrooms')
feature_list.append('price')


# In[ ]:


feature_list


# In[ ]:


modeltrain, modeltest = train_test_split(train, test_size = 0.2,random_state=0)

#For testing
Xmodeltrain = modeltrain[feature_list]
Ymodeltrain = modeltrain['interest_level']

Xmodeltest = modeltest[feature_list]
Ymodeltest = modeltest['interest_level']

#For fitting all the data
Xtrain = train[feature_list]
Ytrain = train['interest_level']

#For prediction from the test tile
Xtest = test[feature_list]


# In[ ]:


#Models
regr = linear_model.LinearRegression()
random_forest = RandomForestClassifier(n_estimators=100)
gaussian = GaussianNB()
logreg = linear_model.LogisticRegression()
kneighbors = KNeighborsClassifier()

#Model List
Models = [regr,random_forest,gaussian,logreg,kneighbors]

#Fitting
for i in Models:
    i.fit(Xmodeltrain,Ymodeltrain)


# In[ ]:


def runscore(x):
    #print('\n')
    print('Test Score: ' + str(x.score(Xmodeltest, Ymodeltest)))
    print('Train Score: ' + str(x.score(Xmodeltrain, Ymodeltrain)))


# In[ ]:


#Run Scores
for model in Models:
    print('\n')
    print(model)
    runscore(model)


# ## Results: Random Forest, KNeighbors, and Logistic are pretty close

# In[ ]:


for model in Models:
    model.fit(Xtrain, Ytrain)


# In[ ]:


for model in Models:
    test['interest_level']=model.predict(Xtest)
    print('\n')
    print(model)
    print(test['interest_level'].value_counts(normalize=True))


# ### Random Forest looks best

# In[ ]:


test['interest_level']=random_forest.predict(Xtest)


# In[ ]:


test['interest_level'].value_counts(normalize=True)


# In[ ]:


maplow = {0: 1, 1: 0, 2:0}
mapmedium = {0: .1, 1: .85, 2:.05}
maphigh = {0:0, 1:0, 3:1}

test['low'] = test['interest_level'].apply(lambda x: maplow.get(x))
test['medium'] = test['interest_level'].apply(lambda x: mapmedium.get(x))
test['high'] = test['interest_level'].apply(lambda x: maphigh.get(x))


# In[ ]:


submission = test[['listing_id','high','medium','low']]


# In[ ]:


#submission = submission.set_index('listing_id')
submission.head()


# In[ ]:


submission.to_csv('submissionjam2.csv', index=False)


# In[ ]:




