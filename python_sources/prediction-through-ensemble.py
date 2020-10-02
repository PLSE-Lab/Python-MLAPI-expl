#!/usr/bin/env python
# coding: utf-8

# I am predicting the interest level of rental listing through Ensembles since these algorithms can give  boost in accuracy on the dataset. In this project i have used Boosting, Bagging and Majority Voting and compare the accuracy of the models on the datasets.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Supress unnecessary warnings so that presentation looks clean

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# ## Load the Data ##

# In[ ]:


#provides data structures to quickly analyze data
#Read the train dataset

dataset = pandas.read_json("../input/train.json") 


# ## Summarizing the Dataset ##

# In[ ]:


# Size of the dataframe
print(dataset.shape)


# In[ ]:


#view sample data
print (dataset.head(3))


# In[ ]:


#view the columns
list(dataset)


# In[ ]:


# Now we can take a look at a summary of each attribute.
# This includes the count, mean, the min and max values as well as some percentiles.
# descriptions are only for continuous variable
print(dataset.describe())


# ## Prepare data ##

# In[ ]:


data = dataset.loc[:, ['bathrooms','bedrooms','created','description','display_address','features',
                        'latitude','longitude','photos','price','street_address']] 

data['created'] = pandas.to_datetime(data['created'], coerce=True)
data['Year'] = data['created'].dt.year
data['Month'] = data['created'].dt.month
data['Day'] = data['created'].dt.day
data["description"] = data["description"].apply(len)
data["display_address"] = data["display_address"].apply(len)
data["features"] = data["features"].apply(len)
data["photos"] = data["photos"].apply(len)
data["street_address"] = data["street_address"].apply(len)
data = data.drop('created', 1)


# In[ ]:


dataset.interest_level.replace(['low','medium', 'high'], [1, 2, 3], inplace=True)
Y = dataset.loc[:,'interest_level']


# ## Evaluate Ensemble Algorithms ##

# ### Bagging###

# In[ ]:


#Bagging
#Bagged Decision Trees for Classification
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, data, Y, cv=kfold)
print(results.mean())


# In[ ]:


#Bagging
# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
seed = 7
num_trees = 100
max_features = 8
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model,data, Y, cv=kfold)
print(results.mean())


# In[ ]:


#Bagging
# Extra Trees Classification

from sklearn.ensemble import ExtraTreesClassifier
seed = 7
num_trees = 100
max_features = 8
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, data, Y, cv=kfold)
print(results.mean())


# ###Boosting###

# In[ ]:


#Boosting
# AdaBoost Classification
from sklearn.ensemble import AdaBoostClassifier

seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, data, Y, cv=kfold)
print(results.mean())


# In[ ]:


#Boosting
# Stochastic Gradient Boosting Classification
from sklearn.ensemble import GradientBoostingClassifier

seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, data, Y, cv=kfold)
print(results.mean())


# ###Voting Ensemble ###

# In[ ]:


# Voting Ensemble for Classification
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = LinearDiscriminantAnalysis()
estimators.append(('LDA', model3))
model4 = KNeighborsClassifier()
estimators.append(('KNN', model4))

# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, data, Y, cv=kfold)
print(results.mean())


# **By comparing all the models we can see Random Forest Classifier is the best.**

# ## Make Predictions ##

# In[ ]:


test_dataset = pandas.read_json("../input/test.json") 

test_data = test_dataset.loc[:, ['bathrooms','bedrooms','created','description','display_address','features',
                        'latitude','longitude','photos','price','street_address']] 

test_data['created'] = pandas.to_datetime(test_data['created'], coerce=True)

test_data['Year'] = test_data['created'].dt.year
test_data['Month'] = test_data['created'].dt.month
test_data['Day'] = test_data['created'].dt.day

test_data["description"] = test_data["description"].apply(len)
test_data["display_address"] = test_data["display_address"].apply(len)
test_data["features"] = test_data["features"].apply(len)
test_data["photos"] = test_data["photos"].apply(len)
test_data["street_address"] = test_data["street_address"].apply(len)

test_data = test_data.drop('created', 1)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
seed = 7
num_trees = 100
max_features = 8
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
classifier = model.fit(data,Y)


# In[ ]:


results = model_selection.cross_val_score(model,data, Y, cv=kfold)
print(results.mean())


# In[ ]:


prediction = classifier.predict(test_data) 
prediction_prob = classifier.predict_proba(test_data)


# In[ ]:


output = pandas.DataFrame()
output['listing_id'] = test_dataset['listing_id']
output['interest_level'] = prediction
output.interest_level.replace([1, 2, 3], ['low', 'medium', 'high'], inplace=True)


# In[ ]:


output_prob = pandas.DataFrame()
output_prob['listing_id'] = test_dataset['listing_id']

output_prob['high'] = prediction_prob[:,2]
output_prob['medium'] = prediction_prob[:,1]
output_prob['low'] = prediction_prob[:,0]


# In[ ]:




