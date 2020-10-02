#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mlp
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Begin cleaning data to feed into random forest model

# In[ ]:


train_data = pd.read_csv(r"/kaggle/input/titanic/train.csv")
train_data.pop('Cabin') #removing cabin column
train_data.pop('Ticket')


#checking for missing data
missing_data = pd.concat([train_data.isnull().sum()], axis=1, keys=['train_data'])
missing_data[missing_data.sum(axis=1) > 0]

#train_data.shape

#filling missing data
#fill in missing embarked with most frequent value in dataset 
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

train_data.head()


# ### Saving cleaned train_data df to new csv file 

# In[ ]:


#train_data_clean = pd.DataFrame({'PassengerId': train_data.PassengerId, 'Survived': train_data.Survived, 'Pclass': train_data.Pclass, 'Name': train_data.Name, 'Sex': train_data.Sex, 'Age': train_data.Age, 'SibSp': train_data.SibSp, 'Parch': train_data.Parch, 'Fare': train_data.Fare, 'Embarked': train_data.Embarked})
#train_data_clean.to_csv(r'Downloads\train_dat.csv', index = False)
#print("file saved successfully")


# In[ ]:


test_data = pd.read_csv(r'/kaggle/input/titanic/test.csv')
test_data.pop('Cabin')
test_data.pop('Ticket')


#checking missing data
missing_data = pd.concat([test_data.isnull().sum()], axis=1, keys=['test_data'])
missing_data[missing_data.sum(axis=1) > 0]

test_data.shape

#filling missing values for age with mean age value
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
test_data['Embarked'] = test_data['Embarked'].fillna(test_data['Embarked'].mode()[0]) # mode for most often

test_data.head()


# ### Saving cleaned test_data df to new csv file 

# In[ ]:


#test_data_clean = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Pclass': test_data.Pclass, 'Name': test_data.Name, 'Sex': test_data.Sex, 'Age': test_data.Age, 'SibSp': test_data.SibSp, 'Parch': test_data.Parch, 'Fare': test_data.Fare, 'Embarked': test_data.Embarked})
#test_data_clean.to_csv(r'Downloads\test_data1_cleaned.csv', index = False)
#print("file saved successfully")


# # Calculating percentage of survival rates based on gender

# In[ ]:


women = train_data.loc[train_data.Sex == 'female']["Survived"]
prop_women = sum(women) / len(women)
print("% of women who survived Titanic Crash:", prop_women)


# In[ ]:


men = train_data.loc[train_data.Sex == 'male']["Survived"]
prop_men = sum(men) / len(men)
print("% of men who survived Titanic Crash:", prop_men)


# ### Building a random forest model to see if predictions for survival can be improved based on more column data
# * Performing one hot encoding and using pd.get_dummies() 

# In[ ]:


#Performing one hot encoding of categorical features 
# getting Dummies from all other categorical vars
train_data['Pclass'] = train_data['Pclass'].apply(str)

for col in train_data.dtypes[train_data.dtypes == 'object'].index:
    for_dummy = train_data.pop(col)
    train_data = pd.concat([train_data, pd.get_dummies(for_dummy, prefix=col)], axis=1)
    
train_data.head()


# In[ ]:


labels = train_data.pop('Survived')


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.25)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from warnings import simplefilter #runs slower but doesnt iterate through all the lines of future warnings
simplefilter(action='ignore', category=FutureWarning) #update scikit learn version to take this snippet away 
from math import sqrt

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
max_depth=7, max_features='auto', max_leaf_nodes=None,
min_impurity_split=1e-07, min_samples_leaf=1,
min_samples_split=0.3, min_weight_fraction_leaf=0.0,
n_estimators=10, n_jobs=-1, oob_score=False, random_state=0,
verbose=0, warm_start=True)
y_pred = rf.predict(x_test)
print(y_pred)

rf_output = pd.DataFrame({'PassengerId': test_set.PassengerId, 'Survived': predictions})
#model_output.head(10)

import os
os.chdir(r'/kaggle/working')

model_output.to_csv('submiss.csv', index=False)
#print("save success")

from IPython.display import FileLink
FileLink(r'submiss.csv')


# In[ ]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# ## Parameter tuning random forest (n_estimators, max_depths, min_samples_split)
# - n_estimators represents the number of trees in the forest. 
# - Usually the higher the number of trees the better to learn the data. 
# - However, adding a lot of trees can slow down the training process considerably, therefore we do a parameter search to find the sweet spot.

# In[ ]:


n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
train_results = []
test_results = []
for estimator in n_estimators:
    rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
    
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")
line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.show()


# ## Tuning max_depths for random forest
# - max_depth represents the depth of each tree in the forest. The deeper the tree, the more splits it has and it captures more information about the data. 
# - We fit each decision tree with depths ranging from 1 to 32 and plot the training and test errors.

# In[ ]:


from warnings import simplefilter #runs slower but doesnt iterate through all the lines of future warnings
simplefilter(action='ignore', category=FutureWarning) #update scikit learn version to take this snippet away 

max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
    rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()


# - We see that our model overfits for large depth values. The trees perfectly predicts all of the train data, however, it fails to generalize the findings for new data

# ## min_samples_split
# - min_samples_split represents the minimum number of samples required to split an internal node. 
# - This can vary between considering at least one sample at each node to considering all of the samples at each node. When we increase this parameter, each tree in the forest becomes more constrained as it has to consider more samples at each node. 
# - Here we will vary the parameter from 10% to 100% of the samples

# In[ ]:


min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True) #10% to 100% varying of the sample
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
    rf = RandomForestClassifier(min_samples_split=min_samples_split)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_splits, train_results, 'b', label="Train AUC")
line2, = plt.plot(min_samples_splits, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples split')
plt.show()


# # max_features
# - max_features represents the number of features to consider when looking for the best split.

# In[ ]:


max_features = list(range(1,train_data.shape[1]))
train_results = []
test_results = []
for max_feature in max_features:
    rf = RandomForestClassifier(max_features=max_feature)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_features, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_features, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('max features')
plt.show()


# # Above shows the parameter tuning of random forest model

# In[ ]:


train_set = pd.read_csv('/kaggle/input/train-clean/train_data1_cleaned.csv')
test_set = pd.read_csv('/kaggle/input/test-clean/test_data1_cleaned.csv')


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

response = train_set["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_set[features])
X_test = pd.get_dummies(test_set[features])
                    
model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
max_depth=7, max_features=3, max_leaf_nodes=None,
min_impurity_split=1e-07, min_samples_leaf=1,
min_samples_split=0.3, min_weight_fraction_leaf=0.0,
n_estimators=27, n_jobs=-1, oob_score=False, random_state=0,
verbose=0, warm_start=True)
model.fit(X, response)
predictions = model.predict(X_test)


model_output = pd.DataFrame({'PassengerId': test_set.PassengerId, 'Survived': predictions})
#model_output.head(10)

import os
os.chdir(r'/kaggle/working')

model_output.to_csv('subTra.csv', index=False)
#print("save success")

from IPython.display import FileLink
FileLink(r'subTra.csv')


# In[ ]:




