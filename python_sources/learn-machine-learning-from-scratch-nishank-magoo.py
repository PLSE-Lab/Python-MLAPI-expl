#!/usr/bin/env python
# coding: utf-8

# <h1>Learning Machine Learning from Scrtach
# <h3>Nishank Magoo -  Dec 2018 
# 
# 
# 
# *! This is an ongoing project that I will it update regularly, so stay tuned !*
# 
# 
# 
# 

# This notebook is going to be focused on learning and understanding all machine learning algorithms .This will be divided into 3 parts for better comprehension -
# 
# * Supervised Learning
# * Unsupervised Learning
# * Deep Learning

# **Introduction**

# **Here are some of the most important Supervised learning algorithms **
# 
# 1. K Nearest neighbour - Distance based Classification Model
# 2. Naive Bayes - Probability based Classification Model
# 3.  Logistic Regression - Classification Model for linearly separable data
# 4.  Linear regression - linear Algebra Based Regression Model
# 5.  Support vector Machines - Similar to logistic Regression Classification Model. The goal is to find the best line that has maximum probability of classifying unseen points correctly. How you define the notion of "best" gives you different models like SVM and LR
# 6.  Decison tree - nested If-Else Classfictaion 
# 7.  Random Forest - Decision tree + Row Sampling + Column Sampling . Low bias + high Variance ---Apply RF---> low bias + Low Variance
# 8.  Gradient boosting - Decision tree + Optimization by Differentiable Loss Function . Examples of loss function - Square Loss, Hinge Loss, Logistic Loss, log loss, Exponential Loss
# 9.  XGBoost - Gradient bosting + row sampling + column sampling . high bias + low Variance ---Apply XGB---> low bias + Low Variance
# 
# **Here are some of the most important UnSupervised learning algorithms**
# 
# 1. K Means - centroid based Model 
# 2. Hierarchichal - hierarchy based Model
# 3. DBSCAN - Density-based spatial clustering of applications with noise - Densiy Based Model . (Dense- Sparse)
# 
# **And Lastly, Deep Learning**
# 
# 1. Artificial Neural network (Input + One Hidden layer + Output)
# 2. Deep Multi Layer Perceptrons (Input + Multiple hidden Layers + Output)
# 3. Convolutional Neural Network - Majorly for Image . Receptive Field. Horizontal Edge Detection. Vertical Edge detection. Padding. Strides
# 4. Recurrent neural network - Majorly for "Sequence of Words"
# 
# 
# 
# 

# **Workflow stages**

# Every Problem goes through these seven stages - 
# 
# 1. Question or problem definition.
# 2. Acquire training and testing data.
# 3. Wrangle, prepare, cleanse the data.
# 4. Analyze, identify patterns, and explore the data.
# 5. Model, predict and solve the problem.
# 6. Visualize, report, and present the problem solving steps and final solution.
# 7. Supply or submit the results.

# **Question or Problem Definition **

# Competition sites like Kaggle define the problem to solve or questions to ask while providing the datasets for training your data science model and testing the model results against a test dataset. 

# **Model 1 :  SVM - Linear Accuracy 79 %**

# In[ ]:


# Import data analysis and wrangling libraries
import pandas as pd
import numpy as np
import random as rnd

# Import visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Import machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#Acquire Data
features = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')


# 

# In[ ]:






def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'
    
def title_map(title):
    if title in ['Mr']:            
        return 1
    elif title in ['Master']:
        return 2
    elif title in ['Ms', 'Mlle', 'Miss']:
        return 3
    elif title in ['Mme', 'Mrs']:
        return 4
    else:
        return 5
    
# Get Title from names    
features['Title'] = features['Name'].apply(get_title).apply(title_map)
    
    
#One Hot Encoding - Convert categorical data to binary vectors - Train data
one_hot = pd.get_dummies(features['Sex'])
one_hot2 = pd.get_dummies(features['Embarked'])
one_hot3 = pd.get_dummies(features['Title'])
features = features.join(one_hot).join(one_hot2).join(one_hot3)

#Clean Data - Drop unwanted Train Data
features = features.drop('Sex',  axis = 1)
features = features.drop('Embarked', axis = 1)
features = features.drop('Cabin', axis = 1)
features = features.drop('Name', axis =1)
features = features.drop('Ticket', axis = 1)

#fill blank values with mean value
features = features.fillna(features.mean())
features.iloc[:,5:].head(5)

labels = np.array(features['Survived'])
features = features.drop('Survived', axis = 1)


#Converting features data frame into np array to be used in regressor model 
features = np.array(features)

#random forest Regression
rf = RandomForestRegressor (n_estimators = 1000, random_state = 42)
rf.fit(features, labels)

test_data['Title'] = test_data['Name'].apply(get_title).apply(title_map)

#One Hot Encoding - Convert categorical data to binary vectors - Test Data
one_hot = pd.get_dummies(test_data['Sex'])
one_hot2 = pd.get_dummies(test_data['Embarked'])
one_hot3 = pd.get_dummies(test_data['Title'])

test_data = test_data.join(one_hot)
test_data = test_data.join(one_hot2)
test_data = test_data.join(one_hot3)

#Clean Data - Drop unwanted Test Data
test_data = test_data.drop('Sex',  axis = 1)
test_data = test_data.drop('Embarked', axis = 1)
test_data = test_data.drop('Cabin', axis = 1)
test_data = test_data.drop('Name', axis =1)
test_data = test_data.drop('Ticket', axis = 1)

test_data = test_data.fillna(test_data.mean())

test_data = np.array(test_data)
predictions = rf.predict(test_data)


print(predictions)

#Export Result to CSV 
test_dataset_copy = pd.read_csv('../input/titanic/test.csv')
submission = pd.DataFrame({
        "PassengerId": test_dataset_copy["PassengerId"],
        "Survived": predictions
})

submission.to_csv('submission_rf.csv', index=False)


# **Model 2 : SVM - Linear Accuracy 79 %**

# In[ ]:



# Import data analysis and wrangling libraries
import pandas as pd
import numpy as np
import random as rnd

# Import visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Import machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#Acquire Data
features = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')


def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'
    
def title_map(title):
    if title in ['Mr']:            
        return 1
    elif title in ['Master']:
        return 2
    elif title in ['Ms', 'Mlle', 'Miss']:
        return 3
    elif title in ['Mme', 'Mrs']:
        return 4
    else:
        return 5
        
# Get Title from names    
features['Title'] = features['Name'].apply(get_title).apply(title_map)
    
#One Hot Encoding - Convert categorical data to binary vectors - Train data
one_hot = pd.get_dummies(features['Sex'])
one_hot2 = pd.get_dummies(features['Embarked'])
one_hot3 = pd.get_dummies(features['Title'])
features = features.join(one_hot).join(one_hot2).join(one_hot3)


#Clean Data - Drop unwanted Train Data
features = features.drop('Sex',  axis = 1)
features = features.drop('Embarked', axis = 1)
features = features.drop('Cabin', axis = 1)
features = features.drop('Name', axis =1)
features = features.drop('Ticket', axis = 1)

#fill blank values with mean value
features = features.fillna(features.mean())


#labels = np.array(features['Survived'])
labels = features['Survived']
features = features.drop('Survived', axis = 1)

feature_list = list(features.columns)
#features = np.array(features)

svclassifier = SVC(kernel = 'linear')
svclassifier.fit(features, labels)

test_data['Title'] = test_data['Name'].apply(get_title).apply(title_map)


#One Hot Encoding - Convert categorical data to binary vectors - Test Data
one_hot = pd.get_dummies(test_data['Sex'])
one_hot2 = pd.get_dummies(test_data['Embarked'])
one_hot3 = pd.get_dummies(test_data['Title'])

test_data = test_data.join(one_hot)
test_data = test_data.join(one_hot2)
test_data = test_data.join(one_hot3)

#Clean Data - Drop unwanted Test Data
test_data = test_data.drop('Sex',  axis = 1)
test_data = test_data.drop('Embarked', axis = 1)
test_data = test_data.drop('Cabin', axis = 1)
test_data = test_data.drop('Name', axis =1)
test_data = test_data.drop('Ticket', axis = 1)

test_data = test_data.fillna(test_data.mean())

predictions = svclassifier.predict(test_data)

print(predictions)


#Export Result to CSV 
test_dataset_copy = pd.read_csv('../input/titanic/test.csv')
submission = pd.DataFrame({
        "PassengerId": test_dataset_copy["PassengerId"],
        "Survived": predictions
})

submission.to_csv('submission_svm.csv', index=False)


# **Model 3 : XGBoost Model**

# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score

#Acquire Data
features = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')


def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'
    
def title_map(title):
    if title in ['Mr']:            
        return 1
    elif title in ['Master']:
        return 2
    elif title in ['Ms', 'Mlle', 'Miss']:
        return 3
    elif title in ['Mme', 'Mrs']:
        return 4
    else:
        return 5
        
# Get Title from names    
features['Title'] = features['Name'].apply(get_title).apply(title_map)
    
#One Hot Encoding - Convert categorical data to binary vectors - Train data
one_hot = pd.get_dummies(features['Sex'])
one_hot2 = pd.get_dummies(features['Embarked'])
one_hot3 = pd.get_dummies(features['Title'])
features = features.join(one_hot).join(one_hot2).join(one_hot3)


#Clean Data - Drop unwanted Train Data
features = features.drop('Sex',  axis = 1)
features = features.drop('Embarked', axis = 1)
features = features.drop('Cabin', axis = 1)
features = features.drop('Name', axis =1)
features = features.drop('Ticket', axis = 1)

#fill blank values with mean value
features = features.fillna(features.mean())


#labels = np.array(features['Survived'])
labels = features['Survived']
features = features.drop('Survived', axis = 1)

feature_list = list(features.columns)
#features = np.array(features)

x = np.array(features)
y = np.array(labels)

clf = xgb.XGBClassifier()
cv = cross_validation.KFold(len(x), n_folds=20, shuffle=True, random_state=1)
scores = cross_validation.cross_val_score(clf, x, y, cv=cv, n_jobs=1, scoring='accuracy')
clf.fit(x,y)


test_data['Title'] = test_data['Name'].apply(get_title).apply(title_map)


#One Hot Encoding - Convert categorical data to binary vectors - Test Data
one_hot = pd.get_dummies(test_data['Sex'])
one_hot2 = pd.get_dummies(test_data['Embarked'])
one_hot3 = pd.get_dummies(test_data['Title'])

test_data = test_data.join(one_hot)
test_data = test_data.join(one_hot2)
test_data = test_data.join(one_hot3)

#Clean Data - Drop unwanted Test Data
test_data = test_data.drop('Sex',  axis = 1)
test_data = test_data.drop('Embarked', axis = 1)
test_data = test_data.drop('Cabin', axis = 1)
test_data = test_data.drop('Name', axis =1)
test_data = test_data.drop('Ticket', axis = 1)

test_data = test_data.fillna(test_data.mean())

predictions = clf.predict(test_data)

print(predictions)

#Export Result to CSV 
test_dataset_copy = pd.read_csv('../input/titanic/test.csv')
submission = pd.DataFrame({
        "PassengerId": test_dataset_copy["PassengerId"],
        "Survived": predictions
})

submission.to_csv('submission_xgb.csv', index=False)


# 
