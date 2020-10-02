#!/usr/bin/env python
# coding: utf-8

# **KNN to analyze adult dataset**

# In[ ]:


#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split,KFold,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#loading dataset
df = pd.read_csv('../input/adult.csv')
df.head()


# In[ ]:


#number of features
print ("Number of features : {}".format(len(df.columns.values)))
print ("Number of sample : {}".format(df.shape[0]))


# Out of these 15 features 'Salary' is the target feature and rest all are covariates.

# In[ ]:


#let see if any cloumn has missing values
df.info()


# 

# It is clear that none on the column has any Nan values. Since number of samples in each column = total number of samples = 32561
# 
# Let's classify features into numerical and categorical:
# 
# Numerical features:
# 
# - age
# - fnlwgt
# - education.num
# - capital.gain
# - capital.loss
# - hours.per.week
# 
# Categorical features:
# 
# - workclass
# - education
# - marital.status
# - occupation
# - relationship
# - race
# - sex
# - native.country
# - salary
# 
# 
# Let's see the classes of the categorical variables

# In[ ]:


print('workclass\n',set(df.workclass))
print('\neducation\n',set(df.education))
print('\nmarital.status\n',set(df['marital.status']))
print('\noccupation\n',set(df.occupation))
print('\nrace\n',set(df.race))
print('\nrelationship',set(df.relationship))
print('\nsex',set(df.sex))
print('\nnative.country',set(df['native.country']))
print('\nsalary',set(df.income))


# As we can see that in workclass and occupation we have unknown class '?'.Now let's represent categorical data in terms of numerical dummy variables

# In[ ]:


set(df['income'].values)


# In[ ]:


df.workclass = df.workclass.map({ '?':0, 'Federal-gov':1, 'Local-gov':2, 'Never-worked':3, 'Private':4, 'Self-emp-inc':5, 'Self-emp-not-inc':6, 'State-gov':7, 'Without-pay':8})

df.income = np.where(df.income == '>50K',1,0)

df.occupation = df.occupation.map({'?':0, 'Adm-clerical':1, 'Armed-Forces':2, 'Craft-repair':3, 'Exec-managerial':4, 'Farming-fishing':5, 'Handlers-cleaners':6, 'Machine-op-inspct':7, 'Other-service':8,'Priv-house-serv':9,'Prof-specialty':10,'Protective-serv':11,'Sales':12,'Tech-support':13,'Transport-moving':14})

df['marital.status'] = df['marital.status'].map({'Divorced':0,'Married-AF-spouse':1,'Married-civ-spouse':2,'Married-spouse-absent':3,'Never-married':4,'Separated':5,'Widowed':6})

df.race = df.race.map({'Amer-Indian-Eskimo':0, 'Asian-Pac-Islander':1, 'Black':2, 'Other':3, 'White':4})

df.sex = np.where(df.sex == 'Male',1,0)

df.relationship = df.relationship.map({'Husband':0,'Not-in-family':1,'Other-relative':2,'Own-child':3,'Unmarried':4,'Wife':5})

df['native.country'] = df['native.country'].map({'?':0,'Cambodia':1,'Canada':2,'China':3,'Columbia':4,'Cuba':5,'Dominican-Republic':6,'Ecuador':7,
 'El-Salvador':8,'England':9,'France':10,'Germany':11,'Greece':12,'Guatemala':13,'Haiti':14,'Holand-Netherlands':15,'Honduras':16,
 'Hong':17,'Hungary':18,'India':19,'Iran':20,'Ireland':21,'Italy':22,'Jamaica':23,'Japan':24,'Laos':25,'Mexico':26,'Nicaragua':27,
 'Outlying-US(Guam-USVI-etc)':28,'Peru':29,'Philippines':30,'Poland':31,'Portugal':32,'Puerto-Rico':33,'Scotland':34,
 'South':35,'Taiwan':36,'Thailand':37,'Trinadad&Tobago':38,'United-States':39,'Vietnam':40,'Yugoslavia':4})

df.education = df.education.map({'10th':0,'11th':1,'12th':2,'1st-4th':3,'5th-6th':4,'7th-8th':5,'9th':6,'Assoc-acdm':7,'Assoc-voc':8,
 'Bachelors':9,'Doctorate':10,'HS-grad':11,'Masters':12,'Preschool':13,'Prof-school':14,'Some-college':15})

df.head()


# ### Feature engineering using Recursive Feature Elimination and Feature importance ranking
# 
# The Recursive Feature Elimination (RFE) method is a feature selection approach. It works by recursively removing attributes and building a model on those attributes that remain. It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute.

# In[ ]:


features = df.drop('income',axis=1)
target = df.income


# In[ ]:


#define a classifier
model = LogisticRegression()

#create RFE model to return top 3 attributes
rfe = RFE(model,3)
rfe = rfe.fit(features,target)

#summarise the selection of attributes
print('\n rfe.support:\n',rfe.support_)
print('\n rfe.ranking:\n',rfe.ranking_)
print('\n features:\n',features.columns.values)


# So the top three features as per RFE are
# 
# - educational-num
# - marital status
# - relationships
# - race
# 
# Let's use Feature Importance to identify the top 3 features
# 
# Methods that use ensembles of decision trees (like Random Forest or Extra Trees) can also compute the relative importance of each attribute. These importance values can be used to inform a feature selection process.
# 
# This recipe shows the construction of an Extra Trees ensemble of the dataset and the display of the relative feature importance.

# In[ ]:


#define and fit a ExtraTreeClassifier to the data
model = ExtraTreesClassifier()
model.fit(features,target)

#display the feature importance
print(model.feature_importances_)
print('\n',features.columns.values)


# In[ ]:


#bar plot of feature importance
values = model.feature_importances_
pos = np.arange(14) + 0.02
plt.barh(pos,values,align = 'center')
plt.title('Feature importance plot')
plt.xlabel('feature importance ')
plt.ylabel('features')
plt.yticks(np.arange(14),('age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status','occupation' ,'relationship', 'race' ,'sex', 'capital-gain', 'capital.loss','hours.per.week', 'native.country'))
plt.grid(True)


# So according to feature importance method, top 3 features are fnlwgt,age,hours-per-week.
# 
# Let's create training and testing datasets

# In[ ]:


#updating features: combining best features from both RFE and feature importance
features = features[['education.num','marital.status','relationship','race','age','fnlwgt']]

#here we have consider best features from both RFE and feature importance results

#spliting data into train and test data
X_train,X_test,y_train,y_test = train_test_split(features,target,random_state = 12)

from sklearn.neighbors import KNeighborsClassifier

k_values = np.arange(1,26)
scores = []

for i in k_values:
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_predict))

print("Accuracy for {} is {}".format(np.argmax(scores),max(scores)))

plt.plot(np.arange(1,26),scores)
plt.title('Varition of accuracy with K value,with best features from RFE and feature importance')
plt.xlabel('K values')
plt.ylabel('Accuracy')


# In[ ]:


#Let's update features with the results of RFE and evaluate how
#accuracy varies

features1 = features[['education.num','marital.status','relationship','race']]

X_train,X_test,y_train,y_test = train_test_split(features1,target,random_state = 12)

from sklearn.neighbors import KNeighborsClassifier

k_values = np.arange(1,26)
scores = []

for i in k_values:
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_predict))

print("Accuracy for {} is {}".format(np.argmax(scores),max(scores)))

plt.plot(np.arange(1,26),scores)
plt.title('Varition of accuracy with K value,with best features from RFE ')
plt.xlabel('K values')
plt.ylabel('Accuracy')


# In[ ]:


#Let's update features with the results of feature importance and evaluate how
#accuracy varies

features2 = features[['age','fnlwgt']]

X_train,X_test,y_train,y_test = train_test_split(features2,target,random_state = 12)

from sklearn.neighbors import KNeighborsClassifier

k_values = np.arange(1,26)
scores = []

for i in k_values:
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_predict))

print("Accuracy for {} is {}".format(np.argmax(scores),max(scores)))

plt.plot(np.arange(1,26),scores)
plt.title('Varition of accuracy with K value,with best features from feature importance')
plt.xlabel('K values')
plt.ylabel('Accuracy')


# What is the need of comparing all three results (ie features from both RFE and feature importance,from RFE only and from feature importance only)?
# 
# In case of KNN, model works well when the input dimensions are small, as the input dimension increases the  performance of KNN decreases, because increase in dimension weakens the most important assumption on which KNN is built, which is  that closer points belongs to same class.
# 
# For more info, you can refer :[K-Nearest Neighbors for Machine Learning](http://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/)
