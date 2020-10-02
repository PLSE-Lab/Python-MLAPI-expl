#!/usr/bin/env python
# coding: utf-8

# ![hkkjkjkjnkjnkjnk](https://upload.wikimedia.org/wikipedia/commons/thumb/d/db/Titanic-Cobh-Harbour-1912.JPG/1600px-Titanic-Cobh-Harbour-1912.JPG)
# 
# **TITANIC**
# 
# In this notebook, we build logistic regression, decision tree, and polynomial features to predict survival on the titanic. Before the model, we build Variance Threshold, SelectKBest,chi2 and SelectFrom which help to choose which indepedent variable best for prediction on titanic dataset.
# 
# 
# 
# 
# 

# In[13]:


#importing library
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd


# In[14]:


# read train datasets
train = pd.read_csv('../input/train.csv')


# In[15]:


#check the train dataset
train.head()


# In[16]:


# check the tail of dataset
train.tail()


# In[17]:


# Display all informations
train.info()


# In[18]:


# to get information three quartiles, mean, count, minimum and maximum values and the standard deviation.
train.describe()


# In[19]:


#heatmap for train dataset
import matplotlib.pyplot as plt
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[ ]:


#now, we are checking start with a pairplot, and check for missing values
sns.heatmap(train.isnull(),cbar=False)


# As we see above, Age  and Cabin column have lots of missing values. But, age column is important data for explorary data analysis. Thus, we get the median of data. 

# In[ ]:


#Data Cleaning and Data Drop Process
train['Fare'] = train['Fare'].fillna(train['Fare'].dropna().median())
train['Age'] = train['Age'].fillna(train['Age'].dropna().median())
# Change to categoric column to numeric
train.loc[train['Sex']=='male','Sex']=0
train.loc[train['Sex']=='female','Sex']=1
# instead of nan values 
train['Embarked']=train['Embarked'].fillna('S') 
# Change to categoric column to numeric
train.loc[train['Embarked']=='S','Embarked']=0
train.loc[train['Embarked']=='C','Embarked']=1
train.loc[train['Embarked']=='Q','Embarked']=2


# In[21]:


#Drop unnecessary columns
drop_elements = ['Name','Cabin','Ticket']
train = train.drop(drop_elements, axis=1)


# In[32]:


# Now, data is clean and read to a analyze
sns.heatmap(train.isnull(),cbar=False)


# In[31]:


# how many people survived or not... %60 percent died %40 percent survived
fig = plt.figure(figsize=(18,6))
train.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()


# In[30]:


#Age with survived
plt.scatter(train.Survived, train.Age, alpha=0.1)
plt.title("Age with Survived")
plt.show()


# In[29]:


#Count the pessenger class
fig = plt.figure(figsize=(18,6))
train.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()


# In[ ]:


#Men Survived
#%20 survived
#train.Survived[train.Sex=='male'].value_counts(normalize=True).plot(kind='bar', alpha=0.5)
#plt.title("Men survived")
#plt.show()


# In[ ]:


#
#female_color='pink'
#train.Survived[train.Sex=='female'].value_counts(normalize=True).plot(kind='bar', alpha=0.5,color=female_color)
#plt.title("Women survived")
#plt.show()


# In[35]:


#Women Men together graph
female_color='pink'
train.Sex[train.Survived==1].value_counts(normalize=True).plot(kind='bar', alpha=0.5,color=[female_color,'b'])
plt.title("Sex of survived")
plt.show()


# In[36]:


# which columns we have
train.columns


# how to retrieve the 5 right informative features in the Titanic #1 dataset.

# **Variance Threshold**
# 
# This feature selection algorithm looks only at the features (X). We set threshold 0.1 which lower than this threshold will be removed. 
# 

# In[34]:


from sklearn.feature_selection import VarianceThreshold

mdlsel = VarianceThreshold(threshold=0.1)
mdlsel.fit(train)
ix = mdlsel.get_support()
#data1 = mdlsel.transform(train) 
data1 = pd.DataFrame(mdlsel.transform(train), columns = train.columns.values[ix])
data1.head()


# **Select K Best**
# 
# Select features according to the k highest scores.

# In[37]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = train.drop("Survived",axis=1)
y = train["Survived"]

mdlsel = SelectKBest(chi2, k=5) # en iyi feature lari alma
mdlsel.fit(X,y)
ix = mdlsel.get_support() # false iyi true lari alacak..
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix]) # en iyi leri aldi... 7 tane...
data2.head(n=5)


# **Select From Model for Logistic Regression**
# 

# In[38]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

X = train.drop("Survived",axis=1)
y = train["Survived"]

# Linear Model
linmdl = LogisticRegression()
linmdl.fit(X,y)
mdl = SelectFromModel(linmdl,prefit=True)
ix = mdl.get_support() # false iyi true lari alacak..
data3 = pd.DataFrame(mdl.transform(X), columns = X.columns.values[ix]) # sadece 5 tane aldi...
data3.head(n=5)


# **Recursive Feature Selection**
# 
# Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. 
# 
# Source: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html 

# In[39]:


#last feature selection
from sklearn.feature_selection import RFE

mdl = RFE(linmdl,n_features_to_select=7)
mdl.fit(X,y)
ix = mdl.get_support() # false iyi true lari alacak..
data4 = pd.DataFrame(mdl.transform(X), columns = X.columns.values[ix]) # sadece 5 tane aldi...
data4.head(n=5)


# **Logistic Regression**

# In[79]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


#classification - > f1
#regresyon -> mse error 


target = train['Survived']
features = train[['Pclass','Sex','SibSp','Parch','Age']]
#features = train[['Pclass','Sex']].values
#print(features.Pclass)

#plt.scatter(features.Pclass,features.Age,c='red')
#Build test and training test
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)


classifier = linear_model.LogisticRegression()
classifier_ = classifier.fit(X_train,y_train)
target_predict=classifier_.predict(X_test)


print("Logistic Regression Score: ",accuracy_score(y_test,target_predict))

from sklearn.metrics import mean_squared_error, r2_score

print ("MSE    :",mean_squared_error(y_test,target_predict))

print ("R2     :",r2_score(y_test,target_predict))


# **Polynomial Features**
# 
# 

# In[41]:


from sklearn import preprocessing
#Build test and training test
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)
poly = preprocessing.PolynomialFeatures(degree=2,include_bias=False)
poly_features = poly.fit_transform(features)

classifier_ = classifier.fit(X_train,y_train)
print("Polynomial Features: ",accuracy_score(y_test,target_predict))


# **Decision Tree**

# In[42]:


from sklearn import tree
from sklearn.metrics import accuracy_score

target = train['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = train[data_features_names].values

#Build test and training test
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)

#features = train[['Pclass','Sex']] 


decision_tree = tree.DecisionTreeClassifier(random_state=1)


decision_tree_ = decision_tree.fit(X_train,y_train)
target_predict=decision_tree_.predict(X_test)

print("Decision tree score: ",accuracy_score(y_test,target_predict))

from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :",mean_squared_error(y_test,target_predict))
#ne kadar dusuk mse o kadar iyi...
print ("R2     :",r2_score(y_test,target_predict))


# In[28]:


import graphviz
generalized_tree = tree.DecisionTreeClassifier(
        random_state = 1,
        max_depth = 5,
        min_samples_split=2
)

generalized_tree_ = generalized_tree.fit(features,target)

print("Generalized tree score: ", generalized_tree_.score(features,target))




dot_data=tree.export_graphviz(generalized_tree_,feature_names=data_features_names,out_file=None)
graph = graphviz.Source(dot_data)
graph


# **CONCLUSION**
# 
# 
# 
# 
# 
# 
# 
# 
