#!/usr/bin/env python
# coding: utf-8

#  **Hey there!**
# 
# This notebook will be dealing with comparing some basic Machine Learning classification algorithms based on their accuracy. So, lets get started! 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Getting to know your data well be of great help...Lets dive into the dataset and see what are the columns we will be dealing with.

# In[ ]:


df = pd.read_csv("../input/diabetes.csv")
df.head()


# Hmm..Looks like it is a supervised learning problem as we have predictor variables and one dependent variable(Outcome).

# To get the total number of classes we have in the outcome column and count of each class, countplot might help us...

# In[ ]:


import seaborn as sns
sns.countplot(df.Outcome)


# Yeah...We have 2 classes(0,1) in which our data is classified..
# 
# Now lets seeHow is the distribution of my data along diffrent rows?

# In[ ]:


df.describe()


# Looks like the data is clean...But..Wait a minute..
# Is the data really clean?
# 
# What about the values that are "0"?...Insulin of a person cant be 0 right...
# Obviously!
# 
# Lets clean it up...To do it in a easy way, replace all zeros from all predictor variables(except Pregnancies) by NaN...This represents that we have got a missing value...Fix it!!

# In[ ]:


df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0,np.NaN)
df.head() #Replaced all 0 values by NaN so it is easy to clean the data


# We can replace missing values by using various methods like mean,median,mode or the number of your choice...We'll do that with mean...You can rather try diffrent methods...And let me know if something works better...(Signs that I'm Lazy:-|)

# In[ ]:


df.fillna(df.mean(), inplace = True) #Filled Mising values with Mean
df.isnull().sum()


# In[ ]:


df.describe()


# **Feature Selection**
# 
# To increase the efficiency of the model, we can eliminate some features. This is done by knowing the importance if a particular feature...
# 
# Lets try to find correlation between the features of our dataset..More the features are correlated, we can eliminate one of them...Heatmap folks!!

# In[ ]:


import matplotlib.pyplot as plt
sns.heatmap(df.corr(),annot=True)
fig = plt.gcf()
fig.set_size_inches(8,8)


# Umm.... I'm not getting the heatmap into my head...Lets try another way....Use random forest to get the importances of feature...

# In[ ]:


#feature selection
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
x=df[df.columns[:8]]
y=df.Outcome
clf.fit(x,y)
feature_imp = pd.DataFrame(clf.feature_importances_,index=x.columns)
feature_imp.sort_values(by = 0 , ascending = False)


# So as we can see... The first 4 features displayed maybe important for us...We might neglect the rest...
# 
# Now get your tools ready to sculpt diffrent models... 

# In[ ]:


from sklearn.cross_validation import train_test_split

features = df[["Glucose",'BMI','Age','DiabetesPedigreeFunction']]
labels = df.Outcome
features.head()


# These are five models we will be seeing...Split the work and get your hands dirty to code the hell up!!
# 
# Note that we have used stratification while splitting so that our data gets splitted in proportion with respect to Outcome column.

# In[ ]:


features_train,features_test,labels_train,labels_test = train_test_split(features,labels,stratify=df.Outcome,test_size=0.4)


# I'll like to start with **Decision Trees**

# In[ ]:


#DTClassifier
from sklearn.tree import DecisionTreeClassifier 
dtclf = DecisionTreeClassifier()
dtclf.fit(features_train,labels_train)
dtclf.score(features_test,labels_test)


# Looks fine...Lets see what Support Vector Machine shows
# **SVM**
# Try to change kernel parameter to other than "linear"..

# In[ ]:


#SVM
from sklearn import svm
clf = svm.SVC(kernel="linear")
clf.fit(features_train,labels_train)
clf.score(features_test,labels_test)


# Will Gaussian **Naive Bayes** do well?

# In[ ]:


#Naive Bayes Classifier
from sklearn import naive_bayes
nbclf = naive_bayes.GaussianNB()
nbclf.fit(features_train,labels_train)
nbclf.score(features_test,labels_test)


# Good...Better than Decision Tree...**K Neighbor?**

# In[ ]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
knnclf = KNeighborsClassifier(n_neighbors=2)
knnclf.fit(features_train,labels_train)
print(knnclf.score(features_test,labels_test))
    
 


# Cool...Now the last one remaining...Lets do **Logistic Regression**

# In[ ]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression()
clf1.fit(features_train,labels_train)
clf1.score(features_test,labels_test)


# You know what?...I will like to see all these algo's accuracy at one place...Also, lets code everything up in single cell itself..

# In[ ]:


algos = ["Support Vector Machine","Decision Tree","Logistic Regression","K Nearest Neighbor","Naive Bayes"]
clfs = [svm.SVC(kernel="linear"),DecisionTreeClassifier(),LogisticRegression(),KNeighborsClassifier(n_neighbors=2),naive_bayes.GaussianNB()]
result = []

for clff in clfs:
    clff.fit(features_train,labels_train)
    acc = clff.score(features_test,labels_test)
    result.append(acc)
result_df = pd.DataFrame(result,index=algos)
result_df.columns=["Accuracy"]
result_df.sort_values(by="Accuracy",ascending=False)


# Great...Looks neat..Also I can see that which models are looking well in terms of accuracy...But did u guys noticed?..We have been working on same training and testing set from a long time..We need to ry diffrent combinations of training and testing sets...Lets bring Cross Validation into picture to help me out! 

# In[ ]:


#Cross Validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold =KFold(n_splits=10)


# By using cross validation, we will be splitting our dataset into 10 equal parts...We keep one part for testing our algorithm and we train models on the rest...Now these parts that we divided the dataset into, keeps interchanging to form diffrent combinations of training and testing data...We get difffrent accuracy score for each combination...This is done by cross_val_score()..It gives us the list of diffrent accuracies...Now by taking the mean of this score, we can find the general accuracy of our model...
# This gives a generalised output..

# In[ ]:


algos = ["Support Vector Machine","Decision Tree","Logistic Regression","K Nearest Neighbor","Naive Bayes"]
clfs = [svm.SVC(kernel="linear"),DecisionTreeClassifier(),LogisticRegression(),KNeighborsClassifier(n_neighbors=2),naive_bayes.GaussianNB()]
cv_results=[]
for classifiers in clfs:
    cv_score = cross_val_score(classifiers,features,labels,cv=kfold,scoring="accuracy")
    cv_results.append(cv_score.mean())
cv_mean = pd.DataFrame(cv_results,index=algos)
cv_mean.columns=["Accuracy"]
cv_mean.sort_values(by="Accuracy",ascending=False)


# And now....We can see the accuracy changed a bit this time...It is because we have done cross validation and trained and tested the algorithms on diffrent combinations of data....
# 
# From the above output, it is clear that for this dataset, SVM, Logistic Regression and Naive Bayes works better....
# 
# So to revise what we did above:
# 
# 1.Cleaned our data by replacing missing values(for this data, we considered 0 as a missing value).
# 2.Feature Selection using Random Forest Classifier.
# 3.Split the data into training and testing sets using train_test_split.
# 4.Trained five diffrent classification algorithms and found their accuracies.
# 5.Cross validation by splitting data into 10 splits to get the generalised accuracy for each algorithm.
