#!/usr/bin/env python
# coding: utf-8

# ![breathtaking-online-shopping-statistics-you-never-knew-1250x600.png](attachment:breathtaking-online-shopping-statistics-you-never-knew-1250x600.png)
# 
# This notebook looks at the features that comprise the dataset and define the revenue of the online store based on various features.
# 
# We shall see how the data is spread in the dataset with visualizations and then procedd to determine the features that most affect the Revenue.
# 
# Once the features are determined we shall use these features to train the following models and determine the best model for classification.
# 
# 1. Logistical Regression Classifier
# 2. KNN Classifier
# 3. SVC
# 4. Naive Bayes
# 5. Random Forest Classifier
# 
# *Now, lets begin!*

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#import additional libraries
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#read the dataset
df = pd.read_csv("/kaggle/input/online-shoppers-intention/online_shoppers_intention.csv")


# In[ ]:


#first few rows of the dataframe
df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


#drop the null values
df.dropna(axis = 0, how = "any", inplace = True)


# In[ ]:


#check for null values again
df.isnull().sum()


# In[ ]:


#describe the dataframe
df.describe().T


# Now, lets visualize the data.

# In[ ]:


#Count of Revenue
sns.countplot(df.Revenue, palette = 'seismic_r')


# In[ ]:


#pie chart for revenue
labels = ['False', 'True']
plt.title("Revenue")
plt.pie(df.Revenue.value_counts(), labels = labels, autopct = '%.4f%%')
plt.legend()


# In[ ]:


#different users
sns.countplot(x = df.VisitorType)


# In[ ]:


#different regions
sns.countplot(df.Region)


# In[ ]:


#Operating system wrt Revenue
sns.countplot(df.OperatingSystems, hue = df.Revenue)


# In[ ]:


#Revenue for each region
sns.countplot(df.Region, hue = df.Revenue)


# In[ ]:


#Revenue with respect to Weekend
sns.countplot(df.Weekend, hue = df.Revenue)


# Now lets check the correlation of the features with respect to Revenue to know the best features to select for modeling.

# In[ ]:


plt.figure(figsize = (10,10))
sns.heatmap(df.corr())


# From the figure it can be determined that the features are all the columns from the BounceRates column.

# In[ ]:


#check datatypes
df.dtypes


# In[ ]:


#split into X and y
X = df.iloc[:,6:-1].values
y = df.iloc[:,-1].values


# In[ ]:


#label encode the objects
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
y = le.fit_transform(y)
X[:,4] = le.fit_transform(X[:,4])
X[:,9] = le.fit_transform(X[:,9])
X[:,10] = le.fit_transform(X[:,10])


# In[ ]:


#split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Lets begin the Classification

# 1. **LOGISTIC REGRESSION**

# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver = 'lbfgs')
classifier.fit(X_train, y_train)


# In[ ]:


#predict the values
y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# Accuracy of Logistic Classifier is 88%

# 2.** KNN Classifier**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(metric = 'minkowski', p = 2, n_neighbors = 5)
classifier.fit(X_train, y_train)


# In[ ]:


#predict the values
y_pred = classifier.predict(X_test)


# In[ ]:


#print report
print(classification_report(y_test, y_pred))


# The accuracy of the KNN model is 88% as well.

# 3. **Support Vector Classifier**

# In[ ]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


# In[ ]:


#predict the values
y_pred = classifier.predict(X_test)


# In[ ]:


#print report
print(classification_report(y_test, y_pred))


# The accuracy is 89%.
# 
# Just 1 more than the Logistic and KNN Classifiers!

# 4. **Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[ ]:


#predict the values
y_pred = classifier.predict(X_test)


# In[ ]:


#print report
print(classification_report(y_test, y_pred))


# Naive Bayes surprisingly has a lower accuracy of 78%

# 5. **Random Forest Classifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[ ]:


#predict the values
y_pred = classifier.predict(X_test)


# In[ ]:


#print report
print(classification_report(y_test, y_pred))


# The accuracy of the Random Forest Classifier is also 89%. 
# 
# Thus by comparing the Precision and the f1 - score we can chose the Random Forest Classifier 
# 
# (To be honest, both are equally good for the given dataset.)
