#!/usr/bin/env python
# coding: utf-8

# ** On this Kernel, I will make a classification about gender voice **
# 
# My steps are:
# *  Firstly I will import libraries
# *  Loading Data
# *  Data Visualization
# *  One Hot Encoding
# *  Normalization
# *  Train - Test Split
# *  Finally, Predictions

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Load the Data**

# In[ ]:


data = pd.read_csv('../input/voice.csv')
data.head()


# In[ ]:


data.info()


# **Data Visualization**

# In[ ]:


seaborn.pairplot(data[['meanfreq', 'Q25', 'Q75', 'skew', 'centroid', 'label']],hue='label', size=3)


# **If you well look at the label column you can see that all genders are sequential. Firstly we have to make shuffle on this data.**

# In[ ]:


data = data.sample(frac=1)

data.head()


# **Categorical Encoding**

# In[ ]:


data['label'] = data['label'].map({'male':1,'female':0})


# In[ ]:


X = data.loc[:, data.columns != 'label']
y = data.loc[:,'label']


# **Normalization**

# In[ ]:


X = (X - np.min(X))/(np.max(X)-np.min(X)).values

X.head()


# **Train - Test Split**

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)


# **Prediction with Logistic Regression**

# In[ ]:


from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)
print("test accuracy: {} ".format(logreg.fit(X_train, y_train).score(X_test, y_test)))
print("train accuracy: {} ".format(logreg.fit(X_train, y_train).score(X_train, y_train)))


# **Prediction with Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

ran_for = RandomForestClassifier(n_estimators=250, max_depth=15, random_state=42)
print("test accuracy: {} ".format(ran_for.fit(X_train, y_train).score(X_test, y_test)))
print("train accuracy: {} ".format(ran_for.fit(X_train, y_train).score(X_train, y_train)))


# **Prediction with KNN**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
print("test accuracy: {} ".format(knn.fit(X_train, y_train).score(X_test, y_test)))
print("train accuracy: {} ".format(knn.fit(X_train, y_train).score(X_train, y_train)))


# **Finding the Best k Values**

# In[ ]:


score = []

for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(X_train, y_train)
    score.append(knn2.score(X_test, y_test))
    
plt.plot(range(1,20), score)
plt.xlabel('k values')
plt.ylabel('sccuracy')
plt.show()


# **I guess it looks like 3 or 4. Therefore I chose 3**

# **Prediction with SVM**

# In[ ]:


from sklearn.svm import SVC

svm = SVC(random_state=42)
print("test accuracy: {} ".format(svm.fit(X_train, y_train).score(X_test, y_test)))
print("train accuracy: {} ".format(svm.fit(X_train, y_train).score(X_train, y_train)))


# **Prediction with Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
print("test accuracy: {} ".format(nb.fit(X_train, y_train).score(X_test, y_test)))
print("train accuracy: {} ".format(nb.fit(X_train, y_train).score(X_train, y_train)))


# **Prediction with Decision Tree**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
print("test accuracy: {} ".format(dt.fit(X_train, y_train).score(X_test, y_test)))
print("train accuracy: {} ".format(dt.fit(X_train, y_train).score(X_train, y_train)))


# So, if you think about something please share with me. 
# 
# Thank you :)

# In[ ]:




