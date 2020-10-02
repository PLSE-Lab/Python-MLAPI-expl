#!/usr/bin/env python
# coding: utf-8

# ### INTRODUCTION
# In this kernel we learn supervised learning algorithms.
# 
#     1. Logistic Regression
#     2. K-Nearest Neighbour (KNN) 
#     3. Support Vector Machine (SVM)
#     4. Naive Bayes
#     5. Decision Tree
#     6. Random Forest
#     7. Confusion Matrix
#     8. Conclusion

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# > Data Information
# Our data is about Intel and AMD CPU comparasion.

# In[ ]:


# read data from csv
data = pd.read_csv("../input/amd-vs-intel/AMDvIntel.csv")
data.info()


# In[ ]:


data.head(10)


# In[ ]:


# We drop "name" and "price" columns. Because these features does not important for classification.
data = data.drop(['Name','Price'],axis=1)
data.head(10)


# In[ ]:


# Y axis is comparison class. Which is Intel or AMD.
y = data['IorA'].values
y = y.reshape(-1,1)
# x_data is all other things without Y axis in data.
x_data = data.drop(['IorA'],axis = 1)
print(x_data)


# In[ ]:


# NORMALIZATION
# We should normalize our dataset for right result. In this dataset we do not need necessary but we usually need normalize datasets. 
# Because maybe in a dataset one column include very high numbers (100000000, 154000000 etc.) and other column include very little number (0,01  0,005 etc.).
# For right result we need normalize datasets.
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x.head(10)


# In[ ]:


# now we organize our test and train data.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)


# >     1. LOGISTIC REGRESSION

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs')  # for FutureWarning Error we need to write solver='lbfgs'
lr.fit(x_train, y_train.ravel())
print("Logistic Regression test accuracy (score): ", lr.score(x_test, y_test))


# >     2. K-Nearest Neighbour (KNN) 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
K = 5   # neighbors number
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(x_train, y_train.ravel())
print("When K = {} neighnors , KNN test accuracy: {}".format(K, knn.score(x_test, y_test)))
print("When K = {} neighnors , KNN train accuracy: {}".format(K, knn.score(x_train, y_train)))


# In[ ]:


# Now we find the best K (hyperparameter, number of neighbors) for our data.
ran = np.arange(1,30)
train_list = []
test_list = []
for i,each in enumerate(ran):
    knn = KNeighborsClassifier(n_neighbors=each)
    knn.fit(x_train, y_train.ravel())
    test_list.append(knn.score(x_test, y_test))
    train_list.append(knn.score(x_train, y_train))


# In[ ]:


# VISUALIZATION RESULT
plt.figure(figsize=[15,10])
plt.plot(ran,test_list,label='Test Score')
plt.plot(ran,train_list,label = 'Train Score')
plt.xlabel('Number of Neighbers')
plt.ylabel('Scores/Accuracy')
plt.xticks(ran)
plt.legend()
print("Best test score is {} and K = {}".format(np.max(test_list), test_list.index(np.max(test_list))+1))
print("Best train score is {} and K = {}".format(np.max(train_list), train_list.index(np.max(train_list))+1))


#     3. Support Vector Machine (SVM)

# In[ ]:


# Third algorithm is Support Vector Machine algorithm. 
from sklearn.svm import SVC
svm = SVC(random_state=42,gamma='scale')   # for FutureWarning, we need to write gamma='auto' or gamma='scale'
# scale accuracy is 0.9230769230769231
# auto  accuracy is 0.8461538461538461
svm.fit(x_train, y_train.ravel())
print("SVM test accuracy: {}".format(svm.score(x_test, y_test)))


#     4. Naive Bayes

# In[ ]:


# import the algorithm
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train.ravel())
print("Naive Bayes test accuracy: ", nb.score(x_test, y_test))


#     5. Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train.ravel())
print("Decision Tree Algorithm test accuracy: ", dtree.score(x_test, y_test))


#     6. Random Forest 

# In this algorithm, we have a hyperparameter N. Train data is divided into N pieces. We get N pieces subsample (subtrain) data. Then run every subtrain data in Decision Tree Algorithm. 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=3)    # n_estimators is number of Decision Tree in this algorithm.
rf.fit(x_train, y_train.ravel())
print("Random Forest Algorithm test accuracy: ",rf.score(x_test, y_test))


# We finished classification algorithms. Lets we check which class more accurate. This means how many Intel CPU the algorithms has marked as AMD CPU.

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
rf = RandomForestClassifier(n_estimators= 10, random_state=3)
rf.fit(x_train, y_train.ravel())
y_predicted = rf.predict(x_test)
matrix = confusion_matrix(y_test, y_predicted)
print("Confusion Matrix: \n",matrix)
print("Classification Report: \n", classification_report(y_test, y_predicted))


# ### CONCLUSION
#  In this kernel we learn classification algorithms with IntelvsAMD CPU dataset.
#  If you can find any bug or mistake, please send me feedback. 
#  Thank you for visit.
