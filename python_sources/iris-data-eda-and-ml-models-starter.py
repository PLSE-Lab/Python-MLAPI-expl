#!/usr/bin/env python
# coding: utf-8

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


# Importing all the libraries over here..

# In[ ]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn import svm
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.cross_validation import cross_val_score
from statistics import mean
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn import datasets


# In[ ]:


data = pd.read_csv("../input/Iris.csv")

labels = []
labels += [0 for i in range(50)]
labels += [1 for i in range(50)]
labels += [2 for i in range(50)]


# importing data in pandas dataframe and adding labels for different species
# 
# 0 for setosa 
# 
# 1 for versicolor 
# 
# 2 for verginica

# In[ ]:


# gives useful description of the data 
data.describe()


# <h2>Data Visualizations</h2>
# Plotting some graphs that tell us about the data and how the species are separated , how values are scattered and outliers in data. then finally plotting pair plots.
# These below plots show how setosa is seperate from the other 2 species for all parameters, hence the classifiers do not have much problem in identifying and predicting iris-setosa.  

# In[ ]:


sns.FacetGrid(data, hue="Species", size=5)    .map(plt.scatter, "SepalWidthCm", "PetalLengthCm")    .add_legend()


# In[ ]:


sns.FacetGrid(data, hue="Species", size=5)    .map(plt.scatter, "SepalWidthCm", "SepalLengthCm")    .add_legend()


# In[ ]:


sns.FacetGrid(data, hue="Species", size=5)    .map(plt.scatter, "PetalLengthCm", "PetalWidthCm")    .add_legend()


# In[ ]:


sns.FacetGrid(data, hue="Species", size=5)    .map(plt.scatter, "PetalLengthCm", "SepalWidthCm")    .add_legend()


# In[ ]:


sns.FacetGrid(data, hue="Species", size=5)    .map(plt.scatter, "PetalWidthCm", "SepalWidthCm")    .add_legend()


# In[ ]:


# finally a Pair Plot 
sns.pairplot(data.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")


# In[ ]:


# deleting these 2 columns from the dataframe
del data["Species"]
del data["Id"]


# In[ ]:


# spliting that data for training and testing with 80% data for training and 20% for testing
X_train,X_test,Y_train,Y_test = train_test_split(data,labels, random_state = 22, test_size = 0.2)


# In[ ]:


from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5)


# <h2>Classification Models and their Results</h2>

# In[ ]:


clfModel1 = LogisticRegression(class_weight='balanced')
# check the accuracy
# the data is split 5 times
scores = cross_val_score(clfModel1,  X_train, Y_train, cv=5, verbose = 1)
print("Accuracy for Logistic Regression : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)


# In[ ]:


clfModel2 = MultinomialNB()
# check the accuracy
# the data is split 5 times
scores = cross_val_score(clfModel2,  X_train, Y_train, cv=5, verbose = 1)
print("Accuracy for MultinomialNB : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)


# In[ ]:


clfModel3 = GaussianNB()
# check the accuracy
# the data is split 5 times
scores = cross_val_score(clfModel3,  X_train, Y_train, cv=5, verbose = 1)
print("Accuracy for Gaussian NB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)


# In[ ]:


clfModel4 = BernoulliNB()
# check the accuracy
# the data is split 5 times
scores = cross_val_score(clfModel4,  X_train, Y_train, cv=5, verbose = 1)
print("Accuracy for BernoulliNB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)


# In[ ]:


clfModel5 = tree.DecisionTreeClassifier(criterion='gini')
# check the accuracy
# the data is split 5 times
scores = cross_val_score(clfModel5,  X_train, Y_train, cv=5, verbose = 1)
print("Accuracy for Decision Tree : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)


# In[ ]:


clfModel6 = svm.SVC()
# check the accuracy
# the data is split 5 times
scores = cross_val_score(clfModel6,  X_train, Y_train, cv=5, verbose = 1)
print("Accuracy for SVM: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)


# In[ ]:


from sklearn.linear_model import SGDClassifier
clfModel7 = SGDClassifier()
# check the accuracy
# the data is split 5 times
scores = cross_val_score(clfModel7,  X_train, Y_train, cv=5, verbose = 1)
print("Accuracy for SGD Classifier : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
# Create KNeighbors classifier object model 
clf8 = KNeighborsClassifier(n_neighbors=6) # default value for n_neighbors is 5
# check the accuracy
# the data is split 5 times
scores = cross_val_score(clf8,  X_train, Y_train, cv=5, verbose = 1)
print("Accuracy for K Neighbours : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
# Create Random Forest object
clf9 = RandomForestClassifier()
# check the accuracy
# the data is split 5 times
scores = cross_val_score(clf9,  X_train, Y_train, cv=5, verbose = 1)
print("Accuracy for Random Forest Clasifier: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
warnings.filterwarnings("ignore")
clf10 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# check the accuracy
# the data is split 5 times
scores = cross_val_score(clf10,  X_train, Y_train, cv=5, verbose = 1)
print("Accuracy for Gradient Boosting Classifier: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)


# <h3>Scaling the features</h3>
# The accuracy score for Bernoulli NB and K Neighbors Classifiers increases after scaling the features  but has either no effect or decreases the accuracy for other classifiers (for this dataset only).
# Please comment if you know the reason...

# In[ ]:


# scaling features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


clfModel4 = BernoulliNB()
# check the accuracy
# the data is split 5 times
scores = cross_val_score(clfModel4, X_train, Y_train, cv=5, verbose = 1)
print("Accuracy for BernoulliNB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
# Create KNeighbors classifier object model 
clf8 = KNeighborsClassifier(n_neighbors=6) # default value for n_neighbors is 5
# check the accuracy
# the data is split 5 times
scores = cross_val_score(clf8, X_train, Y_train, cv=5, verbose = 1)
print("Accuracy for K Neighbours : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)


# <h3>Verification Against Testing data</h3>
# As SVM has the highest accuracy score we test it against testing data to print accuracy and plot heatmaps for confusion matrix

# In[ ]:


classifierModel = svm.SVC()
classifierModel.fit(X_train,Y_train)
predictions = classifierModel.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[ ]:


print(accuracy_score(Y_test,predictions))


# In[ ]:


# precision, recall, f1-score for setosa is perfect 
# because it is easily seperable from the other 2 species as seen in the plots 
print(classification_report(Y_test,predictions))


# In[ ]:


conf = confusion_matrix(Y_test,predictions)


# In[ ]:


label = ["Setosa","Versicolor","Virginica"]
sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label, cmap = 'YlGnBu')

