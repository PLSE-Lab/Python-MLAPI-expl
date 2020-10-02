#!/usr/bin/env python
# coding: utf-8

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
import matplotlib.pyplot as plt
from numpy import random
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # A Report on Iris Classification
# 
# The purpose of this notebook is to create a simple classifier on Iris Dataset, to make things more fun here are some challenges to play with Iris Dataset including:
# 
# 1. Create the simplest classification model from scratch (with acceptable accuracy, more or less 80%)
# 2. Find the most accurate classification model with as little of overfitting possible
# 3. Create a human friendly knowledge mined from classification model (minimal accuracy of 90%)

# In[ ]:


dataset = pd.read_csv("../input/iris/Iris.csv").drop("Id", axis = 1 ,)


#independent feature
indFeat = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]


# In[ ]:


dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


dataset.describe()


# In[ ]:


dataset["Species"].value_counts()


# The cells above show summary of the data set, we can conclude that the data is made up of 4 independent features that that are continous those features are :
# 
# 1. Sepal Length
# 2. Sepal Width
# 3. Petal Length
# 4. Petal Width
# 
# and one categorical dependent feature
# 
# The distribution of data are also pretty close where mostly the STD Deviations are under 2, with ranges of each independent in the single digit,this means we might be able to process data without much hassle such as normalization.
# 
# The class distribution also pretty much equals for each class, a very beatiful data

# # Understanding the Data

# In[ ]:


sns.pairplot(dataset, hue = "Species")
plt.show()


# From the pairplot above we can observe that petals feature have a better separation than sepal features to better see the comparison between Sepal's and Petal's data distribution we will observe the following graph

# In[ ]:


plt.figure(figsize = (15,10))
sns.scatterplot(dataset["SepalLengthCm"], dataset["SepalWidthCm"], hue = dataset["Species"], s = 100)
plt.show()


# In[ ]:


plt.figure(figsize = (15,10))
sns.scatterplot(dataset["PetalLengthCm"], dataset["PetalWidthCm"], hue = dataset["Species"], s = 100)
plt.show()


# Using this knowledge we could try to implement Principal Component Analysis to reduce the dimension of Petal into one dimension only feature and use threshold to classify iris, compromised accuracy but it would be the simplest way to model a classifier

# # Creating Simple Classification Model (Treshold + Probability Based Classifier)

# In[ ]:


#Dimensionality reduction

from sklearn.decomposition import PCA

pca = PCA(n_components = 1)
dataDecompose = pd.DataFrame(pca.fit_transform(dataset[["PetalLengthCm", "PetalWidthCm"]]))
dataDecompose["Species"] = dataset["Species"]


# In[ ]:


plt.figure(figsize = (20,10))
sns.pairplot(dataDecompose, hue = "Species", height = 5)


# # On Simple Classifier:
# 
# to make the simples classifier we can 100% predict Iris-setosa using simple thresholding which will give 30% accuracy already, and we can also use thresholding on both versicolor and virginica for certain value range with guaranteed success, the rest of them will do with simple naive bayes and or probability theorem
# 
# let's combine tresholding and simple probability

# In[ ]:


def NaiveClassifier(data, input = 0):
    
    if input <= data[data["Species"] == "Iris-setosa"].max()[0] and input >= data[data["Species"] == "Iris-setosa"].min()[0]:
        return "Iris-setosa"
    
    elif input < data[data["Species"] == "Iris-virginica"].min()[0]:
        return "Iris-versicolor"
    
    elif input > data[data["Species"] == "Iris-versicolor"].max()[0]:
        return "Iris-virginica"
    
    else:
        return random.choice(["Iris-versicolor", "Iris-virginica"])
    


# In[ ]:


sumasi = 0
for x in range(5):
    test = []
    for data in dataDecompose[0]:
        test.append(NaiveClassifier(dataDecompose, data))
    
    sumasi +=((pd.Series(test) == dataDecompose["Species"]).astype(int)).sum()


# In[ ]:


akurasi = sumasi / (150*5)

print("Accuration on Simple Tresholding and Random :", akurasi)


# The model could work even better with weighted probability based on input, as we can see, there are certain value where the probability of it being Virginica is higher than versicolor, but hey it works like a charm, and we are looking for simple!
# 
# so simple you got!

# # Modelling an accurate classifier

# To prove that a model is not overfitted we can use the k-fold validation as an evaluation method and take the model with highest accuracy, let's use k = 5 in this instance.
# 
# The Classifier used in this datasets are as follows and the hypothesis of choosing it: (even though Iris Datasets are a very beautiful dataset which almost every alghoritm will give a great result)
# 
# - Using Gaussian naive Bayes classifier because the data distribution is gaussian distribution
# - kNN becausee of it's very simple alghoritm and having a very balanced class ratio also helps this algorithm a lot

# In[ ]:


# data partitioning for k=5 fold

def partitioning(dataset, k = 5):
    grouped = list(dataset.groupby("Species"))
    test = []
    train = []
    for i in range (k):
        temp_concat = ""
        temp_concat = pd.concat([grouped[0][1][i*10:(i+1)*10],grouped[1][1][i*10:(i+1)*10],grouped[2][1][i*10:(i+1)*10]])
        test.append(temp_concat)
        train.append(pd.DataFrame(dataset[~(dataset.isin(test[i]))].dropna(how = "all")))
    return test, train

test,train = partitioning(dataset)


# In[ ]:


from sklearn.naive_bayes import GaussianNB

gaussNB = GaussianNB()

acc = 0
for i in range(5):
    NB = gaussNB.fit( X = train[i][indFeat], y = train[i]["Species"])
    pred = NB.predict(test[i][indFeat])
    acc += (pred == test[i]["Species"]).sum()

acc = acc/150

print("Naive Bayes Accuracy =",acc)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors = 7, metric = "euclidean")

for i in range(5):
    neigh = kNN.fit(X = train[i][indFeat], y = train[i]["Species"])
    pred = neigh.predict(test[i][indFeat])
    acc += (pred == test[i]["Species"]).sum()

acc = acc/150
print("kNN Accuracy=",acc)


# In[ ]:


from sklearn import svm
VecMachine = svm.SVC(gamma = "auto")
for i in range(5):
    neigh = VecMachine.fit(X = train[i][indFeat], y = train[i]["Species"])
    pred = VecMachine.predict(test[i][indFeat])
    acc += (pred == test[i]["Species"]).sum()

acc = acc/150
print("SVM Accuracy=",acc)


# * It seems that SVM is accurate enough on classifying the Iris Dataset

# # Modelling a human friendly classifier

# One of the most human friendly alghoritm is no other than decision tree, so let's try to create a decision tree classifier and print out the tree so we can gain knowledge on how to differentiate iris species in the wild, how Exciting!

# In[ ]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz


tree_model = DecisionTreeClassifier()
tree_model.fit(X = dataset[indFeat], y=dataset['Species'])

print("Decision Tree Accuracy =",(tree_model.predict(dataset[indFeat])
       ==dataset["Species"]).sum()/len(dataset))

dec = export_graphviz(tree_model)

from IPython.display import display

display(graphviz.Source(dec))


# In[ ]:





# In[ ]:




