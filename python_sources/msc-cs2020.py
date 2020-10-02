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


# In[ ]:


import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
dataset = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
dataset.head(12)
print(dataset.shape)
print(dataset.describe())
#class distribution
print(dataset.groupby('species').size())
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#pyplot.show()
# histograms
dataset.hist()
pyplot.show()


# In[ ]:


# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()


# In[ ]:


# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y= array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)


# In[ ]:


from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,Y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_validation)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_validation, y_pred))


# In[ ]:


#from IPython.display import Image  
from sklearn import tree
tree.plot_tree(clf)  


# In[ ]:


#now we try mlp classifier
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(3,3), max_iter=3000,activation = 'relu',solver='adam',random_state=1)
classifier=classifier.fit(X_train, Y_train)
#Using the trained network to predict

#Predicting y for X_val
Y_pred = classifier.predict(X_validation)

#Calculating the accuracy of predictions
#Importing Confusion Matrix
from sklearn.metrics import confusion_matrix
#Comparing the predictions against the actual observations in y_val
cm = confusion_matrix(Y_pred, Y_validation)

#Printing the accuracy
print("Accuracy of MLPClassifier : ",metrics.accuracy_score(Y_validation, Y_pred))
print(cm)


# In[ ]:



 #Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = clf.predict(X_validation)
cm = confusion_matrix(Y_pred, Y_validation)

#Printing the accuracy
print("Accuracy of SVM Classifier : ",metrics.accuracy_score(Y_validation, Y_pred))
print(cm)


# In[ ]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = classifier.coefs_[0].min(), classifier.coefs_[0].max()
for coef, ax in zip(classifier.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(2, 2), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())


# In[ ]:


# importing the required module 
import matplotlib.pyplot as plt 

 


# In[ ]:


import pandas as pd
plt.rcParams['figure.figsize'] = (15, 10)
MC = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
MC.head()
#plotting 
X=MC['Age']
Y=MC.iloc[:,3]
Z=MC.iloc[:,4]
plt.xlabel('AGE') 
# naming the y axis 
plt.ylabel('INCOME') 

#plt.scatter(X, Y)
plt.ylabel('Spending')
plt.scatter(X, Z)
#X

MC.head()


# In[ ]:


plt.xlabel('INCOME') 
# naming the y axis 
plt.ylabel('Spending')
plt.scatter(Y, Z)


# In[ ]:


x = MC.iloc[:, [3, 4]].values

# let's check the shape of x
print(x.shape)


# In[ ]:


from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()


# In[ ]:


km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'miser')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'general')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'cyan', label = 'target')
plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')
plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'orange', label = 'careful')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')

plt.style.use('fivethirtyeight')
plt.title('K Means Clustering', fontsize = 20)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
#plt.legend()
plt.grid()

plt.legend()
plt.show()


# In[ ]:


import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogam', fontsize = 20)
plt.xlabel('Customers')
plt.ylabel('Ecuclidean Distance')
plt.show()


# **To determine the optimal number of clusters by visualizing the data, imagine all the horizontal lines as being completely horizontal and then after calculating the maximum distance between any two horizontal lines, draw a horizontal line in the maximum distance calculated******
# 
# For more details please read https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/

# In[ ]:


from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)
#hc.labels_=['A',"B","C","D","E"]
plt.scatter(x[:,0],x[:,1], c=hc.labels_, cmap='rainbow')


# In[ ]:


from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)
#plt.scatter(x[0],x[1], c=cluster.labels_)
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'pink', label = 'miser')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'yellow', label = 'general')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'cyan', label = 'target')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'orange', label = 'careful')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')
plt.legend()


# In[ ]:


from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=3, min_samples=2).fit(x)
clustering.labels_

clustering


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)
print(os.listdir("../input"))


# In[ ]:


from sklearn.cluster import Birch
brc = Birch(branching_factor=500, n_clusters=5, threshold=1.5)
brc.fit(x)
#We use the predict method to obtain a list of points and their respective cluster.
labels = brc.predict(x)
plt.scatter(x[:,0], x[:,1], c=labels, cmap='rainbow')


# To read more about the BIRCH algorithm 
# read
# https://towardsdatascience.com/machine-learning-birch-clustering-algorithm-clearly-explained-fb9838cbeed9

# To see How A k medoids is implemented,See https://towardsdatascience.com/k-medoids-clustering-on-iris-data-set-1931bf781e05

# how to create datasets https://machinelearningmastery.com/generate-test-datasets-python-scikit-learn/

# In[ ]:


import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from matplotlib import pyplot
from pandas import DataFrame
# generate 2d classification dataset
X, y = make_moons(n_samples=1000, noise=0.1)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
   group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()


# In[ ]:


import pandas as pd
from sklearn.cluster import OPTICS, cluster_optics_dbscan 
from sklearn.preprocessing import normalize, StandardScaler 
MC = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
MC.head()
x = MC.iloc[:, [2, 3]].values

# let's check the shape of x
print(x.shape)
optics_model = OPTICS(min_samples = 10, xi = 0.05, min_cluster_size = 0.05)
optics_model.fit(x) 

labels = optics_model.labels_[optics_model.ordering_]
plt.scatter(x[:,0], x[:,1], c=labels, cmap='rainbow')


# In[ ]:


from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

digits = load_digits()
print(digits.data.shape)
print(digits.keys())
# set up the figure
fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the digits: each image is 8x8 pixels
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))
X_train, X_validation, Y_train, Y_validation = train_test_split(digits.data, digits.target,
                                                random_state=0)


# In[ ]:


from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = clf.predict(X_validation)
cm = confusion_matrix(Y_pred, Y_validation)

#Printing the accuracy
print("Accuracy of SVM Classifier : ",accuracy_score(Y_validation, Y_pred))
print(cm)


# In[ ]:


#now we try mlp classifier
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(3,3), max_iter=3000,activation = 'relu',solver='adam',random_state=1)
classifier=classifier.fit(X_train, Y_train)
#Using the trained network to predict

#Predicting y for X_val
Y_pred = classifier.predict(X_validation)
print("Accuracy of MLP Classifier : ",accuracy_score(Y_validation, Y_pred))
print(cm)


# In[ ]:


from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=3000,activation = 'relu',solver='adam',random_state=1)
classifier=classifier.fit(X_train, Y_train)
#Using the trained network to predict

#Predicting y for X_val
Y_pred = classifier.predict(X_validation)
print("Accuracy of MLP Classifier : ",accuracy_score(Y_validation, Y_pred))
#print(cm)


# In[ ]:


import seaborn as sns
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(Y_pred, Y_validation)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');

