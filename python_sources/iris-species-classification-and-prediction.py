# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importing the libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


#importing dataset
dataset = pd.read_csv("../input/Iris.csv")
X=dataset.iloc[:,0:5].values
y=dataset.iloc[:,-1].values


################################################################################

# Fitting K-Means to the dataset
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_kmeans=kmeans.fit_predict(X)


# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 1], X[y_kmeans == 0, 2], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 1], X[y_kmeans == 1, 2], s = 50, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 1], X[y_kmeans == 2, 2], s = 50, c = 'green', label = 'Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s = 80, c = 'yellow', label = 'Centroids')
plt.title('Clusters of flowers')
plt.xlabel('sepel length')
plt.ylabel('sepel width')
plt.legend()
plt.show()


#################################################################################


#Applying K-NN
#encoding categorical variable---flower names
from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
y =labelencoder_y.fit_transform(y)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

#predicting result on test set

y_pred=classifier.predict(X_test)




#constructing confusion matrix
from sklearn.metrics import confusion_matrix
print("confusion matix obtained by K-NN")
print(confusion_matrix(y_test, y_pred))


############################################################################
#applying randomforest as regressor
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(X_test)
k_pred=[]
for i in range(0,len(y_pred)):
    k_pred.append(int(y_pred[i]))
print("predicted values using Random forest regressor")
print(k_pred)


#######################################################################################################

#fitting randomforst classifier
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier2.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier2.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix obtained by random forest classifier")
print(cm)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 2].min() - 1, stop = X_set[:, 2].max() + 1, step = 0.01))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 1], X_set[y_set == j, 2],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Sepal-length')
plt.ylabel('Sepal-width')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 2].min() - 1, stop = X_set[:, 2].max() + 1, step = 0.01))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 1], X_set[y_set == j, 2],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('sepal-length')
plt.ylabel('sepal-width')
plt.legend()
plt.show()
########################################################################################
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier1 = SVC(kernel = 'rbf', random_state = 0)
classifier1.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier1.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
print("SVM - confusion_matrix:\n",confusion_matrix(y_test, y_pred))
#########################################################################################