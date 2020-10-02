#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotting and graphs
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Exploration

# 
# 
# Here is a snapshot of what our data looks like. We want to be able to predict what species of Iris a flower is based on the length and width of its sepal and petal.

# In[ ]:


data = pd.read_csv('../input/Iris.csv')
data.head()


# Here, we will split our data into features (our descriptive variables) and labels (our dependent variable, or what we want to predict). We will call our features X and our labels y.

# In[ ]:


X = data.drop(['Id', 'Species'], axis=1).values[:, 2:4]
y = data.Species.values

print("First 5 rows of X:\n", X[:5, :])
print("\nFirst 5 labels in y:\n", y[:5])


# It can be useful to plot your data to get a better understanding of the underlying structure. In this case, we can see that the different classes of Iris look like that can indees be distinguished by the features that we have been given.

# In[ ]:


markers = ('s', 'x', 'o')
colors = ('red', 'blue', 'lightgreen')
cmap = ListedColormap(colors[:len(np.unique(y))])
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
               c=cmap(idx), marker=markers[idx], label=cl)


# And, it looks like the classes in our data are balanced. This helps making training predictive models a little easier.

# In[ ]:


sns.countplot(x='Species', data=data)
plt.title("Value Counts of Iris Classes")
plt.show()


# # Data Preprocessing

# First of all, we need to transform our data labels (y) from words to numbers to that our model can process them.

# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

print("Now y is numeric!\n", y)


# Now, we shuffle the examples in our data and create a training and test set.

# In[ ]:


shuffle_index = np.arange(len(y))
np.random.shuffle(shuffle_index)

X_shuffle = X[shuffle_index]
y_shuffle = y[shuffle_index]

y_shuffle


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_shuffle, y_shuffle, test_size=0.3)


# In[ ]:


print("Training set has {} examples".format(X_train.shape[0]))
print("Test set has {} examples".format(X_test.shape[0]))


# # Build Classifier

# Building a classifier can often be as simple as passing it your training features and labels, like we did below.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

dt = DecisionTreeClassifier(random_state=0)
svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)

dt = dt.fit(X_train, y_train)
svm = svm.fit(X_train, y_train)


# # Evaluate Classifier

# In[ ]:


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


# Below we can see the decision boundaries that our classifiers came up with with the test examples plotted on top. Judging by the plots, it looks like our classifiers did very well.

# In[ ]:


plot_decision_regions(X_test, y_test, dt)


# In[ ]:


dt.score(X_test, y_test)


# In[ ]:


plot_decision_regions(X_test, y_test, svm)


# In[ ]:


svm.score(X_test, y_test)

