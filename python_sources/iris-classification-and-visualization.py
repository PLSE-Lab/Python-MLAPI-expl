#!/usr/bin/env python
# coding: utf-8

# Start of by doing importing the dataset, which will the Iris dataset.
# [Link to slideshow](https://docs.google.com/presentation/d/1vAQ8x1VF6rLQnPbirkrxY12T8V6l0F1qJeKoBZZPTbE/edit?usp=sharing)
# ![](https://cdn-images-1.medium.com/max/1600/0*7H_gF1KnslexnJ3s)
# 

# In[118]:


import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("../input/Iris.csv", index_col = 0)
data.head()


# We want to break up the dataset according to our features and targets. This is important for training our model.
# ![](https://jakevdp.github.io/PythonDataScienceHandbook/figures/05.02-samples-features.png)

# In[119]:


X = data.iloc[:, :4]  #feature matrix
y = data.iloc[:, 4]  #target vector


# Let's see the iris species we are dealing with. Their count is 150, each species is 50.

# In[120]:


y.value_counts()


# In[121]:


X = X.values   # convert to numpy array for compatibility with other packages
y = y.values


# Simple matplotlib functions will tell us relationships between couple of features, but we have 4 in total so we need a better way to visualize everything. The seaborn library has the pairplot function which we can utilize.

# In[122]:


import seaborn as sns  # we dont want to look at all plots individually so we use pair plots
sns.pairplot(data, hue = "Species", height=3) 


# This is just to show off the 3D graphic potential of matplotlib; It looks pretty cool.

# In[123]:


#credits to https://www.kaggle.com/skalskip/iris-data-visualization-and-knn-classification
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
fig = plt.figure(1, figsize=(8, 5))
ax = Axes3D(fig, elev=48, azim=134)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s = X[:, 3]*50)

for name, label in [('Virginica', 0), ('Setosa', 1), ('Versicolour', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean(),
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'),size=15)

ax.set_title("3D visualization", fontsize=20)
ax.set_xlabel("Sepal Length [cm]", fontsize=15)
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Sepal Width [cm]", fontsize=15)
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Petal Length [cm]", fontsize=15)
ax.w_zaxis.set_ticklabels([])

plt.show()


# Enough with the visualization and let's do some machine learning. First we need to decide on what model we should use. Since this is a classification problem there are several: Decision Trees, SVMs(Support Vector Machines), K-Nearest Neighbors, etc., all of which are built into the ML package, Scikit-learn. **In this tutorial we will use decision trees because they are the simplest.** Here is an example of a tree:
# ![](https://cdn-images-1.medium.com/max/1200/0*Yclq0kqMAwCQcIV_.jpg)
# Here is a visual on a graph: 
# ![](https://i.stack.imgur.com/8IxwL.png)

# We then have to break up our features and targets further into training and testing sets. The training set will be used to fit our model while the testing set will be used to measure the performance.

# In[124]:


from sklearn.model_selection import train_test_split  #used to split to train and test samples
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.30) #test sample = 0.30 of the data , or 45 samples


# Now time to declare and initialize the Deicision Tree Classifier model from sklearn:

# In[125]:


from sklearn import tree
iris_classifier = tree.DecisionTreeClassifier()


# We have to now train the data which is simple since we already defined the training sets:

# In[126]:


iris_classifier.fit(x_train, y_train)  # notice the hyperparameters and optimizations below after execution


# We now test our model and gauge its performance:

# In[127]:


predictions = iris_classifier.predict(x_test)

from sklearn.metrics import accuracy_score   # another import, this time for accuracy score
print(accuracy_score(y_test,predictions))  # raw score


# For our very first model, 90+ accuracy is not too bad. However we can optimize our hyperparameters to fine-tune our model to get even higher accuracy. We can also try other kinds models, too, like SVMs. Next week we will dig further into the different types.
# 
# Created by rae385 for CHS AI Club
