#!/usr/bin/env python
# coding: utf-8

# #  Background of the very popular Iris Dataset
# 
# The Iris data set also known as 'Fisher's Iris data' is one of the most popular multivariate dataset.
# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). 
# 
# Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.

# In[ ]:



## We will be importing the major libraries that we will need for our EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data_df = pd.read_csv('../input/Iris.csv')


# # Preview of Data
# 
# The Iris dataset is a cleaned dataset. By cleaned I mean there are no Null values or any garbage data.
# 
# - There are 150 observations with 4 features each (sepal length, sepal width, petal length, petal width).
# - There are no null values, so we don't have to worry about that.
# - There are 50 observations of each species (setosa, versicolor, virginica).

# In[ ]:


## We will first have a look at the top five records. Note that since the first 50 records belong to Iris-sentosa
## so will will not be seeing any other records apart from those belonging to Iris-sentosa
data_df.head()


# In[ ]:


## Next we are interested in finding out a little more about the dataset like the data columns/features,their data types 
## and also total number of values for each feature.
data_df.info()


# In[ ]:


## Next we will be exploring the count,mean,standard deviation,min/max and the percentiles off the numeric values.
## On observation it is found that the mean of the features lie within the 50th percentile.

data_df.describe()


# In[ ]:


## Next we will have a look at our label column
## using the function value_counts(), we will see the total unique count of records
## So we see that there are 50 labels for Iris-setosa, 50 fir Iris-versicolor and 50 for Iris-virginica

data_df['Species'].value_counts()


# # Data Cleansing
# 
# Since the Iris data set is already a clean dataset we do not need to spend any time for data cleansing activity.
# However while executing the data.info() command , we found that there is a column called 'id' which does not add any value 
# to our analysis activity. Hence we will be dropping that column from our dataset.

# In[ ]:


data = data_df.drop('Id', axis = 1)
## axis = 1 means, drop column
## axis = 0 means, drop labels


# # Data Analysis And Visualization
# 
# Next we will try to find out if there is any correlation between the  four attributes SepalLengthCm,SepalWidthCm,PetalLengthCm and PetalWidthCm. We will be implementing the pearson method for this purpose.
# 
# What we find is that SepalLengthCm has strong positive correlation with PetalLengthCm and PetalWidthCm.
# Also PetalWidthCm has strong positive correlation with SepalLengthCm and PetalLengthCm.

# In[ ]:


data.corr(method = 'pearson')


# Next we will create a heatmap to represent the above correlation in form of a graph.
# To draw a heatmap we need to import the seaborn package.
# THis is a good website to refer: https://www.absentdata.com/python-graphs/create-a-heat-map-with-seaborn/

# In[ ]:


correlation = data.corr(method = 'pearson')
heat_map = sns.heatmap(correlation,annot = True, cmap = 'coolwarm', linewidth = .5)
plt.show()


# In[ ]:


## Also we will be demonstrating the pair-wise feature correlation using pairplot from the seaborn library
## We are giving the parameter hue = 'Species' so that for each species, the colour marker will be different.

sns.pairplot(data, hue = 'Species')


# So from the graph we can see that iris-sentosa is distinctly different from the other two groups namely iris-versicolor and iris-virginica. Iris-sentosa can be very well classified based on PetalLengthCm and PetalWidthCm. There is a small amount of overlap between iris-versicolor and iris-virginica.

# Next we will be exploring the range of values for SepalLengthCm,SepalWidthCm ,PetalLengthCm, PetalWidthCm.
# We will be plotting one graph for each.
# We will be using violinplot from seaborn to visually represent them.

# In[ ]:


g = sns.violinplot(y='Species', x='SepalLengthCm', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='Species', x='SepalWidthCm', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='Species', x='PetalLengthCm', data=data, inner='quartile')
plt.show()
g = sns.violinplot(y='Species', x='PetalWidthCm', data=data, inner='quartile')
plt.show()


# # Modeling with scikit-learn

# Using the various data model packages avialable in Scikit-Learn, we will train our Iris dataset to find the best 
# model that has the highest accuracy.

# ## Split the dataset into a training set and a testing set
# 
# ### Advantages
# - By splitting the dataset pseudo-randomly into a two separate sets, we can train using one set and test using another.
# - This ensures that we won't use the same observations in both sets.
# - More flexible and faster than creating a model using all of the dataset for training.
# 
# ### Disadvantages
# - The accuracy scores for the testing set can vary depending on what observations are in the set. 
# - This disadvantage can be countered using k-fold cross-validation.
# 
# ### Notes
# - The accuracy score of the models depends on the observations in the testing set, which is determined by the seed of the pseudo-random number generator (random_state parameter).
# - As a model's complexity increases, the training accuracy (accuracy you get when you train and test the model on the same data) increases.
# - If a model is too complex or not complex enough, the testing accuracy is lower.
# - For KNN models, the value of k determines the level of complexity. A lower value of k means that the model is more complex.

# In[ ]:


## In X, we are storing all the features while in y we are storing the lables
X = data.drop(['Species'], axis=1)
y = data['Species']

## 50% training date, 50% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ##### We will be investigating the value of k for which the accuracy of prediction is the highest.Since the training set has 90 rows, we can have a maximum value of k = 90. However here we will test till k = 50

# In[ ]:


# experimenting with different n values
k_range = list(range(1,51))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.grid(True)
plt.show()


# ## Choosing KNN to Model Iris Species Prediction with k = 20
# After seeing that a value of k = 20 is a pretty good number of neighbors for this model, I used it to fit the model for the entire dataset instead of just the training set.

# In[ ]:


## K Nearest Neighbour
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
metrics.accuracy_score(y_test, y_pred)


# In[ ]:


## Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
metrics.accuracy_score(y_test, y_pred)


# In[ ]:


## Decision Tree
tree = DecisionTreeClassifier(max_depth = 3)
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)
metrics.accuracy_score(y_test,y_pred)


# # Re-Doing All the tests once again after splitting the data into 70% training and 30% testing set

# In[ ]:


## Now lets split the train and test data into 80% training data and 20% testing data 
## And run our tests again

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(X_test.shape)


# In[ ]:


## K Nearest Neighbour
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
metrics.accuracy_score(y_test, y_pred)


# In[ ]:


## Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
metrics.accuracy_score(y_test, y_pred)


# In[ ]:


## Decision Tree
tree = DecisionTreeClassifier(max_depth = 3)
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)
metrics.accuracy_score(y_test,y_pred)


# If you found this notebook useful, give me a upvote. Thanks!!
