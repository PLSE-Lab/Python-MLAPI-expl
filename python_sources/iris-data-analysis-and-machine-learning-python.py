#!/usr/bin/env python
# coding: utf-8

# # Getting Data:

# In[ ]:


# Importing the libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Importing the dataset
dataset = pd.read_csv('../input/Iris.csv')
dataset.head()


# In[ ]:


#drop Id column
dataset = dataset.drop('Id',axis=1)
dataset.head()


# # Summary Of the Dataset

# In[ ]:


# shape
print(dataset.shape)


# In[ ]:


# more info on the data
print(dataset.info())


# In[ ]:


# descriptions
print(dataset.describe())


# In[ ]:


# class distribution
print(dataset.groupby('Species').size())


# # Visualizations:

# In[ ]:


# box and whisker plots
dataset.plot(kind='box', sharex=False, sharey=False)


# In[ ]:


# histograms
dataset.hist(edgecolor='black', linewidth=1.2)


# In[ ]:


# boxplot on each feature split out by species
dataset.boxplot(by="Species",figsize=(10,10))


# In[ ]:


# violinplots on petal-length for each species
sns.violinplot(data=dataset,x="Species", y="PetalLengthCm")


# In[ ]:


from pandas.plotting import scatter_matrix
# scatter plot matrix
scatter_matrix(dataset,figsize=(10,10))
plt.show()


# In[ ]:


# Using seaborn pairplot to see the bivariate relation between each pair of features
sns.pairplot(dataset, hue="Species")


# From the plot, we can see that the species setosa is separataed from the other two across all feature combinations

# We can also replace the histograms shown in the diagonal of the pairplot by kde.

# In[ ]:


# updating the diagonal elements in a pairplot to show a kde
sns.pairplot(dataset, hue="Species",diag_kind="kde")


# # Applying different Classification models:

# In[ ]:


# Importing metrics for evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[ ]:


# Seperating the data into dependent and independent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


# LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[ ]:


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[ ]:


# Support Vector Machine's 
from sklearn.svm import SVC

classifier = SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[ ]:


# K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[ ]:


# Decision Tree's
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))

