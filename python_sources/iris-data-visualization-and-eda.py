#!/usr/bin/env python
# coding: utf-8

# # Iris EDA

# ## Dataset: [Iris Dataset](https://www.kaggle.com/uciml/iris)

# ### Import all the necessary header files as follows:
# 
# **pandas** : An open source library used for data manipulation, cleaning, analysis and visualization. <br>
# **numpy** : A library used to manipulate multi-dimensional data in the form of numpy arrays with useful in-built functions. <br>
# **matplotlib** : A library used for plotting and visualization of data. <br>
# **seaborn** : A library based on matplotlib which is used for plotting of data. <br>
# **sklearn.metrics** : A library used to calculate the accuracy, precision and recall. <br>

# In[ ]:


# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


# ### Read the data from the dataset using the read_csv() function from the pandas library.

# In[ ]:


# Importing the dataset
data = pd.read_csv('../input/Iris.csv')


# ### Inspecting and cleaning the data

# In[ ]:


# Printing the 1st 5 columns
data.head()


# In[ ]:


# Printing the dimenions of data
data.shape


# In[ ]:


# Viewing the column heading
data.columns


# In[ ]:


# Inspecting the target variable
data.Species.value_counts()


# In[ ]:


data.dtypes


# In[ ]:


# Identifying the unique number of values in the dataset
data.nunique()


# In[ ]:


# Checking if any NULL values or any inconsistancies are present in the dataset
data.isnull().sum()


# In[ ]:


# See rows with missing values
data[data.isnull().any(axis=1)]


# In[ ]:


# Viewing the data statistics
data.describe()


# In[ ]:


data.info()


# In[ ]:


# Dropping the Id column as it is unnecessary for our model
data.drop('Id',axis=1,inplace=True)


# ## Data Visualization

# In[ ]:


fig = data[data.Species=='Iris-setosa'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='blue', label='Setosa')
data[data.Species=='Iris-versicolor'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='orange', label='Versicolor', ax=fig)
data[data.Species=='Iris-virginica'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='green', label='Virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length vs Sepal Width")
fig=plt.gcf()
# fig.set_size_inches(20,10)
plt.show()


# In[ ]:


fig = data[data.Species=='Iris-setosa'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='blue', label='Setosa')
data[data.Species=='Iris-versicolor'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='orange', label='Versicolor', ax=fig)
data[data.Species=='Iris-virginica'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='green', label='Virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title("Petal Length vs Petal Width")
fig=plt.gcf()
# fig.set_size_inches(20,10)
plt.show()


# As we can see that the Petal Features are giving a better cluster division compared to the Sepal features. This is an indication that the Petals can help in better and accurate Predictions over the Sepal.

# In[ ]:


data.hist(edgecolor='black', linewidth=2)
fig=plt.gcf()
fig.set_size_inches(12,10)
plt.show()


# In[ ]:


sns.boxplot(x="Species", y="PetalLengthCm", data=data)


# In[ ]:


sns.boxplot(x="Species", y="PetalWidthCm", data=data)


# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=data)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=data)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=data)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=data)


# The violinplot shows density of the length and width in the species. The thinner part denotes that there is less density whereas the fatter part conveys higher density

# KDE: Kernel Density Estimate. This shows the distribution density more clearly. We use a FacetGrid with hue = 'Species'. .add_legend() adds the legend on the top rights.

# In[ ]:


sns.stripplot(x="Species", y="PetalLengthCm", data=data, jitter=True, edgecolor="gray")


# In[ ]:


# Distribution density plot KDE (kernel density estimate)
sns.FacetGrid(data, hue="Species", size=6).map(sns.kdeplot, "PetalLengthCm").add_legend()


# In[ ]:


# Plotting bivariate relations between each pair of features (4 features x4 so 16 graphs) with hue = "Species"
sns.pairplot(data, hue="Species", size=4)


# In[ ]:


# Finding out the correlation between the features
corr = data.corr()
corr.shape


# In[ ]:


# Plotting the heatmap of correlation between features
plt.figure()
sns.heatmap(corr, cbar=True, square= True, fmt='.2f', annot=True, annot_kws={'size':15}, cmap = 'Greens')


# The Sepal Width and Length are not correlated The Petal Width and Length are highly correlated.

# In[ ]:


# Spliting target variable and independent variables
X = data.drop(['Species'], axis = 1)
y = data['Species']


# #### Once the data is cleaned, we split the data into training set and test set to prepare it for our machine learning model in a suitable proportion.

# In[ ]:


# Splitting the data into training set and testset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
print("Size of training set:", X_train.shape)
print("Size of training set:", X_test.shape)


# ### Logistic Regression

# In[ ]:


# Logistic Regression

# Import library for LogisticRegression
from sklearn.linear_model import LogisticRegression

# Create a Logistic regression classifier
logreg = LogisticRegression()

# Train the model using the training sets 
logreg.fit(X_train, y_train)


# In[ ]:


# Prediction on test data
y_pred = logreg.predict(X_test)


# In[ ]:


# Calculating the accuracy
acc_logreg = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of Logistic Regression model : ', acc_logreg )


# ### Gaussian Naive Bayes

# In[ ]:


# Gaussian Naive Bayes

# Import library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(X_train,y_train)


# In[ ]:


# Prediction on test set
y_pred = model.predict(X_test)


# In[ ]:


# Calculating the accuracy
acc_nb = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of Gaussian Naive Bayes model : ', acc_nb )


# ### Decision Tree Classifier

# In[ ]:


# Decision Tree Classifier

# Import Decision tree classifier
from sklearn.tree import DecisionTreeClassifier

# Create a Decision tree classifier model
clf = DecisionTreeClassifier(criterion = "gini" , min_samples_split = 100, min_samples_leaf = 10, max_depth = 50)

# Train the model using the training sets 
clf.fit(X_train, y_train)


# In[ ]:


# Prediction on test set
y_pred = clf.predict(X_test)


# In[ ]:


# Calculating the accuracy
acc_dt = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of Decision Tree model : ', acc_dt )


# ### Random Forest Classifier

# In[ ]:


# Random Forest Classifier

# Import library of RandomForestClassifier model
from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest Classifier
rf = RandomForestClassifier()

# Train the model using the training sets 
rf.fit(X_train,y_train)


# In[ ]:


# Prediction on test data
y_pred = rf.predict(X_test)


# In[ ]:


# Calculating the accuracy
acc_rf = round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 )
print( 'Accuracy of Random Forest model : ', acc_rf )


# ### Support Vector Machine

# In[ ]:


# SVM Classifier

# Creating scaled set to be used in model to improve the results
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


# Import Library of Support Vector Machine model
from sklearn import svm

# Create a Support Vector Classifier
svc = svm.SVC()

# Train the model using the training sets 
svc.fit(X_train,y_train)


# In[ ]:


# Prediction on test data
y_pred = svc.predict(X_test)


# In[ ]:


# Calculating the accuracy
acc_svm = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of SVM model : ', acc_svm )


# ### K - Nearest Neighbors

# In[ ]:


# Random Forest Classifier

# Import library of RandomForestClassifier model
from sklearn.neighbors import KNeighborsClassifier

# Create a Random Forest Classifier
knn = KNeighborsClassifier()

# Train the model using the training sets 
knn.fit(X_train,y_train)


# In[ ]:


# Prediction on test data
y_pred = knn.predict(X_test)


# In[ ]:


# Calculating the accuracy
acc_knn = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of KNN model : ', acc_knn )


# ## Evaluation and comparision of all the models

# In[ ]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'Support Vector Machines', 
              'K - Nearest Neighbors'],
    'Score': [acc_logreg, acc_nb, acc_dt, acc_rf, acc_svm, acc_knn]})
models.sort_values(by='Score', ascending=False)


# ## Hence Naive Bayes classification works perfectly on this dataset.

# ### Please upvote if you found this kernel useful! :) <br>
# ### Feedbacks appreciated.
