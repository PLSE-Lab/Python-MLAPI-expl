#!/usr/bin/env python
# coding: utf-8

# # General Steps to be followed for an end-to-end Machine Learning Project
# 1) Define Problem
# 
#     What is the problem and what are we trying to solve?
#     
#     What are the general intentions behind doing this project and how is it going to be helpful?
#     
#     

# 2) Prepare the dataset
# 
#     One of the major issues with working with datasets is that they are not always formatted and in a way, we want. 
#     We need to clean up the data in various ways. We need to make sure that the data we have is properly formatted
#     and without any issues. We might need to add some data and remove some data, which can be done by viewing the
#     data. In the example provided, we have a clean dataset, but that might not be always the case.
# 

# 3) Evaluate our algorithms
# 
# We will import all the packages we need for our program to run in an end to end basis
# 
# Visualize the data/see how the dataset looks like
# 
# Get statistical Measures to evaluate our algorithm

# # Fruits Classifier
# 
# # Stage 1: Ask A Question:
# 
# What is the questions/answer you expect to find ?
# 
# The goal of this project is to create a fruit classifier with high accurracy.
# 
# We will analyze the data and find out the good features for the classifier. We expect to see the performance of the different classifiers on our dataset.
# 
# 
# # Stage 2: Get the Data
# 
# 
# What is the DataSet ? 
# 
# We will use the fruits dataset created by Dr. Iain Murray from University of Edinburgh. He bought a few dozen oranges, lemons and apples of different varieties, and recorded their measurements in a table. And then the professors at University of Michigan formatted the fruits data slightly and it can be downloaded.
# 
# The dataset can be found at 
# https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/fruit_data_with_colors.txt
# 
# If you have trouble downloading the dataset, you can use this link: https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/fruit_data_with_colors.txt

# In[ ]:


# First import the required packages
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# # Stage 3: Explore the Data:
# 
# In this stage, we will vizualize the data and try to get the information about the dataset. We will visualize the distribution of the input variables, and also see the mean, median and standard deviation of all the features.

# In[ ]:


# Import the data and check the first few rows
fruits = pd.read_table('../input/IPythonData_04042018.txt')
fruits.head(20)


# From the  above table, we can see that:
# 
# Each of the row represents a piece of fruit with its several features like mass, width, height, etc
# 
# Lets see how many different pieces of fruits we have in the dataset

# In[ ]:


#lets see how many rows and columns we have in our data
print(fruits.shape)


# We have 59 pieces of fruits in our dataset

# Now Lets see how many unique fruits are there in the dataset.

# In[ ]:


print(fruits['fruit_name'].unique())


# We can see that the dataset has 4 different types of fruits, ['apple' 'mandarin' 'orange' 'lemon']
# 
# Now lets see the total number of fruits we have for each of them.
# 
# 

# In[ ]:


sns.countplot(fruits['fruit_name'],label="Count")
plt.show()


#  We can see that we have less data for mandarin compared to other fruits. This might hamper the accurracy of our classifier.
#  
#  Box plot for each numeric variable will give us a clearer idea of the distribution of the input variables.

# In[ ]:


fruits.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9), 
                                        title='Box Plot for each input variable')
plt.savefig('fruits_box')
plt.show()


# In[ ]:


fruits.describe()


# # Stage 4: Model the Data
# 
# What you want to predict from the data?
# 
#     With the given four features, we want to predict the type of the fruit.
# 
# Explain how you select this ML model.
# 
#     We are trying to use 3 models here. 2 of them are supervised and 1 is unsupervised.
# 
#     First one is multinomial logistic regression. As our dataset is really simple and we have only 4 features we can use a simple model like logistic regression.
# 
#     Again with simple 4 features, we want to see if decision tree will perform better or not.
#     
#     Next model we want to compare is an unsupervised one. We want to see how KNN performs
# 

# In[ ]:


#We have the following features of the fruit
feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]

# this is the name of the fruit we are predicting
y = fruits['fruit_label']

#we are splitting our main dataset to training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Build Model - Multinomial Logistic Regression
# 
# 

# In[ ]:


# initialize logistic regression from sklearn
logreg = LogisticRegression()

# fit the data in the model
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

logreg_acc = accuracy_score(y_test, y_pred)

# calculate the accurracy in test set
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg_acc))


# # Now lets do the classification on new data
# 

# In[ ]:


#create couple of new fruit features data
X = [[192,10,9,0.88],[80, 5.9, 4.3,0.81]]

logreg.predict(X)


# #  Build Model - Decision Tree
# 

# In[ ]:


#import decision tree classifier from sklearn

from sklearn.tree import DecisionTreeClassifier

# fit the data into the model
clf = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = clf.predict(X_test)

logreg_acc = accuracy_score(y_test, y_pred)

# calculate the accurracy in test set
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg_acc))


# In[ ]:


clf.predict(X)


# In[ ]:


#import KNN from sklearn

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

# fit the data into the model
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

logreg_acc = accuracy_score(y_test, y_pred)

# calculate the accurracy in test set
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg_acc))


# 

# # Stage 5: Communicate the Data
# 
# 

# Our objective is to learn a model that has a good generalization performance. Such a model maximizes the prediction accuracy. We identified the machine learning algorithm that is best-suited for the problem at hand (i.e. fruit types classification); therefore, we compared different algorithms and selected the best-performing one.
# 
# After the traning  all the models, we can see that KNN has best performance on the test set while Decision tree has the best performance training set.
# 

# In[ ]:




