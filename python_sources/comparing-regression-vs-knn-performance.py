#!/usr/bin/env python
# coding: utf-8

# The aim of this kernel is to compare various types of regressions vs K-nearest neighbor alogrithm on the Biomechanical features of orthopedic patients data. 
# 
# This kernel will also contain basic EDA to understand the data. 
# 
# Types of regression covered in this kernel:
# 1. Linear Regression
# 2. Random Forest
# 3. Decision Trees
# 4. Logisitic Regression
# 
# So, lets get started with the data set:  Biomechanical features of orthopedic patients
# 
# This data set contains features for orthopedic patients. Copying over the description from data set page:
# 
# Each patient is represented in the data set by six biomechanical attributes derived from the shape and orientation of the pelvis and lumbar spine (each one is a column):
# 
# * pelvic incidence
# * pelvic tilt
# * lumbar lordosis angle
# * sacral slope
# * pelvic radius
# * grade of spondylolisthesis
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/column_2C_weka.csv")
data.head()


# Let's look at the correlation between different variables.

# In[ ]:


data.corr()


# Looking at the correlated data above, here are some observations:
# 
# Highly correlated:
# - pelvic_incidence and sacral_slope are highly correlated
# - pelvic_tilt numeric and pelvic_incidence are highly correlated
# - lumbar_lordosis_angle and pelvic_incidence are highly correlated
# - pelvic_radius and pelvic_tilt numeric are highly correlated
# 
# Weakly correlated:
# - pelvic_incidence and pelvic_radius are weakly correlated.
# - pelvic_tilt numeric and pelvic_radius are weakly correlated.
# 
# **Heat map:**
# 
# Heat map allows us to look at the correlated data in a visual manner, which might help us quickly point out variables which are highly correlated, hence helping us on a path towards using those variables for further analysis.
# 
# We will create a heat map from the correlated data below:

# In[ ]:


f, axis = plt.subplots(figsize=(12,12))
sns.heatmap(data.corr(), annot=True, linewidths=0.4, fmt='.2f', ax = axis)
plt.show()


# As you can see from the heat map, pelvic_radius is weakly correlated with most of the other features in the dataset. This is helpful as we would have to spend more time with a tabular dataset to come to the above conclusion.
# 
# **Count Plot:**
# 
# Lets take a look at the counts of abnormal vs normal classes in the dataset.
# 

# In[ ]:


sns.countplot(x="class", data=data)
data.loc[:, 'class'].value_counts()


# As we can see from the above plot, abnormal counts are more than double of the of the normal class counts. 

# **Pair Plot**
# 
# We would be using one of the most common plots used by data scientists when trying to understand the data. Pair plot helps us display correlation between different features via scatter plot and historgrams. 
# 
# By default pairplot will return scatter plots in the main entries and a histogram in the diagonal. pairplot is oftentimes the first thing that a data scientist will throw at their data, and it works fantastically well in that capacity, even if sometimes the scatter-and-histogram approach isn't quite appropriate, given the data types.

# In[ ]:


sns.pairplot(data,hue="class",palette="Set2", diag_kind = 'kde')
plt.show()


# As you can see from the pairplot above, the distribution of classes (Abnormal and Normal) are shown over all the features. 
# 
# For example, you can look at pelvic_incidence and degree_spondyiosthesis pair, which shows you the distribution of classes. As the pelvic_incidence increases, the degree_spondyiosthesis does not increase beyond 200 value. 
# 
# Another example, pelvic_tilt_numeric and lumbar_lordosis_angle, as pelvic_tilt_numeric increases, so does lumbar_lordosis_angle, hence proving them to be linearly related to each other. 
# 
# You can find more information on how to create pairplots and play around with various nobs for pair plot here: https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166

# **Linear Regression:**
# 
# Lets look at the basica regression algorithm, linear regression. Our aim is to plot the values vs predicted values from linear regression.

# In[ ]:


from sklearn.linear_model import LinearRegression

data1=data[data['class'] == "Abnormal"]

linear_Reg = LinearRegression()
x = np.array(data1.loc[:, 'pelvic_incidence']).reshape(-1,1)
y = np.array(data1.loc[:, 'sacral_slope']).reshape(-1,1)

linear_Reg.fit(x, y)
y_head = linear_Reg.predict(x)


# In[ ]:


plt.figure(figsize=(15, 5))
plt.scatter(x, y, color="green")
plt.plot(x, y_head, color="black")
plt.xlabel("pelvic_incidence")
plt.ylabel("sacrel_scope")
plt.show()


# Based on above plot, the black line is the prediction from linear regression and the green circles are the actual values for correlation between pelvic_incidence and sacrel_scrope. 
# 
# Lets take a another example for linear regression, this time looking at relation between: pelvic_tilt_numeric and lumbar_lordosis_angle.

# In[ ]:


linear_Reg = LinearRegression()
x = np.array(data1.loc[:, 'pelvic_tilt numeric']).reshape(-1,1)
y = np.array(data1.loc[:, 'lumbar_lordosis_angle']).reshape(-1,1)

linear_Reg.fit(x, y)
y_head = linear_Reg.predict(x)


# In[ ]:


plt.figure(figsize=(15, 5))
plt.scatter(x, y, color="green")
plt.plot(x, y_head, color="black")
plt.xlabel("pelvic_tilt numeric")
plt.ylabel("lumbar_lordosis_angle")
plt.show()


# **Logistic Regression:**
# 
# Logistic regression is a statistical method for predicting binary classes. The outcome or target variable is dichotomous in nature, meaning there are only two possible outcomes or classes. It computes the probability of an event occurring. 
# 
# It is a special case of linear regression where the target variable is categorical in nature. It uses a log of odds as the dependent variable. 
# 
# Linear Regression Equation:
# 
# ![](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1534281880/image1_ga8gze.png)
# 
# y: dependent variable
# x: features
# 
# Sigmoid function: 
# ![](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1534281880/image3_qldafx.png)
# 
# Apply sigmoid function on linear regression:
# 
# ![](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1534281880/image3_qldafx.png)
# 
# The above equation is used for logistic regression.
# 
# Logistic Regression is estimated using maximum likelihood estimation approach. In this approach, we determine the parameters that are mostly likely to produce the desired outcome. MLE sets the mean and variance parameters in determining the specific parameter value for a given model. This set of parameters can be used for predicting the data needed in a normal distribution.
# 
# Types of logisitc regression:
# - Binary logistic regression: the target variable has only two possible outcomes such as Spam or Not Spam, Cancer or No cancer.
# - Multinominal logistic regression: the target variable has three or more nominal categories such as predicting the type of wine. 
# - ordinal logistic regression: the target variable has three or more ordinal categories such as a restaurant or product rating from 1 to 5. 
# 
# You can read more about logisitc regression here: https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python
# 

# In[ ]:


data.columns


# In[ ]:


#split dataset in features and target variable
feature_cols = ['pelvic_incidence', 'pelvic_tilt numeric', 'lumbar_lordosis_angle',
       'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis']

X = data[feature_cols] # Features
y = data['class'] # Target variable


# Lets split the data into a training set and testing set by using model selection from sklearn package. 

# In[ ]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[ ]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

# predict
y_pred = logreg.predict(X_test)


# **Model Evaluation using Confusion Matrix**
# 
# A confusion matrix is a table that is used to evaluate the performance of a classification model. You can also visualize the performance of an algorithm. The fundamental of a confusion matrix is the number of correct and incorrect predictions are summed up class-wise.

# In[ ]:


# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# Here. you can see the confusion matrix in the form of the array object. The dimension of this matrix is a 2 by 2 because this model is binary classification. You can have two classes: Abnormal and Normal. 
# 
# Digonal values represent accurate predictions ( 46, 17) and non-diagonal values are inaccurate predictions (7, 8).
# 
# **Visualizing the confusion matrix using heatmap:**
# 
# Let's visualize the results of the model in  the form of a confusion matrix using matplotlib and seaborn. 

# In[ ]:


class_names=["Abnormal", "Normal"] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# **Confusion Matrix Evaluation Metrics**
# 
# Let's evaluate the model using model evaluation metrics such as accuracy.

# In[ ]:


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred, labels=['Normal', 'Abnormal'], average=None))
print("Recall:",metrics.recall_score(y_test, y_pred, labels=['Normal', 'Abnormal'], average=None))


# Lets look at the values:
# 
# Accuracy of 80%. Our logistic regression model is accurate 80% of time. 
# 
# Precision:
# It expresses the proportion of the data points our model says was relevant actually was relevant. For our use-case, for the class Normal, it was precise 68% of time. For abnormal class, it was precise 86% of time.
# 
# Recall:
# It expresses  the ability of the model to find all relevant instances in a dataset. For our use-case, for the class Normal, it was able to find the class 70% of  time. For the class Abnormal, it was able to find the class 85% of the time. 

# **Precision - Recall:**
#     
# Precision-Recall is a useful measure of success of prediction when the classes are very imbalanced. In information retrieval, precision is a measure of result relevancy, while recall is a measure of how many truly relevant results are returned.
# 
# The precision-recall curve shows the tradeoff between precision and recall for different threshold. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate. High scores for both show that the classifier is returning accurate results (high precision), as well as returning a majority of all positive results (high recall).
# 
# A system with high recall but low precision returns many results, but most of its predicted labels are incorrect when compared to the training labels. A system with high precision but low recall is just the opposite, returning very few results, but most of its predicted labels are correct when compared to the training labels. An ideal system with high precision and high recall will return many results, with all results labeled correctly.
# 
#  While recall expresses the ability to find all relevant instances in a dataset, precision expresses the proportion of the data points our model says was relevant actually were relevant.
#  
#  More information on precision-recall can found here: https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c

# **ROC Curve (Receiver operating characertisitic):**
# 
# It is a plot of the true positive rate against the false positive rate. It shows the tradeoff between sensitivity and specificity. 

# In[ ]:


y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba, pos_label="Abnormal")
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:


y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba, pos_label="Normal")
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:




