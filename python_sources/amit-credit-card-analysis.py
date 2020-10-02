#!/usr/bin/env python
# coding: utf-8

# # ** What do we plan to do here? **
# * Read the dataset
# * Analyze the data for missing values and outliers
# * Perform uni-variate analysis
# * Perform bi-variate analysis
# * We will not do any feature engineering in this particular problem
# * We will create a lot of visualizations to do a thorough analysis of the problem at hand
# * Pick a list of algorithms which we can choose to apply in this case
# * Pick the best algorithm
# * Score the algorithm based on the evaluation criteria
# * Fine tune algorithms to achieve the best possible value of the evaluation metric

# ### Q1) Read the dataset into the notebook

# In[ ]:


import os
os.chdir("../input")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("creditcard.csv")
data.head()


# ### Q2) Print the shape of the data

# In[ ]:


data.shape


# ### Q3) List out the feature variables and their data-types

# In[ ]:


f_variables = data.iloc[:,0:30]
f_variables.dtypes


# ### Q4) List out response variable and its data type

# In[ ]:


data['Class'].dtypes


# ### Q5) Check for null values in the feature variables

# In[ ]:


#f_variables.isnull().any
f_variables.describe()
#by observing count values, all the variables have same count which means no missing values


# In[ ]:


#f_variables.isnull().values.any() gives if there any missing values(true/false)
#f_variables.isnull().sum() gives number of missing values by each column


# ### Q6) Treat the null variables. What is your strategy? Why did you use that? What other strategies could be taken? Explain

# In[ ]:


### There are no missing values in the data 


# ### Q7) Check for outliers in the feature variables

# In[ ]:


from numpy import percentile
Q1 = f_variables.quantile(0.25)
Q3 = f_variables.quantile(0.75)
IQR = Q3-Q1
#print("quartiles are ",Q3,Q1)
print(IQR)


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.figure(figsize = (15,100))
gs = gridspec.GridSpec(10,3)
for i,cn in enumerate(f_variables.columns):
    ax = plt.subplot(gs[i])
    sns.boxplot(f_variables[cn])
    ax.set_xlabel('')
    ax.set_title('feature: ' + str(cn))
plt.show() 


# ### Q8) Treat outliers. What is your strategy?

# In[ ]:


lower_band = IQR - 1.5*Q1
upper_band = IQR +1.5*Q3
f_variables.clip(lower = lower_band, upper = upper_band, axis = 1)
f_variables.shape


# ### Q9) Pick each one of the feature variables and perform univariate analysis (be as creative as possible in your analysis)
# * ### Q9.1) Visualize the shape of the distribution of data.Is every feature variable normally distributed? Why is normal distribution important for data?
# * ### Q9.2) Is the data distribution skewed? If highly skewed,do you still find outliers which you did not treat?
# * ### Q9.3) Draw box and whiskers plot of each of the feature variables
# * ### Q9.4) How do the distributions look in terms of variation? Which features are widely spread and which are kind of concentrated towards the mean?

# In[ ]:


#9.1
f_variables.hist(figsize = (20,20))
plt.show()


# ### checking the class distribution
# 

# In[ ]:


data['Class'].value_counts()


# In[ ]:


sns.countplot(x = 'Class', data= data)


# ### Q10) Pick the feature variables and perform bi-variate analysis (be as creative as possible)
# * ### Q10.1) Try creating correlation matrices. See if there are variables which are strongly or weakly related
# * ### Q10.2) Try build joint distribution charts
# * ### Q10.3) If there are variables showing high correlation, what corrective action is needed? Why is this a matter of concern? What if we do not treat the variables showing high degree of correlation?

# In[ ]:


##10.1)
corr = data.corr()
corr


# In[ ]:


plt.figure(figsize = (30,10))
sns.heatmap(corr,xticklabels = corr.columns, yticklabels = corr.columns)


# In[ ]:


plt.figure(figsize = (15,100))
gs = gridspec.GridSpec(10,3)
for i, c in enumerate(f_variables.columns):
    plt.subplot(gs[i])
    plt.scatter(f_variables['V1'], f_variables[c])
    plt.xlabel("V1")
    plt.ylabel("feature Variable:"+str(c))


# ### Q11.1) What is the type of machine learning problem at hand? (Supervised or Unsupervised?) Why?
# ### Q11.2) What is the category of the machine learning problem at hand? (Classification or Regression?) Why?

# 11.1) It is a Supervised learning problem, because in supervised learning for a given input 
# we have an output values, here we are predicting the class outcome by giving input fields.
# 
# 11.2) It is a Classification problem, because the response variable is in terms of yes or no i.e. 1 or 0, 
# where as in regression type of problems, we will predict the continuous variable.

# ### Q12.1) Draw univariate plots for each of the feature variables, color each plotted point as red if the class value = 0 else green.
# ### Q12.2) Which feature segregates the data the cleanest way? How would you calculate the misclassification rate?
# ### Q12.3) Now take two features at a time, again color each plotted point as mentioned in 12.1. Calculate and comment on the misclassification rate?

# In[ ]:


import numpy as np
colors = np.where(data['Class']==0,'red','green')


# In[ ]:


plt.figure(figsize = (15,100))
gs = gridspec.GridSpec(10,3)
for i, c in enumerate(f_variables.columns):
    plt.subplot(gs[i])
    plt.scatter(f_variables['V1'], f_variables[c], c = colors)
    plt.xlabel("V1")
    plt.ylabel("feature Variable:" +str(c))
plt.show()


#  ### Q13.1) List down all the algorithms known to you which you think might be applicable in this case?
# Decision Trees
# Random Forest
# KNN
# Logistic Regression
# SVM

# ### Q14) Pick each of the algorithm and perform the below steps : 
# ### Q14.1) Split your data between test, train and validation steps. Why 3 and not just test and train? 
# ### Q14.2) Build your model
# ### Q14.3) List down the evaluation metrics you would use to evaluate the performance of the model?
# ### Q14.4) Evaluate the model on training data
# ### Q14.5) Predict the response variables for the validation test data
# ### Q14.6) Evaluate the model on test data
# ### Q14.7) How are the two scores? Are they significantly different? Are they the same? Is the test score better than training score?

# In[ ]:


f_variables['Class'] = data['Class']
f_variables.head()


# In[ ]:


split1 = int(0.8*len(f_variables))
split2 = int(0.9*len(f_variables))
train = f_variables[:split1]
validation = data[split1:split2]
test = data[split2:]


# In[ ]:


x_train = train.drop('Class', axis = 1)
y_train = train['Class']
x_validation = validation.drop('Class', axis =1)
y_validation = validation['Class']
x_test = test.drop('Class', axis = 1)
y_test = test['Class']


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 40, random_state = 10)
rf.fit(x_train, y_train)


# In[ ]:


pred = rf.predict(x_validation)
pred


# In[ ]:


from sklearn.metrics import accuracy_score
score = accuracy_score(y_validation, pred.round())
score


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)


# In[ ]:


c_pred = clf.predict(x_validation)
score = accuracy_score(y_validation, c_pred)
score


# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)


# In[ ]:


log_pred = logreg.predict(x_validation)
score = accuracy_score(y_validation, log_pred)
score


# In[ ]:




