#!/usr/bin/env python
# coding: utf-8

# The dataset consists of 10000 individuals and whether their credit card has defaulted or not. The main aim is to build the model using Logistic Regression and predict the accuracy of it . <br>
# 
# Attributes:<br>
#     
# Default : Yes or No (Whether defaulted or Not). <br>
# Student : Yes or Nor (Whether Student or not). <br>
# Balance : Total Balance for given credit card holder.<br>
# Income : Gross Annual Income of credit card holder.<br>
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, integrate
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.float_format = '{:.2f}'.format
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14


# In[ ]:


cred_df=pd.read_csv("../input/attachment_default.csv")
import re
cred_df.head(10)


# In[ ]:


cred_df.head()


# In[ ]:


cred_df.info()


# In[ ]:


cred_df.info()


# From the above graph, we can easily classify the majority of student by simply drawing a horizontal line (from balance =1250) that separate Yes  & no (default).This is nothimg but a linear classifier describing the relationship between input and output variable and proves that its a condition for the logistic regression. 

# In[ ]:


sns.boxplot(x='default', y='income', data=cred_df)
plt.show()


# In[ ]:


sns.lmplot(x='balance', y='income', hue = 'default', data=cred_df, aspect=1.5, fit_reg = False)
plt.show()


# **A scatter plot and box and whisker diagram seem to suggest that there is a relationship between credit card balance and default, while income is not related. The diagram also suggest that the default rates are higher when balance is high.**

# In[ ]:


pd.crosstab(cred_df['default'], cred_df['student'], rownames=['Default'], colnames=['Student'])


# # Generating Dummy variable
# 
# The dummy variable <b>default_yes</b> reflecting the class value 0 or 1. When class value is 1 then we have the default case. 

# In[ ]:


# Convert Categorical to Numerical
default_dummies = pd.get_dummies(cred_df.default, prefix='default')
default_dummies.drop(default_dummies.columns[0], axis=1, inplace=True)
cred_df = pd.concat([cred_df, default_dummies], axis=1)
cred_df.head()
#default_dummies


# In[ ]:


# Building Linear Regression Model
from sklearn.linear_model import LinearRegression
    
X = cred_df[['balance']]
y = cred_df['default_Yes']

linreg = LinearRegression()
linreg.fit(X, y)

print(linreg.coef_)
print(linreg.intercept_)


# <b>Even though through linear regression, we are getting value of coefficient and intercept from the equation but this is not correct because our Output is in 0 or 1. Whereas the regression equation is generating a value between 0 to 1. So this doesnot make any sense and same has been suggested through the lmplot where datapoints are plotted as 0 and 1.</b> 

# In[ ]:


sns.lmplot(x='balance', y='default_Yes', data=cred_df, aspect=1.5, fit_reg = True)


# # Creating logistic model for demonstration purpose with all data.

# In[ ]:


#calling logistic regression  ( fitting all the data for demonstration purpose. The training & test data is excuted after this.)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X, y)
print(logreg.coef_)
print(logreg.intercept_)

y_pred = logreg.predict_proba(X)
plt.scatter(X.values, y_pred[:,1])
#plt.scatter(X.values, y)
plt.show()


# As we are intersted in only default =1, So we have plotted the only the y_pred[:,1], which gives us a sigmoid. 

# In[ ]:


# probability of  (class 0 , class 1)

y_pred


# In[ ]:


# probability of class 0 only.

y_pred[:,0]


# # Creating logistic model with Train & Test Data

# In[ ]:


cred_df.head()


# In[ ]:


X.head()


# In[ ]:


#splitting the data into train and test with 70:30 ratio
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.30, random_state=13)


# In[ ]:


#calling logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(X, y)
print(logreg.coef_)
print(logreg.intercept_)


# In[ ]:


#fitting the model with x and y attributes of train data
#in this it is goin to learn the pattern
logreg.fit(xTrain, yTrain)


# In[ ]:


#now applying our learnt model on test and also on train data
y_log_pred_test = logreg.predict(xTest)
y_log_pred_train = logreg.predict(xTrain)


# In[ ]:


y_log_pred_test.shape


# In[ ]:


y_log_pred_train.shape


# In[ ]:


y_log_pred_test


# # CONFUSION MATRIX

# In[ ]:


#creating a confusion matrix to understand the classification
conf = metrics.confusion_matrix(yTest, y_log_pred_test)
conf


# In[ ]:


# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(yTest, y_log_pred_test)
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
print ("TP",TP)
print ("TN",TN)
print("FN",FN)
print ("FP",FP)


# In[ ]:


cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(conf,cmap = cmap,xticklabels=['predicted_default_yes=0','predicted_default_yes=1'],yticklabels=['actual_default_yes=0','actual_default_yes=1'],annot=True, fmt="d")


# **Result:**
# 
# TrueNegative(TN) = 2508 cases, which are nondefault and predicted as nondefault as well.
# 
# TruePositive(TP) = 88 cases, which are default and predicted as default as well.
# 
# FalseNegative(FN) = 21 cases, which are actually default but predicted as nondefault.
# 
# FalsePositive(FP) = 383 cases, which are actually nondefault but predicted as default.
# 

# In[ ]:


# print the first 25 true and predicted responses
print('True', yTest.values[0:15])
print('Pred', y_log_pred_test[0:15])


# # Metrics computed from a confusion matrix

# **1.Classification Accuracy: Overall, how often is the classifier correct?** This is discussed above in  detail under the classification accuracy.

# In[ ]:


#comparing the metrics of predicted lebel and real label of test data
print('Accuracy_Score:', metrics.accuracy_score(yTest, y_log_pred_test))


# This suggest that 86.5% observations of credit defaults rates are correctly or accurately observe by our model. 

# <b>2.Classification Error: Overall, how often is the classifier incorrect?</b>. It is nothing but (1-classification accuracy)
# 
# **Also known as "Misclassification Rate"**

# In[ ]:


# Method to calculate Classification Error
   

print('Classification Error:',1 - metrics.accuracy_score(yTest, y_log_pred_test))


# <b>3.Sensitivity or Recall:</b> When the actual value is positive, how often is the prediction correct? .

# In[ ]:


# Method to calculate Sensitivity

print('Sensitivity or Recall:', metrics.recall_score(yTest, y_log_pred_test))


# **4.Specificity: When the actual value is negative, how often is the prediction correct?**
# 

# In[ ]:


specificity = TN / (TN + FP)

print(specificity)


# <b>7.Classification Report.</b>

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(yTest, y_log_pred_test))


# In[ ]:


cred_df.head()


# In[ ]:


#Defining a sample data to test the model
# As we discussed earlier, income has no significance in default. So only balance is considered as input & X = cred_df[['balance']]

feature_cols = ['balance']
data =[817.18]
studentid_2=pd.DataFrame([data],columns=feature_cols)
studentid_2.head()


# In[ ]:


predictions_default=logreg.predict(studentid_2)
print(predictions_default)


# **output is zero means, the Studentid_2 is a nondefault case and not going to have default anytime soon.**

# # Adjusting the classification threshold

# In[ ]:


# print the first 10 predicted responses
# 1D array (vector) of binary values (0, 1)
logreg.predict(xTest)[0:10]


# In[ ]:


# print the first 10 predicted probabilities of class membership
logreg.predict_proba(xTest)[0:10]


# In[ ]:


# print the first 10 predicted probabilities for class 1   ( predicting diabetic cases =1)
logreg.predict_proba(xTest)[0:10, 1]


# In[ ]:


# store the predicted probabilities for class 1
y_pred_prob = logreg.predict_proba(xTest)[:, 1]


# In[ ]:


y_pred_prob[0:10]


# In[ ]:


# Plotting predicion through histogram of predicted probabilities
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# 8 bins
plt.hist(y_pred_prob, bins=8)

# x-axis limit from 0 to 1
plt.xlim(0,1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of default')
plt.ylabel('Frequency')


# Histogram suggest that the predicted probabilities are positively-skewed distribution with a long tail on right side and most of the probabilities are <0.1. So we going to change the threshold for probability from 0.5 to 0.1 with binarize function.

# In[ ]:


# predict diabetes if the predicted probability is greater than 0.1
from sklearn.preprocessing import binarize
from sklearn.preprocessing import Binarizer
from sklearn import preprocessing
# it will return 1 for all values above 0.1 and 0 otherwise
# results are 2D so we slice out the first column

y_pred = binarize(y_pred_prob.reshape(-1,1), 0.1) 


# In[ ]:


y_pred.shape


# In[ ]:


# probability with revised threshold =0.1

y_pred_prob[0:10]


# In[ ]:


# Outcome with revised threshold =0.3

y_pred[0:10]


# In[ ]:


# previous confusion matrix (default threshold of 0.5)
print(confusion)


# In[ ]:


#The new confusion matrix (threshold of 0.1)
   
print(metrics.confusion_matrix(yTest, y_pred))


# <b> We can see that earlier we are able to correctly classified (TP) 88 cases of diabetic. Now, we are able to correctly classify 106 diabetic cases by lowering the threshold.</b>
# 

# In[ ]:


# sensitivity has increased (used to be 0.81)
print (106 / float(3 + 106))


# In[ ]:


# specificity has decreased (used to be 0.86)
print(1812 / float(1812 + 1079))


# <b>We are more interested in higher sensitivity value because false positives (normal transactions that are flagged as possible fraud) are more acceptable than false negatives (fraudulent transactions that are not detected).</b>

# 
# # Receiver Operating Characteristic (ROC) Curves
# 

# In[ ]:


# IMPORTANT: first argument is true values, second argument is predicted probabilities

# we pass y_test and y_pred_prob
# we do not use y_pred, because it will give incorrect results without generating an error
# roc_curve returns 3 objects fpr, tpr, thresholds
# fpr: false positive rate
# tpr: true positive rate
fpr, tpr, thresholds = metrics.roc_curve(yTest, y_pred_prob)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for default classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# # AUC 

# In[ ]:


# IMPORTANT: first argument is true values, second argument is predicted probabilities

print(metrics.roc_auc_score(yTest, y_pred_prob))


# # END
