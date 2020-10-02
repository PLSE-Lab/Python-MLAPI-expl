#!/usr/bin/env python
# coding: utf-8

# Installing packages used to clean and visualize the data. There are other packages/functions used later down for each type of regression.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
import sklearn
import seaborn as sns
import scipy as sp
from scipy import stats

import os
print(os.listdir("../input"))


# Next, I uploaded and looked at a rough summary of the datasets.

# In[ ]:




train = pd.read_csv('../input/launchds-classification/bank-train.csv')
test = pd.read_csv('../input/launchds-classification/bank-test.csv')

train.head() # lots of categorical variables
train.describe() # pdays, previous, y is very skewed


# The features Pdays and Previous were heavily skewed in their distribution of values, so I thought it would be interesting to plot them and see what's up.

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(10, 2), sharey=False, dpi=100)
sns.distplot(train['pdays'] , color="dodgerblue", ax=axes[0], axlabel='Pdays')
sns.distplot(train['previous'] , color="deeppink", ax=axes[1], axlabel='Previous')


# Next, I wanted to see if there were any null values, what the balance of 1s versus 0s were in the response variable, and see how many observations there are.

# In[ ]:


print(train.isnull().apply(sum), '\n') # no null values
print(train.groupby('y').count()['id'], '\n') # 29245=0, 3705=1
print('There are',len(test),'testing observations') # 8238 testing observations


# Next came the task of transforming the categorical variables. I wanted to leave the features default, housing, loan, and poutcome as a [0,1,2] for [No,Yes,Unknown]. That could be revised later as a potential way to cut down on information. I also wanted day of the week and month to preserve the order as they fall in a calendar, so I modified them to 1-5 for Mon-Fri and 1-12 for jan-dec. Two things to note here: (1) there were no jan or feb data in the month column and no sat or sun in the day column (2) I went back later to see if coding each month/day in it's own column would be useful and it significantly hurt my model, so I switched back to keeping them within the same variable.

# In[ ]:


train.head()
'''categorical: 
job: 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
       'retired', 'self-employed', 'services', 'student', 'technician',
       'unemployed', 'unknown'
marital: 'divorced', 'married', 'single', 'unknown'
education: 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
       'professional.course', 'university.degree', 'unknown'
default: 'no', 'unknown', 'yes'
housing: 'no', 'unknown', 'yes'
loan: 'no', 'unknown', 'yes'
contact: 'cellular', 'telephone'
month: 'apr', 'aug', 'dec', 'jul', 'jun', 'mar', 'may', 'nov', 'oct',
       'sep'
day_of_week: 'fri', 'mon', 'thu', 'tue', 'wed'
poutcome: 'failure', 'nonexistent', 'success'
''' 

def yes_no_uk(x):
  '''used to transform default, housing, and loan'''
  if x=='yes':
    return(1)
  elif x=='no':
    return(0)
  elif x=='unknown':
    return(2)
  
def poutcome(x):
  '''used to transform poutcome'''
  if x=='success':
    return(1)
  elif x=='failure':
    return(0)
  elif x=='nonexistent':
    return(2)

def day_of_week(x):
  '''used to transform day_of_week'''
  if x=='mon':
    return(1)
  elif x=='tue':
    return(2)
  elif x=='wed':
    return(3)
  elif x=='thu':
    return(4)
  elif x=='fri':
    return(5)
  
def month(x):
  '''used to transform month'''
  if x=='jan':
    return(1)
  elif x=='feb':
    return(2)
  elif x=='mar':
    return(3)
  elif x=='apr':
    return(4)
  elif x=='may':
    return(5)
  elif x=='jun':
    return(6)
  elif x=='jul':
    return(7)
  elif x=='aug':
    return(8)
  elif x=='sep':
    return(9)
  elif x=='oct':
    return(10)
  elif x=='nov':
    return(11)
  elif x=='dec':
    return(12)


# In[ ]:


# transforming training ordinal variables
default_labels = train['default'].apply(yes_no_uk)
housing_labels = train['housing'].apply(yes_no_uk)
loan_labels = train['loan'].apply(yes_no_uk)
month_labels = train['month'].apply(month)
day_labels = train['day_of_week'].apply(day_of_week)
poutcome_labels = train['poutcome'].apply(poutcome)

# transforming test data rdinal variables
default_labels2 = test['default'].apply(yes_no_uk)
housing_labels2 = test['housing'].apply(yes_no_uk)
loan_labels2 = test['loan'].apply(yes_no_uk)
month_labels2 = test['month'].apply(month)
day_labels2 = test['day_of_week'].apply(day_of_week)
poutcome_labels2 = test['poutcome'].apply(poutcome)


# Next I had to transform the categorical features that did not have a pre-defined order through one-hot encoding. These were the marital, job, education, and contact columns. I made sure to specify which 'unknown' corresponded to which variable for easy interpretation later.

# In[ ]:


# transforming categorical variables
marital_labels = pd.get_dummies(train['marital'])
job_labels = pd.get_dummies(train['job'])
education_labels = pd.get_dummies(train['education'])
contact_labels = pd.get_dummies(train['contact'])

# making sure the unknowns have a specific label
marital_labels.columns = ['divorced', 'married', 'single', 'unknown.marital']
job_labels.columns = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
       'retired', 'self-employed', 'services', 'student', 'technician',
       'unemployed', 'unknown.job']
education_labels.columns = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
       'professional.course', 'university.degree', 'unknown.education']



# transforming test data
marital_labels2 = pd.get_dummies(test['marital'])
job_labels2 = pd.get_dummies(test['job'])
education_labels2 = pd.get_dummies(test['education'])
contact_labels2 = pd.get_dummies(test['contact'])

# making sure the unknowns have a specific label
marital_labels2.columns = ['divorced', 'married', 'single', 'unknown.marital']
job_labels2.columns = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
       'retired', 'self-employed', 'services', 'student', 'technician',
       'unemployed', 'unknown.job']
education_labels2.columns = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
       'professional.course', 'university.degree', 'unknown.education']


# Then it came time ot piece the training and testing data back together, with the numerical features from the original data and the modified ordinal/categoricla features just created.

# In[ ]:


train_y = train['y']
train2 = train[['id', 'age', 'duration', 'campaign','pdays', 'previous', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed']]

train2['default'] = default_labels
train2['housing'] = housing_labels
train2['loan'] = loan_labels
train2['month'] = month_labels
train2['day'] = day_labels
train2['poutcome'] = poutcome_labels

train2 = pd.concat([train2, marital_labels], axis=1)
train2 = pd.concat([train2, job_labels], axis=1)
train2 = pd.concat([train2, education_labels], axis=1)
train2 = pd.concat([train2, contact_labels], axis=1)

train2['y'] = train_y


# In[ ]:


test2 = test[['id', 'age', 'duration', 'campaign','pdays', 'previous', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed']]

test2['default'] = default_labels2
test2['housing'] = housing_labels2
test2['loan'] = loan_labels2
test2['month'] = month_labels2
test2['day'] = day_labels2
test2['poutcome'] = poutcome_labels2

test2 = pd.concat([test2, marital_labels2], axis=1)
test2 = pd.concat([test2, job_labels2], axis=1)
test2 = pd.concat([test2, education_labels2], axis=1)
test2 = pd.concat([test2, contact_labels2], axis=1)


# Then I began a journey of trying different feature selection methods to see what woudl yield the best result. I started by comparing the correlation of each feature with the y variable. Through that I created two new feature sets, one that had features with abs(corr)>0.01 and one wiht abs(corr)>0.05. Then I used all three feature sets to go throgh some basic models that I will touch on later.

# In[ ]:


corr = pd.DataFrame()
for a in list('y'):
    for b in list(train2.columns.values):
        corr.loc[b, a] = train2.corr().loc[a, b]
        
sns.heatmap(corr)
print(corr['y'].sort_values())

# variables with abs(corr)<0.01
'''
basic.4y              -0.009658
high.school           -0.009604
housemaid             -0.008696
self-employed         -0.008180
unknown.job           -0.003743
technician            -0.001249
management            -0.000280
loan                   0.000409
professional.course    0.000415
unknown.marital        0.002550
illiterate             0.007441
day                    0.008814
'''
train3 = train2[['age', 'duration', 'campaign', 'pdays', 'previous',
       'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
       'nr.employed', 'default', 'housing', 'month', 'poutcome',
       'divorced', 'married', 'single', 'admin.',
       'blue-collar', 'entrepreneur', 'retired',
       'services', 'student', 'unemployed',
       'basic.6y', 'basic.9y', 'university.degree',
       'unknown.education', 'cellular', 'telephone']]
test3 = test2[['age', 'duration', 'campaign', 'pdays', 'previous',
       'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
       'nr.employed', 'default', 'housing', 'month', 'poutcome',
       'divorced', 'married', 'single', 'admin.',
       'blue-collar', 'entrepreneur', 'retired',
       'services', 'student', 'unemployed',
       'basic.6y', 'basic.9y', 'university.degree',
       'unknown.education', 'cellular', 'telephone']]

# variables with abs(corr)<0.05
"""
basic.9y              -0.043711
married               -0.042574
services              -0.031471
basic.6y              -0.024711
entrepreneur          -0.016653
divorced              -0.010230
basic.4y              -0.009658
high.school           -0.009604
housemaid             -0.008696
self-employed         -0.008180
unknown.job           -0.003743
technician            -0.001249
management            -0.000280
loan                   0.000409
professional.course    0.000415
unknown.marital        0.002550
illiterate             0.007441
day                    0.008814
housing                0.011729
unemployed             0.014542
unknown.education      0.016053
age                    0.027631
admin.                 0.030412
month                  0.036602
"""
train4 = train2[['duration', 'campaign', 'pdays', 'previous',
       'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
       'nr.employed', 'default', 'poutcome', 'single', 'blue-collar','retired',
       'student', 'university.degree', 'cellular', 'telephone']]

test4 = test2[['duration', 'campaign', 'pdays', 'previous',
       'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
       'nr.employed', 'default', 'poutcome', 'single', 'blue-collar','retired',
       'student', 'university.degree', 'cellular', 'telephone']]


# Then I made my own split on the training set to have a train_training and train_testing set to use for checking the validity of the model. There are definitley better ways that taking the last 2000 rows as the testing set, but that was the easiest way to replicate across all datasets. The y split wil also be the same for all feature sets, so i only had to define that once.

# In[ ]:


train2_lite = train2.iloc[:-2000, :-1]
trainy_lite = train2.iloc[:-2000, -1]
train2_test = train2.iloc[-2000:, :-1]
trainy_test = train2.iloc[-2000:, -1]

train3_lite = train3.iloc[:-2000, :-1]
train3_test = train3.iloc[-2000:, :-1]

train4_lite = train4.iloc[:-2000, :-1]
train4_test = train4.iloc[-2000:, :-1]


# To start I performed a basic logistic regression on all 3 feature sets, and printed the results below. I used f1 score (which wants to be maximized) because I couldn't find the f mean score metric. The accuracy score is also a good measure of how well the model does at prediction. Further analysis could try different types of logistic regression other than 'liblinear'.

# In[ ]:


from sklearn.linear_model import LogisticRegression
logis = LogisticRegression(solver='liblinear',fit_intercept=True)

logis_test = logis.fit(train2_lite, trainy_lite)
preds = logis_test.predict(train2_test)
print(sklearn.metrics.f1_score(trainy_test, preds))
print(sklearn.metrics.accuracy_score(trainy_test, preds))

log_test2 = logis.fit(train3_lite, trainy_lite)
preds2 = log_test2.predict(train3_test)
print(sklearn.metrics.f1_score(trainy_test, preds2))
print(sklearn.metrics.accuracy_score(trainy_test, preds2))

log_test3 = logis.fit(train4_lite, trainy_lite)
preds3 = log_test3.predict(train4_test)
print(sklearn.metrics.f1_score(trainy_test, preds3))
print(sklearn.metrics.accuracy_score(trainy_test, preds3))


# Removing the features I did barely improved the model, so I tried selecting a different subset of features using an ANOVA F-test. I used all variables that were significant at an alpha=0.01 level and ran another logistic regression. 

# In[ ]:


from sklearn.feature_selection import f_regression
(F_vals, p_vals) = f_regression(train2_lite, trainy_lite)

cols = list(train2_lite.columns[p_vals<0.01])
trainF = train2[cols]

trainF_lite = trainF.iloc[:-2000, :-1]
trainF_test = trainF.iloc[-2000:, :-1]


# In[ ]:


log_testF = logis.fit(trainF_lite, trainy_lite)
predsF = log_testF.predict(trainF_test)
print(sklearn.metrics.f1_score(trainy_test, predsF))
print(sklearn.metrics.accuracy_score(trainy_test, predsF))
# no better than original


# So since that turned out any better than the original logistic regressions, I tried a new approach with Linear, Ridge, and Lasso regressions. (Spoiler this didn't really work out either) I messed around with different values of lambda, but the higher values caused the performance of the model to decrease, and the lower values were the same as the linear regression. This is something I could come back to and spend time refining, but I decided to see if there was a better group of features that I hadn't uncovered yet.

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


lr = LinearRegression()
linreg = lr.fit(train2_lite, trainy_lite)
predsLR = linreg.predict(train2_test)>0.32
print(sklearn.metrics.f1_score(trainy_test, predsLR))
print(sklearn.metrics.accuracy_score(trainy_test, predsLR))

rr = Ridge(alpha=0.000001, normalize=True)
ridge = rr.fit(train2_lite, trainy_lite)
predsR = ridge.predict(train2_test)>0.32
print(sklearn.metrics.f1_score(trainy_test, predsR))
print(sklearn.metrics.accuracy_score(trainy_test, predsR))

lasso = Lasso(alpha=0.00000000001, normalize=True)
lass = lasso.fit(train2_lite, trainy_lite)
predsLass = lass.predict(train2_test)>0.32
print(sklearn.metrics.f1_score(trainy_test, predsLass))
print(sklearn.metrics.accuracy_score(trainy_test, predsLass))


# Welcome to Decision trees! I did a Decision Tree Classification and Random Forest Classification to see what features they decided were important. My thought was that I could take the top 10 or so features that were of note and go back and try some of the simpler regressions with that new subset of features.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

tree = DecisionTreeClassifier()
treeD = tree.fit(train2_lite, trainy_lite)
predsTD = treeD.predict(train2_test)
print(sklearn.metrics.f1_score(trainy_test, predsTD))
print(sklearn.metrics.accuracy_score(trainy_test, predsTD))

forest = RandomForestClassifier(criterion = 'entropy')
forestR = forest.fit(train2_lite, trainy_lite)
predsF = forestR.predict(train2_test)
print(sklearn.metrics.f1_score(trainy_test, predsF))
print(sklearn.metrics.accuracy_score(trainy_test, predsF))


# In[ ]:


print(pd.DataFrame({'Gain': treeD.feature_importances_}, index = train2_lite.columns).sort_values('Gain', ascending = False))
print(pd.DataFrame({'Importance': forestR.feature_importances_}, index = train2_lite.columns).sort_values('Importance', ascending = False))


# I found that there was significant overlap between the two methods of determining feature importance. The top 10 features from decision tree and random forest are shown below with dashes next to features that were included in both.
# 
# Decision Tree
# duration               0.302235-
# id                     0.153253-
# age                    0.070679-
# euribor3m              0.053408-
# nr.employed            0.037945-
# pdays                  0.034321-
# campaign               0.032640-
# day                    0.030208-
# emp.var.rate           0.024689
# month                  0.021207
# 
# Random Forest
# duration             0.327945-
# nr.employed          0.154874-
# id                   0.116405-
# age                  0.074104-
# euribor3m            0.037914-
# campaign             0.032394-
# cons.conf.idx        0.023777
# day                  0.023379-
# pdays                0.022481-
# housing              0.013133
# 
# Using this imformation, I created a new set of the training data with all of these features shown above. the ones not included in both top 10 were within the top 15 of the other feature list, so I felt comfortable including them. I didn't include month however, because I felt that variable had too much going on with it.

# In[ ]:


trainT = train2[['duration', 'id', 'age', 'euribor3m', 'nr.employed', 
                 'pdays', 'campaign', 'day', 'cons.conf.idx', 'housing',
                'emp.var.rate']]
trainT_lite = trainT.iloc[:-2000,:]
trainT_test = trainT.iloc[-2000:, :]

testT = test2[['duration', 'id', 'age', 'euribor3m', 'nr.employed', 
                 'pdays', 'campaign', 'day', 'cons.conf.idx', 'housing',
                'emp.var.rate']]


# Then I went back to the two tree models and input the new set of features to see if the model performance improved. they actually performed slightly better with the removal of variables, so I think I reduced a bit of overfitting that was occuring. I'd be curious to go back and try parsing down the variables even more later.

# In[ ]:


treeD2 = tree.fit(trainT_lite, trainy_lite)
predsTD2 = treeD2.predict(trainT_test)
print(sklearn.metrics.f1_score(trainy_test, predsTD2))
print(sklearn.metrics.accuracy_score(trainy_test, predsTD2))


forest2 = RandomForestClassifier(criterion = 'gini')
forestR2 = forest2.fit(trainT_lite, trainy_lite)
predsF2 = forestR2.predict(trainT_test)
print(sklearn.metrics.f1_score(trainy_test, predsF2))
print(sklearn.metrics.accuracy_score(trainy_test, predsF2))


# I then ran a Linear/Ridge/Lasso/Bayes Regression on this new set of features selected by the trees, and I reached my best model so far. Again, I tried playing around with values of lambda, but I couldn't get a good compromise without reducing the model performance. I did adjust the cutoff value to 0.3 (instead of 0.5) for the Linear/Ridge/Lasso becuase that gave the bet accuracy and f1 score. My first submission was of this linear regression model.

# In[ ]:


linregT = lr.fit(trainT_lite, trainy_lite)
predsLRT = linregT.predict(trainT_test)>0.3
print(sklearn.metrics.f1_score(trainy_test, predsLRT))
print(sklearn.metrics.accuracy_score(trainy_test, predsLRT))


rr = Ridge(alpha=0.000001, normalize=True)
ridgeT = rr.fit(trainT_lite, trainy_lite)
predsRT = ridgeT.predict(trainT_test)>0.3
print(sklearn.metrics.f1_score(trainy_test, predsRT))
print(sklearn.metrics.accuracy_score(trainy_test, predsRT))


lassoT = Lasso(alpha=0.000001, normalize=True)
lassT = lassoT.fit(trainT_lite, trainy_lite)
predsLassT = lassT.predict(trainT_test)>0.3
print(sklearn.metrics.f1_score(trainy_test, predsLassT))
print(sklearn.metrics.accuracy_score(trainy_test, predsLassT))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit(trainT_lite, trainy_lite)
predsG = model.predict(trainT_test)
print(sklearn.metrics.f1_score(trainy_test, predsG))
print(sklearn.metrics.accuracy_score(trainy_test, predsG))


# I decided to run another logistic regression. It performed fairly well, and although the accuracy score was one of the highest to date, the f1 score was still fairly low, so I believe there still may be overfitting happening. I will go back later and try an even more reduced subset of variables.

# In[ ]:


logis_test2 = logis.fit(train2_lite, trainy_lite)
predsL = logis_test2.predict(train2_test)
print(sklearn.metrics.f1_score(trainy_test, predsL))
print(sklearn.metrics.accuracy_score(trainy_test, predsL))


# Next, I moved onto Support Vector Machines (SVM), which gave me my second best entry to date. This took a minute to run, but was really a

# In[ ]:


from sklearn.svm import SVC
from sklearn.linear_model import  LogisticRegression
classifier = SVC(kernel="linear")

svm = classifier.fit(trainT_lite, trainy_lite)
predsSVM = svm.predict(trainT_test)
print(sklearn.metrics.f1_score(trainy_test, predsSVM))
print(sklearn.metrics.accuracy_score(trainy_test, predsSVM))


# I also tried K-Nearest Neighbors which was fairly successful, but not as good as SVM or Linear Regression. 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier()
knn.fit(trainT_lite, trainy_lite)
predKNN = knn.predict(trainT_test)
print(sklearn.metrics.f1_score(trainy_test, predKNN))
print(sklearn.metrics.accuracy_score(trainy_test, predKNN))


# It was at this point that I was convinced that I needed to go back and parse down my feature subset even more, so I took only the features that were included in the the top 10 of both Decision Tree and Random Forest. No different selection of features improved Linear/Ridge/Lasso/Bayes/Logistic, so I went back to the drawing board. Something way back when I switched categorical variables to numeric probably want optimized.

# When we joined as a group, we attmpted imputing the unknown values, which was very difficult and did not get us anywhere. We also tried different methods of feature selection (stepwise, further reduction based on tree importance) and those did not yield good results either. We also tried oversampling to help balance out the number of successes and failures in the response column, but that actually hurt our model performance. 
# 
# 
# 
# Based on all of the attempts above, we submitted our top three models:
# 1. Linear regression with a cutoff of 0.3 on the top 13 features most important to decision tree and random forest
# 2. SVM on the top 13 features most important to decision tree and random forest
# 3. Random Forest on the top 13 features most important to decision tree and random forest
# 
# The code for producing the final predictions is shown below. The accuracy and f1score of the same models but with the training data can be found in the previous sections.

# In[ ]:


# 1. Linear Regression
lr = LinearRegression()
linregT = lr.fit(trainT, train.iloc[:, -1]) # best so far
prediction1 = linregT.predict(testT)>0.3

def TF(x):
  if x==True:
    return(1)
  elif x==False:
    return(0)

submission = pd.concat([test.id, pd.Series(prediction1)], axis = 1)
submission.columns = ['id', 'Predicted']
submission['Predicted'] = submission['Predicted'].apply(TF)
submission.to_csv('submission.csv', index=False)

# 2. SVM
from sklearn.svm import SVC
from sklearn.linear_model import  LogisticRegression
#classifier = SVC(kernel="linear")
#svm = classifier.fit(trainT, train.iloc[:, -1])
predictions3 = svm.predict(testT)
submission = pd.concat([test.id, pd.Series(predictions3)], axis = 1)
submission.columns = ['id', 'Predicted']
submission.to_csv('submission.csv', index=False)


# 3. Random Forest
forest2 = RandomForestClassifier(criterion = 'gini')
forestR2 = forest2.fit(trainT, train.iloc[:, -1])
predsF2 = forestR2.predict(testT)
submission = pd.concat([test.id, pd.Series(predsF2)], axis = 1)
submission.columns = ['id', 'Predicted']
submission.to_csv('submission.csv', index=False)

