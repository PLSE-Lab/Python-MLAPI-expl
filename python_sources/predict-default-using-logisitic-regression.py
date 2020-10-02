#!/usr/bin/env python
# coding: utf-8

# # Importing the dataset

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


df = pd.read_csv("../input/accepted_2007_to_2017Q3.csv.gz", compression='gzip')


# We are only going to look at loans that either have the loan status 'Default' or 'Fully Paid'

# In[ ]:


df = df[(df['loan_status']=='Fully Paid') | (df['loan_status']=='Charged Off')]


# Let's see how the dataset looks like.

# In[ ]:


df.head(5)


# # Data visualization and feature enigineering

# At first we transform the loan-status into a binary variable where 'Charged Off' = 1 and 'Fully Paid' = 0.

# In[ ]:


df['loan_status_bin'] = df['loan_status'].map({'Charged Off': 1, 'Fully Paid': 0})


# Which grades occur most often?

# In[ ]:


(df['grade'].value_counts().sort_index()/len(df)).plot.bar()


# Before we can have a look at the distribution of the employment length we need to transform it into numerical values.

# In[ ]:


def emp_to_num(term):
    if pd.isna(term):
        return None
    elif term[2]=='+':
        return 10
    elif term[0]=='<':
        return 0
    else:
        return int(term[0])

df['emp_length_num'] = df['emp_length'].apply(emp_to_num)
(df['emp_length_num'].value_counts().sort_index()/len(df)).plot.bar()


# Is there a connection between employment length and default rate?

# In[ ]:


df.groupby('emp_length_num')['loan_status_bin'].mean().plot.bar()


# Intrestingly there is no obvious connection between employment length and default rate. For example employment lengths of 8 and 9 years have almost the same default rate as employment lengths of up to 1 year.
# But what we can see is that an employment length of 0 or 1 has a high default rate and an emloyment length of more than 10 years has a really low default rate. So we are going to transform the feature into two features called 'long_emp' and 'short_emp'. 

# In[ ]:


df['long_emp'] = df['emp_length'].apply(lambda x: 1*(x=='10+ years'))
df['short_emp'] = df['emp_length'].apply(lambda x: 1*(x=='1 year' or x=='< 1 year'))


# Let's have a look at the distirubtion of the interest rates.

# In[ ]:


(df['int_rate']/len(df)).plot.hist(bins=10)


# Let's have a look at the distribution of the annual income (excluding incomes above 200,000). This seems so be log-normal-distributed so we do a log-transform..

# In[ ]:


df[df['annual_inc']<200000]['annual_inc'].plot.hist(bins=20)
df['annual_inc_log'] = df['annual_inc'].apply(np.log)


# Which are the most common reasons for requesting a loan on lending club?

# In[ ]:


(df['purpose'].value_counts()/len(df)).plot.bar()


# How high is the default rate for different grades? It looks like there is a linear connection between the grade an the default rate.

# In[ ]:


df.groupby('grade')['loan_status_bin'].mean().plot.line()


# How high ist the interest rate for different grades? Is there a linear connection too?

# In[ ]:


df.groupby('grade')['int_rate'].mean().plot.line(color='blue')


# As we have seen there is a linear connection between grade and default rate and also between grade and interest rate. But this meens that there is also a linear connection between the default rate and the interest rate. We will come back to this in the next section.

# As we can see there are only two possible values for the term on Lending Club, i.e. 36 months or 60 months, and months with a higher term have a significant higher default rate.

# In[ ]:


(df['term'].value_counts()/len(df)).plot.bar(title='value counts')


# In[ ]:


df.groupby('term')['loan_status_bin'].mean().plot.bar(title='default rate')


# Let's do the same for the home_ownership feature.

# In[ ]:


(df['home_ownership'].value_counts()/len(df)).plot.bar(title='value counts')


# In[ ]:


df[(df['home_ownership']=='MORTGAGE') | (df['home_ownership']=='OWN')| (df['home_ownership']=='RENT')].groupby('home_ownership')['loan_status_bin'].mean().plot.bar(title='default rate')


# Let's have a look at the distribution of the FICO score and the installment.

# In[ ]:


df['fico_range_high'].plot.hist(bins=20, title='FICO-Score')


# In[ ]:


df['installment'].plot.hist(bins=40, title='installment')


# # Describing the linear connection between default and interest rate
# 
# As said before there seems to be a linear connection between the interest rate and the default rate. We will use Linear Regression to calculate the linear function that maps from the default rate to the interest rate.

# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(df.groupby('sub_grade')['loan_status_bin'].mean().values.reshape(-1,1), y=df.groupby('sub_grade')['int_rate'].mean())


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(df.groupby('sub_grade')['loan_status_bin'].mean(), df.groupby('sub_grade')['int_rate'].mean())
plt.plot(df.groupby('sub_grade')['loan_status_bin'].mean(), lr.predict(df.groupby('sub_grade')['loan_status_bin'].mean().values.reshape(-1,1)))
plt.xlabel('default rate')
plt.ylabel('interest rate')


# In[ ]:


print('interest rate = ', lr.intercept_, '+', lr.coef_[0], '* default rate')


# # Training the Logistic Regression

# Reduce the dataset to the following columns that are known to investors **before** the loan is funded.

# In[ ]:


columns = ['loan_amnt', 'term', 'int_rate',
       'installment', 'grade', 'emp_length',
       'home_ownership', 'annual_inc_log', 'verification_status',
       'loan_status_bin', 'purpose',
       'addr_state', 'dti', 'delinq_2yrs',
       'fico_range_low', 'inq_last_6mths', 'open_acc',
       'pub_rec', 'revol_bal', 'revol_util', 'total_acc']
df = df[columns]


# Drop all rows that contain null-values.

# In[ ]:


df.dropna(inplace=True)


# Transform the grade into numerical values.

# In[ ]:


df['grade']=df['grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})


# Get the dummy-variables for categorical features.

# In[ ]:


df_dummies = pd.get_dummies(df)


# We are going to drop all dummy-variables which contain not at least 1% ones. In this case we can simply look at the mean of the features because all non-dummy variables have means greater than 0.01.

# In[ ]:


drop_columns = df_dummies.columns[(df_dummies.mean()<0.01)]
df_dummies.drop(drop_columns, axis=1, inplace=True)


# Add the two different verification status variables that indicate verified to one variable.

# In[ ]:


df_dummies['verification_status_Verified_sum'] = df_dummies['verification_status_Source Verified']+df_dummies['verification_status_Verified']
df_dummies.drop(['verification_status_Source Verified', 'verification_status_Verified'], axis=1, inplace=True)


# How does the transformed dataset look like?

# In[ ]:


df_dummies.head()


# Seperate features from targets.

# In[ ]:


X = df_dummies.drop('loan_status_bin', axis=1)
y = df_dummies['loan_status_bin']


# Split the data into training and testing data.

# In[ ]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Set up the pipeline. We will use $L_1$-penalty for built-in feature selectiob

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sc = MinMaxScaler()
clf = LogisticRegression(penalty='l1', C=0.01)

pipe_lr = Pipeline([('scaler', sc), ('clf', clf)])


# Train the logisitc regression model.

# In[ ]:


pipe_lr.fit(X_train, y_train)


# # Evaluating the Logistic Regression
# 
# Let's see how well the model works by plotting the ROC-curve and calculating the ROC-AUC-Score.

# In[ ]:


test_probas = pipe_lr.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

fpr, tpr, tresholds = roc_curve(y_test, test_probas)
plt.plot(fpr, tpr)
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')

print('ROC-AUC-score: ', roc_auc_score(y_test, test_probas))


# Let's see which features are most important.

# In[ ]:


for i in np.argsort(-np.abs(pipe_lr.named_steps['clf'].coef_[0])):
    print(X.columns[i], ': ', round(pipe_lr.named_steps['clf'].coef_[0,i], 4))


# In the next step we will see whats the connection between default and interest rate when using the Logistic Regression and compare it to the connection we got from the grades. One can control the risk of the investments by choosing different tresholds. The higher one chooses the probability where to seperate *good* from *bad* loans the higher the default rate will be. We now write a function that transforms default probabilities to predictions.
# $$ y_{pred} = \begin{cases} 1, \text{ if } P(y_{pred}=1)\geq\theta\\
# 0, text{ else } \end{cases}$$

# In[ ]:


def prob_to_pred(theta, proba):
    return [(p<theta) for p in proba]


# Now we calculate the mean interest rate and default rate for different thetas and compare it to the interest rate we would get by just looking at the grade.
# 
# 1. column: parameter $\theta$
# 2. column: default rate on the test data when using logisitc regression with treshold $\theta$
# 3. column: mean interest rate when using logisitc regression with treshold $\theta$
# 4. column: mean interes rate when looking just at the grade

# In[ ]:


probs = pipe_lr.predict_proba(X_test)[:,1]
for theta in np.arange(0.03,0.21,0.01):
    print('theta =', round(theta,2), end="  ")
    print(round(y_test.values[prob_to_pred(theta, probs)].mean(),2), end="  ")
    print(round(X_test.values[prob_to_pred(theta, probs), 1].mean(),2), end="  ")
    print(round(lr.predict(y_test.values[prob_to_pred(theta, probs)].mean().reshape(-1,1))[0],2))

