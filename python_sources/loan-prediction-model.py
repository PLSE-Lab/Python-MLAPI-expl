#!/usr/bin/env python
# coding: utf-8

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

import matplotlib as plt
data=pd.read_csv("../input/loanPred_train_data.csv")


# Any results you write to the current directory are saved as output.


# 

# Data is added now.Also the required libraries are imported. Now lets get some insights in to the data.

# **Data Exploration**

# Lets findout whats there in the top 10 rows of the data.

# In[ ]:


data.head(10)


# Lets findout some statics about the data

# In[ ]:


data.describe()


# From the above, we can see that there are 614 rows in 'ApplicantIncome' and 'CoapplicantIncome'.But other columns have less rows.That means there are missing values.
# Also, the mean (average) applicant income is '5403.459283' and mean Loan amount is '146.412162'.Also, we can infer that 84.2% of the applicant have credit history.This can be infered from the mean value of 'Credit_History' column.
# Now, lets get the type of variables.

# In[ ]:


data.info()


# **Distribution Analysis**   
# 
# Lets understnad the distribution of varibales.
# For 'ApplicantIncome'
# 

# In[ ]:


data['ApplicantIncome'].hist()


# The graph shows some extreme values.Also distribution ins not cliear.So,lets use the bins to plot the graph.

# In[ ]:


data['ApplicantIncome'].hist(bins=50)


# The above graph is show more clear distribution.To see the extreme values(outliers), lets plot the box plot.

# In[ ]:


data.boxplot('ApplicantIncome')


# Hmm..Too many outliers.This is due the wide spread of the income based on education or some other factors.Lets compare based on the education groups.

# In[ ]:


data.boxplot('ApplicantIncome',by='Education')


# So, the applicant income is more spread for Gradutes than the Non-graduates.But mean income is almost same.

# *Loan Amount distrinution.*

# In[ ]:


data['LoanAmount'].hist(bins=100)


# In[ ]:


data.boxplot('LoanAmount')


# As per the graphs, thre are many outliers in the Loan Amount.
# Also, there are many missing values in the 'LoanAmount'column.This requires data pre-processing.We will do that shortly.

# *Categorical Variables* <br>
# Lets us look at the chances of getting loan based on Credit History.<br>
# We will set value of '1' for loan status as 'Yes' and '0' for 'No'.

# In[ ]:


temp1 = data['Credit_History'].value_counts(ascending=True)
temp2 = data.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print ('Frequency Table for Credit History:')
print(temp1)


# Get the probability of getting Loan based on Credit history

# In[ ]:


print ('\nProbility of getting loan for each Credit History class:')
print (temp2)


# As we can see, there is 79.57 % probability of getting loan when applicant has credit history

# **Data Preperation**

# Data has missing values and has outliers. So we need to fix these.

# *Treating Missing Values*

# In[ ]:


data.apply(lambda x: sum(x.isnull()),axis=0)


# In[ ]:


#Addin LoanAmount missing values
data['LoanAmount'].fillna(data['LoanAmount'].mean(),inplace=True)


# In[ ]:


#Verify the Missing Values in LoanAmount
data.apply(lambda x: sum(x.isnull()),axis=0)


# Add the missing values in all major columns

# In[ ]:


data.fillna(data.mean(),inplace=True)


# In[ ]:


#Verify the Missing Values
data.apply(lambda x: sum(x.isnull()),axis=0)


# Add the missing values for Self_Employed

# In[ ]:


data['Self_Employed'].fillna('No',inplace=True)


# In[ ]:


#Verify the Missing Values
data.apply(lambda x: sum(x.isnull()),axis=0)


# **Treating Missing Values**

# *Loanamount*<br>
# Since Loan amount can vary from smaller to bigger range, we need to transform the values. We will use logarthamic transformation.

# In[ ]:


data['LoanAmount_log']=np.log(data['LoanAmount'])


# In[ ]:


#Histogram
data['LoanAmount_log'].hist()


# Distribuation looks almost normal.This neutralized the outliers.

# *ApplicantIncome*<br>
# We need to loook and combining both applicant and co-applicant data

# In[ ]:


#combining incomes
data['TotalIncome']=data['ApplicantIncome']+data['CoapplicantIncome']


# In[ ]:


#Lets look at top 5 values of combined incomes
data['TotalIncome'].head(5)


# Since income also can vary with a large range, we need to transform to bring to same scale. Again will use log transformation.

# In[ ]:


#Log Transformation
data['TotalIncome_log']=np.log(data['TotalIncome'])


# In[ ]:


#Histogram
data['TotalIncome_log'].hist()


# Distribution is very near to normal.

# Lets replace missing values of other variables

# In[ ]:


data['Gender'].fillna(data['Gender'].mode()[0],inplace=True)
data['Married'].fillna(data['Married'].mode()[0],inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0],inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0],inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0],inplace=True)


# In[ ]:


#Check missing values
data.apply(lambda x: sum(x.isnull()),axis=0)


# So all missing values are replaced

# **Modelling**

# Convert all the categorical variables to numeric

# In[ ]:


from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])
data.dtypes 


# We will use the Scikit learn and import all the required libraries

# In[ ]:


#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


# Next,we will define a generic classification function, which takes a model as input and determines the Accuracy and Cross-Validation scores.

# In[ ]:


#Function for making a classification model and verifying the performance
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 


# **Logistic Regression**

# We can consider few variables based on the inution for prediction of Loan.

# In[ ]:


outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(model, data,predictor_var,outcome_var)


# So, the accuracy is 80.95%

# In[ ]:


#We can try different combination of variables:
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model, data,predictor_var,outcome_var)


# Again accuary is same

# **Decision Tree**

# In[ ]:


model = DecisionTreeClassifier()
predictor_var = ['Credit_History','Gender','Married','Education']
classification_model(model, data,predictor_var,outcome_var)


# So,the accuracy is 81.27%. Slightly better the Logistic regression

# In[ ]:




