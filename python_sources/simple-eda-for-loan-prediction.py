#!/usr/bin/env python
# coding: utf-8

# *Following code sample and datasets has been refered from AV*

# Reading the train dataset using the pandas in a dataframe

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/loan-prediction/train_loan.csv")


# To check the first 3 rows from the dataframe.

# In[ ]:


df.head(3)


# To look the summary of the Numerical fields.

# In[ ]:


df.describe()


# In[ ]:


df.shape


# In[ ]:


quantitative = [f for f in df.columns if df.dtypes[f] != 'object']
print(quantitative)


# In[ ]:


fig,ax = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(),ax=ax,annot= True,linewidth= 0.02,fmt='.2f',cmap = 'Blues')
plt.show()


# **Distribution Analysis**

# In[ ]:


df['ApplicantIncome'].hist(bins=50)


# In[ ]:


df.boxplot(column='ApplicantIncome')


# In[ ]:


df.boxplot(column='ApplicantIncome', by = 'Education')


# In[ ]:


df['LoanAmount'].hist(bins=50)


# In[ ]:


df.boxplot(column='LoanAmount')


# **Probability of getting loan.**

# In[ ]:


temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print ('Frequency Table for Credit History:') 
print (temp1)

print ('\nProbility of getting loan for each Credit History class:')
print (temp2)


# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar',color='Black')


temp2.plot(kind = 'bar',color='Black')
ax2 = fig.add_subplot(122)
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")


# In[ ]:


temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['Red','black'], grid=False)


# **Checking the Missing values.**

# In[ ]:


df.apply(lambda x: sum(x.isnull()),axis=0) 


# In[ ]:


missing = df.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
print("Number of attributes having missing values " + str(len(missing)))


# Now we are going replace the null values with the mean of loan amount.

# In[ ]:


df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['LoanAmount'].isnull().sum()


# We will percentage of employment

# In[ ]:


df['Self_Employed'].value_counts()


# In[ ]:


df['Self_Employed'].isnull().sum()


# From the above value count almost 80% of persons not self employed,so we will replace the missing values with 'NO'

# In[ ]:


df['Self_Employed'].fillna('NO', inplace=True)
df['Self_Employed'].isnull().sum()


# Now we are goind to replace the null values in Loanamount column with the median values using the function.

# In[ ]:


#table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
# Define function to return value of this pivot_table
#def fage(x):
 #return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values
#df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)


# In[ ]:


df['LoanAmount'].hist(bins=50)


# In[ ]:


df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)


# Sometimes loan will be also give based on the co applicant income , so we are adding applicant income and co applicant income.

# In[ ]:


df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=20) 


# Replacing the other column null values with mode values

# In[ ]:


df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)


# As sklearn requires all inputs with numeric , we should convert all our categorical varialbe into numeric by encoding categories.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes 


# In[ ]:


#import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_splits=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome])


# In[ ]:


outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model, df,predictor_var,outcome_var)


# In[ ]:


model = DecisionTreeClassifier()
predictor_var = ['Credit_History','Gender','Married','Education']
classification_model(model, df,predictor_var,outcome_var)

