#!/usr/bin/env python
# coding: utf-8

# 
# 
# The analysis is divided into four main parts:
# 1. Data understanding 
# 2. Data cleaning (cleaning missing values, removing redundant columns etc.)
# 3. Data Analysis 
# 4. Recommendations
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

loan = pd.read_csv("../input/credit_assessment_data.csv", sep=",")
loan.info()


# ## Data Understanding

# In[ ]:


# let's look at the first few rows of the df
loan.head()


# In[ ]:


# Looking at all the column names
loan.columns


# Some of the important columns in the dataset are loan_amount, term, interest rate, grade, sub grade, annual income, purpose of the loan etc.
# 
# The **target variable**, which we want to compare across the independent variables, is loan status. The strategy is to figure out compare the average default rates across various independent variables and identify the  ones that affect default rate the most.
# 
# 

# # Data Cleaning
# 
# Some columns have a large number of missing values, let's first fix the missing values and then check for other types of data quality problems.

# In[ ]:


# summarising number of missing values in each column
loan.isnull().sum()


# In[ ]:


# percentage of missing values in each column
round(loan.isnull().sum()/len(loan.index), 2)*100


# You can see that many columns have 100% missing values, some have 65%, 33% etc. First, let's get rid of the columns having 100% missing values.

# In[ ]:


# removing the columns having more than 90% missing values
missing_columns = loan.columns[100*(loan.isnull().sum()/len(loan.index)) > 90]
print(missing_columns)


# In[ ]:


loan = loan.drop(missing_columns, axis=1)
print(loan.shape)


# In[ ]:


# summarise number of missing values again
100*(loan.isnull().sum()/len(loan.index))


# In[ ]:


# There are now 2 columns having approx 32 and 64% missing values - 
# description and months since last delinquent

# let's have a look at a few entries in the columns
loan.loc[:, ['desc', 'mths_since_last_delinq']].head()


# The column description contains the comments the applicant had written while applying for the loan. Although one can use some text analysis techniques to derive new features from this column (such as sentiment, number of positive/negative words etc.), we will not use this column in this analysis. 
# 
# There is an important reason we shouldn't use this column in analysis - since at the time of loan application, we will not have this data (it gets generated months after the loan has been approved), it cannot be used as a predictor of default at the time of loan approval. 
# 
# Thus let's drop the two columns.

# In[ ]:


# dropping the two columns
loan = loan.drop(['desc', 'mths_since_last_delinq'], axis=1)


# In[ ]:


# summarise number of missing values again
100*(loan.isnull().sum()/len(loan.index))


# There are some more columns with missing values, but let's ignore them for now (since we are not doing any modeling, we don't need to impute all missing values anyway). 
# 
# But let's check whether some rows have a large number of missing values.

# In[ ]:


# missing values in rows
loan.isnull().sum(axis=1)


# In[ ]:


# checking whether some rows have more than 5 missing values
len(loan[loan.isnull().sum(axis=1) > 5].index)


# The data looks clean by and large. Let's also check whether all columns are in the correct format.

# In[ ]:


loan.info()


# In[ ]:


# The column int_rate is character type, let's convert it to float
loan['int_rate'] = loan['int_rate'].apply(lambda x: pd.to_numeric(x.split("%")[0]))


# In[ ]:


# checking the data types
loan.info()


# In[ ]:


# also, lets extract the numeric part from the variable employment length

# first, let's drop the missing values from the column (otherwise the regex code below throws error)
loan = loan[~loan['emp_length'].isnull()]

# using regular expression to extract numeric values from the string
import re
loan['emp_length'] = loan['emp_length'].apply(lambda x: re.findall('\d+', str(x))[0])

# convert to numeric
loan['emp_length'] = loan['emp_length'].apply(lambda x: pd.to_numeric(x))


# In[ ]:


# looking at type of the columns again
loan.info()


# ## Data Analysis
# 
# Let's now move to data analysis. To start with, let's understand the objective of the analysis clearly and identify the variables that we want to consider for analysis. 
# 
# The objective is to identify predictors of default so that at the time of loan application, we can use those variables for approval/rejection of the loan. Now, there are broadly three types of variables - 1. those which are related to the applicant (demographic variables such as age, occupation, employment details etc.), 2. loan characteristics (amount of loan, interest rate, purpose of loan etc.) and 3. Customer behaviour variables (those which are generated after the loan is approved such as delinquent 2 years, revolving balance, next payment date etc.).
# 
# Now, the customer behaviour variables are not available at the time of loan application, and thus they cannot be used as predictors for credit approval. 
# 
# Thus, going forward, we will use only the other two types of variables.
# 
# 

# In[ ]:


behaviour_var =  [
  "delinq_2yrs",
  "earliest_cr_line",
  "inq_last_6mths",
  "open_acc",
  "pub_rec",
  "revol_bal",
  "revol_util",
  "total_acc",
  "out_prncp",
  "out_prncp_inv",
  "total_pymnt",
  "total_pymnt_inv",
  "total_rec_prncp",
  "total_rec_int",
  "total_rec_late_fee",
  "recoveries",
  "collection_recovery_fee",
  "last_pymnt_d",
  "last_pymnt_amnt",
  "last_credit_pull_d",
  "application_type"]
behaviour_var


# In[ ]:


# let's now remove the behaviour variables from analysis
df = loan.drop(behaviour_var, axis=1)
df.info()


# Typically, variables such as acc_now_delinquent, chargeoff within 12 months etc. (which are related to the applicant's past loans) are available from the credit bureau. 

# In[ ]:


# also, we will not be able to use the variables zip code, address, state etc.
# the variable 'title' is derived from the variable 'purpose'
# thus let get rid of all these variables as well

df = df.drop(['title', 'url', 'zip_code', 'addr_state'], axis=1)


# Next, let's have a look at the target variable - loan_status. We need to relabel the values to a binary form - 0 or 1, 1 indicating that the person has defaulted and 0 otherwise.
# 
# 

# In[ ]:


df['loan_status'] = df['loan_status'].astype('category')
df['loan_status'].value_counts()


# You can see that fully paid comprises most of the loans. The ones marked 'current' are neither fully paid not defaulted, so let's get rid of the current loans. Also, let's tag the other two values as 0 or 1. 

# In[ ]:


# filtering only fully paid or charged-off
df = df[df['loan_status'] != 'Current']
df['loan_status'] = df['loan_status'].apply(lambda x: 0 if x=='Fully Paid' else 1)

# converting loan_status to integer type
df['loan_status'] = df['loan_status'].apply(lambda x: pd.to_numeric(x))

# summarising the values
df['loan_status'].value_counts()


# Next, let's start with univariate analysis and then move to bivariate analysis.
# 
# ## Univariate Analysis
# 
# First, let's look at the overall default rate.
# 

# In[ ]:


# default rate
round(np.mean(df['loan_status']), 2)


# The overall default rate is about 14%.  

# Let's first visualise the average default rates across categorical variables.
# 

# In[ ]:


# plotting default rates across grade of the loan
sns.barplot(x='grade', y='loan_status', data=df)
plt.show()


# In[ ]:


# lets define a function to plot loan_status across categorical variables
def plot_cat(cat_var):
    sns.barplot(x=cat_var, y='loan_status', data=df)
    plt.show()
    


# In[ ]:


# compare default rates across grade of loan
plot_cat('grade')


# Clearly, as the grade of loan goes from A to G, the default rate increases. This is expected because the grade is decided by Lending Club based on the riskiness of the loan. 

# In[ ]:


# term: 60 months loans default more than 36 months loans
plot_cat('term')


# In[ ]:


# sub-grade: as expected - A1 is better than A2 better than A3 and so on 
plt.figure(figsize=(16, 6))
plot_cat('sub_grade')


# In[ ]:


# home ownership: not a great discriminator
plot_cat('home_ownership')


# In[ ]:


# verification_status: surprisingly, verified loans default more than not verifiedb
plot_cat('verification_status')


# In[ ]:


# purpose: small business loans defualt the most, then renewable energy and education
plt.figure(figsize=(16, 6))
plot_cat('purpose')


# In[ ]:


# let's also observe the distribution of loans across years
# first lets convert the year column into datetime and then extract year and month from it
df['issue_d'].head()


# In[ ]:


from datetime import datetime
df['issue_d'] = df['issue_d'].apply(lambda x: datetime.strptime(x, '%b-%y'))


# In[ ]:


# extracting month and year from issue_date
df['month'] = df['issue_d'].apply(lambda x: x.month)
df['year'] = df['issue_d'].apply(lambda x: x.year)



# In[ ]:


# let's first observe the number of loans granted across years
df.groupby('year').year.count()


# You can see that the number of loans has increased steadily across years. 

# In[ ]:


# number of loans across months
df.groupby('month').month.count()


# Most loans are granted in December, and in general in the latter half of the year.

# In[ ]:


# lets compare the default rates across years
# the default rate had suddenly increased in 2011, inspite of reducing from 2008 till 2010
plot_cat('year')


# In[ ]:


# comparing default rates across months: not much variation across months
plt.figure(figsize=(16, 6))
plot_cat('month')


# Let's now analyse how the default rate varies across continuous variables.

# In[ ]:


# loan amount: the median loan amount is around 10,000
sns.distplot(df['loan_amnt'])
plt.show()


# The easiest way to analyse how default rates vary across continous variables is to bin the variables into discrete categories.
# 
# Let's bin the loan amount variable into small, medium, high, very high.

# In[ ]:


# binning loan amount
def loan_amount(n):
    if n < 5000:
        return 'low'
    elif n >=5000 and n < 15000:
        return 'medium'
    elif n >= 15000 and n < 25000:
        return 'high'
    else:
        return 'very high'
        
df['loan_amnt'] = df['loan_amnt'].apply(lambda x: loan_amount(x))


# In[ ]:


df['loan_amnt'].value_counts()


# In[ ]:


# let's compare the default rates across loan amount type
# higher the loan amount, higher the default rate
plot_cat('loan_amnt')


# In[ ]:


# let's also convert funded amount invested to bins
df['funded_amnt_inv'] = df['funded_amnt_inv'].apply(lambda x: loan_amount(x))


# In[ ]:


# funded amount invested
plot_cat('funded_amnt_inv')


# In[ ]:


# lets also convert interest rate to low, medium, high
# binning loan amount
def int_rate(n):
    if n <= 10:
        return 'low'
    elif n > 10 and n <=15:
        return 'medium'
    else:
        return 'high'
    
    
df['int_rate'] = df['int_rate'].apply(lambda x: int_rate(x))


# In[ ]:


# comparing default rates across rates of interest
# high interest rates default more, as expected
plot_cat('int_rate')


# In[ ]:


# debt to income ratio
def dti(n):
    if n <= 10:
        return 'low'
    elif n > 10 and n <=20:
        return 'medium'
    else:
        return 'high'
    

df['dti'] = df['dti'].apply(lambda x: dti(x))


# In[ ]:


# comparing default rates across debt to income ratio
# high dti translates into higher default rates, as expected
plot_cat('dti')


# In[ ]:


# funded amount
def funded_amount(n):
    if n <= 5000:
        return 'low'
    elif n > 5000 and n <=15000:
        return 'medium'
    else:
        return 'high'
    
df['funded_amnt'] = df['funded_amnt'].apply(lambda x: funded_amount(x))


# In[ ]:


plot_cat('funded_amnt')


# In[ ]:


# installment
def installment(n):
    if n <= 200:
        return 'low'
    elif n > 200 and n <=400:
        return 'medium'
    elif n > 400 and n <=600:
        return 'high'
    else:
        return 'very high'
    
df['installment'] = df['installment'].apply(lambda x: installment(x))


# In[ ]:


# comparing default rates across installment
# the higher the installment amount, the higher the default rate
plot_cat('installment')


# In[ ]:


# annual income
def annual_income(n):
    if n <= 50000:
        return 'low'
    elif n > 50000 and n <=100000:
        return 'medium'
    elif n > 100000 and n <=150000:
        return 'high'
    else:
        return 'very high'

df['annual_inc'] = df['annual_inc'].apply(lambda x: annual_income(x))


# In[ ]:


# annual income and default rate
# lower the annual income, higher the default rate
plot_cat('annual_inc')


# In[ ]:


# employment length
# first, let's drop the missing value observations in emp length
df = df[~df['emp_length'].isnull()]

# binning the variable
def emp_length(n):
    if n <= 1:
        return 'fresher'
    elif n > 1 and n <=3:
        return 'junior'
    elif n > 3 and n <=7:
        return 'senior'
    else:
        return 'expert'

df['emp_length'] = df['emp_length'].apply(lambda x: emp_length(x))


# In[ ]:


# emp_length and default rate
# not much of a predictor of default
plot_cat('emp_length')


# ## Segmented Univariate Analysis
# 
# We have now compared the default rates across various variables, and some of the important predictors are purpose of the loan, interest rate, annual income, grade etc.
# 
# In the credit industry, one of the most important factors affecting default is the purpose of the loan - home loans perform differently than credit cards, credit cards are very different from debt consolidation loans etc. 
# 
# This comes from business understanding, though let's again have a look at the default rates across the purpose of the loan.
# 

# In[ ]:


# purpose: small business loans defualt the most, then renewable energy and education
plt.figure(figsize=(16, 6))
plot_cat('purpose')


# In the upcoming analyses, we will segment the loan applications across the purpose of the loan, since that is a variable affecting many other variables - the type of applicant, interest rate, income, and finally the default rate. 

# In[ ]:


# lets first look at the number of loans for each type (purpose) of the loan
# most loans are debt consolidation (to repay otehr debts), then credit card, major purchase etc.
plt.figure(figsize=(16, 6))
sns.countplot(x='purpose', data=df)
plt.show()


# Let's analyse the top 4 types of loans based on purpose: consolidation, credit card, home improvement and major purchase.

# In[ ]:


# filtering the df for the 4 types of loans mentioned above
main_purposes = ["credit_card","debt_consolidation","home_improvement","major_purchase"]
df = df[df['purpose'].isin(main_purposes)]
df['purpose'].value_counts()


# In[ ]:


# plotting number of loans by purpose 
sns.countplot(x=df['purpose'])
plt.show()


# In[ ]:


# let's now compare the default rates across two types of categorical variables
# purpose of loan (constant) and another categorical variable (which changes)

plt.figure(figsize=[10, 6])
sns.barplot(x='term', y="loan_status", hue='purpose', data=df)
plt.show()


# In[ ]:


# lets write a function which takes a categorical variable and plots the default rate
# segmented by purpose 

def plot_segmented(cat_var):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cat_var, y='loan_status', hue='purpose', data=df)
    plt.show()

    
plot_segmented('term')


# In[ ]:


# grade of loan
plot_segmented('grade')


# In[ ]:


# home ownership
plot_segmented('home_ownership')


# In general, debt consolidation loans have the highest default rates. Lets compare across other categories as well.

# In[ ]:


# year
plot_segmented('year')


# In[ ]:


# emp_length
plot_segmented('emp_length')


# In[ ]:


# loan_amnt: same trend across loan purposes
plot_segmented('loan_amnt')


# In[ ]:


# interest rate
plot_segmented('int_rate')


# In[ ]:


# installment
plot_segmented('installment')


# In[ ]:


# debt to income ratio
plot_segmented('dti')


# In[ ]:


# annual income
plot_segmented('annual_inc')


# A good way to quantify th effect of a categorical variable on default rate is to see 'how much does the default rate vary across the categories'. 
# 
# Let's see an example using annual_inc as the categorical variable.

# In[ ]:


# variation of default rate across annual_inc
df.groupby('annual_inc').loan_status.mean().sort_values(ascending=False)


# In[ ]:


# one can write a function which takes in a categorical variable and computed the average 
# default rate across the categories
# It can also compute the 'difference between the highest and the lowest default rate' across the 
# categories, which is a decent metric indicating the effect of the varaible on default rate

def diff_rate(cat_var):
    default_rates = df.groupby(cat_var).loan_status.mean().sort_values(ascending=False)
    return (round(default_rates, 2), round(default_rates[0] - default_rates[-1], 2))

default_rates, diff = diff_rate('annual_inc')
print(default_rates) 
print(diff)


# Thus, there is a 6% increase in default rate as you go from high to low annual income. We can compute this difference for all the variables and roughly identify the ones that affect default rate the most.

# In[ ]:


# filtering all the object type variables
df_categorical = df.loc[:, df.dtypes == object]
df_categorical['loan_status'] = df['loan_status']

# Now, for each variable, we can compute the incremental diff in default rates
print([i for i in df.columns])


# In[ ]:


# storing the diff of default rates for each column in a dict
d = {key: diff_rate(key)[1]*100 for key in df_categorical.columns if key != 'loan_status'}
print(d)


# In[ ]:


df.shape


# In[ ]:


df['loan_status'].value_counts()


# In[ ]:


df = df.dropna()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


# drop id,member_id,emp_title
df.drop(['id','member_id','emp_title'],axis=1,inplace=True)


# ### Making Prediction

# In[ ]:


# import required libraries
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from imblearn.metrics import sensitivity_specificity_support
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


# In[ ]:


le = preprocessing.LabelEncoder()


# In[ ]:


df = df.apply(le.fit_transform)


# In[ ]:


df.head()


# In[ ]:


X = df.drop('loan_status',axis=1)
y = df.loan_status
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state = 4, stratify = y)


# In[ ]:


# print shapes of train and test sets
X_train.shape
y_train.shape
X_test.shape
y_test.shape


# ## Model Building

# In[ ]:


import statsmodels.api as sm


# In[ ]:


# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# ### Feature Selection using RFE

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[ ]:


from sklearn.feature_selection import RFE
rfe = RFE(logreg, 15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[ ]:


rfe.support_


# In[ ]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[ ]:


col = X_train.columns[rfe.support_]


# In[ ]:


X_train.columns[~rfe.support_]


# ### Assessing the model with StatsModel

# In[ ]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[ ]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[ ]:


y_train_pred_final = pd.DataFrame({'loan_status':y_train.values, 'loan_status_prob':y_train_pred})
y_train_pred_final['CustID'] = y_train.index
y_train_pred_final.head()


# #### Creating new column 'predicted' with 1 if loan_status_prob > 0.5 else 0

# In[ ]:


y_train_pred_final['predicted'] = y_train_pred_final.loan_status_prob.map(lambda x: 1 if x > 0.3 else 0)

# Let's see the head
y_train_pred_final.head()


# In[ ]:


from sklearn import metrics


# In[ ]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.loan_status, y_train_pred_final.predicted )
print(confusion)


# In[ ]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.loan_status, y_train_pred_final.predicted))


# #### Checking VIFs

# In[ ]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# There are a few variables with high VIF. It's best to drop these variables as they aren't helping much with prediction and unnecessarily making the model complex. The variable 'PhoneService' has the highest VIF. So let's start by dropping that.

# In[ ]:


col = col.drop('issue_d',1)
col


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[ ]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[ ]:


y_train_pred[:10]


# In[ ]:


y_train_pred_final['loan_status_prob'] = y_train_pred


# In[ ]:


# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.loan_status_prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[ ]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.loan_status, y_train_pred_final.predicted))


# So ,Overall accuracy increases

# ### Checking VIF again

# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


# Let's drop TotalCharges since it has a high VIF
col = col.drop('sub_grade')
col


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[ ]:


y_train_pred[:10]


# In[ ]:


y_train_pred_final['loan_status_prob'] = y_train_pred


# In[ ]:


# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.loan_status_prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[ ]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.loan_status, y_train_pred_final.predicted))


# ### Let's now check the VIFs again

# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


# Let's drop TotalCharges since it has a high VIF
col = col.drop('year')
col


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[ ]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[ ]:


y_train_pred[:10]


# In[ ]:


y_train_pred_final['loan_status_prob'] = y_train_pred


# In[ ]:


# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.loan_status_prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[ ]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.loan_status,y_train_pred_final.predicted))


# ### Metrics beyond simply accuracy

# In[ ]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[ ]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[ ]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[ ]:


# Let us calculate specificity
TN / float(TN+FP)


# In[ ]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[ ]:


# positive predictive value 
print (TP / float(TP+FP))


# In[ ]:


# Negative predictive value
print (TN / float(TN+ FN))


# ### Plotting the ROC Curve

# An ROC curve demonstrates several things:
# 
# - It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).
# - The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.
# - The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.

# In[ ]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[ ]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.loan_status, y_train_pred_final.loan_status_prob, drop_intermediate = False )


# In[ ]:


draw_roc(y_train_pred_final.loan_status, y_train_pred_final.loan_status_prob)


# ### Finding Optimal CutOff point

# Optimal cutoff probability is that prob where we get balanced sensitivity and specificity

# In[ ]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.loan_status_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# ### PCA

# In[ ]:


# apply pca to train data
pca = Pipeline([('scaler', StandardScaler()), ('pca', PCA())])


# In[ ]:


pca.fit(X_train)
churn_pca = pca.fit_transform(X_train)


# In[ ]:


# extract pca model from pipeline
pca = pca.named_steps['pca']

# look at explainded variance of PCA components
print(pd.Series(np.round(pca.explained_variance_ratio_.cumsum(), 4)*100))


# ~ 12 components explain 90% variance
# 
# ~ 15 components explain 95% variance

# In[ ]:


# plot feature variance
features = range(pca.n_components_)
cumulative_variance = np.round(np.cumsum(pca.explained_variance_ratio_)*100, decimals=4)
plt.figure(figsize=(175/20,100/20)) # 100 elements on y-axis; 175 elements on x-axis; 20 is normalising factor
plt.plot(cumulative_variance)


# ### PCA and Logistic Regression

# In[ ]:


# create pipeline
PCA_VARS = 10
steps = [('scaler', StandardScaler()),
         ("pca", PCA(n_components=PCA_VARS)),
         ("logistic", LogisticRegression(class_weight='balanced'))
        ]
pipeline = Pipeline(steps)


# In[ ]:


# fit model
pipeline.fit(X_train, y_train)

# check score on train data
pipeline.score(X_train, y_train)


# ### Evaluate on test data

# In[ ]:


# predict churn on test data
y_pred = pipeline.predict(X_test)

# create onfusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')

# check area under curve
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
print("AUC:    \t", round(roc_auc_score(y_test, y_pred_prob),2))


# ### Hyperparameter tuning - PCA and Logistic Regression

# In[ ]:


# class imbalance
y_train.value_counts()/y_train.shape


# In[ ]:


# PCA
pca = PCA()

# logistic regression - the class weight is used to handle class imbalance - it adjusts the cost function
logistic = LogisticRegression(class_weight={0:0.1, 1: 0.9})

# create pipeline
steps = [("scaler", StandardScaler()), 
         ("pca", pca),
         ("logistic", logistic)
        ]

# compile pipeline
pca_logistic = Pipeline(steps)

# hyperparameter space
params = {'pca__n_components': [10, 15], 'logistic__C': [0.1, 0.5, 1, 2, 3, 4, 5, 10], 'logistic__penalty': ['l1', 'l2']}

# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)

# create gridsearch object
model = GridSearchCV(estimator=pca_logistic, cv=folds, param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=1)


# In[ ]:


# fit model
model.fit(X_train, y_train)


# In[ ]:


# cross validation results
pd.DataFrame(model.cv_results_)


# In[ ]:


# print best hyperparameters
print("Best AUC: ", model.best_score_)
print("Best hyperparameters: ", model.best_params_)


# In[ ]:


# predict churn on test data
y_pred = model.predict(X_test)

# create onfusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')

# check area under curve
y_pred_prob = model.predict_proba(X_test)[:, 1]
print("AUC:    \t", round(roc_auc_score(y_test, y_pred_prob),2))


# ### Random Forest

# In[ ]:


# random forest - the class weight is used to handle class imbalance - it adjusts the cost function
forest = RandomForestClassifier(class_weight={0:0.1, 1: 0.9}, n_jobs = -1)

# hyperparameter space
params = {"criterion": ['gini', 'entropy'], "max_features": ['auto', 0.4]}

# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)

# create gridsearch object
model = GridSearchCV(estimator=forest, cv=folds, param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=1)


# In[ ]:


# fit model
model.fit(X_train, y_train)


# In[ ]:


# print best hyperparameters
print("Best AUC: ", model.best_score_)
print("Best hyperparameters: ", model.best_params_)


# In[ ]:


# predict churn on test data
y_pred = model.predict(X_test)

# create onfusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')

# check area under curve
y_pred_prob = model.predict_proba(X_test)[:, 1]
print("AUC:    \t", round(roc_auc_score(y_test, y_pred_prob),2))


# ### Choosing Best Features

# In[ ]:


# run a random forest model on train data
max_features = int(round(np.sqrt(X_train.shape[1])))    # number of variables to consider to split each node
print(max_features)

rf_model = RandomForestClassifier(n_estimators=100, max_features=max_features, class_weight={0:0.1, 1: 0.9}, oob_score=True, random_state=4, verbose=1)


# In[ ]:


# fit model
rf_model.fit(X_train, y_train)


# In[ ]:


# OOB score
rf_model.oob_score_


# In[ ]:


# predict churn on test data
y_pred = rf_model.predict(X_test)

# create onfusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')

# check area under curve
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
print("ROC:    \t", round(roc_auc_score(y_test, y_pred_prob),2))


# ### Feature Importance

# In[ ]:


# predictors
features = df.drop('loan_status', axis=1).columns

# feature_importance
importance = rf_model.feature_importances_

# create dataframe
feature_importance = pd.DataFrame({'variables': features, 'importance_percentage': importance*100})
feature_importance = feature_importance[['variables', 'importance_percentage']]

# sort features
feature_importance = feature_importance.sort_values('importance_percentage', ascending=False).reset_index(drop=True)
print("Sum of importance=", feature_importance.importance_percentage.sum())
feature_importance

