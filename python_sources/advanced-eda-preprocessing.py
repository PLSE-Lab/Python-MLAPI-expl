#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visualization
import seaborn as sns #data visualization
from scipy import stats #Statistics
from sklearn.cluster import DBSCAN  #outlier detection
from collections import Counter
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from fancyimpute import KNN #KNN imputation

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Importing CSV files.**

# In[ ]:


test_df = pd.read_csv("../input/test_AV3.csv")
train_df = pd.read_csv("../input/train_AV3.csv")


# Looking at basic structure of data frames: 

# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.shape


# In[ ]:


train_df['Property_Area'].hist(color = 'orange')
plt.xlabel('Property Area')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


train_df.info()


# In[ ]:


train_df.dtypes


# Observations till now:
# 1. There are various NULL objects in many attributes. Surely requires attention.
# 2. There are 13 attributes of train_df and 12 of test_df.
# 3. Categorical Attributes : Gender, married, Education, self employed, Propert_area, Loan_status.
# 3. Numercal Categorical : Dependencies, Credit History.
# 4. Continuous Numerical : ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
# 5. Most houses are in suburban area. 
# 6. Data type of few attributes need to be changed.

# In[ ]:


train_df.describe()


# Observation : Mean and median of almost all the attributes are far apart.
# Around 84% people have credit history.
# 

# Mode of all features:

# In[ ]:


train_df.mode().iloc[0][1:]   


# ### Correcting data(Imputation of missing values)

# In[ ]:


#Number of null values
train_df.isnull().sum()


# In[ ]:


#train_df.LoanAmount.value_counts(dropna=False)
fig,axes = plt.subplots(nrows=1, ncols=2)
mode = train_df.LoanAmount.mode().iloc[0]
fig1 = train_df.LoanAmount.fillna(mode).plot(kind = 'hist', bins = 30, ax = axes[1])
print(train_df.LoanAmount.fillna(mode).describe())
print(train_df.LoanAmount.describe())
train_df.LoanAmount.plot(kind = 'hist', bins = 30, ax = axes[0])
plt.show()


# In[ ]:


#Using mean to fill NAN values
fig,axes = plt.subplots(nrows=1, ncols=2)
mean = train_df.LoanAmount.mean()
train_df.LoanAmount.fillna(mean).plot(kind = 'hist', bins = 30, ax = axes[1])
print(train_df.LoanAmount.fillna(mean).describe())
print(train_df.LoanAmount.describe())
train_df.LoanAmount.plot(kind = 'hist', bins = 30, ax = axes[0])


# In[ ]:


#Using median to fill out NAN values
#Using mean to fill NAN values
fig,axes = plt.subplots(nrows=1, ncols=2)
median = train_df.LoanAmount.median()
train_df.LoanAmount.fillna(median).plot(kind = 'hist', bins = 30, ax = axes[1])
print(train_df.LoanAmount.fillna(median).describe())
print(train_df.LoanAmount.describe())
train_df.LoanAmount.plot(kind = 'hist', bins = 30, ax = axes[0])


# Since median < mean, we can say that our disribution is skewed, so median can be considered as better way of dealing with center. Skewness is easily seen from the histograms.

# In[ ]:


#Using KNN to fill the values
train_df.ApplicantIncome = train_df.ApplicantIncome.astype('float')
train_df_numeric = train_df.select_dtypes('float')
df_filled = KNN(k=5).complete(train_df_numeric)
df_filled = pd.DataFrame(df_filled)
df_filled.shape


# In[ ]:


df_filled.index = train_df_numeric.index
df_filled.columns = train_df_numeric.columns


# In[ ]:


df_filled.info()


# In[ ]:


#Make a copy of train_df to fill up the values from df_filled
train_df_c = train_df.copy()
train_df_c.LoanAmount = df_filled.LoanAmount


# In[ ]:


fig, axes = plt.subplots(nrows=1,ncols=2)
train_df.LoanAmount.hist(bins = 30, ax = axes[0])
train_df_c.LoanAmount.hist(bins = 30, ax = axes[1])
print(train_df.LoanAmount.describe())
print(train_df_c.LoanAmount.describe())


# **Observation**: kNN is perfect method as magnitude of difference between std of original and imputed loan amount is approximately 0.4, also in this case mean is almost same.
# **Observation**: Dataset is small and applying KNN is good to impute other values also.
# Let us check for other continuous numerical attributes also: 

# In[ ]:


train_df_c.ApplicantIncome = df_filled.ApplicantIncome
train_df_c.CoapplicantIncome = df_filled.CoapplicantIncome
train_df_c.Loan_Amount_Term = df_filled.Loan_Amount_Term


# In[ ]:


plt.hist(train_df_c.LoanAmount, bins = 30, alpha = 0.5, label = 'Imputed Loan Amount', color = 'orange', stacked = True)
plt.hist(train_df.LoanAmount.dropna(), bins = 30, alpha = 0.5, label = 'Original Loan Amount', color = 'green')
plt.legend()
plt.show()


# In[ ]:


plt.hist(train_df_c.Loan_Amount_Term, bins = 10, alpha = 0.5, label = 'Imputed Loan Amount Term', color = 'orange', stacked = True)
plt.hist(train_df.Loan_Amount_Term.dropna(), bins = 10, alpha = 0.5, label = 'Original Loan Amount Term', color = 'green')
plt.legend()
plt.show()


# As we can see that there is not much difference in those two graphs we can go ahead with KNN imputation for continuous numerical attributes. Now we'll find missing values for categorical data.

# In[ ]:


train_df.describe(include = ['O'])


# Observations from Categorical Data: (No NAN included) 
# * 480 are graduate.
# * Most houses are from semiurban area(233)
# * Most Loans were approved.
# * Most of the aplicants are Married and does not have dependents.
# * There are null values in Credit History, self employed, Dependents and gender and married. Let us try fill these NA values.
# 

# In[ ]:


train_df = train_df_c.copy()


# In[ ]:


#Transforming Loan_Status, Gender, Married, Education, & Property area from object to proper data type like bool, int, categoryetc.
d = {'Y':True, 'N':False}
train_df.Loan_Status = train_df.Loan_Status.map(d)
d = {'Male':False, 'Female':True, 'NaN':np.nan}
train_df.Gender = train_df.Gender.map(d)


# In[ ]:


#We want to find if there is relationship between gender & Applicants income so that we can find missing gender more clearly.
train_df_c = train_df[train_df.Gender.notnull()].copy()


# In[ ]:


train_df_c.Gender = train_df_c.Gender.astype('int64')
fig, ax = plt.subplots(figsize = (18,18))
sns.heatmap(train_df_c.corr(), ax = ax, annot = True)


# We can simply see that there is almost no relationship between Gender and any other attributes (Beautiful, isn't it?).So we may go ahead and put all Nan in gender to be Female. Similarly we can fill all other nan values of all variables.

# In[ ]:


train_df.Gender = train_df.Gender.fillna(True)


# Credit History has around 50 Null values let's try to fill it first.

# In[ ]:


#Filling Nan using mode
train_df.Credit_History = train_df.Credit_History.fillna(train_df.Credit_History.mode()[0])


# Now Self_Employed has 32 missing values. We'll again impute it, after seeing it's dependencies.

# In[ ]:


d = {'No':False, 'Yes':True, 'Nan':np.nan}
train_df.Self_Employed = train_df.Self_Employed.map(d)
train_df_c = train_df[train_df.Self_Employed.notnull()]
train_df_c.Self_Employed = train_df_c.Self_Employed.astype('bool')


# In[ ]:


fig, ax = plt.subplots(figsize = (10,10))
sns.heatmap(train_df_c.corr(), annot = True, ax = ax)


# Again we can see that Self_employed is not very much correlated to any other attributes. So let us fill nan using mode again, also There are total of 15 Nan in dependants and 3 in marrige in these cases again we'll use mode as missing values are comparitively less.

# In[ ]:


train_df.Self_Employed = train_df.Self_Employed.fillna(train_df.Self_Employed.mode()[0])
train_df.Dependents = train_df.Dependents.fillna(train_df.Dependents.mode()[0])
train_df.Married = train_df.Married.fillna(train_df.Married.mode()[0])


# In[ ]:


train_df.isnull().sum()


# In[ ]:


#Now we hav no null values in our data, also in this cell we delete our useless data frames.
del(train_df_c)
del(train_df_numeric)


# Exploratory analysis

# In[ ]:


train_df.head()


# In[ ]:


#Visualizing scatter plot of all continuous numerical data, 
#Selecting all numerical attributes with Loan Status
train_df_c = train_df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term','Loan_Status']]
color = ['orange' if i==True else 'green' for i in train_df_c.loc[:,'Loan_Status']]
pd.plotting.scatter_matrix(train_df_c.loc[:, train_df_c.columns != 'Loan_Status'], c = color, figsize=(15,15), diagonal='hist', alpha = 0.5,
                          s= 200)
plt.show()


# **Some Observations from above matrix:**
# * Mostly there are just three Loan amount term, in which 360 is given to most applicants.
# * From histograms we can see that other than Loan_Amount_term all other distributions are left skewed.
# * Most applicant's are applying for loan amount of less than 300.

# In[ ]:


fig = sns.FacetGrid(train_df, row = 'Gender', col='Education', size = 4)
fig = fig.map(plt.hist, 'LoanAmount', color = 'g', bins = 30)


# Male Candidates are applying more, moreover those who are graduate are applying for higher amount of loan. Female those who are not graduate are applying for very less amount.

# In[ ]:


fig = sns.FacetGrid(train_df, row = 'Gender', col='Education', size = 4)
fig = fig.map(plt.hist, 'ApplicantIncome', color = 'c', bins = 30)


# Graduate males are earning more than graduate female. 

# In[ ]:


fig = sns.FacetGrid(train_df, col='Credit_History', row = 'Loan_Status', size = 4)
fig = fig.map(plt.hist, 'LoanAmount', color = 'k', bins = 30)


# * Most of the people are taking loan again.
# * Loan approval rate is high for applicants with credit history.

# In[ ]:


train_df.boxplot(by = 'Loan_Status', column = 'LoanAmount', figsize = (7,5))
plt.show()


# In[ ]:


train_df.boxplot(by = 'Loan_Status', column = 'ApplicantIncome', figsize = (7,5))
plt.show()


# * Applicants with higher income have slightly higher chance of getting their loan approved.

# In[ ]:


train_df.boxplot(by = 'Loan_Status', column = 'CoapplicantIncome', figsize = (7,5))
plt.show()


# There are a lot of Outliers which will be handled afterwards.

# In[ ]:


train_df.boxplot(by = 'Education', column = 'ApplicantIncome', figsize = (7,5))
plt.show()


# * Few applicants have very high income. (Imbalance in society)
# * Graduate have more median income than not graduate.

# ** Creeatingg New Variables ** 

# In[ ]:


# Applicant inclome + Coapplicant income = Family income
train_df['Family_Income'] = train_df.ApplicantIncome + train_df.CoapplicantIncome


# In[ ]:


train_df.Family_Income.plot(kind = 'hist', bins = 50)
plt.show()


# In[ ]:





# In[ ]:


#Applicant's class 
# 0 - 1000 Lower class
# 1001 - 5000 Lower Middle class
# 5001 - 10000 Upper middle Class
# 10000+ Upper class
values = []
for i in train_df.loc[:,'ApplicantIncome']:
    if i <= 1000:
        values.append('Lower Class')
    elif i > 1000 and i <= 5000:
        values.append('Lower Middle Class')
    elif i > 5000 and i <= 10000:
        values.append('Upper Middle Class')
    else:
        values.append('Upper Class')


# In[ ]:


values = np.array(values)
train_df['ApplicantClass'] = values


# In[ ]:


fig, ax = plt.subplots(figsize = (10,10))
sns.countplot(x = 'ApplicantClass',data =  train_df, ax = ax)


# * Most of the people are from Middle Class (Lower + Upper). Lower Class don't apply for loan, Upper don't need it.
# * There is some intuition that classes may have some relation with loan amount. (Let's try to find out)

# In[ ]:


fig = sns.FacetGrid(train_df, col = 'ApplicantClass')
fig.map(plt.hist, 'LoanAmount', bins = 50)
plt.show()


# * Upper Class people are asking for more money, Lower Middle class is asking for maximum of approx 300. (There is atleast a lower class applicant who asked for approx 375)

# In[ ]:


train_df.head()


# In[ ]:


train_df[['Family_Income','LoanAmount']].corr()


# * So we can see tht there is some corelation between Loan Amount and Family Income

# In[ ]:


train_df.boxplot(column='Family_Income', by = 'Loan_Status')


# There are a lot of outliers which needs to be handled.
# 

# In[ ]:


train_df.Credit_History = train_df.Credit_History.astype('bool')


# In[ ]:


Cols = [i for i in train_df.columns if train_df[i].dtype == 'float64']
Cols


# We saw in box plots that there were several outliers, in the continuous numerical data. These many outliers can't be simply discarded as they might contribute well in our anyalysis so finding iutliers using IQR method will simply find so many outliers, so we can use DBSCAN to find outliers also, Applicant income and Family Income etc are huge numbers we can simply apply log to normalize them. (So that we can keep significantly low values of eps)

# In[ ]:


train_df_c = np.log(train_df[['ApplicantIncome', 'Family_Income','LoanAmount']]).copy()


# In[ ]:


model = DBSCAN(eps = 0.7, min_samples= 25).fit(train_df_c)


# In[ ]:


print(train_df_c[model.labels_ == -1])
print(Counter(model.labels_))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




