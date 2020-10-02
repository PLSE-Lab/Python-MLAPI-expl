#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import seaborn as sns #for Data visualisation
import matplotlib.pyplot as plt #for Data visualisation
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#importing datasets 
train=pd.read_csv('/kaggle/input/Train_v2.csv')
test=pd.read_csv('/kaggle/input/Test_v2.csv')
submission=pd.read_csv('/kaggle/input/SubmissionFile.csv')
variables=pd.read_csv('/kaggle/input/VariableDefinitions.csv')


# In[ ]:


#copying original datasets for future usage in the program
train_copy=train.copy()
test_copy=test.copy()


# In[ ]:


#combining dataset for easy data impuation in all datasets
combine = [train,test]


# In[ ]:


#view first five rows
train.head()


# In[ ]:


#view last five rows
train.tail()


# In[ ]:


#display columns in train dataset
print(train.columns.values)


# In[ ]:


#display columns in test dataset
print(test.columns.values)


# In[ ]:


#display shape for all datasets
train.shape,test.shape,submission.shape,variables.shape


# In[ ]:


#checking data types in the datasets
train.info(),test.info()


# In[ ]:


#view variables with missing values from train data
train.isna().sum()


# In[ ]:


#view variables with missing values from test data
test.isna().sum()


# * Data cleaning 
# * We found no missing values or null values in the train and test datasets

# * Now Doing Exploratory Data Analysis 
# * Univariate Data analysis
# * There are 14% of people have  bank accounts
# * There are 86% of people who deosn't have bank accounts
# 

# In[ ]:


#getting variable data into seperate dataset
dummy1=pd.get_dummies(train['bank_account'])
dummy1.head()


# In[ ]:


#summary of the data
dummy1.describe()


# In[ ]:


#Display total numbers of people qualifying for accounts and not in the train dataset
print("Number of people qualifying for bank accounts",(dummy1['Yes'].sum()))
print("Number of people doesn't qualify for bank accounts",(dummy1['No'].sum()))


# In[ ]:


#bar graph
names = ['Yes', 'No']
values = [3312,20212]

plt.figure(figsize=(30, 5))

plt.subplot(131)
plt.bar(names, values)
plt.title('bank_account')
plt.show()


# Now performing univariate analysis for numerical data for Household size and age respondents
# * The count of train data set is 23524 from popultion dataset
# looking at variable household size 
# 1. We have mean of 1,standard deviation of 2.23 
# 2. We have minimum household size of 1 and maximum household size of 21
# 3. The data is skewed to the left with fair outliers
# looking at variable age respondents
# 1. We have mean of 38,81,standard deviation of 16 
# 2. We have minimum 16 and maximum 100 of age respondents
# 3. The data is skewed to the left 
# 
# 

# In[ ]:


#data summary for household sizes
train['household_size'].describe()


# In[ ]:


#indepent Variable(Numerical)
plt.figure(3) 
plt.subplot(121) 
sns.distplot(train['household_size']);
plt.subplot(122) 
train['household_size'].plot.box(figsize=(16,5))
plt.show()


# In[ ]:


#data summary for age of respondents
train['age_of_respondent'].describe()


# In[ ]:


#indepent Variable(Numerical)
plt.figure(4) 
plt.subplot(131) 
sns.distplot(train['age_of_respondent']);
plt.subplot(132) 
train['age_of_respondent'].plot.box(figsize=(16,5))
plt.show()


# Bivariate analysis

# In[ ]:


cellphone_access=pd.crosstab(train['cellphone_access'],train['bank_account'])
cellphone_access.div(cellphone_access.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))


# In[ ]:


gender_of_respondent=pd.crosstab(train['gender_of_respondent'],train['bank_account'])
gender_of_respondent.div(gender_of_respondent.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))


# In[ ]:


relationship_with_head=pd.crosstab(train['relationship_with_head'],train['bank_account'])
relationship_with_head.div(relationship_with_head.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
marital_status=pd.crosstab(train['marital_status'],train['bank_account'])
marital_status.div(marital_status.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
education_level=pd.crosstab(train['education_level'],train['bank_account'])
education_level.div(education_level.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
job_type=pd.crosstab(train['job_type'],train['bank_account'])
job_type.div(job_type.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))


# In[ ]:


#categorising data from numrical data into ordinal data
for dataset in combine:    
   
    dataset.loc[(dataset['age_of_respondent'] > 0) & (dataset['age_of_respondent'] <= 16), 'age_of_respondent'] = 0
    dataset.loc[(dataset['age_of_respondent'] > 16) & (dataset['age_of_respondent'] <=200), 'age_of_respondent'] = 1
 


# In[ ]:


#categorising data from numrical data into ordinal data
for dataset in combine:    
    dataset.loc[(dataset['household_size'] > 0) & (dataset['household_size'] <= 3), 'household_size'] = 3
    dataset.loc[(dataset['household_size'] > 3) & (dataset['household_size'] <= 6), 'household_size'] = 2
    dataset.loc[(dataset['household_size'] > 6) & (dataset['household_size'] <=9), 'household_size'] = 1
    dataset.loc[(dataset['household_size'] > 9) & (dataset['household_size'] <=21), 'household_size'] = 0


# In[ ]:


#getting gender_of_respondent data into seperate dataset
dummy=pd.get_dummies(train['gender_of_respondent'])
dummy.head()


# In[ ]:


#categorical Variable(binary)
plt.figure(4) 
plt.subplot(131) 
sns.distplot(dummy['Female']);
plt.subplot(132) 
dummy['Female'].plot.box(figsize=(16,5))

plt.show()


# In[ ]:


#categorical Variable(binary)
plt.figure(4) 
plt.subplot(131) 
sns.distplot(dummy['Male']);
plt.subplot(132) 
dummy['Male'].plot.box(figsize=(16,5))

plt.show()


# In[ ]:


#summary of the data
dummy.describe()


# In[ ]:


#Display total numbers of Females and Males in the train dataset
print("Number of Females",(dummy['Female'].sum()))
print("Number of Males",(dummy['Male'].sum()))


# In[ ]:


#bar graph
names = ['Female', 'Male']
values = [13877,9647]

plt.figure(figsize=(30, 5))

plt.subplot(131)
plt.bar(names, values)
plt.title('gender_of_respondent')
plt.show()


# From the data we have more females than males from train dataset
# Around 59% is Females and around 41% is Males

# In[ ]:


#getting variable data into seperate dataset
dummytest1=pd.get_dummies(test['gender_of_respondent'])
dummytest1.head()


# In[ ]:


#getting variable data into seperate dataset
dummy3=pd.get_dummies(train['cellphone_access'])
dummy3.head()


# In[ ]:


#summary data
dummy3.describe()


# In[ ]:


#categorical Variable(binary)
plt.figure(4) 
plt.subplot(131) 
sns.distplot(dummy3['No']);
plt.subplot(132) 
dummy3['No'].plot.box(figsize=(16,5))

plt.show()


# In[ ]:


#categorical Variable(binary)
plt.figure(4) 
plt.subplot(131) 
sns.distplot(dummy3['Yes']);
plt.subplot(132) 
dummy3['Yes'].plot.box(figsize=(16,5))

plt.show()


# In[ ]:


#display number of people having cellphones and not having cellphones
print("Number of people who doesn't cell phones",(dummy3['No'].sum()))
print("Number of people who has cell phones",(dummy3['Yes'].sum()))


# Around 26% does not have cellphones and 74% have cellphones

# In[ ]:


#bar graph
names = ['No','Yes']
values = [6070,17454]

plt.figure(figsize=(20, 5))

plt.subplot(131)
plt.bar(names, values)
plt.title('cellphone_access')
plt.show()


# * Looking at the bar graph more people have cellphones

# In[ ]:


#getting variable data into seperate dataset
dummytest3=pd.get_dummies(test['cellphone_access'])
dummytest3.head()


# In[ ]:


#getting variable data into seperate dataset
dummy4=pd.get_dummies(train['education_level'])
dummy4.head()


# In[ ]:


#summary data
dummy4.describe()


# In[ ]:


#display total numbers 
print("Number of people with No formal education",(dummy4['No formal education'].sum()))
print("Number of people with Other/Dont know/RTA",(dummy4['Other/Dont know/RTA'].sum()))
print("Number of people with Primary education",(dummy4['Primary education'].sum()))
print("Number of people with Secondary education",(dummy4['Secondary education'].sum()))
print("Number of people with Tertiary education",(dummy4['Tertiary education'].sum()))
print("Number of people with Vocational/Specialised training",(dummy4['Vocational/Specialised training'].sum()))


# * Around 19% of people have No formal Education
# * Around 0.1% of people is Other/Don't Know/ RTA
# * Around 54% of people have Primary Education
# * Around 18% of people have secondary education
# * Around Around 4% of people have Tertiary education
# * Around 3% of people have Vacational/specialised training

# In[ ]:


#bar graph
names = ['No formal education','Other/Dont know/RTA','Primary education', 'Secondary education','Tertiary education','Vocational/Specialised training']
values = [4515,35,12791,4223,1157,803]

plt.figure(figsize=(55, 5))

plt.subplot(131)
plt.bar(names, values)
plt.title('Education level')
plt.show()


# Looking at Education level, more people have primary education.

# In[ ]:


#getting variable data into seperate dataset
dummytest4=pd.get_dummies(test['education_level'])
dummytest4.head()


# In[ ]:


#getting variable data into seperate dataset
dummy5=pd.get_dummies(train['job_type'])
dummy5.head()


# In[ ]:


#getting variable data into seperate dataset
dummytest5=pd.get_dummies(test['job_type'])
dummytest5.head()


# In[ ]:


#sammary data
dummy5.describe()


# In[ ]:


#display number totals
print("Number of people Dont Know/Refuse to answer",(dummy5['Dont Know/Refuse to answer'].sum()))
print("Number of people with Farming and Fishing",(dummy5['Farming and Fishing'].sum()))
print("Number of people with Formally employed Government",(dummy5['Formally employed Government'].sum()))
print("Number of people with Formally employed Private",(dummy5['Formally employed Private'].sum()))
print("Number of people with Government Dependent",(dummy5['Government Dependent'].sum()))
print("Number of people with Informally employed",(dummy5['Informally employed'].sum()))
print("Number of people with No Income",(dummy5['No Income'].sum()))
print("Number of people with Other Income",(dummy5['Other Income'].sum()))
print("Number of people with Remittance Dependent",(dummy5['Remittance Dependent'].sum()))
print("Number of people with Self employed",(dummy5['Self employed'].sum()))


# * Arournd 0.5% of people don't know/refuse to answer
# * Around 23% of people are in farming and fishing
# * Around 1.6% of people are formally employed Government
# * Around 4.5% of people are formally employed private
# * Around 1.% of people are Government dependents
# * Around 24.% Are informally employed
# * Around 2.7% are with No Income
# * Around 4.6% are with other income
# * Around 10.7% are Remittance dependents
# * Around 27% are self employed 

# In[ ]:


#bar graph
names = ['Dont Know/Refuse','Farming and Fishing','Formally employed Government', 'Formally employed Private','Government Dependent','Informally employed','No Income','Other Income','Remittance Dependent','Self employed']
values = [126,5441,387,1055,247,5597,627,1080,2527,6437]

plt.figure(figsize=(100,10))

plt.subplot(131)
plt.bar(names, values)
plt.title('job_type')
plt.show()


# * Looking at the bar graph the number of people who are self employed greater 

# In[ ]:


#getting variable data into seperate dataset
dummy55=pd.get_dummies(train['marital_status'])
dummy55.head()


# In[ ]:


#getting variable data into seperate dataset
dummytest55=pd.get_dummies(test['marital_status'])
dummytest55.head()


# In[ ]:


#sammary data
dummy55.describe()


# In[ ]:


#display totals
print("Number of people Divorced/Seperated",(dummy55['Divorced/Seperated'].sum()))
print("Number of people Dont know",(dummy55['Dont know'].sum()))
print("Number of people Married/Living together",(dummy55['Married/Living together'].sum()))
print("Number of people Single/Never Married",(dummy55['Single/Never Married'].sum()))
print("Number of people Widowed",(dummy55['Widowed'].sum()))


# In[ ]:


#bar graph
names = ['Dont know', 'Divorced/Seperated', 'Widowed','Single/Never Married','Married/Living together']
values = [2076,8,10749,7983,2708]

plt.figure(figsize=(37, 7))

plt.subplot(131)
plt.bar(names, values)
plt.title('marital_status')
plt.show()


# In[ ]:


#getting variable data into seperate dataset
dummy6=pd.get_dummies(train['relationship_with_head'])
dummy6.head()


# In[ ]:


#sammary data
dummy6.describe()


# In[ ]:


#display numbers totals
print("Number of children",(dummy6['Child'].sum()))
print("Number of Head of Households",(dummy6['Head of Household'].sum()))
print("Number of Other non-relatives",(dummy6['Other non-relatives'].sum()))
print("Number of Other relatives",(dummy6['Other relative'].sum()))
print("Number of Parent",(dummy6['Parent'].sum()))
print("Number of Spouse",(dummy6['Spouse'].sum()))


# In[ ]:


#bar graph
names = ['Child', 'Head of Household', 'Other non-relatives','Other relative','Parent','Spouse']
values = [2229,12831,190,668,1086,6520]

plt.figure(figsize=(37, 7))

plt.subplot(131)
plt.bar(names, values)
plt.title('relationship_with_head')
plt.show()


# In[ ]:


#getting variable data into seperate dataset
dummytest6=pd.get_dummies(test['relationship_with_head'])
dummytest6.head()


# In[ ]:


#getting variable data into seperate dataset
dummy7=pd.get_dummies(train['location_type'])
dummy7.head()


# In[ ]:


#sammary data
dummy7.describe()


# In[ ]:


#display numbers totals
print("Number of people in Rural",(dummy7['Rural'].sum()))
print("Number of of people in Urban",(dummy7['Urban'].sum()))


# In[ ]:


#bar graph
names = ['Rural','Urban']
values = [14343,9181]

plt.figure(figsize=(20, 5))

plt.subplot(131)
plt.bar(names, values)
plt.title('location_type')
plt.show()


# In[ ]:


#getting variable data into seperate dataset
dummytest7=pd.get_dummies(test['location_type'])
dummytest7.head()


# In[ ]:


#concatinating datasets into one dataset
train2=pd.concat((train,dummy,dummy3,dummy4,dummy5,dummy55,dummy6,dummy7),axis=1)


# In[ ]:


#concatinating datasets into one dataset
test2=pd.concat((test,dummytest1,dummytest3,dummytest4,dummytest5,dummytest55,dummytest6,dummytest7),axis=1)


# In[ ]:


#view first five rows of data transformed
train2.head().T


# In[ ]:


#view first five rows of data transformed
test2.head().T


# In[ ]:



#rename columns of interest 
train2=train2.rename(columns={"Female":"gender_of_respondent_F","Male":"gender_of_respondent_M","Rural":"location_type_R","Urban":"location_type_U","Yes":"cellphone_access_Y","No":"cellphone_access_N"})
train2.head()


# In[ ]:


#rename columns of interest
test2=test2.rename(columns={"Female":"gender_of_respondent_F","Male":"gender_of_respondent_M","Rural":"location_type_R","Urban":"location_type_U","Yes":"cellphone_access_Y","No":"cellphone_access_N"})
test2.head()


# In[ ]:


#view first five rows transformed
train2.head().T


# In[ ]:


#view first five rows transformed
test2.head().T


# In[ ]:


#display columns in the dataset
train2.columns


# In[ ]:


#drop unnecessary columns
train2=train2.drop(['location_type','cellphone_access','gender_of_respondent', 'relationship_with_head', 'marital_status',
       'education_level','job_type'], axis = 1) 


# In[ ]:


#drop unnecessary columns
test2=test2.drop(['location_type','cellphone_access','gender_of_respondent', 'relationship_with_head', 'marital_status',
       'education_level','job_type'], axis = 1) 


# In[ ]:


#view first five rows
train2.head()


# In[ ]:


#view last five rows
train2.tail()


# In[ ]:


#view first five rows
test2.head()


# In[ ]:


#getting variable data into seperate dataset
dummy1=pd.get_dummies(train2['bank_account'])
dummy1.head()


# In[ ]:


#concatinating datasets
train2=pd.concat((train2,dummy1),axis=1)
train2.head()


# In[ ]:


#drop unnecessary columns
train2=train2.drop(['bank_account','No'], axis = 1)


# In[ ]:


#rename column of interest
train2=train2.rename(columns={"Yes":"bank_account"})
train2.head().T


# In[ ]:


import seaborn as sns;
plt.subplots(figsize=(20,15))
sns.heatmap(train2.corr(), annot = True, linewidths = 6, fmt= '.2g')
plt.show()


# In[ ]:


#feature engineering
train2['Education level']=train2['Secondary education']+train2['Tertiary education']+train2['Vocational/Specialised training']
test2['Education level']=test2['Secondary education']+test2['Tertiary education']+test2['Vocational/Specialised training']


# In[ ]:


#feature engineering
train2['job_type']=train2['Farming and Fishing']+train2['Formally employed Government']+train2['Formally employed Private']+train2['Government Dependent']+train2['Other Income']+train2['Self employed']
test2['job_type']=test2['Farming and Fishing']+test2['Formally employed Government']+test2['Formally employed Private']+train2['Government Dependent']+test2['Other Income']+test2['Self employed']


# In[ ]:


#feature engineering
train2['marital status']=train2['Divorced/Seperated']+train2['Married/Living together']+train2['Single/Never Married']+train2['Widowed']
test2['marital status']=test2['Divorced/Seperated']+test2['Married/Living together']+test2['Single/Never Married']+test2['Widowed']


# In[ ]:


#feature engineering
train2['relationship head']=train2['Head of Household']+train2['Other relative']+train2['Parent']+train2['Spouse']
test2['relationship head']=test2['Head of Household']+test2['Other relative']+test2['Parent']+test2['Spouse']


# In[ ]:



#feature engineering
train2['gender']=train2['gender_of_respondent_F']+train2['gender_of_respondent_M']
test2['gender']=test2['gender_of_respondent_F']+test2['gender_of_respondent_M']


# In[ ]:



#feature engineering
train2['location_type']=train2['location_type_R']+train2['location_type_U']
test2['location_type']=test2['location_type_R']+test2['location_type_U']


# In[ ]:


#feature engineering
#train2['cellphone']=train2['cellphone_access_N']+train2['cellphone_access_Y']
#test2['cellphone']=test2['cellphone_access_N']+test2['cellphone_access_Y']


# In[ ]:


train2.head().T


# In[ ]:


# machine learning packages to be used
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import Pool, CatBoostClassifier, cv


# In[ ]:


#drop unnecessary columns to be used in the models
train2=train2.drop("uniqueid", axis=1)
test2=test2.drop("uniqueid", axis=1)


# In[ ]:


#drop unnecessary columns to be used in the models
train2=train2.drop("country", axis=1)
test2=test2.drop("country", axis=1)


# In[ ]:


#drop unnecessary columns to be used in models
train2=train2.drop("year", axis=1)
test2=test2.drop("year", axis=1)


# In[ ]:


#drop unnecessary columns to be used in models
train2=train2.drop(['Other/Dont know/RTA','No formal education','Primary education','Secondary education','Tertiary education','Vocational/Specialised training'], axis=1)
test2=test2.drop(['Other/Dont know/RTA','No formal education','Primary education','Secondary education','Tertiary education','Vocational/Specialised training'], axis=1)


# In[ ]:


train2=train2.drop(['Dont Know/Refuse to answer','Farming and Fishing','Formally employed Government', 'Formally employed Private','Government Dependent','Informally employed','No Income','Other Income','Remittance Dependent','Self employed'], axis=1)
test2=test2.drop(['Dont Know/Refuse to answer','Farming and Fishing','Formally employed Government', 'Formally employed Private','Government Dependent','Informally employed','No Income','Other Income','Remittance Dependent','Self employed'], axis=1)


# In[ ]:


train2=train2.drop(['Child','Head of Household', 'Other non-relatives','Other relative','Parent','Spouse'], axis=1)
test2=test2.drop(['Child', 'Head of Household', 'Other non-relatives','Other relative','Parent','Spouse'], axis=1)


# In[ ]:


train2=train2.drop(['Dont know', 'Divorced/Seperated', 'Widowed','Single/Never Married','Married/Living together'], axis=1)
test2=test2.drop(['Dont know', 'Divorced/Seperated', 'Widowed','Single/Never Married','Married/Living together'], axis=1)


# In[ ]:



train2=train2.drop(['gender_of_respondent_F','gender_of_respondent_M'], axis=1)
test2=test2.drop(['gender_of_respondent_F','gender_of_respondent_M'], axis=1)


# In[ ]:



train2=train2.drop(['cellphone_access_N'], axis=1)
test2=test2.drop(['cellphone_access_N'], axis=1)


# In[ ]:


train2=train2.drop(['location_type_R','location_type_U'], axis=1)
test2=test2.drop(['location_type_R','location_type_U'], axis=1)


# In[ ]:


#display shape for all datasets
train2.shape,test2.shape


# In[ ]:


test2.head()


# In[ ]:


test2.head().T


# In[ ]:


#calculate splitting data percentages
#finding total count of datasets
print("Total population",23524+10086)
#finding test dataset percentage
print("test dataset",10086/33610*100,"%")
#finding train dataset percentage
print("train dataset",23524/33610*100,"%")


# In[ ]:


#Model, predict and solve
X_train = train2.drop("bank_account", axis=1)
Y_train = train2.bank_account

X_test  = test2

X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


LogisticRegression()


# In[ ]:


# Logistic Regression

logreg = LogisticRegression(n_jobs=100,random_state=30)


# In[ ]:


logreg.fit(X_train, Y_train)
Y_pred1 = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


Y_pred1.sum()


# In[ ]:


coeff_df = pd.DataFrame(train2.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# In[ ]:


SVC()


# In[ ]:


# Support Vector Machines

svc = SVC(random_state=30)
svc.fit(X_train, Y_train)
Y_pred2 = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:


Y_pred2.sum()


# In[ ]:


KNeighborsClassifier()


# In[ ]:


knn = KNeighborsClassifier(n_jobs=30,n_neighbors = 5)
knn.fit(X_train, Y_train)
Y_pred3 = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[ ]:


Y_pred3.sum()


# In[ ]:


GaussianNB()


# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred4 = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[ ]:


Y_pred4.sum()


# In[ ]:


Perceptron()


# In[ ]:


# Perceptron

perceptron = Perceptron(n_jobs=30,random_state=100)
perceptron.fit(X_train, Y_train)
Y_pred5 = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[ ]:


Y_pred5.sum()


# In[ ]:


LinearSVC()


# In[ ]:


# Linear SVC

linear_svc = LinearSVC(random_state=100)
linear_svc.fit(X_train, Y_train)
Y_pred6 = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# In[ ]:


Y_pred6.sum()


# In[ ]:


SGDClassifier()


# In[ ]:


# Stochastic Gradient Descent

sgd = SGDClassifier(n_jobs=30,random_state=100)
sgd.fit(X_train, Y_train)
Y_pred7 = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[ ]:


Y_pred7.sum()


# In[ ]:


DecisionTreeClassifier()


# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier(random_state=100)
decision_tree.fit(X_train, Y_train)
Y_pred8 = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


Y_pred8.sum()


# In[ ]:


XGBClassifier()


# In[ ]:


# XGBoost

model_xgboost=XGBClassifier(n_estimators=100,n_jobs=30, random_state=100,max_depth=4)
model_xgboost.fit(X_train, Y_train)
Y_pred9 = model_xgboost.predict(X_test)
model_xgboost.score(X_train, Y_train)
acc_xgboost = round(model_xgboost.score(X_train, Y_train) * 100, 2)
acc_xgboost


# In[ ]:


Y_pred9.sum()


# In[ ]:


RandomForestClassifier()


# In[ ]:


#Random forest
random_forest = RandomForestClassifier(n_estimators=100,random_state=1)
random_forest.fit(X_train, Y_train)
Y_pred10 = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


Y_pred10.sum()


# In[ ]:


CatBoostClassifier()


# In[ ]:


#Cat Boost
catboost= CatBoostClassifier(random_seed=100)


# In[ ]:


catboost.fit(X_train, Y_train)
Y_pred11 = catboost.predict(X_test)
catboost.score(X_train, Y_train)
acc_catboost= round(catboost.score(X_train, Y_train) * 100, 2)
acc_catboost


# In[ ]:


Y_pred11.sum()


# In[ ]:


#choose the features we want to train, just forget the float data
cate_features_index = np.where(X_train.dtypes != float)[0]


# In[ ]:


#display models accuracy scores in descending order
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree','XGBoost','Catboost'],
    'Accuracy score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree,acc_xgboost,acc_catboost]})
models.sort_values(by='Accuracy score', ascending=False)


# improving the accuracy score 

# In[ ]:


#dividing train-test dataset to have validation part
from sklearn.model_selection import train_test_split
x_train,x_cv,y_train,y_cv=train_test_split(X_train,Y_train,test_size=0.3)


# In[ ]:


#Applying dummy variable
X_train=pd.get_dummies(X_train)
train2=pd.get_dummies(train2)
test2=pd.get_dummies(test2)


# In[ ]:


#import Accuracy_score from sklearn 

from sklearn.metrics import accuracy_score


# In[ ]:


from sklearn.model_selection import StratifiedKFold


# In[ ]:


#Random Forest
i=1 
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X_train ,Y_train):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl=X_train .loc[train_index],X_train .loc[test_index]
    ytr,yvl=Y_train[train_index],Y_train[test_index] 
    model= RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=500,
                       n_jobs=200, oob_score=False, random_state=500,
                       verbose=0, warm_start=False)
    model.fit(xtr,ytr)
    pred_test=model.predict(xvl)
    score=accuracy_score(yvl,pred_test)
    print('accuracy_score',score) 
    i+=1
pred_test=model.predict(test2)
print(pred_test)
pred=model.predict_proba(xvl)[:,1]
print(pred)


# In[ ]:


i=1 
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X_train ,Y_train):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl=X_train .loc[train_index],X_train .loc[test_index]
    ytr,yvl=Y_train[train_index],Y_train[test_index] 
    model1= CatBoostClassifier(random_seed=100)
    model1.fit(xtr,ytr)
    pred_test1=model1.predict(xvl)
    score1=accuracy_score(yvl,pred_test1)
    print('accuracy_score',score1) 
    i+=1
pred_test1=model1.predict(test2)
print(pred_test1)
pred1=model1.predict_proba(xvl)[:,1]
print(pred1)


# In[ ]:


i=1 
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X_train ,Y_train):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl=X_train .loc[train_index],X_train .loc[test_index]
    ytr,yvl=Y_train[train_index],Y_train[test_index] 
    model2= DecisionTreeClassifier()
    model2.fit(xtr,ytr)
    pred_test2=model2.predict(xvl)
    score2=accuracy_score(yvl,pred_test2)
    print('accuracy_score',score2) 
    i+=1
pred_test2=model2.predict(test2)
print(pred_test2)
pred2=model2.predict_proba(xvl)[:,1]
print(pred2)


# In[ ]:


i=1 
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X_train ,Y_train):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl=X_train .loc[train_index],X_train .loc[test_index]
    ytr,yvl=Y_train[train_index],Y_train[test_index] 
    model3= DecisionTreeClassifier()
    model3.fit(xtr,ytr)
    pred_test3=model3.predict(xvl)
    score3=accuracy_score(yvl,pred_test3)
    print('accuracy_score',score3) 
    i+=1
pred_test3=model3.predict(test2)
print(pred_test3)
pred3=model3.predict_proba(xvl)[:,1]
print(pred3)


# In[ ]:


i=1 
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X_train ,Y_train):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl=X_train .loc[train_index],X_train .loc[test_index]
    ytr,yvl=Y_train[train_index],Y_train[test_index] 
    model4=XGBClassifier(n_estimators=100,n_jobs=30, random_state=100,max_depth=4)
    model4.fit(xtr,ytr)
    pred_test4=model4.predict(xvl)
    score4=accuracy_score(yvl,pred_test4)
    print('accuracy_score',score4) 
    i+=1
pred_test4=model4.predict(test2)
print(pred_test4)
pred4=model4.predict_proba(xvl)[:,1]
print(pred4)


# In[ ]:



i=1 
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X_train ,Y_train):
   print('\n{} of kfold {}'.format(i,kf.n_splits))
   xtr,xvl=X_train .loc[train_index],X_train .loc[test_index]
   ytr,yvl=Y_train[train_index],Y_train[test_index] 
   model5=LinearSVC()
   model5.fit(xtr,ytr)
   pred_test5=model5.predict(xvl)
   score5=accuracy_score(yvl,pred_test5)
   print('accuracy_score',score5) 
   i+=1
pred_test5=model5.predict(test2)
print(pred_test5)


# In[ ]:


i=1 
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X_train ,Y_train):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl=X_train .loc[train_index],X_train .loc[test_index]
    ytr,yvl=Y_train[train_index],Y_train[test_index] 
    model6= LogisticRegression(max_iter=50,n_jobs=100,random_state=50)
    model6.fit(xtr,ytr)
    pred_test6=model6.predict(xvl)
    score6=accuracy_score(yvl,pred_test6)
    print('accuracy_score',score6) 
    i+=1
pred_test6=model6.predict(test2)
print(pred_test6)
pred6=model6.predict_proba(xvl)[:,1]
print(pred6)


# In[ ]:


# Gaussian Naive Bayes
GaussianNB()


# In[ ]:


i=1 
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X_train ,Y_train):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl=X_train .loc[train_index],X_train .loc[test_index]
    ytr,yvl=Y_train[train_index],Y_train[test_index] 
    model7= GaussianNB()
    model7.fit(xtr,ytr)
    pred_test7=model7.predict(xvl)
    score7=accuracy_score(yvl,pred_test7)
    print('accuracy_score',score7) 
    i+=1
pred_test7=model7.predict(test2)
print(pred_test7)
pred7=model7.predict_proba(xvl)[:,1]
print(pred7)


# In[ ]:


# Support Vector Machines

svc = SVC(random_state=30)
svc.fit(X_train, Y_train)
Y_pred2 = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:


LogisticRegression(n_jobs=100,random_state=30)


# In[ ]:


i=1 
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X_train ,Y_train):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl=X_train .loc[train_index],X_train .loc[test_index]
    ytr,yvl=Y_train[train_index],Y_train[test_index] 
    model9=LogisticRegression(n_jobs=100,random_state=30)
    model9.fit(xtr,ytr)
    pred_test9=model9.predict(xvl)
    score9=accuracy_score(yvl,pred_test9)
    print('accuracy_score',score9) 
    i+=1
pred_test9=model9.predict(test2)
print(pred_test9)


# In[ ]:


i=1 
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X_train ,Y_train):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl=X_train .loc[train_index],X_train .loc[test_index]
    ytr,yvl=Y_train[train_index],Y_train[test_index] 
    model8=SVC(random_state=30)
    model8.fit(xtr,ytr)
    pred_test8=model8.predict(xvl)
    score8=accuracy_score(yvl,pred_test8)
    print('accuracy_score',score8) 
    i+=1
pred_test8=model8.predict(test2)
print(pred_test8)


# In[ ]:


output = pd.DataFrame({'uniqueid':submission['uniqueid'],
                       'bank_account':pred_test1})
output.to_csv('SubmissionFile.csv', index=False)
print("successfully saved!")


# In[ ]:


print(pred_test.sum())
print(pred_test1.sum())
print(pred_test2.sum())
print(pred_test3.sum())
print(pred_test4.sum())
print(pred_test5.sum())
print(pred_test7.sum())
print(pred_test8.sum())


# In[ ]:


10086-748


# In[ ]:


9338/10086*100


# In[ ]:


#bar graph
names = ['Yes', 'No']
values = [748,9338]

plt.figure(figsize=(10,6))

plt.subplot(131)
plt.bar(names, values)
plt.title('bank_account')
plt.show()


# ***
