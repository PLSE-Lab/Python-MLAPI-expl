#!/usr/bin/env python
# coding: utf-8

#  # Exploratory analysis on credit risk analysis datasets

# In the below notebook you will learn how to do exploratory analysis for any credit risk analysis problem. The dataset that we have taken is from the famous competition problem of Home Credit.
# 
# **About Home Credit**
# Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. They wish to find out which customer is likely to default so that they can accordingly decide the customers they wish to lend to

# - <a href='#1'>1. Importing necessary libraries and packages and reading files</a>  
# - <a href='#2'>2. Handling non-numerical variables</a>
# - <a href='#3'>3. Aligning Training and Testing Data</a>
# - <a href='#4'> 4. Handling missing values (using Iterative Imputer)</a>
# - <a href='#5'>5. Outlier Detection (using Isolation Forest)</a>
#   - <a href='#5-1'>5.1 Anomaly detection</a>
# - <a href='#6'>6. Missing data in application_train</a>
# - <a href='#7'>7. Duplicate data in application_train</a>
# - <a href='#8'>8. Checking for data imbalance</a>
# - <a href='#9'>9. Exploratory Data Analysis for application_train by visualisation</a>
#    - <a href='#9.1'>9.1. Distribution of income</a>
#    - <a href='#9.2'>9.2. Distribution of credit</a>
#    - <a href='#9.3'>9.3. Distribution of loan types</a>
#    - <a href='#9.4'>9.4. Distribution of NAME_INCOME_TYPE</a>
#    - <a href='#9.5'>9.5. Distribution of NAME_TYPE_SUITE</a>
#    - <a href='#9.6'>9.6. Distribution of NAME_EDUCATION_TYPE</a>
#    - <a href='#9.7'>9.7. Effect of marital status on ability to pay back loans</a>
#    - <a href='#9.8'>9.8. Distribution of NAME_HOUSING_TYPE</a>
#    - <a href='#9.9'>9.9. Distribution of Age</a>
#    - <a href='#9.10'>9.10. Effect of OCCUPATION_TYPE on default 
# - <a href='#10'>10. Preparation of Data</a>
#    - <a href='#10.1'>10.1. Feature Engineering of Application data </a>
#    - <a href='#10.2'>10.2 Using Bureau Data</a>
#    - <a href='#10.3'>10.3. Using Previous Application Data</a>
#    - <a href='#10.4'>10.4. Using POS_CASH_balance data</a>
#    - <a href='#10.5'>10.5 Using installments_payments data</a>
#    - <a href='#10.6'>10.6. Using Credit card balance data </a>
# - <a href='#11'>11. Dividing data into train, valid and test   </a>
# - <a href='#12'>12. Feature Selection using Information Value and Weight of Evidence </a>
# - <a href='#13'> 13. Data Imputation before applying machine learning algorithms</a>
# - <a href='#14'>14. Applying Machine Learning Algorithms </a>
#   - <a href='#14.1'>14.1. Applying Logistic Regression</a>
#   - <a href='#14.2'>14.2. Applying XGBoost </a>
#   - <a href='#14.3'>14.3. Applying CATBOOST</a>
#   - <a href='#14.4'>14.4. Applying LightGBM</a>
#   - <a href='#14.5'>14.5. Applying RandomForest</a>
# - <a href='#15'> 15. Evaluating machine learning algorithms accuracy on training and testing sets </a>
# - <a href='#16'> 16. Optimising selected machine learning model further by choosing best hyperparameters </a>
# - <a href='#17'> 17. Final Predictions </a>

# ## <a id="1"> 1. Importing necessary libraries and packages and reading files</a>

# In[ ]:


import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import cufflinks as cf
cf.go_offline()


# In[ ]:


application_train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
pos_cash= pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv')
bureau_balance = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv')
previous_application = pd.read_csv('../input/home-credit-default-risk/previous_application.csv')
insta_payments = pd.read_csv('../input/home-credit-default-risk/installments_payments.csv')
credit_card_balance = pd.read_csv('../input/home-credit-default-risk/credit_card_balance.csv')
bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv')
application_test = pd.read_csv('../input/home-credit-default-risk/application_test.csv')


# In[ ]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

application_train.head()


# ## <a id="2"> 2. Handling non-numerical variables</a>
# Machines can understand only numbers. Hence let us convert all non-numeric columns into numbers. Categorical variables will be converted into dummy columns , ordinal variables are converted into numbers by mapping and variables which are non-numeric and cannot be converted into numbers will be dropped from the model.

# In[ ]:


#List of non-numerical variables
application_train.select_dtypes(include=['O']).columns


# In[ ]:


#We cannot have non-numerical columns for modelling. We can have only numerical columns. Non-numerical columns can also be ordinal or categorical variables.  
col_for_dummies=application_train.select_dtypes(include=['O']).columns.drop(['FLAG_OWN_CAR','FLAG_OWN_REALTY','EMERGENCYSTATE_MODE'])
application_train_dummies = pd.get_dummies(application_train, columns = col_for_dummies, drop_first = True)
application_test_dummies = pd.get_dummies(application_test, columns = col_for_dummies, drop_first = True)


# In[ ]:


application_train_dummies.select_dtypes(include=['O']).columns


# In[ ]:


application_train_dummies['EMERGENCYSTATE_MODE'].value_counts()


# In[ ]:


#We cannot convert flag_own_car and flag_own_realty to column with yes or no etc. Lets rather map yes to 1 and no to 0
application_train_dummies['FLAG_OWN_CAR'] = application_train_dummies['FLAG_OWN_CAR'].map( {'Y':1, 'N':0})
application_train_dummies['FLAG_OWN_REALTY'] = application_train_dummies['FLAG_OWN_REALTY'].map( {'Y':1, 'N':0})
application_train_dummies['EMERGENCYSTATE_MODE'] = application_train_dummies['EMERGENCYSTATE_MODE'].map( {'Yes':1, 'No':0})

application_test_dummies['FLAG_OWN_CAR'] = application_train_dummies['FLAG_OWN_CAR'].map( {'Y':1, 'N':0})
application_test_dummies['FLAG_OWN_REALTY'] = application_train_dummies['FLAG_OWN_REALTY'].map( {'Y':1, 'N':0})
application_test_dummies['EMERGENCYSTATE_MODE'] = application_train_dummies['EMERGENCYSTATE_MODE'].map( {'Yes':1, 'No':0})
print(application_train_dummies.shape)
print(application_test_dummies.shape)


# In[ ]:


#We have 4 columns less in application_test_dummies. Lets see which are those 4 columns
#Sometimes test data does not have certain columns.
application_train_dummies.columns.difference(application_test_dummies.columns)


# ### <a id="3"> 3. Aligning Training and Testing Data</a>
# 
# There need to be the same features (columns) in both the training and testing data.
# One-hot encoding has created more columns in the training data because there were some categorical variables
# with categories not represented in the testing data. To remove the columns in the training data that are not in the testing
# data, we need to `align` the dataframes. First we extract the target column from the training data (because this is not in
# the testing data but we need to keep this information). When we do the align, we must make sure to set `axis = 1` 
# to align the dataframes based on the columns and not on the rows!
# 

# In[ ]:


train_labels = application_train_dummies['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
application_train_dummies, application_test_dummies = application_train_dummies.align(application_test_dummies, join = 'inner', axis = 1)

# Add the target back in
application_train_dummies['TARGET'] = train_labels

print('Training Features shape: ', application_train_dummies.shape)
print('Testing Features shape: ', application_test_dummies.shape)


# The training and testing datasets now have the same features which is required for machine learning. The number of features has grown significantly due to one-hot encoding. At some point we probably will want to try [dimensionality reduction (removing features that are not relevant)](https://en.wikipedia.org/wiki/Dimensionality_reduction) to reduce the size of the datasets.

# ### <a id="4"> 4. Handling missing values (using Iterative Imputer) prior to outlier detection </a>
# 
# We need to handle our missing values before we can do any kind of outlier detection.
# There are many ways to handle missing values. We can use fillna() and replace missing values with data's mean, median or most frequent value. The approach that we shall use below will be Iterative Imputer. Iterative imputer will consider the missing variable to be the dependent variable and all the other features will be independent variables. So there will be a regression and the independent variables will be used gto determine the dependent variable (which is the missing feature).

# In[ ]:


from sklearn.experimental import enable_iterative_imputer
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
import random


# In[ ]:


y=application_train_dummies[['SK_ID_CURR','TARGET']]
X=application_train_dummies.drop(columns=['TARGET'], axis=1)
X.head()


# In[ ]:


X_imputation = X.loc[:, (X.nunique() > 1000)]


# In[ ]:


X_imputation.columns


# In[ ]:


imputer = IterativeImputer(BayesianRidge())
imputed_total = pd.DataFrame(imputer.fit_transform(X_imputation))
imputed_total.columns = X_imputation.columns


# ###  <a id="5">5. OUTLIERS DETECTION</a>
# 
# In statistics, an outlier is an observation point that is distant from other observations. There are many ways for outlier detection. 
# 
# **Visual methods to spot and remove outliers**
# 1. Box-plot
# 2. Scatter plots
# 
# **Outliers detection and removal using mathematical function**
# 1. Z-score: Threshold of -3 to 3 is taken, and any point with z score not in this range is removed as an outlier.
# 2. IQR Score : This works similar to a box plot and z - score in the sense that a threshold IQR value is defined. IQR is the first quartile subtracted from the third quartile. Any point below the threshold IQR is removed. 
# 
# 
# **Clustering methods for outlier detection**
# 1. DBScan clustering (making clusters around data points). Minimum number of points are required to be in a cluster. There will be points that do not belong to any cluster or else points which are single in an entire cluster. So we can remove such noise points.  
# 2. Isolation Forest: Isolation Forest will output the predictions for each data point in an array. If the result is -1, it means that this specific data point is an outlier. If the result is 1, then it means that the data point is not an outlier
# 
# Here we shall use Isolation Forest method because it can handle missing values well and does not require scaling of inputs. 

# In[ ]:


from sklearn.ensemble import IsolationForest
rs=np.random.RandomState(0)
clf = IsolationForest(max_samples=100,random_state=rs, contamination=.1) 
clf.fit(imputed_total)
if_scores = clf.decision_function(imputed_total)

pred = clf.predict(imputed_total)
imputed_total['anomaly']=pred
outliers=imputed_total.loc[imputed_total['anomaly']==-1]
outlier_index=list(outliers.index)
#print(outlier_index)
#Find the number of anomalies and normal points here points classified -1 are anomalous
print(imputed_total['anomaly'].value_counts())


# In[ ]:


outlier_ID=list(outliers['SK_ID_CURR'])
X_new = X[~X.SK_ID_CURR.isin(outlier_ID)]
y_new = y[~y.SK_ID_CURR.isin(outlier_ID)]


# In[ ]:


print(X_new.shape)
print(X.shape)


# ### <a id='5.1'> 5.1. Anomaly detection </a>
# 
# Though we have removed outliers using Isolation Forest we will still see the data once to check for any anomalies. Isolation Forest or any outlier detection method assumes that outlier is a point which is in minority and does not resemble the other majority points. However sometimes some abberation points are too many in numbers too. Let us see if there is any such anamoly that we find

# In[ ]:


X_new.describe()


# **Univariate outliers detection**
# 
# **Negative numbers:**
# DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH, DAYS_LAST_PHONE_CHANGE
# 
# Numbers are negative since they are taken relative to the date of application. So we need to change them to positive.
# 
# **Maximum value discrepancy**
# DAYS_EMPLOYED: 365243 days(over 1000 years)
# OWN_CAR_AGE: 91 Years
# 

# In[ ]:


#Checking the anamalous variables values in years
print('DAYS_BIRTH stats in years:','\n',(X_new['DAYS_BIRTH'] / -365).describe(),'\n')
print('Check the stats in years to see if there is any anomalous behavior')
print('DAYS_EMPLOYED stats in years:','\n',(X_new['DAYS_EMPLOYED'] / -365).describe(),'\n')
print('DAYS_REGISTRATION stats in years:','\n',(X_new['DAYS_REGISTRATION'] / -365).describe(),'\n')
print('DAYS_ID_PUBLISH stats in years:','\n',(X_new['DAYS_ID_PUBLISH'] / -365).describe(),'\n')
print('DAYS_LAST_PHONE_CHANGE stats in years:','\n',(X_new['DAYS_LAST_PHONE_CHANGE'] / -365).describe(),'\n')


# As we can see there is anomaly in Days_employed as it is highly unlikely that a person will be employed for 1000 years.

# In[ ]:


X_new['DAYS_EMPLOYED'].max()


# In[ ]:


# Replace the error values in Days_employed with nan
X_new['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
application_test_dummies['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)


# In[ ]:


X_new.describe()


# The data now looks nice and clean.

# ## <a id="6">6. Missing data in application_train</a>

# In[ ]:


# checking missing data
total = X_new.isnull().sum().sort_values(ascending = False)
percent = (X_new.isnull().sum()/X_new.isnull().count()*100).sort_values(ascending = False)
missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_application_train_data.head(20)


# ## <a id="7">7. Duplicate data in application_train </a>

# In[ ]:


columns_without_id = [col for col in X_new.columns if col!='SK_ID_CURR']
#Checking for duplicates in the data.
X_new[X_new.duplicated(subset = columns_without_id, keep=False)]
print('The no of duplicates in the data:',X_new[X_new.duplicated(subset = columns_without_id, keep=False)]
      .shape[0])


# ## <a id="8">8. Checking for data imbalance</a>

# In[ ]:


y_new['TARGET'].value_counts()


# We see that the class is clearly imbalanced with cases of default as very low compared to overall cases. So we need to balance the data when we use Machine learning models.

# ## <a id="9">  9. Exploratory Data Analysis for application_train by visualisation </a>

# In[ ]:


X_new.head()


# ### <a id="9.1"> 9.1. Distribution of income</a>
# 

# In[ ]:


import seaborn as sns # for making plots with seaborn
color = sns.color_palette()

plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_INCOME_TOTAL")
ax = sns.distplot(X_new["AMT_INCOME_TOTAL"])


# The distribution is right skewed and there are extreme values, we can apply log distribution.

# In[ ]:


X_new["AMT_INCOME_TOTAL"].describe()


# In[ ]:


application_train=pd.merge(X_new,y_new,on='SK_ID_CURR')


# In[ ]:


(application_train[application_train['AMT_INCOME_TOTAL'] > 1000000]['TARGET'].value_counts())/len(application_train[application_train['AMT_INCOME_TOTAL'] > 1000000])*100


# People with high income tend to not default

# In[ ]:


#boxcox=0 means we are taking log transformation of data to show it as normal form

from scipy.stats import boxcox
from matplotlib import pyplot


np.log(application_train['AMT_INCOME_TOTAL']).iplot(kind='histogram', bins=100,
                               xTitle = 'log(INCOME_TOTAL)',yTitle ='Count corresponding to Incomes',
                               title='Distribution of log(AMT_INCOME_TOTAL)')


# We see that income variable gets normal distribution when it is log transformed. 

# ### <a id="9.2">9.2. Distribution of credit</a>
# 

# In[ ]:


import seaborn as sns # for making plots with seaborn
color = sns.color_palette()

plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_CREDIT")
ax = sns.distplot(application_train["AMT_CREDIT"])


# In[ ]:


application_train["AMT_CREDIT"].describe()


# In[ ]:


(application_train[application_train['AMT_CREDIT']>2000000]['TARGET'].value_counts())/len(application_train[application_train['AMT_CREDIT']>2000000])*100


# People who take more credit default less.

# ### <a id="9.3"> 9.3. Distribution of loan types</a>

# In[ ]:


original_train_data = pd.read_csv('../input/home-credit-default-risk/application_train.csv')


contract_val = original_train_data['NAME_CONTRACT_TYPE'].value_counts()
contract_df = pd.DataFrame({'labels': contract_val.index,
                   'values': contract_val.values
                  })
contract_df.iplot(kind='pie',labels='labels',values='values', title='Types of Loan')


# More people are interested to take cash loans than revolving loans.

# ### <a id="9.4"> 9.4. Distribution of NAME_INCOME_TYPE</a>
# 

# In[ ]:


original_train_data["NAME_INCOME_TYPE"].iplot(kind="histogram", bins=20, theme="white", title="Passenger's Income Types",
                                            xTitle='Name of Income Types', yTitle='Count')


# In[ ]:


education_val = original_train_data['NAME_INCOME_TYPE'].value_counts()

education_val_y0 = []
education_val_y1 = []
for val in education_val.index:
    education_val_y1.append(np.sum(original_train_data['TARGET'][original_train_data['NAME_INCOME_TYPE']==val] == 1))
    education_val_y0.append(np.sum(original_train_data['TARGET'][original_train_data['NAME_INCOME_TYPE']==val] == 0))

data = [go.Bar(x = education_val.index, y = ((education_val_y1 / education_val.sum()) * 100), name='Default' ),
        go.Bar(x = education_val.index, y = ((education_val_y0 / education_val.sum()) * 100), name='No default' )]

layout = go.Layout(
    title = "Income of people affecting default on loans",
    xaxis=dict(
        title='Income of people',
       ),
    yaxis=dict(
        title='Count of people accompanying in %',
        )
)

fig = go.Figure(data = data, layout=layout) 
fig.layout.template = 'plotly_dark'
py.iplot(fig)


# ### <a id="9.5"> 9.5. Distribution of NAME_TYPE_SUITE</a>

# Who accompanied the person while taking the loan? 

# In[ ]:


original_train_data["NAME_TYPE_SUITE"].iplot(kind="histogram", bins=20, theme="white", title="Accompanying Person",
                                            xTitle='People accompanying', yTitle='Count')


# Most people are unaccompanied.

# ### <a id="9.6">9.6. Distribution of NAME_EDUCATION_TYPE</a>
# 

# In[ ]:


education_val = original_train_data['NAME_EDUCATION_TYPE'].value_counts()

education_val_y0 = []
education_val_y1 = []
for val in education_val.index:
    education_val_y1.append(np.sum(original_train_data['TARGET'][original_train_data['NAME_EDUCATION_TYPE']==val] == 1))
    education_val_y0.append(np.sum(original_train_data['TARGET'][original_train_data['NAME_EDUCATION_TYPE']==val] == 0))

data = [go.Bar(x = education_val.index, y = ((education_val_y1 / education_val.sum()) * 100), name='Default' ),
        go.Bar(x = education_val.index, y = ((education_val_y0 / education_val.sum()) * 100), name='No default' )]

layout = go.Layout(
    title = "Education sources of Applicants in terms of loan is repayed or not  in %",
    xaxis=dict(
        title='Education of Applicants',
       ),
    yaxis=dict(
        title='Count of applicants in %',
        )
)

fig = go.Figure(data = data, layout=layout) 
fig.layout.template = 'plotly_dark'
py.iplot(fig)


# People with a degree are able to pay back mostly.

# ### <a id="9.7"> 9.7. Effect of marital status on ability to pay back loans</a>

# In[ ]:


education_val = original_train_data['NAME_FAMILY_STATUS'].value_counts()

education_val_y0 = []
education_val_y1 = []
for val in education_val.index:
    education_val_y1.append(np.sum(original_train_data['TARGET'][original_train_data['NAME_FAMILY_STATUS']==val] == 1))
    education_val_y0.append(np.sum(original_train_data['TARGET'][original_train_data['NAME_FAMILY_STATUS']==val] == 0))

data = [go.Bar(x = education_val.index, y = ((education_val_y1 / education_val.sum()) * 100), name='Default' ),
        go.Bar(x = education_val.index, y = ((education_val_y0 / education_val.sum()) * 100), name='No default' )]

layout = go.Layout(
    title = "Family status of Applicant in terms of loan is repayed or not in %",
    xaxis=dict(
        title='Family status of Applicants',
       ),
    yaxis=dict(
        title='Count of applicants in %',
        )
)

fig = go.Figure(data = data, layout=layout) 
fig.layout.template = 'plotly_dark'
py.iplot(fig)


# ### <a id="9.8"> 9.8. Distribution of NAME_HOUSING_TYPE</a>
# 

# In[ ]:


education_val = original_train_data['NAME_HOUSING_TYPE'].value_counts()

education_val_y0 = []
education_val_y1 = []
for val in education_val.index:
    education_val_y1.append(np.sum(original_train_data['TARGET'][original_train_data['NAME_HOUSING_TYPE']==val] == 1))
    education_val_y0.append(np.sum(original_train_data['TARGET'][original_train_data['NAME_HOUSING_TYPE']==val] == 0))

data = [go.Bar(x = education_val.index, y = ((education_val_y1 / education_val.sum()) * 100), name='Default' ),
        go.Bar(x = education_val.index, y = ((education_val_y0 / education_val.sum()) * 100), name='No default' )]

layout = go.Layout(
    title = "Housing Type of Applicant in terms of loan is repayed or not in %",
    xaxis=dict(
        title='Housing Type of Applicants',
       ),
    yaxis=dict(
        title='Count of applicants in %',
        )
)

fig = go.Figure(data = data, layout=layout) 
fig.layout.template = 'plotly_dark'
py.iplot(fig)


# People in office apartment, co-op apartment almost never default.

# ### <a id="9.9"> 9.9. Distribution of AGE</a>
# 

# In[ ]:


(original_train_data["DAYS_BIRTH"]/-365).iplot(kind="histogram", bins=20, theme="white", title="Customer's Ages",
                                            xTitle='Age of customer', yTitle='Count')


# ### <a id="9.10">9.10. Effect of OCCUPATION_TYPE on default </a>

# In[ ]:


parameter_val = original_train_data['OCCUPATION_TYPE'].value_counts()

parameter_val_y0 = []
parameter_val_y1 = []
for val in parameter_val.index:
    parameter_val_y1.append(np.sum(original_train_data['TARGET'][original_train_data['OCCUPATION_TYPE']==val] == 1))
    parameter_val_y0.append(np.sum(original_train_data['TARGET'][original_train_data['OCCUPATION_TYPE']==val] == 0))

data = [go.Bar(x = parameter_val.index, y = ((parameter_val_y1 / parameter_val.sum()) * 100), name='Default' ),
        go.Bar(x = parameter_val.index, y = ((parameter_val_y0 / parameter_val.sum()) * 100), name='No default' )]

layout = go.Layout(
    title = "Occupation type of people affecting default on loans",
    xaxis=dict(
        title='Occupation type of people',
       ),
    yaxis=dict(
        title='Count of people Occupation that type of housing in %',
        )
)

fig = go.Figure(data = data, layout=layout) 
fig.layout.template = 'plotly_dark'
py.iplot(fig)


# Highly skilled people more likely to pay back and low skilled not so likely to pay back loans

# ## <a id="10"> 10. Combining other tables to extract their data </a> 
# 
# There are many tables apart from application_train. Due to memory and space restrictions on Kaggle, I am unable to describe them here. But one can easily look up the description on the competition data page. We need to extract information from these 

# <a id='3.4.1'></a>
# <h3> 10.1 Feature Engineering of Application data </h3>

# In[ ]:


#Flag to represent when credit > income
application_train_dummies['Credit_flag'] = application_train_dummies['AMT_INCOME_TOTAL'] > application_train_dummies['AMT_CREDIT']
application_train_dummies['Percent_Days_employed'] = application_train_dummies['DAYS_EMPLOYED']/application_train_dummies['DAYS_BIRTH']*100
application_train_dummies['Annuity_as_percent_income'] = application_train_dummies['AMT_ANNUITY']/ application_train_dummies['AMT_INCOME_TOTAL']*100
application_train_dummies['Credit_as_percent_income'] = application_train_dummies['AMT_CREDIT']/application_train_dummies['AMT_INCOME_TOTAL']*100

application_test_dummies['Credit_flag'] = application_test_dummies['AMT_INCOME_TOTAL'] > application_test_dummies['AMT_CREDIT']
application_test_dummies['Percent_Days_employed'] = application_test_dummies['DAYS_EMPLOYED']/application_test_dummies['DAYS_BIRTH']*100
application_test_dummies['Annuity_as_percent_income'] = application_test_dummies['AMT_ANNUITY']/ application_test_dummies['AMT_INCOME_TOTAL']*100
application_test_dummies['Credit_as_percent_income'] = application_test_dummies['AMT_CREDIT']/application_test_dummies['AMT_INCOME_TOTAL']*100


# <a id='10.2'></a>
# <h3> 10.2 Using Bureau Data </h3>

# In[ ]:


# Combining numerical features
grp = bureau.drop(['SK_ID_BUREAU'], axis = 1).groupby(by=['SK_ID_CURR']).mean().reset_index()
grp.columns = ['BUREAU_'+column if column !='SK_ID_CURR' else column for column in grp.columns]
application_bureau = application_train_dummies.merge(grp, on='SK_ID_CURR', how='left')
application_bureau.update(application_bureau[grp.columns].fillna(0))

application_bureau_test = application_test_dummies.merge(grp, on='SK_ID_CURR', how='left')
application_bureau_test.update(application_bureau_test[grp.columns].fillna(0))


# In[ ]:


# Combining categorical features
bureau_categorical = pd.get_dummies(bureau.select_dtypes('object'))
bureau_categorical['SK_ID_CURR'] = bureau['SK_ID_CURR']
grp = bureau_categorical.groupby(by = ['SK_ID_CURR']).mean().reset_index()
grp.columns = ['BUREAU_'+column if column !='SK_ID_CURR' else column for column in grp.columns]
application_bureau = application_bureau.merge(grp, on='SK_ID_CURR', how='left')
application_bureau.update(application_bureau[grp.columns].fillna(0))

application_bureau_test = application_bureau_test.merge(grp, on='SK_ID_CURR', how='left')
application_bureau_test.update(application_bureau_test[grp.columns].fillna(0))


# <a id='10.2.1'></a>
# <h3> 10.2.1. Feature Engineering of Bureau Data </h3>

# In[ ]:


# Number of past loans per customer
grp = bureau.groupby(by = ['SK_ID_CURR'])['SK_ID_BUREAU'].count().reset_index().rename(columns = {'SK_ID_BUREAU': 'BUREAU_LOAN_COUNT'})

application_bureau = application_bureau.merge(grp, on='SK_ID_CURR', how='left')
application_bureau['BUREAU_LOAN_COUNT'] = application_bureau['BUREAU_LOAN_COUNT'].fillna(0)

application_bureau_test = application_bureau_test.merge(grp, on='SK_ID_CURR', how='left')
application_bureau_test['BUREAU_LOAN_COUNT'] = application_bureau_test['BUREAU_LOAN_COUNT'].fillna(0)


# In[ ]:


# Number of types of past loans per customer 
grp = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by = ['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})

application_bureau = application_bureau.merge(grp, on='SK_ID_CURR', how='left')
application_bureau['BUREAU_LOAN_TYPES'] = application_bureau['BUREAU_LOAN_TYPES'].fillna(0)

application_bureau_test = application_bureau_test.merge(grp, on='SK_ID_CURR', how='left')
application_bureau_test['BUREAU_LOAN_TYPES'] = application_bureau_test['BUREAU_LOAN_TYPES'].fillna(0)


# In[ ]:


# Debt over credit ratio 
bureau['AMT_CREDIT_SUM'] = bureau['AMT_CREDIT_SUM'].fillna(0)
bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)

grp1 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM': 'TOTAL_CREDIT_SUM'})

grp2 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_DEBT':'TOTAL_CREDIT_SUM_DEBT'})

grp1['DEBT_CREDIT_RATIO'] = grp2['TOTAL_CREDIT_SUM_DEBT']/grp1['TOTAL_CREDIT_SUM']

del grp1['TOTAL_CREDIT_SUM']

application_bureau = application_bureau.merge(grp1, on='SK_ID_CURR', how='left')
application_bureau['DEBT_CREDIT_RATIO'] = application_bureau['DEBT_CREDIT_RATIO'].fillna(0)
application_bureau['DEBT_CREDIT_RATIO'] = application_bureau.replace([np.inf, -np.inf], 0)
application_bureau['DEBT_CREDIT_RATIO'] = pd.to_numeric(application_bureau['DEBT_CREDIT_RATIO'], downcast='float')

application_bureau_test = application_bureau_test.merge(grp1, on='SK_ID_CURR', how='left')
application_bureau_test['DEBT_CREDIT_RATIO'] = application_bureau_test['DEBT_CREDIT_RATIO'].fillna(0)
application_bureau_test['DEBT_CREDIT_RATIO'] = application_bureau_test.replace([np.inf, -np.inf], 0)
application_bureau_test['DEBT_CREDIT_RATIO'] = pd.to_numeric(application_bureau_test['DEBT_CREDIT_RATIO'], downcast='float')


# In[ ]:


(application_bureau[application_bureau['DEBT_CREDIT_RATIO'] > 0.5]['TARGET'].value_counts()/len(application_bureau[application_bureau['DEBT_CREDIT_RATIO'] > 0.5]))*100


# In[ ]:


# Overdue over debt ratio
bureau['AMT_CREDIT_SUM_OVERDUE'] = bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(0)
bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)

grp1 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM_OVERDUE']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})

grp2 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_DEBT':'TOTAL_CUSTOMER_DEBT'})

grp1['OVERDUE_DEBT_RATIO'] = grp1['TOTAL_CUSTOMER_OVERDUE']/grp2['TOTAL_CUSTOMER_DEBT']

del grp1['TOTAL_CUSTOMER_OVERDUE']

application_bureau = application_bureau.merge(grp1, on='SK_ID_CURR', how='left')
application_bureau['OVERDUE_DEBT_RATIO'] = application_bureau['OVERDUE_DEBT_RATIO'].fillna(0)
application_bureau['OVERDUE_DEBT_RATIO'] = application_bureau.replace([np.inf, -np.inf], 0)
application_bureau['OVERDUE_DEBT_RATIO'] = pd.to_numeric(application_bureau['OVERDUE_DEBT_RATIO'], downcast='float')

application_bureau_test = application_bureau_test.merge(grp1, on='SK_ID_CURR', how='left')
application_bureau_test['OVERDUE_DEBT_RATIO'] = application_bureau_test['OVERDUE_DEBT_RATIO'].fillna(0)
application_bureau_test['OVERDUE_DEBT_RATIO'] = application_bureau_test.replace([np.inf, -np.inf], 0)
application_bureau_test['OVERDUE_DEBT_RATIO'] = pd.to_numeric(application_bureau_test['OVERDUE_DEBT_RATIO'], downcast='float')


# In[ ]:


import gc

gc.collect()


# <a id='10.3'></a>
# <h3> 10.3 Using Previous Application Data </h3>

# In[ ]:


def isOneToOne(df, col1, col2):
    first = df.drop_duplicates([col1, col2]).groupby(col1)[col2].count().max()
    second = df.drop_duplicates([col1, col2]).groupby(col2)[col1].count().max()
    return first + second == 2

isOneToOne(previous_application,'SK_ID_CURR','SK_ID_PREV')


# In[ ]:


# Number of previous applications per customer
grp = previous_application[['SK_ID_CURR','SK_ID_PREV']].groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index().rename(columns={'SK_ID_PREV':'PREV_APP_COUNT'})

# Take only the IDs which are present in application_bureau
application_bureau_prev = application_bureau.merge(grp, on =['SK_ID_CURR'], how = 'left')
application_bureau_prev_test = application_bureau_test.merge(grp, on =['SK_ID_CURR'], how = 'left')

#Fill NA for previous application counts (lets say there was an application ID present in application_bureau but not present
# in grp, then that means that person never took loan previously, so count of previous loan for that person = 0)
application_bureau_prev['PREV_APP_COUNT'] = application_bureau_prev['PREV_APP_COUNT'].fillna(0)
application_bureau_prev_test['PREV_APP_COUNT'] = application_bureau_prev_test['PREV_APP_COUNT'].fillna(0)


# In[ ]:


# Combining numerical features

#Take the mean of all the parameters (grouping by SK_ID_CURR)
grp = previous_application.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()

#Add prefix prev in front of all columns so that we know that these columns are from previous_application
prev_columns = ['PREV_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]

#Change the columns
grp.columns = prev_columns

application_bureau_prev = application_bureau_prev.merge(grp, on =['SK_ID_CURR'], how = 'left')
application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))
application_bureau_prev_test = application_bureau_prev_test.merge(grp, on =['SK_ID_CURR'], how = 'left')
application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))


# In[ ]:


# Combining categorical features
prev_categorical = pd.get_dummies(previous_application.select_dtypes('object'))
prev_categorical['SK_ID_CURR'] = previous_application['SK_ID_CURR']

grp = prev_categorical.groupby('SK_ID_CURR').mean().reset_index()
grp.columns = ['PREV_'+column if column != 'SK_ID_CURR' else column for column in grp.columns]

application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

application_bureau_prev_test = application_bureau_prev_test.merge(grp, on=['SK_ID_CURR'], how='left')
application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))


# In[ ]:


gc.collect()


# <a id='10.4'></a>
# <h3> 10.4. Using POS_CASH_balance data </h3>

# In[ ]:


# Combining numerical features
grp = pos_cash.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
prev_columns = ['POS_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]
grp.columns = prev_columns


# In[ ]:


application_bureau_prev = application_bureau_prev.merge(grp, on =['SK_ID_CURR'], how = 'left')
application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

application_bureau_prev_test = application_bureau_prev_test.merge(grp, on =['SK_ID_CURR'], how = 'left')
application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))


# In[ ]:


# Combining categorical features
pos_cash_categorical = pd.get_dummies(pos_cash.select_dtypes('object'))
pos_cash_categorical['SK_ID_CURR'] = pos_cash['SK_ID_CURR']

grp = pos_cash_categorical.groupby('SK_ID_CURR').mean().reset_index()
grp.columns = ['POS_'+column if column != 'SK_ID_CURR' else column for column in grp.columns]


# In[ ]:


application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

application_bureau_prev_test = application_bureau_prev_test.merge(grp, on=['SK_ID_CURR'], how='left')
application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))


# In[ ]:


gc.collect()


# <a id='10.5'></a>
# <h3> 10.5. Using installments_payments data</h3>

# In[ ]:


# Combining numerical features and there are no categorical features in this dataset
grp = insta_payments.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
prev_columns = ['INSTA_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]
grp.columns = prev_columns
application_bureau_prev = application_bureau_prev.merge(grp, on =['SK_ID_CURR'], how = 'left')
application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))
application_bureau_prev_test = application_bureau_prev_test.merge(grp, on =['SK_ID_CURR'], how = 'left')
application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))


# In[ ]:


gc.collect()


# <a id='10.6'></a>
# <h3> 10.6. Using Credit card balance data </h3>

# In[ ]:


credit_card=credit_card_balance
# Combining numerical features
grp = credit_card.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
prev_columns = ['CREDIT_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]
grp.columns = prev_columns
application_bureau_prev = application_bureau_prev.merge(grp, on =['SK_ID_CURR'], how = 'left')
application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

application_bureau_prev_test = application_bureau_prev_test.merge(grp, on =['SK_ID_CURR'], how = 'left')
application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))


# In[ ]:


# Combining categorical features
credit_categorical = pd.get_dummies(credit_card.select_dtypes('object'))
credit_categorical['SK_ID_CURR'] = credit_card['SK_ID_CURR']

grp = credit_categorical.groupby('SK_ID_CURR').mean().reset_index()
grp.columns = ['CREDIT_'+column if column != 'SK_ID_CURR' else column for column in grp.columns]

application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

application_bureau_prev_test = application_bureau_prev_test.merge(grp, on=['SK_ID_CURR'], how='left')
application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))


# <h3> <a id='11'> 11. Dividing data into train, valid and test </a> </h3> 

# `X=application_bureau_prev.drop(columns=['TARGET'])
# y=application_bureau_prev['TARGET']`
# 
# `from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)`

# <h3> <a id='12'> 12. Feature Selection using Information Value and Weight of Evidence </a> </h3>

# There are many methods for feature selection. Some of them include feature importance using XGBoost and RandomForest. Other methods are forward or backward elimination and Boruta. Here we use one of the most common methods for feature selection, Information value and weight of evidence to determine the feature selection for credit risk analysis.
# 
# 
# **Code to calculate IV and WOE**
# 
# max_bin = 20
# 
# force_bin = 3
# 
# #Define a binning function
# 
# def mono_bin(Y, X, n = max_bin):
# 
#     df1 = pd.DataFrame({"X": X, "Y": Y})
#     justmiss = df1[['X','Y']][df1.X.isnull()]
#     notmiss = df1[['X','Y']][df1.X.notnull()]
#     r = 0
#     while np.abs(r) < 1:
#         try:
#             d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
#             d2 = d1.groupby('Bucket', as_index=True)
#             r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
#             n = n - 1 
#         except Exception as e:
#             n = n - 1
# 
#     if len(d2) == 1:
#         n = force_bin         
#         bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
#         if len(np.unique(bins)) == 2:
#             bins = np.insert(bins, 0, 1)
#             bins[1] = bins[1]-(bins[1]/2)
#         d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
#         d2 = d1.groupby('Bucket', as_index=True)
#     
#     d3 = pd.DataFrame({},index=[])
#     d3["MIN_VALUE"] = d2.min().X
#     d3["MAX_VALUE"] = d2.max().X
#     d3["COUNT"] = d2.count().Y
#     d3["EVENT"] = d2.sum().Y
#     d3["NONEVENT"] = d2.count().Y - d2.sum().Y
#     d3=d3.reset_index(drop=True)
#     
#     if len(justmiss.index) > 0:
#         d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
#         d4["MAX_VALUE"] = np.nan
#         d4["COUNT"] = justmiss.count().Y
#         d4["EVENT"] = justmiss.sum().Y
#         d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
#         d3 = d3.append(d4,ignore_index=True)
#     
#     d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
#     d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
#     d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
#     d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
#     d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
#     d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
#     d3["VAR_NAME"] = "VAR"
#     d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       
#     d3 = d3.replace([np.inf, -np.inf], 0)
#     d3.IV = d3.IV.sum()
#     
#     return(d3)
# 
# 
# def char_bin(Y, X):
#         
#     df1 = pd.DataFrame({"X": X, "Y": Y})
#     justmiss = df1[['X','Y']][df1.X.isnull()]
#     notmiss = df1[['X','Y']][df1.X.notnull()]    
#     df2 = notmiss.groupby('X',as_index=True)
#     
#     d3 = pd.DataFrame({},index=[])
#     d3["COUNT"] = df2.count().Y
#     d3["MIN_VALUE"] = df2.sum().Y.index
#     d3["MAX_VALUE"] = d3["MIN_VALUE"]
#     d3["EVENT"] = df2.sum().Y
#     d3["NONEVENT"] = df2.count().Y - df2.sum().Y
#     
#     if len(justmiss.index) > 0:
#         d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
#         d4["MAX_VALUE"] = np.nan
#         d4["COUNT"] = justmiss.count().Y
#         d4["EVENT"] = justmiss.sum().Y
#         d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
#         d3 = d3.append(d4,ignore_index=True)
#     
#     d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
#     d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
#     d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
#     d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
#     d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
#     d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
#     d3["VAR_NAME"] = "VAR"
#     d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
#     d3 = d3.replace([np.inf, -np.inf], 0)
#     d3.IV = d3.IV.sum()
#     d3 = d3.reset_index(drop=True)
#     
#     return(d3)
# 
# -------------------------------------------------------------------------------------------------------------------
# 
# def data_vars(df1, target):
#     
#     stack = traceback.extract_stack()
#     filename, lineno, function_name, code = stack[-2]
#     vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
#     final = (re.findall(r"[\w']+", vars_name))[-1]
#     
#     x = df1.dtypes.index
#     count = -1
#     
#     for i in x:
#         if i.upper() not in (final.upper()):
#             if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
#                 conv = mono_bin(target, df1[i])
#                 conv["VAR_NAME"] = i
#                 count = count + 1
#             else:
#                 conv = char_bin(target, df1[i])
#                 conv["VAR_NAME"] = i            
#                 count = count + 1
#                 
#             if count == 0:
#                 iv_df = conv
#             else:
#                 iv_df = iv_df.append(conv,ignore_index=True)
#     
#     iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
#     iv = iv.reset_index()
#     return(iv_df,iv)
#     
# -------------------------------------------------------------------------------------------------------------------
# 
# `import pandas as pd
# import numpy as np
# import pandas.core.algorithms as algos
# from pandas import Series
# import scipy.stats.stats as stats
# import re
# import traceback
# import string
# final_iv, IV = data_vars(X_train, y_train)
# IV`
# 
# **Output of IV**
# 
# Index         | VAR_NAME | IV
# ----------------------|-----------|---------------
#  Logistic Regression with Selected features  |    AMT_ANNUITY  | 4.050335e-04
#  Random Forest with Selected features  |    AMT_CREDIT |    2.415783e-03  
#  LightGBM with Selected features |    AMT_GOODS_PRICE |    3.591973e-02
#  CATBoost with Selected features |    AMT_INCOME_TOTAL |    2.504913e-03
#  XGBoost with Selected features |    AMT_REQ_CREDIT_BUREAU_DAY |    1.289777e-02
# 
# 
# In case of Information value, predictions with information value < 0.02 are useless for predictions, so we will only consider columns with IV > 0.02.
# 
# `list_of_columns=IV[IV['IV'] > 0.02]['VAR_NAME'].to_list()
# print(len(list_of_columns))`
# 
# 63
# 
# We find that only 63 columns are efficient in predicting the default by a customer. Hence we shall only consider those columns 
# 
# `X_train_selected_features=X_train[list_of_columns]
# X_test_selected_features=X_test[list_of_columns]
# X_train_selected_features['SK_ID_CURR']=X_train['SK_ID_CURR']
# X_test_selected_features['SK_ID_CURR']=X_test['SK_ID_CURR']`
# 
# `application_bureau_prev_test_selected_features=application_bureau_prev_test[list_of_columns]
# application_bureau_prev_test_selected_features['SK_ID_CURR']=application_bureau_prev_test['SK_ID_CURR']`

# ## <a id ='13'> 13. Data Imputation before applying machine learning algorithms </a>

# There are many ways to handle missing values. We can use fillna() and replace missing values with data's mean, median or most frequent value. The approach that we shall use below will be Iterative Imputer. Iterative imputer will consider the missing variable to be the dependent varibale and all the other features will be independent variables. Then it will apply regression and the independent variables will be used to determine the dependent variable (which is the missing feature).
# 
# `imputer = IterativeImputer(BayesianRidge())
# X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_selected_features))
# X_train_imputed.columns = X_train_selected_features.columns`
# 
# `imputer = IterativeImputer(BayesianRidge())
# application_bureau_prev_test_selected_features_subset1=application_bureau_prev_test_selected_features.iloc[:, np.r_[62,0:30]]
# app_bur_prev_test_imputed_subset1 = pd.DataFrame(imputer.fit_transform(application_bureau_prev_test_selected_features_subset1))
# app_bur_prev_test_imputed_subset1.columns = application_bureau_prev_test_selected_features_subset1.columns`
# 
# `application_bureau_prev_test_selected_features_subset2=application_bureau_prev_test_selected_features.iloc[:, np.r_[31:63]]
# app_bur_prev_test_imputed_subset2 = pd.DataFrame(imputer.fit_transform(application_bureau_prev_test_selected_features_subset2))
# app_bur_prev_test_imputed_subset2.columns = application_bureau_prev_test_selected_features_subset2.columns`
# 
# `app_bur_prev_test_imputed=pd.merge(app_bur_prev_test_imputed_subset1, app_bur_prev_test_imputed_subset2, on= 'SK_ID_CURR')`
# 
# `imputer = IterativeImputer(BayesianRidge())
# X_test_imputed = pd.DataFrame(imputer.fit_transform(X_test_selected_features))
# X_test_imputed.columns = X_test_selected_features.columns`
# 
# `print(X_test_imputed.shape)
# print(X_train_imputed.shape)
# print(app_bur_prev_test_imputed.shape)`
# Output
# 
# (61503, 63)
# 
# (246008, 63)
# 
# (48744, 62)
# 
# **Align the training and testing dataframes, keep only columns present in both dataframes**
# 
# We see above that the number of columns in test and training set are not same.
# 
# `X_train_imputed, app_bur_prev_test_imputed  = app_bur_prev_test_imputed.align(X_train_imputed, join = 'inner', axis = 1)
# X_train_imputed,X_test_imputed= X_train_imputed.align(X_test_imputed, join = 'inner', axis = 1)`
# 

# ## <a id='14'> 14. Applying Machine Learning Algorithms (using cross validation) </a>

# **14.1. Applying Logistic Regression**
# 
# `from sklearn.linear_model import LogisticRegression
# lr_clf = LogisticRegression(random_state = 0, class_weight='balanced')
# lr_clf.fit(X_train_imputed, y_train)
# from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import cross_val_score
# y_train_pred_lr=cross_val_predict(lr_clf, X_train_imputed, y_train, cv=3)
# print('Accuracy on Training set:',cross_val_score(lr_clf, X_train_imputed,y_train, cv=3, scoring='accuracy'))
# print('Accuracy on Test set:',cross_val_score(lr_clf, X_test_imputed,y_test, cv=3, scoring='accuracy'))`
# 
# *Accuracy on Training set: [0.66357739 0.68726373 0.6783615 ],
# Accuracy on Test set: [0.66164277 0.66167504 0.65356098]*
# 
# -----------------------------------------------------------------------------------------------------------------------
# 
# **14.2. Applying XGBoost**
# 
# `#Scale_pos_weight if set to sum(negative instances)/ sum(negative instances) will take care of imbalanced data in the dataset
# scale_pos_weight_value=y_train.value_counts().values.tolist()[0]/y_train.value_counts().values.tolist()[1]
# from xgboost import XGBClassifier
# XGB_clf = XGBClassifier(scale_pos_weight=scale_pos_weight_value)
# XGB_clf.fit(X_train_imputed, y_train)
# print('Accuracy on Training set:',cross_val_score(XGB_clf, X_train_imputed, y_train, cv=3, scoring='accuracy'))
# print('Accuracy on Test set:',cross_val_score(XGB_clf, X_test_imputed, y_test, cv=3, scoring='accuracy'))`
# 
# *Accuracy on Training set: [0.74863421 0.75014024 0.74932319],
# Accuracy on Test set: [0.81938347 0.82391103 0.82102439]*
# 
# -----------------------------------------------------------------------------------------------------------------------
# 
# **14.3. Applying CATBOOST**
# 
# `cols_numeric = X_train_imputed.select_dtypes([np.number]).columns
# cols_categorical=X_train_imputed.columns.difference(cols_numeric)
# #We find that there are no categorical columns.
# from catboost import CatBoostClassifier
# CatBoost_clf=CatBoostClassifier(scale_pos_weight=scale_pos_weight_value)
# #CatBoost_clf=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')
# CatBoost_clf.fit(X_train_imputed, y_train)
# print('Accuracy on Training set:',cross_val_score(CatBoost_clf, X_train_imputed, y_train, cv=3, scoring='accuracy'))
# print('Accuracy on Test set:',cross_val_score(CatBoost_clf, X_test_imputed, y_test, cv=3, scoring='accuracy'))`
# 
# *Accuracy on Training set: [0.76207258 0.76210336 0.75900588],
# Accuracy on Test set: [0.81211589 0.8081557  0.81302439]*
# 
# -----------------------------------------------------------------------------------------------------------------------
# 
# **14.4. Applying LightGBM**
# 
# `import lightgbm as lgb
# LightGBM_clf=lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight_value)
# LightGBM_clf.fit(X_train_imputed, y_train)
# print('Accuracy on Training set:',cross_val_score(LightGBM_clf, X_train_imputed, y_train, cv=3, scoring='accuracy'))
# print('Accuracy on Test set:',cross_val_score(LightGBM_clf, X_test_imputed, y_test, cv=3, scoring='accuracy'))`
# 
# *Accuracy on Training set: [0.70613629 0.70886076 0.70815346],
# Accuracy on Test set: [0.75158521 0.74947564 0.75639024]*
# 
# -----------------------------------------------------------------------------------------------------------------------
# 
# **14.5. Applying RandomForest**
# 
# `#class_weight = 'balanced' ensures that RandomForest works well on imbalanced datasets.
# from sklearn.ensemble import RandomForestClassifier
# rf_clf = RandomForestClassifier(n_estimators = 10, random_state = 0, n_jobs=-1, class_weight="balanced")
# rf_clf.fit(X_train_imputed, y_train)
# print('Accuracy on Training set:',cross_val_score(rf_clf, X_train_imputed, y_train, cv=3, scoring='accuracy'))
# print('Accuracy on Test set:',cross_val_score(rf_clf, X_test_imputed, y_test, cv=3, scoring='accuracy'))`
# 
# *Accuracy on Training set: [0.86872348 0.86855077 0.8684654 ],
# Accuracy on Test set: [0.86878841 0.86829667 0.8684878 ]*

# ## <a id='15'> 15. Evaluating machine learning algorithms accuracy on training and testing sets </a>
# This perfromance is without tuning any hyperparameters and without any optimisation

# Model         | Train Accuracy | Test AUC
# ----------------------|-----------|---------------
#  Logistic Regression with Selected features  |    0.66  | 0.66
#  Random Forest with Selected features  |    0.86 |    0.86  
#  LightGBM with Selected features |    0.71 |    0.75
#  CATBoost with Selected features |    0.76 |    0.81
#  XGBoost with Selected features |    0.75 |    0.82
#  
#  
# 

# Since we get the best performance in Random Forest, let us try to enhance Random Forest by tuning hyperparameters

# ## <a id='16'> 16. Optimising selected machine learning model further by choosing best hyperparameters </a>

# Since we got best performance with Random Forest, let's try to optimise it further by using the correct parameters. We shall use GridSearchCV or RandomizedSearchCV to choose the best parameters.
# 
# **First choose the range of hyperparameters using RandomizedSearchCV.**
# 
# `from sklearn.model_selection import RandomizedSearchCV`
# 
# `#Hyperparameters`
# 
# `n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# max_features = ['auto', 'sqrt']
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# min_samples_split = [ 10, 25, 50, 100]
# min_samples_leaf = [1, 2, 4,10,20,30]
# bootstrap = [True, False]`
# 
# `#Create the random grid`
# 
# `random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}`
# 
# `from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(class_weight="balanced")
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# #Fit the random search model
# rf_random.fit(X_train_imputed, y_train)
# rf_random.best_params_`
# 
# 
# **Random_grid Best Parameters output**
# 
# `{
#     'bootstrap': [True],
#     'max_depth': [100],
#     'max_features': [2],
#     'min_samples_leaf': [ 4],
#     'min_samples_split': [10],
#     'n_estimators': [200]
# }`
# 
# **GridSearchCV**
# 
# After getting the best parameters from RandomSearchCV , we have understood the range for hyperparameters. For example the number of trees\estimators to be used should be in the range of 200, maximum features should be in range of 2 and so on.
# Now let us test which hyperparameter to use by using GridSearchCV. We will include the parameters in the range as found by random_grid output.
# 
# `from sklearn.model_selection import GridSearchCV`
# 
# `#Create the parameter grid based on the results of random search`
# 
# `param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100, 110],
#     'max_features': [2, 3],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 200, 300, 1000]
# }`
# 
# `#Create a RandomForets based model
# rf = RandomForestRegressor()`
# 
# `#Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 3, n_jobs = -1, verbose = 2)
# grid_search.best_params_`
# 
# **Output of grid search best parameters**
# 
# `max_depth= 20,
#  max_features= 'sqrt',
#  min_samples_leaf= 5,
#  min_samples_split= 40,
#  n_estimators= 200`
#  
# Fitting the parameters obtained by GridSearchCV on the RandomForest classification and rechecking the accuracy obtained on training and testing set.
# 
# `from sklearn.ensemble import RandomForestClassifier
# rf_clf_grid = RandomForestClassifier(random_state = 0, n_jobs=-1, class_weight="balanced",bootstrap= True,
#  max_depth= 20,
#  max_features= 'sqrt',
#  min_samples_leaf= 5,
#  min_samples_split= 40,
#  n_estimators= 200)
# rf_clf_grid.fit(X_train_imputed, y_train)`
# 
# `print('Accuracy on Training set:',cross_val_score(rf_clf_grid, X_train_imputed, y_train, cv=3, scoring='accuracy'))
# print('Accuracy on Test set:',cross_val_score(rf_clf_grid, X_test_imputed, y_test, cv=3, scoring='accuracy'))`
# 
# *Accuracy on Training set: [0.88676162 0.88575888 0.88496622]
# Accuracy on Test set: [0.89396156 0.89868787 0.89497561]*
# 
# 
# Note that the accuracy obtained via GridsearchCV and RandomizedSearchCV is more than the accuracy obtained without hyperparameters tuning.

# ## <a id='17'> 17. Final predictions </a>

# Fitting the parameters found out by using GridSearchCV and predicting the outputs by fitting the best machine learning model.
# 
# `rf_clf_grid.fit(X_train_imputed, y_train)
# predictions_grid=rf_clf_grid.predict(app_bur_prev_test_imputed)`
# 
# Saving the results in csv files 
# 
# `predictions_grid_df=pd.DataFrame(data={"SK_ID_CURR":app_bur_prev_test_imputed["SK_ID_CURR"],"TARGET":predictions_grid}) 
# predictions_grid_df['SK_ID_CURR'] = predictions_grid_df['SK_ID_CURR'].astype(int)
# predictions_grid_df.to_csv(path_or_buf="predictions_grid_df.csv",index=False)`

# In[ ]:




