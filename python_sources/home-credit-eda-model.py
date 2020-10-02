#!/usr/bin/env python
# coding: utf-8

# 

# - <a href='#1'>1. Introduction</a>  
# - <a href='#2'>2. Retrieving the Data</a>
# - <a href='#3'>3. Glimpse of Data</a>
# - <a href='#4'> 4. Check for missing data</a>
# - <a href='#5'>5. Data Exploration</a>
#     - <a href='#5-1'>5.1 Distribution of AMT_CREDIT</a>
#     - <a href='#5-2'>5.2 Distribution of AMT_INCOME_TOTAL</a>
#     - <a href='#5-3'>5.3 Distribution of AMT_GOODS_PRICE</a>
#     <a href='#6'>6. Pearson Correlation of features</a>
# - <a href='#7'>7. Feature Importance using Random forest</a>

# # <a id='1'>1. Introduction</a>

# Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.
# 
# While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

#  # <a id='2'>2. Retrieving the Data</a>

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
#import plotly.graph_objs.layout.YAxis as goly
#import plotly.graph_objs.layout.scene.YAxis as golsy
import plotly.offline as offline
offline.init_notebook_mode()
# from plotly import tools
# import plotly.tools as tls
# import squarify
# from mpl_toolkits.basemap import Basemap
# from numpy import array
# from matplotlib import cm
	
from  sklearn.utils import resample
from sklearn import preprocessing 

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

#code to run the xgboost model
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# from sklearn import preprocessing
# # Supress unnecessary warnings so that presentation looks clean
# import warnings
# warnings.filterwarnings("ignore")

# # Print all rows and columns
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

from sklearn.preprocessing import Imputer
my_imputer = Imputer()
#data_with_imputed_values = my_imputer.fit_transform(original_data)


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


application_train = pd.read_csv('../input/application_train.csv')  # creating a data frame named application_train using application_train.csv
#POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
#bureau_balance = pd.read_csv('../input/bureau_balance.csv')
#previous_application = pd.read_csv('../input/previous_application.csv')
#installments_payments = pd.read_csv('../input/installments_payments.csv')
#credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
bureau = pd.read_csv('../input/bureau.csv')
application_test = pd.read_csv('../input/application_test.csv')


# In[ ]:



# Downsample majority class
train_downsampled = resample(application_train, 
                                 replace=False,    # sample without replacement
                                 n_samples=25000,     # to match minority class
                                 random_state=123) # reproducible results


# In[ ]:


train_downsampled.shape


# In[ ]:


train_downsampled.iloc[1]


# In[ ]:


#application_train.csv table
["NAME_CONTRACT_TYPE",
"CODE_GENDER",
"FLAG_OWN_CAR",
"FLAG_OWN_REALTY",
"CNT_CHILDREN",
"AMT_INCOME_TOTAL",
"AMT_CREDIT",
"AMT_ANNUITY",
"AMT_GOODS_PRICE",
"NAME_TYPE_SUITE",
"NAME_INCOME_TYPE",
"NAME_EDUCATION_TYPE",
"NAME_FAMILY_STATUS",
"NAME_HOUSING_TYPE",
"REGION_POPULATION_RELATIVE",
"DAYS_BIRTH",
"DAYS_EMPLOYED",
"OWN_CAR_AGE",
"OCCUPATION_TYPE",
"CNT_FAM_MEMBERS",
"REGION_RATING_CLIENT_W_CITY",
"ORGANIZATION_TYPE",
"EXT_SOURCE_1",
"EXT_SOURCE_2",
"EXT_SOURCE_3",
"YEARS_BUILD_AVG",
"FLAG_DOCUMENT_2",
"FLAG_DOCUMENT_3",
"FLAG_DOCUMENT_4",
"FLAG_DOCUMENT_5",
"FLAG_DOCUMENT_6",
"FLAG_DOCUMENT_7",
"FLAG_DOCUMENT_8",
"FLAG_DOCUMENT_9",
"FLAG_DOCUMENT_10",
"FLAG_DOCUMENT_11",
"FLAG_DOCUMENT_12",
"FLAG_DOCUMENT_13",
"FLAG_DOCUMENT_14",
"FLAG_DOCUMENT_15",
"FLAG_DOCUMENT_16",
"FLAG_DOCUMENT_17",
"FLAG_DOCUMENT_18",
"FLAG_DOCUMENT_19",
"FLAG_DOCUMENT_20",
"FLAG_DOCUMENT_21",
"AMT_REQ_CREDIT_BUREAU_HOUR",
"AMT_REQ_CREDIT_BUREAU_MON",
"AMT_REQ_CREDIT_BUREAU_YEAR"]
#bureau.csv table below
SK_ID_CURR
SK_BUREAU_ID
CREDIT_ACTIVE
DAYS_CREDIT
CREDIT_DAY_OVERDUE
AMT_CREDIT_MAX_OVERDUE
CNT_CREDIT_PROLONG
AMT_CREDIT_SUM
AMT_CREDIT_SUM_DEBT
AMT_CREDIT_SUM_LIMIT
AMT_CREDIT_SUM_OVERDUE
CREDIT_TYPE
AMT_ANNUITY


# In[ ]:


train = pd.merge(train_downsampled, bureau, how='left', on='SK_ID_CURR')
#train = pd.merge(train_b, bureau_balance, how='left', on= 'SK_ID_BUREAU')
#train = train_b.drop('SK_ID_BUREAU', axis=1)


# In[ ]:


test = pd.merge(application_test, bureau, how='left', on='SK_ID_CURR')
#test = pd.merge(test_b, bureau_balance, how='left', on= 'SK_ID_BUREAU')
#test = test_b.drop('SK_ID_BUREAU', axis=1)


# In[ ]:


print (train_downsampled.shape)
print (bureau.shape)
print (train_b.shape)
print (bureau_balance.shape)
print (train.shape)


# In[ ]:


print (bureau.columns)


# In[ ]:


train.iloc[2]


# In[ ]:


lb = preprocessing.LabelEncoder()
lb.fit((list (bureau.columns)+list (bureau_balance.columns)))
print (lb.transform(bureau.columns))
print (lb.transform(bureau_balance.columns.values.astype('str')))


# In[ ]:


categorical_featrs = [
    f for f in application_train.columns if application_train[f].dtype == 'object'
]

for col in categorical_featrs:
    lb = preprocessing.LabelEncoder()
    lb.fit(list(application_train[col].values.astype('str')) + list(application_test[col].values.astype('str')))
    application_train[col] = lb.transform(list(application_train[col].values.astype('str')))
    application_test[col] = lb.transform(list(application_test[col].values.astype('str')))


# In[ ]:


categorical_featrs = [
    k for k in train.columns if train[k].dtype == 'object'
]
for col in categorical_featrs:
    lb = preprocessing.LabelEncoder()
    lb.fit(list(train[col].values.astype('str')) ) 
    train[col] = lb.transform(list(train[col].values.astype('str')))
   


# In[ ]:


categorical_featrs = [
    k for k in test.columns if test[k].dtype == 'object'
]
for col in categorical_featrs:
    lb = preprocessing.LabelEncoder()
    lb.fit(list(test[col].values.astype('str')) ) 
    test[col] = lb.transform(list(test[col].values.astype('str')))


# In[ ]:


train.fillna(-999, inplace = True)


# In[ ]:


list(train)


# In[ ]:


X= train.drop(["TARGET"], axis=1)
y= train["TARGET"]

test_size = 0.30
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    random_state=0)


# In[ ]:



# fit model no training data
model = XGBClassifier()
#x_train= application_train[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]]
#x_test= application_test[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]]
#y_train= application_train["TARGET"]
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


from sklearn import metrics
from ggplot import *
from sklearn.metrics import roc_curve, roc_auc_score

preds = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test, preds)

rs=roc_auc_score(y_test, preds)
print ("Ther ROC is ",rs  ) 

df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
ggplot(df, aes(x='fpr', y='tpr')) + geom_line() + geom_abline(linetype='dashed') +    xlim(-0.01,1) +    ylim(0,1.01) +    xlab("False Positive Rate (FPR) ")+    ylab("True Positive Rate (TPR) ")


# In[ ]:


#preds_see = model.predict_proba(X_test)
preds = model.predict_proba(test)[:,1]
# We will look at the predicted prices to ensure we have something sensible.
#print(preds_see)
#print(preds)


# In[ ]:


my_submission = pd.DataFrame({'SK_ID_CURR': test.SK_ID_CURR, 'TARGET': preds})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# In[ ]:


#submission_csv = pd.read_csv('submission.csv')
#print (submission_csv)


# In[ ]:


from sklearn.preprocessing import Imputer
my_imputer = Imputer()
X_test_imp = my_imputer.fit_transform(X_test)
X_train_imp= my_imputer.fit_transform(X_train)
#y_train_imp= my_imputer.fit_transform(y_train)


# In[ ]:


print('Size of application_train data', application_train.shape)
print('Size of POS_CASH_balance data', POS_CASH_balance.shape)
print('Size of bureau_balance data', bureau_balance.shape)
print('Size of previous_application data', previous_application.shape)
print('Size of installments_payments data', installments_payments.shape)
print('Size of credit_card_balance data', credit_card_balance.shape)
print('Size of bureau data', bureau.shape)


# # <a id='3'>3. Glimpse of Data</a>

# **application_train data**

# In[ ]:


application_train.columns.values


# **bureau_balance data**

# In[ ]:


bureau_balance.head()


# **previous_application data**

# In[ ]:


previous_application.head()


# In[ ]:


previous_application.columns.values


# **installments_payments data**

# In[ ]:


installments_payments.head()


# **credit_card_balance data**

# In[ ]:


credit_card_balance.head()


# In[ ]:


credit_card_balance.columns.values


# **bureau data**

# In[ ]:


bureau.head()


# # <a id='4'> 4 Check for missing data</a>

# **checking missing data in application_train **

# In[ ]:


# checking missing data
total = application_train.isnull().sum().sort_values(ascending = False)
percent = (application_train.isnull().sum()/application_train.isnull().count()*100).sort_values(ascending = False)
missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_application_train_data[total>0] 


# **checking missing data in POS_CASH_balance **

# In[ ]:


# checking missing data
total = POS_CASH_balance.isnull().sum().sort_values(ascending = False)
percent = (POS_CASH_balance.isnull().sum()/POS_CASH_balance.isnull().count()*100).sort_values(ascending = False)
missing_POS_CASH_balance_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_POS_CASH_balance_data[total>0]


# **checking missing data in bureau_balance **

# In[ ]:


# checking missing data
total = bureau_balance.isnull().sum().sort_values(ascending = False)
percent = (bureau_balance.isnull().sum()/bureau_balance.isnull().count()*100).sort_values(ascending = False)
missing_bureau_balance_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_bureau_balance_data


# **checking missing data in previous_application **

# In[ ]:


# checking missing data
total = previous_application.isnull().sum().sort_values(ascending = False)
percent = (previous_application.isnull().sum()/previous_application.isnull().count()*100).sort_values(ascending = False)
missing_previous_application_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_previous_application_data[total>0]


# **checking missing data in installments_payments **

# In[ ]:


# checking missing data
total = installments_payments.isnull().sum().sort_values(ascending = False)
percent = (installments_payments.isnull().sum()/installments_payments.isnull().count()*100).sort_values(ascending = False)
missing_installments_payments_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_installments_payments_data.head(3)


# **checking missing data in credit_card_balance **

# In[ ]:


# checking missing data
total = credit_card_balance.isnull().sum().sort_values(ascending = False)
percent = (credit_card_balance.isnull().sum()/credit_card_balance.isnull().count()*100).sort_values(ascending = False)
missing_credit_card_balance_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_credit_card_balance_data.head(10)


# **checking missing data in bureau **

# In[ ]:


# checking missing data
total = bureau.isnull().sum().sort_values(ascending = False)
percent = (bureau.isnull().sum()/bureau.isnull().count()*100).sort_values(ascending = False)
missing_bureau_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_bureau_data.head(8)


# # <a id='5'>5. Data Exploration</a>

# ## <a id='5-1'>5.1 Distribution of AMT_CREDIT</a>

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_CREDIT")
ax = sns.distplot(application_train["AMT_CREDIT"])


# ## <a id='5-2'>5.2 Distribution of AMT_INCOME_TOTAL</a>

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_INCOME_TOTAL")
ax = sns.distplot(application_train["AMT_INCOME_TOTAL"].dropna())


# ## <a id='5-3'>5.3 Distribution of AMT_GOODS_PRICE</a>

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_GOODS_PRICE")
ax = sns.distplot(application_train["AMT_GOODS_PRICE"].dropna())


# ## <a id='5-4'>5.4 Who accompanied client when applying for the  application</a>

# In[ ]:


temp = application_train["NAME_TYPE_SUITE"].value_counts()
#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Who accompanied client when applying for the  application in % ",
    xaxis=dict(
        title='Name of type of the Suite',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of Name of type of the Suite in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')


# ## <a id='5-5'>5.5 Data is balanced or imbalanced</a>

# In[ ]:


temp = application_train["TARGET"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Loan Repayed or not')


# * As we can see data is highly imbalanced.

# ## <a id='5-6'>5.6 Types of loan</a>

# * **Rovolving loans :**  Arrangement which allows for the loan amount to be withdrawn, repaid, and redrawn again in any manner and any number of times, until the arrangement expires. Credit card loans and overdrafts are revolving loans. Also called evergreen loan

# In[ ]:


temp = application_train["NAME_CONTRACT_TYPE"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      #"name": "Types of Loans",
      #"hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Types of loan",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Loan Types",
                "x": 0.17,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')


# * Most of the loans are Cash loans which were taken by applicants. **90.5 %** loans are Cash loans.

# ## <a id='5-7'>5.7 Purpose of loan</a>

# In[ ]:


temp1 = application_train["FLAG_OWN_CAR"].value_counts()
temp2 = application_train["FLAG_OWN_REALTY"].value_counts()

fig = {
  "data": [
    {
      "values": temp1.values,
      "labels": temp1.index,
      "domain": {"x": [0, .48]},
      "name": "Own Car",
      "hoverinfo":"label+percent+name",
      "hole": .6,
      "type": "pie"
    },
    {
      "values": temp2.values,
      "labels": temp2.index,
      "text":"Own Realty",
      "textposition":"inside",
      "domain": {"x": [.52, 1]},
      "name": "Own Reality",
      "hoverinfo":"label+percent+name",
      "hole": .6,
      "type": "pie"
    }],
  "layout": {
        "title":"Purpose of loan",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Own Car",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Own Realty",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')


# ## <a id='5-8'>5.8 Income sources of Applicant's who applied for loan</a>

# In[ ]:


temp = application_train["NAME_INCOME_TYPE"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Income sources of Applicant\'s', hole = 0.5)


# * 51.6 % Applicants mentioned that they are working.  23.3 % are Commercial Associate and 18 % are Pensioner etc. 

# ## <a id='5-9'>5.9 Family Status of Applicant's who applied for loan</a>

# In[ ]:


temp = application_train["NAME_FAMILY_STATUS"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Family Status of Applicant\'s', hole = 0.5)


# * 63.9 % applicants are married. 14.8 % are single etc.

# ## <a id='5-10'>5.10 Occupation of Applicant's who applied for loan</a>

# In[ ]:


temp = application_train["OCCUPATION_TYPE"].value_counts()
temp.iplot(kind='bar', xTitle = 'Occupation', yTitle = "Count", title = 'Occupation of Applicant\'s who applied for loan', color = 'green')


# * **Top Applicant's who applied for loan :**
#   * Laborers - Apprx. 55 K
#   * Sales Staff - Approx. 32 K
#   * Core staff - Approx. 28 K
#   * Managers - Approx. 21 K
#   * Drivers - Approx. 19 K

# ## <a id='5-11'>5.11 Education of Applicant's who applied for loan</a>

# In[ ]:


temp = application_train["NAME_EDUCATION_TYPE"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Education of Applicant\'s', hole = 0.5)


# * 71 % applicants have secondary and 24.3 % having higher education.

# ## <a id='5-12'>5.12 For which types of house higher applicant's applied for loan ?</a>

# In[ ]:


temp = application_train["NAME_HOUSING_TYPE"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Type of House', hole = 0.5)


# * Approx. 89 % peoples applied for loan, they mentioned type of house is **House / Appartment**.

# ## <a id='5-13'>5.13 Types of Organizations who applied for loan </a>

# In[ ]:


temp = application_train["ORGANIZATION_TYPE"].value_counts()
temp.iplot(kind='bar', xTitle = 'Organization Name', yTitle = "Count", title = 'Types of Organizations who applied for loan ', color = 'red')


# * **Types of Organizations who applied for loan :**
#   * Business Entity Type 3 - Approx. 68 K
#   * XNA - Approx. 55 K
#   * Self employed - Approx. 38 K
#   * Others - Approx. 17 K
#   * Medicine - Approx. 11 K
#  

# ## <a id='5-14'>5.14 Exploration in terms of loan is repayed or not</a>

# ## <a id='5-14-1'>5.14.1 Income sources of Applicant's in terms of loan is repayed or not in %</a>

# In[ ]:


temp = application_train["NAME_INCOME_TYPE"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(application_train["TARGET"][application_train["NAME_INCOME_TYPE"]==val] == 1))
    temp_y0.append(np.sum(application_train["TARGET"][application_train["NAME_INCOME_TYPE"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='YES'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='NO'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Income sources of Applicant's in terms of loan is repayed or not  in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='Income source',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ## <a id='5-14-5'>5.14.5 For which types of house higher applicant's applied for loan in terms of loan is repayed or not in %</a>

# In[ ]:


temp = application_train["NAME_HOUSING_TYPE"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(application_train["TARGET"][application_train["NAME_HOUSING_TYPE"]==val] == 1))
    temp_y0.append(np.sum(application_train["TARGET"][application_train["NAME_HOUSING_TYPE"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='YES'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='NO'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "For which types of house higher applicant's applied for loan in terms of loan is repayed or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='types of house',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ## <a id='5-14-6'>5.14.6 Types of Organizations in terms of loan is repayed or not in %</a>

# In[ ]:


temp = application_train["ORGANIZATION_TYPE"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(application_train["TARGET"][application_train["ORGANIZATION_TYPE"]==val] == 1))
    temp_y0.append(np.sum(application_train["TARGET"][application_train["ORGANIZATION_TYPE"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='YES'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='NO'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Types of Organizations in terms of loan is repayed or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='Types of Organizations',
        tickfont=dict(
            size=10,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# # <a id='5-15'>5.15 Exploartion of previous application data</a>

# ## <a id='5-15-1'>5.15.1 Contract product type of previous application</a>

# In[ ]:


temp = previous_application["NAME_CONTRACT_TYPE"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      #"name": "Types of Loans",
      #"hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Contract product type of previous application",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Contract product type",
                "x": 0.12,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')


# * **Contract product type of previous application :**
#   * Cash loans - 44.8 %
#   * Consumer loans - 43.7 %
#   * Rovolving loan - 11.6 %
#   * XNA - 0.0207 %

# ## <a id='5-15-3'>5.15.3 Purpose of cash loan in previous application</a>

# In[ ]:


temp = previous_application["NAME_CASH_LOAN_PURPOSE"].value_counts()
#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
    marker=dict(
        color=(temp / temp.sum())*100,
        colorscale = 'Blues',
        reversescale = True
    ),
)
data = [trace]
layout = go.Layout(
    title = "Purpose of cash loan in previous application in % ",
    xaxis=dict(
        title='Purpose of cash loan',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# * **Main purpose of the cash loan was  :**
#   * XAP - 55 %
#   * XNA - 41 %

# ## <a id='5-15-7'>5.15.7 Who accompanied client when applying for the previous application</a>

# In[ ]:


temp = previous_application["NAME_TYPE_SUITE"].value_counts()
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
    marker=dict(
        color=(temp / temp.sum())*100,
        colorscale = '#ea7c96',
        #reversescale = True
    ),
)
data = [trace]
layout = go.Layout(
    title = "Who accompanied client when applying for the previous application in % ",
    xaxis=dict(
        title='Name of type of the Suite',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of Name of type of the Suite in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# * **Who accompanied client when applying for the previous application :**
#   * Unaccompanied : Approx. 60 % times
#   * Family : Approx. 25 % times
#   * Spouse, Partner : Approx. 8 %
#   * Childrens : Approx. 4 %

# ## <a id='5-15-8'>5.15.8 Was the client old or new client when applying for the previous application</a>

# In[ ]:


temp = previous_application["NAME_CLIENT_TYPE"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Was the client old or new client when applying for the previous application', hole = 0.7,colors=['#ea7c96','#75e575',])


# * Approximately 74 % was repeater clients who applied for previous application.

# ## <a id='5-15-9'>5.15.9 What kind of goods did the client apply for in the previous application</a>

# In[ ]:


temp = previous_application["NAME_GOODS_CATEGORY"].value_counts()
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
    marker=dict(
        color=(temp / temp.sum())*100,
        colorscale = 'Greens',
        reversescale = True
    ),
)
data = [trace]
layout = go.Layout(
    title = "What kind of goods did the client apply for in the previous application in % ",
    xaxis=dict(
        title='Name of the goods',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ## <a id='6'>6 Pearson Correlation of features</a>

# In[ ]:


data = [
    go.Heatmap(
        z= application_train.corr().values,
        x=application_train.columns.values,
        y=application_train.columns.values,
        colorscale='Viridis',
        reversescale = False,
        text = True ,
        opacity = 1.0 )
]

layout = go.Layout(
    title='Pearson Correlation of features',
    xaxis = dict(ticks='', nticks=36),
    yaxis = dict(ticks='' ),
    width = 900, height = 700,
margin=dict(
    l=240,
),)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')

