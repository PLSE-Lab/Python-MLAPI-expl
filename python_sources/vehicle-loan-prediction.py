#!/usr/bin/env python
# coding: utf-8

# ## Work in Progress....Any suggestions are welcome

# ## Introduction
# 
# Financial institutions incur significant losses due to the default of vehicle loans. This has led to the tightening up of vehicle loan underwriting and increased vehicle loan rejection rates. The need for a better credit risk scoring model is also raised by these institutions. This warrants a study to estimate the determinants of vehicle loan default. A financial institution has hired you to accurately predict the probability of loanee/borrower defaulting on a vehicle loan in the first EMI (Equated Monthly Instalments) on the due date. Following Information regarding the loan and loanee are provided:
# 
# * Loanee Information (Demographic data like age, income, Identity proof etc.)
# * Loan Information (Disbursal details, amount, EMI, loan to value ratio etc.)
# * Bureau data & history (Bureau score, number of active accounts, the status of other loans, credit history etc.)
# 
# Doing so will ensure that clients capable of repayment are not rejected and important determinants can be identified which can be further used for minimising the default rates.

# ## Data Dictionary
# ...

# ## Importing the required packages

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import datetime
import seaborn as sns


# ## Importing the data

# In[2]:


print(os.listdir("../input"))
Train_df= pd.read_csv('../input/train.csv')
Test_df= pd.read_csv('../input/test.csv')


# ## Dimensions of the data

# In[3]:


print(Train_df.shape)
print(Test_df.shape)


# ## Appending the Train and Test data

# In[4]:


Train_df['Train_Test']='Train'
Test_df['Train_Test']='Test'
df= Train_df.append(Test_df)


# ## Glimpse of the data

# In[5]:


df.head()


# ## Descriptive Stats

# In[6]:


df.describe()


# ## Missing value check

# In[7]:


missing_var= df.isnull().sum()
print(missing_var)


# In[8]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# We have missing values only in the variable "EMPLOYMENT_TYPE" which we will impute it using the backward fill method

# In[9]:


df['EMPLOYMENT_TYPE'] = df['EMPLOYMENT_TYPE'].fillna(method = 'bfill')


# ## Structure of the data

# In[10]:


df.info()


# We need to convert the data type of the categorical variables into category 

# In[11]:


for col in ['UNIQUEID','AADHAR_FLAG', 'BRANCH_ID', 'CURRENT_PINCODE_ID', 'DRIVING_FLAG',
            'EMPLOYEE_CODE_ID', 'LOAN_DEFAULT', 'MANUFACTURER_ID' , 
            'MOBILENO_AVL_FLAG', 'STATE_ID', 'SUPPLIER_ID', 'VOTERID_FLAG',
            'PAN_FLAG', 'PASSPORT_FLAG', 'PERFORM_CNS_SCORE_DESCRIPTION',
            'EMPLOYMENT_TYPE']:
    df[col] = df[col].astype('category')


# Similarly, we need to convert the date time variables into its respective data type.

# In[12]:


df['DATE_OF_BIRTH'] = pd.to_datetime(df['DATE_OF_BIRTH'])
df['DISBURSAL_DATE'] = pd.to_datetime(df['DISBURSAL_DATE'])


# ## Analysis of Categorical Variables

# We will begin by checking the number of unique categories in the categorical variables.

# In[13]:


df.select_dtypes(include=['category']).nunique()


# We will proceed to drop those variables that have too many categories. These will only add noise to the model.

# In[14]:


df=df.drop(['BRANCH_ID','CURRENT_PINCODE_ID','EMPLOYEE_CODE_ID','SUPPLIER_ID'],axis=1)


# There are multiple categories in the variable 'PERFORM_CNS_SCORE_DESCRIPTION' which are similar and can be clubbed.

# In[15]:


df.PERFORM_CNS_SCORE_DESCRIPTION.value_counts()


# In[16]:


df = df.replace({'PERFORM_CNS_SCORE_DESCRIPTION':{'C-Very Low Risk':'Low', 'A-Very Low Risk':'Low',
                                                       'B-Very Low Risk':'Low', 'D-Very Low Risk':'Low',
                                                       'F-Low Risk':'Low', 'E-Low Risk':'Low', 'G-Low Risk':'Low',
                                                       'H-Medium Risk': 'Medium', 'I-Medium Risk': 'Medium',
                                                       'J-High Risk':'High', 'K-High Risk':'High','L-Very High Risk':'Very High',
                                                       'M-Very High Risk':'Very High','Not Scored: More than 50 active Accounts found':'Not Scored',
                                                       'Not Scored: Only a Guarantor':'Not Scored','Not Scored: Not Enough Info available on the customer':'Not Scored',
                                                        'Not Scored: No Activity seen on the customer (Inactive)':'Not Scored','Not Scored: No Updates available in last 36 months':'Not Scored',
                                                       'Not Scored: Sufficient History Not Available':'Not Scored', 'No Bureau History Available':'Not Scored'
                                                       }})
df.PERFORM_CNS_SCORE_DESCRIPTION.value_counts()


# ## Analysis of Date Time Variables

# We will convert the variables 'AVERAGE_ACCT_AGE' and 'CREDIT_HISTORY_LENGTH' in number of months from its present format.

# In[17]:


df['AVERAGE_ACCT_AGE']=df['AVERAGE_ACCT_AGE'].map(lambda x : re.sub("[^0-9]+"," ",x))
df['AVERAGE_ACCT_AGE']=df['AVERAGE_ACCT_AGE'].str.split(" ",expand=True)[0].astype(int)*12+df['AVERAGE_ACCT_AGE'].str.split(" ",expand=True)[1].astype(int)

df['CREDIT_HISTORY_LENGTH']=df['CREDIT_HISTORY_LENGTH'].map(lambda x : re.sub("[^0-9]+"," ",x))
df['CREDIT_HISTORY_LENGTH']=df['CREDIT_HISTORY_LENGTH'].str.split(" ",expand=True)[0].astype(int)*12+df['CREDIT_HISTORY_LENGTH'].str.split(" ",expand=True)[1].astype(int)


# Next we will use the variables 'DATE_OF_BIRTH' and 'DISBURSAL_DATE' to calculate the age of the customer and the age of the loan respectively.

# In[18]:


df['TODAY']=pd.to_datetime(datetime.date.today())

df['AGE']=df['TODAY']-df['DATE_OF_BIRTH']
new = df["AGE"].astype(str).str.split(" ", n = 1, expand = True)
df['AGE']=new[0].astype(int)

df['LOAN_AGE']=df['TODAY']-df['DISBURSAL_DATE']
new = df["LOAN_AGE"].astype(str).str.split(" ", n = 1, expand = True)
df['LOAN_AGE']=new[0].astype(int)


# ## Correlation

# In[19]:


print(df.corr())
sns.heatmap(df.corr(), cmap='BuGn')


# ## Variance Infaltion Factor

# In[20]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
df_Numeric=pd.DataFrame(df[['ASSET_COST',
'AVERAGE_ACCT_AGE',
'CREDIT_HISTORY_LENGTH',
'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS',
'DISBURSED_AMOUNT','LTV','NEW_ACCTS_IN_LAST_SIX_MONTHS',
'NO_OF_INQUIRIES','PERFORM_CNS_SCORE',
'PRIMARY_INSTAL_AMT',
'PRI_ACTIVE_ACCTS',
'PRI_CURRENT_BALANCE',
'PRI_DISBURSED_AMOUNT',
'PRI_NO_OF_ACCTS',
'PRI_OVERDUE_ACCTS',
'PRI_SANCTIONED_AMOUNT',
'SEC_ACTIVE_ACCTS',
'SEC_CURRENT_BALANCE',
'SEC_DISBURSED_AMOUNT',
'SEC_INSTAL_AMT',
'SEC_NO_OF_ACCTS',
'SEC_OVERDUE_ACCTS',
'SEC_SANCTIONED_AMOUNT',
'AGE',
'LOAN_AGE']])
X = df_Numeric.assign(const=0)
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)


# ## Event Imbalance Check

# In[21]:


Event_Imbalance = df.groupby('LOAN_DEFAULT').agg({'LOAN_DEFAULT':'count'})
Event_Imbalance['PERCENTAGE_CONTRIBUTION']=Event_Imbalance['LOAN_DEFAULT']/sum(Event_Imbalance['LOAN_DEFAULT'])
print(Event_Imbalance)
df.LOAN_DEFAULT.value_counts().plot(kind='bar')


# The event and non-event ratio is 78% to 22% and hence its absolutely fine.

# In[22]:


Predictors= df[['AADHAR_FLAG','DRIVING_FLAG','MOBILENO_AVL_FLAG','STATE_ID',
                'VOTERID_FLAG','PAN_FLAG','PASSPORT_FLAG', 'PERFORM_CNS_SCORE_DESCRIPTION',
                'EMPLOYMENT_TYPE']]

One_hot_encoded_training_predictors = pd.get_dummies(Predictors)

df=pd.merge(df[['ASSET_COST','AVERAGE_ACCT_AGE',
'CREDIT_HISTORY_LENGTH','DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS',
'DISBURSED_AMOUNT','LTV','NEW_ACCTS_IN_LAST_SIX_MONTHS','NO_OF_INQUIRIES','PERFORM_CNS_SCORE',
'PRIMARY_INSTAL_AMT','PRI_ACTIVE_ACCTS','PRI_CURRENT_BALANCE','PRI_DISBURSED_AMOUNT',
'PRI_NO_OF_ACCTS','PRI_OVERDUE_ACCTS','PRI_SANCTIONED_AMOUNT','SEC_ACTIVE_ACCTS',
'SEC_CURRENT_BALANCE','SEC_DISBURSED_AMOUNT','SEC_INSTAL_AMT','SEC_NO_OF_ACCTS',
'SEC_OVERDUE_ACCTS','SEC_SANCTIONED_AMOUNT','AGE','LOAN_AGE','LOAN_DEFAULT','Train_Test']],
One_hot_encoded_training_predictors,left_index=True, right_index=True)


# ## Splitting into Train and Test data

# In[23]:


Train_df= df[df.Train_Test=='Train']
Train_df=Train_df.drop(['Train_Test'],axis=1)
Test_df= df[df.Train_Test=='Test']
Test_df=Test_df.drop(['Train_Test'],axis=1)


# In[24]:


from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(Train_df.drop('LOAN_DEFAULT',axis=1),Train_df['LOAN_DEFAULT'],
                                                    test_size=0.30,random_state=101)


# ## Logistic Regression

# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression()
logreg.fit(X_Train, Y_Train)


# In[30]:


import statsmodels.api as sm
logit_model=sm.Logit(Y_Train,X_Train)
result=logit_model.fit()
result.summary()


# ## Confusion Matrix

# In[27]:



from sklearn.metrics import classification_report
predictions = logreg.predict(X_Train)
print(classification_report(Y_Train,predictions))
print("Accuracy:",metrics.accuracy_score(Y_Train, predictions))


# In[28]:


from sklearn.metrics import confusion_matrix
predictions = logreg.predict(X_Train)
print(confusion_matrix (Y_Train, predictions))

