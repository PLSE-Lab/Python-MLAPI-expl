#!/usr/bin/env python
# coding: utf-8

# - <a href='#1'>1. Problem Statement </a>  
# - <a href='#2'>2. Reading the data</a>
# - <a href='#3'>3. Feature Engineering</a>
#     - <a href='#3-1'>3.1 Creating new feature for bureau</a>
#     - <a href='#3-2'> 3.2 Function to count and normalize values of categorical variables </a>
# - <a href='#4'>4. Grouping the data</a>
# - <a href='#5'>5. Exploratory Data Analysis</a>
#        - <a href='#5-1'>5.1  Analyzing Target Variable</a>
#      - <a href='#5-2'>5.2  Visualizing basic info of the applicant </a>
#       - <a href='#5-3'>5.3 Client accompanied by ? </a>
# - <a href='#6'>6. Merging the data</a>     
# - <a href='#7'>7. Combining Training and Testing data</a>     
# - <a href='#7'>8. Feature Engineering Continued</a>     
#      -<a href='#8_1'>8.1. Deleting features </a>
#      -<a href='#8_2'>8.2  Handling Missing Values </a>
#       -<a href='#8_3'>8.3 Scaling Numerical Features </a>
#       -<a href='#8_3'>8.4 Converting into Categorical </a>
#    
# - <a href='#9'>9.Modelling</a>     

# > # <a id='1'>1. Problem Statement</a>

# Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.
# 
# **Evalutaion**  - Area under the ROC Curve
# 
# **Data  ** -   the problem has 7 files. 
# 
# * **application_train/application_test**: the main training and testing data with information about each loan application at Home Credit. Every loan has its own row and is identified by the feature SK_ID_CURR.  
# * **bureau **: All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample). Each previous credit has its own row in bureau, but one loan in the application data can have multiple previous credits.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import for plotting 
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# >  # <a id='2'>2. Reading the Data</a>

# In[ ]:


app_train = pd.read_csv('../input/application_train.csv')
app_test = pd.read_csv('../input/application_test.csv')
bureau = pd.read_csv('../input/bureau.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
pos_cash_balance = pd.read_csv('../input/POS_CASH_balance.csv')

previous_app = pd.read_csv('../input/previous_application.csv')
installments_payments = pd.read_csv('../input/installments_payments.csv')
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')


# In[ ]:


print(app_test.shape)


# > # <a id='3'>3. Feature Engineering</a>

# ## <a id='3-1'>3.1 Creating new feature for bureau</a>

# In[ ]:


# Groupby the client id (SK_ID_CURR), count the number of previous loans, and rename the column
previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'})
previous_loan_counts.head()


# ## <a id='3-2'>3.2 Function to count and normalize values of categorical variables </a>

# In[ ]:


def normalize_categorical(df, group_var, col_name):
    
    """Computes counts and normalized counts for each observation
    of `group_var` for each unique category in every categorical variable
    
    Parameters 
    ----------
    df - DataFrame for which we will calculate count
    
    group_var  = string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    col_name = string
            Variable added to the front of column names to keep track of columns
            
            """
    # select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))
    
    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]
    
    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])                                              
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (col_name, var, stat))
    
    categorical.columns = column_names
    
    return categorical
    


# In[ ]:


bureau_counts = normalize_categorical(bureau, group_var = 'SK_ID_CURR', col_name = 'bureau')
bureau_counts.head()


# > # <a id='4'>4 Grouping the data </a>

# In[ ]:


# Grouping data  so  that we can merge all the files in 1 dataset

data_bureau_agg=bureau.groupby(by='SK_ID_CURR').mean()
data_credit_card_balance_agg=credit_card_balance.groupby(by='SK_ID_CURR').mean()
data_previous_application_agg=previous_app.groupby(by='SK_ID_CURR').mean()
data_installments_payments_agg=installments_payments.groupby(by='SK_ID_CURR').mean()
data_POS_CASH_balance_agg=pos_cash_balance.groupby(by='SK_ID_CURR').mean()

data_bureau_agg.head()


# > # <a id='5'>5. Exploratory Data Exploration</a>

# ## <a id='5-2'>5.2  Visualizing basic info of the applicant </a>

# In[ ]:


# we will be plotting gender, occupation, has car, has flat  

plt.figure(1)
plt.subplot(221)
app_train['CODE_GENDER'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender')

plt.subplot(222)
app_train['FLAG_OWN_CAR'].value_counts(normalize=True).plot.bar(title= 'Own Car?')

plt.subplot(223)
app_train['CNT_CHILDREN'].value_counts(normalize=True).plot.bar(title= 'Count Children')

plt.subplot(224)
app_train['FLAG_OWN_REALTY'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Has Realty?')



plt.show()


# # Inference - 
# 1. We see that most of the applicants were female and without any children.
# 2. An interesting fact is that most of the applicants owned a realty but not a car. 
# 

# ## <a id='5-3'>5.3 Client accompanied by ? </a>

# In[ ]:


plt.figure(2)

plt.subplot(321)
app_train['NAME_TYPE_SUITE'].value_counts(normalize=True).plot.bar(figsize=(20,20), title= 'Accompanient')

plt.subplot(322)
app_train["NAME_CONTRACT_TYPE"].value_counts(normalize=True).plot.pie(figsize=(20,20), title='Loan Type')

plt.subplot(323)
app_train["NAME_FAMILY_STATUS"].value_counts(normalize=True).plot.pie(figsize=(20,20), title='Family status of applicants')

plt.subplot(324)
app_train["OCCUPATION_TYPE"].value_counts(normalize=True).plot.bar(figsize=(20,20), title='Occupation')
plt.show()


# ## <a id='5-3'>5.3  Loan is replayed or not? </a>

# In Progress

# > # <a id='6'>6. Merging the data</a>

# In[ ]:


def merge(df):
    df = df.join(data_bureau_agg, how='left', on='SK_ID_CURR', lsuffix='1', rsuffix='2') 
    df = df.join(bureau_counts, on = 'SK_ID_CURR', how = 'left')
    df = df.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')
    df = df.join(data_credit_card_balance_agg, how='left', on='SK_ID_CURR', lsuffix='1', rsuffix='2')    
    df = df.join(data_previous_application_agg, how='left', on='SK_ID_CURR', lsuffix='1', rsuffix='2')   
    df = df.join(data_installments_payments_agg, how='left', on='SK_ID_CURR', lsuffix='1', rsuffix='2') 
    
    return df

train = merge(app_train)
test = merge(app_test)
display(train.head())


# In[ ]:


print(train.shape)
print(test.shape)


# > # <a id='7'>7. Combining training and testing data</a>

# In[ ]:


#combining the data
ntrain = train.shape[0]
ntest = test.shape[0]

y_train = train.TARGET.values

#train_df = train_df.drop

all_data = pd.concat([train, test]).reset_index(drop=True)
all_data.drop(['TARGET'], axis=1, inplace=True)


# > # <a id='8'>8. Feature Engineering Continued</a>

# In[ ]:


# Now we will convert days employed and days registration and days id publish to a positive no. 
def correct_birth(df):
    
    df['DAYS_BIRTH'] = round((df['DAYS_BIRTH'] * (-1))/365)
    return df

def convert_abs(df):
    df['DAYS_EMPLOYED'] = abs(df['DAYS_EMPLOYED'])
    df['DAYS_REGISTRATION'] = abs(df['DAYS_REGISTRATION'])
    df['DAYS_ID_PUBLISH'] = abs(df['DAYS_ID_PUBLISH'])
    df['DAYS_LAST_PHONE_CHANGE'] = abs(df['DAYS_LAST_PHONE_CHANGE'])
    return df

# Now we will fill misisng values in OWN_CAR_AGE. 
#Most probably there will be missing values if the person does not own a car. So we will fill with 0

def missing(df):
    
    features = ['previous_loan_counts','NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_MEDI','OWN_CAR_AGE']
    
    for f in features:
        df[f] = df[f].fillna(0 )
    return df

def transform_app(df):
    df = correct_birth(df)
    df = convert_abs(df)
    df = missing(df)
    return df

   

all_data = transform_app(all_data)

    


# In[ ]:


# counting no of phones given by the company and delete the irrelevant features
all_data['NO_OF_CLIENT_PHONES'] = all_data['FLAG_MOBIL'] + all_data['FLAG_EMP_PHONE'] + all_data['FLAG_WORK_PHONE']
all_data.head()


# In[ ]:


# add a feature to determine if client's permanent city does not match with contact/work city
all_data['FLAG_CLIENT_OUTSIDE_CITY'] = np.where((all_data['REG_CITY_NOT_WORK_CITY']==1) & (all_data['REG_CITY_NOT_LIVE_CITY']==1),1,0)
all_data.head()


# In[ ]:


# add a feature to determine if client's permanent city does not match with contact/work region
all_data['FLAG_CLIENT_OUTSIDE_REGION'] = np.where((all_data['REG_REGION_NOT_LIVE_REGION']==1) & (all_data['REG_REGION_NOT_WORK_REGION']==1),1,0)
all_data.head()


#  ## <a id='8'>8.1. Deleting features</a>

# In[ ]:


# deleting useless features
def delete(df):
   # useless=['FLAG_MOBIL', 'FLAG_EMP_PHONE' ,'FLAG_WORK_PHONE','REG_CITY_NOT_WORK_CITY','REG_CITY_NOT_LIVE_CITY','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION']
    #for feature in useless:
     return df.drop(['FLAG_MOBIL', 'FLAG_EMP_PHONE' ,'FLAG_WORK_PHONE','REG_CITY_NOT_WORK_CITY','REG_CITY_NOT_LIVE_CITY','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION'], axis=1)
def transform(df):
   # df = convert_abs(df)
    df = delete(df)
   
    return df

all_data = transform(all_data)
all_data.head()


# In[ ]:


# delete Ids

def delete_id(df):
    return df.drop(['SK_ID_CURR', 'SK_ID_PREV','SK_ID_BUREAU'], axis = 1)

all_data = delete_id(all_data)


# In[ ]:


all_data.head()


# In[ ]:


print(all_data.columns)


# ## <a id='8-2'>8.2  Handling Missing Values </a>

# In[ ]:


# handling missing values

def miss_numerical(df):
    
    features = ['previous_loan_counts','NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_MEDI','OWN_CAR_AGE']
    numerical_features = all_data.select_dtypes(exclude = ["object"] ).columns
    #print(numerical_features)
    for f in numerical_features:
        #print(f)
        if f not in features:
            df[f] = df[f].fillna(df[f].median())
      
    return df

def miss_categorical(df):
    
    categorical_features = all_data.select_dtypes(include = ["object"]).columns
    
    for f in categorical_features:
        df[f] = df[f].fillna(df[f].mode()[0])
        
    return df

def transform_feature(df):
    df = miss_numerical(df)
    df = miss_categorical(df)
    #df = fill_cabin(df)
    return df

all_data = transform_feature(all_data)


all_data.head()
        


# ## <a id='8-3'>8.3 Scaling Numerical Features </a>

# In[ ]:


# Scaling the data 

from sklearn.preprocessing import MinMaxScaler

def encoder(df):
    scaler = MinMaxScaler()
    numerical = all_data.select_dtypes(exclude = ["object"]).columns
    features_transform = pd.DataFrame(data= df)
    features_transform[numerical] = scaler.fit_transform(df[numerical])
    display(features_transform.head(n = 5))
    return df

all_data = encoder(all_data)

#display(all_data.head())


# ## <a id='8-4'>8.4 Converting into categorical features </a>

# In[ ]:


# Converting into categorical features

# Create a label encoder object
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le_count = 0


# Iterate through the columns
for col in all_data:
    if all_data[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(all_data[col].unique())) <= 2:
            # Train on the training data
            le.fit(all_data[col])
            # Transform both training and testing data
            all_data[col] = le.transform(all_data[col])
            #test[col] = le.transform(test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
           
print('%d columns were label encoded.' % le_count)


# In[ ]:


# dummy variables
all_data = pd.get_dummies(all_data)

display(all_data.shape)


# > # <a id='9'>9. Modelling</a>

# In[ ]:


### Splitting features
train = all_data[:ntrain]
test = all_data[ntrain:]

print("Training shape", train.shape)
print("Testing shape", test.shape)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(train, y_train, test_size = 0.3, random_state = 200)
print("X Training shape", X_train.shape)
print("X Testing shape", X_test.shape)
print("Y Training shape", Y_train.shape)
print("Y Testing shape", Y_test.shape)


# In[ ]:


from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

logreg = LogisticRegression(random_state=0, class_weight='balanced', C=100)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict_proba(X_test)[:,1]

#Y_pred_proba = logreg.predict_proba(X_test)

print('Train/Test split results:')
#print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(Y_test, Y_pred))
print("ROC",  roc_auc_score(Y_test, Y_pred))
#print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))


# In[ ]:


pred_test = logreg.predict_proba(test)
#print("ROC",  roc_auc_score(Y_test, pred_test))
submission = pd.read_csv('../input/sample_submission.csv')

submission['SK_ID_CURR']=app_test['SK_ID_CURR']
print(len(app_test['SK_ID_CURR']))
submission['TARGET']=pred_test
#converting to csv
#print(submission['TARGET'])
pd.DataFrame(submission, columns=['SK_ID_CURR','TARGET'],index=None).to_csv('homecreditada.csv')

