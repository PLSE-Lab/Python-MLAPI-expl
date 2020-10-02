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

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy
from scipy.stats import norm 
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()


# In[ ]:


application_train=pd.read_csv(r"../input/application_train.csv")
application_test=pd.read_csv(r"../input/application_test.csv")
bureau_balance=pd.read_csv(r"../input/bureau_balance.csv")
bureau=pd.read_csv(r"../input/bureau.csv")
credit_card_balance=pd.read_csv(r"../input/credit_card_balance.csv")
POS_cash=pd.read_csv(r"../input/POS_CASH_balance.csv")
bureau=pd.read_csv(r"../input/bureau.csv")
previous_application=pd.read_csv(r"../input/previous_application.csv")
install_payment=pd.read_csv(r"../input/installments_payments.csv")


# ||||||||Checking the head of each dataset 

# In[ ]:


application_train.head()


# In[ ]:


application_train.columns.values


# In[ ]:


application_test.head()


# In[ ]:


application_test.columns.values


# In[ ]:


bureau.head()


# In[ ]:


bureau.info()


# In[ ]:


credit_card_balance.head()


# |||||**Now we to have to check whether is there any missing value or not 
# if there is  missing  value then we can find it by how much total and what is the percentage of missing value in that particular columns**

# In[ ]:


application_train.shape


# In[ ]:


application_train.describe()


# In[ ]:


#finding missing value of train dataset
total_null=application_train.isnull().sum().sort_values(ascending=False)
percentage=(application_train.isnull().sum()/application_train.isnull().count() *100).sort_values(ascending=False)
missing_train_data=pd.concat([total_null,percentage],axis=1,keys=["Total_null","Percentage"])


# In[ ]:


missing_train_data.head()


# In[ ]:


POS_cash.head()


# In[ ]:


POS_cash.shape


# In[ ]:


#finding missing value of POS_cash dataset
total_null=POS_cash.isnull().sum().sort_values(ascending=False)
percentage=(POS_cash.isnull().sum()/POS_cash.isnull().count() *100).sort_values(ascending=False)
missing_POS_data=pd.concat([total_null,percentage],axis=1,keys=["Total_null","Percentage"])


# In[ ]:


missing_POS_data


# In[ ]:


#finding missing value of bureau dataset
total_null=bureau.isnull().sum().sort_values(ascending=False)
percentage=(bureau.isnull().sum()/bureau.isnull().count() *100).sort_values(ascending=False)
missing_bureau_data=pd.concat([total_null,percentage],axis=1,keys=["Total_null","Percentage"])
missing_bureau_data.head(15)


# In[ ]:


#missing value of bureau_balance dataset
total_null=bureau_balance.isnull().sum().sort_values(ascending=False)
percentage=(bureau_balance.isnull().sum()/bureau_balance.isnull().count() *100).sort_values(ascending=False)
missing_bureau_balance=pd.concat([total_null,percentage],axis=1,keys=["Total_null","Percentage"])
missing_bureau_balance.head(15)


# In[ ]:


#missing value of previous_application dataset
total_null=previous_application.isnull().sum().sort_values(ascending=False)
percentage=(previous_application.isnull().sum()/previous_application.isnull().count() *100).sort_values(ascending=False)
missing_previous_application=pd.concat([total_null,percentage],axis=1,keys=["Total_null","Percentage"])
missing_previous_application.head(15)


# In[ ]:


#missing value of install_payment dataset
total_null=install_payment.isnull().sum().sort_values(ascending=False)
percentage=(install_payment.isnull().sum()/install_payment.isnull().count() *100).sort_values(ascending=False)
missing_installment=pd.concat([total_null,percentage],axis=1,keys=["Total_null","Percentage"])
missing_installment.head()


# In[ ]:


#missing value of application_test dataset
total_null=application_test.isnull().sum().sort_values(ascending=False)
percentage=(application_test.isnull().sum()/application_test.isnull().count() *100).sort_values(ascending=False)
missing_app_test=pd.concat([total_null,percentage],axis=1,keys=["Total_null","Percentage"])
missing_app_test.head(15)


# In[ ]:


#lets checkout the discriptive plot of the amt_credit present in training dataset 
plt.figure(figsize=(18,8),dpi=100)
sns.distplot(application_train['AMT_CREDIT'],fit=norm,color="red");
plt.title("Amt_credit distribution");


# In[ ]:


plt.figure(figsize=(10,5),dpi=80)
plt.hist(application_train['TARGET'],bins=50);


# |||||**we can see that the data is highly imbalanced**

# In[ ]:





# In[ ]:





# In[ ]:


target_count = application_train.TARGET.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

count_class_0, count_class_1 = application_train.TARGET.value_counts()

# Divide by class
df_class_0 = application_train[application_train['TARGET'] == 0]
df_class_1 = application_train[application_train['TARGET'] == 1]
df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_test_under.TARGET.value_counts())

df_test_under.TARGET.value_counts().plot(kind='bar', title='Count (TARGET)');


# In[ ]:


temp=application_train['NAME_CONTRACT_TYPE'].value_counts()
x=temp.index
y=temp.values
plt.pie(x=temp.values,explode=(0.1,0),labels=temp.index,startangle=90,autopct='%1.1f%%',
        colors=['#F1BF1B','#B1F11B'],frame=False,radius=1.5)


# ||||Most of the loans are Cash loans which were taken by applicants. 90.5 % loans are Cash loans.
# 

# In[ ]:


#purpose of loan
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8),dpi=100)
car_loan=application_train['FLAG_OWN_CAR'].value_counts()
x=car_loan.index
y=car_loan.values
plt.subplot(2,2,1)
plt.pie(x=car_loan.values,explode=(0.1,0),labels=car_loan.index,
        startangle=90,autopct='%1.1f%%');
plt.subplot(2,2,2)
realty_loan=application_train['FLAG_OWN_REALTY'].value_counts()
X=realty_loan.index
Y=realty_loan.values
plt.pie(x=realty_loan.values,explode=(0.1,0),labels=realty_loan.index,
        startangle=90,autopct='%1.1f%%');


# |||||**Income Source of Applicant who applied for loan**

# In[ ]:


plt.figure(figsize=(13,5),dpi=200)
loan_applicant=application_train['NAME_INCOME_TYPE'].value_counts()
value=(loan_applicant/application_train['NAME_INCOME_TYPE'].count()*100)
sns.barplot(y=value,x=loan_applicant.index,hue=loan_applicant.index,ci=100)
plt.ylabel("pecentage of loan applicant",size=15)


# |||||Family Status of Applicant

# In[ ]:


plt.figure(figsize=(10,5),dpi=100)
applicant_status=application_train['NAME_FAMILY_STATUS'].value_counts()
sns.barplot(y=applicant_status.values,x=applicant_status.index,hue=applicant_status.index,ci=100)
plt.title("Family Status of Applicant",size=20,color="red")


# |||||housing type

# In[ ]:


plt.figure(figsize=(10,6),dpi=100)
temp=application_train['NAME_HOUSING_TYPE'].value_counts()
sns.barplot(y=temp.values,x=temp.index,hue=temp.index);
plt.xlabel("Housing Type",size=14,color="red")
plt.ylabel("frequency",size=14,color="red")
plt.title("Housing Type Name",size=20,color="blue")


# In[ ]:


#Occupation of Apllicant 
plt.figure(figsize=(15,10),dpi=100)
occ=application_train['OCCUPATION_TYPE'].value_counts()
sns.barplot(y=occ.values,x=occ.index)
plt.xlabel("Occupation",size=15,color="blue")
plt.ylabel("Frequency",size=15,color="blue")
plt.title("Occupation Loan Applicant",size=20,color="red")
plt.xticks(rotation=45);


# |||||Here we can see  that the top applicant who were applied for loan :
# 
# Laborers:55-57k
# 
# Sales-Staff:32-33k 
# 
# core_staff:28-29k
# 
# managers:approx 22k 
# 

# In[ ]:


#Education Of Applicant
plt.figure(figsize=(10,5),dpi=100)
edu=application_train["NAME_EDUCATION_TYPE"].value_counts()
sns.barplot(x=edu.index,y=edu.values)
plt.xlabel("Education",size=15,color="blue")
plt.ylabel("Frequency",size=15,color="blue")
plt.title("Education of  Loan Applicant",size=20,color="red")
plt.xticks(rotation=45,size=10);


# **Loan Is Repayed Or Not**

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


# In[ ]:


temp = application_train["OCCUPATION_TYPE"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(application_train["TARGET"][application_train["OCCUPATION_TYPE"]==val] == 1))
    temp_y0.append(np.sum(application_train["TARGET"][application_train["OCCUPATION_TYPE"]==val] == 0))    
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
    title = "Occupation of Applicant's in terms of loan is repayed or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='Occupation of Applicant\'s',
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


# In[ ]:





# In[ ]:


prev=previous_application['NAME_CLIENT_TYPE'].value_counts()
trace = go.Pie(labels=prev.index, values=prev.values)
py.iplot([trace], filename='basic_pie_chart')


# In[ ]:


prev1=previous_application["NAME_CONTRACT_TYPE"].value_counts()
trace = go.Pie(labels=prev1.index, values=prev1.values)
py.iplot([trace], filename='basic_pie_chart')


# In[ ]:


prev2=previous_application["CHANNEL_TYPE"].value_counts()
trace = go.Pie(labels=prev2.index, values=prev2.values)
py.iplot([trace], filename='basic_pie_chart')


# In[ ]:


suite = previous_application['NAME_TYPE_SUITE'].value_counts()
trace = go.Pie(labels=suite.index, values=suite.values)
py.iplot([trace], filename='basic_pie_chart')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


application_train.select_dtypes('object').info()


# In[ ]:


application_train.select_dtypes('int').info()


# |||||**We have seen through the info that only categorical  value have null value** 

# In[ ]:


application_train.select_dtypes("object").nunique()


# most of the catogrical value having small number of unique entity.

# As we seen that in [dytpes=object] we have few unique entries so we have to deal with these categorical variable.
# for this we have to use label encoder and pandas dummy
# we use label encoder to deal  to with categorical variable who have only two unique entries so that it can asign only 0 and 1 and more than two entites we use one hot encoder to dea with

# In[ ]:


from sklearn.preprocessing import LabelEncoder
lbl=LabelEncoder()
lbl_count=0
for i in application_train:
    if application_train[i].dtype=='object':
        if len(list(application_train[i].unique())) <= 2:
            lbl.fit(application_train[i])
            #through this code we encode only those column who have less 
            #than or equal to 2 categorical variable
            application_train[i]=lbl.transform(application_train[i])
            application_test[i]=lbl.transform(application_test[i])
            lbl_count +=1
print('%d column were encoded.'%lbl_count)
  
            


# In[ ]:


application_train.select_dtypes("object").nunique()


# In[ ]:


#for one hot enocder we use pd.get_dummies
application_train=pd.get_dummies(application_train)
application_test=pd.get_dummies(application_test)
print("application_train feature shape:",application_train.shape)
print("application_test feature shape:",application_test.shape)


# In[ ]:


train_target=application_train['TARGET']
application_train,application_test=application_train.align(application_test,axis=1,join='inner')


# In[ ]:


application_train['TARGET']=train_target


# In[ ]:


application_train.head()


# In[ ]:


#as DAYS_BIRTH given negative in the dataset we hav to make positive to analyse in years
(application_train['DAYS_BIRTH']/(-365)).describe()


# thats looks good there is no outlier in the age field

# In[ ]:


application_train.isnull().sum()


# In[ ]:





# In[ ]:


application_train.isnull().sum()


# In[ ]:


prev_category = pd.get_dummies(previous_application)
bureau_category = pd.get_dummies(bureau)
pos_category = pd.get_dummies(POS_cash)
credit_category= pd.get_dummies(credit_card_balance)


# In[ ]:


application_train=application_train.fillna(0)
application_test=application_test.fillna(0)


# In[ ]:


from sklearn.model_selection import train_test_split 
import lightgbm as lgb

application_test['is_test'] = 1 
application_test['is_train'] = 0
application_train['is_test'] = 0
application_train['is_train'] = 1

# target variable
Y = application_train['TARGET']
train_X = application_train.drop(['TARGET'], axis = 1)

# test ID
test_id = application_train['SK_ID_CURR']
test_X = application_test

# merge train and test datasets for preprocessing
data = pd.concat([train_X, test_X], axis=0)


# In[ ]:


prev_apps = previous_application[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
previous_application['SK_ID_PREV'] = previous_application['SK_ID_CURR'].map(prev_apps['SK_ID_PREV'])

## Average values for all other features in previous applications
prev_apps_avg = previous_application.groupby('SK_ID_CURR').mean()
prev_apps_avg.columns = ['p_' + col for col in prev_apps_avg.columns]
data = data.merge(right=prev_apps_avg.reset_index(), how='left', on='SK_ID_CURR')


# In[ ]:


bureau_avg = bureau.groupby('SK_ID_CURR').mean()
bureau_avg['buro_count'] = bureau[['SK_ID_BUREAU','SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
bureau_avg.columns = ['b_' + f_ for f_ in bureau_avg.columns]
data = data.merge(right=bureau_avg.reset_index(), how='left', on='SK_ID_CURR')


# In[ ]:


install_pay= install_payment[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
install_payment['SK_ID_PREV'] = install_payment['SK_ID_CURR'].map(install_pay['SK_ID_PREV'])

## Average values for all other variables in installments payments
avg_inst = install_payment.groupby('SK_ID_CURR').mean()
avg_inst.columns = ['i_' + f_ for f_ in avg_inst.columns]
data = data.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')


# In[ ]:


pos_cash = POS_cash[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
POS_cash['SK_ID_PREV'] = POS_cash['SK_ID_CURR'].map(pos_cash['SK_ID_PREV'])

## Average Values for all other variables in pos cash
POS_avg = POS_cash.groupby('SK_ID_CURR').mean()
data = data.merge(right=POS_avg.reset_index(), how='left', on='SK_ID_CURR')


# In[ ]:


credit_balns= credit_card_balance[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
credit_card_balance['SK_ID_PREV'] = credit_card_balance['SK_ID_CURR'].map(credit_balns['SK_ID_PREV'])

### average of all other columns 
avg_credit_bal = credit_card_balance.groupby('SK_ID_CURR').mean()
avg_credit_bal.columns = ['credit_bal_' + f_ for f_ in avg_credit_bal.columns]
data = data.merge(right=avg_credit_bal.reset_index(), how='left', on='SK_ID_CURR')


# In[ ]:


#final training and testing data
ignore_features = ['SK_ID_CURR', 'is_train', 'is_test']
relevant_features = [col for col in data.columns if col not in ignore_features]
trainX = data[data['is_train'] == 1][relevant_features]
testX = data[data['is_test'] == 1][relevant_features]


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(trainX, Y, test_size=0.2, random_state=18)
lgb_train = lgb.Dataset(data=x_train, label=y_train)
lgb_eval = lgb.Dataset(data=x_val, label=y_val)


# In[ ]:


import lightgbm as lgb
params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 
          'learning_rate': 0.01, 'num_leaves': 48, 'num_iteration': 5000, 'verbose': 0 ,
          'colsample_bytree':.8, 'subsample':.9, 'max_depth':7, 'reg_alpha':.1, 'reg_lambda':.1, 
          'min_split_gain':.01, 'min_child_weight':1}
model = lgb.train(params, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=170, verbose_eval=200)


# In[ ]:





# In[ ]:


lgb.plot_importance(model, max_num_features=100, figsize=(15, 30),color="red")


# In[ ]:


preds = model.predict(testX)
sub = application_test[['SK_ID_CURR']].copy()
sub['TARGET'] = preds
sub.to_csv('sub.csv', index= False)
sub.head(10)


# In[ ]:





# In[ ]:





# In[ ]:




