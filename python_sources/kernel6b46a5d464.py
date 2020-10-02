#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


data= pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[ ]:


data


# In[ ]:


data.info()


# In[ ]:


get_ipython().system('pip install pycaret')
from pycaret.classification import *


# In[ ]:


print ("Rows     : " ,data.shape[0])
print ("Columns  : " ,data.shape[1])
print ("\nFeatures : \n" ,data.columns.tolist())
print ("\nMissing values :  ", data.isnull().sum().values.sum())
print ("\nUnique values :  \n",data.nunique())


# In[ ]:


data['TotalCharges']=data["TotalCharges"].replace(r'\s+',np.nan,regex=True)
data['TotalCharges']=pd.to_numeric(data['TotalCharges'])


# In[ ]:


data.Partner.value_counts(normalize=True).plot(kind='bar')


# In[ ]:


data.SeniorCitizen.value_counts(normalize=True).plot(kind='bar')


# In[ ]:


data.gender.value_counts(normalize=True).plot(kind='bar')


# In[ ]:


data.tenure.value_counts(normalize=True).plot(kind='bar',figsize=(16,7))


# In[ ]:


data.PhoneService.value_counts(normalize=True).plot(kind='bar')


# In[ ]:


data.MultipleLines.value_counts(normalize=True).plot(kind='bar')


# In[ ]:


data.InternetService.value_counts(normalize=True).plot(kind='bar')


# In[ ]:


data.Contract.value_counts(normalize=True).plot(kind='bar')


# In[ ]:


data.PaymentMethod.value_counts(normalize=True).plot(kind='bar')


# In[ ]:


data.Churn.value_counts(normalize=True).plot(kind='bar')


# In[ ]:


print(pd.crosstab(data.gender,data.Churn,margins=True))
pd.crosstab(data.gender,data.Churn,margins=True).plot(kind='bar',figsize=(7,5))


# In[ ]:


print('Percent of females that left the company {0}'.format((939/1869)*100))
print('Percent of males that left the company {0}'.format((930/1869)*100))


# In[ ]:


print(pd.crosstab(data.Contract,data.Churn,margins=True))
pd.crosstab(data.Contract,data.Churn,margins=True).plot(kind='bar',figsize=(7,5))


# In[ ]:


print("% off month to month ",((1655/1869)*100))
print("% off one year ",((166/1869)*100))
print("% off two year ",((48/1869)*100))


# In[ ]:


print(pd.crosstab(data.InternetService,data.Churn,margins=True))
pd.crosstab(data.InternetService,data.Churn,margins=True).plot(kind='bar',figsize=(7,5))


# In[ ]:


print("% of DSL service",((459/1869)*100))
print("% of fibre optic",((1297/1869)*100))
print("% of No internet",((113/1869)*100))


# In[ ]:


print(pd.crosstab(data.tenure.median(),data.Churn,margins=True))
pd.crosstab(data.tenure.median(),data.Churn,margins=True).plot(kind='bar',figsize=(7,5))


# In[ ]:


print(pd.crosstab(data.Partner,data.Dependents,margins=True))
pd.crosstab(data.Partner,data.Dependents,margins=True).plot(kind='bar',figsize=(7,5))


# In[ ]:


print("% of partner that had dependents",((1749/2110)*100))
print("% of non-partner that had dependents",((361/2110)*100))


# In[ ]:


print(pd.crosstab(data.Partner,data.Churn,margins=True))
pd.crosstab(data.Partner,data.Churn,margins=True).plot(kind='bar',figsize=(7,5))


# In[ ]:


plt.figure(figsize=(17,8))
sns.countplot(x=data['tenure'],hue=data.Partner)


# In[ ]:


print(pd.crosstab(data.SeniorCitizen,data.Churn,margins=True))
pd.crosstab(data.SeniorCitizen,data.Churn,margins=True).plot(kind='bar',figsize=(7,5))


# In[ ]:


data.boxplot('MonthlyCharges')


# In[ ]:


data.boxplot('TotalCharges')


# In[ ]:


data.describe()


# In[ ]:


data.isnull().sum()


# In[ ]:


fill=data.MonthlyCharges*data.tenure


# In[ ]:


data.TotalCharges.fillna(fill,inplace=True)


# In[ ]:


data.isnull().sum()


# In[ ]:


data.loc[(data.Churn=='Yes'),'MonthlyCharges'].median()


# In[ ]:


data.loc[(data.Churn=='Yes'),'TotalCharges'].median()


# In[ ]:


data.loc[(data.Churn=='Yes'),'tenure'].median()


# In[ ]:


data.loc[(data.Churn=='Yes'),'PaymentMethod'].value_counts(normalize=True)


# In[ ]:


df=data


# In[ ]:


def changeColumnsToString(df):
    columnsNames=['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
    for col in columnsNames:
        df[col]=df[col].astype('str').str.replace('Yes','1').replace('No','0').replace('No internet service','0').replace('No phone service',0)

changeColumnsToString(df)

df['SeniorCitizen']=df['SeniorCitizen'].astype(bool)
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')


# In[ ]:


df.head(2)


# In[ ]:



print("Payment methods: ",df.PaymentMethod.unique())
print("Contract types: ",df.Contract.unique())
print("Gender: ",df.gender.unique())
print("Senior Citizen: ",df.SeniorCitizen.unique())
print("Internet Service Types: ",df.InternetService.unique())


# In[ ]:



df['gender']=df['gender'].astype('category')
df['PaymentMethod']=df['PaymentMethod'].astype('category')
df['Contract']=df['Contract'].astype('category')
df['SeniorCitizen']=df['SeniorCitizen'].astype('category')
df['InternetService']=df['InternetService'].astype('category')
df.dtypes


# In[ ]:


dfPaymentDummies = pd.get_dummies(df['PaymentMethod'], prefix = 'payment')
dfContractDummies = pd.get_dummies(df['Contract'], prefix = 'contract')
dfGenderDummies = pd.get_dummies(df['gender'], prefix = 'gender')
dfSeniorCitizenDummies = pd.get_dummies(df['SeniorCitizen'], prefix = 'SC')
dfInternetServiceDummies = pd.get_dummies(df['InternetService'], prefix = 'IS')

print(dfPaymentDummies.head(3))
print(dfContractDummies.head(3))
print(dfGenderDummies.head(3))
print(dfSeniorCitizenDummies.head(3))
print(dfInternetServiceDummies.head(3))


# In[ ]:



df.drop(['gender','PaymentMethod','Contract','SeniorCitizen','InternetService'], axis=1, inplace=True)

df = pd.concat([df, dfPaymentDummies], axis=1)
df = pd.concat([df, dfContractDummies], axis=1)
df = pd.concat([df, dfGenderDummies], axis=1)
df = pd.concat([df, dfSeniorCitizenDummies], axis=1)
df = pd.concat([df, dfInternetServiceDummies], axis=1)
df.head(2)


# In[ ]:



df.columns = ['customerID', 'Partner', 'Dependents', 'tenure', 'PhoneService',
       'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
       'MonthlyCharges', 'TotalCharges', 'Churn',
       'payment_Bank_transfer_auto', 'payment_Credit_card_auto',
       'payment_Electronic_check', 'payment_Mailed_check',
       'contract_Month_to_month', 'contract_One_year', 'contract_Two_year',
       'gender_Female', 'gender_Male', 'SC_False', 'SC_True', 'IS_DSL',
       'IS_Fiber_optic', 'IS_No']


# In[ ]:



numericColumns=np.array(['Partner', 'Dependents', 'tenure', 'PhoneService',
       'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
       'MonthlyCharges', 'TotalCharges', 'Churn',
       'payment_Bank_transfer_auto', 'payment_Credit_card_auto',
       'payment_Electronic_check', 'payment_Mailed_check',
       'contract_Month_to_month', 'contract_One_year', 'contract_Two_year',
       'gender_Female', 'gender_Male', 'SC_False', 'SC_True', 'IS_DSL',
       'IS_Fiber_optic', 'IS_No'])

for columnName in numericColumns:
    df[columnName]=pd.to_numeric(df[columnName],errors='coerce')
df.dtypes


# In[ ]:


df


# In[ ]:


train=df[:6000]
train


# In[ ]:


test=df[6000:]
test


# In[ ]:


new_test=test['Churn']


# In[ ]:


test.drop(['Churn'],axis=1,inplace=True)


# In[ ]:


test


# In[ ]:


from pycaret.classification import *


# In[ ]:


clf = setup(data = train, 
             target = 'Churn'
           )


# In[ ]:


compare_models()


# In[ ]:


lgbm  = create_model('lightgbm')    


# In[ ]:


tuned_lightgbm = tune_model('lightgbm')


# In[ ]:


plot_model(estimator = tuned_lightgbm, plot = 'learning')


# In[ ]:


plot_model(estimator = tuned_lightgbm, plot = 'auc')


# In[ ]:


plot_model(estimator = tuned_lightgbm, plot = 'confusion_matrix')


# In[ ]:


plot_model(estimator = tuned_lightgbm, plot = 'feature')


# In[ ]:


evaluate_model(tuned_lightgbm)


# In[ ]:


interpret_model(tuned_lightgbm)


# In[ ]:


predict_model(tuned_lightgbm, data=test)


# In[ ]:


predictions = predict_model(tuned_lightgbm, data=test)
predictions.head(20)


# In[ ]:


new_test3=round(predictions['Score']).astype(int)


# In[ ]:


new_test3


# In[ ]:


new_test


# In[ ]:


new_test.to_csv('submission1.csv',index=False)


# In[ ]:


new_test3.to_csv('submission2.csv',index=2)


# In[ ]:


d=pd.read_csv('submission1.csv')


# In[ ]:


d1=pd.read_csv('submission2.csv')
d1


# In[ ]:


d['pred_churn']=d1['0.1']


# In[ ]:


d.to_csv('final_sub.csv')


# In[ ]:


d


# In[ ]:





# 
