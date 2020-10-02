#!/usr/bin/env python
# coding: utf-8

# # Loan_Prediction_project

# In[ ]:


import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,roc_auc_score,roc_curve,auc
import statsmodels.api as sm


# # Reading Test Dataset......

# In[ ]:


df=pd.read_csv('test_Y.csv')


# # Reading Training Dataset....

# In[ ]:


df1=pd.read_csv('train_u.csv')


# In[ ]:


df.head(5)


# In[ ]:


df1.head(5)


# # Detail of related variable(i.e. columns name)

# In[ ]:


print(df.columns)
print(df1.columns)


# In[ ]:


#summery of numerical variable for training dataset
df1.describe()


# # Data Visulization of training dataset

# In[ ]:


#Types of variable
df1.dtypes


# In[ ]:


#Getting the percentage value for the Property_area
r=pd.get_dummies(df1['Property_Area'])
r1=(sum(r["Semiurban"]),sum(r["Urban"]),sum(r["Rural"]))
plt.pie(r1,labels=["Semiurban","Urban","Rural"],shadow=True,explode=(.1,.1,.1), autopct='%1.1f%%')


# In[ ]:


#Get, how many male and feamale take the loan
r2=pd.get_dummies(df1['Gender'])
r3=(sum(r2["Female"]),sum(r2["Male"]))
plt.pie(r3,labels=["Male","Female"],shadow=True,explode=(.1,.1), autopct='%1.1f%%')


# In[ ]:


#Get,percent value for variable Education
p=pd.get_dummies(df1['Education'])
p1=(sum(p["Graduate"]),sum(p["Not Graduate"]))
plt.pie(p1,labels=["Graduate","Not Graduate"],shadow=True,explode=(.1,.1), autopct='%1.1f%%')


# #Now we see Graduated people are take loan

# # Understanding the distribution and to observe the outliers

# In[ ]:


sns.distplot(df1['ApplicantIncome'],bins=100)


# In[ ]:


df1['ApplicantIncome'].plot(kind="kde",figsize=(20,5))


# In[ ]:


sns.catplot(data=df1,x='Education',y='ApplicantIncome',hue='Loan_Status',kind='box')
#df1.boxplot(column='ApplicantIncome',by='Education')


# #box plot show the both mean value is same. Some graduate people high income but tose are not eligible for loan

# In[ ]:


sns.catplot(data=df1,x='Education',y='LoanAmount',hue='Loan_Status',kind='boxen')


# In[ ]:


sns.catplot(data=df1,x='Married',y='ApplicantIncome',hue='Loan_Status',kind='box')


# #box plot shows married people have high income but those are not eligible for loan

# In[ ]:


df1['LoanAmount_log']=np.log(df1['LoanAmount'])
sns.distplot(df1['LoanAmount_log'],bins=20)


# In[ ]:


sns.boxplot(data=df1,y='LoanAmount',x='Education',hue='Loan_Status')


# In[ ]:


sns.boxplot(data=df1,y='LoanAmount',x='Married',hue='Loan_Status')


# In[ ]:


sns.boxplot(data=df1,y='LoanAmount',x='Gender',hue='Loan_Status')


# #loanAmount has missing values as well as outlier and ApplicantIncome has some outliers

# In[ ]:


sns.distplot(df['LoanAmount'],bins=20)


# # Visulization of categorical variable 

# In[ ]:


loan_app=df1['Loan_Status'].value_counts()["Y"]
print(loan_app)

# 422 number of loans approved
# In[ ]:


print(pd.crosstab(df1["Credit_History"],df1["Loan_Status"],margins=True))
plt.figure(figsize=(10,4))
sns.countplot(x='Credit_History',hue='Loan_Status',data=df1,order=df1['Credit_History'].value_counts().index);


# In[ ]:


plt.figure(figsize=(10,4))
sns.countplot(x='Education',hue='Loan_Status',data=df1,order=df1['Education'].value_counts().index);


# In[ ]:


plt.figure(figsize=(10,4))
sns.countplot(x='Gender',hue='Loan_Status',data=df1,order=df1['Gender'].value_counts().index);


# In[ ]:


def percentagecon(ser):
    return ser/float(ser[-1])
df_1=pd.crosstab(df1["Credit_History"],df1["Loan_Status"],margins=True).apply(percentagecon,axis=1)
loan_app_wcredit_1=df_1["Y"][1]
print(loan_app_wcredit_1*100)


# 79.5% loan are approved whose credit_history is 1

# In[ ]:


plt.plot(df1['ApplicantIncome'])
plt.plot(df1['CoapplicantIncome'])
df1['TotalIncome']=df1['ApplicantIncome']+df1['CoapplicantIncome']


# In[ ]:


sns.distplot(df1['TotalIncome'],bins=20)


# # Transform the categorical data into numerical data

# In[ ]:


cat=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area','Loan_Status']
for var in cat:
    le=preprocessing.LabelEncoder()
    df1[var]=le.fit_transform(df1[var].astype('str'))
df1.dtypes


# # Dealing the missing/outlier values

# In[ ]:


x2=['LoanAmount','ApplicantIncome','CoapplicantIncome','Loan_Amount_Term']
for i in x2:
    df1[i].fillna(df1[i].mean(),inplace=True)
df1['Credit_History'].fillna(df1['Credit_History'].mode()[0],inplace=True)


# # Analysis the training dataset

# In[ ]:


#we choose dependent and independent variables
x=df1[['Credit_History','Education','Gender']]
y=df1['Loan_Status']


# In[ ]:


#Summery of logistic Regression model
import statsmodels.api as sm
x1=sm.add_constant(x)
logist_modal=sm.Logit(y,x1)
result=logist_modal.fit()
print(result.summary())


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20)
log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)


# In[ ]:


y_pred=log_reg.predict(x_test)


# In[ ]:


accuracy=accuracy_score(y_test,y_pred)
print(accuracy)


#  prediction accuracy is 80.48%

# In[ ]:


tp,fp,fn,tn=confusion_matrix(y_test,y_pred).ravel()
print("True_positive",tp)
print("False_positive",fp)
print("False_negative",fn)
print("True_negative",tn)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


y_prob_train=log_reg.predict_proba(x_train)[:,1]
y_prob_train.reshape(1,-1)


# In[ ]:


fpr_p,tpr_p,threshol=roc_curve(y_train,y_prob_train)
roc_auc_p=auc(fpr_p,tpr_p)
print(roc_auc_p)


# In[ ]:


plt.figure()
plt.plot(fpr_p,tpr_p,color='green',label='ROC curve'% roc_auc_p)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Recurcive Operating characterstics')
plt.legend(loc='lower right')
plt.show()


# # Analysis the dataset using training as well as test 

# In[ ]:


df1['Type']='Train'
df['Type']='Test'
N_data=pd.concat([df1,df],axis=0,sort=True)
N_data.isnull().sum()


# In[ ]:


idcol=['Loan_ID']
tar_col=["Loan_Status"]
new_col=['Credit_History','Dependents','Gender','Married','Education','Property_Area','Self_Employed']
for var in new_col:
    num=LabelEncoder()
    N_data[var]=num.fit_transform(N_data[var].astype('str'))
    
train_mod=N_data[N_data['Type']=='Train']
test_mod=N_data[N_data['Type']=='Test']
train_mod["Loan_Status"]=num.fit_transform(train_mod["Loan_Status"])


# In[ ]:


N_data['TotalIncome']=N_data['ApplicantIncome']+N_data['CoapplicantIncome']
N_data['TotalIncome_log']=np.log(N_data['TotalIncome'])
sns.distplot(N_data['TotalIncome_log'],bins=20)


# In[ ]:


predict_log=['Credit_History','Education','Gender']
x_train_c=train_mod[predict_log]
y_train_c=train_mod['Loan_Status']
x_test_c=test_mod[predict_log]


# In[ ]:


result_c=log_reg.fit(x_train_c,y_train_c)
predicted_c=log_reg.predict(x_test_c)
predicted_c=num.inverse_transform(predicted_c)
test_mod['Loan_Status']=predicted_c


# In[ ]:


y_c=y[:367]
accuracy_c=accuracy_score(predicted_c,y_c)
print(accuracy_c)


# In[ ]:




