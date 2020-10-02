#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os
from datetime import datetime
import re

from sklearn import metrics


# In[ ]:


pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)


# In[ ]:


data_dictionary=pd.read_csv('../input/lt-vehicle-loan-default-prediction/data_dictionary.csv')
data_dictionary


# In[ ]:


df=pd.read_csv('../input/lt-vehicle-loan-default-prediction/train.csv')
df.head()


# In[ ]:


df.dtypes


# In[ ]:


df.nunique()


# In[ ]:


df.describe()


# In[ ]:


import missingno


# In[ ]:


missingno.matrix(df,figsize=(40,15))


# From this matrix we find that Employment Type has missing values

# In[ ]:


df.nunique()


# In[ ]:


df.shape


# In[ ]:


plt.figure(figsize=(30,20))
sns.heatmap(df.corr())
plt.show()


# In[ ]:


plt.figure(figsize=(40,30))
sns.heatmap(df.corr(),linewidths=0.05,annot=True)
plt.show()


# In[ ]:


df['disbursed_amount'].corr(df['ltv'])


# In[ ]:


np.corrcoef(df['disbursed_amount'],df['ltv'])


# In[ ]:


df.corr()


# In[ ]:


df.drop(['MobileNo_Avl_Flag','UniqueID'],axis=1,inplace=True)


# In[ ]:


df1=df[df['loan_default']==1]
df0=df[df['loan_default']==0]


# ### disbursed_amount

# In[ ]:


sns.boxplot(df['disbursed_amount'])


# In[ ]:


plt.figure(figsize=(10,4))
sns.distplot(df0['disbursed_amount'],kde=False)
sns.distplot(df1['disbursed_amount'],kde=False)
plt.legend(labels=['Not Defaulted','Defaulted'])
plt.show()


# In[ ]:


sns.distplot(df['disbursed_amount'])


# In[ ]:


sns.distplot(df['disbursed_amount'].apply(lambda x:np.log(x)))


# In[ ]:


df['disbursed_amount']=df['disbursed_amount'].apply(lambda x:np.log(x))


# ### asset_cost

# In[ ]:


df


# In[ ]:


plt.figure(figsize=(10,4))
sns.distplot(df0['asset_cost'],kde=False)
sns.distplot(df1['asset_cost'],kde=False)
plt.legend(labels=['Not Defaulted','Defaulted'])
plt.show()


# In[ ]:


sns.distplot(df['asset_cost'])


# In[ ]:


sns.distplot(df['asset_cost'].apply(lambda x:np.log(x)))


# In[ ]:


df['asset_cost']=df['asset_cost'].apply(lambda x:np.log(x))


# ### ltv

# In[ ]:


sns.boxplot(df['ltv'])


# In[ ]:


sns.distplot(df['ltv'])


# In[ ]:


plt.figure(figsize=(10,4))
sns.distplot(df0['ltv'],kde=False)
sns.distplot(df1['ltv'],kde=False)
plt.legend(labels=['Not Defaulted','Defaulted'])
plt.show()


# ### supplier_id

# In[ ]:


df['supplier_id'].value_counts()


# In[ ]:


supplier_loan=pd.crosstab(df['supplier_id'],df['loan_default'])


# In[ ]:


plt.figure(figsize=(30,5))
sns.countplot(df['supplier_id'])


# In[ ]:


pd.crosstab(df['supplier_id'],df['loan_default']).plot(kind='bar',figsize=(30,5))


# In[ ]:


import  scipy.stats                     as  stats


# In[ ]:


chi_sq, p_value, deg_freedom, exp_freq = stats.chi2_contingency(supplier_loan)
print('Chi Square Statistics',chi_sq)
print('p-value',p_value)
print('Degree of freedom',deg_freedom)
p_value


# ### branch_id

# In[ ]:


df['branch_id'].value_counts()


# In[ ]:


plt.figure(figsize=(30,5))
sns.countplot(df['branch_id'])
plt.show()


# In[ ]:


pd.crosstab(df['branch_id'],df['loan_default'])


# In[ ]:


chi_sq, p_value, deg_freedom, exp_freq = stats.chi2_contingency(supplier_loan)
print('Chi Square Statistics',chi_sq)
print('p-value',p_value)
print('Degree of freedom',deg_freedom)


# ### manufacture_id

# In[ ]:


df['manufacturer_id'].value_counts()


# In[ ]:


plt.figure(figsize=(30,7))
sns.countplot(df0['manufacturer_id'],color='blue')
sns.countplot(df1['manufacturer_id'],color='orange')
plt.legend(labels=['Not-Defaulted','Defaulted'])
plt.show()


# In[ ]:


plt.figure(figsize=(30,7))
sns.countplot(x='manufacturer_id',hue='loan_default',data=df)
plt.legend(labels=['Not-Defaulted','Defaulted'])
plt.show()


# In[ ]:


manu_loan=pd.crosstab(df['manufacturer_id'],df['loan_default'])
print(manu_loan)


# In[ ]:


chi_sq, p_value, deg_freedom, exp_freq = stats.chi2_contingency(manu_loan)
print('Chi Square Statistics',chi_sq)
print('p-value',p_value)
print('Degree of freedom',deg_freedom)


# ### age of disbursal
# Changed to **Age**

# In[ ]:


pres_date='01-01-20' # Present date

def days_between(d1,d2):
    d1=datetime.strptime(d1,'%d-%m-%y')
    d2=datetime.strptime(d2,'%d-%m-%y')
    return abs((d2-d1).days)


# In[ ]:


df['Date.of.Birth']=df['Date.of.Birth'].apply(lambda x:days_between(x,pres_date)/365) # converting date of birth in years
df['DisbursalDate']=df['DisbursalDate'].apply(lambda x:days_between(x,pres_date)/365) # converting disbursed date to the present date


# In[ ]:


df.head()


# In[ ]:


df[df['loan_default']==0]['DisbursalDate']


# In[ ]:


plt.figure(figsize=(40,20))
sns.distplot(df[df['loan_default']==0]['DisbursalDate'],color='b')
sns.distplot(df[df['loan_default']==1]['DisbursalDate'],color='r')
plt.legend(labels=['Not-Defaulted','Defaulted'],prop={'size': 25})
plt.show()


# In[ ]:


plt.figure(figsize=(30,20))
sns.distplot(df[df['loan_default']==0]['Date.of.Birth'],color='b',kde=True)
sns.distplot(df[df['loan_default']==1]['Date.of.Birth'],color='r',kde=True)
plt.legend(labels=['Not-Defaulted','Defaulted'],prop={'size': 25})
plt.show()


# ### Employment Type

# In[ ]:


df['Employment.Type'].value_counts()


# In[ ]:


df['Employment.Type']=df['Employment.Type'].fillna('unknown')


# In[ ]:


sns.countplot(y='Employment.Type',data=df,hue='loan_default')
plt.legend(labels=['Not-Defaulted','Defaulted'])
plt.show()


# In[ ]:


ct=pd.crosstab(df['Employment.Type'], df['loan_default'])
ct.plot.bar(stacked=True,figsize=(12,5))
plt.legend(labels=['Not Defaulted','Defaulted'])
plt.show()


# In[ ]:


df['Employment.Type'].value_counts().plot(kind='bar')


# In[ ]:


df['Employment.Type'].value_counts(normalize=True)


# In[ ]:


emp_loan=pd.crosstab(df['Employment.Type'],df['loan_default'])
print(emp_loan)


# In[ ]:


chi_sq, p_value, deg_freedom, exp_freq = stats.chi2_contingency(emp_loan)
print('Chi Square Statistics',chi_sq)
print('p-value',p_value)
print('Degree of freedom',deg_freedom)


# ## State ID

# In[ ]:


plt.figure(figsize=(30,7))
sns.countplot(x='State_ID',hue='loan_default',data=df)
plt.legend(labels=['Not-Defaulted','Defaulted'])
plt.show()


# ## Employee ID

# In[ ]:


'''plt.figure(figsize=(7,20))
sns.countplot(y='Employee_code_ID',hue='loan_default',data=df)
plt.legend(labels=['Not-Defaulted','Defaulted'])
plt.show()'''


# In[ ]:





# ### State_ID: Passport_flag

# In[ ]:


df.loc[:,'Aadhar_flag':'Passport_flag']


# In[ ]:


temp=df.loc[:,'Aadhar_flag':'Passport_flag'].columns
print(temp)


# In[ ]:


df['Aadhar_flag']+df['PAN_flag']+df['VoterID_flag']+df['Driving_flag']+df['Passport_flag']


# In[ ]:


df['Flag']=df['Aadhar_flag'].astype('object')+df['PAN_flag'].astype('object')+df['VoterID_flag'].astype('object')+df['Driving_flag'].astype('object')+df['Passport_flag'].astype('object')


# In[ ]:


df['Flag'].value_counts()


# In[ ]:


df['Flag']=df['Flag'].astype('int32')


# In[ ]:


sns.countplot(x='Flag',hue='loan_default',data=df)


# In[ ]:


for i in ['State_ID', 'Employee_code_ID','Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag', 'Passport_flag']:
    ct=pd.crosstab(df[i], df['loan_default'])
    ct.plot.bar(stacked=True,figsize=(12,5))
    plt.legend(labels=['Not Defaulted','Defaulted'])
    plt.show()


# In[ ]:


for i in temp:
    print('Feature:',i)
    chi_sq, p_value, deg_freedom, exp_freq = stats.chi2_contingency(pd.crosstab(df[i],df['loan_default']))
    print('Chi Square Statistics',chi_sq)
    print('p-value',p_value)
    print('Degree of freedom',deg_freedom)
    print()


# In[ ]:


df=df.drop(['State_ID', 'Employee_code_ID','Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag', 'Passport_flag'],axis=1)


# ### 'PERFORM_CNS.SCORE', 'PERFORM_CNS.SCORE.DESCRIPTION'

# In[ ]:



df[['PERFORM_CNS.SCORE', 'PERFORM_CNS.SCORE.DESCRIPTION']]


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(y='PERFORM_CNS.SCORE.DESCRIPTION',hue='loan_default',data=df)
plt.legend(labels=['Not Defalted','Defaulted'])
plt.show()


# In[ ]:


plt.figure(figsize=(30,5))
sns.distplot(df['PERFORM_CNS.SCORE'])
plt.show()


# In[ ]:


def cns_score(score):
    if score<100:
        return 0
    elif (score>=100) & (score<200):
        return 1
    elif (score>=200) & (score<300):
        return 2
    elif (score>=300) & (score<400):
        return 3
    elif (score>=400) & (score<500):
        return 4
    elif (score>=500) & (score<600):
        return 5
    elif (score>=600) & (score <700):
        return 6
    elif (score>=700) & (score <800):
        return 7
    elif (score>=800) & (score <900):
        return 8
    elif (score>=900) & (score <1000):
        return 9
    else:
        return 10


# In[ ]:




cns_score(1004)


# In[ ]:


df['PERFORM_CNS.SCORE'].map(lambda x:cns_score(x)).value_counts()


# In[ ]:


df['PERFORM_CNS.SCORE']=df['PERFORM_CNS.SCORE'].map(lambda x:cns_score(x))


# In[ ]:


df[ 'PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[ ]:


df['PERFORM_CNS.SCORE.DESCRIPTION']=le.fit_transform(df['PERFORM_CNS.SCORE.DESCRIPTION'])


# In[ ]:


df['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()


# In[ ]:


count=1
for i in ['PERFORM_CNS.SCORE', 'PERFORM_CNS.SCORE.DESCRIPTION']:
    plt.subplot(2,1,count)
    ct=pd.crosstab(df[i], df['loan_default'])
    ct.plot.bar(stacked=True,figsize=(12,5))
    plt.legend(labels=['Not Defaulted','Defaulted'])
    plt.show()
    count+=1


# In[ ]:


for i in ['PERFORM_CNS.SCORE', 'PERFORM_CNS.SCORE.DESCRIPTION']:
    print('Feature:',i)
    chi_sq, p_value, deg_freedom, exp_freq = stats.chi2_contingency(pd.crosstab(df[i],df['loan_default']))
    print('Chi Square Statistics',chi_sq)
    print('p-value',p_value)
    print('Degree of freedom',deg_freedom)
    print()


# ### 'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS', 'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE','PRI.SANCTIONED.AMOUNT', 'PRI.DISBURSED.AMOUNT'

# In[ ]:


df.loc[:,[ 'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS', 'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE','PRI.SANCTIONED.AMOUNT', 'PRI.DISBURSED.AMOUNT']]


# In[ ]:


primary=df.loc[:,[ 'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS', 'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE','PRI.SANCTIONED.AMOUNT', 'PRI.DISBURSED.AMOUNT']]


# In[ ]:


primary.describe()


# In[ ]:


primary['PRI.NO.OF.ACCTS'].value_counts()


# In[ ]:


pri_col=['PRI.CURRENT.BALANCE','PRI.SANCTIONED.AMOUNT', 'PRI.DISBURSED.AMOUNT']


# In[ ]:


count=1
plt.figure(figsize=(25,10))
for i in pri_col:
    plt.subplot(2,2,count)
    sns.distplot(df[i])
    count+=1
plt.tight_layout()


# In[ ]:


df.loc[:,pri_col].corr()


# In[ ]:


sns.heatmap(df.loc[:,pri_col].corr(),annot=True)


# In[ ]:


#sns.distplot(df['disbursed_amount'].apply(lambda x:np.log1p(x)))


# In[ ]:


np.log(0+1)


# ### 'SEC.NO.OF.ACCTS','SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE','SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT'

# In[ ]:


df[['SEC.NO.OF.ACCTS',
       'SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE',
       'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT']]


# In[ ]:


secondary=df[['SEC.NO.OF.ACCTS',
       'SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE',
       'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT']]


# In[ ]:


secondary['SEC.NO.OF.ACCTS'].value_counts()


# In[ ]:


secondary.describe()


# ## Totals COMBINING PRIMARY AND SECONDARY TO ONE

# In[ ]:


df


# In[ ]:


df.loc[:,'total.no.of.accts']=df['PRI.NO.OF.ACCTS']+df['SEC.NO.OF.ACCTS']
df.loc[:,'pri.inactive.accts']=df['PRI.NO.OF.ACCTS']-df['PRI.ACTIVE.ACCTS']
df.loc[:,'sec.inactive.accts']=df['SEC.NO.OF.ACCTS']-df['SEC.ACTIVE.ACCTS']
df.loc[:,'total.inactive.accts']=df['pri.inactive.accts']-df['sec.inactive.accts']
df.loc[:,'total.overdue.accts']=df['PRI.OVERDUE.ACCTS']+df['SEC.OVERDUE.ACCTS']
df.loc[:,'total.current.balance']=df['PRI.CURRENT.BALANCE']+df['SEC.CURRENT.BALANCE']
df.loc[:,'total.disbursed.amount']=df['PRI.DISBURSED.AMOUNT']+df['SEC.CURRENT.BALANCE']
df.loc[:,'total.sanctioned.amount']=df['PRI.SANCTIONED.AMOUNT']+df['SEC.SANCTIONED.AMOUNT']
df.loc[:,'total.installment']=df['PRIMARY.INSTAL.AMT']+df['SEC.SANCTIONED.AMOUNT']
# df.loc[:,'bal.to.disburse']=np.round((1+df['total.disbursed.amount'])/(1+df['total.current.balance']),2) # balance to disbursed anount ratio
df.loc[:,'pri.tenure']=(df['PRI.DISBURSED.AMOUNT']/(df['PRIMARY.INSTAL.AMT']+1)).astype(int)
df.loc[:,'sec.tenure']=(df['SEC.DISBURSED.AMOUNT']/(df['SEC.INSTAL.AMT']+1)).astype(int)
df.loc[:,'disburse.to.sanctioned']=np.round((1+df['total.disbursed.amount'])/(1+df['total.sanctioned.amount']),2)


# In[ ]:


df=df.drop(['PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS', 'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT', 'PRI.DISBURSED.AMOUNT','SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT'],axis=1)


# In[ ]:


df.columns


# ### NEW.ACCTS.IN.LAST.SIX.MONTHS

# In[ ]:


df['NEW.ACCTS.IN.LAST.SIX.MONTHS']


# In[ ]:


df['NEW.ACCTS.IN.LAST.SIX.MONTHS'].value_counts()


# ### NEW.ACCTS.IN.LAST.SIX.MONTHS

# In[ ]:


df['NEW.ACCTS.IN.LAST.SIX.MONTHS'].value_counts()


# In[ ]:


ct=pd.crosstab(df['NEW.ACCTS.IN.LAST.SIX.MONTHS'], df['loan_default'])
ct.plot.bar(stacked=True,figsize=(12,5))
plt.legend(labels=['Not Defaulted','Defaulted'])
plt.show()


# ### DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS

# In[ ]:


df['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS']


# In[ ]:


df['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS'].value_counts().plot(kind='bar')


# In[ ]:


ct=pd.crosstab(df['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS'], df['loan_default'])
ct.plot.bar(stacked=True,figsize=(12,5))
plt.legend(labels=['Not Defaulted','Defaulted'])
plt.show()


# ### AVERAGE.ACCT.AGE

# In[ ]:


df['AVERAGE.ACCT.AGE']


# In[ ]:


df['AVERAGE.ACCT.AGE']=df['AVERAGE.ACCT.AGE'].apply(lambda x:(re.sub('[a-z]','',x)).split())
df['AVERAGE.ACCT.AGE']=df['AVERAGE.ACCT.AGE'].apply(lambda x:int(x[0])*12+int(x[1]))


# In[ ]:


df['AVERAGE.ACCT.AGE']


# In[ ]:


plt.figure(figsize=(20,7))
sns.distplot(df[df['loan_default']==0]['AVERAGE.ACCT.AGE'],kde=False)
sns.distplot(df[df['loan_default']==1]['AVERAGE.ACCT.AGE'],kde=False)
plt.legend(labels=['Not-Defaulted','Defaulted'])
plt.show()


# ### CREDIT.HISTORY.LENGTH

# In[ ]:


df['CREDIT.HISTORY.LENGTH']


# In[ ]:


df['CREDIT.HISTORY.LENGTH']=df['CREDIT.HISTORY.LENGTH'].apply(lambda x:
                                                                   (re.sub('[a-z]','',x)).split())
df['CREDIT.HISTORY.LENGTH']=df['CREDIT.HISTORY.LENGTH'].apply(lambda x:
                                                                   int(x[0])*12+int(x[1]))


# In[ ]:


df['CREDIT.HISTORY.LENGTH']


# In[ ]:


plt.figure(figsize=(20,7))
sns.distplot(df[df['loan_default']==0]['CREDIT.HISTORY.LENGTH'],kde=False)
sns.distplot(df[df['loan_default']==1]['CREDIT.HISTORY.LENGTH'],kde=False)
plt.legend(labels=['Not-Defaulted','Defaulted'])
plt.show()


# ### NO.OF_INQUIRIES

# In[ ]:


df['NO.OF_INQUIRIES'].value_counts()


# In[ ]:


df['NO.OF_INQUIRIES'].value_counts().plot(kind='bar')


# In[ ]:


df['NO.OF_INQUIRIES'].describe()


# In[ ]:


plt.figure(figsize=(20,7))
sns.distplot(df[df['loan_default']==0]['NO.OF_INQUIRIES'],kde=False)
sns.distplot(df[df['loan_default']==1]['NO.OF_INQUIRIES'],kde=False)
plt.legend(labels=['Not-Defaulted','Defaulted'])
plt.show()


# In[ ]:


df.corr()


# ### loan_default (Target/Dependent Variable)

# In[ ]:


print(df['loan_default'].value_counts())
sns.countplot(df['loan_default'])
plt.show()


# In[ ]:


df['loan_default'].value_counts(normalize=True)


# In[ ]:


df


# In[ ]:


plt.figure(figsize=(20,10))
sns.scatterplot(x='CREDIT.HISTORY.LENGTH',y='disbursed_amount',style='loan_default',hue='loan_default',alpha=0.6,data=df)
plt.show()


# In[ ]:


plt.figure(figsize=(50,50))
sns.heatmap(df.corr(),linewidths=0.05,annot=True)
plt.show()


# In[ ]:


sns.scatterplot(x='CREDIT.HISTORY.LENGTH',y='disbursed_amount',hue='loan_default',data=df)


# In[ ]:


df=df.drop(['supplier_id','branch_id','Current_pincode_ID'],axis=1)


# ## Base Model

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)


# In[ ]:


ohe.fit_transform(df[['Employment.Type']])


# In[ ]:


print(ohe.categories_)


# In[ ]:


temp=pd.DataFrame(ohe.fit_transform(df[['Employment.Type']]),columns=ohe.categories_[0])


# In[ ]:


manu =pd.get_dummies(df['manufacturer_id'], columns=['manufacturer_id'], drop_first=False)
print(manu)


# In[ ]:


df=df.drop(['Employment.Type','manufacturer_id'],axis=1)


# In[ ]:


df=pd.concat([df,manu],axis=1)


# In[ ]:


df=pd.concat([df,temp],axis=1)


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


y=df['loan_default']
X=df.drop(['loan_default'],axis=1)


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report


# In[ ]:


classifiers = [
    LogisticRegression(),
    DecisionTreeClassifier(criterion='entropy',random_state=0),
    DecisionTreeClassifier(criterion='gini',random_state=0),
    RandomForestClassifier(n_estimators=100),
    GaussianNB(),
    BaggingClassifier(random_state=0),
    AdaBoostClassifier(),
    XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,
                  n_estimators=100, max_depth=3)
]


# In[ ]:


results = []
for item in classifiers:
    print(item,"\n")
    clf = item
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    #print(y_pred)
    #results.append(accuracy_score(y_test,y_pred))
    results.append(y_pred)
    print("accuracy train:",clf.score(X_train,y_train),"\n")
    print("accuracy test:",clf.score(X_test,y_test),"\n")
    print("precision :",precision_score(y_test,y_pred),"\n")
    print('Recall score:',recall_score(y_test,y_pred),'\n')
    print("f1 score:",f1_score(y_test,y_pred),"\n")
    from sklearn.metrics import classification_report
    print(classification_report(y_test,y_pred))
    print("-------------------------------------------------------------------------------------------------------")


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[ ]:


X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[ ]:


classifiers = [
    LogisticRegression(),
    DecisionTreeClassifier(criterion='entropy',random_state=0),
    DecisionTreeClassifier(criterion='gini',random_state=0),
    RandomForestClassifier(n_estimators=100),
    GaussianNB(),
    BaggingClassifier(random_state=0),
    AdaBoostClassifier(),
    XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,
                  n_estimators=100, max_depth=3)
]


# In[ ]:


results = []
for item in classifiers:
    print(item,"\n")
    clf = item
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    #print(y_pred)
    #results.append(accuracy_score(y_test,y_pred))
    results.append(y_pred)
    print("accuracy train:",clf.score(X_train,y_train),"\n")
    print("accuracy test:",clf.score(X_test,y_test),"\n")
    print("precision :",precision_score(y_test,y_pred),"\n")
    print('Recall score:',recall_score(y_test,y_pred),'\n')
    print("f1 score:",f1_score(y_test,y_pred),"\n")
    from sklearn.metrics import classification_report
    print(classification_report(y_test,y_pred))
    print("-------------------------------------------------------------------------------------------------------")


# ## SMOTE

# **SMOTE not working in Kaggle Notebook , ill find an alternate soon.**

# In[ ]:


get_ipython().system('pip install scikit-learn --upgrade')


# In[ ]:


from sklearn.utils import resample
from imblearn.over_sampling import SMOTE


# In[ ]:


y=df['loan_default']
X=df.drop(['loan_default'],axis=1)


# In[ ]:


print("X shape",X.shape)
print('y shape',y.shape)


# In[ ]:


sm = SMOTE(random_state=0)
X_smote,y_smote = sm.fit_sample(X,y)


# In[ ]:


print("X shape",X_smote.shape)
print('y shape',y_smote.shape)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_smote,y_smote,test_size=0.30,random_state=0)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[ ]:


X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[ ]:


classifiers = [
    LogisticRegression(),
    DecisionTreeClassifier(criterion='entropy',random_state=0),
    DecisionTreeClassifier(criterion='gini',random_state=0),
    RandomForestClassifier(n_estimators=100),
    GaussianNB(),
    BaggingClassifier(random_state=0),
    AdaBoostClassifier(),
    XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,
                  n_estimators=100, max_depth=3)
]


# In[ ]:


results = []
for item in classifiers:
    print(item,"\n")
    clf = item
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    #print(y_pred)
    #results.append(accuracy_score(y_test,y_pred))
    results.append(y_pred)
    print("accuracy train:",clf.score(X_train,y_train),"\n")
    print("accuracy test:",clf.score(X_test,y_test),"\n")
    print("precision :",precision_score(y_test,y_pred),"\n")
    print('Recall score:',recall_score(y_test,y_pred),'\n')
    print("f1 score:",f1_score(y_test,y_pred),"\n")
    from sklearn.metrics import classification_report
    print(classification_report(y_test,y_pred))
    print("-------------------------------------------------------------------------------------------------------")


# ## pca

# In[ ]:


from sklearn.decomposition import PCA
pca=PCA(32)


# In[ ]:


X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)


# In[ ]:


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(y=0.95, color='r', linestyle='-')
plt.show()


# In[ ]:


X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[ ]:


classifiers = [
    LogisticRegression(),
    DecisionTreeClassifier(criterion='entropy',random_state=0),
    DecisionTreeClassifier(criterion='gini',random_state=0),
    RandomForestClassifier(n_estimators=100),
    GaussianNB(),
    BaggingClassifier(random_state=0),
    AdaBoostClassifier(),
    XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,
                  n_estimators=100, max_depth=3)
]


# In[ ]:


results = []
for item in classifiers:
    print(item,"\n")
    clf = item
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    #print(y_pred)
    #results.append(accuracy_score(y_test,y_pred))
    results.append(y_pred)
    print("accuracy train:",clf.score(X_train,y_train),"\n")
    print("accuracy test:",clf.score(X_test,y_test),"\n")
    print("precision :",precision_score(y_test,y_pred),"\n")
    print('Recall score:',recall_score(y_test,y_pred),'\n')
    print("f1 score:",f1_score(y_test,y_pred),"\n")
    from sklearn.metrics import classification_report
    print(classification_report(y_test,y_pred))
    print("-------------------------------------------------------------------------------------------------------")


# ## TRYING DOWN SAMPLING/UNDERSAMPLING

# In[ ]:


not_default = df[df.loan_default==0]
default = df[df.loan_default==1]


# In[ ]:


not_default_downsampled = resample(not_default,
                                replace = True, # sample without replacement
                                n_samples = len(default), # match minority n
                                random_state = 0) 


# In[ ]:


downsampled = pd.concat([not_default_downsampled, default])


# In[ ]:


downsampled


# In[ ]:


downsampled['loan_default'].value_counts()


# In[ ]:


y=downsampled['loan_default']
X=downsampled.drop(['loan_default'],axis=1)


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_smote,y_smote,test_size=0.30,random_state=0)


# In[ ]:


X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[ ]:


results = []
for item in classifiers:
    print(item,"\n")
    clf = item
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    #print(y_pred)
    #results.append(accuracy_score(y_test,y_pred))
    results.append(y_pred)
    print("accuracy train:",clf.score(X_train,y_train),"\n")
    print("accuracy test:",clf.score(X_test,y_test),"\n")
    print("precision :",precision_score(y_test,y_pred),"\n")
    print('Recall score:',recall_score(y_test,y_pred),'\n')
    print("f1 score:",f1_score(y_test,y_pred),"\n")
    from sklearn.metrics import classification_report
    print(classification_report(y_test,y_pred))
    print("-------------------------------------------------------------------------------------------------------")


# In[ ]:





# In[ ]:





# In[ ]:




