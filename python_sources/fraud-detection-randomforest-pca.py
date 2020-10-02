#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df=pd.read_csv("/kaggle/input/insurance-claim/insurance_claims.csv")


# In[ ]:


df.head(5)


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.describe().transpose()


# In[ ]:


df.columns


# In[ ]:


df['fraud_reported'].value_counts()


# In[ ]:


fraud=df[df['fraud_reported']=='Y']


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

col=['months_as_customer','age','policy_deductable','policy_annual_premium','capital-gains','capital-loss','total_claim_amount','injury_claim','property_claim','vehicle_claim']


plt.figure(figsize=(16,14))
k=1
for i in col :
    plt.subplot(4,4,k)
    sns.distplot(df[i])
    k=k+1
plt.show()


# All the above data are almost normally distributed

# In[ ]:


df['umbrella_limit'].value_counts()


# In[ ]:


a=df[df['umbrella_limit'] >0]


# In[ ]:


sns.distplot(a['umbrella_limit'])


# In[ ]:


df.columns


# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='age', y='months_as_customer',hue='fraud_reported',data=df)
plt.grid(True)
plt.show()


# We do not have any pattern which justifies if customers with more years with the company are claiming fraud insurance.

# In[ ]:


plt.figure(figsize=(12,10))
sns.countplot(x='age',hue='fraud_reported',data=df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


a=df[df['fraud_reported'] == 'Y']

a[['policy_number','insured_occupation','insured_education_level','total_claim_amount']].sort_values('total_claim_amount',ascending=False)[:20]


# We have top 20 fraud claims with policy number and their occupation listed
# Policy number 217938 has claimed highest amount of 112320$ and is working as craft-repair

# In[ ]:


a['insured_education_level'].value_counts()
a['insured_occupation'].value_counts()


# People who are working as exec-manager has claimed more fraud transactions

# In[ ]:


sns.set(style="darkgrid")

plt.figure(figsize=(20,12))
plt.subplot(1,2,1)
plt.title("Count plot for Fraud transaction 'Y' wrt Education level",fontsize=20)
sns.countplot('insured_education_level',data=a)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.subplot(1,2,2)
plt.title("Count plot for Fraud transaction 'Y' wrt Occupation",fontsize=20)
sns.countplot('insured_occupation',data=a)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# People with occupation as Exec Manager seems to be doing more fraud transactions and people with JD level of education are also involved in more fraud transactions. Comparitively people with less education are claiming more fraud claims.

# In[ ]:


#Looking at below claims
a_claims=pd.pivot_table(a,values='total_claim_amount',index=['insured_occupation','insured_education_level']).sort_values('total_claim_amount',ascending=False)

cm = sns.light_palette("blue", as_cmap=True)
a_claims.style.background_gradient(cmap=cm)


# People from occupation sector Protective-services and education level of JD has highest fraud claimed amount of 87,890$

# In[ ]:


plt.figure(figsize=(16,6))
plt.title("Count plot for Fraud transaction 'Y' wrt Hobbies",fontsize=20)
sns.countplot('insured_hobbies',data=a)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# People who play more chess have claimed more fraud followed by crossfit

# In[ ]:


plt.figure(figsize=(14,5))
plt.title("Bar plot for Fraud transaction wrt Gender",fontsize=20)
sns.barplot(x='insured_sex',y='total_claim_amount',hue='fraud_reported',data=df)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# Both Male and female have caimed same amount which are fraud

# In[ ]:


a['insured_relationship'].value_counts()


# In[ ]:


profit=df['capital-gains']-df['capital-loss']
df1=df
df1['profit']=profit


# In[ ]:


df1.columns


# In[ ]:


df[['policy_number','profit']].sort_values('profit',ascending=False)[0:20]


# In[ ]:


df[['incident_date', 'incident_type', 'collision_type', 'incident_severity',
       'authorities_contacted', 'incident_state', 'incident_city',
       'incident_location', 'incident_hour_of_the_day',
       'number_of_vehicles_involved']]


# In[ ]:


pd.pivot_table(a,values=['number_of_vehicles_involved','total_claim_amount','vehicle_claim','incident_hour_of_the_day'],index=['incident_type','collision_type']).sort_values('vehicle_claim',ascending=False)


# For auto claims, single vehicle side collision have claimed highest

# In[ ]:


df['number_of_vehicles_involved'].value_counts()


# In[ ]:


df['incident_type'].value_counts()


# In[ ]:


df['collision_type'].value_counts()


# In[ ]:


# For collision type we have few ? values, which are nan values and has to be replaced/removed
#Let us see for what kind of incident we have for value ?
coll=a[['incident_type','collision_type']]


# In[ ]:


res=coll.loc[coll['collision_type']=='?']


# In[ ]:


res['incident_type'].value_counts()


# 8 Cars which are parked and 8 cars which are theft have claimed fraud.

# In[ ]:


coll_df=df[['incident_type','collision_type']]


# In[ ]:


res_df=coll_df.loc[coll_df['collision_type']=='?']


# In[ ]:


res_df['incident_type'].value_counts()


# Cars which are theft and Parked are marked as ?, we can replace them wiht either NA or No collisison

# In[ ]:


df['collision_type']=df['collision_type'].replace("?","Not Applicable")


# In[ ]:


df['collision_type'].value_counts()


# In[ ]:


a['incident_city'].value_counts()


# People from Arlington have more auto related incidents which are claimed to be fraud

# In[ ]:


pd.pivot_table(a,values=['total_claim_amount','vehicle_claim'],index=['incident_state','incident_city']).sort_values('total_claim_amount',ascending=False)[:20]


# Riverwood city from SC state have claimed maximum amount of fraud for auto insurance

# In[ ]:


df.columns


# In[ ]:


a.loc[(a['property_claim'] == 0.0 )&(a['vehicle_claim'] != 0.0 )]


# In[ ]:



plt.figure(figsize=(16,8))
plt.subplot(2,1,1)
plt.title("Count plot for Auto make",fontsize=20)
sns.countplot('auto_make',data=df)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.subplot(2,1,2)
plt.title("Count plot for Auto model",fontsize=20)
sns.countplot('auto_model',data=df)
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


a['auto_make'].value_counts()


# In[ ]:


plt.figure(figsize=(16,8))
plt.title("Count plot of Auto make which have Fraud claim",fontsize=20)
sns.countplot('auto_make',data=a)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# Auto make of 'Ford' and 'Mercedes' are having highest Fraud claim, 'Chevorlet' and 'Audi' also seems to be claimed equally having Fraud claims 
# 

# In[ ]:


pd.pivot_table(a,values=['vehicle_claim'],index=['auto_model','auto_make','policy_number']).sort_values('vehicle_claim',ascending=False)[0:10]


# Policy# 217938 who have Impreza-Suburu has claimed amount of 77,760$ and is identified as Fraud claim

# In[ ]:


df.columns


# In[ ]:


a['bodily_injuries'].value_counts()


# In[ ]:


a['police_report_available'].value_counts()


# Even for the policies which have police report have done fraud claims

# In[ ]:


df['police_report_available']=df['police_report_available'].replace("?","Unknown")


# In[ ]:


df.columns


# In[ ]:


df['umbrella_limit']=df['umbrella_limit'].replace(-1000000,0)


# In[ ]:


df2=df


# In[ ]:


df2=df2.drop(['policy_number','policy_bind_date','insured_zip','incident_date','authorities_contacted','profit','auto_make','auto_model'],axis=1)


# In[ ]:


df2.columns


# In[ ]:


df2=pd.get_dummies(df2,columns=['policy_state','policy_csl','insured_sex','insured_education_level','insured_occupation',
                                'insured_hobbies','insured_relationship','incident_type','collision_type','incident_severity','incident_state',
                                'incident_city','incident_location','property_damage','police_report_available'],drop_first=True)


# In[ ]:


df2.shape


# In[ ]:


df2.columns


# In[ ]:


x=df2.drop(['fraud_reported'],axis=1)


# In[ ]:


y=df2['fraud_reported']


# In[ ]:


from imblearn.over_sampling import SMOTE

x_upsample, y_upsample  = SMOTE().fit_resample(x, y)

print(x_upsample.shape)
print(y_upsample.shape)


# In[ ]:


y_upsample.value_counts()


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[ ]:


x_scale=sc.fit_transform(x_upsample)


# In[ ]:


from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
x_scaled=pca.fit_transform(x_scale)


# In[ ]:


r=pca.explained_variance_ratio_


# In[ ]:


np.sum(r)


# In[ ]:


len(r)


# 1039 features are converted into 944 principal components which constitutes 95.02% variance.
# 
# The explained variance tells how much information (variance) can be attributed to each of the principal components

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y_upsample,test_size=0.3)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf=RandomForestClassifier()


# In[ ]:


#Hyperparamenter tuning
from sklearn.model_selection import GridSearchCV
parameters={'criterion':['gini','entropy'], 'max_depth': np.arange(1,30)}


# In[ ]:


grid=GridSearchCV(rf,parameters)


# In[ ]:


grid.fit(x_train,y_train)


# In[ ]:


model=grid.best_estimator_


# In[ ]:


grid.best_score_


# In[ ]:


y_pred=model.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report

cm=confusion_matrix(y_pred,y_test)
sns.heatmap(cm,annot=True)


# In[ ]:


clf=classification_report(y_pred,y_test)
print(clf)


# In[ ]:


from sklearn.model_selection import cross_val_score
cross_val=cross_val_score(model,x_train,y_train,cv=10)


# In[ ]:


print(cross_val)


# In[ ]:


print("Acccuracy of the model is :", np.mean(cross_val))

