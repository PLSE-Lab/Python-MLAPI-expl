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


fraud['policy_number'].value_counts()[:10]


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

col=['months_as_customer','age','policy_deductable','policy_annual_premium','capital-gains', 'capital-loss','total_claim_amount','injury_claim', 'property_claim', 'vehicle_claim']

plt.figure(figsize=(14,12))
k=1
for i in col:
    plt.subplot(4,4,k)
    sns.distplot(df[i])
    k=k+1
plt.show()


# Almost all the data above are normally distributed.

# In[ ]:


df['umbrella_limit'].value_counts()


# In[ ]:


a=df[df['umbrella_limit']>0]


# In[ ]:


sns.distplot(a['umbrella_limit'])


# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='age',y='months_as_customer',hue='fraud_reported',data=df)
plt.grid(True)
plt.show()


# There is no pattern that justifies if the customers are with the company for more years and are claiming the fraud claims.

# In[ ]:


plt.figure(figsize=(12,10))
sns.countplot(x=df['age'],hue='fraud_reported',data=df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


a=df[df['fraud_reported']=='Y']
a[['policy_number','insured_occupation','insured_education_level','total_claim_amount']].sort_values('total_claim_amount',ascending=False)[:20]


# There are 20 Policies that have frauds and are listed above and the Policy_Number ='217938' has claimed '$112320' amount.

# In[ ]:


a['insured_occupation'].value_counts()


# In[ ]:


a['insured_education_level'].value_counts()


# People whose education level is 'JD' and working as 'exec-manegerial' have more fraud claims.

# In[ ]:


sns.set(style="darkgrid")

plt.figure(figsize=(20,12))
plt.subplot(1,2,1)
plt.title("Count plot for Fraud transactions 'Y' wrt Education level",fontsize=20)
sns.countplot('insured_education_level',data=a)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.subplot(1,2,2)
plt.title("Count plot for Fraud transactions 'Y' wrt Occupation",fontsize=20)
sns.countplot('insured_occupation',data=a)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# People with education level as 'JD' are much involved in fraud transactions.
# People with occupation as 'exec-managerial' are much involved in fraud transactions.

# In[ ]:


#Looking at below claim_amount:
a_claims=pd.pivot_table(a,values='total_claim_amount',index=['insured_occupation','insured_education_level']).sort_values('total_claim_amount',ascending=False)

cm=sns.light_palette("black", as_cmap=True)
a_claims.style.background_gradient(cmap=cm)


# People from occupation sector as Protective-services and education level as JD have highest fraud claimed amount of 87,890$.

# In[ ]:


plt.figure(figsize=(16,6))
plt.title("Count plot for Fraud transactionn 'Y' wrt Hobbies",fontsize=20)
sns.countplot('insured_hobbies',data=a)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# People who have the hobbies of 'chess' and 'cross-fit' are tend to do the fraud claims.

# In[ ]:


plt.figure(figsize=(10,8))
plt.title("Bar plot for Fraud transactions wrt Gender",fontsize=20)
sns.barplot(x='insured_sex',y='total_claim_amount',hue='fraud_reported',data=df)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# Both Male and Female are tend to do the fraud claims similarly.

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


df[['incident_date','incident_type','collision_type','incident_severity','authorities_contacted','incident_state','incident_city',
   'incident_location','incident_hour_of_the_day','number_of_vehicles_involved']]


# In[ ]:


pd.pivot_table(a,values=['number_of_vehicles_involved','total_claim_amount','vehicle_claim','incident_hour_of_the_day'],index=['incident_type','collision_type']).sort_values('vehicle_claim',ascending=False)


# For Auto claims, single vehicle collision have claimed the highesht amount in the current data set.

# In[ ]:


df['number_of_vehicles_involved'].value_counts()


# In[ ]:


df['incident_type'].value_counts()


# In[ ]:


df['collision_type'].value_counts()


# In[ ]:


#For collision_type there are junk values as '?' that are nan values and has to be replaced/removed:
#Let's see what kind of incident we have for the junk (?) values:
coll=a[['incident_type','collision_type']]


# In[ ]:


res=coll.loc[coll['collision_type']=='?']


# In[ ]:


res['incident_type'].value_counts()


# 8 Parked Cars and 8 Vehicle Theft cars have fraud claimed.

# In[ ]:


coll_df=df[['incident_type','collision_type']]


# In[ ]:


res_df=coll_df.loc[coll_df['collision_type']=='?']


# In[ ]:


res_df['incident_type'].value_counts()


# Cars that are stolen (Vehicle Theft) and cars that are parked (Parked Cars) are marked as '?', so we shall replace with these values with either NA or No collision

# In[ ]:


df['collision_type']=df['collision_type'].replace("?","Not Applicable")


# In[ ]:


df['collision_type'].value_counts()


# In[ ]:


a['incident_city'].value_counts()


# People from the city - 'Arlington' have more auto related incidents that are fraud claimed.

# In[ ]:


pd.pivot_table(a,values=['total_claim_amount','vehicle_claim'],index=['incident_state','incident_city']).sort_values('total_claim_amount',ascending=False)[:20]


# Riverwood city from the state - SC have claimed maximum number of frauds for auto insurance.

# In[ ]:


df.columns


# In[ ]:


a.loc[(a['property_claim'] == 0.0)&(a['vehicle_claim'] != 0.0)]


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


plt.figure(figsize=(12,10))
sns.countplot('auto_make',data=a)
plt.title("Count of Auto make which have Fraud claim",fontsize=20)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# Auto make of the company's Mercedes, Ford, Audi are having highest fraud claims.Chevrolet, Dodge, BMW are as wellhave the fraud claims.

# In[ ]:


pd.pivot_table(a,values=['vehicle_claim'],index=['auto_make','auto_model','policy_number']).sort_values('vehicle_claim',ascending=False)[:20]


# Person having Policy number - 217938 on the vehicle - Suburu-Impreza has claimed an amount of $77760 which is identified as fraud.

# In[ ]:


df.columns


# In[ ]:


a['bodily_injuries'].value_counts()


# In[ ]:


a['police_report_available'].value_counts()


# Even when the police report is available there policies that have fraud claimed.

# In[ ]:


df['police_report_available']=df['police_report_available'].replace("?","Unknown")


# In[ ]:


df['police_report_available'].value_counts()


# In[ ]:


df.columns


# In[ ]:


df['umbrella_limit'].value_counts()


# In[ ]:


df['umbrella_limit']=df['umbrella_limit'].replace(-1000000 ,0)


# In[ ]:


df['umbrella_limit'].value_counts()


# In[ ]:


df['policy_csl'].value_counts()


# In[ ]:


df2=df


# In[ ]:


df2.columns


# In[ ]:


df2=df2.drop(['policy_number', 'policy_bind_date', 'insured_zip','incident_date','authorities_contacted','profit','auto_make', 'auto_model'],axis=1)


# In[ ]:


df2.columns


# In[ ]:


df2=pd.get_dummies(df2,columns=['policy_state','policy_csl','insured_sex','insured_education_level', 'insured_occupation',
       'insured_hobbies', 'insured_relationship','incident_type', 'collision_type', 'incident_severity',
       'incident_state', 'incident_city', 'incident_location','property_damage','police_report_available'],drop_first=True)


# In[ ]:


df2.shape


# In[ ]:


df2.columns


# In[ ]:


x=df2.drop(['fraud_reported'],axis=1)


# In[ ]:


y=df2['fraud_reported']


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[ ]:


x_scale=sc.fit_transform(x)


# In[ ]:


from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
x_scaled=pca.fit_transform(x_scale)


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.3)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf=RandomForestClassifier(criterion='entropy')


# In[ ]:


rf.fit(x_train,y_train)


# In[ ]:


y_pred=rf.predict(x_test)


# In[ ]:


rf.score(x_test,y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_pred,y_test)
sns.heatmap(cm,annot=True)


# In[ ]:


clf=classification_report(y_pred,y_test)
print(clf)

