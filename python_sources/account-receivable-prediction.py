#!/usr/bin/env python
# coding: utf-8

# **Context**:
# 
# The biggest challenge of Factoring is to predict if and when invoices will be paid. The factor provides funds against this future payment to the business by buying their invoice. The factor then collects the payment and charges their interest rate. If the invoice isn't paid, the factor loses their advanced funds. Try using this data set for predicting when payments will be made.**

# Accounts Receivable
# Understand the factors of successful collection efforts. You can Predict which customers will pay fastest and recover more money and improve collections efficiency.

# In[ ]:


import math
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
#import statsmodels.api as sm


# #### Get the Data

# In[ ]:



df_receivable = pd.read_csv('../input/finance-factoring-ibm-late-payment-histories/WA_Fn-UseC_-Accounts-Receivable.csv')
#df.dropna(inplace=True)
df_receivable.info()


# In[ ]:


df_receivable.head(5)


# In[ ]:


df_receivable['InvoiceDate']= pd.to_datetime(df_receivable.InvoiceDate)


# generate one more variable 'Late' using another varable called 'DaysLate'

# In[ ]:


df_receivable['Late'] = df_receivable['DaysLate'].apply(lambda x: 1 if x >0 else 0)


# In[ ]:


df_receivable.head(3)


# Generate a rolling count of the amount of late payments for each customer

# In[ ]:


df_receivable['countlate']=df_receivable.Late.eq(1).groupby(df_receivable.customerID).apply(
    lambda x : x.cumsum().shift().fillna(0)).astype(int)


# In[ ]:


df_receivable.head(3)


# In[ ]:


df_receivable.info()


# In[ ]:


df_receivable.describe()


# In[ ]:


temp = pd.DataFrame(df_receivable.groupby(['countryCode'], axis=0, as_index=False)['DaysLate'].mean())
plt.figure(figsize=(10,6))
sns.barplot(x="countryCode", y="DaysLate",data=temp,linewidth=2.5, facecolor=(1, 1, 1, 0),
                 errcolor=".4", edgecolor="red")


# Identified that country code 818 has maxiimum lateday and minimum late days for 391.<br>
# Check the details of categorical features

# In[ ]:


df_receivable.describe(include=np.object)


# Count of categorical features

# In[ ]:


print(pd.crosstab(index=df_receivable["PaperlessBill"], columns="count"))
print(pd.crosstab(index=df_receivable["countryCode"], columns="count"))
print(pd.crosstab(index=df_receivable["Late"], columns="count"))


# Checking for the customer who is late to pay

# In[ ]:


df_receivable.head(3)


# In[ ]:


customer_late =pd.crosstab(index=df_receivable["customerID"], columns=df_receivable['Late'])
customer_late.sort_values(by=[1], ascending = False)


# In[ ]:


df1 = df_receivable[df_receivable['DaysLate']>0].copy()


# In[ ]:


df2 = pd.DataFrame(df1.groupby(['customerID'], axis=0, as_index=False)['DaysLate'].count())


# In[ ]:


df2.columns = (['customerID','repeatCust'])


# In[ ]:


df3 = pd.merge(df_receivable, df2, how='left', on='customerID')


# In[ ]:


df3['repeatCust'].fillna(0, inplace=True)


# In[ ]:


df_receivable = df3


# In[ ]:



temp = pd.DataFrame(df_receivable.groupby(['repeatCust'], axis=0, as_index=False)['DaysLate'].mean())
plt.figure(figsize=(14,7))
sns.barplot(x="repeatCust", y="DaysLate",data=temp,color='olive')


# In[ ]:


def func_IA (x):
    if x>60: return "b. more than 60"
    else: return "a. less than 60"
df_receivable['InvoiceAmount_bin'] = df_receivable['InvoiceAmount'].apply(func_IA)


# In[ ]:


df_receivable.head(3)


# In[ ]:


temp = pd.DataFrame(df_receivable.groupby(['InvoiceAmount_bin'], axis=0, as_index=False)['DaysLate'].mean())


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x="InvoiceAmount_bin", y="DaysLate",data=temp,color='purple')


# **
# Generate more features and map the some of the categorical variables to integers**<br>
# Map some of the categorical variables to integers. It is also helpful to generate some more insights about a customer given the data. For example if the order occurs at the end of the year is a company more likely to pay on time?

# In[ ]:


df_receivable['Disputed'] = df_receivable['Disputed'].map({'No':0,'Yes':1})
df_receivable['PaperlessBill'] = df_receivable['PaperlessBill'].map({'Paper': 0,'Electronic': 1})


# In[ ]:


df_receivable['InvoiceQuarter']= pd.to_datetime(df_receivable['InvoiceDate']).dt.quarter


# In[ ]:


df_receivable.head(3)


# Check for relation to late with  other variable to

# In[ ]:


plt.figure(figsize=(10,8))

ax = sns.countplot(df_receivable['countryCode'],hue=df_receivable['Late'],palette="YlGn")


# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(df_receivable['InvoiceQuarter'],hue=df_receivable['Late'],palette='bright')


# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(df_receivable['PaperlessBill'],hue=df_receivable['Late'],palette='YlOrRd')


# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(df_receivable['Disputed'],hue=df_receivable['Late'],palette='BuGn')


# 
# ### Distributions of Invoice Amounts and Days to settle
# It may be useful to understand the distribution of some variables. This can be helpful if we wish to know within reasonable assumptions what our confidence intervals are for payments or how long it takes for a customer to settle.

# In[ ]:


plt.figure(figsize=(8,8))
plt.figure(1)
sns.distplot(df_receivable['InvoiceAmount'],color='green')
plt.figure(figsize=(8,8))
plt.figure(2)
sns.distplot(df_receivable['DaysToSettle'],color='blue')


# 
# Finally, label customers with integers for processing in models

# In[ ]:


labels = df_receivable['customerID'].astype('category').cat.categories.tolist()


# In[ ]:


replace_map_comp = {'customerID' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}


# In[ ]:


#replace the customerID with Integers
df_receivable.replace(replace_map_comp, inplace=True)


# In[ ]:


df_receivable.head(3)


# 
# ## Train a Model to Predict if a Payment will be late

# In[ ]:


df_receivable.head(3)


# In[ ]:


corremat = df_receivable.corr()
plt.figure(figsize=(10,10))
g= sns.heatmap(df_receivable.corr(),annot=True,cmap='viridis',linewidths=.5)


# In[ ]:


corremat


# In[ ]:


corremat.columns


# In[ ]:


cat_feats = ['InvoiceAmount_bin']
final_data = pd.get_dummies(df_receivable,columns=cat_feats,drop_first=True)


# In[ ]:


final_data.head(3)


# In[ ]:


features=['countryCode', 'customerID', 'InvoiceAmount',
       'Disputed', 'PaperlessBill','repeatCust','Late', 'DaysToSettle',
       'countlate']


# In[ ]:


features


# In[ ]:


X = final_data[features]
y = final_data['DaysLate']


# In[ ]:


y.head(5)


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)


# In[ ]:


linear=LinearRegression()


# In[ ]:


linear.fit(X_train,y_train)


# In[ ]:


# linear Regression

y_pred1 = linear.predict(X_test)


# In[ ]:


#Checking the accuracy
linear_accuracy = round(linear.score(X_train,y_train)*100,2)
print(round(linear_accuracy,2),'%')


# In[ ]:


from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,y_pred)


# In[ ]:


GBR_model = GradientBoostingRegressor(n_estimators=100, max_depth=4)


# In[ ]:


#Fit
GBR_model.fit(X_train, y_train)


# In[ ]:


y_pred2 = GBR_model.predict(X_test)


# In[ ]:


#Checking the accuracy
GBR_model_accuracy = round(GBR_model.score(X_train,y_train)*100,2)
print(round(GBR_model_accuracy,2),'%')


# In[ ]:


mean_squared_error(y_test,y_pred2)


# In[ ]:


random_model = RandomForestRegressor(n_estimators=1000)


# In[ ]:


#Fit
random_model.fit(X_train, y_train)


# In[ ]:


y_pred3 = random_model.predict(X_test)


# In[ ]:


#Checking the accuracy
random_model_accuracy = round(random_model.score(X_train,y_train)*100,2)
print(round(random_model_accuracy,2),'%')


# In[ ]:


from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,y_pred3)


# The ideal value for Mean squared error is zero.GBR model is giving high accuracy.

# In[ ]:


y = pd.concat([y_test,pd.DataFrame(y_pred2)],axis=1)
y.columns = ('act','pred')

def act_decile (x):
    if x == 0: return "a. 0"
    elif x <= 2: return "b. (0-2] days"
    elif x <= 4: return "b. (2-4] days"
    elif x <= 6: return "c. (4-6] days"
    elif x <= 8: return "c. (6-8] days"
    elif x <= 10: return "c. (8-10] days"
    else: return "d. more (10-) days"
y['act_bin'] = y['act'].apply(act_decile)

temp = pd.DataFrame(y.groupby(['act_bin'], axis=0, as_index=False)['act','pred'].mean())
temp.index = temp['act_bin']
temp.plot(marker='o',figsize=(12,6))


# In[ ]:




