#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df=pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
df.head(10)


# In[ ]:


df.tail(10)


# In[ ]:


#df['arriavl_date']=pd.to_datetime(df['arrival_date_year']+df['arrival_date_month']+df['arrival_date_day_of_month'])


# In[ ]:


print('shape of dataset',df.shape)
print('\n')
print('size of dataset',df.size)


# In[ ]:


df.info()


# In[ ]:


df.describe().T


# In[ ]:


df.describe(include='object').T


# ### EDA

# In[ ]:


df.isna().sum()


# In[ ]:


cat=df.select_dtypes(include='object').columns
cat


# In[ ]:


df=df.drop(['agent','company','reservation_status_date'],axis=1)


# In[ ]:


df['country'].mode()


# In[ ]:


df['country']=df['country'].replace(np.nan,'PRT')


# In[ ]:


df.isnull().sum()


# In[ ]:


sns.countplot(df['hotel'])
plt.show()


# In[ ]:


plt.figure(figsize=(15 ,10 ))
sns.countplot(df['arrival_date_month'])
plt.show()


# In[ ]:



sns.countplot(df['is_canceled'])
plt.show()


# In[ ]:


df.is_canceled.value_counts()


# In[ ]:


plt.figure(figsize=(15 ,10 ))
sns.countplot(df['meal'])
plt.show()


# In[ ]:


plt.figure(figsize=(15 ,10 ))
sns.countplot(df['market_segment'])
plt.show()


# In[ ]:


plt.figure(figsize=(15 ,10 ))
sns.countplot(df['distribution_channel'])
plt.show()


# In[ ]:


plt.figure(figsize=(15 ,10 ))
sns.countplot(df['reserved_room_type'])
plt.show()


# In[ ]:


plt.figure(figsize=(15 ,10 ))
sns.countplot(df['assigned_room_type'])
plt.show()


# In[ ]:



sns.countplot(df['deposit_type'])
plt.show()


# In[ ]:





# In[ ]:



sns.countplot(df['customer_type'])
plt.show()


# In[ ]:


plt.figure(figsize=(15 ,10 ))
sns.countplot(df['reservation_status'])
plt.show()


# In[ ]:


plt.figure(figsize=(15 ,10 ))
sns.barplot(df['reservation_status'],df['arrival_date_year'],)
plt.show()


# In[ ]:





# In[ ]:


df.corr()


# In[ ]:


plt.figure(figsize=(15 ,10 ))
sns.barplot(df['arrival_date_year'],df['previous_cancellations'])
plt.show()


# In[ ]:


plt.figure(figsize=(15 ,10 ))
sns.barplot(df['arrival_date_year'],df['previous_bookings_not_canceled'])
plt.show()


# In[ ]:


plt.figure(figsize=(15 ,10 ))
sns.barplot(df['arrival_date_month'],df['previous_cancellations'])
plt.show()


# In[ ]:


plt.figure(figsize=(15 ,10 ))
sns.barplot(df['arrival_date_month'],df['previous_bookings_not_canceled'])
plt.show()


# In[ ]:


plt.figure(figsize=(15 ,10 ))
sns.barplot(df['arrival_date_month'],df['is_canceled'])
plt.show()


# In[ ]:


plt.figure(figsize=(15 ,10 ))
sns.barplot(df['arrival_date_year'],df['is_canceled'])
plt.show()


# ### converting categorical to numerical

# In[ ]:


cat


# In[ ]:


df=pd.get_dummies(df,prefix=['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
       'distribution_channel', 'reserved_room_type', 'assigned_room_type',
       'deposit_type', 'customer_type', 'reservation_status'])


# In[ ]:


df.head()


# In[ ]:


print('shape of dataset',df.shape)
print('\n')
print('size of dataset',df.size)


# In[ ]:


for i in df.columns:
    if (df[i].isnull().sum())!=0:
        print("{} {}".format(i, df[i].isnull().sum()))


# In[ ]:


df.children.mode()


# In[ ]:


df['children']=df['children'].replace(np.nan,'0')


# In[ ]:


df['children']=df['children'].astype('int')


# In[ ]:


df.corr()


# In[ ]:


plt.figure(figsize=(25 ,20 ))
sns.heatmap(df.corr())


# In[ ]:





# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[ ]:


X=df.drop('is_canceled',axis=1 )
y=df['is_canceled']


# In[ ]:


LR=LogisticRegression()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=2)


# In[ ]:


LR.fit(X_train,y_train)


# In[ ]:


y_pred = LR.predict(X_test)
y_pred


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test, y_pred)
accuracy 


# In[ ]:





# ### on ols

# In[ ]:


import statsmodels.api as sm
X=df.drop('is_canceled',axis=1 )
y=df['is_canceled']


# In[ ]:


Xc=sm.add_constant(X)
model=sm.OLS(y,X).fit()
model.summary()


# In[ ]:


cols = X.columns.tolist()

while len(cols)>0:
    
    x_1 = X[cols]
    model = sm.OLS(y, x_1).fit()
    p = pd.Series(model.pvalues.values, index = cols)
    pmax = max(p)
    feature_max_p = p.idxmax()
    
    if(pmax > 0.05):
        cols.remove(feature_max_p)
    else:
        break


# In[ ]:


print(len(cols))
print(cols)


# In[ ]:


X=df[cols]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=2)


# In[ ]:


LR.fit(X_train,y_train)


# In[ ]:


y_pred1=LR.predict(X_test)
y_pred1


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred1)
confusion_matrix


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test, y_pred1)
accuracy 


# In[ ]:





# In[ ]:




