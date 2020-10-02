#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="ticks")
# logistic regression and random forest 


# In[79]:


df = pd.read_csv('../input/sales_data.csv')


# In[80]:


df.info()


# In[81]:


df.head()


# In[82]:


# first lets look how many null values each column contains
df.count()/df.shape[0]


# In[83]:


df.dropna(subset=['education'],inplace=True)


# In[84]:


# fill the house_owner feature
df['house_owner'].value_counts().plot(kind='bar')


# In[85]:


# As most of the valuas are owner we will the missing values with owner 
df['house_owner']=df['house_owner'].fillna('Owner')
df['house_owner'].value_counts()


# In[86]:


# but as most of the values are missing in the marriage feature I think we cannot direclty fill the values of feature.
# As the age column filled 100 percent i will try to fill them with the help of age.
df['age'].value_counts()


# In[87]:


df['marriage'].value_counts()


# In[88]:


df.groupby('age')['marriage'].value_counts().plot(kind='bar')


# In[89]:


df.loc[(df['age']=='2_<=25'),'marriage']='Single'


# In[90]:


df['marriage']=df['marriage'].fillna('Married')


# In[91]:


# lets start analyzing with the data we have 
print(df['flag'].value_counts())
sns.countplot(x='flag',data=df)


# In[92]:


# there is no much differnce between the both  categories..
# That is there are almost equal number of people who bought the prodcut and also who didn't purchase it.
# lets seee how different feature are depending on the flag (target) variable.


# In[93]:


# Age Variable 
f,ax=plt.subplots(1,3,figsize=(16,6))
sns.countplot(x='gender',hue='flag',data=df,ax=ax[0])
sns.countplot(x='online',hue='flag',data=df,ax=ax[1])
sns.countplot(x='marriage',hue='flag',data=df,ax=ax[2])


# In[94]:


# most of the males purchased the product and unkown,gender didn't show much interest
# Customers who have online experince have bought the product where as most of the people who are married purchased the product


# As we can only the house_val good correlation with the flag variable so we will direclty work on those variables..

# In[95]:


f,ax=plt.subplots(1,2,figsize=(16,6))
sns.boxplot(x=df["house_val"],ax=ax[0])
sns.distplot(df['house_val'],ax=ax[1])


# In[96]:


# first we will remove some outliers..


# In[97]:


df['house_val'].describe()


# In[98]:


df['house_valbins']=pd.cut(df['house_val'],bins=5)


# In[99]:


# print(df.groupby('house_valbins')['flag'].value_counts())
# f,ax=plt.subplots(figsize=(16,6))
# sns.countplot(x='house_valbins',hue='flag',data=df)

df.groupby('house_valbins')['flag'].value_counts()


# In[100]:


#Increasing with the salary the people are purchasing the product..


# In[101]:


sns.countplot(x='occupation',hue='flag',data=df)
# sns.countplot(x='online',hue='flag',data=df,ax=ax[1])


# most of the professional are purchaing the product when compared with the other occupation...

# In[102]:


sns.factorplot(x='occupation',hue='flag',y='house_val',data=df)


# In each occupation if the house_val of the customer is high then the chances of purchasing the product is high

# In[103]:


sns.factorplot(x='occupation',hue='flag',y='car_prob',data=df)


# If the customer have a car then they are mostly likely to buy a car.

# In[104]:


sns.countplot(x='fam_income',hue='flag',data=df)


# In[105]:


print(df['child'].value_counts())
sns.countplot(x='child',hue='flag',data=df)


# In[106]:


df_new=df.copy()
df_new=df_new.drop(['house_val'],axis=1)


# In[107]:


# before going further we will try pick some most important variables
# To calculate the conditional probability of the features (Categorical)


from collections import Counter
import scipy.stats as ss
import math

def conditional_entropy(x,y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def theil_u(x,y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

theilu = pd.DataFrame(index=['flag'],columns=df_new.columns)
columns = df_new.columns
for j in range(0,len(columns)):
    u = theil_u(df_new['flag'].tolist(),df_new[columns[j]].tolist())
    theilu.loc[:,columns[j]] = u
theilu.fillna(value=np.nan,inplace=True)
plt.figure(figsize=(20,1))
sns.heatmap(theilu,annot=True,fmt='.2f')
plt.show()


# In[108]:


# get dummies
# df=df

df_model=df.copy()


# In[109]:


# logistic regression

X=pd.DataFrame(columns=['house_val'],data=df_model)
y=df_model['flag']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)




# In[111]:



from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[ ]:




