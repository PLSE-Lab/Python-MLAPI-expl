#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# # Importing Data

# In[ ]:


df = pd.read_csv('../input/insurance.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# # Data Cleaning

# We have integers and object types in our dataset..

# In[ ]:


df.isnull().sum()


# "here we observe that we don't have any missing values." 

# In[ ]:


sns.heatmap(df.isnull(),cmap="YlGnBu")


# In[ ]:


df.dtypes


# In[ ]:


df[['sex','smoker','region']].head()


# we have 3 categorical featurs in our dataset.so we need to encode for better results

# ##  Importing LabelEncoder for encoding categorical feaures

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le = LabelEncoder()
le.fit(df.sex.drop_duplicates())
df.sex = le.transform(df.sex)


# In[ ]:


df['sex'].head()


# In[ ]:


le.fit(df.smoker.drop_duplicates())


# In[ ]:


df.smoker=le.transform(df.smoker)


# In[ ]:


le.fit(df.region.drop_duplicates())


# In[ ]:


df.region = le.transform(df.region)


# In[ ]:


df.head()


# Now its all clear and our data is ready to fit a Machine learning model.

# In[ ]:


df.dtypes


# #### We need to find a better features for getting more accuracy and performance of the mdel. so we find correlation between the independent and dependent variavles.

# # Data analysis and visualization

# In[ ]:


df.corr()


# In[ ]:


sns.heatmap(df.corr(),cmap="YlGnBu")


# #### We observe that  smoker and age are much effects the charges

# In[ ]:


corr_matrix = df.corr()


# In[ ]:


corr_matrix['charges'].sort_values(ascending=False)


# In[ ]:


sns.countplot(data=df,x='smoker',hue='sex')


# In[ ]:


sns.countplot(data=df,x='smoker')


# In[ ]:


sns.countplot(data=df,x='region')


# In[ ]:


sns.countplot(data=df,x='sex')


# In[ ]:


sns.countplot(data=df,x='smoker',hue='sex')


# In[ ]:


df['charges'].hist()


# In[ ]:


df['charges'].min()


# In[ ]:


df.charges.max()


# Observe the minimum and maximum charges..

# In[ ]:


f = plt.figure(figsize=(12,7))
ax = f.add_subplot(121)
sns.distplot(df[df['smoker']==1]['charges'],color='g',ax=ax)
plt.title('Distrubution of  charges for Smokers')

ax = f.add_subplot(122)
sns.distplot(df[df['smoker']==0]['charges'],color='b',ax=ax)
plt.title('Distrubution of  charges for Non-Smokers')


# In[ ]:


df[df['smoker']==0]['charges'].max()


# In[ ]:





# In[ ]:


df[df['smoker']==0]['charges'].min()


# In[ ]:


df[df['smoker']==1]['charges'].min()


# In[ ]:


df[df['smoker']==1]['charges'].max()


# In[ ]:


sns.catplot(x='sex',y='charges',hue='smoker',kind='violin',data=df)


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Box plot for charges of women")
sns.boxplot(y="smoker", x="charges", data =  df[(df.sex == 1)] , orient="h", palette = 'magma')


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Box plot for charges of women")
sns.boxplot(y="smoker", x="charges", data =  df[(df.sex == 0)] , orient="h", palette = 'magma')


# In[ ]:


df.age.plot(kind='hist')


# In[ ]:


df['age'].min()


# In[ ]:


df['age'].max()


# In[ ]:


plt.figure(figsize=(12,8))
sns.catplot(x='smoker',kind='count',hue='sex',data=df[(df.age<=23)])


# In[ ]:


df[(df['smoker']==0) & (df['age']<=23) &(df['sex']==0)].count()


# In[ ]:



sns.jointplot(x="age", y="charges", data = df[(df.smoker == 0)],kind="kde", color="m")


# In[ ]:


sns.jointplot(x="age", y="charges", data = df[(df.smoker == 1)],kind="kde", color="g")


# In[ ]:





# In[ ]:


sns.distplot(df['bmi'])


# In[ ]:


plt.figure(figsize=(14,8))
plt.title('Distribution of charges for patients with BMI greater than 30')
ax = sns.distplot(df[df.bmi>=30]['charges'],color='m')


# In[ ]:


plt.figure(figsize=(14,8))
plt.title('Distribution of charges for patients with BMI lesser than 30')
ax = sns.distplot(df[df.bmi<30]['charges'],color='g')


# In[ ]:


sns.jointplot(x="bmi", y="charges", data = df,kind="kde", color="r")


# In[ ]:





# In[ ]:


sns.catplot(x="children", kind="count", data=df, size = 6)


# In[ ]:


sns.catplot(x="children", kind="count", hue='smoker',data=df, size = 6)


# In[ ]:





# In[ ]:


X = df.drop(['charges'],axis=1)
y= df['charges']


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)


# In[ ]:


l_reg = LinearRegression()


# In[ ]:


l_reg.fit(X_train,y_train)


# In[ ]:


y_pred = l_reg.predict(X_test)


# In[ ]:


from sklearn.metrics import r2_score,mean_squared_error


# In[ ]:


print(r2_score(y_test,y_pred))


# In[ ]:


print(mean_squared_error(y_test,y_pred))


# In[ ]:


print(l_reg.score(X_test,y_test))


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# In[ ]:


l_reg.predict([[48,1,35.625,4,0,0]])


# In[ ]:





# In[ ]:





# In[ ]:


corr_matrix['charges']


# In[ ]:


X = df.drop(['region','charges'],axis=1)  


# In[ ]:


y = df.iloc[:,-1]


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)


# In[ ]:


l_reg.fit(X_train,y_train)


# In[ ]:


y_pred = l_reg.predict(X_test)


# In[ ]:


l_reg.score(X_test,y_test)


# In[ ]:


mean_squared_error(y_test,y_pred)


# In[ ]:


print(r2_score(y_test,y_pred))


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# In[ ]:


l_reg.predict([[18,0,38.280,0,0]])


# # <b>Thank you. here i worked with a simle linear regression technique for beginers to understand how it works.</b>

# In[ ]:




