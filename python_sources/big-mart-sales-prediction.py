#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing libraries
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Reading train & test data of dmart
dm_train=pd.read_csv("/kaggle/input/big-mart-sales-prediction/Train.csv")
dm_test=pd.read_csv('/kaggle/input/big-mart-sales-prediction/Test.csv')


# In[ ]:


#dmart train data
dm_train


# In[ ]:


#dmart test data
dm_test


# In[ ]:


dm_train.shape


# In[ ]:


dm_test.shape


# In[ ]:


dm_train.dtypes


# In[ ]:


dm_train.columns


# In[ ]:


# checking the null values and  filling with respect to their data type for both train and test
dm_train.isnull().sum()


# In[ ]:


dm_test.isnull().sum()


# In[ ]:


dm_train['Item_Weight'].fillna(dm_train['Item_Weight'].median(),inplace=True)
dm_train['Outlet_Size'].fillna(dm_train['Outlet_Size'].mode()[0],inplace=True)
dm_train.isnull().sum()


# In[ ]:


dm_test['Item_Weight'].fillna(dm_test['Item_Weight'].median(),inplace=True)
dm_test['Outlet_Size'].fillna(dm_test['Outlet_Size'].mode()[0],inplace=True)
dm_test.isnull().sum()


# In[ ]:


dm_train.describe()


# In[ ]:


# selecting only oblect type data to have a look over on it

dm_train.select_dtypes(include=['object'])


# In[ ]:


dm_train['Item_Fat_Content'].value_counts()


# In[ ]:


# replacing repeated values with the appropriate name in test and train

dm_train.Item_Fat_Content.replace('LF','Low Fat',inplace=True)
dm_train.Item_Fat_Content.replace('low fat','Low Fat',inplace=True)
dm_train.Item_Fat_Content.replace('reg','Regular',inplace=True)


# In[ ]:


dm_test.Item_Fat_Content.replace('LF','Low Fat',inplace=True)
dm_test.Item_Fat_Content.replace('low fat','Low Fat',inplace=True)
dm_test.Item_Fat_Content.replace('reg','Regular',inplace=True)


# In[ ]:


dm_train['Item_Fat_Content'].value_counts()


# In[ ]:


dm_test['Item_Fat_Content'].value_counts()


# In[ ]:


dm_train['Item_Fat_Content'].value_counts().plot.bar()


# In[ ]:


dm_train.Item_Type.value_counts()


# In[ ]:


dm_train.Item_Type.value_counts().plot.bar(color='pink',figsize=(12,8))


# In[ ]:


dm_train.Outlet_Identifier.value_counts()


# In[ ]:


dm_train.Outlet_Identifier.value_counts().plot(kind='bar')
plt.ylim(0,1000)


# In[ ]:


# grouping outlet identifier with outlet sales
grp=dm_train[['Outlet_Identifier','Item_Outlet_Sales']]
grp.head()


# In[ ]:


grp=grp.groupby(['Outlet_Identifier'],as_index=False).sum()
grp


# In[ ]:


# ploting the grouped values to visualize the rise of the sales
plt.figure(figsize=(12,8))
plt.plot(grp['Outlet_Identifier'],grp['Item_Outlet_Sales'],'go-')# here 'go-' is the format type to show
plt.grid()
plt.title('sales graph according to outlet',fontsize=15,color='r')
plt.ylim(0,)
plt.xlabel('Outlet_Identifier',fontsize=15)
plt.ylabel('Item_outlet sales in(10^6)',fontsize=15)


# In[ ]:


dm_train.Outlet_Establishment_Year.value_counts()


# In[ ]:


# groping years with sales 
grp=dm_train[['Outlet_Establishment_Year','Item_Outlet_Sales']]
grp.head()


# In[ ]:


grp1=grp.groupby(['Outlet_Establishment_Year'],as_index=False).sum()
grp1


# In[ ]:


# visualizing the graph of sales with years
plt.figure(figsize=(12,8))
plt.plot(grp1['Outlet_Establishment_Year'],grp1['Item_Outlet_Sales'],'r*-')
plt.title('sales price graph with respect to year',fontsize=15,color='g')
plt.xlabel('years',fontsize=15,color='red')
plt.ylabel('sales price in (10^6)',fontsize=15,color='red')
plt.ylim(0,4000000)
plt.xlim(1984,2010)
plt.grid()


# In[ ]:


#  Geting  the comprsed(with less decimal point ) values of mrp,weight using round keyword
dm_train['Item_MRP']=dm_train['Item_MRP'].round()
dm_train['Item_Weight']=dm_train['Item_Weight'].round()
dm_train['Item_Visibility']=dm_train['Item_Visibility'].round(3)
dm_train


# In[ ]:


dm_test['Item_MRP']=dm_test['Item_MRP'].round()
dm_test['Item_Weight']=dm_test['Item_Weight'].round()
dm_test['Item_Visibility']=dm_test['Item_Visibility'].round(3)
dm_test


# In[ ]:


dm_train['Item_Weight'].value_counts()


# In[ ]:


dm_train['Item_Weight'].describe()


# In[ ]:


# finding the relation of item types towards weight using crosstab
pd.crosstab(dm_train['Item_Type'],dm_train['Item_Weight'],margins=True)

# if margins is true it will give the totalcount in rows & columns


# In[ ]:


dm_train['Outlet_Type'].unique()


# In[ ]:


# grouging outlet tye with sales
super_grp=dm_train[['Outlet_Type','Item_Outlet_Sales']]
super_grp.head()


# In[ ]:


grp2=super_grp.groupby(['Outlet_Type'],as_index=False).sum()
grp2


# In[ ]:


# visualizing sales with respect to outlet type 
plt.figure(figsize=(12,9))
plt.plot(grp2['Outlet_Type'],grp2['Item_Outlet_Sales'],'b*-')
plt.title('sales price with supermarket',fontsize=15,color='r')
plt.xlabel('out let type',fontsize=15,color='green')
plt.ylabel('sales price in crores(10^7)',fontsize=15,color='green')
plt.grid()


# In[ ]:


dm_train.head()


# In[ ]:


dm_train['Outlet_Size'].value_counts()


# In[ ]:


dm_train['Outlet_Size'].value_counts().plot.bar(color='r')


# In[ ]:


dm_train['Outlet_Type'].value_counts()


# In[ ]:


dm_train['Outlet_Type'].value_counts().plot.bar(color='g')


# In[ ]:


dm_train['Item_MRP'].describe()


# In[ ]:


# correlating the train data
matrix=dm_train.corr()
plt.figure(figsize=(12,9))
sns.heatmap(matrix,vmin=-1,vmax=1,square=True,annot=True,cmap='BuGn')


# In[ ]:


# by default correlation
sns.heatmap(matrix,square=True,annot=True)


# In[ ]:


# finding the inter Quartile rang using boxplot
dm_train['Item_Outlet_Sales'].plot.box(figsize=(12,8))


# In[ ]:


sns.boxplot(y=dm_train['Item_Outlet_Sales'])


# In[ ]:


# ploting the distance graph using seaborn 
sns.distplot(dm_train['Item_Outlet_Sales'])


# In[ ]:


#boxplot for sale with columns in matplotlib
dm_train.boxplot(column='Item_Outlet_Sales',by='Outlet_Type',figsize=(12,8),grid=False)
plt.suptitle(" ")


# In[ ]:


# boxplot with seaborn
sns.boxplot(y=dm_train['Item_Outlet_Sales'],x=dm_train['Outlet_Type'])


# In[ ]:


# checking duplicates
dm_train=dm_train.drop_duplicates()
dm_train


# In[ ]:


dm_test=dm_test.drop_duplicates()
dm_test


# In[ ]:


dm_train['Item_MRP'].plot.box(figsize=(12,9))
plt.ylim(0,300)
plt.title('box plot mrp',fontsize=14)


# In[ ]:


pd.crosstab(dm_train['Item_Type'],dm_train['Item_Fat_Content'])


# In[ ]:


pd.crosstab(dm_train['Item_Type'],dm_train['Item_Fat_Content']).plot.bar(figsize=(15,9))


# In[ ]:


pd.crosstab(dm_train['Outlet_Size'],dm_train['Item_Fat_Content'])


# In[ ]:


pd.crosstab(dm_train['Outlet_Size'],dm_train['Item_Fat_Content']).plot.bar(figsize=(15,9))


# In[ ]:


pd.crosstab(dm_train['Outlet_Size'],dm_train['Item_Fat_Content'],normalize=True).plot.bar(figsize=(15,9))


# In[ ]:


sns.stripplot(dm_train['Outlet_Establishment_Year'],dm_train['Item_Outlet_Sales'])


# In[ ]:


dm_train=dm_train.drop(['Item_Identifier','Item_Visibility'],1)


# In[ ]:


dm_train.head()


# In[ ]:


#label Encoding technicue
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[ ]:


dm_train['Item_Fat_Content']=le.fit_transform (dm_train['Item_Fat_Content'])
dm_train['Outlet_Location_Type']=le.fit_transform(dm_train['Outlet_Location_Type'])
dm_train['Item_Type']=le.fit_transform(dm_train['Item_Type'])
dm_train['Outlet_Size']=le.fit_transform(dm_train['Outlet_Size'])
dm_train['Outlet_Identifier']=le.fit_transform(dm_train['Outlet_Identifier'])
dm_train['Outlet_Type']=le.fit_transform(dm_train['Outlet_Type'])


# In[ ]:


dm_test['Item_Fat_Content']=le.fit_transform (dm_test['Item_Fat_Content'])
dm_test['Outlet_Location_Type']=le.fit_transform(dm_test['Outlet_Location_Type'])
dm_test['Item_Type']=le.fit_transform(dm_test['Item_Type'])
dm_test['Outlet_Size']=le.fit_transform(dm_test['Outlet_Size'])
dm_test['Outlet_Identifier']=le.fit_transform(dm_test['Outlet_Identifier'])
dm_test['Outlet_Type']=le.fit_transform(dm_test['Outlet_Type'])


# In[ ]:


dm_train


# In[ ]:


dm_test=dm_test.drop(['Item_Identifier','Item_Visibility'],1)
dm_test


# In[ ]:


train=dm_train
test=dm_test


# In[ ]:


x=train.drop('Item_Outlet_Sales',1)
y=train.Item_Outlet_Sales
# we are using the logirthmic values of the output to get better respones in score it is stored in Y
Y=np.log(y)
Y


# In[ ]:


#  ploting boxplot & distance for log values
plt.boxplot(Y)


# In[ ]:


sns.distplot(Y)


# In[ ]:


# generating dummie values for the label encoded columns
x=pd.get_dummies(x,columns=['Item_Type','Item_Fat_Content','Outlet_Identifier','Outlet_Location_Type','Outlet_Size','Outlet_Type'])
x


# In[ ]:


test=pd.get_dummies(test,columns=['Item_Type','Item_Fat_Content','Outlet_Identifier','Outlet_Location_Type','Outlet_Size','Outlet_Type'])
test


# In[ ]:


# spliting &fitting values of TRAIN DATA 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size =0.3)


# In[ ]:


#linear regression
model=LinearRegression()
model.fit(x_train,y_train)
predy=model.predict(x_test)


# In[ ]:


predy


# In[ ]:


y_test


# In[ ]:


#  finding error from predected values (-) true value
error=predy-y_test
error


# In[ ]:


# finding mean square error
totalerror=np.sum(error*error)# error^2=error*error
mse=totalerror/len(predy)
print('MSE:',mse)


# In[ ]:


# finding root mean square
rms=np.sqrt(mse)
rms


# In[ ]:


model.score(x_train,y_train)


# In[ ]:


sns.distplot(predy)


# In[ ]:


sns.distplot(y)


# In[ ]:


#comparing the outputs predected vs true output
ax1=sns.distplot(y_test,hist=False)
sns.distplot(predy,hist=False,ax=ax1,color='r')


# In[ ]:


# performing standard scaler operation fot it
from sklearn.preprocessing import StandardScaler
x_trainsc=StandardScaler().fit_transform(x_train)
x_testsc=StandardScaler().fit_transform(x_test)


# In[ ]:


model=LinearRegression()
model.fit(x_trainsc,y_train)
predy=model.predict(x_testsc)


# In[ ]:


model.score(x_trainsc,y_train)


# In[ ]:


from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(y_test,predy)
r2=r2_score(y_test,predy)


# In[ ]:


mse


# In[ ]:


r2


# In[ ]:


# using log  of the y
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.3)
model=LinearRegression()
model.fit(x_train,y_train)
predection=model.predict(x_test)


# In[ ]:


model.score(x_train,y_train)


# In[ ]:


model.score(x_test,y_test)


# In[ ]:


from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(predection,y_test)
r2=r2_score(predection,y_test)
print('mse:',mse,'\n','rmse:',r2)


# In[ ]:


ax=sns.distplot(predection,hist=False)
sns.distplot(y_test,hist=False,color='r')


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(criterion='mse',random_state=40)


# In[ ]:


model.fit(x_train,y_train)
pred=model.predict(x_test)


# In[ ]:


model.score(x_train,y_train)


# In[ ]:


model.score(x_test,y_test)


# In[ ]:


ax1=sns.distplot(y_test,hist=False)
sns.distplot(pred,hist=False,color='r')    


# In[ ]:


from sklearn.model_selection import cross_val_score
model=RandomForestRegressor()
score=cross_val_score(model,x,Y,cv=5)
score


# In[ ]:


# converting data into standard scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_sc = sc.fit_transform(x_train)
test_sc = sc.transform(x_test)
model=RandomForestRegressor()
model.fit(train_sc,y_train)
predection=model.predict(test_sc)


# In[ ]:


model.score(train_sc,y_train)


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(x_train, y_train)
y_prediction = knn.predict(x_test)
knn.score(x_test,y_test)


# In[ ]:


knn.score(x_train,y_train)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
x_train, x_test, y_train, y_test= train_test_split(x,Y, random_state=51,test_size =0.3)
model=DecisionTreeRegressor(criterion='mse')
model.fit(x_train,y_train)
predection=model.predict(x_test)
model.score(x_train,y_train)


# In[ ]:


ax1=sns.distplot(y_test,hist=False)
sns.distplot(predection,hist=False,color='r') 


# In[ ]:


# reconverting the log values into normal expotenial values
predection=np.exp(predection)
y_test=np.exp(y_test)


# In[ ]:


ax1=sns.distplot(y_test,hist=False)
sns.distplot(predection,hist=False,color='r')


# In[ ]:


# preforming decission tree regressor with log Y values
model=DecisionTreeRegressor(criterion='mse')
model.fit(x,Y)
model.score(x,Y)


# In[ ]:


# preforming decission tree regressor with normal values
model=DecisionTreeRegressor(criterion='mse')
model.fit(x,y)
model.score(x,y)


# In[ ]:


out=model.predict(test)


# In[ ]:


out


# In[ ]:


samplesubmission=pd.read_csv("/kaggle/input/big-mart-sales-prediction/Submission.csv")
samplesubmission.head()


# In[ ]:


samplesubmission['Item_Outlet_Sales']=out
samplesubmission


# In[ ]:


# saving the data in csv format
samplesubmission.to_csv(' big mart output.csv')


# In[ ]:


get_ipython().run_line_magic('ls', '')


# In[ ]:




