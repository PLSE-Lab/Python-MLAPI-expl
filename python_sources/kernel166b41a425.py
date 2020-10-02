#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing required libraries
import numpy as np
import pandas as pd


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Loading data 
train = pd.read_csv('../input/black-friday/train.csv')
test = pd.read_csv('../input/black-friday/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.tail()


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


train.columns


# In[ ]:


train.shape


# In[ ]:


#checking that null values in Product_Category_3 and Product_Category_2


# In[ ]:


train.Product_Category_3.value_counts(dropna=False).head()


# In[ ]:


train.Product_Category_2.value_counts(dropna=False).head()


# In[ ]:


test.head()


# In[ ]:


test.tail()


# In[ ]:


test.info()


# In[ ]:


test.describe()


# In[ ]:


test.Product_Category_3.value_counts(dropna=False).head()


# In[ ]:


test.Product_Category_2.value_counts(dropna=False).head()


# In[ ]:


#calculating missing values in terms of percentage
PC_2 = (173638/550068)*100
PC_3 = (383247/550068)*100
print ('Missing values in Product_Category_2 of train dataset is {0}%'.format(PC_2))
print ('Missing values in Product_Category_3 of train dataset is {0}%'.format(PC_3))
pc_2 = (72344/233599)*100
pc_3 = (162562/233599)*100
print ('Missing values in Product_Category_2 of test dataset is {0}%'.format(pc_2))
print ('Missing values in Product_Category_3 of test dataset is {0}%'.format(pc_3))


# In[ ]:


# working on missing data or null values
# removing Product_Category_3 from both data sets as more than 50% data is missing from the column
train.drop(columns='Product_Category_3',axis=1,inplace=True)
test.drop(columns='Product_Category_3',axis=1,inplace=True)


# In[ ]:


test.head()


# In[ ]:


train.head()


# In[ ]:


# Now filling the null values in Product_Category_2
sns.countplot(x='Product_Category_2', data=train)


# In[ ]:


train['Product_Category_2'].min()


# In[ ]:


train['Product_Category_2'].max()


# In[ ]:


test['Product_Category_2'].min()


# In[ ]:


test['Product_Category_2'].max()


# In[ ]:


# Filling null values randomly 


# In[ ]:


def fillNaN_with_random(data):
    a = data.values
    m = np.isnan(a) 
    
    a[m] = np.random.randint(2, 18, size=m.sum())
    return data


# In[ ]:


fillNaN_with_random(train['Product_Category_2'])


# In[ ]:


fillNaN_with_random(test['Product_Category_2'])


# In[ ]:


train.Product_Category_2.value_counts(dropna=False).head()


# In[ ]:


test.Product_Category_2.value_counts(dropna=False).head()


# In[ ]:


assert train.notnull().all().all()  # this line will through error if there will be any null value in data


# In[ ]:


assert test.notnull().all().all()


# In[ ]:


# checking unique entries in columns with object data type
for col_name in ['Gender', 'Age', 'City_Category','Stay_In_Current_City_Years']:
    print(sorted(train[col_name].unique()))


# In[ ]:


train.dtypes


# In[ ]:


# saving the names of all the columns having obect data type in obj_cols
obj_cols=train.select_dtypes(include=['object']).columns
print(obj_cols)


# In[ ]:


# applying label encoder to convert object data types into int data type
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[ ]:


train[obj_cols] = train[obj_cols].apply(lambda col: le.fit_transform(col))
train[obj_cols].head()


# In[ ]:


test[obj_cols] = test[obj_cols].apply(lambda col: le.fit_transform(col))
test[obj_cols].head()


# In[ ]:


#Visualizing Data


# In[ ]:


plt.figure(figsize = (10, 10))
sns.heatmap( train.corr(),annot=True)
plt.title('Correlation Of Train Data')


# In[ ]:


plt.figure(figsize = (10, 10))
sns.heatmap( test.corr(),annot=True)
plt.title('Correlation Of Test Data')


# In[ ]:


fig = plt.figure(figsize=(15,4))
train['Purchase'] = train.Purchase.apply(lambda amount : amount-(amount%1000))
sns.countplot(x='Purchase', data=train)


# In[ ]:


# checking for outliers in data 
sns.boxplot(x='Purchase',data=train)


# In[ ]:


sns.boxplot(x='Age',data=train)


# In[ ]:


sns.boxplot(x='Product_Category_1',data=train)


# In[ ]:


sns.boxplot(x='Product_Category_2',y='Purchase',data=train)


# In[ ]:


sns.boxplot(x='Product_Category_1',y='Purchase',data=train)


# In[ ]:


sns.boxplot(x='Age',y='Purchase',hue='Gender',data=train)


# In[ ]:


sns.boxplot(x='City_Category',y='Purchase',data=train)


# In[ ]:


train.shape


# In[ ]:


# there are outliers in three columns, removing outliers from the data 
outliers = train[['Age','Product_Category_1','Purchase']]
outliers.head()


# In[ ]:


Q1 = outliers.quantile(0.25)
Q3 = outliers.quantile(0.75)
IQR = Q3 - Q1
print(IQR) #prints IQR for each column


# In[ ]:


lb = Q1 -( 1.5 * IQR)
print('lower bound is \n',lb)
up = Q3 + (1.5 * IQR)
print('uper bound is \n',up)


# In[ ]:


print(outliers < lb ) |(outliers > up )


# In[ ]:


outliers.shape


# In[ ]:


train= train [~((outliers < lb ) |(outliers > up )).any(axis=1) ]


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


outliersTest = test[['Age','Product_Category_1']]
outliers.head()


# In[ ]:


Q1 = outliersTest.quantile(0.25)
Q3 = outliersTest.quantile(0.75)
IQR = Q3 - Q1
print(IQR) 


# In[ ]:


print(outliersTest <  Q1 -( 1.5 * IQR)) |(outliersTest >  Q3 + (1.5 * IQR) )


# In[ ]:


test= test [~((outliersTest < Q1 -( 1.5 * IQR) ) |(outliersTest > Q3 + (1.5 * IQR) )).any(axis=1) ]


# In[ ]:


test.shape


# In[ ]:


sns.boxplot(x='Product_Category_1',data=train)


# In[ ]:


sns.boxplot(x='Age',data=train)


# In[ ]:


sns.boxplot(x='Purchase',data=train)


# In[ ]:


train.to_csv("clean_train.csv",index=False, encoding='utf8')
test.to_csv("clean_test.csv",index=False, encoding='utf8')


# In[ ]:


testData = pd.read_csv('clean_test.csv')
testData.head()


# In[ ]:


ytrain = train ['Purchase']
Xtrain = train.drop(['Purchase'],axis=1)
Xtest = testData


# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


tree = DecisionTreeRegressor()


# In[ ]:


tree.fit(Xtrain,ytrain)


# In[ ]:


prediction = tree.predict(Xtest)


# In[ ]:


prediction


# In[ ]:


print(" Accuaracy is {0}%".format(tree.score(Xtest,prediction)*100))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


regressor = RandomForestRegressor(n_estimators = 10, random_state = 42)


# In[ ]:


regressor.fit(Xtrain,ytrain)


# In[ ]:


reg_pred = regressor.predict(Xtest)


# In[ ]:


reg_pred = [round(x) for x in reg_pred]


# In[ ]:


reg_pred 


# In[ ]:


ytrain


# In[ ]:


print("Accuracy is {0} %".format(regressor.score(Xtest,reg_pred)*100))


# In[ ]:


# making new dataframe
submission = pd.DataFrame()


# In[ ]:


#copying test data to this new data frame
submission = Xtest
submission.head()


# In[ ]:


#Assign all predictions of test Data to this new Data frame
submission['Purchase']=reg_pred


# In[ ]:


submission.head()


# In[ ]:


submission.shape


# In[ ]:


submission.to_csv("BlackFridayResults.csv",index=False, encoding='utf8')


# In[ ]:





# In[ ]:





# In[ ]:




