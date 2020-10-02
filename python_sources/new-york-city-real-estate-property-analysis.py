#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# The purpose of this analysis is to reinforce the data science methotology and process from reviewing the problem in question, getting and cleaning the data, analyzing it and further creating a predictive model. This report will revolve around the use case of New York Real Estate market. 
# 
# I am interested in automating the process of figuring out estimated price of a real estate properly in the recent future. Can I predict the sale price of a property within a certain area? Can I figure out estimated price of a property within a certain neighbourhood? Can I predict the future up and coming neighborhood within a borough? These are some of the questions I will try to answer with my analysis.
# 
# I will start downloading my libraries.

# ## Importing Modules and Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# Besides these standard libraries, i will import sklearn, warnings for my predictive model and itertools moduole for iteration and looping.

# In[ ]:


import itertools
import sklearn as sk
import warnings


# I will also set the style use of seaborn and matplotlib for my visualizations.

# In[ ]:


sns.set(style='white', context='notebook', palette='deep')
import matplotlib.style as style
style.use('fivethirtyeight')


# Finally will import regression , metrics and other model libraries for ML from sklearn

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet


# In[ ]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


# I will further import my data and start the cleaning process. I grabbed the data "nyc-rolling-sales" from Kaggle. It is in csv format and easy to import.

# ## Import the Data

# In[ ]:


df=pd.read_csv('../input/nyc-rolling-sales.csv')


# In[ ]:


df.head(5)


# When I initially look at the data set, I see "Unnamed o" column as a possible index coming from the csv and "Easement" variable that doesnt have any value in the rows. I will drop these columns.

# ## Data Cleaning

# In[ ]:


df.drop(columns={'Unnamed: 0', 'EASE-MENT'}, inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.dtypes


# The Boroguh column has integer type; based on the value of the variable, this is correct; however, Borough should have me a name such as "Manhattan" or "Brooklyn". According the data set from Kaggle I will convert the numeric values into the Borough names.

# In[ ]:


df['BOROUGH'][df['BOROUGH']==1]='Manhattan'


# In[ ]:


df['BOROUGH'][df['BOROUGH']==2]='Bronx'


# In[ ]:


df['BOROUGH'][df['BOROUGH']==3]='Brooklyn'


# In[ ]:


df['BOROUGH'][df['BOROUGH']==4]='Queens'


# In[ ]:


df['BOROUGH'][df['BOROUGH']==5]='Staten Island'


# In[ ]:


df.head()


# In[ ]:


df.info()


# Part of the cleaning process I will look for the missing and duplicate data. 3 Steps to cleaning the missing data, I will identify them and either remove, correct or replace them.

# In[ ]:


missing_data=df.isnull()


# In[ ]:


missing_data.head()


# In[ ]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")


# In[ ]:


sum(df.duplicated(df.columns))


# There isnt any missing data value in the dataset however there are 765 duplicates. I will remove the duplicated data so it doesnt hinder the analysis.

# In[ ]:


df=df.drop_duplicates(df.columns, keep='last')


# In[ ]:


sum(df.duplicated(df.columns))


# Since there are no duplicated data, I can look at the values and make sure they have correct data types.

# In[ ]:


df.dtypes


# In[ ]:


df.head(2)


# "SALE PRICE', "YEAR BUILT", "LAND SQUARE FEET", "GROSS SQUARE FEET" have 'object' values but should all be numeric. "SALE DATE" has an 'object' value but should be datatime, "TAX CLASS AT THE TIME OF SALE" has 'integer' value but should be categorical. "TAX CLASS AT PRESENT" has 'object' value and should also have categorical value.

# In[ ]:


df['SALE PRICE']=pd.to_numeric(df['SALE PRICE'], errors='coerce')


# In[ ]:


df['YEAR BUILT']=pd.to_numeric(df['YEAR BUILT'], errors='coerce')


# In[ ]:


df['LAND SQUARE FEET']=pd.to_numeric(df['LAND SQUARE FEET'], errors='coerce')


# In[ ]:


df['GROSS SQUARE FEET']=pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')


# In[ ]:


df['SALE DATE']=pd.to_datetime(df['SALE DATE'], errors='coerce')


# In[ ]:


df['TAX CLASS AT TIME OF SALE'] = df['TAX CLASS AT TIME OF SALE'].astype('category')
df['TAX CLASS AT PRESENT'] = df['TAX CLASS AT PRESENT'].astype('category')
df['ZIP CODE'] = df['ZIP CODE'].astype('category')


# In[ ]:


df.head()


# There are NaN values in the "SALE PRICE" variable, I will replace them with average Sale Price

# In[ ]:


df.isnull().sum()


# There are 26054 missing data for "Land Square Feet", 27385 for "Gross Square Feet" and 14176 missing values for "SALE PRICE". I will;
# <ul>
#     <li> Avg each missing data variable</li>
#     <li> Replace the missing dat value with the avg</li>

# In[ ]:


avg_land=df['LAND SQUARE FEET'].astype('float').mean(axis=0)


# In[ ]:


df['LAND SQUARE FEET'].replace(np.nan, avg_land, inplace=True)


# In[ ]:


avg_gross=df['GROSS SQUARE FEET'].astype('float').mean(axis=0)


# In[ ]:


df['GROSS SQUARE FEET'].replace(np.nan, avg_gross, inplace=True)


# In[ ]:


avg_sale_price=df['SALE PRICE'].astype('float').mean(axis=0)


# In[ ]:


df['SALE PRICE'].replace('np.nan, avg_sale_price', inplace=True)


# In[ ]:


df.head()


# There are null values in the SALE PRICE data. 

# In[ ]:


var=df.columns
count=[]
for variable in var:
    length=df[variable].count()
    count.append(length)


# In[ ]:


plt.figure(figsize=(40,8))
sns.barplot(x=var, y=count)
plt.title('Percentage of Available Data', fontsize=30)
plt.show()


# There are 10,000 values showing up as null for SALE DATA. These needs to be removed from our observations. 

# Standardization or normalization is not required in this case. I corrected the data types, removed duplicate data and there are no missing data values. I can further start analyszing the data set. Woot!

# ## Data Exploration

# In[ ]:


df.corr()


# In[ ]:


df[['COMMERCIAL UNITS', 'SALE PRICE']].corr()


# In[ ]:


sns.regplot(x='COMMERCIAL UNITS', y='SALE PRICE', data=df)


# There is not much correlation with commercial units in a neighborhood and sale price

# In[ ]:


df[['LAND SQUARE FEET', 'SALE PRICE']].corr()


# In[ ]:


sns.regplot(x='LAND SQUARE FEET', y='SALE PRICE', data=df)


# Interestingly there correlation between Land Square Feet and Sale Price is weak as well

# In[ ]:


df[['GROSS SQUARE FEET', 'SALE PRICE']].corr()


# In[ ]:


sns.regplot(x='GROSS SQUARE FEET', y='SALE PRICE', data=df)


# There is definately strong correlation between Gross Square Feet and Sale Price. Regression line is not that bad in this case so I can use Gross Square Feet as a predictor variable.

# I will explore more around the neighboorhood and boroughs. 

# In[ ]:


sns.boxplot(x='BOROUGH', y='SALE PRICE', data=df)


# In[ ]:


df.head(2)


# In[ ]:


df.describe(include=['object'])


# Grouping the dataset based on Boroughs will help segmenting my analysis.

# In[ ]:


df_borough=df[['BOROUGH','SALE PRICE', 'SALE DATE']]


# In[ ]:


df_borough.head(5)


# In[ ]:


df_borough=df_borough.groupby(['BOROUGH'], as_index=False).mean()


# In[ ]:


df_borough.head(100)


# In[ ]:


sns.boxplot(x='BOROUGH', y='SALE PRICE', data=df_borough)


# Manhattan has the most expensive houses in averege, followed by Brooklyn, Queens, Bronx and Staten Island. The median gap between Manhattan and other boroughs is huge.

# Let's examine "Manhattan" in more detail.

# In[ ]:


df_manhattan=df[(df['BOROUGH']=='Manhattan')]


# In[ ]:


df_manhattan


# In[ ]:


df_manhattan_neighborhood=df[['NEIGHBORHOOD', 'RESIDENTIAL UNITS','SALE PRICE', 'SALE DATE']]


# In[ ]:


df_manhattan_neighborhood=df_manhattan_neighborhood.groupby(['NEIGHBORHOOD', 'SALE PRICE'], as_index=False).mean()


# In[ ]:


df_manhattan_neighborhood


# In[ ]:


fig, ax = plt.subplots(figsize=(40,20))
plt.xticks(fontsize=30) 
plt.yticks(fontsize=30)
ax.set_title('Neighborhood Sale Price Analysis', fontweight="bold", size=30)
ax.set_ylabel('Neighborhood', fontsize = 30)
ax.set_xlabel('Sale Price', fontsize = 30)
sns.boxplot(x='SALE PRICE', y='NEIGHBORHOOD', data=df_manhattan)


# In[ ]:


df_manhattan.head(2)


# In[ ]:


fig, ax = plt.subplots(figsize=(40,20))
plt.xticks(fontsize=30) 
plt.yticks(fontsize=30)
ax.set_title('Neighborhood vs Residential Units Analysis', fontweight="bold", size=30)
ax.set_ylabel('Neighborhood', fontsize = 30)
ax.set_xlabel('Residential Units', fontsize = 30)
sns.barplot(x='RESIDENTIAL UNITS', y='NEIGHBORHOOD', data=df_manhattan)


# In[ ]:


plt.figure(figsize=(12,4))
sns.countplot(x='BOROUGH', data=df)


# We have a lot more data on "Brooklyn" than any other borough within the data set. This might indicate there are more real estate properties in Brooklyn than other boroughs. Interesting that Queens is the largest borough in NYC but has less real estate properties. This might also indicate that our data set might not be accurate. 

# We have Sale Date data as a variable in our dataset. However using sale data, considering the amount of possible variations might make it difficult to use as a predictor variable. So I will categorize them and create a season variable.

# In[ ]:


def get_season(x):
    if x==1:
        return 'Summer'
    elif x==2:
        return 'Fall'
    elif x==3:
        return 'Winter'
    elif x==4:
        return 'Spring'
    else:
        return ''
df['seasons']=df['SALE DATE'].apply(lambda x:x.month)
df['seasons']=df['seasons'].apply(lambda x:(x%12+3)//3)
df['seasons']=df['seasons'].apply(get_season)


# In[ ]:


plt.figure(figsize=(18,8))
df_wo_manhattan=df.loc[df['BOROUGH']!='Manhattan']
sns.relplot(x="BOROUGH", y="SALE PRICE",hue='seasons' ,kind="line", data=df_wo_manhattan,legend='full')


# There is no correlation between sale price and seaons. For example; Sale prices go up and down during the summer for all boroughs. There is definately a correlation between boroughs and sale price though. Keep in mind, correlation does not mean causation. Cause of something will require more investigation. In this case, I can't use seasons as a predictor variable for my model.

# Lets see if there is a linear correlation between "Land Square Feet" and "Sales Price" . Simple Linear Regression is a very straight forward data model; it can help us understand the relationship between Land Square Feet and Sales Price.

# In[ ]:


sns.regplot(x='SALE PRICE', y='LAND SQUARE FEET', data=df)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


df[['SALE PRICE', 'GROSS SQUARE FEET', 'LAND SQUARE FEET']]


# In[ ]:


df.dropna(subset=["SALE PRICE"], axis=0, inplace = True)


# In[ ]:


df.reset_index(drop = True, inplace = True)


# In[ ]:


df.dropna(subset=['GROSS SQUARE FEET', 'LAND SQUARE FEET'], axis=0, inplace=True)


# In[ ]:


lm = LinearRegression()
lm


# In[ ]:


X = df[['SALE PRICE']]
Y = df['GROSS SQUARE FEET']


# In[ ]:


lm.fit(X,Y)


# Here is our prediction model (single regression) using Gross Square Feet as a predictive variable.

# In[ ]:


Yhat=lm.predict(X)
Yhat[0:5]   


# In[ ]:


## intercept value is
lm.intercept_


# In[ ]:


## slope
lm.coef_


# <ul>
#     <li>Final Predictive Model using Single Linear Regression </li>
#     <li>Yhat=a+bx </li>
#     <li>a is intercept</li>
#     <li>b is slope</li>
#     <li> final formula == SALE PRICE = 2422.1667333157084 + 0.00095306 * GROSS SQUARE FEET  

# In[ ]:


plt.figure(figsize=(12, 10))
sns.regplot(x="GROSS SQUARE FEET", y="SALE PRICE", data=df)
plt.ylim(0,)


# Conclusion: We can use single regression to create a predictive model GROSS SQUARE FEET AND OR LAND SQUARE FEET as the predictive variable.

# Let's see if we can use the BOROUGH

# In[ ]:


sns.pairplot(data=df, hue='BOROUGH')


# Based on the data exploration, I can use GROSS SQUARE FEET, LAND SQAURE FEET as predictive variable for Single or Multi Linear Regresstion data model. Important data variables are;
# 
# <ul>
#     <li>Residential Units</li>
#     <li>Land Square Feet</li>
#     <li>Gross Square Feet</li>
#     <li>Age of the Building Sale</li>
# 

# ## Data Preperation

# In[ ]:


variable_model=['BOROUGH','BUILDING CLASS CATEGORY','COMMERCIAL UNITS','GROSS SQUARE FEET',
               'SALE PRICE','Building Age During Sale','LAND SQUARE FEET','RESIDENTIAL UNITS','seasons']
data_model=df.loc[:,variable_model]


# In[ ]:


important_features=['BOROUGH','BUILDING CLASS CATEGORY','seasons']
longest_str=max(important_features,key=len)
total_num_of_unique_cat=0
for feature in important_features:
    num_unique=len(data_model[feature].unique())
    print('{} : {} unique categorical values '.format(feature,num_unique))
    total_num_of_unique_cat+=num_unique
print('Total {} will be added with important feature adding'.format(total_num_of_unique_cat))


# "SALE PRICE" is the dependent variable meaning , we are trying to predict the sale price in our 

# In[ ]:


df[df['SALE PRICE']==0.0].sum().count()


# In[ ]:


important_features_included = pd.get_dummies(data_model[important_features])
important_features_included.info(verbose=True, memory_usage=True, null_counts=True)


# In[ ]:


data_model.drop(important_features,axis=1,inplace=True)
data_model=pd.concat([data_model,important_features_included],axis=1)
data_model.head()


# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(data_model['SALE PRICE'],bins=2)
plt.title('Histogram of SALE PRICE')
plt.show()


# In[ ]:


data_model.head()


# In[ ]:


data_model=data_model[data_model['SALE PRICE']!=0]
data_model


# In[ ]:


data_model['SALE PRICE'] = StandardScaler().fit_transform(np.log(data_model['SALE PRICE']).values.reshape(-1,1))
plt.figure(figsize=(10,6))
sns.distplot(data_model['SALE PRICE'])
plt.title('Histogram of Normalised SALE PRICE')
plt.show()


# In[ ]:


data_model.describe()


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[ ]:


lm.fit(X, Y)


# In[ ]:


lm.score(X, Y)


# In[ ]:


Yhat=lm.predict(X)
Yhat[0:4]


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


mean_squared_error(df['SALE PRICE'], Yhat)


# ## Split the Data Set

# In[ ]:


y=data_model['SALE PRICE']
X=data_model.drop('SALE PRICE',axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)
print('Size of Training data: {} \n Size of test data: {}'.format(X_train.shape[0],X_test.shape[0]))


# We split our data set into two sections; Train and Test data sets.

# In[ ]:


data_model.shape[0]


# By our above exploration, we know we will be using single and multi linear regression.
# 

# In[ ]:


sns.distplot(y_test)
plt.show()


# Conclusion:
# 
# <ul>
#     <li>Dependent Variable is 'SALE PRICE'</li>
#     <li>Predictive variables are 'LAND SQUARE FEET', GROSS SQUARE FEET', 'BOROUGH'</li>
#     <li>Predictive Models to be used - Single Linear or Multi Linear Regression</li>
#     

# In[ ]:




