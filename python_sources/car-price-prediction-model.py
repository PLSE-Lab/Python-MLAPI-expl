#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# # Reading and Understanding the Data

# setting file path

# In[ ]:


df=pd.read_csv('../input/car-price-prediction/CarPrice_Assignment.csv')
df.head()


# understanding the dataframe

# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.describe()


# # Cleaning the Data

# dropping the car_ID as it is not affecting the car price

# In[ ]:



df.drop('car_ID',axis=1,inplace=True)
df.head()


# checking if the dataframe has any missing values

# In[ ]:


print(df.isnull().values.any())


# datafram columns and their respective datatypes

# In[ ]:


df.dtypes


# putting all subcategories into a single category

# In[ ]:


df['CarName']=df['CarName'].str.split(' ',expand=True)
df['CarName'].head()


# checking the unique car companies

# In[ ]:


df['CarName'].unique()


# renaming the typing errors in Car Company Names
# 
# syntax : 'wrong one' : 'correct one'

# In[ ]:



df['CarName']=df['CarName'].replace({'maxda':'mazda',
                                     'nissan':'Nissan',
                                     'toyouta':'toyota',
                                     'porcshce':'porsche',
                                     'vokswagen':'volkswagen',
                                     'vw':'volkswagen'
                                     })


# changing symboling to a string datatype from integer as it is mentioned in the dictionary excel file

# In[ ]:


df['symboling']=df['symboling'].astype(str)
df['symboling'].head()


# checking for duplicated values

# In[ ]:


df.loc[df.duplicated()]


# thus, we see that there are no duplicate values in the dataframe

# Segregation of Numerical and Categorical Variables/Columns

# In[ ]:


cat_col=df.select_dtypes(include='object').columns
num_col=df.select_dtypes(exclude='object').columns
df_cat=df[cat_col]
df_num=df[num_col]


# In[ ]:


df_cat.head(2)


# In[ ]:


df_num.head(2)


# # Visualising the Data

# In[ ]:


df['CarName'].value_counts()


# visualizing the different car names available

# In[ ]:


plt.figure(figsize=(10, 10))
ax=df['CarName'].value_counts().plot(kind='bar')
plt.title(label='CarName')
plt.xlabel("Names of the Car",fontweight = 'bold')
plt.ylabel("Count of Cars",fontweight = 'bold')
plt.show()


# pairplot, equivalent to correlation graph

# In[ ]:


ax=sns.pairplot(df[num_col])
plt.show()


# thus, 
# *   positive correlation with price : wheelbase, carlength, carwidth, curbweight, enginesize, horsepower, 
# *   negative correlation with price : citympg, highwaympg
# 
# 
# 

# visualising the categorical data

# In[ ]:


plt.figure(figsize=(20, 15))
plt.subplot(3,3,1)
sns.boxplot(x = 'doornumber', y = 'price', data = df)
plt.subplot(3,3,2)
sns.boxplot(x = 'fueltype', y = 'price', data = df)
plt.subplot(3,3,3)
sns.boxplot(x = 'carbody', y = 'price', data = df)
plt.subplot(3,3,4)
sns.boxplot(x = 'drivewheel', y = 'price', data = df)
plt.subplot(3,3,5)
sns.boxplot(x = 'enginelocation', y = 'price', data = df)
plt.subplot(3,3,6)
sns.boxplot(x = 'cylindernumber', y = 'price', data = df)
plt.subplot(3,3,7)
sns.boxplot(x = 'enginetype', y = 'price', data = df)
plt.subplot(3,3,8)
sns.boxplot(x = 'fuelsystem', y = 'price', data = df)
plt.subplot(3,3,9)
sns.boxplot(x = 'aspiration', y = 'price', data = df)
plt.show()


# car name grouped with respect to their average prices

# In[ ]:


ax=df.groupby(['CarName'])['price'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 10))
ax.plot.bar()
plt.title('Car Company Name vs Average Price')
plt.show()


# thus, we see jaguar has the highest average price

# car body grouped with respect to their average prices

# In[ ]:


ax=df.groupby(['carbody'])['price'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 10))
ax.plot.bar()
plt.title('Car Body Name vs Average Price')
plt.show()


# thus, car body with hardtop has the highest average price

# binning the car companies based on their average prices

# In[ ]:


df['price'] = df['price'].astype('int')
df_auto_temp = df.copy()
grouped = df_auto_temp.groupby(['CarName'])['price'].mean()
print(grouped)
df_auto_temp = df_auto_temp.merge(grouped.reset_index(), how='left', on='CarName')
bins = [0,10000,20000,40000]
label =['Budget_Friendly','Medium_Range','TopNotch_Cars']
df['Cars_Category'] = pd.cut(df_auto_temp['price_y'], bins, right=False, labels=label)
df.head()


# significant columns from visualised date:
# *   symboling
# *   fueltype
# *   aspiration
# *   carbody
# *   drivewheel
# *   enginelocation
# *   wheelbase
# *   carlength
# *   carwidth
# *   curbweight
# *   enginetype
# *   boreratio
# *   horsepower
# *   peakrpm
# *   citympg
# *   highwaympg  
# *   enginesize
# *   cylindernumber
# *   Cars_Category
# 
# 

# In[ ]:


sig_col = ['price','Cars_Category','enginetype','fueltype', 'aspiration','carbody','cylindernumber', 'drivewheel',
            'wheelbase','curbweight', 'enginesize', 'boreratio','horsepower', 
                    'citympg','highwaympg', 'carlength','carwidth']


# updating the dataframe, including only the significant columns

# In[ ]:


df=df[sig_col]


# # Data Preparation

# creating dummy variables for sig_cat_col,
# <br>
# sig_cat_col -> significant categorical columns

# In[ ]:


sig_cat_col=['Cars_Category','enginetype','fueltype','aspiration','carbody','cylindernumber','drivewheel']


# In[ ]:


dummies=pd.get_dummies(df[sig_cat_col])
print(dummies.shape)
dummies.head()


# avoiding dummy trap by removing the first column of each dummy variable

# In[ ]:


dummies=pd.get_dummies(df[sig_cat_col],drop_first=True)
print(dummies.shape)
dummies.head()


# concatenating the dataframe with the dummy variables

# In[ ]:


df=pd.concat([df,dummies],axis=1)


# dropping the significant categorial columns as we have already made and added the dummy variables for the same in the dataframe

# In[ ]:


df.drop(sig_cat_col,axis=1,inplace=True)
df.shape


# # Splitting the dataset into training and test sets

# In[ ]:


df


# With the seed reset (every time), the same set of numbers will appear every time. We specify this so that the train and test data set always have the same rows, respectively
# 
# We divide the df into 70/30 ratio

# In[ ]:


np.random.seed(0) 

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, train_size=0.7, test_size = 0.3, random_state = 100)


# In[ ]:


df_train.head()


# #Rescaling the features
# Rescaling the data using Standardisation Scaling.
# Scaling needs to be done on the significant num columns.
# The significant categorical columns have already been converted into dummies 

# In[ ]:


from sklearn.preprocessing import StandardScaler 
scaler=StandardScaler()


# sig_num_col -> significant num columns

# In[ ]:


sig_num_col = ['wheelbase','carlength','carwidth','curbweight','enginesize','boreratio','horsepower','citympg','highwaympg','price']


# applying scaler() to all the columns except the 'dummy' variables

# In[ ]:


df_train[sig_num_col]=scaler.fit_transform(df_train[sig_num_col])


# In[ ]:


df_train.head()


# checking the correlation co-efficients to see which variables are highly correlated

# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(df_train.corr(), cmap= 'RdYlGn')
plt.show()


# #Splitting train dataset into x_train and y_train

# In[ ]:


y_train=df_train.pop('price')


# In[ ]:


x_train=df_train


# # Building Linear Model

# In[ ]:


import statsmodels.api as sm

x_train_copy = x_train


# In[ ]:


x_train_copy1=sm.add_constant(x_train_copy['horsepower'])

#1st model
lr1=sm.OLS(y_train,x_train_copy1).fit()


# In[ ]:


lr1.params


# In[ ]:


print(lr1.summary())


# In[ ]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(x_train, y_train)


# #Recursive Feature Elimination (RFE)
# as there are too many independent variabls, we will use RFE

# In[ ]:


from sklearn.feature_selection import RFE

rfe=RFE(lm,15)
rfe=rfe.fit(x_train,y_train)


# checking which variables support RFE

# In[ ]:


list(zip(x_train.columns,rfe.support_,rfe.ranking_))


# selecting the variables which support RFE

# In[ ]:


col_sup=x_train.columns[rfe.support_]
col_sup


# creating x_train dataframe with RFE selected variables

# In[ ]:


x_train_rfe=x_train[col_sup]
x_train_rfe


# dropping variables having:
# *   high p-value, high vif
# *   high p-value, low vif or low p-value, high vif
# *   low p-value, low vif
# 
# 
# 
# 

# In[ ]:


import statsmodels.api as sm

x_train_rfec = sm.add_constant(x_train_rfe)
lm_rfe = sm.OLS(y_train,x_train_rfec).fit()

#Summary of linear model
print(lm_rfe.summary())


# #Variance Inflation Factor (VIF)
# it gives a basic quantitative idea about how much the feature variables are correlated with each other. It is an extremely important parameter to test our linear model.
# 
# vif values of variables should be less than 5 to be accepted

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['Features'] = x_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(x_train_rfe.values, i) for i in range(x_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Dropping cylindernumber_twelve beacuse its p-value is 0.393 and we want p-value less than 0.05 and hence rebuilding the model

# In[ ]:


x_train_rfe1=x_train_rfe.drop('cylindernumber_twelve',axis=1)

x_train_rfe1c=sm.add_constant(x_train_rfe1)
lm_rfe1=sm.OLS(y_train,x_train_rfe1c).fit()

print(lm_rfe1.summary())


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = x_train_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_train_rfe1.values, i) for i in range(x_train_rfe1.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Dropping cylindernumber_six beacuse its p-value is 0.493 and we want p-value less than 0.05 and hence rebuilding the model

# In[ ]:


x_train_rfe2=x_train_rfe1.drop('cylindernumber_six',axis=1)

x_train_rfe2c=sm.add_constant(x_train_rfe2)
lm_rfe2=sm.OLS(y_train,x_train_rfe2c).fit()

print(lm_rfe2.summary())


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = x_train_rfe2.columns
vif['VIF'] = [variance_inflation_factor(x_train_rfe2.values, i) for i in range(x_train_rfe2.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Dropping carbody_hardtop beacuse its p-value is 0.238 and we want p-value less than 0.05 and hence rebuilding the model

# In[ ]:


x_train_rfe3=x_train_rfe2.drop('carbody_hardtop',axis=1)

x_train_rfe3c=sm.add_constant(x_train_rfe3)
lm_rfe3=sm.OLS(y_train,x_train_rfe3c).fit()

print(lm_rfe3.summary())


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = x_train_rfe3.columns
vif['VIF'] = [variance_inflation_factor(x_train_rfe3.values, i) for i in range(x_train_rfe3.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Dropping enginetype_ohc beacuse its p-value is 0.110 and we want p-value less than 0.05 and hence rebuilding the model

# In[ ]:


x_train_rfe4=x_train_rfe3.drop('enginetype_ohc',axis=1)

x_train_rfe4c=sm.add_constant(x_train_rfe4)
lm_rfe4=sm.OLS(y_train,x_train_rfe4c).fit()

print(lm_rfe4.summary())


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = x_train_rfe4.columns
vif['VIF'] = [variance_inflation_factor(x_train_rfe4.values, i) for i in range(x_train_rfe4.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Dropping cylindernumber_five beacuse its p-value is 0.104 and we want p-value less than 0.05 and hence rebuilding the model

# In[ ]:


x_train_rfe5=x_train_rfe4.drop('cylindernumber_five',axis=1)

x_train_rfe5c=sm.add_constant(x_train_rfe5)
lm_rfe5=sm.OLS(y_train,x_train_rfe5c).fit()

print(lm_rfe5.summary())


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = x_train_rfe5.columns
vif['VIF'] = [variance_inflation_factor(x_train_rfe5.values, i) for i in range(x_train_rfe5.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Dropping enginetype_ohcv beacuse its p-value is 0.180 and we want p-value less than 0.05 and hence rebuilding the model

# In[ ]:


x_train_rfe6=x_train_rfe5.drop('enginetype_ohcv',axis=1)

x_train_rfe6c=sm.add_constant(x_train_rfe6)
lm_rfe6=sm.OLS(y_train,x_train_rfe6c).fit()

print(lm_rfe6.summary())


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = x_train_rfe6.columns
vif['VIF'] = [variance_inflation_factor(x_train_rfe6.values, i) for i in range(x_train_rfe6.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Dropping curbweight beacuse its VIF is 8.1 and we want VIF less than 5 and hence rebuilding the model

# In[ ]:


x_train_rfe7=x_train_rfe6.drop('curbweight',axis=1)

x_train_rfe7c=sm.add_constant(x_train_rfe7)
lm_rfe7=sm.OLS(y_train,x_train_rfe7c).fit()

print(lm_rfe7.summary())


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = x_train_rfe7.columns
vif['VIF'] = [variance_inflation_factor(x_train_rfe7.values, i) for i in range(x_train_rfe7.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Dropping cylindernumber_four beacuse its VIF is 5.66 and we want VIF less than 5 and hence rebuilding the model

# In[ ]:


x_train_rfe8=x_train_rfe7.drop('cylindernumber_four',axis=1)

x_train_rfe8c=sm.add_constant(x_train_rfe8)
lm_rfe8=sm.OLS(y_train,x_train_rfe8c).fit()

print(lm_rfe8.summary())


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = x_train_rfe8.columns
vif['VIF'] = [variance_inflation_factor(x_train_rfe8.values, i) for i in range(x_train_rfe8.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# checking both vif and p-value
# <br>
# Dropping highly correlated variables and insignificant variables

# In[ ]:


x_train_rfe9=x_train_rfe8.drop('carbody_sedan',axis=1)

x_train_rfe9c=sm.add_constant(x_train_rfe9)
lm_rfe9=sm.OLS(y_train,x_train_rfe9c).fit()

print(lm_rfe9.summary())


# we can see that the R-squared value did not change significantly, thus we can go ahead and drop carbody_sedan

# In[ ]:


vif = pd.DataFrame()
vif['Features'] = x_train_rfe9.columns
vif['VIF'] = [variance_inflation_factor(x_train_rfe9.values, i) for i in range(x_train_rfe9.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Dropping carbody_wagon beacuse its p-value is 0.315 and we want p-value less than 0.05 and hence rebuilding the model

# In[ ]:


x_train_rfe10=x_train_rfe9.drop('carbody_wagon',axis=1)

x_train_rfe10c=sm.add_constant(x_train_rfe10)
lm_rfe10=sm.OLS(y_train,x_train_rfe10c).fit()

print(lm_rfe10.summary())


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = x_train_rfe10.columns
vif['VIF'] = [variance_inflation_factor(x_train_rfe10.values, i) for i in range(x_train_rfe10.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# # Residual Analysis of train data

# predicting price of the training set

# In[ ]:


y_train_pred=lm_rfe10.predict(x_train_rfe10c)


# plotting histogram of the error terms

# In[ ]:


sns.distplot((y_train-y_train_pred),bins=20)
plt.title('Error Term Analysis')
plt.xlabel('Errors')
plt.show()


# # Making Predictions Using the Model

# In[ ]:


df_test[sig_num_col]=scaler.transform(df_test[sig_num_col])
df_test.shape


# splitting test set into x_test and 

# In[ ]:


y_test=df_test.pop('price')
x_test=df_test


# adding constant

# In[ ]:


x_test_1=sm.add_constant(x_test)

x_test_new=x_test_1[x_train_rfe10c.columns]


# making prediction

# In[ ]:


y_pred=lm_rfe10.predict(x_test_new)


# In[ ]:


y_pred


# # RMSE Score

# In[ ]:


from sklearn.metrics import r2_score

r2_score(y_test,y_pred)


# # Conclusions
# *   R-squared and Adjusted R-squared - 0.912 and 0.909
# * p - values for all coefficients seems to be less than the significance level of 0.05 i.e all the predictors are statistically significant
# *   90% variance explained
# 
# # Closing Statement : 
# thus, we can say that the model is good enough to predict the car prices which explains the variance of data upto 90% and the model is significant.
# 
# 
# 
#  

# In[ ]:




