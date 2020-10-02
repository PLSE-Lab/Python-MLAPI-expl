#!/usr/bin/env python
# coding: utf-8

# ## Step 1: Reading and Understanding the data

# In[ ]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# for train-test split of dataset
from sklearn.model_selection import train_test_split

# for scaling of dataset
from sklearn.preprocessing import MinMaxScaler

# RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# to create linear model
import statsmodels.api as sm

# to check VIFs
from statsmodels.stats.outliers_influence import variance_inflation_factor

# R-squared
from sklearn.metrics import r2_score

# MSE
from sklearn.metrics import mean_squared_error

# Supress Warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


print(os.listdir('../input/geely-auto'))


# In[ ]:


# importing csv file
car_df = pd.read_csv("../input/geely-auto/CarPriceAssignment.csv")
car_df.head()


# ####  # Inspecting the Car Dataset

# In[ ]:


print(car_df.shape)
car_df.info()


# There are **`205`** `rows` and **`26`** `columns` without any missing values...its a good thing, we need not to perform any missing value treatments.

# In[ ]:


# Describing the numerical vars
car_df.describe()


# Looks pretty decent....

# ## Step 2: Visualising the Data

# #### # Visualising the Numerical Variables:

# In[ ]:


sns.pairplot(car_df)
plt.show()


# -  More or less every Numerical variables are Normally distributed, whereas `Price` is highly Right-skewed which is our `Dependent Variable`,`horsepower` also right-skewed.
# -  `compressionratio` is the only variable having different spikes, one at left another at right, we should inspect it more...

# In[ ]:


# searching for outliers in compressionratio

# Box-plot
plt.figure(figsize=(15,6))
plt.subplot(121)
sns.boxplot(data=car_df.compressionratio, width=0.5, palette="colorblind")
plt.title('Box-Plot Compression Ratio')

# Scatter plot
plt.subplot(122)
sns.scatterplot(data=car_df.compressionratio, palette="colorblind")
plt.title('Scatter Plot Compression Ratio')
plt.show()


# So, majority of the values lies between `0-12`, whereas few others lying arround `22`, those seems to be outliers, lets check their percentage...

# In[ ]:


# percentage of outliers
comp_ratio = car_df[['compressionratio']].copy().sort_values(by='compressionratio',ascending=False)
comp_ratio_outlier = car_df[car_df['compressionratio']>12]
print(len(comp_ratio))
print(len(comp_ratio_outlier))

comp_ratio_outlier_perc = round(100*(len(comp_ratio_outlier) / len(comp_ratio)),2)
print('Outlier percentage of compressionratio: ' + str(comp_ratio_outlier_perc))


# So, the outlier percentage is around `10%`, removing 20 rows out of 205 seems to be expensive, lets keep them for now, in future we may handle them.

# In[ ]:


# heatmap
plt.figure(figsize = (20,10))  
sns.heatmap(car_df.corr(), cmap= 'YlGnBu',annot = True)


# -  As we can see, there is high correlation(0.97) between `citympg` and `highwaympg`, so we can get rid one of them, as they will have same impact on dataset.
# -  High collinearity also exist among `carlength`, `curbweight`, `wheelbase` and `carwidth` around 0.84 to 0.88, so we can keep only one of them and drop others

# In[ ]:


# Keeping 'citympg' and 'carlength' from above
car_df = car_df.drop(['carwidth','curbweight','wheelbase','highwaympg'], axis=1)
car_df.head()


# We may drop the `car_ID` column as it just a serial no. which is not putting any significance.

# In[ ]:


# dropping car_ID
car_df = car_df.drop('car_ID', 1)
car_df.head()


# #### # Visualising the Categorical Variables:

# In[ ]:


# Categorical columns
car_df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# In[ ]:


plt.figure(figsize=(20, 20))
plt.subplot(3,3,1)
sns.boxplot(x = 'fueltype', y = 'price', data = car_df)
plt.subplot(3,3,2)
sns.boxplot(x = 'aspiration', y = 'price', data = car_df)
plt.subplot(3,3,3)
sns.boxplot(x = 'doornumber', y = 'price', data = car_df)
plt.subplot(3,3,4)
sns.boxplot(x = 'carbody', y = 'price', data = car_df)
plt.subplot(3,3,5)
sns.boxplot(x = 'drivewheel', y = 'price', data = car_df)
plt.subplot(3,3,6)
sns.boxplot(x = 'enginelocation', y = 'price', data = car_df)
plt.subplot(3,3,7)
sns.boxplot(x = 'enginetype', y = 'price', data = car_df)
plt.subplot(3,3,8)
sns.boxplot(x = 'cylindernumber', y = 'price', data = car_df)
plt.subplot(3,3,9)
sns.boxplot(x = 'fuelsystem', y = 'price', data = car_df)
plt.show()


# ## Step 3: Data Preparation

# - **CarName** comprised of `car comapny` and `car model`, as per direction of this assignment we have to consider only `Company name`.

# In[ ]:


car_df.CarName.head(20)


# So, we've to pick the company name from CarName by using delimitters `-` and `space`

# In[ ]:


# split and taking the Company name
car_df['CarName'] = car_df['CarName'].apply(lambda x:re.split('-| ',x)[0])
car_df['CarName'].head(10)


# In[ ]:


# Names and count of Cars according to Companies
car_df['CarName'].value_counts()


# 
# So, there are so many different comapnies, some of them belongs to only one company having different names, like:
# -  `toyota` = `toyouta`
# -  `vw`=`volkswagen`
# -  `vokswagen`=`volkswagen`
# -  `toyouta`=`toyota`
# -  `porcshce`=`porsche`
# -  `maxda`=`mazda`
# -  `Nissan`=`nissan`

# In[ ]:


# mapping similar companies into one
car_df['CarName'] = car_df.CarName.str.replace('vw','volkswagen')
car_df['CarName'] = car_df.CarName.str.replace('vokswagen','volkswagen')
car_df['CarName'] = car_df.CarName.str.replace('toyouta','toyota')
car_df['CarName'] = car_df.CarName.str.replace('porcshce','porsche')
car_df['CarName'] = car_df.CarName.str.replace('maxda','mazda')
car_df['CarName'] = car_df.CarName.str.replace('Nissan','nissan')
car_df['CarName'].value_counts()


# -  We can see that 4 categorical columns `fueltype`,`aspiration`,`doornumber`,`enginelocation` having only two types of data in them, so in order to build a regression model we need to quantify them into Numerical values like 1 and 0.

# In[ ]:


print(car_df['fueltype'].value_counts())
print(car_df['aspiration'].value_counts())
print(car_df['doornumber'].value_counts())
print(car_df['enginelocation'].value_counts())


# -  Whichever is higher between two values of each category that will be denoted by 1, another will be 0, and column name will be changed according to it.

# In[ ]:


# quantifying into 1 and 0
car_df['fueltype'] = car_df['fueltype'].map({'gas': 1, 'diesel':0})
car_df['aspiration'] = car_df['aspiration'].map({'std': 1, 'turbo':0})
car_df['doornumber'] = car_df['doornumber'].map({'four': 1, 'two':0})
car_df['enginelocation'] = car_df['enginelocation'].map({'front': 1, 'rear':0})

car_df.head()


# #### # Dummy Variables for categorical columns:

# In[ ]:


# creating dummy variables for categorical columns
dummy_car_df = pd.get_dummies(car_df, drop_first=True)
dummy_car_df.head()


# In[ ]:


dummy_car_df.info()


# ## Step 4: Splitting the Data into Train and Test sets

# In[ ]:


# train-test split
np.random.seed(0)
df_train, df_test = train_test_split(dummy_car_df, train_size = 0.7, random_state=100)
print(df_train.shape)
df_train.head()


# #### # Scaling the Features:

# We should not rescale the `symboling` column as it defines `insurance risk rating` in range of `-3 to +3`.

# In[ ]:


# Apply scalar to all columns except 'quantified' and 'dummy' variables
vars_list = ['price','carlength','enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg']

scalar = MinMaxScaler()
df_train[vars_list] = scalar.fit_transform(df_train[vars_list])
df_train.head()


# In[ ]:


df_train.describe()


# #### # Dividing the Dataset into X_train and y_train for model building:

# In[ ]:


y_train = df_train.pop('price')
X_train = df_train
print(y_train.head())
X_train.head()


# ## Step 5: Building Linear Regression Model

# #### # Recursive Feature Elimination:

# In[ ]:


# RFE
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm,12)      # choosing top 12 features
rfe = rfe.fit(X_train,y_train)
list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[ ]:


# selecting the top 12 features
col = X_train.columns[rfe.support_]
col


# In[ ]:


# variables which are redundant
X_train.columns[~rfe.support_]


# In[ ]:


# eatures selected for model building
X_train_rfe = X_train[col]
X_train_rfe.head()


# #### # Building model using Statsmodel:

# In[ ]:


# add constant variable
X_train_rfe_1 = sm.add_constant(X_train_rfe)

# 1st linear model
lr_model_1 = sm.OLS(y_train,X_train_rfe_1).fit()

# summary
lr_model_1.summary()


# From summary what we get:
# 1.  None of the co-efficients are 0, they are having some values (positive as well as negative), so all of them are adding some efforts into the model. 
# 2.  `F-statistics` is 120.2, resulting in the Probablity 7.94e-63, so the overall model fit is significant.
# 3.  `Adjusted R-squraed` value is 0.902, that means 90% variance in `price` is described by the selected features.
# 4.  `p-values` for all of the features are 0 telling that they all are significant.

# Let's see the VIFs of the features for `multicollinearity`..

# #### # VIF:

# In[ ]:


# VIF of model_1
vif = pd.DataFrame()
X = X_train_rfe_1.drop('const',1)  # no need of 'const' in finding VIF
vif['features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending=False)
vif


# -  `cylindernumber_two` and `enginetype_rotor` are showing VIF as `inf`, because they are highly correlated with each other having `Pearson Correlation factor(R)` nearly equal to 1. we can see it by the heatmap also...

# In[ ]:


# heatmap
plt.figure(figsize=(15,8))
sns.heatmap(X.corr(), cmap='YlGnBu', annot=True)


# -  So, we should drop one of the `inf` feature..

# In[ ]:


# dropping 'enginetype_rotor' 
X_train_rfe_2 = X_train_rfe_1.drop('enginetype_rotor', axis=1)


# In[ ]:


# add constant variable
X_train_rfe_2 = sm.add_constant(X_train_rfe_2)

# 2nd linear model
lr_model_2 = sm.OLS(y_train,X_train_rfe_2).fit()

# summary
lr_model_2.summary()


# In[ ]:


# VIF of model_2
vif = pd.DataFrame()
X = X_train_rfe_2.drop('const',1)  # no need of 'const' in finding VIF
vif['features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending=False)
vif


# -  Now from heatmap we can see, `cylindernumber_four` is highly negatively correlated(-0.61) with `enginesize`, also having VIF = 22.95. Let's drop it.

# In[ ]:


# dropping 'cylindernumber_four'
X_train_rfe_3 = X_train_rfe_2.drop('cylindernumber_four',1)

# add constant variable
X_train_rfe_3 = sm.add_constant(X_train_rfe_3)

# 3rd linear model
lr_model_3 = sm.OLS(y_train,X_train_rfe_3).fit()

# summary
lr_model_3.summary()


# -  `Adj. R-squared` dropped to 0.885, and some of the p-values arises.

# In[ ]:


# VIF of model_3
vif = pd.DataFrame()
X = X_train_rfe_3.drop('const',1)  # no need of 'const' in finding VIF
vif['features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending=False)
vif


# -  `boreratio` having VIF `13.35` with p-value `0.013`, we'll drop it now...

# In[ ]:


# dropping 'boreratio'
X_train_rfe_4 = X_train_rfe_3.drop('boreratio',1)

# add constant variable
X_train_rfe_4 = sm.add_constant(X_train_rfe_4)

# 4th linear model
lr_model_4 = sm.OLS(y_train,X_train_rfe_4).fit()

# summary
lr_model_4.summary()


# Summary:
# -  `Adj. R-squared` is reduced `0.881`.
# -  `p-value` increased of some of them:
#     -  stroke 0.071
#     -  cylindernumber_three 0.069
#     -  cylindernumber_twelve 0.029

# In[ ]:


# VIF of model_4
vif = pd.DataFrame()
X = X_train_rfe_4.drop('const',1)  # no need of 'const' in finding VIF
vif['features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending=False)
vif


# -  Let's see the heatmap also, to get some insight...

# In[ ]:


# heatmap
plt.figure(figsize=(12,5))
sns.heatmap(X.corr(), cmap='YlGnBu', annot=True)


# -  `carlength` is correlated (0.7) with `enginesize` and also having VIF `13.52`.

# In[ ]:


# dropping 'carlength'
X_train_rfe_5 = X_train_rfe_4.drop('carlength',1)

# add constant variable
X_train_rfe_5 = sm.add_constant(X_train_rfe_5)

# 5th linear model
lr_model_5 = sm.OLS(y_train,X_train_rfe_5).fit()

# summary
lr_model_5.summary()


# In[ ]:


# VIF of model_5
vif = pd.DataFrame()
X = X_train_rfe_5.drop('const',1)  # no need of 'const' in finding VIF
vif['features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending=False)
vif


# - `stroke` is having p-value `0.023` as well as VIF `4.96`

# In[ ]:


# dropping 'stroke'
X_train_rfe_6 = X_train_rfe_5.drop('stroke',1)

# add constant variable
X_train_rfe_6 = sm.add_constant(X_train_rfe_6)

# 6th linear model
lr_model_6 = sm.OLS(y_train,X_train_rfe_6).fit()

# summary
lr_model_6.summary()


# -  p-value of `cylindernumber_three` is 0.218, which is not considerable.

# In[ ]:


# VIF of model_6
vif = pd.DataFrame()
X = X_train_rfe_6.drop('const',1)  # no need of 'const' in finding VIF
vif['features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending=False)
vif


# -  VIFs of all of them is in significant range (<2)
# -  So, `cylindernumber_three` should be dropped based on its p-value.

# In[ ]:


# dropping 'cylindernumber_three'
X_train_rfe_7 = X_train_rfe_6.drop('cylindernumber_three',1)

# add constant variable
X_train_rfe_7 = sm.add_constant(X_train_rfe_7)

# 7th linear model
lr_model_7 = sm.OLS(y_train,X_train_rfe_7).fit()

# summary
lr_model_7.summary()


# -  No significant change in `Adj. R-squared` (0.867), let's see the VIF also..

# In[ ]:


# VIF of model_7
vif = pd.DataFrame()
X = X_train_rfe_7.drop('const',1)  # no need of 'const' in finding VIF
vif['features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending=False)
vif


# -  VIFs are normal, let's see the correlation..

# In[ ]:


# heatmap
plt.figure(figsize=(8,5))
sns.heatmap(X.corr(), cmap='YlGnBu', annot=True)


# -  So, `cylindernumber_twelve` is positively correlated (0.41) with `enginesize`, and also having negative co-efficient (-0.2382), which we get from summary stats. There is a pssibility of `Multicollinearity`.

# In[ ]:


# dropping 'cylindernumber_twelve'
X_train_rfe_8 = X_train_rfe_7.drop('cylindernumber_twelve',1)

# add constant variable
X_train_rfe_8 = sm.add_constant(X_train_rfe_8)

# 8th linear model
lr_model_8 = sm.OLS(y_train,X_train_rfe_8).fit()

# summary
lr_model_8.summary()


# In[ ]:


# VIF of model_8
vif = pd.DataFrame()
X = X_train_rfe_8.drop('const',1)  # no need of 'const' in finding VIF
vif['features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending=False)
vif


# - Now, `VIF`s of all of the features are in considerable range, as well as `p-value`s are also 0 for all of them. So, we can consider it as our final model, having `Adj. R-squared: 0.861` and `F-statistic: 177.6` telling us that overall model fit is good, all of the final independent variables are able to define `86%` variance of our dependent variable `price`, claiming it is a pretty significant model. We can confirm it by feeding the Test data into the model.
# 

# -  Let's see the heatmap of the final independent features once for satisfaction..

# In[ ]:


# heatmap
plt.figure(figsize=(8,5))
sns.heatmap(X.corr(), cmap='YlGnBu', annot=True)


# ## Step 6: Residual Analysis on Train Data

# > #### # Distribution of error terms:
# which should be normally distributed by validating our assumptions.

# In[ ]:


y_train_pred = lr_model_8.predict(X_train_rfe_8)
residual = y_train - y_train_pred

plt.figure()
sns.distplot(residual, bins = 20)
plt.title('Error Terms', fontsize = 18)     
plt.xlabel('Errors')   


# -  So, the Error terms are normally distributed, which validates our one of the assumptions on residuals.

# #### # Finding patterns in the residuals:

# In[ ]:


fig = plt.figure(figsize=(18,5))
x_axis_range = [i for i in range(1,144,1)]
fig.suptitle('Error Terms', fontsize=20)

plt.subplot(1,2,1)
plt.scatter(x_axis_range, residual)
plt.ylabel('Residuals')

plt.subplot(1,2,2)
plt.plot(x_axis_range,residual, color="green", linewidth=2.5, linestyle="-")


# -  So, there is no such patterns in the Error terms, which validates our one of the major assumptions of residuals.

# ## Step 7: Prediction on Test Set

# #### # Scaling the test data:

# In[ ]:


vars_list = ['price','carlength','enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg']

df_test[vars_list] = scalar.transform(df_test[vars_list])
df_test.head()


# #### # Dividing tset set into X_test and y_test:

# In[ ]:


y_test = df_test.pop('price')
X_test = df_test
print(y_test.head())
X_test.head()


# In[ ]:


# creating X_test_new dataframe by selected columns from final model
X = X_train_rfe_8.drop('const',1)
X_test_new = X_test[X.columns]

# adding contant
X_test_new = sm.add_constant(X_test_new)
X_test_new.head()


# In[ ]:


# making prediction using final model
y_pred = lr_model_8.predict(X_test_new)
y_pred.head()


# ## Step 8: Model Evaluation

# #### # R-squared of Test set:

# In[ ]:


# R-squared value of test set
r2_score(y_test, y_pred)


# -  So R-squared of Test set is `0.866` which nearly same for Train data `0.861`, telling that none of the independent features in this model are redundant.

# #### # RMSE:

# In[ ]:


np.sqrt(mean_squared_error(y_test, y_pred))


# -  RMSE is lesser the better, for our case it is `0.085` telling our Prediction is very good.

# #### # Visualising the fit on Test set:

# In[ ]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure(figsize=(20,5))
fig.suptitle('y_test vs y_pred', fontsize=20)              

plt.subplot(1,2,1)
plt.scatter(y_test,y_pred)
plt.xlabel('y_test', fontsize=18)                          
plt.ylabel('y_pred', fontsize=16)                          

plt.subplot(1,2,2)
sns.regplot(y_test,y_pred)
plt.xlabel('y_test', fontsize=18)                          


# ### Conclusion:
# -  So, we got our best fitted line, and it is clearly showing the Linear Relationship between Train and Test data.
# -  The final equation of of Best-fitted line:
# 
# $ price = -0.0816 + 1.1205 \times enginesize + 0.2233 \times CarNamebmw + 0.2311 \times CarNameporsche + 0.1474 \times cylindernumberfive + 0.2513 \times cylindernumbertwo $
# 
# - The variables which are significant in predicting the Price of a car:
#     -  enginesize
#     -  CarName_bmw
#     -  cylindernumber_five
#     -  CarName_porsche	
#     -  cylindernumber_two

# In[ ]:




