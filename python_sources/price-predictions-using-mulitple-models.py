#!/usr/bin/env python
# coding: utf-8

# # 1. Data Wrangling
# ## Importing libraries and relevant data

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from scipy import stats
from scipy.stats import norm, skew


# In[ ]:


train_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train_df.head()


# Let us check the various features included in the dataframe.

# In[ ]:


train_df.columns.shape


# In[ ]:


train_df.columns


# As we have seen, we have about 81 different features in the dataframe. Hence, it is extremely important to downsize the number of features for model fitting. More over we need to check how the various features are distributed here.

# From the training dataframe, we are sure that we need to calculate the sale price of the various properties. Let us separate the sale price column.

# # Data preprocessing and visualisation

# In[ ]:


target=pd.DataFrame(train_df.iloc[:,-1],columns=['SalePrice'])
target.head()


# Moreover, ID column has no use for training purpose. Hence, we can simply drop this column

# In[ ]:


train_df.drop('Id',axis=1,inplace=True)


# Let us now check for any missing values that are present in the various features.

# In[ ]:


train_df.isna().any()


# As we can clearly see, there are some particular features which require our attention  due to the presence of empty values. Let us see how many empty entries are present in these features.

# In[ ]:


missing=pd.DataFrame(train_df.isna().sum().sort_values(ascending=False)[0:19],columns=['Missing values'])
missing.reset_index(inplace=True)
missing.rename(columns={'index':'Feature name'},inplace=True)


# In[ ]:


missing


# Let us visualise the above as barplots for better understanding.

# In[ ]:


plt.figure(figsize=(10,8))
sns.barplot('Feature name','Missing values',data=missing)
plt.xticks(rotation=45)


# As it can be clearly seen, we have extremely high number of missing values that have to be taken care of.

# ### Target variable distribution

# Let us also check how is the target variable distributed for us. 

# In[ ]:


plt.figure(figsize=(10,8))
sns.distplot(target['SalePrice'],fit=norm);
(mu, dev) = norm.fit(target['SalePrice'])
plt.xticks(rotation=45)
mu=np.round(target['SalePrice'].mean(),2)
dev=np.round(target['SalePrice'].std(),2)
plt.title('Target variable distribution')
plt.ylabel('Frequency')
plt.legend(['Normal dist. ($\mu=$ {} and $\sigma=$ {} )'.format(mu, dev)])


# In[ ]:


fig = plt.figure()
res = stats.probplot(target['SalePrice'], plot=plt)
plt.show()


# A scipy probplot generates a probability plot of sample data against the quantiles of a specified theoretical distribution (the normal distribution by default). probplot optionally calculates a best-fit line for the data and plots the results using Matplotlib or a given plot function.
# 
# If the blue and red line are on the same linear path at 45 degrees, we can confirm that our data is normally distributed.

# From the above distplot, it is clearly visible that the data is right skewed and hence, deviates from a normal distribution. Moreover, the data deviates from a linear red line. Hence, we must do some preprocessing to somehow, convert our distribution into a normal distribution. This will also help us use a linear model for the  purpose of regression.
# 
# 
# The reason for preprocessing to have a normal distribution is because most regression models perform far superior when it has a normally distributed data. Moreover, regression models such as the OLS linear model make the assumptions that it's residuals (or errors) are nomrally distributed in nature.  
# 
# To understand it better, this link below is helpful.
# 
# https://towardsai.net/p/data-science/how-when-and-why-should-you-normalize-standardize-rescale-your-data-3f083def38ff

# ## Transforming the data into a normal distribution
# 
# ### 1. Using power_transform

# In[ ]:


from sklearn.preprocessing import power_transform


# We can use two different methods here. These are namely:
# 
# 1. yeo-johnson: works with positive and negative values
# 2. box-cox: works with strictly postivie values
# 
# Let us check some more information on the SalePrice column

# In[ ]:


target['SalePrice'].describe()


# As we can see, all values here are positive. Hence, we can use any of the power_transform methods here.

# In[ ]:


target['Box-cox']=power_transform(target['SalePrice'].values.reshape(-1,1),method='box-cox',standardize=False)


# In[ ]:


sns.distplot(target['Box-cox'],fit=norm,color='indianred')
(mu, dev) = norm.fit(target['Box-cox'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, dev)])
plt.ylabel('Frequency')


# After checking the mean and standard deviation, it is clear that the distribution is normally distributed

# Let us check if the residuals are normally distributed using the probplot of scipy

# In[ ]:


fig = plt.figure()
res = stats.probplot(target['Box-cox'], plot=plt)
plt.show()


# As we can clearly see, our residuals are following the red line which basically means that they are normally distributed in nature. This is quite advatageous for model training purpose.

# ### 2. Logarithmic transformation

# In this case, instead of using the sklearn power_transform feature, we manually convert our SalePrices using a numpy transformation as shown below.
# 
# Generally, for a non normal data with high skewdness, a log transformation reduces the skew and makes the data approximately normally distributed.

# In[ ]:


target['Log prices']=np.log(target['SalePrice'])


# In[ ]:


sns.distplot(target['Log prices'],fit=norm,color='green')
(mu, dev) = norm.fit(target['Log prices'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, dev)])
plt.ylabel('Frequency')


# In[ ]:


fig = plt.figure()
res = stats.probplot(target['Log prices'], plot=plt)
plt.show()


# ### 3. Min-Max transformation

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler=MinMaxScaler()
target['Min-max']=scaler.fit_transform(target['SalePrice'].values.reshape(-1,1))


# In[ ]:


sns.distplot(target['Min-max'],fit=norm)


# In[ ]:


fig = plt.figure()
res = stats.probplot(target['Min-max'], plot=plt)
plt.show()


# As it can be clearly seen, the min-max transformation failed to normalise the results.

# ### 4. Robust Scaler

# In[ ]:


from sklearn.preprocessing import RobustScaler


# In[ ]:


rob=RobustScaler()
target['Robust scaler']=rob.fit_transform(target['SalePrice'].values.reshape(-1,1))


# In[ ]:


sns.distplot(target['Robust scaler'],fit=norm)


# In[ ]:


fig = plt.figure()
res = stats.probplot(target['Robust scaler'], plot=plt)
plt.show()


# Clearly, the robust scaler failed to normalise our data aswell.

# Hence, at the end, we can either select the power_transform or log transformation. Let us go ahead with the log transformation.

# ## Outlier detection for some features

# Let us check the scatter plots of numerical data with sale prices. We know, traditionally, area of the house is a strong indicator of price. After going through the other data, another few features that could play an important role in price can be shown here.
# 
# * GrLivArea
# * TotalBsmtSF
# * YrSold
# * OverQual
# * GarageArea
# * TotRmsAbvGrd
# * YearBuilt
# 
# 
# So, let us check their scatter plots and boxplots.

# ### A) GrLivArea Vs SalePrice

# In[ ]:


plt.figure(figsize=(10,8))
sns.regplot(train_df['GrLivArea'],target['SalePrice'],line_kws={"color": "red"})
plt.xlabel('Living area')
plt.ylabel('Sale price')
plt.title('Sale price vs Living Area')


# Let us visualise the above data with boxplots.

# In[ ]:


sns.boxplot(target['SalePrice'],orient='v')


# In[ ]:


target['SalePrice'].median()


# Let us drop the entries with outliers here. In our case, we will be dropping all the values that have a living area > 4500 sq feet and a sale price below 350000 will be dropped as this data is generating quite a lot of noise.

# In[ ]:


train_df = train_df.drop(train_df[(train_df['GrLivArea']>4500) & (train_df['SalePrice']<350000)].index)


# After dropping all these entries, we can make a regplot to see if the variations have reduced compared to what it was previously.

# In[ ]:


plt.figure(figsize=(10,8))
sns.regplot(train_df['GrLivArea'],train_df['SalePrice'],line_kws={"color": "red"})
plt.xlabel('Living area')
plt.ylabel('Sale price')
plt.title('Sale price vs Living Area without outliers')


# As we can clearly see, the variation of the red line is much lesser now. This shows that the data has much lesser noise now.

# ### B) TotalBsmtSF Vs SalePrice

# TotatBsmtSF feature tells us about the total basement area in square feet area. Let us check if there is enough correlation amongst saleprice and basement area.

# In[ ]:


plt.figure(figsize=(10,8))
sns.regplot(train_df['TotalBsmtSF'],train_df['SalePrice'],line_kws={'color':'green'})


# As we can clearly see, the outliers are little and hence, noise is not effecting the data much. It will be wise to not drop any data under these conditions. The general trend of the data seems to be linear aswell.

# ### C) YrSold Vs SalePrice

# Let us see how the year the property was sold in affects the sale price. This would give us an idea if prices are increasing or reducing in recent years. However, this visualisation needs to be taken with a grain of salt as properties with varying feature values such as area will be compared with one another. However, this will give us a preliminary idea of how the sale prices are varying with each passing year.
# 
# We shall compare the  data using boxplots. 

# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot('YrSold','SalePrice',data=train_df)


# As we can see from the boxplots, median prices of the properties have remained constant for all the years. 

# ### D) OverQual Vs SalePrice

# Let us check how does the over all quality of the houses relate to the sale prices. For some background, the numbers given to overquality maybe read as follows:
# 
#       10	Very Excellent
#        9	Excellent
#        8	Very Good
#        7	Good
#        6	Above Average
#        5	Average
#        4	Below Average
#        3	Fair
#        2	Poor
#        1	Very Poor

# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot('OverallQual','SalePrice',data=train_df)
plt.title('Overall quality Vs Sale prices')


# As was expected, with higher overall quality of the property, median prices of the properties are increasing.

# ### E) Garage area Vs SalePrice

# Let us check if garage area is correlated to sale prices in any way.

# In[ ]:


plt.figure(figsize=(10,8))
sns.regplot('GarageArea','SalePrice',data=train_df,line_kws={'color':'red'})
plt.title('Gara area vs Sale price')


# As we can see, we have a few outliers here that should take care of. Upon eyeballing at the data, we can see that there are cases where for very large garage area, the price is too low. Similary, we have cases with extremely high prices even with avaerage garage area. Let us drop some of these entries and check the plot again.

# In[ ]:


train_df=train_df.drop(train_df[(train_df['GarageArea']>1200)&(train_df['SalePrice']<300000)].index)


# In[ ]:


train_df=train_df.drop(train_df[(train_df['GarageArea']>600)&(train_df['GarageArea']<1000)&(train_df['SalePrice']>550000)].index)


# Let us now check the regplot after removing some of the outliers.

# In[ ]:


plt.figure(figsize=(10,8))
sns.regplot('GarageArea','SalePrice',data=train_df,line_kws={'color':'red'})
plt.title('Gara area vs Sale price')
plt.ylim(0,700000)


# The data looks far cleaner now with slightly lower variance to the linear regression line.

# ### F) TotRmsAbvGrd Vs SalePrice

# Let us check if there is any correlation between total rooms above ground and sale price

# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot('TotRmsAbvGrd','SalePrice',data=train_df)
plt.title('Total rooms above ground Vs Sale Price')


# The general trend is that as number of rooms increase above ground, median sale prices increase. Most houses have about 3-7 rooms above ground. Data for more than 7 rooms is low and hence, we can find a lot of variance in the data. However, this indicates that there is somewhat of a linear relation between total rooms and sale price.

# ### G) YearBuilt Vs SalePrice

# Let us check if the year in which, the house was built has any correlation with sale price.

# In[ ]:


plt.figure(figsize=(20,8))
sns.boxplot('YearBuilt','SalePrice',data=train_df)
plt.xticks(rotation=90)


# Although the correlation is a little weak, it can be said that houses built 1983 seems to have higher median prices with each passing year.

# However, there are extremely high number of features that have to be checked manually. We could reduce all this work using a correlation heat map to check which features may be of particular interest to us.

# ## Correlation heat map

# In[ ]:


corr=train_df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr,annot=True)


# As we can clearly see from the last row of the heatmap, some features are highly correlated while some are not. Infact, we should take care of possible multicollinearity in the data. Some data like Garage cars, Garage area show similar correlations. Hence, we can safely remove these features.
# 
# Let us check the top 10 most correlated features with sale price.

# In[ ]:


k=10
cols=corr.nlargest(k,'SalePrice')['SalePrice'].index
cols


# Hence, the features above are the most highly correlated. Let us make a smaller heatmap for the above.

# In[ ]:


corr_highest=train_df[cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_highest,annot=True,fmt='g')


# Some of the important conclusions that could be drawn are:
# 
# * Features like GarageCars and GarageArea show the same data. More cars would mean more garage area.
# * GrLivArea and 1stFlrSF should show the same results since 1st, 2nd or any other higher floors can be only as big as the ground floor area.

# Out of all these important correlations, we are yet to visualise how FullBath and YearRemodAdd correlates with SalePrice.
# 
# The full bath feature tells us the number of bathrooms present which are above the ground floor.
# 
# ### G) FullBath Vs SalePrice

# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot(train_df['FullBath'],train_df['SalePrice'])
plt.title('SalePrice Vs Full bath')


# Let us neglect the case where there are no full baths above ground (i.e. FullBath=0)
# 
# For other cases, it can be seen that the median prices are increasing as number of bathrooms above the ground floor is rising.

# ### H) YearRemodAdd Vs SalePrice

# In[ ]:


plt.figure(figsize=(20,8))
sns.boxplot(train_df['YearRemodAdd'],train_df['SalePrice'])
plt.title('Year remodelled Vs Sale Price')
plt.xticks(rotation=45)


# As the remodelled year is more recent, median sale prices are increasing. This confirms the well known fact that as the property has been remodelled in the recent past, it's prices will be higher.

# ## Processing the missing values

# In any machine learning model, we need to take care of the missing values. Let us check the presence of missing values in the features that we are planning to use in our models.

# In[ ]:


train_df.head()


# In[ ]:


train_df.isna().any()


# As we can see, quite a features have no missing values and hence, requires no intervention. Let us take care of the features that have missing values. Each feature requires different treatment. Let us start.
# 
# ### A) LotFrontage

# In[ ]:


train_df['LotFrontage'].describe()


# In[ ]:


sns.boxplot(train_df['LotFrontage'],orient='v')


# Due to the presence of outliers, it will be wise to fill the missing values with median value as shown below.

# In[ ]:


train_df['LotFrontage'].fillna(train_df['LotFrontage'].median(),inplace=True)


# ### B) Alley

# In[ ]:


train_df['Alley'].isna().value_counts()


# According to the provided data, 1358 properties have missing values. According to the data description, missing values means there is no alley access. Hence, we shall fill the missing values with "None" 

# In[ ]:


train_df['Alley'].fillna('None',inplace=True)


# ### C)  MasVnrType

# In[ ]:


train_df['MasVnrType'].isna().value_counts()


# Only 8 entries have missing values.

# In[ ]:


train_df['MasVnrType'].value_counts()


# In[ ]:


train_df['MasVnrType'].fillna(train_df['MasVnrType'].mode()[0],inplace=True)


# We have filled all missing values with the most common (or mode) of this feature.

# ### D) MasVnrArea

# In[ ]:


train_df['MasVnrArea'].isna().value_counts()


# In[ ]:


train_df['MasVnrArea'].fillna(train_df['MasVnrArea'].median(),inplace=True)


# ### E) 'BsmtQual' ,'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'

# For all the basement features, missing values actually means there are no basements. Hence, missing values will be replaced by none.

# In[ ]:


basement_features=['BsmtQual' ,'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

for feature in basement_features:
    train_df[feature]=train_df[feature].fillna('None')


# In[ ]:


train_df['BsmtQual'].isna().value_counts()


# In[ ]:


train_df['BsmtQual'].value_counts()


# ### F) Electrical

# All properties should have electrical connections as a mandatory facility. Hence, it is odd to see any empty values in this feature. Let us check how many empty values are there.

# In[ ]:


train_df['Electrical'].isna().value_counts()


# As expected, we have only one single value with an empty value. This could be due to a mistake in entering the data. Hence, we will fill this value with the most common data under this feature.

# In[ ]:


train_df['Electrical'].fillna(train_df['Electrical'].mode()[0],inplace=True)


# ### G) FireplaceQu, GarageType, GarageYrBlt,GarageFinish ,GarageQual,GarageCond ,PoolQC,Fence,                         MiscFeature 

# For all the above features, missing values denotes that this feature is not present in the property. This has been clearly mentioned in the data description. Hence, we replace the empty values as "None"

# In[ ]:


misc_features=['FireplaceQu', 'GarageType', 'GarageYrBlt','GarageFinish' ,
               'GarageQual','GarageCond' ,'PoolQC','Fence','MiscFeature' ]


# In[ ]:


for misc in misc_features:
    train_df[misc].fillna('None',inplace=True)


# After all this is done, we should not expect anymore missing values in the dataframe. Let us check once more to confirm.

# In[ ]:


train_df.isna().any()


# ### Ordinal encoding the catergorical features

# In[ ]:


from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()


# In[ ]:


cat_features=['FireplaceQu', 'BsmtQual', 'BsmtCond','GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold']


# In[ ]:


for cat in cat_features:
    train_df[cat]=oe.fit_transform(train_df[cat].values.reshape(-1,1))
    


# In[ ]:


train_df.head()


# Let us drop the garage year built feature since it isn't giving us much information that will help us predict sale price

# In[ ]:


train_df.drop('GarageYrBlt',axis=1,inplace=True)


# ### One hot encoding the remaining categorical data

# In[ ]:


train_df=pd.get_dummies(train_df)


# Upon doing the above operation, we must now have all the data in either int or float form. Let us take a quick look if it is indeed the case.

# In[ ]:


train_df.dtypes


# As we can see, all the given data is now in numerical form. We should now separate the target feature which is the SalePrice from the training dataframe.

# In[ ]:


target_df=pd.DataFrame(train_df['SalePrice'],columns=['SalePrice'])
target_df.head()


# In[ ]:


train_df.drop('SalePrice',axis=1,inplace=True)


# In[ ]:


train_df.head()


# The above dataframe is now ready for machine learning purpose. However, as we will be utilising a few linear models, it is important for us to use standard scaler in our dataframe. This will help us give equal important to the features.
# 
# ### Standard scaling the input data

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler=StandardScaler()
X_scaled_input=scaler.fit_transform(train_df)


# ### Log transforming the target data

# In[ ]:


target_df['LogSalePrice']=np.log(target_df['SalePrice'])
target_df.head()


# # 2. Machine Learning
# 
# ## Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics.regression import r2_score,explained_variance_score
from sklearn.model_selection import train_test_split


# In[ ]:


reg_lin=LinearRegression()


# In[ ]:


X=train_df
y=target_df['SalePrice'].values.astype(float)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[ ]:


reg_lin.fit(X_train,y_train)


# In[ ]:


y_pred=reg_lin.predict(X_test)


# In[ ]:


reg_lin.score(X_train,y_train)


# In[ ]:


reg_lin.score(X_test,y_test)


# In[ ]:


lin_pred_df=pd.DataFrame(columns=['Actual values','Predicted values'])


# In[ ]:


lin_pred_df['Actual values']=y_test
lin_pred_df['Predicted values']=y_pred
lin_pred_df['Absolute difference']=abs(lin_pred_df['Actual values']-lin_pred_df['Predicted values'])
lin_pred_df['Residual']=lin_pred_df['Actual values']-lin_pred_df['Predicted values']
lin_pred_df.head()


# In[ ]:


lin_pred_df['Residual'].describe()


# In[ ]:


plt.scatter(lin_pred_df['Actual values'],lin_pred_df['Predicted values'])


# In[ ]:


sns.distplot(lin_pred_df['Residual'])


# The residuals seem to be normally distributed which is a good sign.

# ## Lasso Regression

# In[ ]:


X=X_scaled_input
y=target_df['LogSalePrice'].values.astype(float)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
reg_las=Lasso(alpha =0.0005, random_state=1,normalize=True)


# In[ ]:


reg_las.fit(X_train,y_train)


# In[ ]:


y_pred=reg_las.predict(X_test)


# In[ ]:


reg_las.score(X_test,y_test)


# In[ ]:


reg_las.score(X_train,y_train)


# In[ ]:


las_pred_df=pd.DataFrame(columns=['Actual values','Predicted values','Absolute difference'])
las_pred_df['Actual values']=np.exp(y_test)
las_pred_df['Predicted values']=np.exp(y_pred)
las_pred_df['Absolute difference']=abs(las_pred_df['Actual values']-las_pred_df['Predicted values'])
las_pred_df['Residual']=las_pred_df['Actual values']-las_pred_df['Predicted values']
las_pred_df['Difference %']=100*las_pred_df['Residual']/las_pred_df['Actual values']
las_pred_df


# In[ ]:


las_pred_df['Absolute difference'].describe()


# Let us check how the residuals are distributed.

# In[ ]:



sns.distplot(las_pred_df['Residual'])
plt.title('Residual PDF',size=20)


# Under best case scenario,our residuals should follow a normal distribution. This case is true. However, we do have a few extremely positive values in the residuals where the model didn't perform too well. This indicates room for model improvement.
# 
# 
# Let us see the scatter plot of predicted vs actual results to get an idea of how the model is performing.

# In[ ]:


sns.regplot(las_pred_df['Actual values'],las_pred_df['Predicted values'],line_kws={'color':'red'})


# As we can see, the model is performing well upto the point where the prices are high. It has a tough time predicting correctly at this point.
# 
# Let us finally check the bias in our model.

# In[ ]:


bias=np.exp(reg_las.intercept_)
bias


# ## Ridge Regression

# In[ ]:


reg_rid=Ridge()


# In[ ]:


reg_rid.fit(X_train,y_train)


# In[ ]:


y_pred=reg_rid.predict(X_test)


# In[ ]:


reg_rid.score(X_train,y_train)


# In[ ]:


reg_rid.score(X_test,y_test)


# In[ ]:


r2_score(y_pred,y_test)


# In[ ]:


explained_variance_score(y_pred,y_test)


# In[ ]:


rid_df=pd.DataFrame(columns=['Actual values','Predicted values','Absolute difference'])
rid_df['Actual values']=np.exp(y_test)
rid_df['Predicted values']=np.exp(y_pred)
rid_df['Absolute difference']=abs(rid_df['Actual values']-rid_df['Predicted values'])
rid_df['Residual']=rid_df['Actual values']-rid_df['Predicted values']
rid_df['Difference %']=100*rid_df['Residual']/rid_df['Actual values']
rid_df.head()


# In[ ]:


sns.distplot(rid_df['Residual'])
plt.title('Residual PDF',size=20)


# The distribution of the residuals is normally distributed in Ridge regression which is better than what we got in Lasso regression.

# In[ ]:


sns.regplot(rid_df['Actual values'],rid_df['Predicted values'],line_kws={'color':'red'})


# ## Ensemble models

# ### Random Forest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


X=train_df
y=target_df['SalePrice']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=0)


# In[ ]:


rfr=RandomForestRegressor()


# In[ ]:


rfr.fit(X_train,y_train)


# In[ ]:


y_pred=rfr.predict(X_test)


# In[ ]:


rfr.score(X_train,y_train)


# In[ ]:


rfr.score(X_test,y_test)


# In[ ]:


rfr_df=pd.DataFrame(columns=['Actual values','Predicted values','Absolute difference'])
rfr_df['Actual values']=y_test
rfr_df['Predicted values']=y_pred
rfr_df['Absolute difference']=abs(rfr_df['Actual values']-rfr_df['Predicted values'])
rfr_df['Residual']=rfr_df['Actual values']-rfr_df['Predicted values']
rfr_df['Difference %']=100*rfr_df['Residual']/rfr_df['Actual values']
rfr_df.head()


# In[ ]:


sns.distplot(rfr_df['Residual'])
plt.title('Residual PDF',size=20)


# In[ ]:


sns.regplot(rfr_df['Actual values'],rfr_df['Predicted values'],line_kws={'color':'green'})


# ### GBDT regressor

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


gbdt=GradientBoostingRegressor()


# In[ ]:


gbdt.fit(X_train,y_train)


# In[ ]:


y_pred=gbdt.predict(X_test)


# In[ ]:


gbdt.score(X_train,y_train)


# In[ ]:


gbdt.score(X_test,y_test)


# In[ ]:


gbdt_df=pd.DataFrame(columns=['Actual values','Predicted values','Absolute difference'])
gbdt_df['Actual values']=y_test
gbdt_df['Predicted values']=y_pred
gbdt_df['Absolute difference']=abs(gbdt_df['Actual values']-gbdt_df['Predicted values'])
gbdt_df['Residual']=gbdt_df['Actual values']-gbdt_df['Predicted values']
gbdt_df['Difference %']=100*gbdt_df['Residual']/gbdt_df['Actual values']
gbdt_df.head()


# In[ ]:


sns.distplot(gbdt_df['Residual'])


# Residuals seem to be normally distributed with some skew in the positive direction.

# In[ ]:


sns.regplot(gbdt_df['Actual values'],gbdt_df['Predicted values'],line_kws={'color':'red'})


# ### XGboost

# In[ ]:


import xgboost as xgb


# In[ ]:


xgb_reg=xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# The hyper parameters were taken from a kaggle kernel for which, the link is shared below:
# 
# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard#Modelling

# In[ ]:


xgb_reg.fit(X_train,y_train)


# In[ ]:


xgb_reg.score(X_train,y_train)


# In[ ]:


xgb_reg.score(X_test,y_test)


# In[ ]:


y_pred=xgb_reg.predict(X_test)


# In[ ]:


xgb_df=pd.DataFrame(columns=['Actual values','Predicted values','Absolute difference'])
xgb_df['Actual values']=y_test
xgb_df['Predicted values']=y_pred
xgb_df['Absolute difference']=abs(xgb_df['Actual values']-xgb_df['Predicted values'])
xgb_df['Residual']=xgb_df['Actual values']-xgb_df['Predicted values']
xgb_df['Difference %']=100*xgb_df['Residual']/xgb_df['Actual values']
xgb_df.head()


# In[ ]:


sns.distplot(xgb_df['Residual'])


# Other than a few residuals at the higher positive values, rest of the residuals are normally distributed.

# In[ ]:


sns.regplot(xgb_df['Actual values'],xgb_df['Predicted values'],line_kws={'color':'green'})
plt.title('Prediction chart',size=20)


# Most values do seem to be approximately correctly predicted by the model. Some are however deviating from the green line.

# ### Light GBM

# In[ ]:


from lightgbm import LGBMRegressor


# In[ ]:


lgb=LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# Hyper parameters have been chosen from the same kaggle kernel used for XGBoost.

# In[ ]:


X=train_df
y=target_df['SalePrice'].values.astype(float)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)


# In[ ]:


lgb.fit(X_train,y_train)


# In[ ]:


lgb.score(X_train,y_train)


# In[ ]:


lgb.score(X_test,y_test)


# In[ ]:


y_pred=lgb.predict(X_test)


# In[ ]:


lgb_df=pd.DataFrame(columns=['Actual values','Predicted values','Absolute difference'])
lgb_df['Actual values']=y_test
lgb_df['Predicted values']=y_pred
lgb_df['Absolute difference']=abs(lgb_df['Actual values']-lgb_df['Predicted values'])
lgb_df['Residual']=lgb_df['Actual values']-lgb_df['Predicted values']
lgb_df['Difference %']=100*lgb_df['Residual']/lgb_df['Actual values']
lgb_df.head()


# In[ ]:


sns.distplot(lgb_df['Residual'])
plt.title('Residual PDF',size=20)


# In[ ]:


sns.regplot(lgb_df['Actual values'],lgb_df['Predicted values'],line_kws={'color':'red'})
plt.title('Prediction chart',size=20)


# # 3. Testing phase

# In[ ]:


test_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


test_df.head()


# In[ ]:


train_df.head()


# In[ ]:


test_id=pd.DataFrame(test_df.iloc[:,0],columns=['Id'])
test_id.head()


# We store all the IDs as a separate dataframe to preserve the order of our data. This dataframe will be concatenated with the predicted prices later.

# In[ ]:


test_df.isna().any()


# In[ ]:


missing_test=pd.DataFrame(test_df.isna().sum().sort_values(ascending=False)[0:33],columns=['Missing values'])
missing_test.reset_index(inplace=True)
missing_test.rename(columns={'index':'Feature name'},inplace=True)
missing_test


# In[ ]:


plt.figure(figsize=(10,8))
sns.barplot('Feature name','Missing values',data=missing_test)
plt.xticks(rotation=90)


# Let us take care of all the missing values in the feature columns.

# In[ ]:


test_df['LotFrontage'].fillna(test_df['LotFrontage'].median(),inplace=True)
test_df['Alley'].fillna('None',inplace=True)
test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode()[0],inplace=True)
test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].median(),inplace=True)
basement_features=['BsmtQual' ,'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

for feature in basement_features:
    test_df[feature]=test_df[feature].fillna('None')
    
test_df['Electrical'].fillna(test_df['Electrical'].mode()[0],inplace=True)


misc_features=['FireplaceQu', 'GarageType', 'GarageYrBlt','GarageFinish' ,
               'GarageQual','GarageCond' ,'PoolQC','Fence','MiscFeature' ]

for misc in misc_features:
    test_df[misc].fillna('None',inplace=True)

test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0],inplace=True)
test_df['Utilities'].fillna(test_df['Utilities'].mode()[0],inplace=True)
test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0],inplace=True)
test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0],inplace=True)
test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].median(),inplace=True)
test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].median(),inplace=True)
test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].median(),inplace=True)
test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].median(),inplace=True)
test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0],inplace=True)


baths=['BsmtFullBath','BsmtHalfBath']
for types in baths:
    test_df[types].fillna(test_df[types].mode()[0],inplace=True)
    
test_df['Functional'].fillna(test_df['Functional'].mode()[0],inplace=True)
test_df['GarageCars'].fillna(test_df['GarageCars'].mode()[0],inplace=True)
test_df['GarageArea'].fillna(test_df['GarageArea'].median(),inplace=True)
test_df['SaleType'].fillna(test_df['SaleType'].mode()[0],inplace=True)


# In[ ]:


test_df.isna().any()


# In[ ]:


for cat in cat_features:
    test_df[cat]=oe.fit_transform(test_df[cat].values.reshape(-1,1))
    


# In[ ]:


test_df.drop('GarageYrBlt',axis=1,inplace=True)


# In[ ]:


test_df.drop('Id',axis=1,inplace=True)


# In[ ]:


test_df=pd.get_dummies(test_df)


# In[ ]:


test_df.dtypes


# ['Exterior1st_Stone', 'HouseStyle_2.5Fin', 'RoofMatl_Metal', 'Exterior2nd_Other', 'Heating_OthW', 'Condition2_RRNn', 'Utilities_NoSeWa', 'Heating_Floor', 'Condition2_RRAn', 'RoofMatl_Membran', 'Condition2_RRAe', 'Exterior1st_ImStucc', 'RoofMatl_Roll', 'MiscFeature_TenC', 'Electrical_Mix']
# 
# 
# As we can see, the above particular columns are present in training dataframe but not in test dataframe. Hence, we will add these features with 0 in the test dataframe to be used in the model.

# In[ ]:


missing_cols=['Condition2_RRAe', 'Utilities_NoSeWa', 'RoofMatl_Metal', 
              'Condition2_RRAn', 'Exterior2nd_Other', 'MiscFeature_TenC', 
              'RoofMatl_Roll', 'Electrical_Mix', 'RoofMatl_Membran', 
              'Exterior1st_ImStucc', 'Condition2_RRNn', 'Heating_Floor', 
              'Heating_OthW','HouseStyle_2.5Fin', 'Exterior1st_Stone']


# In[ ]:


for cols in missing_cols:
    test_df[cols]=0


# In[ ]:


test_df=test_df[train_df.columns]


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# The above dataframe is now preprocessed and can be used for final model prediction.
# 
# 
# ### Linear regression model

# In[ ]:


X_test=test_df
X_scaled_test=scaler.fit_transform(X_test)


# In[ ]:


y_pred=reg_lin.predict(X_test)


# In[ ]:


linear_reg_final=pd.DataFrame(columns=['Id','SalePrice'])
linear_reg_final['Id']=test_id['Id']
linear_reg_final['SalePrice']=y_pred
linear_reg_final.head()


# ### Lasso regression model

# In[ ]:


y_pred=reg_las.predict(X_scaled_test)


# In[ ]:


lasso_final=pd.DataFrame(columns=['Id','SalePrice'])
lasso_final['Id']=test_id['Id']
lasso_final['SalePrice']=np.exp(y_pred)
lasso_final.head()


# ### Ridge regression model

# In[ ]:


y_pred=reg_rid.predict(X_scaled_test)


# In[ ]:


ridge_final=pd.DataFrame(columns=['Id','SalePrice'])
ridge_final['Id']=test_id['Id']
ridge_final['SalePrice']=np.exp(y_pred)
ridge_final.head()


# ### Random forest regressor

# In[ ]:


y_pred=rfr.predict(X_test)


# In[ ]:


rf_final=pd.DataFrame(columns=['Id','SalePrice'])
rf_final['Id']=test_id['Id']
rf_final['SalePrice']=y_pred
rf_final.head()


# ### XGboost regressor

# In[ ]:


y_pred=xgb_reg.predict(X_test)


# In[ ]:


xgb_final=pd.DataFrame(columns=['Id','SalePrice'])
xgb_final['Id']=test_id['Id']
xgb_final['SalePrice']=y_pred
xgb_final.head()


# ### GBDT regressor

# In[ ]:


y_pred=gbdt.predict(X_test)


# In[ ]:


gbdt_final=pd.DataFrame(columns=['Id','SalePrice'])
gbdt_final['Id']=test_id['Id']
gbdt_final['SalePrice']=y_pred
gbdt_final.head()


# ### Light GBM

# In[ ]:


y_pred=lgb.predict(X_test)


# In[ ]:


lgb_final=pd.DataFrame(columns=['Id','SalePrice'])
lgb_final['Id']=test_id['Id']
lgb_final['SalePrice']=y_pred
lgb_final.head()


# In[ ]:


lgb_final.to_csv('LGBM_Regression.csv',index=False)


# In[ ]:




