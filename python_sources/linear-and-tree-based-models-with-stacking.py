#!/usr/bin/env python
# coding: utf-8

# # Predicting house prices with advanced regression models

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,RobustScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.set() ## set Seaborn defaults


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train.head()


# ## Exploratory data analysis

# - First, let us take a close look at the correlations between the predictor variables and the target variable.

# In[ ]:


# Correlation matrix
plt.figure(figsize=(14,10))
sns.heatmap(train.corr())  ## only for numerical features


# What is interesting in the above plot is the last row, i.e. `SalePrice`. We notice quite a few red squares there which signifies high correlation. These features will have the the highest predictive power in our ML models. We will now take a detailed look at these features. 
# 
# Note: One can see two 2x2 white squares along the diagonal. It implies that the pairs `TotalBsmtSF` & `1stFlrSF`, and `GarageCars` & `GarageArea` are highly correlated, which is a sign of **multicollinearity**. 

# In[ ]:


train.corr()['SalePrice'].sort_values(ascending=False)[1:]


# #### What we will do now is to do an exploratory analysis of those features which show the *strongest correlation* with our target variable `SalePrice`.

# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x=train['OverallQual'],y = train['SalePrice'])


# 
# Such a nice relationship! Up to `OVerallQual` = 6 it hs a relatively gentle slope, but in the regime `OVerallQual` > 6, `SalePrice` really takes off and shows a steeper trend. One can tell that this will be one of the strong predictor variables while analyzing feature importances. I believe that this feature is a combination of other predictor variables in the dataset and therefore we get such a smooth and clean relationship.

# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey= True,sharex=True)
sns.scatterplot(x=train['GrLivArea'], y=train['SalePrice'],ax=axs[0])
sns.scatterplot(train['TotalBsmtSF'],train['SalePrice'], ax=axs[1])


# As expected, both `TotalBsmtSF` and `GrLivArea` show strong linear correlation with `SalePrice`. Furthermore, `SalePrice` shows a steeper relationship with `TotalBsmtSF` (both plots are drawn on the same scale).

# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(train['GrLivArea'], label = 'GrLivArea',kde=False)
sns.distplot(train['TotalBsmtSF'], label = 'TotalBsmtSF',kde=False)
plt.legend()
plt.xlabel('')
plt.ylabel('Count')
plt.title('Histogram of GrLivArea and TotalBsmtSF')


# Both **GrLivArea** and **TotalBsmtSF** show positive skewness. We will do the normality tests later in the notebook.

# ### Removing outliers

# In[ ]:


sns.scatterplot(train['GrLivArea'], train['SalePrice'])


# - As evident from the living area vs. SalePrice plot, there are 2 clear outliers that one would wish to remove as part of the data cleaning process before proceeding to the modelling section. The [paper](http://jse.amstat.org/v19n3/decock.pdf) which introduced this dataset recommends removing all the 4 data points in the training set which has more than 4000 sqft of living area. But I chose to remove only those two points which correspond to huge houses priced rather inappropriately.

# In[ ]:


outliers_index = train.loc[train['GrLivArea']>4500, :].index
train = train.drop(outliers_index).reset_index(drop=True)
target = train['SalePrice'].reset_index(drop=True)  ## target variable that we need to predict


# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(x=train['GarageCars'], y=target,ax=axs[0])
sns.scatterplot(x=train['GarageArea'],y=target,ax=axs[1])


# In[ ]:


sns.distplot(train.loc[train['GarageArea']>0, 'GarageArea'],kde=False)


# In[ ]:


train['GarageCars'].value_counts()
#train.groupby('GarageCars').size()


# In[ ]:


train['YearBuilt'].describe()


# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(train['YearBuilt'],kde=False)
plt.ylabel('Count')


# 50% of the houses sold were constructed after 1973, i.e. within the last 4 decades of the dataset. It is probably not a surprise considering the fact that the dataset contains sales data from 2006 to 2010 only. You are much less likely to buy a hundred-year old house.

# In[ ]:


fig, axs = plt.subplots(2, 2, figsize=(14, 10))
sns.scatterplot(x=train['1stFlrSF'], y=train['SalePrice'],ax=axs[0][0])
sns.scatterplot(x=train['YearBuilt'], y=train['SalePrice'],hue=train['GrLivArea'] ,ax=axs[0][1])
sns.boxplot(train['FullBath'],train['SalePrice'], ax=axs[1][0])
sns.boxplot(x=train['TotRmsAbvGrd'], y=train['SalePrice'],ax=axs[1][1])


# We can see a strong correlation between `1stFlrSF` and `SalePrice`. The correlation of SalePrice with YearBuilt has to be taken with a pich of salt since the sales data is only collected between 2006-2010. That's exactly why older houses fetch lower prices with the exception of very big houses (living area is colour-coded in the figure).

# In[ ]:


garage_cat_cols = ['GarageQual', 'GarageCond','GarageFinish']
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for i in range(len(garage_cat_cols)):
    sns.boxplot(x=train[garage_cat_cols[i]],y=target,ax=axs[i])


# While looking at the boxplots, it is worthwhile to keep in mind the number of data points available in each category. Please have a look at the category-wise value counts in the table below.
# 
# Label explanation from the documentation:
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
#        Fin	Finished
#        RFn	Rough Finished	
#        Unf	Unfinished
#        NA	No Garage

# In[ ]:


pd.concat([train['GarageQual'].value_counts(),train['GarageCond'].value_counts()],axis=1,sort=True)


# In[ ]:


sns.scatterplot(train['YearRemodAdd'],train['SalePrice'])
# YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)


# Houses remodelled before 1990 does not seem to attract a high price. One is much better off buying a relatively recently remodelled house. 
# 
# Out of curiosity, I decided to check the OverallQual of all the remodelled houses. It's reassuring that recent houses are better rated overall (on average).

# In[ ]:


sns.scatterplot(train['YearRemodAdd'], train['OverallQual'])
train[['YearRemodAdd', 'OverallQual']].groupby('YearRemodAdd').mean().plot()
#train[['YearBuilt', 'OverallQual']].groupby('YearBuilt').mean().plot()


# #### How does Neigbourhood influence SalePrice?

# In[ ]:


plt.figure(figsize=(15,8))
sns.boxplot(train['Neighborhood'], target)
plt.xticks(rotation='90')


# It appears that **NoRidge** (Northridge), **NridgHt** (Northridge Heights) and **StoneBr** (Stone Brook) are the areas where the rich people of Ames prefer to stay.

# ### Missing values

# In[ ]:


train = train.drop(['SalePrice','Id'], axis=1)
test_id = test['Id']
test = test.drop('Id', axis=1)
full_data = pd.concat([train,test],axis=0,ignore_index=True)


# In[ ]:


def missing_data_df(df):
    '''Returns a pandas series showing which features of the input dataframe have missing values'''
    missing_df = df.isnull().sum(axis=0)
    missing_df =  missing_df[missing_df!=0]
    return missing_df


# In[ ]:


missing_train_test = pd.concat([train.isnull().sum(), test.isnull().sum(),full_data.isnull().sum()], axis=1, keys=['Train', 'Test', 'Full'], sort=True)
missing_train_test[missing_train_test.sum(axis=1) > 0]


# In[ ]:


plt.figure(figsize=(20,8))
sns.barplot(x=missing_data_df(full_data).index,y = missing_data_df(full_data)) # How many missing values are there in the trainset?
plt.xticks(rotation='90')
plt.title('Number of missing values in the (train + test) set')


# ### Imputing missing values

# - Numeric columns

# In[ ]:


num_cols = train.select_dtypes(include=[np.number]).columns
cat_cols = train.select_dtypes(exclude=[np.number]).columns


# In[ ]:


missing_data_df(full_data[num_cols])


# - It seems that `LotFrontage` has the most number of missing values. From the scatterplot below we can even spot an outlier. I will not remove it since I don't expect this variable to be crucial in determining `SalePrice`.

# In[ ]:


sns.scatterplot(train['LotFrontage'], target)


# In[ ]:


train.loc[train['LotFrontage']>300, 'GrLivArea']


# In[ ]:


# This cell imputes missing values for numeric columns
fill_with_zero = ['MasVnrArea','GarageYrBlt','GarageArea', 'GarageCars']
fill_with_mean = ['LotFrontage','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']
fill_with_mode = ['BsmtFullBath','BsmtHalfBath']

full_data.loc[:,fill_with_zero] = full_data[fill_with_zero].fillna(0)
full_data.loc[:,fill_with_mean] = full_data[fill_with_mean].fillna(train[fill_with_mean].mean())
for col in fill_with_mode:
    full_data[col] = full_data[col].fillna(train[col].mode()[0])


# - Categorical columns

# In[ ]:


missing_data_df(full_data[cat_cols])


# In[ ]:


# This cell imputes missing values for categorical columns

fill_with_mode_cat = ['MSZoning','MasVnrType','Electrical', 'KitchenQual','Functional', 'SaleType']
fill_with_none = ['Alley','Exterior1st', 'Exterior2nd','BsmtQual', 'BsmtCond', 
                  'BsmtExposure','BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
                 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature' ]

full_data.loc[:,fill_with_none] = full_data[fill_with_none].fillna('None')
for col in fill_with_mode_cat:
    full_data[col] = full_data[col].fillna(train[col].mode()[0])


# In[ ]:


full_data['Utilities'].value_counts()


# As you can see from the value counts above, all but one house does not have water connection (NoSeWa). Therefore, it makes no sense to keep the `Utilities` predictor.

# In[ ]:


full_data = full_data.drop('Utilities', axis=1)


# In[ ]:


print('Number of missing values present = %i' %np.sum(full_data.isnull().sum(axis=0)))


# ### Feature engineering

# - Add the total area of a house (including Garage) as a new predictor variable in out dataset

# In[ ]:


## Add the total area of a house (including Garage) as a new predictor variable in out dataset
full_data['total_area'] = full_data[['GrLivArea','TotalBsmtSF','GarageArea']].sum(axis=1)


# - Since `GrLivArea` is the feature which seems to predict the `SalePrice` best, let's create a feature which is square of the living area.

# In[ ]:


full_data['GrLivArea_sq'] = full_data['GrLivArea']**2


# - Total number of bathrooms might be a good choice? The half-bathrooms are weighted accordingly.

# In[ ]:


full_data['Total_bath'] = full_data['FullBath']+0.5*full_data['HalfBath']+full_data['BsmtFullBath']+0.5*full_data['BsmtHalfBath']


# - Total porch area

# In[ ]:


full_data['Total_porchSF'] = full_data['WoodDeckSF']+full_data['OpenPorchSF']+full_data['EnclosedPorch']+full_data['3SsnPorch']+full_data['ScreenPorch']


# In[ ]:


full_data['TotRmsandBathAbvGrd']  = full_data['TotRmsAbvGrd'] + full_data['FullBath'] + full_data['HalfBath']


# In[ ]:


full_data['yrs_since_renov'] = full_data['YearRemodAdd'] - full_data['YearBuilt']
full_data.loc[full_data['yrs_since_renov']<0, 'yrs_since_renov'] = 0


# In[ ]:


sns.scatterplot(full_data.loc[:len(train)-1, 'yrs_since_renov'], target)
#plt.ylim(0,3e5)


# In[ ]:


np.corrcoef(full_data.loc[:len(target)-1,'yrs_since_renov'], target)


# In[ ]:


num_cols = full_data.select_dtypes(include=[np.number]).columns
cat_cols = full_data.select_dtypes(exclude=[np.number]).columns


# ### Analying `SalePrice`: the target variable
# 
#  - The target variable for linear regression should ideally be normally distributed. Since the living area and overall quality show strong linear trends, it seems that linear models might be good predictors of sale price.
#  - The sale price in this case is **positively skewed**, meaning tht it has a thicker til to the right. This seems plausible because there will be very few rich people who can afford the most expensive houses.

# In[ ]:


plt.figure(figsize=(10,6))
z = (target- np.mean(target))/np.std(target)
ax = sns.distplot(z,rug=True)
ax.set(xlabel = 'Normalized Sale price (z-scores)')


#  - The Q-Q plot is a good visual test of normality. If the data is perfectly normal dstributed, then all the points should lie along the red line.

# In[ ]:


from scipy import stats
stats.probplot(target, plot=plt)


# In[ ]:


target = np.log1p(target)
# https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html


# In[ ]:


stats.probplot(target, plot=plt)
plt.title('after transformation of SalePrice')


# ### Analyzing skewed variables

# - Let's look at the QQ plots of the highest correlated variables

# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(12,6))
stats.probplot(full_data.loc[full_data['GrLivArea']>0, 'GrLivArea'], plot=ax[0])
stats.probplot(full_data.loc[full_data['TotalBsmtSF']>0, 'TotalBsmtSF'], plot=ax[1])


# - After log transformation

# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(12,6))
stats.probplot(np.log(full_data.loc[full_data['GrLivArea']>0, 'GrLivArea']), plot=ax[0])
stats.probplot(np.log(full_data.loc[full_data['TotalBsmtSF']>0, 'TotalBsmtSF']), plot=ax[1])
ax[0].set_title('GrLivArea after Log transform')
ax[1].set_title('TotalBsmtSF after Log transform')


# In[ ]:


def skew_features_df(df,cols):
    skew_list = []
    for col in cols:
        skew_list.append([col, df[col].skew()])
    skew_df = pd.DataFrame(skew_list, columns=['Feature', 'Skewness']).sort_values(by='Skewness', ascending=False).reset_index(drop=True)        
    return skew_df


# In[ ]:


skew_df =skew_features_df(full_data,num_cols)
skew_df.loc[np.abs(skew_df['Skewness']>0.8), :]


# In[ ]:


skew_correction_arr = list(skew_df.loc[np.abs(skew_df['Skewness'])>0.8,'Feature'])


# In[ ]:


full_data.loc[:, skew_correction_arr] = full_data.loc[:, skew_correction_arr].apply(np.log1p)


# - The skewness has decreased after log-transformation and there are only 9 features with skewness>0.8

# In[ ]:


skew_df =skew_features_df(full_data,num_cols)
skew_df.loc[np.abs(skew_df['Skewness']>0.8), :]


# - Top 10 features correlated with price after feature engineering

# In[ ]:


pd.concat([full_data.loc[:len(train)-1,:],target],axis=1).corr()['SalePrice'].sort_values(ascending=False)[1:].head(10)


# ## Encoding categorical features

# In[ ]:


ordinal_cat_cols = ['Alley', 'Street',  'LandSlope', 'LotShape',
                    'ExterQual', 'ExterCond','BsmtQual', 'BsmtCond', 'BsmtExposure',
                    'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir',
                     'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish',
                    'GarageQual','GarageCond','PavedDrive', 'PoolQC', 'Fence', 'HouseStyle', 'Electrical']


nominal_cat_cols = [ 'MSZoning', 'LandContour', 'LotConfig', 
                    'Neighborhood', 'Condition1', 'Condition2', 
                    'BldgType', 'RoofStyle','RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
                    'Foundation', 'Heating', 'MiscFeature', 'SaleType', 'SaleCondition','GarageType']


# In[ ]:


len(ordinal_cat_cols) + len(nominal_cat_cols) == len(cat_cols)


# ### Label Encoding ordinal categorical variables

# In[ ]:


enc =LabelEncoder()
full_data[ordinal_cat_cols] = full_data[ordinal_cat_cols].apply(enc.fit_transform)
#for ordinal_cat_col in ordinal_cat_cols:
#    enc = LabelEncoder()
#    train_copy.loc[:,ordinal_cat_col] = enc.fit_transform(train[ordinal_cat_col])
#    test_copy.loc[:,ordinal_cat_col] = enc.transform(test[ordinal_cat_col])


# ### Feature scaling numeric features

# In[ ]:


train = full_data.iloc[:len(train)]  
test  = full_data.iloc[len(train):]


# In[ ]:


num_cols = full_data.select_dtypes(include=[np.number]).columns


#  - We will use **RobustScaler** since it is more *robust* to outliers.

# In[ ]:


scaler = RobustScaler()
train_copy = train.copy()
test_copy = test.copy()
train_copy.loc[:,num_cols] = scaler.fit_transform(train[num_cols])
test_copy.loc[:,num_cols] = scaler.transform(test[num_cols])
train = train_copy
test = test_copy
full_data = pd.concat([train,test], axis=0,ignore_index=True)


# ### One-Hot Encoding nominal categorical variables

# In[ ]:


full_data = pd.get_dummies(full_data,drop_first=False)
print('The shape of the data matrix after pre-processing: {}' .format(full_data.shape))


# ### Dropping irrelevant features

# In[ ]:


full_data.drop(skew_df.loc[np.abs(skew_df['Skewness']>1), 'Feature'].values, axis=1,inplace=True)


# In[ ]:


#Removing highly correlated predictors
full_data = full_data.drop(['GarageArea', '1stFlrSF'], axis=1)


# In[ ]:


train= full_data.iloc[:len(train)]
test = full_data.iloc[len(train):]


# In[ ]:


print(train.shape) # 'Id' column dropped
print(test.shape)


# ## Regression models

# In[ ]:


from sklearn.model_selection import train_test_split, KFold


# In[ ]:


# Split the dataset into training and dev set
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.3, random_state=123)


# ## Linear models

# In[ ]:


from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error as mse


# ### Lasso

# In[ ]:


kf = KFold(n_splits=5, shuffle=True, random_state=123)


# In[ ]:


alpha_list_lasso = [1e-5,5e-5,8e-5,1e-4, 5e-4,1e-3, 1e-2,0.1]

val_sc_mean = []
for alpha in alpha_list_lasso:
    model = Lasso(alpha = alpha, random_state=10,max_iter=1e5)
    val_sc = cross_val_score(model,X_train,y_train,cv=kf,scoring='neg_mean_squared_error')
    val_sc_mean.append(np.mean(np.sqrt(np.abs(val_sc))))
    
index = val_sc_mean.index(np.min(val_sc_mean))
print('\nThe best mean CV score (rmse)=  %.7f which corresponds to alpha = %.4f' 
      %(np.min(val_sc_mean), alpha_list_lasso[index])) 
pd.DataFrame(list(zip(alpha_list_lasso,val_sc_mean)),columns=['alpha','Mean CV score (RMSE)'])


# In[ ]:


training_score = []
test_score = []
feature_selection = []
for alpha in alpha_list_lasso:
    lasso = Lasso(alpha = alpha,max_iter=2e4)
    lasso.fit(X_train, y_train)
    training_score.append(lasso.score(X_train, y_train))  # This score is R_squared
    test_score.append(lasso.score(X_test,y_test))
    feature_selection.append(np.sum(lasso.coef_!=0))


# ![](http://)

# - With increase in the regularization parameter alpha, more coefficients are reduced to zero by the Lasso model (Why exactly to zero? Refer Tibshirani's book or CS109 lecture on regression by Joe Blitzstein). For the alpha with the best CV score, only about 70 features were used.

# In[ ]:


plt.figure(figsize=(10,6))
plt.semilogx(alpha_list_lasso, feature_selection, 'ko-',lw=2)
plt.xlabel(r'$\alpha_{Lasso}$')
plt.ylabel('# of features selected')


# In[ ]:


plt.figure(figsize=(10,6))
plt.semilogx(alpha_list_lasso,training_score, marker='o',color='blue', lw=2, label='train set')
plt.semilogx(alpha_list_lasso,test_score, 'ko-', lw=2, label='test set')
plt.xlabel('$\\alpha$')
plt.ylabel('$R^{2}$')
plt.legend()
plt.title('Lasso regression')


# In[ ]:


lasso_final = Lasso(alpha = 0.0005, random_state=10,max_iter=1e6)
lasso_final.fit(X_train,y_train)


# In[ ]:


coef = pd.Series(lasso_final.coef_, index = X_train.columns)


# In[ ]:


## Top 20 important features as determined by Lasso
imp_coef = pd.concat([coef.sort_values(ascending=False).head(10),
                     coef.sort_values(ascending=False).tail(10)])


# In[ ]:


plt.figure(figsize=(10,10))
sns.barplot(imp_coef.values, imp_coef.index,  orient='h',color='b')
plt.title("Coefficients in the Lasso Model")


# ### Ridge regression

# In[ ]:


alpha_list_ridge = [0.01,0.1,0.6, 1,1.5, 2, 3,  5, 7, 8, 10,15,20]

validation_score_mean = []
for alpha in alpha_list_ridge:
    validation_score = cross_val_score(Ridge(alpha = alpha, solver='svd',
                                             random_state=123,max_iter=5000),X_train,y_train,cv=5,scoring='neg_mean_squared_error')
    validation_score_mean.append(np.mean(np.sqrt(np.abs(validation_score))))
#    print('alpha = ', alpha, 'mean CV score = ', validation_score_mean[-1])

index = np.where(validation_score_mean == np.min(validation_score_mean))[0][0]
print('\nThe best mean CV score (rmse)=  %.7f which corresponds to alpha = %.1f' 
      %(np.min(validation_score_mean), alpha_list_ridge[index])) 

pd.DataFrame(list(zip(alpha_list_ridge,validation_score_mean)),columns=['alpha','Mean CV score (RMSE)'])    


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(alpha_list_ridge,validation_score_mean, marker='o',color='blue', lw=2)
plt.xlabel('$\\alpha$')
plt.ylabel('Mean RMSE of 5-fold cross validation')
plt.title('Ridge regression')


# In[ ]:


validation_score_r2_mean = []
for alpha in alpha_list_ridge:
    validation_score_r2 = cross_val_score(Ridge(alpha = alpha, solver='svd',
                                             random_state=123,max_iter=5000),X_train,y_train,cv=5,scoring='r2')
    train_score_r2 = Ridge(alpha = alpha, solver='svd',
                                             random_state=123,max_iter=5000)
    validation_score_r2_mean.append(np.mean(np.sqrt(np.abs(validation_score_r2))))

plt.figure(figsize=(10,6))
plt.plot(alpha_list_ridge,validation_score_r2_mean, marker='o',color='blue', lw=2, label='Validation set')
plt.xlabel('$\\alpha$')
plt.ylabel('$R^{2}$')
plt.legend()
plt.title('Ridge regression')


# In[ ]:


## test exercise to verify the realtion between R^2 and mse.

#ridge_ex = Ridge(alpha = 1)
#ridge_ex.fit(X_train, y_train)
#TSS = np.sum(((y_train - y_train.mean())**2).values)
#SSE = np.sum((ridge_ex.predict(X_train) - y_train)**2)
#R_squared = ridge_ex.score(X_train, y_train)
#R_squared -1 +SSE/TSS


# In[ ]:


ridge_final = Ridge(alpha = 5, random_state=123,max_iter=5000)
ridge_final.fit(X_train,y_train)


# ### Elastic net

# In[ ]:


#If a is L1 coefficient and b is L2 coefficient, then alpha = a + b and l1_ratio = a / (a + b)

alpha_list_enet = [ 1e-5,5e-5,5e-4, 0.001, 0.005, 0.01, 0.05, 0.1]

validation_score_mean = []
for alpha in alpha_list_enet:
    validation_score = cross_val_score(ElasticNet(alpha = alpha, l1_ratio = 0.3,
                                             random_state=123,max_iter=1e6),X_train,y_train,cv=5,scoring='neg_mean_squared_error')
    validation_score_mean.append(np.mean(np.sqrt(np.abs(validation_score))))

index = np.where(validation_score_mean == np.min(validation_score_mean))[0][0]
print('\nThe best mean CV score (rmse)=  %.7f which corresponds to alpha = %.4f' 
      %(np.min(validation_score_mean), alpha_list_enet[index])) 
pd.DataFrame(list(zip(alpha_list_enet,validation_score_mean)),columns=['alpha','Mean CV score (RMSE)'])        


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(alpha_list_enet,validation_score_mean, marker='o',color='blue', lw=2)
plt.xlabel('$\\alpha$')
plt.ylabel('Mean RMSE of 5-fold cross validation')
plt.title('Elastic net regression')


# In[ ]:


def grid_search(clf, parameters, X, y, n_jobs= -1, n_folds=5, score_func=None):
    if score_func:
        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=score_func,verbose =2)
    else:
        print('Doing grid search')
        gs = GridSearchCV(clf, param_grid=parameters, n_jobs=n_jobs, cv=n_folds, verbose =2)
    gs.fit(X, y)
    print("mean test score (weighted by split size) of CV rounds [RMSE]: ",np.sqrt(abs(gs.cv_results_['mean_test_score'])))
    print ("\nBest parameter set", gs.best_params_, "Corresponding mean CV score (rmse)",np.sqrt(abs(gs.best_score_)))
    best = gs.best_estimator_
    return best


# In[ ]:


enet_gs = ElasticNet(random_state=13,max_iter=50000)
param = {
'alpha' : [1e-4,5e-4, 0.001, 0.005, 0.01, 0.05, 0.1],
'l1_ratio': [0.1,0.3,0.5,0.7]
}
enet_gs  = grid_search(enet_gs, param,X_train,y_train, n_folds=5, score_func= 'neg_mean_squared_error')


# In[ ]:


enet_final = ElasticNet(alpha=.001,l1_ratio=0.3, max_iter=50000)
enet_final.fit(X_train, y_train)


# ### XGBoost

# In[ ]:


def rmsle(log_pred, log_actual):
#    log_pred = np.array([np.log(val + 1) for val in pred])
#    log_actual = np.array([np.log(val + 1) for val in actual])
    logsqerror = (log_pred - log_actual) ** 2
    return np.sqrt(np.mean(logsqerror))


# In[ ]:


from xgboost.sklearn import XGBRegressor


# In[ ]:


param = {}
param['learning_rate'] = 0.05
param['verbosity'] = 1
param['colsample_bylevel'] = 0.7
param['colsample_bytree'] = 0.8
param['subsample'] = 0.6
param['reg_lambda']= 1.5
param['max_depth'] = 3
param['n_estimators'] = 800
param['seed']= 10
param['min_child_weight'] = 4
#param['gamma'] = 0.05
xgb= XGBRegressor(**param)
xgb.fit(X_train, y_train, eval_metric=['rmse'], eval_set=[(X_train, y_train),(X_test, y_test)],early_stopping_rounds=40)


# In[ ]:


xgb_imp_df = pd.DataFrame({'feature_imp':xgb.feature_importances_, 'Feature_name': X_train.columns}).sort_values(by='feature_imp', ascending=False).iloc[:20]
plt.figure(figsize=(15,6))
plt.barh(xgb_imp_df['Feature_name'][::-1],xgb_imp_df['feature_imp'][::-1],align='center')
plt.xlabel('Relative importance in XGBoost')
plt.ylabel('Features')
plt.title('Top 20 important features determied by XGBoost')
plt.show()


# In[ ]:


xgbgrid_search = XGBRegressor()
param = {
'learning_rate': [0.05,0.1],#[0.2],
#'verbosity': [1],
'colsample_bylevel': [0.7,0.8],
'colsample_bytree': [0.7,0.8],
'subsample' : [0.6,0.7], #0.8
'n_estimators': [500],
'reg_lambda': [2.5], #1.5,2,
'max_depth': [2,3],#4
 'min_child_weight': [1,2,4],   
 'seed': [10]   
}
xgbCV  = grid_search(xgbgrid_search, param,X_train,y_train, n_folds=5, score_func= 'neg_mean_squared_error')


# In[ ]:


xgbCV.fit(X_train, y_train, eval_metric=['rmse'], eval_set=[(X_train, y_train),(X_test, y_test)],early_stopping_rounds=40)


# ### Random Forest regressor

# In[ ]:


param ={
'n_estimators': [200,500],#300,700
 'max_features': [50,70,80], #30,40,100
    'max_depth': [10,None], #5,8
    'min_samples_split': [3,5],#5
    'min_samples_leaf' : [1]#2
}
rfmodelgrid_search = RandomForestRegressor(n_jobs=-1, random_state=123)
rfmodelCV  = grid_search(rfmodelgrid_search, param,X_train,y_train, n_folds=5,score_func='neg_mean_squared_error')

rfmodelCV.fit(X_train,y_train)


# In[ ]:


rf_imp_df = pd.DataFrame({'feature_imp':rfmodelCV.feature_importances_, 'Feature_name': X_train.columns}).sort_values(by='feature_imp', ascending=False).iloc[:20]
plt.figure(figsize=(15,10))
plt.barh(rf_imp_df['Feature_name'][::-1],rf_imp_df['feature_imp'][::-1],align='center')
plt.xlabel('Relative importance in Random Forest')
plt.ylabel('Features')
plt.title('Top 20 important features determied by Random Forest')
plt.show()


# - Random Forest regressor shows quite bad CV scores compared to the linear models. Can you suggest why this is the case? Maybe because the strongest features show strong linear relationship with the housing price?
# I will discard Random forest model while stacking.

# ## Stacked models

# In[ ]:


from mlxtend.regressor import StackingCVRegressor, StackingRegressor


# In[ ]:


stack = StackingCVRegressor(regressors=(xgbCV,enet_final, ridge_final, lasso_final),
                            meta_regressor=lasso_final, use_features_in_secondary=False,
                            cv=kf,pre_dispatch=2)
print('5-fold cross validation scores:\n')

for clf, label in zip([xgbCV,enet_final, ridge_final, lasso_final, stack], ['XGBoost', 'Elastic Net', 'Ridge', 'Lasso', 
                                                'StackingCVRegressor (Lasso)']):
    scores = cross_val_score(clf, X_train.values, y_train.values, cv=kf,scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(np.abs(scores))
    print("RMSE: %0.4f (+/- %0.4f) [%s]" % (rmse_scores.mean(), rmse_scores.std(), label))


# In[ ]:


stack.fit(X_train,y_train)


# ## Final predictions

# In[ ]:


print(rmsle(xgbCV.predict(X_test), y_test))
print(rmsle(rfmodelCV.predict(X_test), y_test))
print(rmsle(lasso_final.predict(X_test.values), y_test))
print(rmsle(ridge_final.predict(X_test.values), y_test))
print(rmsle(enet_final.predict(X_test.values), y_test))
print(rmsle(stack.predict(X_test), y_test))


# - Ideally the test set should not be touched while tuning hyperparameters because that might introduce additional variance which leads to overfitting. Hyperparameter tuning should be based on validation set only. The final scores on my test set show that stacked regressor outperforms all the individual regressor models.

# In[ ]:


xgb_pred = np.expm1(xgbCV.predict(test))
rf_pred = np.expm1(rfmodelCV.predict(test))
lasso_pred = np.expm1(lasso_final.predict(test))
ridge_pred = np.expm1(ridge_final.predict(test))
enet_pred = np.expm1(enet_final.predict(test))
stack_pred = np.expm1(stack.predict(test))


# I hve tried out a few ensemble combinations below by hand. But this is not optimized at all. You are welcome to do so.

# In[ ]:


#final_pred = np.mean([xgb_pred, stack_pred, lasso_pred, ridge_pred],axis=0,dtype=np.float64)
ensemble1_pred = (0.6*stack_pred + 0.4*lasso_pred)
ensemble2_pred =(0.2*stack_pred + 0.8*lasso_pred)
ensemble3_pred =(0.3*stack_pred + 0.3*lasso_pred + 0.4*xgb_pred)
ensemble4_pred =(0.5*stack_pred + 0.5*lasso_pred)
final_pred = (stack_pred + lasso_pred + xgb_pred)/3


# In[ ]:


submission = pd.concat([test_id,pd.Series(final_pred, name='SalePrice')],axis=1)
submission.to_csv("final_pred.csv", index = False)


# In[ ]:


submission.head()


# In[ ]:




