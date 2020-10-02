#!/usr/bin/env python
# coding: utf-8

# <font color='Blue'>**House Prices: Advanced Regression Techniques**</font><br/>

# <font color='Blue'>**Competition description**</font><br/>

# <font color='Black'>**Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.**</font><br/>

# <font color='Blue'>**Libraries**</font><br/>

# <font color='Black'>**"When in doubt, go to the library."
#  J.K. Rowling**</font><br/>

# In[ ]:


#DO NOT DISTURB!
import warnings
warnings.filterwarnings("ignore")

# the King and the Queen of libraries
import pandas as pd
import numpy as np

#friendly stats
from scipy import stats
from scipy.stats import norm, skew

#plots
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.style.use('ggplot')
import seaborn as sns

#Modeling
from sklearn import ensemble, metrics
from sklearn import linear_model, preprocessing
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb


# <font color='Blue'>**The Datasets**</font><br/>

# <font color='Black'>**Let's load the train and test dataframes using Pandas the king of the libraries, let's then have a look to their shape and to their the firsts rows**</font><br/>

# In[ ]:


df_train=pd.read_csv('../input/train.csv', index_col='Id')
df_test=pd.read_csv('../input/test.csv', index_col='Id')
#Save the 'Id' column
train_ID = df_train.index
test_ID = df_test.index


# In[ ]:


print('the Train dataframe shape is:',df_train.shape,' while the Test dataframe shape is:',df_test.shape)


# In[ ]:


df_train.head(5)


# In[ ]:


df_test.head(5)


# <font color='Blue'>**Processing Data**</font><br/>

# <font color='red'>**Outliers**</font><br/>

# <font color='Black'>**Reading the Description file and some more information about the data, provided by the author http://jse.amstat.org/v19n3/decock.pdf I found out that there are outliers present in the training data. We can explore these outliers using a scatter plot!**</font><br/>

# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'], color='blue')
plt.ylabel('Sale_Price', fontsize=13)
plt.xlabel('Ground_Living_Area', fontsize=13)
plt.show()


# <font color='Black'>**It's easy to spot at the bottom right two values with extremely large Ground_Living_Area that are offered at a low price. These values are oultliers, therefore it is good to delete them.**</font><br/>

# In[ ]:


df_train = df_train.drop((df_train[df_train['GrLivArea']>4000]).index)


# In[ ]:


#looking at the df_train without the two outliers
fig, ax = plt.subplots()
ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'], color='blue')
plt.ylabel('Sale_Price', fontsize=13)
plt.xlabel('Ground_Living_Area', fontsize=13)
plt.show()


# <font color='Red'>**The target: Sale Price**</font><br/>

# <font color='Black'>**Let's see if Sales Price follows a normal distribution, which would be great for linear models**</font><br/>

# In[ ]:


sns.distplot(df_train['SalePrice'] , fit=norm, color='blue');
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df_train['SalePrice'])
print('mu:',mu,'sigma:',sigma)

#Let's plot the distribution of Sale Price
plt.legend(['Normal dist'],loc='best')
plt.ylabel('Frequency')
plt.title('Sale_Price distribution')


# <font color='Black'>**Sale Price variable is right skewed. But we can log-transform it thanks to Queen Numpy!**</font><br/>

# In[ ]:


#Numpy Abrakadabra!
SALEPRICE_ABRAKADABRA=df_train["SalePrice"]
SALEPRICE_ABRAKADABRA= np.log1p(SALEPRICE_ABRAKADABRA)

#And now
sns.distplot(SALEPRICE_ABRAKADABRA , fit=norm,color='blue');
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(SALEPRICE_ABRAKADABRA)
print('mu:',mu,'sigma:',sigma)

#Let's plot again the distribution of Sale Price
plt.legend(['Normal dist'],loc='best')
plt.ylabel('Frequency')
plt.title('Sale_Price distribution')


# <font color='Black'>**We got it!**</font><br/>

# <font color='Blue'>**Features engineering**</font><br/>

# <font color='Red'>**Quantitative and qualitative**</font><br/>

# <font color='Black'>**Let's find out which features are quantitative and which instead are qualitative**</font><br/>

# In[ ]:


quantitative = [i for i in df_train.columns if df_train.dtypes[i] != 'object']
quantitative.remove('SalePrice')
qualitative = [j for j in df_train.columns if df_train.dtypes[j] == 'object']
print('Quantitative:',quantitative)
print('')
print('Qualitative:',qualitative)


# <font color='Black'>**It is good to continue our Feature engineering using the train and test dataframes**</font><br/>

# In[ ]:


merged_data = df_train.append(df_test, sort=False).reset_index(drop=True)
print("The size of the merged data is:", merged_data.shape)
ntrain = df_train.shape[0]
ntest = df_test.shape[0]
y_train = df_train.SalePrice.values


# In[ ]:





# <font color='Red'>**Missing data**</font><br/>

# In[ ]:


missing = merged_data.isnull().sum()
missing = missing[missing > 0]
missing50=missing[missing>=2915/2] #feature with_more than 50 percent of all data missing
missing.sort_values(inplace=True)
del missing['SalePrice']
del missing50['SalePrice']
missing.plot.bar()


# In[ ]:


print('Features with missing values:\n',missing)
print('')
print('Total of features with missing values:\n',len(missing))
print('')
print('Total of the missing values of the features with more than 50 percent of all data missing:\n',missing50)


# <font color='Black'>**DON'T JUDGE A BOOK BY ITS COVER!!! Most of times NA means lack of subject described by attribute, like missing pool, no garage and basement etc...indeed if we have a look to the file description we find out that**</font><br/>

# <font color='Black'><br/>
# - Alley : Data description says NA means "no alley access"
# - Fence : Data description says NA means "no fence"
# - MiscFeature : Data description says NA means "no misc feature"
# - PoolQC : Data description says NA means "No Pool".
# - FireplaceQu : Data description says NA means "no fireplace"
# - LotFrontage : The area of each street connected to the house property often has a similar area to other houses in its neighborhood, so let's fill in missing values by the median LotFrontage of the neighborhood.
# - GarageType, GarageFinish, GarageQual and GarageCond : Data description says NA means None
# - GarageYrBlt, GarageArea and GarageCars : we can replace missing data with 0
# - BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : Missing values are likely zero for having no basement
# - BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related features, NaN means that there is no basement.
# - MasVnrArea and MasVnrType : NA means no masonry veneer for these houses.
# - MSZoning: We could fill in missing values with the mode
# - Utilities : Since the house with 'NoSewa' is in the training set, other values are set on 'AllPub' this feature won't help in predictive modelling. We can remove it!
# - Functional : data description says NA means typical.
# - Electrical : We could fill in missing values with the mode
# - KitchenQual: We could fill in missing values with the mode
# - Exterior1st and Exterior2nd: We could fill in missing values with the mode
# - SaleType : We could fill in missing values with the mode
# Finally
# - MSSubClass :Here NA means No building class, so we can replace missing values with None</font><br/>

# <font color='Black'>**GOODBYE MISSING VALUES!**</font><br/>

# In[ ]:


merged_data["Alley"] = merged_data["Alley"].fillna("None")

merged_data["Fence"] = merged_data["Fence"].fillna("None")

merged_data["MiscFeature"] = merged_data["MiscFeature"].fillna("None")

merged_data["PoolQC"] = merged_data["PoolQC"].fillna("None")

merged_data["FireplaceQu"] = merged_data["FireplaceQu"].fillna("None")

merged_data["LotFrontage"] = merged_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    merged_data[col] = merged_data[col].fillna('None')
    
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    merged_data[col] = merged_data[col].fillna(0)
    
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    merged_data[col] = merged_data[col].fillna(0)
    
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    merged_data[col] = merged_data[col].fillna('None')
    
merged_data["MasVnrType"] = merged_data["MasVnrType"].fillna("None")

merged_data["MasVnrArea"] = merged_data["MasVnrArea"].fillna(0)

merged_data['MSZoning'] = merged_data['MSZoning'].fillna(merged_data['MSZoning'].mode()[0])

merged_data = merged_data.drop(['Utilities'], axis=1)

merged_data["Functional"] = merged_data["Functional"].fillna("Typ")

merged_data['Electrical'] = merged_data['Electrical'].fillna(merged_data['Electrical'].mode()[0])

merged_data['KitchenQual'] = merged_data['KitchenQual'].fillna(merged_data['KitchenQual'].mode()[0])

merged_data['Exterior1st'] = merged_data['Exterior1st'].fillna(merged_data['Exterior1st'].mode()[0])

merged_data['Exterior2nd'] = merged_data['Exterior2nd'].fillna(merged_data['Exterior2nd'].mode()[0])

merged_data['SaleType'] = merged_data['SaleType'].fillna(merged_data['SaleType'].mode()[0])

merged_data['MSSubClass'] = merged_data['MSSubClass'].fillna("None")


# In[ ]:


merged_data.head(10)


# In[ ]:


#Check
merged_data_NA_VALUES =merged_data.isnull().sum()


# <font color='Black'>**Let's create a new variable: the total squarefootage!**</font><br/>

# In[ ]:


merged_data['TotalSF'] = merged_data['TotalBsmtSF'] + merged_data['1stFlrSF'] + merged_data['2ndFlrSF']


# <font color='Black'>**Some of the quantitative variables such as "MSSubClass","OverallCond","YrSold",'MoSold" are qualitative features!! We need to tranform them!!**</font><br/>

# In[ ]:


merged_data['MSSubClass'] = merged_data['MSSubClass'].apply(str)
merged_data['OverallCond'] = merged_data['OverallCond'].astype(str)
merged_data['YrSold'] = merged_data['YrSold'].astype(str)
merged_data['MoSold'] = merged_data['MoSold'].astype(str)


# <font color='Black'>**Label Encoding thanks to sklearn preprocessing :)**</font><br/>

# In[ ]:


from sklearn.preprocessing import LabelEncoder
columns = ['LandContour',  'MSZoning',  'Alley',
      'MasVnrType',  'ExterQual',  'ExterCond',
      'BsmtQual',  'BsmtCond',  'BsmtExposure',
      'BsmtFinType1',  'BsmtFinType2', 'HeatingQC',
      'CentralAir',  'KitchenQual',  'FireplaceQu',
      'GarageFinish',  'GarageQual',  'GarageCond',
      'PavedDrive',  'PoolQC',  'MiscFeature']
for col in columns:
    lbl = LabelEncoder() 
    lbl.fit(list(merged_data[col].values)) 
    merged_data[col] = lbl.transform(list(merged_data[col].values))
    
columns_Qual=['LotShape', 'LotConfig', 'LandSlope', 'Neighborhood',
       'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
       'Foundation', 'Heating', 'Electrical', 'Functional',
       'GarageType', 'GarageCond', 'Fence', 'SaleType',
       'SaleCondition', 'Street']
temp = pd.get_dummies(merged_data[columns_Qual], drop_first=True)
merged_data = merged_data.drop(columns_Qual, axis=1)
merged_data = pd.concat([merged_data, temp], axis=1)


# <font color='Blue'>**Prediction**</font><br/>

# <font color='Red'>**Model**</font><br/>

# <font color='Black'>**Let's get back our df_train and df_test with their new cool appearence**</font><br/>

# In[ ]:


df_train = merged_data[merged_data['SalePrice'].notnull()]
df_test = merged_data[merged_data['SalePrice'].isnull()].drop('SalePrice', axis=1)


# In[ ]:


x_train = df_train.drop(['SalePrice'], axis=1)
y_train = df_train['SalePrice']
x_test  = df_test


# <font color='Black'>**Scaling**</font><br/>

# In[ ]:


scaler = preprocessing.RobustScaler();
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)


# In[ ]:


df_test.shape


# <font color='Black'>**Prediction**</font><br/>

# <font color='Black'>**KERNEL RIDGE REGRESSION**</font><br/>

# In[ ]:


KRR = KernelRidge(alpha=0.05, kernel='polynomial', degree=1, coef0=2.5)


# <font color='Black'>**LASSO REGRESSION**</font><br/>

# In[ ]:


lasso = linear_model.Lasso(alpha=0.001, max_iter=5000, random_state=42)


# <font color='Black'>**Gradient Boosting REGRESSION**</font><br/>

# In[ ]:


GBoost = ensemble.GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, max_features='sqrt', loss='huber', random_state=42)


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
#It is very useful to create a class in order to make predictions with the above defined models! 

class Averaging_the_models(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, cool_models, peso):
        self.cool_models = cool_models
        self.peso = peso
        
    def fit(self, X, y):
        self.cool_models_ = [clone(x) for x in self.cool_models]
        for model in self.cool_models_:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = np.column_stack([(model.predict(X) * peso) for model, peso in zip(self.cool_models_, self.peso)])
        return np.sum(predictions, axis=1)


# In[ ]:


regression = Averaging_the_models(cool_models=(KRR, lasso, GBoost), peso=[0.25, 0.25, 0.50])


# In[ ]:


regression.fit(x_train_scaled, np.log1p(y_train))
result = np.expm1(regression.predict(x_test_scaled))


# In[ ]:


subFDS = pd.DataFrame({
    "Id": test_ID,
    "SalePrice": result
})
subFDS.to_csv("subFDS.csv", index=False)


# <font color='Black'>**Root Mean Square Error of cross validation**</font><br/>

# In[ ]:


def rmse_cv(model, x, y):
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=5))
    return rmse

score = rmse_cv(regression, x_train_scaled, np.log1p(y_train))
print(round(score.mean(), 5))


# In[ ]:


subFDS

