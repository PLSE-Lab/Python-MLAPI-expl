#!/usr/bin/env python
# coding: utf-8

# # Contents
# 1. [Identify the Problem](#IP)
# 2. [Import the Data](#ID)
# 3. [Exploratory Data Analysis](#EDA)
# 4. [Data Cleaning](#DC)
#      1. [Outlier Detection](#OD)
#      2. [Missing Value Imputation](#MVI)
# 5. [Feature Transformation](#FT)
#      1. [Convert Categorical to Numerical Data](#C2N)
#      2. [Reduce Skewness](#RS)
#      3. [Normalise Data Columns](#NDC)
# 6. [Modelling](#Mod)
#      1. [Model Learning](#ML)
#      2. [Model Validation](#MV)
# 

# # 1. Identify the Problem <a id='IP'></a>

# Since, the problem statement says to predict the House Prices of given multiple variables.
# So. it is a Multivariable Regression Problem.

# # 2. Import the Data <a id='ID'></a>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from scipy import stats
from scipy.stats import norm

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


# Print some useful info for the train set
print(f'Train size is: {train.shape}')
print(f'Test size is: {test.shape}')


# # 3. Exploratory Data Analysis <a id='EDA'></a>

# First, let's Analyze the dependent Variable

# In[ ]:


train['SalePrice'].describe()


# *Skewness*

# In[ ]:


def distribution_plot_and_qqplot(data):
    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data)
    print('mu = {:.2f} and sigma = {:.2f}'.format(mu, sigma))

    # Plot the distribution
    g = sns.distplot(data, fit=norm)
    legend1 = plt.legend(['Skewness : {:.2f}'.format(data.skew())], loc=4)
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.gca().add_artist(legend1)
    plt.ylabel('Frequency')
    plt.title(f'{data.name} distribution')

    # Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(data, plot=plt)
    plt.show()
    
distribution_plot_and_qqplot(train['SalePrice'])


# We can observe that the target variable:
# * deviate from the normal distribution
# * have appreciable positive skewness
# * show peakedness
#    
#    
# As, the data in linear models should be normally distributed, later we will transform this variable and make it more normally distributed.

# *Correlation*

# To explore the data, we will start with 
# * Correlation matrix ie plotting heatmap.
# * Scatter plots between the most correlated variables.

# In[ ]:


f,ax = plt.subplots(figsize = (15,15))
sns.heatmap(train.corr(), annot = True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()

Here, we can see few variables are highly correlated. Lets plot the 'SalePrice' correlation matrix (highly correlated variables).
# In[ ]:


corre = train.corr()
top_corr_features = corre.index[abs(corre['SalePrice'])>0.5]
g = sns.heatmap(train[top_corr_features].corr(),annot=True)


# These are the variables most correlated with 'SalePrice'. We can say:
# * 'OverallQual', 'GrLivArea' and 'TotalBsmtSF' are strongly correlated with 'SalePrice'.
# * 'GarageCars' and 'GarageArea' are also some of the most strongly correlated variables. However, we can say, the number of       cars that fit into the garage is a consequence of the garage area. 'GarageCars' and 'GarageArea' are like twin brothers.       You'll never be able to distinguish them. Therefore, we just need one of these variables in our analysis (we can keep          'GarageCars' since its correlation with 'SalePrice' is higher).
# * 'TotalBsmtSF' and '1stFlrSF' also seem to be twin brothers. We can keep 'TotalBsmtSF' just to say that our first guess was       right.
# * 'TotRmsAbvGrd' and 'GrLivArea', twin brothers again.
# * 'YearBuilt'... It seems that 'YearBuilt' is slightly correlated with 'SalePrice'.

# Scatter plots between 'SalePrice' and highly correlated variables

# In[ ]:


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show()


# Although we already know some of the main figures, this mega scatter plot gives us a reasonable idea about variables relationships.
# 
# One of the figures we may find interesting is the one between 'TotalBsmtSF' and 'GrLiveArea'. In this figure we can see the dots drawing a linear line, which almost acts like a border. It totally makes sense that the majority of the dots stay below that line. Basement areas can be equal to the above ground living area, but it is not expected a basement area bigger than the above ground living area.
# 
# The plot concerning 'SalePrice' and 'YearBuilt' can also make us think. In the bottom of the 'dots cloud', we see what almost appears to be an exponential function. We can also see this same tendency in the upper limit of the 'dots cloud'. Also, notice how the set of dots regarding the last years tend to stay above this limit (I just wanted to say that prices are increasing faster now).

# # 4. Data Cleaning <a id='DC'></a>

# *Outlier Detection*<a id='OD'></a>

# In[ ]:


# Before plotting let's create a useful function to use it again later
def plot_scatter(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x=x, y=y)
    plt.xlabel(x.name, fontsize=12)
    plt.ylabel(y.name, fontsize=12)
    plt.show()
    


# In[ ]:


plot_scatter(train['GrLivArea'], train['SalePrice'])
plot_scatter(train['TotalBsmtSF'], train['SalePrice'])
plot_scatter(train['1stFlrSF'], train['SalePrice'])
plot_scatter(train['OverallQual'], train['SalePrice'])
plot_scatter(train['GarageCars'], train['SalePrice'])


# We can see that there are two point with (very) large value of GrLivArea and with (very) low price. Similarily, one point each  with (very) large value of TotalBsmtSF & 1stFlrSF and with (very) low price.
# These are outliers and we can safely remove them.

# In[ ]:


train[train.GrLivArea>4500]
train[train.TotalBsmtSF>4000]
train[train['1stFlrSF']>4000]


# Deleting Outliers

# In[ ]:


train = train.drop(train[train['Id']==524].index)
train = train.drop(train[train['Id']==1299].index)
train.shape


# *Missing Value Imputation*<a id='MVI'></a>

# Let's combine both train and test data to save our time and energy. :)

# In[ ]:


df = pd.concat([train,test])

pd.set_option('display.max_rows',5000)
pd.set_option('display.max_columns',500)


# Removing the highly correlated variables

# In[ ]:


df = df.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd'], axis =1)


# In[ ]:


#checkig the columns for categorical and numerical values
print(df.select_dtypes(include = ['int64','float64']).columns)
print(df.select_dtypes(include = ['object']).columns)


# In[ ]:


df = df.set_index('Id')


# In[ ]:


#missing data
total_miss = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.shape[0]*100).sort_values(ascending=False)
missing_data = pd.concat([total_miss,percent], axis=1, keys=['Total','Percent'])

missing_data.head(35)


# In[ ]:


columns_drop =percent[percent > 20].keys()

columns_drop


# In[ ]:


df = df.drop(columns_drop, axis = 1)

print(df.shape)
df.describe(include = 'all')


# Now, lets impute the missing value.

# In[ ]:


missing_cols = df.columns[df.isnull().any()]

missing_cols


# First impute the missing values in Bsmt Features.

# In[ ]:


bsmt_cols = ['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1',
       'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF','TotalBsmtSF']

bsmt_feat = df[bsmt_cols]
bsmt_feat.info()


# Lets first impute the missing values in rows.

# In[ ]:


bsmt_feat = bsmt_feat[bsmt_feat.isnull().any(axis=1)]

#print(bsmt_feat)
print(bsmt_feat.shape)


# replace the NaN in categorical with NA(ie No Basement) and with 0 in Numerical data.

# In[ ]:


bsmt_feat_all_nan = bsmt_feat[(bsmt_feat.isnull() | bsmt_feat.isin([0])).all(1)]

#print(bsmt_feat_all_nan)
print(bsmt_feat_all_nan.shape)


# In[ ]:


qual = list(df.loc[:,df.dtypes=='object'].columns.values)

for i in bsmt_cols:
    if i in qual:
        bsmt_feat_all_nan[i] = bsmt_feat_all_nan[i].replace(np.nan,'NA')
    else:
        bsmt_feat_all_nan[i] = bsmt_feat_all_nan[i].replace(np.nan,0)

bsmt_feat.update(bsmt_feat_all_nan)
df.update(bsmt_feat_all_nan)


# In[ ]:


#Finding remaining rows which have null columns

bsmt_feat = bsmt_feat[bsmt_feat.isin([np.nan]).any(axis=1)]

#print(bsmt_feat)
print(bsmt_feat.shape)


# Replace the BsmtFinType2 based on BsmtFinSF2 by bucketing the BsmtFinSF2.
# 

# In[ ]:


#Bucket the continuous columns
print(df['BsmtFinSF2'].max())
print(df['BsmtFinSF2'].min())

#Bucket this  range in 5 buckets.
#pd.cut(range(0,1526),5)


# In[ ]:


df_slice = df[(df['BsmtFinSF2'] >= 305) & (df['BsmtFinSF2'] <= 610)]

#Impute this particular row
bsmt_feat.at[333,'BsmtFinType2'] = df_slice['BsmtFinType2'].mode()[0]


# In[ ]:


#Impute the missing BsmtExposure value with the slice of BsmtExposure when BsmtQual is Gd.
bsmt_feat['BsmtExposure'] = bsmt_feat['BsmtExposure'].replace(np.nan, df[df['BsmtQual'] == 'Gd']['BsmtExposure'].mode()[0])

#Similarily
bsmt_feat['BsmtCond'] = bsmt_feat['BsmtCond'].replace(np.nan, df['BsmtCond'].mode()[0])
bsmt_feat['BsmtQual'] = bsmt_feat['BsmtQual'].replace(np.nan, df['BsmtQual'].mode()[0])


# In[ ]:


df.update(bsmt_feat)

df.columns[df.isnull().any()]


# In[ ]:


#Now impute the missing values in Garage Features.

garage_cols = ['GarageCars', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType',
       'GarageYrBlt']

gar_feat = df[garage_cols]
gar_feat.info()


# In[ ]:


gar_feat = gar_feat[gar_feat.isnull().any(axis=1)]

#print(gar_feat)
print(gar_feat.shape)


# In[ ]:


gar_feat_all_nan = gar_feat[(gar_feat.isnull() | gar_feat.isin([0])).all(1)]

#print(gar_feat_all_nan)
print(gar_feat_all_nan.shape)


# In[ ]:


for i in garage_cols:
    if i in qual:
        gar_feat_all_nan[i] = gar_feat_all_nan[i].replace(np.nan,'NA')
    else:
        gar_feat_all_nan[i] = gar_feat_all_nan[i].replace(np.nan,0)
gar_feat.update(gar_feat_all_nan)
df.update(gar_feat_all_nan)


# In[ ]:


gar_feat = gar_feat[gar_feat.isnull().any(axis=1)]

#gar_feat


# In[ ]:


for i in garage_cols:
    gar_feat[i] = gar_feat[i].replace(np.nan, df[df['GarageType'] == 'Detchd'][i].mode()[0])

#gar_feat


# In[ ]:


df.update(gar_feat)

df.columns[df.isnull().any()]


# In[ ]:


df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])

df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])

df['Functional'] = df['Functional'].fillna(df['Functional'].mode()[0])

df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])

df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])

df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])

df['Utilities'] = df['Utilities'].fillna(df['Utilities'].mode()[0])

df['MasVnrType'] = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])


# In[ ]:


df[df['MasVnrArea'].isnull() == True]['MasVnrType'].unique()


# In[ ]:


df.loc[(df['MasVnrType'] == 'None') & (df['MasVnrArea'].isnull() == True), 'MasVnrArea'] = 0


# In[ ]:


#print(df['MasVnrArea'].isnull().sum())
#print(df['MasVnrType'].isnull().sum())
#print(df.columns[df.isnull().any()])

Now impute the LotFrontage based on LotConfig
# In[ ]:


lotconfig = ['Corner','Inside','CulDSac','FR2','FR3']

for i in lotconfig:
    df['LotFrontage'] = pd.np.where((df['LotFrontage'].isnull() == True) & (df['LotConfig'] == i), df[df['LotConfig'] == i]['LotFrontage'].mean(),df['LotFrontage'])

df.isnull().sum().max()


# # 5. Feature Transformation <a id='FT'></a>

# *Dealing with the Categorical Data*<a id='C2N'></a>

# In[ ]:


#Few Features are in numerical in nature but actually are of Categorical

cat_con_columns = ['MSSubClass', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt','YrSold']
for i in cat_con_columns:
    df[i] = df[i].astype(str)


# In[ ]:


import calendar
df['MoSold'] = df['MoSold'].apply(lambda x : calendar.month_abbr[x])

df['MoSold'].unique()


# In[ ]:


quan = list(df.loc[:,df.dtypes != 'object'].columns.values)


# In[ ]:


# Ordered Data
from pandas.api.types import CategoricalDtype

df['BsmtCond'] = df['BsmtCond'].astype(CategoricalDtype(categories=['NA','Po','Fa','TA','Gd','Ex'], ordered = True)).cat.codes

df['BsmtExposure'] = df['BsmtExposure'].astype(CategoricalDtype(categories=['NA','No','Mn','Av','Gd'], ordered = True)).cat.codes

df['BsmtFinType1'] = df['BsmtFinType1'].astype(CategoricalDtype(categories=['NA','Unf','LwQ','Rec','BLQ','ALQ','GLQ'], ordered = True)).cat.codes

df['BsmtFinType2'] = df['BsmtFinType2'].astype(CategoricalDtype(categories=['NA','Unf','LwQ','Rec','BLQ','ALQ','GLQ'], ordered = True)).cat.codes

df['BsmtQual'] = df['BsmtQual'].astype(CategoricalDtype(categories=['NA','Po','Fa','TA','Gd','Ex'], ordered = True)).cat.codes

df['ExterQual'] = df['BsmtQual'].astype(CategoricalDtype(categories=['Po','Fa','TA','Gd','Ex'], ordered = True)).cat.codes

df['ExterCond'] = df['ExterCond'].astype(CategoricalDtype(categories=['Po','Fa','TA','Gd','Ex'], ordered = True)).cat.codes

df['Functional'] = df['Functional'].astype(CategoricalDtype(categories=['Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'], ordered = True)).cat.codes

df['GarageCond'] = df['GarageCond'].astype(CategoricalDtype(categories=['NA','Po','Fa','TA','Gd','Ex'], ordered = True)).cat.codes

df['GarageQual'] = df['GarageQual'].astype(CategoricalDtype(categories=['NA','Po','Fa','TA','Gd','Ex'], ordered = True)).cat.codes

df['GarageFinish'] = df['GarageFinish'].astype(CategoricalDtype(categories=['NA','Unf','RFn','Fin'], ordered = True)).cat.codes

df['HeatingQC'] = df['HeatingQC'].astype(CategoricalDtype(categories=['Po','Fa','TA','Gd','Ex'], ordered = True)).cat.codes

df['KitchenQual'] = df['KitchenQual'].astype(CategoricalDtype(categories=['Po','Fa','TA','Gd','Ex'], ordered = True)).cat.codes

df['PavedDrive'] = df['PavedDrive'].astype(CategoricalDtype(categories=['N','P','Y'], ordered = True)).cat.codes

df['Utilities'] = df['Utilities'].astype(CategoricalDtype(categories=['ELO','NoSeWa','NoSewr','AllPub'], ordered = True)).cat.codes


# *Reducing Skewness among numerical data* <a id='RS'></a>

# In[ ]:


skewed_features = ['2ndFlrSF','3SsnPorch',
 'BedroomAbvGr','BsmtFinSF1','BsmtFinSF2',
 'BsmtFullBath','BsmtHalfBath','BsmtUnfSF',
 'EnclosedPorch','Fireplaces','FullBath',
 'GarageCars','GrLivArea', 'HalfBath',
 'KitchenAbvGr','LotArea','LotFrontage',
 'LowQualFinSF','MasVnrArea','MiscVal',
 'OpenPorchSF','PoolArea','ScreenPorch',
 'TotalBsmtSF','WoodDeckSF']


# In[ ]:


## Remove Skewness from the data
for i in skewed_features:
    df[i] = np.log1p(df[i])

log_SalePrice = np.log1p(train['SalePrice'])

distribution_plot_and_qqplot(log_SalePrice)


# *Normalisation* <a id='NDC'></a>

# In[ ]:


# Create Dummies for all non ordinal categorical data
qual1 = list(df.loc[:,df.dtypes == 'object'].columns.values)
print(len(qual1))

df_with_dummies = pd.get_dummies(df, columns=qual1, drop_first=True)
df_with_dummies.shape


# In[ ]:


##Normalize

df_inputs = df_with_dummies.copy()
targets = log_SalePrice.copy()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_inputs)
df_inputs_scaled = scaler.transform(df_inputs)


# # 6. Modelling <a id='Mod'></a>

# In[ ]:


#Segregate data into original train and test
train_len = len(train)
train_scaled = df_inputs_scaled[:train_len]
test_scaled = df_inputs_scaled[train_len:]

print(train_scaled.shape)

print(test_scaled.shape)


# In[ ]:


# Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_scaled, targets, test_size=0.2, random_state=365)


# Model Learning <a id='ML'></a>

# In[ ]:


import xgboost
regressor = xgboost.XGBRegressor(learning_rate = 0.06, max_depth= 3, n_estimators = 350, random_state= 0)
regressor.fit(x_train,y_train)


# In[ ]:


y_hat = regressor.predict(x_train)

plt.scatter(y_train, y_hat, alpha = 0.2)
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
plt.show()


# In[ ]:


regressor.score(x_train,y_train)


# In[ ]:


##Testing
y_hat_test = regressor.predict(x_test)


plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.show()


# In[ ]:


y_predict = regressor.predict(test_scaled)
y_predict = np.expm1(y_predict)


# Model Validation <a id='MV'></a>

# In[ ]:


## k-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = x_train, y = y_train, cv = 10)


# In[ ]:


print(accuracies.mean())
print(accuracies.std())

