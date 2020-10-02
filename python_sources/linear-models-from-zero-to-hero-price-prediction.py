#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction

# <img src="https://ritu-19.github.io/images/housing_0_2.jpg" width="800" height="400">

# In this Kernel we will try to predict house **Saleprice** using different linear regression techniques. We will see how we can modify the features and fit into a linear model and get good results. Before going further we must understand the assumptions of linear regression.

# 
# **Assumptions**
# 
# 1. **Linear relationship: There should be a linear relationship between input and target variables**
# 
# 2. **Multivariate Normality: Assumes that the resiudals after prediction are normally distributed.**
# 
# 3. **No Multicollinearity: Assumes that input variables are not highly coorelated with each other**
# 
# 4. **Homoscedacity: Assumes variance of error terms are similar across input variables.**
# 
# You can read more from [here](https://www.statisticssolutions.com/assumptions-of-linear-regression/)
# 
# We must keep these points in mind before proceeding further 

# ### Importing libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler,RobustScaler
import scipy
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,KFold,cross_val_score,cross_validate
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import GridSearchCV
import warnings
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
import matplotlib.style as style
warnings.filterwarnings('ignore')


# ### Loading data

# In[ ]:


# Loading data
train_csv = '/kaggle/input/house-prices-advanced-regression-techniques/train.csv'
test_csv = '/kaggle/input/house-prices-advanced-regression-techniques/test.csv'

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)
print(f'The shape of train data is {df_train.shape}')
print(f'The shape of test data is {df_test.shape}')
df_train.head()


# # Exploratory data analysis
# 
# ## EDA on SalePrice
# 
# Let's create a histogram to see if the target variable is Normally distributed. If we want to create a linear model, it is better that the features follow a normal distribution.
# 
# * We will plot a histplot(Also we will check skweness) and Q-Q plot to check this.
# * Also we will plot a boxplot to check outliers
# 
# 

# In[ ]:


#function for ploting Histogram,Q-Q plot and 
# Box plot of target and also print skewness
def target_analysis(target):
    fig = plt.figure(constrained_layout=True, figsize=(14,10))
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title('Histogram')
    sns.distplot(target,norm_hist=True,ax=ax1)
    ax2 = fig.add_subplot(grid[1, :2])
    ax2.set_title('Q-Q Plot')
    stats.probplot(target,plot=ax2)
    ax3 = fig.add_subplot(grid[:,2])
    ax3.set_title('Box Plot')
    sns.boxplot(target,orient='v',ax=ax3)
    print(f'skweness is { target.skew()}')
    plt.show()


# In[ ]:


target_analysis(df_train['SalePrice'])


# Observations: 
# 
# * We can see that SalePrice are not normaly distributed.
# 
# * It is right skwed by 1.8828757597682129(Positive skweness). Skweness means is the degree of distortion from the symmetrical bell curve or the normal curve. Exact normal distribution will have skweness of 0.
# 
# * Here Positive skweness means more than half of the houses will be sold at less than average price.
# 
# * When performing regression, sometimes it makes sense to log-transform the target variable when it is skewed. One reason for this is to improve the linearity of the data.
# 
# * We can see some outliers(above 700000) using box plot[outliers]. For now we will keep it.
# 
# * To make distribution symmeric we can apply log to it (np.log1()).We will handle that later.
# 

# In[ ]:


target_analysis(np.log1p(df_train['SalePrice']))


# Observation: 
# 
# * We can see that now our target variable is more skwed towards zero and symmetric(follows a approx normal distribution). 

# ## EDA of continous variables
# 
# * Continous variables plays an important role in prediction of linear regression models. Let us explore more on these

# In[ ]:



num_cols_tr = df_train.select_dtypes('number').columns.tolist()
num_cols_te = df_test.select_dtypes('number').columns.tolist()

print(f'There are {len(num_cols_tr)} numeric columns in train data')
df_num = df_train[num_cols_tr]
df_test_num = df_test[num_cols_te]

print('Train/test numeric shapes')
print(df_num.shape)
print(df_test_num.shape)


# In[ ]:


df_num.dtypes


# In[ ]:


# droping column 'Id' as it is not needed
df_num = df_num.drop(columns=['Id'])
df_test_num = df_test_num.drop(columns=['Id'])


# ## Coorelation matrix and removing multicollinearity

# In[ ]:


corr = df_num.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, cmap=sns.diverging_palette(20, 220, n=200))
plt.show()


# * Heatmap is the better way to understand coorelation independent variables as well as with target variables.
# 
# * We know as per assumption of linear regression multicollinearity should not exist.That means two or more independent features cannot be strongly coorelated.Coorelation values exist between [-1,1]. 
# 
# * Here we have no  strong negative coorelation,but there are some strong positive coorelations 
# 
# * From the first sight itself we see deep blue at intersection of features '1stFlrSF'(First Floor square feet) and 'TotalBSMTSF'(Total square feet of basement area) .Another blue spot we see is between 'Garage area' and 'Garage Cars'. That means that they are highly coorelated.We can say that they almost give the same information, so multicollinearity really occurs. 
# 
# * So a better approach is to keep one of them.For eg, that we will either keep Garage area or Garage Cars, as they both give same information. We will not be abe to seperate those. So we can keep one of them 
# 
# * Similarly features like 'YearBuilt' and 'GarageYrBlt' have strong coorelation.Much more features behave like this.
# 
# * We will analyse strong multicollinear pairs.They are ('GarageArea','GarageCars'),('GarageYrBlt','YearBuilt')('TotRmsAbvGrd','GrLivArea'),('1stFlrSF','TotalBsmtSF')
# 
# * The above pairs are multicollinear. We will kee only one in each pair.We do this on an assumption that we are using Linear regression.For ridge they will automatically take are of multicollinearity.
# 
# 
# **How to decide out of two which feature we should remove?**
# 
# For that we will check coorelation with SalePrice and their linearity which is an important assumption for linear regression
# 
# 

# In[ ]:


multicoll_pairs = ['GarageArea','GarageCars',
        'GarageYrBlt','YearBuilt','TotRmsAbvGrd','GrLivArea',
                   '1stFlrSF','TotalBsmtSF']

fig,axes = plt.subplots(4,2,figsize=(15,20))

def plot_two(feat,i,j):
    sns.regplot(x=df_num[feat], y=df_num['SalePrice'], ax=axes[i,j])
    sns.scatterplot(y=df_num['SalePrice'],x=df_num[feat],color=('orange'),ax=axes[i,j])   
    fig.tight_layout(pad=5.0)
    

for i,feat in enumerate(multicoll_pairs):
    j = i%2 #0 or 1
    plot_two(feat,i//2,j)


# In[ ]:


# multicoll_pairs.append('SalePrice')
df_num.corr()['SalePrice'].sort_values(ascending=False)


# * So based on this we will drop 'GarageArea','TotRmsAbvGrd' and 'GarageYrBlt','1stFlrSF'.
#  
# * Here we can't see any negative strong coorelation
# 
# *  We can see that 'OverallQual', 'GrLivArea' and 'TotalBsmtSF' are strongly correlated with 'SalePrice'.

# In[ ]:


# remaning numeric columns(input variables)
df_num = df_num.drop(columns = ['1stFlrSF','GarageArea','TotRmsAbvGrd','GarageYrBlt'])
df_test_num = df_test_num.drop(columns = ['1stFlrSF','GarageArea','TotRmsAbvGrd','GarageYrBlt'])


# ## Checking for Linearity
# 
# Linearity of independent variables with target variable is another imporatnt assumption of linear regression. Let us plot features with SalePrice and check
# 

# In[ ]:


fig,axes = plt.subplots(16,2,figsize=(15,60))

linear_num_cols = df_num.select_dtypes(include='number').columns.tolist() 
linear_num_cols.remove('SalePrice')

def plot_two(feat,i,j):
    sns.regplot(x=df_num[feat], y=df_num['SalePrice'], ax=axes[i,j])
    sns.scatterplot(y=df_num['SalePrice'],x=df_num[feat],color=('orange'),ax=axes[i,j])   
    fig.tight_layout(pad=5.0)
    

for i,feat in enumerate(linear_num_cols):
    j = i%2 #0 or 1
    plot_two(feat,i//2,j)


# We can see that there are many non linear features.Some behave like categorical features.
# 
# Note: If there are some features which exhibit little bit of non linearity, but having strong coorelation with target, we keep it
# 
# Non linear features that can be converted to categorical: 'YrSold','MSSold','PoolArea','BsmtFullBath','BsmtHalfBath','Halfbath','BedroomAbvGvr','Fireplaces',
# 
# Non-linear features that we will drop 'OverallCond','LowQualFinSF', 'MiscVal',
# 

# In[ ]:


df_num = df_num.drop(columns=['OverallCond','LowQualFinSF', 'MiscVal'])
df_test_num = df_test_num.drop(columns=['OverallCond','LowQualFinSF', 'MiscVal'])
print('Train/test numeric shapes')
print(df_num.shape)
print(df_test_num.shape)


# In[ ]:


# removing outliers
# removing outliers
# removing some outliers
df_num = df_num[df_num['LotFrontage'] < 300]
df_num = df_num[df_num['BsmtFinSF1'] < 5000]
df_num = df_num[df_num['TotalBsmtSF'] < 6000]
df_num = df_num[df_num['GrLivArea'] < 4600]
df_num = df_num[df_num['SalePrice'] < 700000]

print(df_num.shape)


# ### Converting some numeric variables to categorical

# In[ ]:


non_linear_cat_cols = ['YrSold','MoSold','PoolArea','BsmtFullBath',
            'BsmtHalfBath','HalfBath','BedroomAbvGr','Fireplaces']

df_num = df_num.drop(columns = non_linear_cat_cols)
df_test_num = df_test_num.drop(columns = non_linear_cat_cols)

for col in non_linear_cat_cols:
    df_train[col] = df_train[col].astype(object)
    df_test[col] = df_train[col].astype(object)
    
print(df_num.shape)
print(df_test_num.shape)


# In[ ]:


df_num.head(3)


# In[ ]:


# missing values in numeric features
def missing_cols(df):
    cols = df.columns[df.isna().any()].tolist()
    print(f'Columns | Percentage missing')
    for column in cols:
        percent = round((sum(df[column].isnull())/df.shape[0])*100,2)
        print(f'{column} : {percent}%')


# Let us check missing values in numeric columns

# In[ ]:


missing_cols(df_num)


# In[ ]:


missing_cols(df_test_num)


# In[ ]:


# lets convert the target variable and keep
target = np.log1p(df_num['SalePrice'])


# ## EDA on Categorical features

# In[ ]:


# helper functions
def categories_plot(df,col,xlabel='Values',size=(8,4)):
    y_train = df[col].value_counts().values
    x_train = df[col].value_counts().index.tolist()
    plt.figure(figsize=size)
    plt.title(col)
    sns.barplot(x_train,y_train)
    plt.xlabel(xlabel)
    plt.xticks(rotation=90, ha='right')
    plt.ylabel('count')
    plt.show()


# In[ ]:


num_idx = df_num.index.to_list()
cat_cols = df_train.select_dtypes(exclude=[np.number]).columns.tolist()

df_cat = df_train.loc[num_idx][cat_cols]
df_cat_test = df_test[cat_cols]
print('Train/test categoric shapes')
print(df_cat.shape)
print(df_cat_test.shape)


# Let us check for the missing values first

# In[ ]:


missing_cols(df_cat)


# In[ ]:


missing_cols(df_cat_test)


# Observations:
# 
# * We have missing values in both train and test data
# 
# * In categorical columns we can clearly see that Alley,MiscFeature,PoolQC,Fence has more than 80% missing values and hence we drop them.
# 
# * For rest of the categorical fetures we will do imputation of max repeating value on later section.
# 

# In[ ]:


# dropping some features
df_cat = df_cat.drop(columns=['Alley','MiscFeature','PoolQC','Fence'])
df_cat_test = df_cat_test.drop(columns=['Alley','MiscFeature','PoolQC','Fence'])


# Let us go through different categorical columns

# In[ ]:


df_cat.describe()


# In[ ]:


#MSZoning Identifies the general zoning classification of the sale. 
categories_plot(df_cat,'MSZoning')


# After much exploration I found that In some columns one category is highly dominating.We will see those features and make some modifications

# In[ ]:


modified_cols = ['PoolArea','Street','MasVnrType','RoofMatl','Utilities']

for col in modified_cols:
    categories_plot(df_cat,col)


# In[ ]:


# some feature engineering on cat features

modified_cols = ['PoolArea','Street','MasVnrType','RoofMatl']


df_cat = df_cat.drop(columns = ['Utilities'])
df_cat_test = df_cat_test.drop(columns = ['Utilities'])

df_cat['PoolArea'] = df_cat['PoolArea'].apply(lambda x: 'Y' if x>1 else 'N') 
df_cat_test['PoolArea'] = df_cat_test['PoolArea'].apply(lambda x: 'Y' if x>1 else 'N')

df_cat['Street'] = df_cat['Street'].apply(lambda x: 'Pave' if x == 'Pave' else 'No Pave')
df_cat_test['Street'] = df_cat_test['Street'].apply(lambda x: 1 if x == 'Pave' else 0)

df_cat['MasVnrType'] = df_cat['MasVnrType'].apply(lambda x: 'N' if x == 'None' else 'Y')
df_cat_test['MasVnrType'] = df_cat_test['MasVnrType'].apply(lambda x: 'N' if x == 'None' else 'Y')

df_cat['RoofMatl'] = df_cat['RoofMatl'].apply(lambda x: 'CompShg' if x == 'CompShg' else 'Other')
df_cat_test['RoofMatl'] = df_cat_test['RoofMatl'].apply(lambda x: 'CompShg' if x == 'CompShg' else 'Other')


# **Concating numeric and categorical features**

# In[ ]:


# saving coumn names
cat_cols = df_cat.columns.to_list()
num_cols = df_num.columns.to_list()

df_test_num = df_test_num.reset_index(drop=True)
df_num = df_num.reset_index(drop=True)
df_cat = df_cat.reset_index(drop=True)
df_test_cat = df_cat_test.reset_index(drop=True)

final_train = pd.concat([df_num,df_cat],axis=1)
final_test = pd.concat([df_test_num,df_test_cat],axis=1)

print('Final shapes:')
print(final_train.shape)
print(final_test.shape)


# In[ ]:


# apply box cox transform to features having skweness > 0.5
def sqrt_skew(df):
    
    sk_feats = df.apply(lambda x: stats.skew(x)).sort_values(ascending=False)
    high_skew = sk_feats[abs(sk_feats) > 0.5].index
    for feat in high_skew:
#         df[feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))
          df[feat] = np.sqrt(df[feat])
        
    return df


# In[ ]:


# ncols = num_cols[:-1]
# final_train[ncols] = sqrt_skew(final_train[ncols])
# final_test[ncols] = sqrt_skew(final_test[ncols])


# ### Train test Split

# In[ ]:


Y = target.values
X = final_train
x_test = final_test.copy()
x_train,x_cv,y_train,y_cv = train_test_split(X,Y,train_size=0.7,random_state=100)
print('Train cv data shape')
print(x_train.shape)
print(x_cv.shape)
print(x_test.shape)


# ### Missing values in numeric features

# In[ ]:


# imputing numeric values
# Featurization of numeric data
# df_num = df_num.drop(columns=['SalePrice'])

num_cols_in = df_num.columns.to_list()
num_cols_in.remove('SalePrice')

imputer = SimpleImputer(strategy='median')
x_train_num = imputer.fit_transform(x_train[num_cols_in])
x_cv_num = imputer.transform(x_cv[num_cols_in])
x_test_num = imputer.transform(x_test[num_cols_in])


# ### Normalization
# 
# Eventhough there is no assumption that in linear regression input data is to be normalized, it is a good option to normalize beacuse,
# 
# * MSE is prone to outliers
# * Normalization helps for faster convergence
# * Helps to remove skweness
# 

# In[ ]:


#Normalizing
scaler = RobustScaler()
x_train_num = scaler.fit_transform(x_train_num)
x_cv_num = scaler.transform(x_cv_num)
x_test_num = scaler.transform(x_test_num)

df_num = pd.DataFrame(x_train_num, columns=num_cols_in)
df_cv_num = pd.DataFrame(x_cv_num, columns=num_cols_in)
df_test_num = pd.DataFrame(x_test_num, columns=num_cols_in)


# ### Missing values in categoric features

# In[ ]:



# cat_cols = df_cat.columns.to_list()

# # missing values in df
# imputer = SimpleImputer(strategy='constant', fill_value='MISSING')
# df_cat = imputer.fit_transform(df_cat[cat_cols])
# df_cat_test = imputer.transform(df_cat_test[cat_cols])

# df_cat = pd.DataFrame(df_cat, columns=cat_cols)
# df_cat_test = pd.DataFrame(df_cat_test, columns=cat_cols)

x_train_cat = x_train[cat_cols]
x_cv_cat = x_cv[cat_cols]
x_test_cat = x_test[cat_cols]

for col in cat_cols:
    val = x_train_cat[col].mode()[0]
    x_train_cat[col] = x_train_cat[col].fillna(val)
    x_cv_cat[col] = x_cv_cat[col].fillna(val)
    x_test_cat[col] = x_test_cat[col].fillna(val)


# ## Onehot encoding categories

# In[ ]:


df_cat_dummy = pd.get_dummies(x_train_cat, columns=cat_cols,drop_first=True)
df_cv_cat_dummy = pd.get_dummies(x_cv_cat, columns=cat_cols,drop_first=True)
df_test_cat_dummy = pd.get_dummies(x_test_cat, columns=cat_cols,drop_first=True)
print(df_cat_dummy.shape)
print(df_cv_cat_dummy.shape)
print(df_test_cat_dummy.shape)

df_cat, df_cat_cv = df_cat_dummy.align(df_cv_cat_dummy, join='left', axis=1) 
df_cat, df_cat_test = df_cat_dummy.align(df_test_cat_dummy, join='left', axis=1) 
df_cat_test = df_cat_test.fillna(0)
df_cat_cv = df_cat_cv.fillna(0)

print('dummy categorical data shapes after aligning with train data')
print(df_cat.shape)
print(df_cat_cv.shape)
print(df_cat_test.shape)


# In[ ]:


# reseting index
df_cat_dummy = df_cat.reset_index(drop=True)
df_cv_cat_dummy = df_cat_cv.reset_index(drop=True)
df_test_cat_dummy = df_cat_test.reset_index(drop=True)
df_num = df_num.reset_index(drop=True)
df_cv_num = df_cv_num.reset_index(drop=True)
df_test_num = df_test_num.reset_index(drop=True)

final_train = pd.concat([df_num,df_cat_dummy],axis=1)
final_cv = pd.concat([df_cv_num,df_cv_cat_dummy],axis=1)
final_test = pd.concat([df_test_num,df_test_cat_dummy],axis=1)

print('Final shapes:')
print(final_train.shape)
print(final_cv.shape)
print(final_test.shape)


# >*Note: It is notable that, In most of the kernels,they have used fitting and transforming whole data first and then splitting to train and cross validation. I think that this can cause data leakage issue as we have to handle cross validation and test data as totally unseen.That means we have to first split data to train, cv and test.Then fit on train and transform on cv and test.
# For example in one hot encoding, train,cv and test should only contain categories from train.
# *

# # ML Models

# ## Linear regression
# 
# You can read more about linear regression and metrics [here](http://https://www.kaggle.com/masumrumi/a-detailed-regression-guide-with-house-pricing)

# In[ ]:


x_train = final_train.copy()
x_cv = final_cv.copy()
x_test = final_test.copy()


# In[ ]:


linear = LinearRegression()
linear.fit(x_train,y_train)

y_pred_train = linear.predict(x_train)
y_pred_cv = linear.predict(x_cv)


print('Root Mean Square Error train = ' + str(np.sqrt(mean_squared_error(y_train, y_pred_train))))
print('Root Mean Square Error test = ' + str(np.sqrt(mean_squared_error(y_cv, y_pred_cv)))) 
print('R2 score = ' + str(r2_score(y_cv,y_pred_cv)))


# In[ ]:


"""
With only numerical columns 
Root Mean Square Error train = 0.13270971547191085
Root Mean Square Error test = 0.14181733405952673

# with categorical cols
Root Mean Square Error train = 0.0864434894176389
Root Mean Square Error test = 0.137230822787648
R2 score = 0.888797531197542

"""


# ## Ridge Regression
# If you wish to read more on regularization [here](http://https://medium.com/@arunm8489/an-overview-on-regularization-f2a878507eae)

# In[ ]:


ridge = Ridge()
ridge.fit(x_train,y_train)

y_pred_train = ridge.predict(x_train)
y_pred_cv = ridge.predict(x_cv)


print('Root Mean Square Error train = ' + str(np.sqrt(mean_squared_error(y_train, y_pred_train))))
print('Root Mean Square Error test = ' + str(np.sqrt(mean_squared_error(y_cv, y_pred_cv)))) 
print('R2 score = ' + str(r2_score(y_cv,y_pred_cv)))


# In[ ]:


"""
Root Mean Square Error train = 0.08799576547555985
Root Mean Square Error test = 0.12886916737699786
"""


# # Lasso

# In[ ]:


lasso = Lasso()
lasso.fit(x_train,y_train)

y_pred_train = lasso.predict(x_train)
y_pred_cv = lasso.predict(x_cv)


print('Root Mean Square Error train = ' + str(np.sqrt(mean_squared_error(y_train, y_pred_train))))
print('Root Mean Square Error test = ' + str(np.sqrt(mean_squared_error(y_cv, y_pred_cv)))) 
print('R2 score = ' + str(r2_score(y_cv,y_pred_cv)))


# ### Feature selection using Lasso(This is done just for demonstaration purpose)
# * Let us check how we can do feature selection using Lasso
# * Default alpha value is 1, using it will select minimum number of features(I end up with 1 feature).So I choose a lower value for alpha, so that it does not penalizes weights much.

# In[ ]:



lasso = Lasso(alpha=0.05)
lasso.fit(x_train,y_train)

y_pred_train = lasso.predict(x_train)
y_pred_cv = lasso.predict(x_cv)


print('Root Mean Square Error train = ' + str(np.sqrt(mean_squared_error(y_train, y_pred_train))))
print('Root Mean Square Error test = ' + str(np.sqrt(mean_squared_error(y_cv, y_pred_cv)))) 
print('R2 score = ' + str(r2_score(y_cv,y_pred_cv)))


# In[ ]:


coef = pd.Series(lasso.coef_, index = x_train.columns).sort_values()
imp_coef = pd.concat([coef.head(10), coef.tail(10)])
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Model")


# Observation: We got 5 important features. 'OverallQual','GrLivArea','TotalBsmtSF','GarageCars' and 'LotArea'.

# ## Elastic Net

# In[ ]:


el = ElasticNet()
el.fit(x_train,y_train)

y_pred_train = el.predict(x_train)
y_pred_cv = el.predict(x_cv)


print('Root Mean Square Error train = ' + str(np.sqrt(mean_squared_error(y_train, y_pred_train))))
print('Root Mean Square Error test = ' + str(np.sqrt(mean_squared_error(y_cv, y_pred_cv)))) 
print('R2 score = ' + str(r2_score(y_cv,y_pred_cv)))


# ## Cross validation on Ridge regression

# In[ ]:


for i in range (-2, 3):
    alpha = 10**i
    rm = Ridge(alpha=alpha)
    ridge_model = rm.fit(x_train, y_train)
    preds_ridge = ridge_model.predict(x_cv)

    plt.scatter(preds_ridge, y_cv, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    ridge_model.score(x_cv, y_cv),
                    np.sqrt(mean_squared_error(y_cv, preds_ridge)))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()


# In[ ]:


ridge = Ridge(alpha=10)
ridge.fit(x_train,y_train)

y_pred_train = ridge.predict(x_train)
y_pred_cv = ridge.predict(x_cv)


print('Root Mean Square Error train = ' + str(np.sqrt(mean_squared_error(y_train, y_pred_train))))
print('Root Mean Square Error test = ' + str(np.sqrt(mean_squared_error(y_cv, y_pred_cv)))) 
print('R2 score = ' + str(r2_score(y_cv,y_pred_cv)))


# Now we will try to interpret the coefficients

# In[ ]:


coef = pd.Series(ridge.coef_, index = x_train.columns).sort_values()
imp_coef = pd.concat([coef.head(10), coef.tail(10)])
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Model")


# How to interpret these coefficients?
# 
# Its rather simple.However I will explain with an example.Take case of GrLivArea. Its coefficient is 0.150052. It means that for 1 unit increase in **GrLivArea**, SalePrice increase by  0.150052 units.

# Now we will check homoscedacity and uniform distribution of residuals with fitted value(Assumptions).

# In[ ]:


x_plot = plt.scatter(y_pred_cv, (y_pred_cv - y_cv), c='b')
plt.title('Residual plot')
plt.show()
sns.distplot((y_pred_cv - y_cv))
plt.show()


# ## Future Scope
# 
# * Our model is slightly overfitting.We can overcome this if we use more advanced models like randomforest,xgboost etc. Soon I will include those in my kernel.
# 
# * There is lot of scope for feature engineering which can improve our result.
# 
# * Feel free to correct me if i had done any mistakes.

# ### If you find my kernel useful do not forget to Upvote.
# 
# Soon I will incoperate more details in this kernel.Until then **Happy Machine Learning!**
