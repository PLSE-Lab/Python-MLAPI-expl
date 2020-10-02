#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor,AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso,Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler,MinMaxScaler
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print(train.shape,test.shape)


# In[ ]:


pd.pandas.set_option('display.max_columns',None)
train.head()


# # WorkFlow
# 
#    1. Data Analysis
#    2. Feature Engineering 
#    3. Model Building
#    4. Ensembling

# # Data Anlysis
# 
#    1. Missing values
#    2. Numerical Variables
#    3. Distribution of numerical variables
#    4. Categorical variables
#    5. Cardinality of categorical variables
#    6. Outliers
#    7. Relationship between dependent and independent variables

# ## 1.Missing values

# In[ ]:


na_columns = [col for col in train.columns if train[col].isna().sum()>=1]
# percentage of missing values
for col in na_columns:
    print(col,np.round(train[col].isna().mean(),4))


# **Now lets find out the relationship between dependent variable and missing value**
# 
# 
# Let's vizualize it by some plots for this i will replace missing values by 1 and non null values by 0

# In[ ]:


for col in na_columns:
    df = train.copy()
    df[col] = np.where(df[col].isna(),1,0)
    #median salesprice for missing values and non missing values
    fig = px.bar(y = df.groupby(col)['SalePrice'].mean(),width=500,height = 500,template="plotly_dark",labels={'x':col,'y':'Mean Saleprice'})
    fig.show()


# We can see that for column with more missing values mean values of houses are more than the non misssing values instances, hence i will replace missing values with some meaningful value in featue engineering section

# ## 2.Numerical variables

# In[ ]:


numerical_col = [col for col in train.columns if train[col].dtypes !='O']
print(numerical_col)
print(len(numerical_col))


# In[ ]:


df[numerical_col].head()


# **Types of numerical variables**
# 
# a. Temporal Variables(datetime)
# 
# b. Descrete Variables
# 
# c. Continuous Variables
# 
#  
# a. We have 4 datetime columns(temporal variables) in the numerical variables let's try to see relationship of these with dependent variable

# In[ ]:


year_col = ['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']
fig = px.line(train.groupby('YrSold')['SalePrice'].median(),width=600,height=400,template='plotly_dark')
fig.show()


# We can see that the house price is decreasing with time but this is not the full picture there are some other raltions with other temporal variables

# In[ ]:


for col in year_col:
    if col != 'YrSold':
        df = train.copy()
        df[col] = df['YrSold']-df[col]
        fig = px.scatter(x = df[col],y = df['SalePrice'],width=650,height=450,labels={'x':col,'y':'Saleprice'},template="plotly_dark")
        fig.show()


# Now here is the full picture we can see that the as year difference between the yearSold and year of build,year of modifiaction or yeargaragebuild is increases the price of houses decreses.
# 
# **Price of old or not recently modified houses are less** 

# b. Descrete Variables
# 
# Let's filter out descrete columns by the rule that if a particular column has less than 25 unique values then it is descrete

# In[ ]:


descrete_col = [col for col in numerical_col if len(train[col].unique())<25 and col not in year_col]
len(descrete_col)


# In[ ]:


train[descrete_col].head()


# In[ ]:


for col in descrete_col:
    df = train.copy()
    plt.style.use('ggplot')
    with plt.style.context('dark_background'):
        df.groupby(col)['SalePrice'].median().plot.bar(color = 'c')
    plt.ylabel('Median SalePrice')
    plt.show()


# From the plots we can see for some of the faetures saleprice is Exponentialy increasing with the descrete variables, for some of them it is decreasing and for some of them it's behaviour is complex

# c. Continuous Varible
# 
# Let's see the distribution and relationship of continuous numerical features with SalePrice

# In[ ]:


conti_col = [col for col in numerical_col if col not in descrete_col+year_col+['Id']]
print(len(conti_col),'continuous features')
print(conti_col)


#    **Distributions of continuons numerical features**

# In[ ]:


for col in conti_col:
    df = train.copy()
    fig = px.histogram(x = col,data_frame= df,width=600,height=450,template='plotly_dark')
    fig.show()


#    **How Saleprice is varying with continuous numerical features**

# In[ ]:


for col in conti_col:
    df = train.copy()
    if 0 in df[col].unique():
        pass
    else:
        df[col] = np.log(df[col])
        fig = px.scatter(x = df[col],y = np.log(df['SalePrice']),labels = {'x' : col,'y':'Saleprice'},width=650,height=400,template='plotly_dark')
        fig.show()


# **Visualizing outliers using boxoplot**

# In[ ]:


for col in conti_col:
    df = train.copy()
    if 0 in df[col].unique():
        pass
    else:
        df[col] = np.log(df[col])
        fig = px.box(data_frame=df,y = df[col],labels={'x':col,'y': col},width=500,height=400,template = 'plotly_dark')
        fig.show()


# # 3.Categorical varibales

# In[ ]:


cat_col = [col for col in train.columns if train[col].dtypes == 'O']
print(cat_col)
print('We have {} categorical features'.format(len(cat_col)))


# In[ ]:


train[cat_col].head()


# ## 5.Cardinality of categorical features 

# In[ ]:


for col in cat_col:
    print(col,': {}'.format(train[col].nunique()))


# **Features Neighborhood, Exterior1st,Exterior2nd have high cardinality hence I will deal with them in feature engineering**
# 
#  Rest of them can be easily dealt by One-Hot Encoding

# **Relationship between categorical and Dependent variable**

# In[ ]:


for col in cat_col:
    df = train.copy()
    fig = px.bar(df.groupby(col)['SalePrice'].mean(),height=400,width=600,template='plotly_dark')
    fig.show()


# **That is all the necessory data analysis from my end I will do the feature engineering in the next version**

# # Feature Engineering
# 
# 1. Missing Values
# 2. Temporal variables
# 3. Trnsforming Numerical Variables for removing outliers
# 4. Categorical variables:remove rare labels
# 5. Encoding Categorical Variables

# In[ ]:


train.head()


# In[ ]:


test.tail()


# In[ ]:


y = train['SalePrice'].reset_index(drop=True)
previous_train = train.copy()


# In[ ]:


all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)


# ## 1.Missing values

# In[ ]:


##Let's first handle categorical variables which are missing
nan_feature = [col for col in all_data.columns if all_data[col].isnull().sum()>=1 and all_data[col].dtypes == 'O']
print(len(nan_feature))
for feture in nan_feature:
    print('{} has {} % missing values'.format(feture,np.round(all_data[feture].isnull().mean(),4)))


# In[ ]:


#replace missing values with a new label
def replace_missing(data,missing_feature):
    df = data.copy()
    df[missing_feature] = df[missing_feature].fillna('missing')
    return df

all_data = replace_missing(all_data,nan_feature)
all_data[nan_feature].isnull().sum()


# In[ ]:


## Now let's check the numerical columns contaning missing values
num_na_col = [col for col in all_data.columns if all_data[col].isnull().sum()>=1 and all_data[col].dtypes != 'O']
for col in num_na_col:
    print('{} has {}% null values'.format(col,np.round(all_data[col].isnull().mean(),4)))


# In[ ]:


#replacing by median since data contain outliers
for col in num_na_col:
    median_value = all_data[col].median()
    
    all_data[col+'NaN'] = np.where(all_data[col].isnull(),1,0)##New Feature to capture missing value
    all_data[col].fillna(median_value,inplace = True) ##Replacing with median in corresponding feature 
    
all_data[num_na_col].isnull().sum()


# ## 2. Temporal variables

# In[ ]:


for col in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    all_data[col] = all_data['YrSold']-all_data[col]
    
all_data[year_col].head()


# ## 3. Numerical Variables
# 
# **Since the numerical variables are skewed we will perform log transformation**

# In[ ]:


num_col = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']

for col in num_col:
    all_data[col] = np.log1p(all_data[col])


# ## 4.Handling categorical variables
# 
# **I will first remove rare label that are present less than 1% in the categorical feature**
# 
# **In the next step i will encode the categorical variables**

# In[ ]:


## Handling Rare lables occuring in categorical variables
cat_col = [col for col in all_data.columns if all_data[col].dtypes == 'O']
for col in cat_col:
    temp = all_data[col].value_counts()/len(all_data)
    temp_df = temp[temp>0.01].index
    all_data[col] = np.where(all_data[col].isin(temp_df),all_data[col],'RareVar')


# In[ ]:


all_data.head()


# In[ ]:


all_data.shape


# In[ ]:


cat_train = all_data.iloc[:len(y), :]

cat_test = all_data.iloc[len(y):, :]


# In[ ]:


cat_train.head()


# In[ ]:


cat_test.tail()


# In[ ]:


all1 = pd.get_dummies(all_data,drop_first=True)


# In[ ]:


all1.shape


# In[ ]:


x_train = all1.iloc[:len(y), :]

x_test = all1.iloc[len(y):, :]


# In[ ]:


x_train.head()


# In[ ]:


x_test.head()


# In[ ]:


x_train.drop('Id',axis = 1,inplace = True)
x_test.drop('Id',axis = 1,inplace = True)
cat_train.drop('Id',axis = 1,inplace = True)
cat_test.drop('Id',axis = 1,inplace = True)


# In[ ]:


x_train.describe()


# In[ ]:


y_transformed = np.log1p(y)


# In[ ]:


y_transformed


# In[ ]:


kf = KFold(n_splits=9, random_state=42, shuffle=True)


# In[ ]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X,y_true):
    rmse = np.sqrt(-cross_val_score(model, X, y_true, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


# In[ ]:


lasso_reg =make_pipeline(Min(),Lasso(alpha=0.0005,random_state=44))


# In[ ]:


scores = {}

score = cv_rmse(lasso_reg,x_train,y_transformed)
print("Lasso: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['Lasso'] = (score.mean(), score.std())


# In[ ]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# In[ ]:


scores = {}

score = cv_rmse(GBoost,x_train,y_transformed)
print("gboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['gboost'] = (score.mean(), score.std())


# In[ ]:


model_xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213,
                             random_state =7, nthread = -1)


# In[ ]:


scores = {}

score = cv_rmse(model_xgb,x_train,y_transformed)
print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['xgboost'] = (score.mean(), score.std())


# In[ ]:


model_lgb = LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# In[ ]:


scores = {}

score = cv_rmse(model_lgb,x_train,y_transformed)
print("lgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['lgbm'] = (score.mean(), score.std())


# In[ ]:


ridge = make_pipeline(RobustScaler(),Ridge(alpha = 0.0005,random_state = 12))


# In[ ]:


scores = {}

score = cv_rmse(ridge,x_train,y_transformed)
print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['ridge'] = (score.mean(), score.std())


# In[ ]:


bag = BaggingRegressor(base_estimator=model_lgb,n_estimators=2,random_state=2,n_jobs=-1)


# In[ ]:


scores = {}

score = cv_rmse(bag,x_train,y_transformed)
print("bag: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['bag'] = (score.mean(), score.std())


# In[ ]:


print('Lightgbm')
model_lgb.fit(x_train,y_transformed)

print('model_xgb')
model_xgb.fit(x_train,y_transformed)

print('Gradient_boosting')
GBoost.fit(x_train,y_transformed)

print('bagging')

bag.fit(x_train,y_transformed)


# In[ ]:


def blend_models_predict(X):
    return (#(0.16  * elastic_model.predict(X)) + \
            #(0.16 * lasso.predict(X)) + \
            (0.2 * GBoost.predict(X)) + \
            (0.3 * model_lgb.predict(X)) + \
            (0.2 * model_xgb.predict(X)) + \
#             (0.1 * xgb_model_full_data.predict(X)) + \
            (0.3 * bag.predict(np.array(X))))


# In[ ]:


print('RMSLE score on train data:')
print(rmsle(y_transformed, blend_models_predict(x_train)))


# In[ ]:


print('Predict submission')
submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.iloc[:,1] = (np.expm1(blend_models_predict(x_test)))


# In[ ]:


submission.to_csv("submission2.csv", index=False)


# In[ ]:


q1 = submission['SalePrice'].quantile(0.0042)
q2 = submission['SalePrice'].quantile(0.99)
#Quantiles helping us get some extreme values for extremely low or high values 
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv("submission3.csv", index=False)


# **Till this version of the notebook i haven't done any hyperparameter tuning maybe in the next version i will do some and imporve the score**

# In[ ]:




