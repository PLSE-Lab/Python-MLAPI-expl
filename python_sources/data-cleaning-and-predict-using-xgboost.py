#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Overview
# I am trying to lear Data Science. After finish some [kaggle's micro courses](https://www.kaggle.com/learn/overview), this is my first kernel on kaggle.
# If you have any advice for me, please left me a comment. It would be greatly appreciated.
# 
# 
# * 1. Data exploring
#     * 1.1 Data overview
#     * 1.2 Exploring numberical columns
#     * 1.3 Exploring categorical columns
# 
# * 2. Data clean and preprocess
#   
# * 3. Modeling

# # 1. Data exploring

# In[ ]:


pd.set_option("display.max_columns", 100)

# load train data
data = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv", index_col="Id")


# ## 1.1 Data overview

# We have 80 columns(included the target) and 1460 rows of training data.
# So we have 79 potential features or variables.

# In[ ]:


data.shape


# Here is some sample rows

# In[ ]:


data.head()


# In[ ]:


categorical_cols = data.select_dtypes(include=['object']).columns
numberical_cols = data.select_dtypes(exclude=['object']).columns
print("Numberical columns: ", len(numberical_cols))
print("Categorical columns: ", len(categorical_cols))


# The numberical columns have following characteristics:

# In[ ]:


data[numberical_cols].describe()


# The categorical columns have following characteristics:

# In[ ]:


data[categorical_cols].describe()


# # 1.2 Exploring numberical variables
# 

# - Find out low correlated variable with SalePrice. We will not using them.
# - Find out strong correlations which we will use for modeling. And look closer.
#     - Because outliers can influence heavily the correlation, we need to detect ouliers(distribution, scatter plot) them remove these rows
#     - Find out missing value and find strategy(constant, mean, median) to fill
#     - Find strong corelated variables(similar feature). We can consider use one of them.

# ## Look at correations with SalePrice
# 
# 

# In[ ]:


fig = plt.figure(figsize=(30,1))
numberical_col_corr = data[numberical_cols].corr().loc[["SalePrice"],:].sort_values(by="SalePrice",axis=1)
sns.heatmap(data=numberical_col_corr, annot=True)


# In[ ]:


weak_corr_threadhold = 0.2
weak_numberical_cols = list(numberical_col_corr[abs(numberical_col_corr) < weak_corr_threadhold].dropna(axis=1))
print("There are %d weak correlated(< %.2f) variables with SalePrice:" % (len(weak_numberical_cols), weak_corr_threadhold))
print(weak_numberical_cols)


# In[ ]:


strong_numberical_cols = numberical_col_corr.columns[len(weak_numberical_cols):-1][::-1]
strong_cols_num = len(strong_numberical_cols)

print("There are %d strong correlated(> %.2f) variables with SalePrice:" % (strong_cols_num-1, weak_corr_threadhold))
print(list(strong_numberical_cols))


# In[ ]:


numberical_null_count = data[numberical_cols].isnull().sum()
print("Missing values: ")
print(numberical_null_count[numberical_null_count>0])


# Let see scatter plot and distribution of each variables

# In[ ]:


plt.rcParams.update({'font.size': 14})

def explore_variable(col_name):
    strong_crr_cols = find_high_corelated_variables(col_name)
    draw_variable([col_name] + strong_crr_cols)


def draw_variable(col_names):
    num_cols = len(col_names)
    fig = plt.figure(figsize=(8, num_cols*4))
    i = 0
    for col in col_names:
        fig.add_subplot(num_cols, 2, 2*i+1)
        sns.regplot(x=data[col], y=data["SalePrice"])
        plt.xlabel(col)
        plt.title('Corr to SalePrice = %.2f'% numberical_col_corr[col])
        fig.add_subplot(num_cols, 2, 2*i+2)
        sns.distplot(data[col].dropna())
        plt.xlabel(col)
        i += 1
        
    plt.tight_layout()

        


variable_corr = data[list(set(numberical_cols)-set(["SalePrice"]))].corr()
high_corr_threadhold = 0.7

def find_high_corelated_variables(col_name):
    corr = variable_corr.loc[[col_name],:]
    strong_corr = corr[(corr>=high_corr_threadhold) & (corr<1)].dropna(axis=1)
    print("Strong corelated variables:")
    print(strong_corr)
    return list(strong_corr.columns)


# ### OverallQual
# - Good shape of distribution
# - no outlier
# - The missing value cand be fill by mean() or median()

# In[ ]:


explore_variable("OverallQual")


# ### GrLivArea
# - Slightly left skew distribution but still in good shape.
# - 2 outliers(> 4500)
# - The missing values can be filled by median()
# - Keep both GrLivArea and TotRmsAbvGrd
# 

# In[ ]:


explore_variable("GrLivArea")


# In[ ]:


data["GrLivArea"].describe()


# In[ ]:


data["GrLivArea"].sort_values().tail()


# ### GarageCars
# - Good shape of  distribution
# - no outliers
# - The missing values can be filled by mean() or median()
# - Keep both GarageCars and GarageArea

# In[ ]:


explore_variable("GarageCars")


# ### 1stFlrSF
# - Left skew of distribution
# - 1 outlier (>5000)
# - The missing value can be filled by median()
# - We can remove strong related variable TotalBsmtSF 

# In[ ]:


explore_variable("1stFlrSF")


# ### TotalBsmtSF
# - We will remove it and use 1stFlrSF.

# In[ ]:


explore_variable("TotalBsmtSF")


# ### TotRmsAbvGrd
# - Slightly left skew distribution
# - No outlier
# - The missing values can be filled by median()
# - Keep both TotRmsAbvGrd and TotRmsAbvGrd

# In[ ]:


explore_variable("TotRmsAbvGrd")


# ### YearBuilt
# - Right skew distribution
# - No outlier
# - This is basic infomation of a hourse, so the columns should not has any missing values.
# - We can remove strong related variable GarageYrBlt

# In[ ]:


explore_variable("YearBuilt")


# ### YearRemodAdd
# - ? distribution
# - At the first look, maybe some outlier where the price went too high. However go back to OverallQual scatter plot, these house has OverallQual 10. It's resonable. -> No oulier
# - Missing value should be fill by YearBuilt value. However, in both train and test dataset there is no null value of YearRemodAdd, so we don't need to fill this variavle.

# In[ ]:


explore_variable("YearRemodAdd")


# ### GarageYrBlt
# We will remove it and use YearBuilt

# In[ ]:


explore_variable("GarageYrBlt")


# ### MasVnrArea
# - Left skew distribution. Almost values = 0
# - No outlier
# - Missing value can be filled by 0

# In[ ]:


explore_variable("MasVnrArea")


# ### Fireplaces
# - ? distribution
# - No outlier
# - Missing value can be filled as 0 (No Fireplaces-most_frequent_value)

# In[ ]:


explore_variable("Fireplaces")


# ### BsmtFinSF1
# - Left skew distribution. Almost values are 0.
# - 1 outlier(>4000)
# - Missing value can be filled by median()

# In[ ]:


explore_variable("BsmtFinSF1")


# ### LotFrontage
# - Left skew distribution
# - 2 outliers( >300)
# - Missing values can be filled by median()

# In[ ]:


explore_variable("LotFrontage")


# ### WoodDeckSF
# - Left skew distribution. Almost values = 0
# - No outlier
# - Missing value can be filled by 0

# In[ ]:


explore_variable("WoodDeckSF")


# ### 2ndFlrSF
# - Left skew distribution. Most values are 0
# - No outliers
# - Missing value can be filled by 0

# In[ ]:


explore_variable("2ndFlrSF")


# ### OpenPorchSF
# - Left skew distribution. Most values are 0
# - No outlier
# - Missing values can be filled by 0

# In[ ]:


explore_variable("OpenPorchSF")


# ### HalfBath
# - ? distribution
# - No outlier
# - Missing value can be fill as 0( most frequent value)

# In[ ]:


explore_variable("HalfBath")


# ### LotArea
# - Left skew distribution
# - No outlier
# - Missing values can be filled by median()

# In[ ]:


explore_variable("LotArea")


# ### BsmtFullBath
# - ? distribution
# - No outlier
# - Missing value can be filled by 0(most frequent value)

# In[ ]:


explore_variable("BsmtFullBath")


# ### BsmtUnfSF
# - Left skew distribution
# - No outlier
# - Missing values can be filled by median()
# 

# In[ ]:


explore_variable("BsmtUnfSF")


# There are 3 of 37 columns has missing value.
# Below are the missing values.(1460 rows in total)

# # 1.3 Exploring categorical variables

# As show bellow, by addition of KitchenQual, the price ranges are seperated quite clearly.
# 
# Basically, we will use all categorical columns and using One Hot Encoding but with large number of unique value we can consider to use LabelEncoder, CountEncoder, TargetEncoder or CatBoostEncoder.
# 
# We also need to remove un-useful columns, which has large number of missing values.

# In[ ]:


fig = plt.figure(figsize=(6,12))
sns.lmplot(x="MasVnrArea", y="SalePrice", hue="KitchenQual", data=data) 


# In[ ]:


fig = plt.figure(figsize=(6,12))
sns.lmplot(x="2ndFlrSF", y="SalePrice", hue="KitchenQual", data=data) 


# The number of unique values each variable are show as bellow.
# Not much of variable so we will apply One Hot Encoding for all categorical variables.

# In[ ]:


unique_val_num_dict = {col:len(data[col].unique())  for col in categorical_cols}
sorted(unique_val_num_dict.items(), key=lambda x: x[1])


# Bellow is the number of null values.
# 
# We will drop Alley, PoolQC, Fence, MiscFeature.
# 
# The other missing values will be fill up by SimpleImputer-most_frequent value

# In[ ]:


null_count = data[categorical_cols].isnull().sum()
null_count[null_count>0].dropna(axis=0)


# ## 2. Data clean and preprocess

# ## Numberical columns
# 
# * Drop columns: 
#     * Have strong corelated with other variables:
#         * ['TotalBsmtSF', 'GarageYrBlt']
#     * Weak corelation with SalePrice: 
#         * ['KitchenAbvGr', 'EnclosedPorch', 'MSSubClass', 'OverallCond', 'YrSold', 'LowQualFinSF', 'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2', '3SsnPorch', 'MoSold', 'PoolArea', 'ScreenPorch', 'BedroomAbvGr']
# 
# 
# * Fill missing values
#     * by zero:
#         * ['MasVnrArea', 'Fireplaces', 'WoodDeckSF', '2ndFlrSF', 'OpenPorchSF', 'HalfBath', 'BsmtFullBath']
#     * by median:
#         * ['OverallQual', 'GrLivArea', 'GarageCars', '1stFlrSF', 'TotRmsAbvGrd', 'BsmtFinSF1', 'LotFrontage', 'LotArea', 'BsmtUnfSF']
# 
# 
# * Remove Outliers:
#     * GrLivArea 2 outliers(> 4500)
#     * 1stFlrSF 1 outlier (>5000)
#     * BsmtFinSF1 1 outlier(>4000)
#     * LotFrontage 2 outliers( >300)
# ## Numberical columns
# * Drop columns: 
#     * ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
#     

# 

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


median_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

zero_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('std_scaler', StandardScaler()),
])
#zero_transformer = SimpleImputer(strategy='constant', fill_value=0)
#median_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[ ]:


drop_cols =  ['Alley', 'PoolQC', 'Fence', 'MiscFeature'] + ['TotalBsmtSF', 'GarageYrBlt'] + ['KitchenAbvGr', 'EnclosedPorch', 'MSSubClass', 'OverallCond', 'YrSold', 'LowQualFinSF', 'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2', '3SsnPorch', 'MoSold', 'PoolArea', 'ScreenPorch', 'BedroomAbvGr']
zero_fill_cols  = ['MasVnrArea', 'Fireplaces', 'WoodDeckSF', '2ndFlrSF', 'OpenPorchSF', 'HalfBath', 'BsmtFullBath']
median_fill_cols =  ['OverallQual', 'GrLivArea', 'GarageCars', '1stFlrSF', 'TotRmsAbvGrd', 'BsmtFinSF1', 'LotFrontage', 'LotArea', 'BsmtUnfSF']

preprocessor = ColumnTransformer(
                transformers=[
                    ('num_zero', zero_transformer, zero_fill_cols),
                    ('median_zero', median_transformer, median_fill_cols),
                    ('cat', categorical_transformer, list(set(categorical_cols) - set(['Alley', 'PoolQC', 'Fence', 'MiscFeature'])))
                ])

train_data = data.copy()
train_data.drop(drop_cols, axis=1, inplace=True)
train_data.drop(train_data[(train_data['GrLivArea'] > 4500) |
                (train_data['1stFlrSF'] > 5000) |
                (train_data['BsmtFinSF1'] > 4000) |
                (train_data['LotFrontage'] > 300)].index
                ,axis=0, inplace=True)

y = train_data.SalePrice
X = train_data.drop('SalePrice', axis=1)
feature_cols = list(X.columns)


# # 3. Modeling

# Following [kaggle micro course](https://www.kaggle.com/learn/intermediate-machine-learning) , I only know how to use DecisionTree, RandomForest and XGBoost.
# XGBoost gets the best accuracy so I use XGBoost. (I will try apply model turing after learn more about ML Algorithms)

# I divide dataset to 2 parts(0.8/0.2).
# Below is the distribution of SalePrice each part.

# In[ ]:


X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
#plt.figure(figsize=(6,6))
#sns.distplot(X_valid_full["OverallQual"])
#sns.distplot(test_data["OverallQual"])
#plt.legend(["OverallQual valid", "OverallQual test"])


# Now is time for modeling

# In[ ]:



model = XGBRegressor(n_estimators=1000,
                         learning_rate=0.05, 
                         early_stopping_rounds=20, 
                         eval_set=[(X_valid_full, y_valid)],
                         random_state=0)

#model = XGBRegressor(n_estimators=1000,learning_rate = 0.01,random_state=0)

my_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', model)])

my_pipeline.fit(X_train_full,y_train)

mae = mean_absolute_error(y_valid, my_pipeline.predict(X_valid_full))
print("MAE:" ,mae)



test_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv", index_col="Id")
predict_result = my_pipeline.predict(test_data[feature_cols])
output = pd.DataFrame({'Id':test_data.index, 
                        'SalePrice':predict_result})
output.to_csv('submission.csv', index=False)

