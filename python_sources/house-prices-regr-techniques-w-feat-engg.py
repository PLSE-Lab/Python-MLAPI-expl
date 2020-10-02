#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import norm
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="darkgrid", color_codes=True,)
sns.set(font_scale=1)

#Data preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


#linear regression
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#AND suppressing warning; they tend to ruin an easy overview
import warnings
warnings.filterwarnings("ignore")


# # 1. Exploratory Data Analysis

# In[ ]:


# read train and test dataset
train_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
test_target = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")


# In[ ]:


# dropping ID column from dataframe
train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


# Shape of training data

print("Training data set shape => (Rows, Columns)= ",train_df.shape)
print("Testing data set shape => (Rows, Columns)= ",test_df.shape)


# #### Checking numerical and catagorical input features in train and test datasets

# In[ ]:


def check_features_count():
    train_cols = train_df.columns
    test_cols = test_df.columns
    train_num_cols = train_df._get_numeric_data().columns
    test_num_cols = test_df._get_numeric_data().columns
    print("Train numerical columns \n",len(train_num_cols))
    print("Test numerical columns \n",len(test_num_cols))

    train_cat_cols = list(set(train_cols) - set(train_num_cols))
    test_cat_cols = list(set(test_cols) - set(test_num_cols))

    print('\n')
    print("Train catagorical columns \n",len(train_cat_cols))
    print("Test catagorical columns \n",len(test_cat_cols))

def check_features():
    train_cols = train_df.columns
    test_cols = test_df.columns
    train_num_cols = train_df._get_numeric_data().columns
    test_num_cols = test_df._get_numeric_data().columns

    train_cat_cols = list(set(train_cols) - set(train_num_cols))
    test_cat_cols = list(set(test_cols) - set(test_num_cols))
    
    return [train_num_cols,test_num_cols,train_cat_cols,test_cat_cols]

check_features_count()


# In[ ]:



stat_features_columns=check_features()
train_num_cols = stat_features_columns[0]
test_num_cols = stat_features_columns[1]
train_cat_cols = stat_features_columns[2]
test_cat_cols = stat_features_columns[3]

print("train_num_cols",train_num_cols)
print("\n test_num_cols",test_num_cols)
print("\n train_cat_cols",train_cat_cols)
print("\n test_cat_cols",test_cat_cols)


# In[ ]:


unique_count_list=[]
for col_name in train_cat_cols:
    temp = (col_name, train_df[col_name].nunique())
    unique_count_list.append(temp)
cat_unique_value_df = pd.DataFrame(data=unique_count_list,columns = ['cat_feature_column_names','unique_value_count'])

cat_unique_value_df.sort_values('unique_value_count',ascending=False)


# In[ ]:


# Total Feature column count

def features_with_dummy():
    cat_feature_column_count = cat_unique_value_df['unique_value_count'].sum()

    total_feature_column_count = len(train_num_cols)+cat_feature_column_count
    
    return total_feature_column_count

print("Total Feature column count before Feature Engineering = ",features_with_dummy())


# ## >>>>>>>>>>>>>>>>>>>>>>>> Exploratory Data Analysis completed <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# # 2. Missing Data Handling

# In[ ]:


train_test_df = pd.concat([train_df,test_df])
train_test_df


# In[ ]:


# Missing data analysis
def missing_data_analysis(dataset,title,col):
    list_missing_data = []

    for column_name in dataset:
        temp = (column_name,dataset[column_name].isnull().sum(),dataset[column_name].isnull().sum()/dataset.shape[0])
        list_missing_data.append(temp)

    missing_data_df = pd.DataFrame(data = list_missing_data, columns = ["feature_columns",col,"percent"])
    missing_data_df = missing_data_df[missing_data_df[col] > 0].sort_values(col,ascending=False)
    return missing_data_df
    
train_test_missing_df = missing_data_analysis(train_test_df,'dataset feature column missing data analysis','missing_count')    

train_test_missing_df


# In[ ]:


# Missing Data chart analysis

f, ax = plt.subplots(figsize=(15, 3))
plt.xticks(rotation='90')
sns.barplot(x=train_test_missing_df['feature_columns'], y=train_test_missing_df['percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Missing data chart for dataset',color='b',size=20)
plt.show()
    
#missing_data_chart(data=train_missing_df,ax=1,suptitle="Train missing data chart")
#missing_data_chart(data=test_missing_df,ax=2,suptitle="Test missing data chart")


# In[ ]:


print("missing column count in data set: ",len(train_test_missing_df))
print()

list_missing_columns=list(train_test_missing_df['feature_columns'])
set_mis=set(list_missing_columns)
list_missing_columns=list(set_mis)
print("test and train missing columns: \n",list_missing_columns)


# ## Detailed analysis of missing values in Train and Test Dataset
# 
# ![image.png](attachment:image.png)

# In[ ]:


None_to_columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType']
zero_to_columns=['BsmtFullBath','BsmtHalfBath','GarageCars','BsmtUnfSF','TotalBsmtSF','BsmtFinSF2','BsmtFinSF1','SalePrice']
median_to_columns=['LotFrontage','MasVnrArea','GarageArea']
mode_to_columns=['MSZoning','Exterior1st','Exterior2nd','SaleType','Utilities','GarageYrBlt']
# GarageYrBlt is ordinal numerical value, hence it will changed to catagorical feature. 

def fillna(feature_columns,dataset,value):
    for col in feature_columns:
        dataset[col]=dataset[col].fillna(value)
        
def fillna_med(feature_columns,dataset):
    for col in feature_columns:
        dataset[col]=dataset[col].fillna(dataset[col].median())
        
def fillna_mode(feature_columns,dataset):
    for col in feature_columns:
        dataset[col]=dataset[col].fillna(dataset.loc[:,col].mode()[0])     
        
        
fillna(feature_columns=None_to_columns,dataset=train_test_df,value="None")

fillna(feature_columns=zero_to_columns,dataset=train_test_df,value=0)

fillna_med(feature_columns=median_to_columns,dataset=train_test_df)

fillna_mode(feature_columns=mode_to_columns,dataset=train_test_df)


train_test_df['Functional']=train_test_df['Functional'].fillna('Typ')
train_test_df['KitchenQual']=train_test_df['KitchenQual'].fillna('TA')
train_test_df['Electrical']=train_test_df['Electrical'].fillna('SBrkr')


# In[ ]:


train_df = train_test_df[train_test_df['Id']<1461]
test_df = train_test_df[train_test_df['Id']>1460]


# In[ ]:


missing_data_analysis(train_df,'dataset feature column missing data analysis','missing_count')


# ## >>>>>>>>>>>>>>>>>>>>>>>> Data cleaning completed  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# 
# # 3. Feature Engineering

# In[ ]:


Ordinal_train_df = train_df[['YearBuilt','YearRemodAdd','GarageYrBlt']].copy()
Ordinal_train_df['que1'] = Ordinal_train_df['YearBuilt'][(Ordinal_train_df['YearBuilt'] >= Ordinal_train_df['YearRemodAdd']) & (Ordinal_train_df['YearBuilt'] <= Ordinal_train_df['GarageYrBlt'])]
Ordinal_train_df['que2'] = Ordinal_train_df['YearBuilt'][(Ordinal_train_df['YearBuilt'] >= Ordinal_train_df['YearRemodAdd'])]
Ordinal_train_df['que3'] = Ordinal_train_df['YearBuilt'][(Ordinal_train_df['YearBuilt'] <= Ordinal_train_df['GarageYrBlt'])]
Ordinal_train_df['que4'] = Ordinal_train_df['YearRemodAdd'][(Ordinal_train_df['YearRemodAdd'] <= Ordinal_train_df['GarageYrBlt'])]
Ordinal_train_df.isnull().sum()


# * Only 11 houses are not built the garage the same year the house built. so we can elimate the GarageYrBlt

# In[ ]:


train_df.drop(['GarageYrBlt'], axis=1,inplace=True)
test_df.drop(['GarageYrBlt'], axis=1,inplace=True)


# As per test data description **MSSubClass** is "The building class", hence this is supposed to be Catagorical input. so this should be changed to str type.
# 
# **MSSubClass**,**'YearBuilt'**, **YearRemodAdd'**,**'MoSold'**, **'YrSold'** are catagorical input column feature but they are ordinal.

# In[ ]:


convert_dict = {'MSSubClass': str, 
                'YearBuilt': str,
                'YearRemodAdd': str, 
                'MoSold': str,
                'YrSold': str
               } 
  
train_df = train_df.astype(convert_dict)
test_df = test_df.astype(convert_dict)


# In[ ]:


stat_features_columns=check_features()
train_num_cols = stat_features_columns[0]
test_num_cols = stat_features_columns[1]
train_cat_cols = stat_features_columns[2]
test_cat_cols = stat_features_columns[3]

print("train_num_cols",train_num_cols)
print("\n test_num_cols",test_num_cols)
print("\n train_cat_cols",train_cat_cols)
print("\n test_cat_cols",test_cat_cols)


# In[ ]:


i = 1
plt.figure(figsize = (30,70))

for col in train_cat_cols:
    plt.subplot(13,4,i)
    sns.boxplot(x = train_df[col],y=train_df['SalePrice'])
    i=i+1
    plt.xticks(rotation='45',size=8)
    plt.yscale('log')

plt.show()


# From the above chart following data are inbalanced or no effect to Saleprice (Target variable) so the feature will create noise to the target variable, hence dropping those feature columns
# * Utilities
# * Condition2
# * RoofMatl
# * YrSold
# * MoSold
# 

# In[ ]:


train_df.drop(['Utilities','Condition2','RoofMatl','YrSold','MoSold'],axis=1,inplace=True)
test_df.drop(['Utilities','Condition2','RoofMatl','YrSold','MoSold'],axis=1,inplace=True)

check_features_count()


# ## Correlation matrix 

# In[ ]:


# pearson Correlated matrix
f, ax = plt.subplots(figsize=(18, 15))

corr_matrix = train_df.corr(method ='pearson')

sns.heatmap(corr_matrix,annot=True,annot_kws={'size': 7},vmax=.9,fmt=".1g",cmap='coolwarm',ax=ax,)
plt.title("Correlated Matrix of quantitative inputs", size = 20,color = 'g')


# GarageArea and GarageCars are 90% correlelated, hence we can remove GarageCars from feature columns

# In[ ]:


train_df.drop(['GarageCars'],axis=1,inplace=True)
test_df.drop(['GarageCars'],axis=1,inplace=True)


# ### Numerical column features relationship with Saleprice

# In[ ]:


stat_features_columns=check_features()
train_num_cols = stat_features_columns[0]
test_num_cols = stat_features_columns[1]
train_cat_cols = stat_features_columns[2]
test_cat_cols = stat_features_columns[3]


# In[ ]:


i = 1
plt.figure(figsize = (15,50))

for col in train_num_cols:
    plt.subplot(10,4,i)
    sns.regplot(x = train_df[col],y=train_df['SalePrice'],color=".1")
    i=i+1
    stp = stats.pearsonr(train_df[col], train_df['SalePrice'])
    str_title = "r = " + "{0:.3f}".format(stp[0]) + "      " "p = " + "{0:.3f}".format(stp[1])
    plt.title(str_title,fontsize=11)
    plt.xticks(rotation='45',size=6)

    plt.tight_layout() 
plt.show()


# In[ ]:


stat_features_columns=check_features()
train_num_cols = stat_features_columns[0]
test_num_cols = stat_features_columns[1]
train_cat_cols = stat_features_columns[2]
test_cat_cols = stat_features_columns[3]

print("train_num_cols",train_num_cols)
print("\n test_num_cols",test_num_cols)
print("\n train_cat_cols",train_cat_cols)
print("\n test_cat_cols",test_cat_cols)


# In[ ]:


collist=[]
lists=[]
for col in train_num_cols: 
    temp = train_df[col].astype(int)
    r,p = stats.pearsonr(temp, train_df['SalePrice'])
    if int(round(p, 2)*100)>5:
        lists.append((col,round(p, 2), int(round(p, 2)*100)))
        collist.append(col)
df=pd.DataFrame(lists,columns=[col,'p value','p %']).sort_values('p %',ascending=False)
df


# In[ ]:


corr_matrix_with_SalePrice = train_df.corr()['SalePrice']

plt.figure(figsize=(10,8))
corr_matrix_with_SalePrice.sort_values(axis=0,ascending=True).plot(kind='barh')
plt.xlabel('Correlation against SalePrice',color='b')


# In[ ]:


corr_matrix_with_SalePrice=corr_matrix_with_SalePrice.where(corr_matrix_with_SalePrice>0.50).dropna()
corr_matrix_with_SalePrice=corr_matrix_with_SalePrice.sort_values(ascending=True)
corr_matrix_with_SalePrice


# In[ ]:


high_corr_df = corr_matrix_with_SalePrice.to_frame()
high_corr_features = high_corr_df.index
print(high_corr_features)


# **From the above figures:**
# 
# Below columns have less than 0.05 P-value and less correlation against SalePrice (target variable). those can be dropped due to strong null hypotheisis or these feature are not significant to target output or create noise to the data set.
# 
# * BsmtFinSF2 
# * BsmtHalfBath
# * MiscVal
# * Id
# * LowQualFinSF
# * 3SsnPorch
# 
# 

# In[ ]:


stat_features_columns=check_features()
train_num_cols = stat_features_columns[0]
test_num_cols = stat_features_columns[1]
train_cat_cols = stat_features_columns[2]
test_cat_cols = stat_features_columns[3]

print("train_num_cols",train_num_cols)
print("\n test_num_cols",test_num_cols)
print("\n train_cat_cols",train_cat_cols)
print("\n test_cat_cols",test_cat_cols)


# In[ ]:


for col in collist:
    if col == 'Id':
        pass
        # Id will be dropped before modelling
    else:
        train_df.drop([col],axis=1,inplace=True)
        test_df.drop([col],axis=1,inplace=True)


# ## Outliers

# In[ ]:


high_corr_features


# In[ ]:


sns.pairplot(train_df[high_corr_features],size = 2.5)
plt.show()


# #### as per above pair plot, outliers are
# 
# * 1stFlrSF above 3000 
# * GrLivArea above 4000 
# * SalePrice above 400000 
# 
# Removing the above outliers

# ## Feature Extraction

# In[ ]:


def feature_engg(df):
    
    df['total_area'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + df['GrLivArea']
    df['porch_area'] = df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch'] + df['ScreenPorch']
    df['total_bathroom'] = df['BsmtFullBath'] + df['FullBath'] + (df['HalfBath']*0.5)
    
    
    df['has_pool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    df['has_2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    df['has_garage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    df['has_bsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    df['has_fireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    
    print("Before dropping",df.shape)
    drop_list = ['TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea',                 'WoodDeckSF','OpenPorchSF','EnclosedPorch','ScreenPorch',                 'BsmtFullBath','FullBath','HalfBath',                 'PoolArea','2ndFlrSF','GarageArea','TotalBsmtSF','Fireplaces']
    df=df.drop(drop_list,axis=1)
    print("After dropping",df.shape)
    
    return df

train_df = feature_engg(train_df)
test_df = feature_engg(test_df)


# ## Distribution of sale prices

# In[ ]:


# Distribution of sale prices

f, ax = plt.subplots(2,2,figsize=(12, 8))

sns.distplot(train_df['SalePrice'],fit=norm, color='b',ax=ax[0][0])
sns.distplot(np.log1p(train_df['SalePrice']),color='b',ax=ax[0][1])
stats.probplot(train_df['SalePrice'],plot=ax[1][0])
stats.probplot(np.log1p(train_df['SalePrice']),plot=ax[1][1])
plt.xlabel('SalePrice')
plt.ylabel('Number of occurances')
plt.tight_layout()


# > fig 0,0 diagram shows the salary distribution is positive skewness
# 
# > fig 0,1 diagram shows the logarithmic of sales in normal distribution
# 
# > fig 1,1 diagram shows the sales in probability chart highly deviates from the line
# 
# > fig 1,1 diagram shows the logarithmic of sales are fit to nor

# ### Handling the skewness
# > Normal distribution target will have good result
# 

# In[ ]:


train_df['SalePrice']= np.log1p(train_df["SalePrice"])


# ## >>>>>>>>>>>>>>>>>>>>>>>> Feature engineering completed <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# # 4. Model Data Preparation

# In[ ]:


#importing the necessary libraries
# from sklearn.preprocessing import LabelEncoder

# #getting categorical variables
# cat_feat = list(train_df.dtypes[train_df.dtypes == 'object'].index)

# #Encoding the categorical variables
# for c in cat_feat:
#     lbl = LabelEncoder() 
#     lbl.fit(list(train_df[c].values)) 
#     train_df[c] = lbl.transform(list(train_df[c].values))
# print(train_df.shape)
# train_df.describe()


# In[ ]:


#Concat and splitting so we get match columns after get_dummies
train_test_df = pd.concat([train_df,test_df])
train_test_df


# In[ ]:


# Get dummy variables to all catagorical columns
print("shape of train_test_df before cat to dummy variable",train_test_df.shape)

train_test_df = pd.get_dummies(train_test_df,drop_first=True)

print("shape of train_test_df before cat to dummy variable",train_test_df.shape)

train_test_df.head()


# Instead of using Label and One Hot encoder, pd.get_dummies will do the same job

# In[ ]:


train_df = train_test_df[train_test_df['Id']<1461]
test_df = train_test_df[train_test_df['Id']>1460]


# In[ ]:


X = train_test_df[train_test_df['Id']<1461]
y = X["SalePrice"]
X = X.drop(["SalePrice",'Id'],axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## Feature scalling

# In[ ]:


# def feature_scale(df):
#     sc = StandardScaler()
#     return sc.fit_transform(df)
# X_train[X_train.columns] = feature_scale(X_train[X_train.columns])


# In[ ]:


# #importing minmax scaler from sklearn.preprocessing and scaling the training dataframe
# from sklearn.preprocessing import MinMaxScaler
# def feature_scale_minmax(df):
#     sc_minmax = MinMaxScaler()
#     return sc_minmax.fit_transform(df)
# X_train[X_train.columns] = feature_scale_minmax(X_train[X_train.columns])
# X_train.describe()


# In[ ]:


# X_test[X_test.columns] = feature_scale_minmax(X_test[X_test.columns])
# X_test.describe()


# ########################### Data preparation completed #################################

# # 5. Modelling

# # 5.1 Scikit Linear regression

# In[ ]:


sk_linear_reg = LinearRegression()

sk_linear_reg.fit(X = X_train,y=y_train)


# In[ ]:


y_pred_sk_Linear = sk_linear_reg.predict(X_test)
print(len(y_pred_sk_Linear))


# In[ ]:


for y in y_pred_sk_Linear:
    print(np.expm1(y))


# In[ ]:


print("Scikit learn linear regression - Goodness score - R**2 value = ", sk_linear_reg.score(X_train,y_train))


# In[ ]:


test_pred_table_linear={"y_test":list(np.expm1(y_test)),"y_pred":list(np.expm1(y_pred_sk_Linear))}

test_pred_table_linear = pd.DataFrame.from_dict(test_pred_table_linear)
test_pred_table_linear['difference']=test_pred_table_linear['y_test']-test_pred_table_linear['y_pred']
test_pred_table_linear['error_percent']=abs(test_pred_table_linear['difference']/test_pred_table_linear['y_test'])*100

print("Error percentage mean from test data",test_pred_table_linear['error_percent'].mean())
test_pred_table_linear.describe()


# In[ ]:


test_pred_table_linear.sort_values('error_percent')


# # 5.2 Scikit Decision tree algorithim

# In[ ]:


sk_tree = tree.DecisionTreeRegressor()

sk_tree.fit(X_train,y_train)

print("Scikit learn decision tree - Goodness score - R**2 value = ", sk_tree.score(X_train,y_train))


# In[ ]:


y_pred_sk_tree = sk_tree.predict(X_test)
print(len(y_pred_sk_tree))


# In[ ]:


test_pred_table_tree={"y_test":list(np.expm1(y_test)),"y_pred":list(np.expm1(y_pred_sk_tree))}

test_pred_table_tree = pd.DataFrame.from_dict(test_pred_table_tree)
test_pred_table_tree['difference']=test_pred_table_tree['y_test']-test_pred_table_tree['y_pred']
test_pred_table_tree['error_percent']=abs(test_pred_table_tree['difference']/test_pred_table_tree['y_test'])*100

print("Error percentage mean from test data",test_pred_table_tree['error_percent'].mean())


# In[ ]:


test_pred_table_tree.describe()


# In[ ]:


test_pred_table_tree.sort_values('error_percent')


# # 5.3 K Nearest Neighbors(KNN)

# In[ ]:


sk_knn = tree.DecisionTreeRegressor()

sk_knn.fit(X_train,y_train)

print("Scikit learn decision tree - Goodness score - R**2 value = ", sk_knn.score(X_train,y_train))


# In[ ]:


y_pred_sk_knn = sk_knn.predict(X_test)
print(len(y_pred_sk_knn))


# In[ ]:


test_pred_table_knn={"y_test":list(np.expm1(y_test)),"y_pred":list(np.expm1(y_pred_sk_knn))}

test_pred_table_knn = pd.DataFrame.from_dict(test_pred_table_knn)
test_pred_table_knn['difference']=test_pred_table_knn['y_test']-test_pred_table_knn['y_pred']
test_pred_table_knn['error_percent']=abs(test_pred_table_knn['difference']/test_pred_table_knn['y_test'])*100

print("Error percentage mean from test data",test_pred_table_knn['error_percent'].mean())


# In[ ]:


test_pred_table_knn.describe()


# In[ ]:


test_pred_table_knn.sort_values('error_percent')


# In[ ]:




