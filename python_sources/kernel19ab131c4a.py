#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, special


# In[5]:


pd.set_option('display.max_columns', 300)

#import cleaned data into df
train_df = pd.read_csv('../input/train.csv')        #train df

test_df = pd.read_csv('../input/test.csv')     #test df

test_id = test_df.Id     #id's of the test data


# In[6]:


#Global variables

dependent_var = 'SalePrice'     #Dependent variable we are trying to predict
correlation_threshold = 0.95    #Variable pairs that are correlated with each other for over 
                                #threshold one will be dropped
corr_with_salePrice = 0.05      #Variable that have correllation with dependent_var of range(-1, 1) 


# ### Discussion by the team:
# This train and test is one dataset spplit into two.
# Decision: Combine it and clean it as one and then split later"

# In[7]:


df = pd.concat([train_df, test_df], sort = True).sort_values(by = 'Id')

df.set_index('Id', inplace = True, verify_integrity = True)


# In[8]:


df.head()


# ### From the data_description.txt it can be seen that MSSubClass, OverallQual, OverallCond are ordinal type. We change them to object type

# In[9]:


for col in ['MSSubClass', 'OverallQual', 'OverallCond']:
    df[col] = df[col].astype('O')


# ##### The method below takes a dataframe and returns a tuple of two lists
# List1: categorical columns that contains NaN values
# List2: numerical columns that contains NaN values

# In[10]:


#get a list of columns with missing values

def get_nan_columns(dataframe):
    '''
    Get two lists of:
        categorical column
        numerical column names of the given df
        params:
            dataframe: Pandas.DataFrame
        returns:
            a tuple of lists
            (categorical, Numerical)
'''
    cat_c = []
    num_c = []
    for col in dataframe.columns.values.tolist():
        if dataframe[col].isnull().any():
            if dataframe[col].dtype == 'O':
                cat_c.append(col)
            else:
                if col != dependent_var:
                    num_c.append(col)
                
    return cat_c, num_c


# In[11]:


get_nan_columns(df)


# #### Handle all the NaN values from the above cell

# In[12]:


#Only one kitchenQual of null. Impute with mode

df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])                                               


# In[13]:


#Garage NaN values when GarageArea = 0.0. This means the house does not have a garage.
#categorical type, NG = No Garage
#numerical types, 0.0

for col in ['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType']:
    df[col] = np.where(df['GarageArea'] == 0.0, 'NG', df[col])

for col in ['GarageCars', 'GarageYrBlt']: 
    df[col] = np.where(df['GarageArea'] == 0.0, 0.0, df[col])


# In[14]:


#GarageArea isnan and some value. 2 rows

for col in ['GarageCond', 'GarageFinish', 'GarageQual']:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in ['GarageCars', 'GarageArea']: 
    df[col] = df[col].fillna(df[col].mean()) 
    
#for garageArea take the year of the house
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])


# In[15]:


#Basement values when TotalBsmtSF = 0.0
#NB = No Basement
for col in ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']:
    df[col] = np.where(df['TotalBsmtSF'] == 0.0, 'NB', df[col])

for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF']:
    df[col] = np.where(df['TotalBsmtSF'] == 0.0, 0.0, df[col])


# In[16]:


#Only one value of TotalBasementArea is zero and all other categories are zero
for col in ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']:
    df[col] = np.where(df['TotalBsmtSF'].isna(), 'NB', df[col])

for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF']:
    df[col] = np.where(df['TotalBsmtSF'].isna(), 0.0, df[col])


# In[17]:


#fill the rest with mode or mean depending on the category

for col in ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'TotalBsmtSF']:
    df[col] = df[col].fillna(df[col].mean())


# In[18]:


df['PoolQC'] = np.where(df['PoolArea'] == 0.0, 'None', df['PoolQC'])
df['FireplaceQu'] = np.where(df['Fireplaces'] == 0, 'None', df['FireplaceQu'])


# In[19]:


df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
df['Alley'] = df['Alley'].fillna('None')
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
df['Functional'] = df['Functional'].fillna('Typ')
df['MiscFeature'] = df['MiscFeature'].fillna('None')


# In[20]:


for col in ['Exterior1st', 'Exterior2nd', 'SaleType']:
    df[col] = df[col].fillna('Other')


# In[21]:


for col in ['MSZoning', 'Utilities', 'Fence']:
    mode =  df[col].mode()[0]
    df[col] = df.groupby('Neighborhood')[col].apply(lambda x: x.fillna(x.value_counts().idxmax() if x.value_counts().max() >=1 else mode , inplace = False))
    df[col]= df[col].fillna(df[col].value_counts().idxmax())


# In[22]:


# 23 out rows have both MasVnrType and MasVnrArea as None.
# Conclusion: MasVnrType = None, MasVnrArea = 0
# IF MasVnrArea = 0.0, MasVnrType = 'None'

df['MasVnrArea'] = df['MasVnrArea'].fillna(0.0)
df['MasVnrType'] = df['MasVnrType'].fillna('None')


# In[23]:


df['PoolQC'] = df['PoolQC'].fillna(df['PoolArea'].mean())


# In[24]:


#df = df[~((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000))]


# ### Get all the numerical columns with high skewness and kurtosis

# In[25]:


#1st get the log1p of all numerical columns

#df[dependent_var] = df[dependent_var].apply(np.log1p)
numerical_columns = df.select_dtypes(exclude=['O']).columns.values.tolist()
df[numerical_columns] = df[numerical_columns].apply(np.log1p)


# In[26]:


def get_skewed_cols(dataframe):
    sk_cols = []
    for col in df.select_dtypes(exclude=['O']).columns.values.tolist():
        if col != dependent_var:
            if dataframe[col].dtype != 'O':
                if abs(dataframe[col].skew()) > 0.1:
                    sk_cols.append(col)
    return sk_cols


# # plot the skew graph here

# In[27]:


#transform skewed columns

df[get_skewed_cols(df)] = special.boxcox1p(df[get_skewed_cols(df)], 1e-4)


# # plot the skew graph here

# In[28]:


df.head()


# # Kurtosis

# In[29]:


def get_high_kurtosis_cols(dataframe):
    kurt_cols = []
    for col in df.select_dtypes(exclude=['O']).columns.values.tolist():
        if col != dependent_var:
            if dataframe[col].dtype != 'O':
                if abs(dataframe[col].kurt()) > 3:
                    kurt_cols.append(col)
    return kurt_cols


# In[30]:


get_high_kurtosis_cols(train_df)


# In[31]:


df.shape


# In[32]:


#get categorical columns and encode them
categorical_columns = df.select_dtypes(include=['O']).columns.values.tolist()

df = pd.get_dummies(df, columns = categorical_columns, drop_first = True)


# In[33]:


df.head()


# ### Separate the dfs and drop SalePrice

# In[34]:


train_df = df.iloc[:len(train_df)]
test_df = df.iloc[len(train_df):].drop(dependent_var, axis = 1)


# ## Data is fairly clean now to start with the analysis
# #### Get a list of columns that correlate with SalePrice by 10% on both ends

# In[35]:


#get pairs of highly correlated variables.
def get_correlated_pairs(dataframe):
    
    correlations = dataframe.corr()
    pairsList = []
    for col in correlations.columns.values.tolist():
        corr_series = correlations.loc[:, col]
    
        for row in corr_series.index.tolist():
            if (abs(corr_series.loc[row]) > correlation_threshold and row != col and abs(corr_series.loc[row]) != 1):
                if sorted([col, row]) not in pairsList: 
                    pairsList.append(sorted([col, row]))
    return pairsList


# In[36]:


get_correlated_pairs(train_df)


# # train_df.corr() HEATMAP here

# In[37]:


#Most frequent columns turns out to be GarageYrBlt, GarageArea, TotalBsmtSF

train_df = train_df.drop(['GarageYrBlt', 'GarageArea', 'TotalBsmtSF', 
                          'Exterior1st_CemntBd', 'Exterior1st_MetalSd', 
                          'Exterior1st_VinylSd'], axis = 1)
test_df = test_df.drop(['GarageYrBlt', 'GarageArea', 'TotalBsmtSF', 
                          'Exterior1st_CemntBd', 'Exterior1st_MetalSd', 
                          'Exterior1st_VinylSd'], axis = 1)


# In[38]:


#get correlation for each col with SalePrice in train_df

def get_corr_with_SalePrice(dataframe):
    
    cor = pd.DataFrame(dataframe.corr()[dependent_var])
    
    return cor[abs(cor.SalePrice) > corr_with_salePrice].index.values.tolist()


# In[39]:


train_df.shape


# In[40]:


test_df.shape


# In[41]:


droplist = get_corr_with_SalePrice(train_df)
train_df = train_df[droplist]
droplist.remove(dependent_var)
test_df = test_df[droplist]


# In[42]:


test_df.shape


# In[43]:


train_df.shape


# ### From list above, get pairs of columns that are highly correlated with each other

# In[44]:


#get pairs of highly correlated variables.
def get_correlated_pairs(dataframe):
    
    correlations = dataframe.corr()
    pairsList = []
    for col in correlations.columns.values.tolist():
        corr_series = correlations.loc[:, col]
    
        for row in corr_series.index.tolist():
            if (abs(corr_series.loc[row]) > correlation_threshold and row != col):
                if sorted([col, row]) not in pairsList: 
                    pairsList.append(sorted([col, row]))
    return pairsList


# ### From the pairs we found, determine which column has the least correlation with SalePrice and drop it

# In[45]:


#get columns that have less correlation with SalePrice from the pairs above

def get_less_correlated(dataframe):
    
    less_correlated = []

    for pair in get_correlated_pairs(dataframe):
    
        if dataframe[pair[0]].corr(dataframe[dependent_var]) > dataframe[pair[1]].corr(dataframe[dependent_var]):
            if pair[1] not in less_correlated:
                less_correlated.append(pair[1])
        else:
            if pair[0] not in less_correlated:
                less_correlated.append(pair[0])
    return less_correlated


# ## Drop less correlated columns from both train_df and test_df (Still need to look at this)

# In[46]:


#Outliers need to figure out how they were identified first, else we do not include them
#train_df = train_df.drop(train_df.index[[30, 88, 462, 631, 1322]])


# In[47]:


train_df.shape


# In[48]:


test_df.shape


# ### Get X and y

# In[49]:


y = train_df[dependent_var]
X = train_df.drop([dependent_var], axis = 1)


# In[50]:


#Scale X and test_df

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
test_df_scaled = scaler.fit_transform(test_df)
test_df_scaled = pd.DataFrame(test_df_scaled, columns = test_df.columns)


# In[51]:


#split train and test samples from X, y

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2)


# In[52]:


#train with LinearRegression
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression(normalize=True, n_jobs=-1)
lr_model.fit(np.reshape(X_train, X_train.shape, order = 'F-contiguous'), y_train)


# In[53]:


#train with Lasso
from sklearn.linear_model import Lasso

l_model = Lasso(alpha = 0.000395)
l_model.fit(np.reshape(X_train, X_train.shape, order = 'F-contiguous'), y_train)


# In[54]:


#train with Ridge
from sklearn.linear_model import Ridge

r_model = Ridge(alpha = 0.0001, max_iter=1000000, normalize=True)
r_model.fit(np.reshape(X_train, X_train.shape, order = 'F-contiguous'), y_train)


# In[55]:


#train with RandomForest
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_jobs=-1, n_estimators=100)
rf_model.fit(np.reshape(X_train, X_train.shape, order = 'F-contiguous'), y_train)


# In[56]:


#train with GradiientBoosting
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor()
gb_model.fit(np.reshape(X_train, X_train.shape, order = 'F-contiguous'), y_train)


# In[57]:


from xgboost import XGBRegressor

xgb_model = XGBRegressor()
xgb_model.fit(np.reshape(X_train, X_train.shape, order = 'F-contiguous'), y_train)


# In[58]:


from lightgbm import LGBMRegressor

lgbm_model = LGBMRegressor()
lgbm_model.fit(np.reshape(X_train, X_train.shape, order = 'F-contiguous'), y_train)


# In[59]:


from mlxtend.regressor import StackingCVRegressor

stack_regr = StackingCVRegressor(regressors = (l_model,
                                               r_model,
                                               gb_model,
                                               rf_model
                                               ), meta_regressor=gb_model)

stack_regr.fit(np.array(X_train), np.array(y_train))


# In[60]:


from sklearn.metrics import mean_squared_log_error

#print('Linear Model: ', np.sqrt(mean_squared_log_error(y_test, 
 #                              lr_model.predict(X_test))))

print('Lasso: ', np.sqrt(mean_squared_log_error(y_test,
                         l_model.predict(X_test))))

print('Ridge: ', np.sqrt(mean_squared_log_error(y_test,
                         r_model.predict(X_test))))

print('RandomForest: ', np.sqrt(mean_squared_log_error(y_test,
                                                       rf_model.predict(X_test))))

print('GradientBoosting: ', np.sqrt(mean_squared_log_error(y_test,
                                                       gb_model.predict(X_test))))

print('XGBoost: ', np.sqrt(mean_squared_log_error(y_test,
                                                       xgb_model.predict(X_test))))

print('LightGBM: ', np.sqrt(mean_squared_log_error(y_test,
                                                       lgbm_model.predict(X_test))))

print('Stack: ', np.sqrt(mean_squared_log_error(y_test,
                                                       stack_regr.predict(np.array(X_test)))))


# In[61]:


'''y_pred = np.expm1(0.2*gb_model.predict(test_df_scaled)+
                   0.15*rf_model.predict(test_df_scaled)+
                   0.2*l_model.predict(test_df_scaled)+
                   0.2*r_model.predict(test_df_scaled)+
                   0.25*stack_regr.predict(np.array(test_df_scaled)))'''
y_pred = np.expm1(l_model.predict(test_df_scaled))


# In[62]:


solution = pd.DataFrame({'Id': test_id, 'SalePrice':y_pred})


# In[63]:


solution.to_csv('finalTest.csv', index = False)


# In[ ]:




