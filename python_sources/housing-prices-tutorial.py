#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Data Viz
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Modelling
from xgboost import XGBRegressor
import lightgbm as lgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Reading Data

# In[3]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')


# Let's take a quick look at the dataset

# In[4]:


train.head(3)


# In[5]:


print(f'Shape of Train Set: {train.shape}')
print(f'Shape of Test Set : {test.shape}')


# ## EDA

# #### Plotting Data Types

# The function *plot_dtypes* identifies the number of distinct data types in our dataset.

# In[6]:


def plot_dtypes(df):
    # Get dataframe with count of each dtype
    dtypes = pd.DataFrame(df.dtypes, columns = ['dtype'])
    dtypes = dtypes.groupby('dtype').size().rename('count').reset_index()
    
    # Plot resulting dataframe
    ax = dtypes.sort_values('count').plot(
        'dtype',
        kind = 'barh', 
        figsize=(8,6), 
        title='Number of Distinct Data Types in Dataset', 
        legend = None, 
        color = '#c19a6b'
        )
    
    # List that will hold plt.patches data
    totals = []

    # Append patch values to list
    for i in ax.patches:
        totals.append(i.get_width())

    # Denominator for percentage
    total = sum(totals)

    # Set individual bar lables
    for i in ax.patches:
        # get_width to move left or right; get_y to move up or down
        # For regular bar, switch get_width and get_y 
        # Change get_width to get_height, get_y to get_x
        ax.text(
            i.get_width() + .3, i.get_y() + .2,
            str(round((i.get_width() / total)*100, 2))+'%', 
            fontsize=12,
            color='blue'
        )
    
    plt.style.use('ggplot')
    plt.show()


# In[7]:


plot_dtypes(train)


# #### Plotting features with missing values

# The function *plot_missing* identifies the percentage of missing rows for every feature in our dataset.

# In[8]:


def plot_missing(df, showplot = True):
    
    # Get dataframe with percentage of missing values
    missing = pd.DataFrame(df.isnull().sum(), columns = ['perc_missing'])
    missing = missing.loc[missing['perc_missing'] > 0]
    missing = (missing/len(df))*100
    
    # Plot resulting dataframe
    missing = missing.sort_values('perc_missing')
    if(showplot):
        ax = missing.plot(
            kind = 'barh',
            figsize=(10,8),
            title = 'Percentage of Missing Values in Dataset by Feature',
            legend = None,
            color = 'coral'
        )

        # Set individual bar lables
        for i in ax.patches:
            # get_width to move left or right; get_y to move up or down
            ax.text(
                i.get_width()+.3, i.get_y(),
                str(round(i.get_width(), 2)), 
                fontsize=12,
                color='blue'
            )

        plt.style.use('ggplot')
        plt.show()
    
    return missing


# In[9]:


missing_train = plot_missing(train)
missing_test = plot_missing(test)


# The distribution of missing values in train set and test set are quite similar. We will deal with these missing values in the data cleaning section.

# #### Plotting features and their interactions

# The funciton *plot_categorical_column* allows us to observe a distribution for the categorical variable and its interactions with the target variable and other subcolumns.

# In[10]:


def plot_categorical_column(df, col, target, size, *subcols):
    num_subcols = len(subcols) 
    # Two charts per subcolumn
    rows_subcols = num_subcols*2
    # Two charts for required categorical feature
    plots_total = 2 + rows_subcols
    
    fig_shape = (plots_total, 1)
    # Height should be a factor of total plots
    fig = plt.figure(figsize=(18, 4*plots_total*size))
    plt.subplots_adjust(hspace=0.5)
    
    # First plot for required categorical feature
    # Plot distribution
    ax1 = plt.subplot2grid(fig_shape, (0,0))
    df[col].value_counts(dropna = False, sort = True).plot(
        kind = 'barh', 
        ax = ax1, 
        title = 'Categorical Count for ' + col
    )
    
    # Second plot for required categorical feature
    # Plot median of target variable for each unique value in main categorical feature
    ax2 = plt.subplot2grid(fig_shape, (1,0))
    df[[col, target]].groupby(col).median().plot(
        kind = 'barh', 
        ax = ax2, 
        title = 'Median of ' + target + ' with respect to ' + col, 
        legend = None
    )
    
    # Generate two plots for subcolumns
    for i, subcol in zip(range(num_subcols), subcols):
        
        # First subcolumn plot
        ax = plt.subplot2grid(fig_shape, (2*i + 2,0))
        # If subcolumn has null values, plot interaction of null values
        if df[subcol].isnull().sum():
            df[[col, subcol]].loc[df[subcol].isnull() == True].groupby(col).size().plot(
                kind = 'barh', 
                ax = ax, 
                title = 'Amount of Nulls in ' + subcol + ' with respect to ' + col
            )
        else:
            if df[subcol].dtype != 'O':
                # If no null values, and a number, plot histogram
                df[subcol].plot(
                    kind = 'hist', 
                    bins = 100,
                    ax = ax, 
                    title = 'Histogram for ' + subcol, 
                    color = 'orange'
                )
            else:
                # If no null values, and an object (categorical feature), plot distribution
                df[subcol].value_counts().plot(
                    kind = 'barh', 
                    ax = ax, 
                    title = 'Categorical counts for ' + subcol
                )
                
        # Second subcolumn plot 
        ax = plt.subplot2grid(fig_shape, (2*i + 3,0))
        # If numeric field, plot median for each unique value in main categorical feature
        if df[subcol].dtype != 'O':
            df[[col, subcol]].groupby(col).median().plot(
                kind = 'barh', 
                ax = ax, 
                title = 'Median Value of ' + subcol + ' with respect to ' + col, 
                legend = None
            )
        else:
            # If categorical field, plot stacked interactions of both categories
            df.groupby([col, subcol]).size().unstack().plot(
                kind = 'barh', 
                ax = ax, 
                title = 'Stacked category interactions for ' + col + ' ' + subcol, 
                cmap = plt.get_cmap('tab20')
            )
        
    plt.style.use('ggplot')
    plt.show()


# In[11]:


plot_categorical_column(train, 'MSZoning', 'SalePrice', 1, 'LotFrontage', 'LotArea')


# In[12]:


plot_categorical_column(train, 'Neighborhood', 'SalePrice', 2, 'LotFrontage')


# #### Correlation Matrix

# In[13]:


fig = plt.figure(figsize = (24, 12))

corr = train.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(
    corr, 
    mask = mask, 
    cmap = 'PiYG', 
    annot = True, 
    fmt=".2f")

plt.yticks(rotation=0) 
plt.xticks(rotation=90)
plt.title('Correlation Matrix for Train Data', fontsize = 15)
plt.show()


# ## Data Cleaning

# Preserve original train and test dataframe

# In[14]:


new_train = train.copy()
new_test = test.copy()

new_train.drop('Id', axis = 1, inplace = True)
new_test.drop('Id', axis = 1, inplace = True)


# ### Handle Missing Values

# #### Lot Frontage

# In our EDA section, we found the relationship between *Neighborhood* and *LotFrontage*. To handle missing values, we can use the median *LotFrontage* for each neighborhood. To dig into this idea a little more, let's draw a boxplot for these two features.

# In[15]:


fig = plt.figure(figsize = (24, 9))
sns.boxplot(
    x = new_train['Neighborhood'],
    y = new_train['LotFrontage'],
)
plt.show()


# In[16]:


lotfrontage_by_neighborhood = new_train.groupby(['Neighborhood'])['LotFrontage'].agg({
    'median': np.median,
    'count': np.size,
    'std': np.std,
}).reset_index()
lotfrontage_by_neighborhood.T


# In[17]:


def fill_lot_frontage(row):
    return lotfrontage_by_neighborhood[lotfrontage_by_neighborhood['Neighborhood'] == row['Neighborhood']]['median'].values[0]


# In[18]:


new_train['LotFrontage'] = new_train.apply(lambda row: fill_lot_frontage(row) if np.isnan(row['LotFrontage']) else row['LotFrontage'], axis=1)
new_test['LotFrontage'] = new_test.apply(lambda row: fill_lot_frontage(row) if np.isnan(row['LotFrontage']) else row['LotFrontage'], axis=1)


# In[19]:


fig = plt.figure(figsize = (8, 6))
plt.subplots_adjust(hspace=0.5)

fig_shape = (2,2)

ax1 = plt.subplot2grid(fig_shape, (0,0))
train.LotFrontage.plot(
    kind = 'hist',
    ax = ax1, 
    bins = 40,
    title = 'Original Distribution: Train'
)


ax1 = plt.subplot2grid(fig_shape, (0,1))
new_train.LotFrontage.plot(
    kind = 'hist', 
    ax = ax1, 
    bins = 40,
    title = 'Without missing values: Train'
)

ax1 = plt.subplot2grid(fig_shape, (1,0))
test.LotFrontage.plot(
    kind = 'hist',
    ax = ax1, 
    bins = 40,
    title = 'Original Distribution: Test'
)


ax1 = plt.subplot2grid(fig_shape, (1,1))
new_test.LotFrontage.plot(
    kind = 'hist', 
    ax = ax1, 
    bins = 40,
    title = 'Without missing values: Test'
)


plt.show()


# #### PoolQC

# In[20]:


new_train["PoolQC"] = new_train["PoolQC"].fillna("None")
new_test["PoolQC"] = new_test["PoolQC"].fillna("None")


# #### MiscFeature

# In[21]:


new_train["MiscFeature"] = new_train["MiscFeature"].fillna("None")
new_test["MiscFeature"] = new_test["MiscFeature"].fillna("None")


# #### Alley

# In[22]:


new_train["Alley"] = new_train["Alley"].fillna("None")
new_test["Alley"] = new_test["Alley"].fillna("None")


# #### Fence

# In[23]:


new_train["Fence"] = new_train["Fence"].fillna("None")
new_test["Fence"] = new_test["Fence"].fillna("None")


# #### FireplaceQu

# In[24]:


new_train["FireplaceQu"] = new_train["FireplaceQu"].fillna("None")
new_test["FireplaceQu"] = new_test["FireplaceQu"].fillna("None")


# #### Garage Features

# In[25]:


# Categorical
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    new_train[col] = new_train[col].fillna('None')
    new_test[col] = new_test[col].fillna('None')
    
# Numeric
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    new_train[col] = new_train[col].fillna(0)
    new_test[col] = new_test[col].fillna(0)


# #### Basement Features

# In[26]:


# Categorical
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    new_train[col] = new_train[col].fillna('None')
    new_test[col] = new_test[col].fillna('None')
    
# Numeric
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    new_train[col] = new_train[col].fillna(0)
    new_test[col] = new_test[col].fillna(0)


# #### Masonry veneer Features

# In[27]:


# Categorical
new_train["MasVnrType"] = new_train["MasVnrType"].fillna("None")
new_test["MasVnrType"] = new_test["MasVnrType"].fillna("None")

#Numeric
new_train["MasVnrArea"] = new_train["MasVnrArea"].fillna(0)
new_test["MasVnrArea"] = new_test["MasVnrArea"].fillna(0)


# #### MSZoning

# In[28]:


new_test['MSZoning'] = new_test['MSZoning'].fillna(train['MSZoning'].mode()[0])


# #### Remaining Features

# In[29]:


missing_new_train = plot_missing(new_train, showplot = False)
missing_new_train


# In[30]:


missing_new_test = plot_missing(new_test, showplot = False)
missing_new_test


# In[31]:


cols_to_drop = []
cols_to_clean = []


# In[32]:


def handle_na(train, test, cols_to_clean, cols_to_drop):
    if(cols_to_drop):
        train.drop(cols_to_drop, axis = 1, inplace = True)
        test.drop(cols_to_drop, axis = 1, inplace = True)
    
    if(cols_to_clean):
        for col in cols_to_clean:
            train[col+'_is_na'] = train[col].isnull()
            test[col+'_is_na'] = test[col].isnull()
        
    for col in test.columns:
        # If numeric, fill with median
        if np.issubdtype(train[col].dtype, np.number):
            train.loc[train[col].isnull() == True, col] = train[col].median()
            test.loc[test[col].isnull() == True, col] = train[col].median()

        # If object, fill with mode
        if (train[col].dtype == 'O'):
            train.loc[train[col].isnull() == True, col] = train[col].mode().iloc[0]
            test.loc[test[col].isnull() == True, col] = train[col].mode().iloc[0]
            
    print(f'Shape of Train Set: {train.shape}')
    print(f'Shape of Test Set : {test.shape}')

    return train, test


# In[33]:


new_train, new_test = handle_na(new_train, new_test, cols_to_clean, cols_to_drop)


# ### Label Encoding

# In[34]:


def label_encode(train, test):
    combined = train.append(test, sort = False)
    for col in test.columns:
        if (train[col].dtype == 'O'):
            le = LabelEncoder()
            le.fit(combined[col])
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])
            
    return train, test


# In[35]:


new_train, new_test = label_encode(new_train, new_test)


# ## Feature Engineering

# #### Total Square Footage

# In[36]:


new_train['TotalSF'] = new_train['TotalBsmtSF'] + new_train['1stFlrSF'] + new_train['2ndFlrSF']
new_test['TotalSF'] = new_test['TotalBsmtSF'] + new_test['1stFlrSF'] + new_test['2ndFlrSF']


# ## Training

# ### Generating Validation Set

# In[37]:


new_train['SalePrice'] = np.log(new_train['SalePrice'])


# In[38]:


def split(df, target, test_size = 0.1):
    X_train, X_valid, y_train, y_valid = train_test_split(df.drop(target, axis = 1), df[target], test_size = test_size, random_state = 42)
    print(f'Shape of Train Set: {X_train.shape}')
    print(f'Shape of Valid Set: {X_valid.shape}')
    return X_train, X_valid, y_train, y_valid


# In[39]:


X_train, X_valid, y_train, y_valid = split(new_train, 'SalePrice')


# ### XGBoost

# In[171]:


def xgtrain(X_train, X_valid, y_train, y_valid):
    regressor = XGBRegressor(
        n_estimators = 50000, 
        learning_rate = 0.01,
        max_depth = 3, 
        subsample = 0.5, 
        colsample_bytree = 0.2
        )
    
    regressor_ = regressor.fit(
        X_train.values, y_train.values, 
        eval_metric = 'rmse', 
        eval_set = [
            (X_train.values, y_train.values), 
            (X_valid.values, y_valid.values)
        ],
        verbose = 1000,
        early_stopping_rounds = 500,
        )
    
    return regressor_


# In[172]:


get_ipython().run_cell_magic('time', '', 'regressor_ = xgtrain(X_train, X_valid, y_train, y_valid)')


# ### LightGBM

# In[205]:


def lgbtrain(X_train, y_train, X_valid, y_valid):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2_root'},
        'max_depth': 3,
        'learning_rate': 0.01,
        'feature_fraction': 0.2,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'num_leaves': 5
    }
    
    gbm = lgb.train(
        params,
        lgb_train,
        valid_sets = lgb_eval,
        num_boost_round=50000,
        early_stopping_rounds=100,
        verbose_eval = 250
    )
    
    return gbm


# In[206]:


get_ipython().run_cell_magic('time', '', 'gbm = lgbtrain(X_train, y_train, X_valid, y_valid)')


# ### Feature Importances

# In[ ]:


def plot_feature_importance():
    df_fi = pd.DataFrame(columns = ['features', 'importance'])
    df_fi['features'] = new_test.columns 
    df_fi['importance'] = regressor_.feature_importances_
    
    ax = df_fi.sort_values('importance').plot(
        'features',
        kind = 'barh',
        legend = None,
        title = 'Feature Importances',
        figsize = (18, 25),
        color = 'crimson'
       )
    
    
    # List that will hold plt.patches data
    totals = []

    # Append patch values to list
    for i in ax.patches:
        totals.append(i.get_width())
    
    # Set individual bar lables
    for i in ax.patches:
        # get_width to move left or right; get_y to move up or down
        ax.text(
            i.get_width() + 0.001, i.get_y(),
            str(round(i.get_width(), 3)), 
            fontsize=14,
            color='blue'
        )
    
    plt.style.use('ggplot')
    plt.yticks(fontsize=14)
    plt.show()


# In[ ]:


plot_feature_importance()


# ## Prepare Submission

# In[209]:


def prepare_submission(comment = 'latest'):
    xgb_preds = np.exp(regressor_.predict(new_test.values))
    lgb_preds = np.expm1(gbm.predict(new_test.values))
    submission['SalePrice'] = 0.75*xgb_preds + 0.25*lgb_preds
    submission.to_csv('submission' + comment + '.csv', index=False)
    print(submission.head())


# In[210]:


prepare_submission('_ensemble')


# In[164]:


from IPython.display import FileLinks
FileLinks('.')


# In[ ]:


new_train.to_feather('new_train')
new_test.to_feather('new_test')


# In[ ]:


# train[['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'YearBuilt']]


# In[ ]:


train['GarageYrBlt'].plot(kind = 'hist')
plt.show()


# In[ ]:


train['YearBuilt'].plot(kind = 'hist')
plt.show()


# In[ ]:


(train['GarageYrBlt'] - train['YearBuilt']).plot(kind = 'hist')
plt.show()

