#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[265]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')

from math import sqrt
import os
from pathlib import Path

from IPython.display import display, FileLink

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_score


# # House price predictions: simple linear model

# This notebook attempts to solve the Housing Prices predictions Kaggle competition using the simplest possible model, with lots of room to improve. A lot the ideas for feature engineering were taken from other Kaggle notebooks including: 
# 
#   * [Stacked Regressions : Top 4% on LeaderBoard](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)
#   * [A study on Regression applied to the Ames dataset](https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset)
# 
# The notebook is broken down into 4 short parts:
# 
#   1. Load dataset.
#   2. Feature engineering.
#   3. Model training.
#   4. Submission.

# ## Load dataset

# In[266]:


PATH = Path("../input/")


# In[267]:


list(PATH.iterdir())


# In[268]:


df_raw = pd.read_csv(PATH / 'train.csv')


# In[269]:


df_raw.head().transpose()


# ## Feature engineering

# ### Find and remove outliers

# The [documentation](http://https://ww2.amstat.org/publications/jse/v19n3/decock/DataDocumentation.txt) for the dataset says *"there are 5 observations that an instructor may wish to remove from the dataset"*. It also says: *"a plot of SALE PRICE versus GR LIV AREA will indicate them quickly"*, so let's do that.

# In[270]:


plt.scatter(x=df_raw['GrLivArea'], y=df_raw['SalePrice'])
plt.title('SALE PRICE vs GR LIV AREA')
plt.show()


# The author also says: *"I would recommend removing any houses with more than 4000 square feet from the data set (which eliminates these 5 unusual observations)*" so let's do that too.

# In[271]:


idx_to_drop = df_raw[(df_raw['GrLivArea']>4000)].index
df_raw.drop(idx_to_drop, inplace=True)


# In[272]:


plt.scatter(x=df_raw['GrLivArea'], y=df_raw['SalePrice'])
plt.title('SALE PRICE vs GR LIV AREA (without outliers)')
plt.show()


# ### Add features
# 
# Add the house's total square feet because it helps a bit. I tried a few other features but nothing helped much.

# In[273]:


df_raw['TotalSF'] = (
    df_raw['BsmtFinSF1'].fillna(0) +
    df_raw['BsmtFinSF2'].fillna(0) +
    df_raw['1stFlrSF'].fillna(0) +
    df_raw['2ndFlrSF'].fillna(0)
)


# In[274]:


df_raw.TotalSF.head()


# ### Extract target variable
# 
# We know the target variable is going to be SalePrice, so I'll pop that off the DataFrame. Kaggle says: *"Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price"* so I'll calculate the log of the sale price.

# In[275]:


sale_price = df_raw.pop('SalePrice')
sale_price_log = np.log(sale_price)

# We also don't need this.
house_ids = df_raw.pop('Id')


# ### Prepare columns
# 
# Continuous columns are columns with numbers that can be any size. Categorical columns have a limited number of choices. For example, `LotArea` can be any size whereas `LandSlope` has to be either `['Sev', 'Mod', 'Gtl']` (or `None`).
# 
# We'll need to process each type separately, so I've defined those by examing each column.

# In[276]:


continuous_columns = [
    'BsmtUnfSF',
    'FullBath',
    'LotFrontage',
    'BsmtFullBath',
    '3SsnPorch',
    'BedroomAbvGr',
    'LowQualFinSF',
    'BsmtFinSF1',
    'WoodDeckSF',
    'GarageArea',
    'MiscVal',
    'BsmtHalfBath',
    'HalfBath',
    'EnclosedPorch',
    'ScreenPorch',
    'TotRmsAbvGrd',
    'Fireplaces',
    'KitchenAbvGr',
    'GarageCars',
    '1stFlrSF',
    'BsmtFinSF2',
    'PoolArea',
    '2ndFlrSF',
    'TotalBsmtSF',
    'TotalSF',
    'GrLivArea',
    'LotArea',
    'OpenPorchSF',
    'MasVnrArea'
]

categorical_columns = [col for col in df_raw.columns if col not in continuous_columns]


# In[277]:


categorical_columns


# In[278]:


assert len(df_raw.columns) == len(categorical_columns + continuous_columns)


# #### Prepare categorical
# 
# Pandas has a categorical datatype that makes life rather easy when dealing with categorical columns. I'll convert all categorical columns as follows:

# In[279]:


for col_name, col in df_raw[categorical_columns].items():
    df_raw[col_name] = col.astype('category').cat.as_ordered()


# For some categories, the order is quite important like `OverallQual` (10 is best, 1 is worst). Those values are called "ordinal". I'll ensure the ordinal columns are ordered.

# In[280]:


ordinal_column_data = [
    ('ExterQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('ExterCond', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('BsmtQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('BsmtExposure', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('BsmtFinType1', ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']),
    ('BsmtFinType2', ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']),
    ('HeatingQC', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('KitchenQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('FireplaceQu', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('GarageFinish', ['Unf', 'Rfn', 'Fin']),
    ('GarageQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('GarageCond', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('PoolQC', ['Fa', 'TA', 'Gd', 'Ex']),
    ('OverallQual', list(range(1, 11))),
    ('OverallCond', list(range(1, 11))),
    ('LandSlope', ['Sev', 'Mod', 'Gtl']),  # Assume less slope is better
    ('Functional', ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']),
    ('YearBuilt', list(range(1800, 2018))),
    ('YrSold', list(range(2006, 2018))),
    ('GarageYrBlt', list(range(1900, 2018))),
    ('YearRemodAdd', list(range(1900, 2018)))
]

ordinal_columns = [o[0] for o in ordinal_column_data]

for col, categories in ordinal_column_data:
    df_raw[col].cat.set_categories(categories, ordered=True, inplace=True)


# For columns with no ordinal relationship, we'll do some special processing later.

# In[281]:


other_cat_columns = [col for col in categorical_columns if col not in ordinal_columns]


# In[282]:


assert len(categorical_columns) == len(ordinal_columns + other_cat_columns)


# #### Prepare continuous
# 
# ##### Replace missing

# The first thing we'll want to do is replace all missing values with some value. For some columns, a missing value is equivalent to 0. For others, we'll use the column's median. We can also add a column `<column_name>_is_na` that will tell the model whether the value was originally missing or not. I got this idea from the [Fast.ai library](https://github.com/fastai/fastai).
# 
# We'll also want to save the median values to be used to replace missing values in the test set.

# In[283]:


NAs = {}


# In[284]:


for col in (
    'GarageArea', 'GarageCars', 'BsmtFinSF1',
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
    'MasVnrArea'
):
    NAs[col] = 0
    df_raw[col] = df_raw[col].fillna(0)
    df_raw[f'{col}_na'] = pd.isna(df_raw[col])


# In[285]:


for col in continuous_columns:
    if not len(df_raw[df_raw[col].isna()]):
        continue
        
    median = df_raw[col].median()
        
    df_raw[f'{col}_na'] = pd.isna(df_raw[col])
    df_raw[col] = df_raw[col].fillna(median)
    
    NAs[col] = median


# ##### Unskew data

# Machine learning models generally want data to be normally distributed. We can examine the skew of our features using the skew function in Scikit-learn.

# In[286]:


skew_feats = df_raw[continuous_columns].apply(skew).sort_values(ascending=False)
skew_feats.head(10)


# For the most skewed column, let's look at the distribution.

# In[287]:


sns.distplot(df_raw[df_raw['MiscVal'] != 0]['MiscVal'])


# We can take the log of the most skewed variables, which seems to help a lot.

# In[288]:


skew_feats = skew_feats[abs(skew_feats) > 0.75]

for feat in skew_feats.index:
    df_raw[feat] = np.log1p(df_raw[feat])


# In[289]:


sns.distplot(df_raw[df_raw['MiscVal'] != 0]['MiscVal'])


# #### Numericalise
# 
# Lastly, we want to convert all categories to their numeric representation.
# 
# We also want to add "dummies" to the dataframe which deals with the unordered categorical variables by replacing the category: `LandSlope` with `LandSlope_Sev`, `LandSlope_Mod` and `LandSlope_Gtl`. ' We generate the dummies, then drop the original columns from the table before concatenating the dummies.

# In[290]:


df_numeric = df_raw.copy()
dummies = pd.get_dummies(df_numeric[other_cat_columns], dummy_na=True)
for col_name in categorical_columns:
    # Use +1 to push the -1 NaN value to 0
    df_numeric[col_name] = df_numeric[col_name].cat.codes + 1


# In[291]:


df_numeric.drop(other_cat_columns, axis=1, inplace=True)
df_numeric = pd.concat([df_numeric, dummies], axis=1)


# ## Model training

# Instead of separating the data into a validation set, we'll use KFolds cross-validation. It basically breaks the model into n train / val splits and trains a model then evaluates the results. It ensures that a particular train / val split is unlikely to bias the outcomes.
# 
# I'm using the `Lasso` model which is just Linear Regression with L1 regularization. Straight Linear Regression tends to overfit a lot on this dataset.

# In[292]:


kf = KFold(n_splits=10, shuffle=True, random_state=42)
model = Lasso(alpha=0.0004)
scores = np.sqrt(
    -cross_val_score(model, df_numeric, sale_price_log, cv=kf, scoring='neg_mean_squared_error'))


# In[293]:


scores.mean()


# I could actually get that result on the leaderboard, I'd be around 8th. This model translates to about 0.11898 on the test set, which puts me around 800th.
# 
# I'll train a model on the full set before submitting my predictions.

# In[294]:


final_model  = Lasso(alpha=0.0004)
final_model.fit(df_numeric, sale_price_log)


# ## Submission
# 
# Need to ensure we do all the same preprocessing on the test set as training set before generating predictions.

# In[295]:


df_test_raw = pd.read_csv(PATH / 'test.csv')


# In[296]:


house_ids = df_test_raw.pop('Id')


# In[297]:


df_test_raw['TotalSF'] = (
    df_test_raw['BsmtFinSF1'].fillna(0) +
    df_test_raw['BsmtFinSF2'].fillna(0) +
    df_test_raw['1stFlrSF'].fillna(0) +
    df_test_raw['2ndFlrSF'].fillna(0)
)


# In[298]:


for col_name in categorical_columns:
    df_test_raw[col_name] = (
        pd.Categorical(
            df_test_raw[col_name],
            categories=df_raw[col_name].cat.categories,
            ordered=True))


# In[299]:


for col in continuous_columns:
    if col not in NAs:
        continue

    df_test_raw[f'{col}_na'] = pd.isna(df_test_raw[col])
    df_test_raw[col] = df_test_raw[col].fillna(NAs[col])


# In[300]:


# Handle any other NAs
df_test_raw[continuous_columns] = df_test_raw[continuous_columns].fillna(
    df_test_raw[continuous_columns].median()
)


# In[301]:


for feat in skew_feats.index:
    df_test_raw[feat] = np.log1p(df_test_raw[feat])


# In[302]:


df_test = df_test_raw.copy()


# In[303]:


test_dummies = pd.get_dummies(df_test[other_cat_columns], dummy_na=True)
for col_name in categorical_columns:
    # Use +1 to push the -1 NaN value to 0
    df_test[col_name] = df_test[col_name].cat.codes + 1
df_test.drop(other_cat_columns, axis=1, inplace=True)
df_test = pd.concat([df_test, test_dummies], axis=1)


# In[304]:


test_preds = final_model.predict(df_test)


# Generate CSV. Note that I have to use `np.exp` to reverse the log of the predictions before submitting.

# In[305]:


pd.DataFrame(
    {'Id': house_ids, 'SalePrice': np.exp(test_preds)}
).to_csv('output.csv')


# In[ ]:




