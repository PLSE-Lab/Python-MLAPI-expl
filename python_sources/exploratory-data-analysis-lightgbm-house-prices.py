#!/usr/bin/env python
# coding: utf-8

# # Housing Regression

# ## Kernel based on Ames Housing Dataset 

# The objective of predicting house prices based on features of the property is a classic regression problem in data science. Here we will conduct some EDA on the housing dataset prepared for Ames, Iowa by Dean De Cock.

# In[ ]:


import pandas as pd
import numpy as np


# Let's load the train and test data and check their shapes

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print("Train Shape: " + str(train.shape))
print("Test Shape: " + str(test.shape))


# ### Let's take a closer look at the training data

# In[ ]:


print(list(train))


# We can see that there are a lot of categorical variables such as 'Neighborhood' and 'GarageType' as well as continuous variables such as 'LotArea' and of course the target 'SalePrice'.

# Let's see how many unique values there are per column. Storing this information could help us later in deciding how to treat the columns.

# In[ ]:


unique_vals_per_col = train.T.apply(lambda x: x.nunique(), axis=1)
print(unique_vals_per_col.head(5))


# ### Plotting Living Area to Sale Price

# Often the price of a house is strongly correlated to the size of the house. Let's see if we can notice that pattern here. For now we will use the GrLivArea column for the non basement living area.

# In[ ]:


import matplotlib.pyplot as plt

# a scatter plot comparing num_children and num_pets
train.plot(kind='scatter',x='GrLivArea',y='SalePrice',color='green')
plt.show()
plt.clf()


# This graph shows us a few things. There does appear to be a strong correlation between the non-basement living area and the sale price. Also we can see that there are a couple of outliers of large properties with a low sale price. Lets remove those outliers.
# 

# In[ ]:


#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)


# ### Exploring Neighborhoods

# In[ ]:


train.groupby('Neighborhood')['SalePrice'].mean().plot(kind='bar')
plt.show()
plt.clf()


# Some neighborhoods seem more expensive than others. Don't forget to consider the sample size before reading too much further into the exact figures. There are 1481 records and 25 neighborhoods. That gives us a mean of just under 60 properties per neighborhood. Common sense tells us that in any given town some neighborhoods may be more desirable than others. Lets take a look at the count of sales per neighborhood. This will give us an idea if some neighborhoods are larger than others, or experienced higher volumes of sales.

# In[ ]:


train.groupby('Neighborhood')['SalePrice'].count().plot(kind='bar')
plt.show()
plt.clf()


# We can see that some neighborhoods had far more property transactions than others. Let's take a look at whether the style of house makes a difference to the sale price.

# ### Housing Style and Kitchen Quality

# In[ ]:


train.groupby('HouseStyle')['SalePrice'].mean().plot(kind='bar')
plt.show()
plt.clf()


# Yes it looks like the style of house makes a difference to the price. We can see for example that style "2.5Fin" is noticeably more expensive than "2.5Unf". Reading the description we find that this means a property of type  "Two and one-half story: 2nd level finished" has a higher mean selling price than "Two and one-half story: 2nd level unfinished". This is not surprising since in the unfinished property the buyer would have to do some renovation or decoration to make the property ready to be lived in comfortably. We will not treat this as an ordinal. Although "2.5Unf" has a higher mean price than "1.5Unf" it has a lower mean than "2Story". Also some of the labels do not contain a number or fit into an obvious ordinal pattern.

# Anecdotal evidence suggests that a good kitchen helps with the sale a of a property. Lets see if that is true in Ames, Iowa. We will plot the KitchenQual column against the Sale Price.
# As a reminder here are the meanings of the categorical variables:
# 
#     KitchenQual: Kitchen quality
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor

# In[ ]:


train.groupby('KitchenQual')['SalePrice'].mean().plot(kind='bar')
plt.show()
plt.clf()


# KitchenQual looks like an ordinal variable. Excellent is better than Good which is better than Typical/Average which is better than Fair. We can see that no properties sold with a kitchen adjudged to be Poor quality. Perhaps this is because the quality assessment process is subjective and the assessors didn't want to be too harsh. We can also see that the quality of the kitchen is strongly correlated to the sale price. So it seems in this case that the anecdotal evidence does have some merit. Of course there are caveats related to the sample size, and perhaps that Kitchen Quality may be highly influenced by other factors that have a larger bearing on the house price. This is something that could be investigated further. However for now it seems like a good time to start preparing some modelling. Often the feature importances output by models can add a lot of insight to what is discovered during EDA.
# 

# ### Preparing to run a model
# Get the target then drop that from train before merging with test
# 

# In[ ]:


y = train['SalePrice']
train.drop("SalePrice", axis = 1, inplace = True)


# #### Categorical Encoding
# A lot of the features we are working with are categoricals. Some but not all of those are ordinals. Lets deal with each appropriately. First we make sure that we are encoding on train and test together.

# In[ ]:


from sklearn import preprocessing
#from sklearn.preprocessing import LabelEncoder
import datetime

ntrain = train.shape[0]

all_data = pd.concat((train, test)).reset_index(drop=True)
print("all_data size is : {}".format(all_data.shape))

def ordinal_encode(df, col, order_list):
    df[col] = df[col].astype('category', ordered=True, categories=order_list).cat.codes
    return df

def label_encode(df, col):
    for c in col:
        #print(str(c))
        encoder = preprocessing.LabelEncoder()
        df[c] = encoder.fit_transform(df[c].astype(str))
    return df 

def split_all_data(all_data, ntrain):
    print(('Split all_data back to train and test: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))
    train_df = all_data[:ntrain]
    test_df = all_data[ntrain:]
    return train_df, test_df

"""
NOW START ENCODING 1. ORDINALS
"""
print(('Ordinal Encoding: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))
order_list = ['Ex', 'Gd', 'TA', 'Fa', 'Po'] #This applies to a few different columns
cols = ['KitchenQual', 'ExterQual', 'ExterCond', 'HeatingQC']
for col in cols:
    all_data = ordinal_encode(all_data, col, order_list)

order_list = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'] #This applies to a few different columns
cols = ['BsmtQual', 'BsmtCond']
for col in cols:
    all_data = ordinal_encode(all_data, col, order_list)

order_list = ['Gd', 'Av', 'Mn', 'No', 'NA']
cols = ['BsmtExposure', 'FireplaceQu', 'GarageQual', 'GarageCond']
for col in cols:
    all_data = ordinal_encode(all_data, col, order_list)
    
order_list = ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal']
cols = ['Functional']
for col in cols:
    all_data = ordinal_encode(all_data, col, order_list)
    
order_list = ['Fin', 'RFn', 'Unf', 'NA']
cols = ['GarageFinish']
for col in cols:
    all_data = ordinal_encode(all_data, col, order_list)
    
order_list = ['Ex', 'Gd', 'TA', 'Fa', 'NA'] 
cols = ['PoolQC']
for col in cols:
    all_data = ordinal_encode(all_data, col, order_list)

"""
ENCODE 2. NON-ORDINAL LABELS
"""
print(('Label Encoding: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())))
cols_to_label_encode = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 
                       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'CentralAir', 
                       'Electrical', 'GarageType', 'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
all_data = label_encode(all_data, cols_to_label_encode)

train, test = split_all_data(all_data, ntrain)

print("Train Shape: " + str(train.shape))
print("Test Shape: " + str(test.shape))

    


# ### Transform the target
# We need to transform the target to match the evaluation criteria. "Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)"

# In[ ]:


def score_transformer(y):
    y = np.log(y)
    
    return y

y = score_transformer(y)


# ### Feature Engineering

# In[ ]:


#Quicker to calculate once for train and test for values where this is appropriate
all_data = pd.concat((train, test)).reset_index(drop=True)

all_data['fe.sum.GrLivArea_BsmtFinSF1_BsmtFinSF2'] = all_data['GrLivArea'] + all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] 
all_data['fe.sum.OverallQual_Overall_Cond'] = all_data['OverallQual'] + all_data['OverallCond']
all_data['fe.mult.OverallQual_Overall_Cond'] = all_data['OverallQual'] * all_data['OverallCond']
all_data['fe.sum.KitchenQual_ExterQual'] = all_data['KitchenQual'] + all_data['ExterQual']
all_data['fe.mult.OverallQual_Overall_Cond'] = all_data['OverallQual'] * all_data['OverallCond']
all_data['fe.ratio.1stFlrSF_2ndFlrSF'] = all_data['1stFlrSF'] / all_data['2ndFlrSF']
all_data['fe.ratio.BedroomAbvGr_GrLivArea'] = all_data['BedroomAbvGr'] / all_data['GrLivArea']


# ### Feature Selection

# In[ ]:


train_features = list(all_data)
#Id should be removed for modelling
train_features = [e for e in train_features if e not in ('ExterQual', 'Condition2', 'GarageCond', 'Street', 'Alley', 'PoolArea', 'PoolQC', 'Utilities', 
                                                         'GarageQual', 'MiscVal', 'MiscFeature')]

train, test = split_all_data(all_data, ntrain)

train_features.remove('Id')

#remove highly correlated variables
#train_features.remove('GarageFinish')
#train_features.remove('GarageArea')


# ### Now lets set up a cross-validation framework

# In[ ]:


from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

nfolds=5
kf = KFold(n_splits=nfolds, shuffle=True, random_state=37) #33 originally
y_valid_pred = 0*y
y_valid_pred_cat = 0*y
fold_scores = [0] * nfolds
fold_scores_cat = [0] * nfolds

importances = pd.DataFrame()
oof_reg_preds = np.zeros(train.shape[0])
sub_reg_preds = np.zeros(test.shape[0])
sub_reg_preds_cat = np.zeros(test.shape[0])


for fold_, (train_index, val_index) in enumerate(kf.split(train, y)):
    trn_x, trn_y = train[train_features].iloc[train_index], y.iloc[train_index]
    val_x, val_y = train[train_features].iloc[val_index], y.iloc[val_index]
    
    reg = LGBMRegressor(
        num_leaves=15,
        max_depth=3,
        min_child_weight=50,
        learning_rate=0.04,
        n_estimators=1000,
        #min_split_gain=0.01,
        #gamma=100,
        reg_alpha=0.01,
        reg_lambda=5,
        subsample=1,
        colsample_bytree=0.21,
        random_state=2
    )
    reg.fit(
        trn_x, trn_y,
        eval_set=[(val_x, val_y)],
        early_stopping_rounds=20,
        verbose=100,
        eval_metric='rmse'
    )    
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')

    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)

    y_valid_pred.iloc[val_index] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    y_valid_pred[y_valid_pred < 0] = 0
    fold_score = reg.best_score_['valid_0']['rmse']
    fold_scores[fold_] = fold_score
    _preds = reg.predict(test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += _preds / nfolds
    
print("LightGBM CV RMSE: " + str(mean_squared_error(y, y_valid_pred) ** .5))
print("LightGBM CV standard deviation: " + str(np.std(fold_scores)))
   


# ### Plot Feature Importance

# In[ ]:


import seaborn as sns
import warnings
#cat_rgr.fit(X_train, y_train, eval_set=(X_valid, y_valid), logging_level='Verbose', plot=False)
warnings.simplefilter('ignore', FutureWarning)

importances['gain_log'] = np.log1p(importances['gain'])
mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])

plt.figure(figsize=(10, 14))
sns.barplot(x='gain_log', y='feature', data=importances.sort_values('mean_gain', ascending=False))


# ### Prepare the Output
# Write the prediction results to a submission file ready to submit to the competition. First convert the target back from log(target).

# In[ ]:


sub_reg_preds = np.exp(sub_reg_preds)
test.is_copy = False #disable the SettingWithCopyWarning
test.loc[:,'SalePrice'] = sub_reg_preds
test[['Id', 'SalePrice']].to_csv("tutorial_sub.csv", float_format='%.8f', index=False)


# In[ ]:




