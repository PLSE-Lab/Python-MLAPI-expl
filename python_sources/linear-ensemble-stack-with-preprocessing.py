#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import r2_score, make_scorer
import lightgbm as lgbm
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Regression modelling with house prices
# 
# This kernel builds a regression model for the Kaggle house price competition.
# 
# There are lots of great kernels already submitted that cover exploration and visualisation, so I will go light on that section and heavier on the data processing and modelling with the hope of getting a respectable score.
# 
# ### Contents
# 1. [Data preprocessing](#preprocessing)
#     1. [Accomodating nulls](#nulls)
#     2. [Examining types](#types)
#     3. [Encoding](#encoding)
#     4. [Engineering](#engineering)
#     5. [Outliers](#outliers)
#     6. [Transformations](#transformations)
#     7. [Scaling](#scaling)
# 2. [Model building](#modelling)
#     1. [Linear regression](#linear)
#     2. [LightGBM](#lightgbm)
#     4. [Stack](#stack)
# 3. [Make submission](#submission)

# ### Data preprocessing <a name="preprocessing"></a>

# There is a fair whack to do here. Good mix of categorical, ordinal and continuous data with all three kinds stored as either numerical or string data. In addition, some of the columns are mostly null or lack variance as a result of being mostly zero, so there are a few decisions to make here that will have a large impact on later modelling performance.
# 
# First step though is to load the train and test data sets. Separate the ID column as metadata as we don't want to use it in processing or modelling.

# In[2]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print('Train size:\t#rows = {}, #cols = {}\nTest size:\t#rows = {}, #cols = {}'      .format(train_df.shape[0], train_df.shape[1], test_df.shape[0], test_df.shape[1]))


# Train set is wider as a result of having the target column, SalePrice. Leave it in for now to help processing. 
# 
# Append the two sets for now to streamline the preprocessing steps and reduce error possibility.

# In[3]:


train_df['train_set'] = 1
test_df['train_set'] = 0
train_ids = train_df.pop('Id')
test_ids = test_df.pop('Id')
y = train_df.pop('SalePrice')
df = train_df.append(test_df).reset_index(drop=True)
df.head()


# #### Nulls <a name="nulls"></a>
# 
# So what do we have? A good first step with checking the quality of any data set is to look for nulls. Lets see which features in this set are affected.

# In[4]:


cols_with_nulls, null_perc = [], []
for col in df.columns:
    if sum(df[col].isnull()) > 0:
        cols_with_nulls.append(col)
        null_perc.append(100 * sum(df[col].isnull()) / len(df))
null_counts = pd.DataFrame({'column': cols_with_nulls, 'null_perc': null_perc}).sort_values(by='null_perc',
                                                                                            ascending=False)


# In[5]:


fig, ax = plt.subplots(1, 1, figsize=[10, 10])
sns.barplot(x='null_perc', y='column', data=null_counts, ax=ax)
ax.set_xlabel('Null percentage (%)')
ax.set_ylabel('')
plt.show()


# So in fairness, the majority of the features are OK. However, there are a handful of columns over 50% null. I will drop these as any machine learning algorithm will struggle to make use of them. Even algorithms like LightGBM that have native handling of nulls would probably ignore them due to a lack of variance.

# In[6]:


null_threshold = .5
for col in df.columns:
    if (sum(df[col].isnull()) / len(df) > null_threshold) & (col != 'SalePrice'):
        df.drop(col, axis=1, inplace=True)


# #### Datatypes <a name="types"></a>

# Next let's check the data *types*. A starting assumption I would make is that integer columns contain counts or ordinal data, floats contain continuous data and objects contain ordinal or categorical data as strings. 

# In[7]:


df.dtypes.value_counts().plot(kind='barh')
plt.title('Data types present with counts')
plt.show()


# So those are the numpy datatypes, but I would like to group them by conceptual datatypes, i.e. continuous, ordinal and categorical, for further use in this kernel.
# 
# To start, I can look at the numerical values (those that are not dtype == object), with the presumption that most will be continuous or ordinal

# In[8]:


numerical_cols = df.columns[df.dtypes != 'object'].tolist()
print(numerical_cols)


# So does this look right? As mentioned, a few, if not most, of these will be ordinal or possibly encoded categoricals. Fortunately, I don't have to infer this (using df.describe() for example) as Kaggle provides the answers in the data_description.txt file (some painstaking ctrl+c and +v'ing later...):

# In[9]:


ordinal_numericals = ['BsmtFullBath', 'BsmtHalfBath', 'Fireplaces', 'FullBath', 'GarageCars', 
                      'HalfBath', 'MoSold', 'OverallCond', 'OverallQual', 'TotRmsAbvGrd']
date_numericals = ['GarageYrBlt', 'YearBuilt', 'YearRemodAdd', 'YrSold']
categorical_numericals = ['MSSubClass']


# The remaining numericals are the continuous ones.

# In[11]:


continuous_feats = [col for col in numerical_cols if col not in ordinal_numericals and                                                     col not in date_numericals and                                                     col not in categorical_numericals and                                                     col != 'SalePrice' and col != 'train_set']


# In[12]:


df[continuous_feats].describe()


# Similarly, I can expect the object dtypes to be non-numerical and therefore mostly categorical. However, from the data_description.txt document, a fair few of the non-numerical datatypes are ordinal. For example: 
# 
# BsmtQual: Evaluates the height of the basement
# - Ex	Excellent (100+ inches)	
# - Gd	Good (90-99 inches)
# - TA	Typical (80-89 inches)
# - Fa	Fair (70-79 inches)
# - Po	Poor (<70 inches
# - NA	No Basement
#        
# It would be a shame to lose this order by making BsmtQual purely categorical, so this may need some thought. In practice, this means only label encoding them in order rather that one-hot encoding. However, there are no guarantees label encoded ordinals will perform better that one-hot encoded categoricals, so I will probably CV to check.
# 
# Define them separately for now.

# In[13]:


non_numerical_cols = df.columns[df.dtypes == 'object'].tolist()
print(non_numerical_cols)


# In[14]:


categorical_strings = ['BldgType', 'Condition1', 'Condition2', 'Electrical', 'Exterior1st', 
                       'Exterior2nd', 'Foundation', 'GarageType', 'Heating', 'HouseStyle', 
                       'LandContour', 'LotConfig', 'LotShape', 'MSZoning', 'MasVnrType', 
                       'Neighborhood', 'PavedDrive', 'RoofMatl', 'RoofStyle', 'SaleCondition',
                       'SaleType', 'Street', 'Utilities']
ordinal_strings = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'ExterCond',
                  'ExterQual', 'FireplaceQu', 'Functional',  'GarageCond', 'GarageFinish', 'GarageQual',
                   'HeatingQC', 'KitchenQual', 'LandSlope']
boolean_strings = ['CentralAir']


# Whilst I have the datatypes broken out, I will use this to inform a null-filling strategy:
# 
# - Continuous and date variables filled by median
# - Ordinal values filled by the mode
# - Categorical values filled with a null category ('NA')

# In[15]:


all_categoricals = categorical_numericals + categorical_strings
df[all_categoricals] = df[all_categoricals].fillna('NA')

all_ordinals = ordinal_numericals + ordinal_strings
mode_df = pd.DataFrame(data=df[all_ordinals].mode().values.tolist()[0], index=all_ordinals)
df[all_ordinals] = df[all_ordinals].fillna(mode_df[0])

df[continuous_feats] = df[continuous_feats].fillna(df[continuous_feats].median())
df[date_numericals] = df[date_numericals].fillna(df[date_numericals].median())


# #### Encoding  <a name="encoding"></a>

# So both my categorical and ordinal features need encoding. All categoricals need one-hot encoding. Non-numerical ordinals need at least label encoding (by painstakingly mapping their order to integers), whilst numerical ordinals *may* be fine as they are.
# 
# ##### Categorical features

# In[16]:


print('Shape before one-hot encoding of categoricals =\t{}'.format(df.shape))
df = pd.get_dummies(df, columns=all_categoricals)
print('Shape after one-hot encoding of categoricals =\t{}'.format(df.shape))


# ##### Ordinal features
# 
# Unfortunately, I will have to create dictionaries to map the non-numerical features to from the data_description.txt file.

# In[17]:


ordinal_dict_1 = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
ordinal_dict_2 = {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
ordinal_dict_3 = {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
ordinal_dict_4 = {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}
ordinal_dict_5 = {'Na': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
ordinal_dict_6 = {'Sev': 0, 'Mod': 1, 'Gtl': 2}


# In[18]:


for cat in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
            'GarageQual', 'GarageCond', ]:
    df[cat] = df[cat].map(ordinal_dict_1)
    
for cat in ['BsmtExposure']:
    df[cat] = df[cat].map(ordinal_dict_2)
    
for cat in ['BsmtFinType1', 'BsmtFinType2']:
    df[cat] = df[cat].map(ordinal_dict_3)
    
for cat in ['Functional']:
    df[cat] = df[cat].map(ordinal_dict_4)
    
for cat in ['GarageFinish']:
    df[cat] = df[cat].map(ordinal_dict_5)
    
for cat in ['LandSlope']:
    df[cat] = df[cat].map(ordinal_dict_6)
    
# Check I haven't introduced more nulls
null_counts.set_index('column', inplace=True)
for col in ordinal_strings:
    if sum(df[col].isnull()) > 0:
        null_counts.loc[col, 'perc_after_encoding'] = 100 * sum(df[col].isnull()) / len(df)
        assert null_counts.loc[col, 'perc_after_encoding'] == null_counts.loc[col, 'null_perc'],        'Nulls introduced in feature {}, check corresponding dictionary'.format(col)


# Boolean string variables just need label encoding

# In[19]:


for cat in boolean_strings:
    encoder = LabelEncoder()
    df[cat] = encoder.fit_transform(df[cat])


# So how does the data set look now after encoding?

# In[20]:


df.head()


# #### Engineering <a name="engineering"></a>
# 
# Perhaps a loose definition of engineering, but in this section I'm looking to create new features from the existing ones. 

# In[21]:


# Total prime living space - indoor, proper space
df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF']

# Total 'sub-prime' living space - outdoors + basements etc.
df['AltTotalSF'] = df['TotalBsmtSF'] + df['ScreenPorch'] + df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch']                   + df['GarageArea'] + df['LowQualFinSF']


# For the date variables, convert them to the difference in years between each feature and YrSold.

# In[22]:


date_numericals.remove('YrSold')
for date_col in date_numericals:
    df[date_col] = df['YrSold'] - df[date_col]

# Create a Boolean of those properties that have had work done
df['RemodelledFlag'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)


# #### Outliers <a name="outliers"></a>
# 
# This is a regression problem, so we want to check it there are any major outliers in the target variable. If we find some, which we probably will, it is best to delete them so the model doesn't overfit to them. This is essentially admitting a limitation of machine learning in that in can only learn the general case, and is by nature unable to predict outlying data that doesn't follow any trends.

# In[23]:


# Guess the most important feature just to break the data out:
fig, ax = plt.subplots(1, 1, figsize=[6, 4])
ax.scatter(df.loc[df['train_set'] == 1, 'TotalSF'], y)
ax.set_xlabel('Total area (feet^2)')
ax.set_ylabel('Sale price ($)')
plt.show()


# So I would probably say there are 4 outliers here. There are two in the lower right that do not follow the trend at all, and there are two in the top right. These two seem to follow the linear relationship here, but they are so far away from the rest of the data that I worry they might be outliers with respect to other features.

# In[24]:


outlier_mask = (df['TotalSF'] > 4000) & (df['train_set'] == 1)
outlier_mask_target = outlier_mask[df['train_set'] == 1]
df = df[~outlier_mask]
y = y[~outlier_mask_target.values]


# #### Transformations <a name="transformations"></a>
# 
# Here, I will be looking to correct the skew in the distribution of the continuous features including the target variable.

# In[25]:


skew_array = skew(df[continuous_feats].fillna(df[continuous_feats].median()), axis=0)
skew_df = pd.DataFrame({'column': continuous_feats, 'skew': skew_array}).sort_values(by='skew', ascending=False)


# Have a look at the worst ones. Seems like the biggest source of skew is having values that are mostly 0.

# In[26]:


fig, axs = plt.subplots(2, 2, figsize=[13, 8])
i = 0
for ax in axs.ravel():
    sns.distplot(df[skew_df.iloc[i, 0]], ax=ax)
    ax.set_title(skew_df.iloc[i, 0])
    i += 1
plt.tight_layout()
plt.show()


# In[27]:


df[continuous_feats] = np.log1p(df[continuous_feats].fillna(df[continuous_feats].median()))
log_y = np.log1p(y)


# In[28]:


new_skew_array = skew(df[continuous_feats].fillna(df[continuous_feats].median()), axis=0)
new_skew_df = pd.DataFrame({'column': continuous_feats, 'skew': new_skew_array})              .sort_values(by='skew', ascending=False)
fig, axs = plt.subplots(2, 2, figsize=[13, 8])
i = 0
for ax in axs.ravel():
    sns.distplot(df[skew_df.iloc[i, 0]], ax=ax)
    ax.set_title(skew_df.iloc[i, 0])
    i += 1
plt.tight_layout()
plt.show()


# So log transformation doesn't fix the mostly zeros columns, but does help otherwise. This raises the issue of low-variance features - should they be dropped due to not providing rich information whilst increasing the dimensionality?

# #### Scaling <a name="transformations"></a>
# 
# To give myself the option of using non-tree methods (e.g. linear models or neural nets), I want to scale my features such that they are all of the same magnitude. This will prevent any feature gaining inaccurate training weight just on account of it being big numbers.
# 
# To do so I will just use standard scaling: scale to a mean of 0 and standard deviation of 1. As per, sklearn has a tidy class for this.

# In[29]:


scaler = MinMaxScaler()
scaled_df = pd.DataFrame(data=scaler.fit_transform(df),
                         columns=df.columns)
scaled_df.head()


# So... I think that leaves the dataset in an OK place for now - we'll see what happens when I try and fit a model to it!

# ### Model Building <a name="modelling"></a>
# 
# Time to try and make some predictions. I aim to apply some of the most popular types of model and stack them. Ideally I would want to include a neural network in the stack but the relatively small amount of data makes me think it wouldn't perform much better than simpler models.
# 
# 1. A linear regression - Simple, understandable and a good stack component
# 2. Gradient boosted trees - Powerful, verstaile and often have simple APIs
# 
# Before I train any models however, I need to re-separate my data set:

# In[30]:


train_df = df[df['train_set'] == df['train_set'].max()].copy()
test_df = df[df['train_set'] == df['train_set'].min()].copy()
for dataset in [train_df, test_df]:
    dataset.drop('train_set', axis=1, inplace=True)


# I also want to define a scoring metric. Makes sense to use the error metric used in the leaderboard for this competition which is root mean squared logarithmic error. Since the target has already been log transformed, if I use an RSME error it'll translate as the same thing.

# In[31]:


def rmsle(y_true, y_pred):
    assert len(y_pred) == len(y_true), 'Input arrays different lengths'
    return np.sqrt(np.mean(np.power(y_pred - y_true, 2)))


# In[32]:


def rmsle_for_lgbm(preds, train_data):
    labels = train_data.get_label()
    return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(preds) - np.log1p(labels), 2))), False


# In[33]:


rmsle_score = make_scorer(rmsle)


# ##### Linear regression - scikit learn API <a name="linear"></a>
# 
# There isn't a huge amount to do in fitting a linear regression; the only thing I want to really examine is whether regularisation will be big help. I'll compare the CV scores of normal, ridge, lasso and elastic net regressions.

# In[34]:


regressions_to_try = [LinearRegression(), Ridge(), Lasso(), ElasticNet()]
preds_list = []
for model in regressions_to_try:
    scores = cross_validate(model, train_df, log_y, cv=3, scoring=rmsle_score, return_train_score=False)
    preds_list.append(cross_val_predict(model, train_df, log_y, cv=3))
    print('{} RSMLE loss = {:.4f}'.format(type(model), scores['test_score'].mean()))


# So ridge seems to perform the best. In the absence of a full gridsearch for brevity's sake, I'll just investigate optimising the regularisation parameter. From the results below, seems like quite a strong regularisation works the best.

# In[35]:


for alph in [1, 3, 6, 9, 12, 15]:
    test_model = Ridge(alpha=alph)
    scores = cross_validate(test_model, train_df, log_y, cv=3, scoring=rmsle_score, return_train_score=False)
    print('Alpha = {}, Ridge RSMLE loss = {:.4f}'.format(alph, scores['test_score'].mean()))


# In[36]:


lin = Ridge(alpha=9)
lin.fit(train_df, log_y)
lin_preds = lin.predict(test_df)


# At this stage, let's have a quick look at a plot of predicted vs. actual values to see if there is anything odd happening.

# In[37]:


fig, (ax, ax1) = plt.subplots(1, 2, figsize=[15,5])

ax.scatter(log_y, preds_list[1], label='Ridge predictions')
ax1.scatter(log_y, preds_list[2], label='Lasso predictions')

for axs in (ax, ax1):
    axs.plot([min(preds_list[1]), max(preds_list[1])], 
             [min(log_y), max(log_y)], label='y = x', color='k', linestyle='--')
    axs.set_xlabel('Prediction value ($)')
    axs.set_ylabel('Actual value ($)')
    axs.legend()

plt.show()


# It's quite clear how much stronger Ridge here is than Lasso - implying that punishing outliers more is a better process in this case.

# ##### Gradient boosted trees - LightGBM <a name="lightgbm"></a>
# 
# I've chosen LightGBM as my gradient booster of choice as it seems to perform at least as well as XGBoost in most cases and is quite a bit faster to train.

# In[63]:


lgbm_model = lgbm.LGBMRegressor(
    boosting_type='gbdt',
    objective='huber',
    learning_rate=0.2,
    min_child_samples=30,
    colsample_bytree=0.9,
    max_depth=-1,
    num_leaves=31,
    reg_lambda=0,
    n_estimators=1000
)

skgbm_model = GradientBoostingRegressor(
    loss='huber',
    n_estimators=700,
    alpha=.5,
    max_depth=3,
    subsample=.6
)


# In[64]:


ensembles_to_try = [lgbm_model, skgbm_model]
preds_list = []

for i in range(len(ensembles_to_try)):
    scores = cross_validate(ensembles_to_try[i], train_df, log_y, 
                            cv=3, 
                            scoring=rmsle_score, 
                            return_train_score=False)
    preds_list.append(cross_val_predict(ensembles_to_try[i], train_df, log_y, cv=3))
    print('{} RSMLE loss = {:.4f}'.format(type(ensembles_to_try[i]), scores['test_score'].mean()))


# In[40]:


fig, (ax, ax1) = plt.subplots(1, 2, figsize=[15,5])

ax.scatter(log_y, preds_list[0], label='LGBM predictions', color='r')
ax1.scatter(log_y, preds_list[1], label='SKLearn GBM predictions', color='r')

for axs in (ax, ax1):
    axs.plot([min(preds_list[1]), max(preds_list[1])], 
             [min(log_y), max(log_y)], label='y = x', color='k', linestyle='--')
    axs.set_xlabel('Prediction value ($)')
    axs.set_ylabel('Actual value ($)')
    axs.legend()

plt.show()


# Having found the right parameters and number of boosting rounds from CV, train the best booster.

# In[41]:


gbm = skgbm_model.fit(train_df, log_y)
gbm_preds = gbm.predict(test_df)


# Have a look at feature importance. The continous features that relate to size of the property dominate the feature importances, with OverallQual and OverallCond coming in amongst them. 
# 
# Also interesting is that none of the one-hotted categorical features are coming through here possibly highlighting their limitation in regression models.

# In[42]:


fig, ax = plt.subplots(1, 1, figsize=[7,10])
features_importance = pd.DataFrame({'Feature': test_df.columns, 'Importance': gbm.feature_importances_})                      .sort_values(by='Importance', ascending=False)

top_n_feats_to_plot = 20
sns.barplot(x='Importance', y='Feature', data=features_importance[:top_n_feats_to_plot], orient='h', ax=ax)
plt.show()


# ### Make submission <a name="submission"></a>
# 
# Now just choose the best set of predicions and write to .csv

# In[43]:


linear_ratio = .6
averaged_preds = (1-linear_ratio)*gbm_preds + linear_ratio*lin_preds


# In[44]:


output = pd.DataFrame({'Id': test_ids, 'SalePrice': np.expm1(averaged_preds)})
output.to_csv('submission.csv', index=False)


# In[ ]:




