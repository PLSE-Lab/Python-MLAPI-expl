#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# This kernel is meant to show you how I used the fast.ai library in the House Price competition without using any manual feature engineering. The only preprocessing made is the builtin methods Categorize, FillMissing and Normalize. 
# 
# The number of epochs and learning rate gave me a good result after some experimenting, but I think you can do even better!

# # Setup

# In[ ]:


from fastai.tabular import *


# In[ ]:


path = Path('../input/house-prices-advanced-regression-techniques')
output_path = Path('../working')

df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'test.csv')

len(df), len(test_df)


# # Explore data

# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.head()


# # Create databunch

# dep_var = the target variable
# 
# procs = preprocesses built in fastai
# 
# The big job here is separating the columns into continous- and categorical value types.

# In[ ]:


dep_var = 'SalePrice'
procs = [FillMissing, Categorify, Normalize]


# In[ ]:


cont_names = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr',
 'EnclosedPorch', 'Fireplaces', 'FullBath',
 'GarageYrBlt', 'GrLivArea',
 'HalfBath', 'KitchenAbvGr', 
 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea',
 'OpenPorchSF', 'PoolArea', 'ScreenPorch',
 'TotRmsAbvGrd', 'WoodDeckSF']

cat_names = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
           'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
           'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 
           'Foundation', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',
           'Electrical', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
           'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'BsmtQual', 'KitchenQual']


# ## Create the test set

# In[ ]:


test = TabularList.from_df(test_df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)


# ## Create the dataset
# I chose to create a validation set by a set slice of the data instead of a random split, for a more comparable validation.
# 
# Since this is a regression problem, for our target variable, we tell fastai that it is a list of floats that we have taken the logarithm of.

# In[ ]:


data = (TabularList.from_df(df, path=output_path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(list(range(600,800)))
                           #.split_by_rand_pct(0.2)
                           .label_from_df(cols=dep_var, label_cls=FloatList, log=True)
                           .add_test(test)
                           .databunch())


# Show the ready databunch. All categorical variables have been mapped to ints behind the scenes, but fastai is showing the strings.

# In[ ]:


data.show_batch(rows=10)


# # Create model

# To further help the model, here we specify the range for the target variable.

# In[ ]:


max_log_y = np.log(np.max(df[dep_var])*1.2)
y_range = torch.tensor([0, max_log_y], device=defaults.device)


# The model has two layers with specified dropout probabilities for each (ps).
# As metric I used exp_rmspe. Tried som a while to get my own RMSLE metric function (as used for the competition) to work, but without success.

# In[ ]:


learn = tabular_learner(data, layers=[200,100], y_range=y_range, ps=[0.05, 0.1], metrics=exp_rmspe)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit(80, 1e-2)


# In[ ]:


learn.recorder.plot_losses(skip_start=100)


# In[ ]:


learn.recorder.plot_metrics(skip_start=200)


# # Predictions and submission

# Get predictions frpom the test data set with get_preds. The model is trained on the log of the target value, here we calculate the real values to submit to the competition.

# In[ ]:


predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.exp(predictions.data).numpy().T[0]

sub_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': labels})
sub_df.to_csv(output_path/'submission.csv', index=False)

