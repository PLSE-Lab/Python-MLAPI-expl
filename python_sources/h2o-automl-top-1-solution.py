#!/usr/bin/env python
# coding: utf-8

# **H2O AutoML Solution for Kaggle Housing Prices Competition.
# Automated model stacking from H2O gives TOP 1% solution.
# I have used this for feature processing, but kept it much shorter:
# https://www.kaggle.com/sagarmainkar/sagar**

# **Import librairies**

# In[ ]:


import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt 
from scipy import stats
from scipy.stats import norm, skew  

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn 

# Limiting floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))  


# **Now read the train and test datasets in pandas dataframes**

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# **Save ID column and drop it from train & test dataframes **

# In[ ]:


# Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

# Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

# check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape))
print("The test data size after dropping Id feature is : {} ".format(test.shape))


# **Delete outliers - incredibly large homes with low prices and drop SalePrice column**

# In[ ]:


# Deleting outliers
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

# We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# **Fill nans**

# In[ ]:


# fillna
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["Alley"] = all_data["Alley"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")


# In[ ]:


# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# In[ ]:


for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')


# **Some more nan filling**

# In[ ]:


all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")


# **Convert some features to str type to be a cathegorical one**

# In[ ]:


# MSSubClass is the building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

# Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

# Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# **Process some features with LabelEncoder**

# In[ ]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
            'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))

# control shape
print('Shape all_data: {}'.format(all_data.shape))


# In[ ]:


# Adding total sqfootage feature
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# **Box Cox Transformation of (highly) skewed features**

# In[ ]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew': skewed_feats})

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))


# **Apply Cox Box transformation and create cleaned train & test data**

# In[ ]:


from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

all_data = pd.get_dummies(all_data)
print(all_data.shape)

train = all_data[:ntrain]
test = all_data[ntrain:]


# **H2O AutoML example. Train it for 2-3 hours to get a TOP 1% solution.
# Link to the documentation:
# http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
# **

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import h2o\nfrom h2o.automl import H2OAutoML\nh2o.init()\n\ntrain[\'SalePrice\'] = y_train\nhtrain = h2o.H2OFrame(train)\nhtest = h2o.H2OFrame(test)\nx = htrain.columns\ny = "SalePrice"\nx.remove(y)\n\n# train the model for 2-3 hours instead of 20 seconds\naml = H2OAutoML(max_runtime_secs = 3600, seed = 1) \naml.train(x=x, y =y, training_frame=htrain)\nlb = aml.leaderboard\nprint (lb)\nprint("generate predictions")\ntest_y = aml.leader.predict(htest)\ntest_y = test_y.as_data_frame()\n\n# submit results\nsub = pd.DataFrame()\nsub[\'Id\'] = test_ID\nsub[\'SalePrice\'] = np.expm1(test_y)\nsub.to_csv(\'submission.csv\',index=False)')


# **If you found this notebook helpful, please upvote ** :-)
