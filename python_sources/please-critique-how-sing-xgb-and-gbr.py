#!/usr/bin/env python
# coding: utf-8

# So I think I'm done here, unless I get any comments that shine light on glaring mistakes/dumb things I've done :P  
#   
# Things to condsider:  
#  - I need to probably change how I impute data and spend more time cleaning it  
#  - Leaderboard score is sitting at ~0.12034  

# ## 1. The Data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


#I/O

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
testid = test_df['Id']
trainid = train_df['Id']

test_df = test_df.drop(['Id'], axis = 1)
train_df = train_df.drop(['Id'], axis = 1);


# Train: 1460 rows x 81 columns  
# Test: 1459 rows x 80 columns  
#   
# Lots of missing values  

# ### Correlation Heatmap of Test Data  
# 
# This will provide a decent amount of insight into features to pay attention to.

# In[ ]:


corrheatmap = train_df.corr()
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corrheatmap, vmax=.8, square=True);


# We can see that there are a few key features that correlate positively with SalePrce in the test data:  
# 
#  - Overall Quality (whatever that actually means)  
#  - Total Basement and 1st Floor Square Feet (I feel like in most cases, these two should be similar)  
#  - Gr Living Area/Total Rooms Above Ground
#  - Garage Area/Cars (I feel like these two mean the same thing  
# 
# It makes sense that people are interested in things like quality, square-foot and area more than things like whether the porch is enclosed.

# ### Visualizations and Brief Discussion

# **Pairplot of Columns That Show Strong Correlation**  (sorry about the small axis labels)

# In[ ]:


#https://seaborn.pydata.org/generated/seaborn.pairplot.html 
plt.figure()
columns = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea',
           '1stFlrSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_df[columns])
plt.show()


# The plot above makes many relationships more clear, and I'll go through each one very briefly now.  
# 
# SalePrice vs:
#  - Overall Quality: Seems to be a *mostly* linear relationship between them  
#  - GrLivArea : Very steep correlation with a *conal shape*  
#  - GarageArea: A couple of zero-values for houses with no garages probably, but otherwise a generally positive correlation that is quite messy  
#  - 1stFlrSF: Very steep correlation with a *conal shape*  
#  - FullBath: A mostly linear relationship where more bathrooms shows an increase in SalePrice, though there is a lot of overlap in price for different values of FullBath  
#    
#    I will need to transform the data using a log1p or boxcox transform later seein as many of the distributions show a clear skew.

# In[ ]:


combined_df = train_df.drop(["SalePrice"], axis=1)
combined_df = pd.concat([combined_df, test_df])


# There are 34 features across train and test datasets with missing values. Let's fix the data feature by feature. It'll be tedious, but it must be done. I'll be deleting features with more than 20% data missing, (~600 missing values and above). The remaining data will be considered depending on the nature of the feature. 

# # 2. Data Cleaning and Transformation

# In[ ]:


#More than 600 Missing Values:
combined_df = combined_df.drop(['MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'PoolQC'], axis=1)

#Filling Missing Data with Mean Value
combined_df['LotFrontage'].fillna(combined_df['LotFrontage'].mean(), inplace=True)

# Filling Garage Features with "None" and 0
for x in ('GarageType', 'GarageFinish', 'GarageCond', 'GarageQual',
          'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
          'BsmtFinType2', 'MasVnrType', 'Utilities'):
    combined_df[x] = combined_df[x].fillna('None')
    
#Filling with 0
for x in ('GarageArea', 'GarageCars', 'MasVnrArea', 'TotalBsmtSF',
          'BsmtUnfSF', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1',
          'BsmtFinSF2'):
    combined_df[x] = combined_df[x].fillna(0)
    
#Filling with Mode
combined_df['YearBuilt'] = combined_df['YearBuilt'].fillna(combined_df['YearBuilt'].mode()[0])
combined_df['GarageYrBlt'] = combined_df['GarageYrBlt'].fillna(combined_df['GarageYrBlt'].mode()[0])
combined_df['MSZoning'] = combined_df['MSZoning'].fillna(combined_df['MSZoning'].mode()[0])
combined_df['KitchenQual'] = combined_df['KitchenQual'].fillna(combined_df['KitchenQual'].mode()[0])
combined_df['Electrical'] = combined_df['Electrical'].fillna(combined_df['Electrical'].mode()[0])
combined_df['Exterior1st'] = combined_df['Exterior1st'].fillna(combined_df['Exterior1st'].mode()[0])
combined_df['Exterior2nd'] = combined_df['Exterior2nd'].fillna(combined_df['Exterior2nd'].mode()[0])
combined_df['SaleType'] = combined_df['SaleType'].fillna(combined_df['SaleType'].mode()[0])
#Misc
combined_df['Functional'] = combined_df['Functional'].fillna('Typ'); #Notes Say that NA == Typical


# In[ ]:


missing_df = combined_df.isnull().sum().sort_values(ascending=False)
print(missing_df.head(1))
#2919 Entries Total x 75 Features


# Combining the data for square-footage into one variable/feature, and removing the old features:

# In[ ]:


combined_df['CombinedSF'] = (combined_df['TotalBsmtSF'] + combined_df['GarageArea'] + combined_df['1stFlrSF'] + combined_df['2ndFlrSF'])

combined_df.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea'], inplace=True, axis=1)


# All fixed! Next, I want to deal with categorical data. I will use a LabelEncoder, because pd.get_dummies creates multiple columns, and I don't want to deal with that just yet.

# In[ ]:


categoricaldata = ('Street', 'BsmtCond', 'GarageCond', 'GarageQual', 
                  'BsmtQual', 'CentralAir', 'ExterQual', 'ExterCond', 'HeatingQC', 
                  'KitchenQual', 'BsmtFinType1','BsmtFinType2', 'Functional',
                  'BsmtExposure', 'GarageFinish','LandSlope', 'LotShape', 'MSZoning',
                  'LandContour', 'Utilities', 'LotConfig', 'Neighborhood', 'Condition1',
                  'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                  'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'GarageType',
                  'PavedDrive','SaleType', 'SaleCondition')
from sklearn.preprocessing import LabelEncoder

lec = LabelEncoder()
for x in categoricaldata:
    lec.fit(list(combined_df[x].values))
    combined_df[x] = lec.transform(list(combined_df[x].values))


# The data has gone through Label Encoding, but I want to fix the skewness of the data before I continue

# In[ ]:


skewcheck = combined_df.dtypes[combined_df.dtypes != "object"].index
skewed = combined_df[skewcheck].skew().sort_values(ascending=False)
skew_df = pd.DataFrame(skewed)
skew_df = abs(skew_df)
skew_df.shape


# In[ ]:


skew_df = skew_df[skew_df > 0.75]
skew_df = skew_df.dropna()
skew_df.shape


# A couple of people have said that a box-cox transform works better than a log transformation, so I'll try using that

# In[ ]:


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.boxcox1p.html
#Using Serigne's general code for this part: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

from scipy.special import boxcox1p
needs_fixing = skew_df.index
lm = 0.25
for x in needs_fixing:
    combined_df[x] = boxcox1p(combined_df[x], lm)
    combined_df[x] += 1


# In[ ]:


train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

combined_df = pd.get_dummies(combined_df, columns=list(categoricaldata))
combined_df.shape


# In[ ]:


X_train = combined_df[:1460]
X_test = combined_df[1460:]
Y_train = train_df['SalePrice']

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)


# # 3. Model Building (WORK IN PROGRESS)   
#   
# #### (I've never stacked models in the past, so I'll try and do it for this project. I found a good video [here](https://www.youtube.com/watch?v=MOqdUoZo5Vo) that encapsulates the use and importance of modeling.)

# In[ ]:


from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn import metrics
import xgboost as xgb
from sklearn.metrics import mean_squared_error


# In[ ]:


#Info from:
#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
#https://www.kaggle.com/wiki/RootMeanSquaredError
#https://www.kaggle.com/apapiu/regularized-linear-models

kf = KFold(n_splits=10, shuffle=True, random_state=42)
kf = kf.get_n_splits(X_train)

def rmsecv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv = kf))
    return (rmse)


# In[ ]:


gbr_clf = GradientBoostingRegressor(n_estimators=8000, learning_rate=0.005, subsample=0.8,
                                   random_state=42, max_features='sqrt', max_depth=5, )

xgb_clf = xgb.XGBRegressor(n_estimators=15000, colsample_bytree=0.8, gamma=0.0, 
                             learning_rate=0.005, max_depth=3, 
                             min_child_weight=1,reg_alpha=0.9, reg_lambda=0.6,
                             subsample=0.2,seed=0, silent=1)

#I got these with gridsearch on my own system and some intuition...took a while 


# In[ ]:


#gbr_score = rmsecv(gbr_clf)
#print(gbr_score.mean())
## it comes out to ~0.11626


# In[ ]:


#xgb_score = rmsecv(xgb_clf)
#print(xgb_score.mean())
# it comes out to ~0.119 but final score is 0.1265....


# In[ ]:


xgb_clf.fit(X_train,Y_train)
gbr_clf.fit(X_train,Y_train)

xx = np.expm1(xgb_clf.predict(X_test))
gg = np.expm1(gbr_clf.predict(X_test))

final = pd.read_csv("../input/sample_submission.csv")

final['SalePrice'] = (xx*0.6 + gg*0.4)

final.to_csv('housing_pred.csv', index=False)


# In[ ]:


xgb.plot_importance(xgb_clf, max_num_features=20)
plt.show()

