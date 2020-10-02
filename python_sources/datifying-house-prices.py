#!/usr/bin/env python
# coding: utf-8

# # Datifying House Prices
# Hey folks! Our goal here is to predict the price of some houses around Ames, Iowa. To do it so, I will follow some steps recommended by "Introduction to Machine Learning with Python", by Andreas C. Muller & Sarah Guido. The book provived my a great initial vision of data science and I would definitely recommend it!
# Please, feel free to comment and criticize this notebook! I would love to hear from you guys, as I am just starting around ML projects and this is my first notebook :)

# ## 1. Framing the problem
# Our goal, as we said, is to predict the value of  houses based on a range of features. The solution can be used as part of the analysis process for real estate investments, as example.
# I understand this challenge as a supervised problem, which will be solved as a batch learning system.
# By now, the sucessfull measurement will be the RMSE.

# ## 2. Get the data

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm, skew

sns.set(style="ticks", palette="pastel")


# In[ ]:


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train = train.drop('Id', axis = 1)

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
ids_test = test['Id']

test = test.drop('Id', axis = 1)


# ## 3.  Explore the data
# I was really wondering how I would do this systematically through this dataset, and I thought it would be the ideal: first going for the numeric data, then the caterogical data and finish with the target.

# ### 3.1 Numeric Data
# So here i went for the direct correlations. I got the highest scores atm to understand a bit how it'd make sense. Then, I'd get those features and understand a bit how they relate to all the others. My goal here is to understand the data and get familiar with it.

# In[ ]:


correlation_target = train.corr()['SalePrice']
pd.DataFrame(correlation_target[np.abs(correlation_target) > 0.6].sort_values(ascending = False))


# * Taking a look at the documentation, Overall Quality is the big shot and it is related to the material and finish of the house. It is such an interesting thing for me, specially because I'd say the feet square area would be a better shot for the first position.
# * Also, checking the above grade (ground) living area seems important - and it seems perfectly normal.
# * The garage cars and garage area look like the same information, as we can think that largers areas can fit more cars. It makes us think that maybe we have similar information between different features
# * After this, we are talking about the basement and the squaree feet of the first floor - at it talks a lot about the living area. Again, it makes me guess (at the moment) that a bit of features can nicely answer our questions

# In[ ]:


attributes = pd.DataFrame(correlation_target[np.abs(correlation_target) > 0.6].sort_values(ascending = False)).index.tolist()

correlation_attributes = train.corr()[attributes]

grid_kws = {"width_ratios": (.9, .05), "wspace": 0.2}
f, (ax1, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize = (9, 9))

cmap = sns.diverging_palette(220, 8, as_cmap=True)

ax1 = sns.heatmap(correlation_attributes, vmin = -1, vmax = 1, cmap = cmap, ax = ax1, square = False, linewidths = 0.5, yticklabels = True,     cbar_ax = cbar_ax, cbar_kws={'orientation': 'vertical',                                  'ticks': [-1, -0.5, 0, 0.5, 1]})
ax1.set_xticklabels(ax1.get_xticklabels(), size = 10); 
ax1.set_title('Correlation Heatmap', size = 15);
cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), size = 12);


# After taking a look at the correlation of these, we can notice that there are a lot of red stuff around. It means that there are some relations like:
# * Simple internal design options (like screen/enclosed/three porch, miscellaneous stuff, kitcken area, low quality finished square feet) does not look to hard impact, as it could be easily remodelled if needed. In the other hand, external stuff (like the open porch, lot frontage, wood decks, masonry veneer) would be a bit harder to 'construct' it and it seems like having higher impact.
# * Overall Conditions does not look that relevant and it was a surprise for me. There could be several reasons, like the criteria was not well trained so it would not make sense. This idea would make some sense especially because there is a great relation between YearBuilt and the SalePrice, so we can understand that new houses tend to be in better conditions. However,  as we are talking about a lot of money invested (ye, we are buying a house), it would be fine to invest some money to make it better. This would be way cheaper than the 'heavy' part as we talked above.

# ### X.X Categorical Data
# Great. For now I'd like to study a bit the distribution of the sale price among the different cattegories. I know it is a lot, but I understood that this experience really made me feel comfy with the data and I really believe it worths.

# In[ ]:


#As we know, MSubClass is not a number, it is a class. Let's transform it
train['MSSubClass'] = train['MSSubClass'].astype(object)
test['MSSubClass'] = test['MSSubClass'].astype(object)

dtypes_ = pd.DataFrame(train.dtypes)

print('We have {:.0f} different categorical features'.format(dtypes_[dtypes_[0] == 'object'].count().iloc[0]))


# In[ ]:


f, axes = plt.subplots(44, 1, figsize=(15,250))

counter_for_axes = 0

for item in dtypes_[dtypes_[0] == 'object'][0].index.tolist():
    sns.boxplot(y='SalePrice', x=item, data=train,  orient='v' , ax=axes[counter_for_axes])
    counter_for_axes = counter_for_axes + 1


# Main takeaways for our initial exploration:
# * We have found several outliers through different features, and it really made me think about this for the next exploration
# * The general zoning and the neighbourhood are features I was really expecting differences, and I am satisfied from what I see.
# * As we saw in in the previous topic, external structure seems to show some difference, like paved or graved alley acess to propriety, the house style, the roof, so we are running in the same direction.
# * Until now, we were looking especially for big and bold structural features as relevant, however the central air, heather and fireplace showed us some potential. The common thing is that those are features that would not be that 'easy' to install in a house.
# * Some features are related to the same classes, but just separated due to different combinations (like Condition 1 and Condition 2). We will have to treat this after.
# * This is the very first moment we are talking about a commercial feature: the sale type and condition seem to be relevant for this decision also.
# 

# ## 4. Prepare the Data and Running the Models
# First of all, we are taking a look at missing values and understanding each way

# In[ ]:


missing_ = pd.DataFrame(train.isna().sum()/len(train)).sort_values(0, ascending = False)
missing_values = missing_[missing_[0] > 0]

plt.figure(figsize=(15,5))
plt.bar(range(len(missing_values)),missing_values[0])
plt.xticks(range(len(missing_values)), missing_values.index, rotation = 'vertical')
plt.title('Missing Values %')

missing_values.index.tolist()


# * For MiscFeatures, Alley, Fence, Fireplace, PoolQC: the lack of information means there is not something of this feature in the house
# * LotFrontage made me think about two situations: just missing the info or not having it. I will go with the second option. We will put it as 0.
# * The garage missing values are the same for all the features. It is probably telling us there is no garage. The GarageCond, GarageType, GarageFinish and GarageQual will take the same destiny as the first group we said. I will take YearBuilt as 0.
# * Same for basement info and for MasVnr
# * Electrical are boolean, so we will go with most frequent values

# From now, let's just create the Pipeline to run cross-validations and select the best model:

# In[ ]:


from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

#Here I will make a class to select some columns according to their types
class SelectType(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        return X.select_dtypes(include = [self.dtype])


# In[ ]:


#Here I am selecting the right columns

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


# In[ ]:


#Here I am applying the pipeline and testing the first model, the ridge one.

from sklearn.pipeline import make_pipeline

preprocessing_pipeline = make_pipeline(
    ColumnSelector(columns=['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',
       'MoSold', 'YrSold', 'SaleType', 'SaleCondition']),
    FeatureUnion(transformer_list = [
    ('Numbers', make_pipeline(
        SelectType(np.number), SimpleImputer(strategy='constant', fill_value = 0), StandardScaler())),
    ('Object', make_pipeline(
        SelectType('object'), SimpleImputer(strategy='constant', fill_value = 'No'), OneHotEncoder(handle_unknown="ignore")))
])
)


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

classifier_pipeline_ridge = make_pipeline(preprocessing_pipeline,
                                    Ridge()
)

param_grid = {"ridge__alpha": [0.01, 0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 100]}

X = train.iloc[:,:79]
y = train.iloc[:,79:]

classifier_model_ridge = GridSearchCV(classifier_pipeline_ridge, param_grid, cv=5, scoring = 'neg_mean_squared_error')
classifier_model_ridge.fit(X,y)


# In[ ]:


#Let's go for the Lasso

from sklearn.linear_model import Lasso

classifier_pipeline_lasso = make_pipeline(preprocessing_pipeline,
                                      Lasso())

param_grid = {"lasso__alpha": [0.01, 0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]}

classifier_model_lasso = GridSearchCV(classifier_pipeline_lasso, param_grid, cv=5, scoring = 'neg_mean_squared_error')
classifier_model_lasso.fit(X,y)
classifier_model_lasso.best_score_


# In[ ]:


#Now its RF time

from sklearn.ensemble import RandomForestRegressor

classifier_pipeline_rf = make_pipeline(preprocessing_pipeline,
                                      RandomForestRegressor())

param_grid = {"randomforestregressor__max_depth": [2,3,4]}

classifier_model_rf = GridSearchCV(classifier_pipeline_rf, param_grid, cv=5, scoring = 'neg_mean_squared_error')
classifier_model_rf.fit(X,y)
classifier_model_rf.best_score_


# In[ ]:


#And Kernel Ridge

from sklearn.kernel_ridge import KernelRidge

classifier_pipeline_krr = make_pipeline(preprocessing_pipeline,
                                      KernelRidge())

param_grid = {"kernelridge__alpha": [0.01, 0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 100]}

classifier_model_krr = GridSearchCV(classifier_pipeline_krr, param_grid, cv=5, scoring = 'neg_mean_squared_error')
classifier_model_krr.fit(X,y)
classifier_model_krr.best_score_


# In[ ]:


#Taking a look at the scores

krr = {'model':'krr','result':classifier_model_krr.best_score_}
ridge = {'model':'ridge','result':classifier_model_ridge.best_score_}
lasso = {'model':'lasso','result':classifier_model_lasso.best_score_}
random_forest = {'model':'random_forest','result':classifier_model_rf.best_score_}

result = pd.DataFrame([krr,ridge,lasso,random_forest])
result.sort_values('result', ascending = False)


# In[ ]:


#It looks we are going for the lasso.

pd.DataFrame(classifier_model_lasso.cv_results_)


# In[ ]:


#Getting ready
classifier_pipeline_lasso_2 = make_pipeline(preprocessing_pipeline,
                                      Lasso(alpha = 200))


# In[ ]:


#Fitting it
classifier_pipeline_lasso_2.fit(X,y)


# In[ ]:


#Going for the test
ans = pd.DataFrame(classifier_pipeline_lasso_2.predict(test))


# In[ ]:


#Just making it according to the sample
ans['Id'] = ids_test
ans2 = ans.set_index('Id')


# In[ ]:


#Final adjustments
ans2.rename(columns={0:'SalePrice'}, inplace = True)


# In[ ]:


#Done!
ans2


# In[ ]:





# In[ ]:




