#!/usr/bin/env python
# coding: utf-8

# # 1) Load Data

# Firstly, we load train and test data into dataframe using pandas library because it makes handling data much easier and efficient.

# In[ ]:


import pandas as pd 

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# # 2) Data Preparation

# ## Deleteing Missing Data

# Now we need to know how many missing values we have in each field and if they are considerable we should delete this field as we don't have it for many instances and it will not be helpful.

# In[ ]:


train_df_na = (train_df.isnull().sum() / len(train_df)) * 100
train_df_na = train_df_na.drop(train_df_na[train_df_na == 0].index).sort_values(ascending=False)[:20]
missing_data = pd.DataFrame({'Missing Ratio' :train_df_na})
missing_data.head(10)


# As you can see there is a high ratio of missing values in some features. I deleted the features having missing ratio greater that 80 percent. 

# In[ ]:


train_df = train_df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
test_df = test_df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)


# ## Adding a New Feature

# Now we need some feature engineering. These new features are some new meaningful features which I added to train and test datasets.

# In[ ]:


for dataset in [train_df, test_df]:
    dataset['SF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF'] 
    dataset['TotalBath'] = dataset['BsmtFullBath'] + (1/2) * dataset['BsmtHalfBath'] + dataset['FullBath'] + (1/2) * dataset['HalfBath']
    dataset['HasPool'] = dataset['PoolArea'].apply(lambda x : 1 if x>0 else 0)
    dataset['Has2ndFloor'] = dataset['2ndFlrSF'].apply(lambda x : 1 if x>0 else 0)
    dataset['HasGarage'] = dataset['GarageArea'].apply(lambda x : 1 if x>0 else 0)


# Then 'Id' feature can be deleted as it doesn't give us any information and it's not needed.

# In[ ]:


train_df = train_df.drop(['Id'], axis=1)


# ## Converting Categorical Features to Numerical

# There are some categorical features which their categories are actually ordinal so it can be a good idea to convert them to numerical features. For instance "ExerQual" feature has values below :<br>
# &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Ex  &emsp; Excellent<br>
# &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Gd	 &emsp; Good<br>
# &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; TA	 &emsp; Average/Typical<br>
# &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Fa	 &emsp; Fair<br>
# &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Po  &emsp; Poor<br>
# so they can be mapped to numbers from 1 to 5 to conserve their ordinal quality.

# In[ ]:


LotShape_mapping = {"Reg": 1, "IR1": 2, "IR2": 3, "IR3": 4}
LandSlope_mapping = {"Gtl": 1, "Mod": 2, "Sev": 3}
ExterQual_mapping = {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5}
ExterCond_mapping = {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5}
BsmtQual_mapping = {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5, "NA": 6}
BsmtCond_mapping = {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5, "NA": 6}
BsmtExposure_mapping = {"Gd": 1, "Av": 2, "Mn": 3, "No": 4, "NA": 5}
BsmtFinType1_mapping = {"GLQ": 1, "ALQ": 2, "BLQ": 3, "Rec": 4, "LwQ": 5, "Unf": 6, "NA": 7}
BsmtFinType2_mapping = {"GLQ": 1, "ALQ": 2, "BLQ": 3, "Rec": 4, "LwQ": 5, "Unf": 6, "NA": 7}
HeatingQC_mapping = {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5}
CentralAir_mapping = {"N": 0, "Y": 1}
KitchenQual_mapping = {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5}
FireplaceQu_mapping = {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5}
GarageQual_mapping = {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5}
GarageCond_mapping = {"Ex": 1, "Gd": 2, "TA": 3, "Fa": 4, "Po": 5}
PavedDrive_mapping = {"Y": 1, "P": 2, "N": 3}

for dataset in [train_df, test_df]:
    dataset['LotShape'] = dataset['LotShape'].map(LotShape_mapping)
    dataset['LandSlope'] = dataset['LandSlope'].map(LandSlope_mapping)
    dataset['ExterQual'] = dataset['ExterQual'].map(ExterQual_mapping)
    dataset['ExterCond'] = dataset['ExterCond'].map(ExterCond_mapping)
    dataset['BsmtQual'] = dataset['BsmtQual'].map(BsmtQual_mapping)    
    dataset['BsmtCond'] = dataset['BsmtCond'].map(BsmtCond_mapping)    
    dataset['BsmtExposure'] = dataset['BsmtExposure'].map(BsmtExposure_mapping)    
    dataset['BsmtFinType1'] = dataset['BsmtFinType1'].map(BsmtFinType1_mapping)    
    dataset['BsmtFinType2'] = dataset['BsmtFinType2'].map(BsmtFinType2_mapping)    
    dataset['HeatingQC'] = dataset['HeatingQC'].map(HeatingQC_mapping) 
    dataset['CentralAir'] = dataset['CentralAir'].map(CentralAir_mapping)        
    dataset['KitchenQual'] = dataset['KitchenQual'].map(KitchenQual_mapping)             
    dataset['GarageQual'] = dataset['GarageQual'].map(GarageQual_mapping)        
    dataset['GarageCond'] = dataset['GarageCond'].map(GarageCond_mapping)        
    dataset['PavedDrive'] = dataset['PavedDrive'].map(PavedDrive_mapping)        


# # 3) Feature Selection

# ## Selecting Numerical Features Using Corrolation

# Below you can see the heatmap of corrolation matrix which shows how corrolated each pair of numerical features are. This can asist us in filtering some features for our model.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

corr_matrix = train_df.corr()
f, ax = plt.subplots(figsize=(25, 20))
sns.heatmap(corr_matrix, vmax=.8, cmap="Blues", square=True)


# We need to know how much each feature is useful for us to predict prices which means it should be corrolated with 'SalePrice'. After calculating corrolations we gain an insight of the most important features which we should use them in our model.

# In[ ]:


import numpy as np

n_largest_top = 31
n_smallest_top = 15
top_large_corr = corr_matrix.nlargest(n_largest_top, 'SalePrice')['SalePrice']
top_small_corr = corr_matrix.nsmallest(n_smallest_top, 'SalePrice')['SalePrice']
print("Top Largest Corrolations :") 
print(top_large_corr)
print("____________________________")
print("Top Smallest Corrolations :") 
print(top_small_corr)
num_attrs = np.append(top_large_corr.index.values, top_small_corr.index.values, axis=0)


# I chose the features which are strongly corrolated with 'SalePrice' as numerical attributed which will be used in future model.

# In[ ]:


cm = train_df[num_attrs].corr()
f, ax = plt.subplots(figsize=(20, 15))
hm = sns.heatmap(cm, cmap="Blues", annot=True, square=True, fmt='.2f', yticklabels=num_attrs, annot_kws={'size': 8}, xticklabels=num_attrs)


# ## Selecting Categorical Features

# Now it's time to select useful categorical features for our model.

# In[ ]:


train_df.describe(include=['O'])


# In[ ]:


for attr in train_df.select_dtypes(include='object'):
    print(train_df[[attr, "SalePrice"]].groupby([attr], as_index=False).mean().sort_values(by='SalePrice', ascending=False))
    print("\n ________________________ \n")


# In[ ]:


cat_attrs =["MSZoning","LandContour", "LotConfig", "Neighborhood", "Condition1", "Condition2", "HouseStyle", "RoofStyle", 
            "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating", "Electrical", "Functional", 
            "GarageType", "GarageFinish", "SaleType", "SaleCondition"]


# # 4) More Data Preparation

# ## Deleting Outliers

# Now it's time to delete outliers in our data as they can degrade our model and prediction. Below we plot the three most corrolated features with 'SalePrice' and if there exist any outlier we will remove them.

# In[ ]:


for var in ['OverallQual', 'SF', 'GrLivArea']:
    data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), label= 'Affect of ' + var + ' on SalePrice')


# In[ ]:


drop_indexes = train_df.sort_values(by='SF', ascending=False)[:2].index
train_df = train_df.drop(drop_indexes)


# In[ ]:


for var in ['OverallQual', 'SF', 'GrLivArea']:
    data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), label= 'Affect of ' + var + ' on SalePrice')


# ## Log Transform

# In[ ]:


from scipy import stats

print(train_df['SalePrice'].describe())
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.distplot(train_df['SalePrice'].dropna() , fit=stats.norm);
plt.subplot(1,2,2)
prob=stats.probplot(train_df['SalePrice'].dropna(), plot=plt)


# You can observe that there is long tail of outlying properties with high sale prices. This causes to biase the mean much higher than the median. For normalizing 'SalePrice' we use log transform.

# In[ ]:


import math

train_df['SalePrice'] = [np.log(x) for x in train_df['SalePrice']]

print(train_df['SalePrice'].describe())
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.distplot(train_df['SalePrice'].dropna() , fit=stats.norm);
plt.subplot(1,2,2)
prob=stats.probplot(train_df['SalePrice'].dropna(), plot=plt)


# # 5) Data Transformation

# ## Transformation Pipelines

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attr_names):
        self.attr_names = attr_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attr_names].values


# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer

class LabelBinarizerPipelineFriendly(MultiLabelBinarizer):
    def fit(self, X, y=None):
        super(LabelBinarizerPipelineFriendly,self).fit(X)
    def transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)
    def fit_transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)


# Pipeline for numerical features which aplies an imputer to put median of each feature for instances which doesn't have value for that feature and after that standardizing features using standard scaler.

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_data_attrs = [x for x in num_attrs if not x=='SalePrice']
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_data_attrs)),
    ('imputer',  SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])


# Another pipeline for categorical features which first use an imputer to replace missing values of features which the most frequent value and then applies a label binarizer.

# In[ ]:


cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attrs)),
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('binarizer', LabelBinarizerPipelineFriendly()),
])


# And the full pipeline for transforming our data which consists of the two pipelines you see above.

# In[ ]:


from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])


# ## Deleting more outliers

# In[ ]:


def find_outliers(model, X, y, sigma=3):
    y_pred = pd.Series(model.predict(X), index=y.index)
        
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    z = (resid - mean_resid)/std_resid    
    outliers = z[abs(z)>sigma].index
    return outliers


# In[ ]:


from sklearn.linear_model import Ridge

train_data = train_df.drop(['SalePrice'], axis=1)
train_labels = train_df['SalePrice']

model = Ridge()
train_prepared = full_pipeline.fit_transform(train_data)
model.fit(train_prepared, train_labels)
outliers = find_outliers(model, train_prepared, train_labels)
train_df = train_df.drop(np.asarray(outliers) )


# ## Preparing Train Data for Prediction

# In[ ]:


train_data = train_df.drop(['SalePrice'], axis=1)
train_labels = train_df['SalePrice']


# In[ ]:


train_prepared = full_pipeline.fit_transform(train_data)


# # 6) Testing Models

# ## Cross Validation

# In[ ]:


from sklearn.model_selection import KFold, cross_val_score, train_test_split

def rms_score_cv(model):
    rmse= np.sqrt(-cross_val_score(model, train_prepared, train_labels, scoring="neg_mean_squared_error", cv = 5))
    return(rmse.mean())


# ## Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
print("Linear Regression : ", rms_score_cv(linear_reg) )


# ## Ridge Regression

# In[ ]:


from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.6, solver='cholesky')
rms_score_cv(ridge)


# ## Lasso Regression

# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
# lasso = Lasso(alpha =0.0005, random_state=1)
rms_score_cv(lasso)


# ## Elastic Net Regression

# In[ ]:


from sklearn.linear_model import ElasticNet

e_net = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
# e_net = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
rms_score_cv(e_net)


# ## Ransac

# In[ ]:


from sklearn.linear_model import RANSACRegressor

ransac = RANSACRegressor()
rms_score_cv(ransac)


# ## SVM Regression

# Linear SVM Regression

# In[ ]:


from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon=0.01)
rms_score_cv(svm_reg)


# Non-linear SVM Regression

# In[ ]:


from sklearn.svm import SVR

svm_poly_reg = SVR(kernel='poly', degree=2, C=100, epsilon=0.1)
rms_score_cv(svm_poly_reg)


# ## Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

decision_tree = DecisionTreeRegressor(max_depth=7, random_state=42)
rms_score_cv(decision_tree)


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rand_forest = RandomForestRegressor(max_depth=15, random_state=42)
rms_score_cv(rand_forest)


# ## Ensemble Learning

# ### Voting

# In[ ]:


from sklearn.ensemble import VotingRegressor

voting_reg = VotingRegressor(
        estimators=[('e_net', e_net), ('ridge', ridge), ('svm_poly_reg', svm_poly_reg), ('rand_forest', rand_forest)])
rms_score_cv(voting_reg)


# ### Bagging

# In[ ]:


from sklearn.ensemble import BaggingRegressor

bagging_reg = BaggingRegressor(e_net, n_estimators=300, bootstrap=True, n_jobs=-1)
# rms_score_cv(bagging_reg)


# ### Pasting

# In[ ]:


from sklearn.ensemble import BaggingRegressor

bagging_reg = BaggingRegressor(e_net, n_estimators=500, bootstrap=False, n_jobs=-1)
# rms_score_cv(bagging_reg)


# ### Extra Tree

# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor

extra_tree = ExtraTreesRegressor(random_state=42)
rms_score_cv(extra_tree)


# ### AdaBoost

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor

ada_boost = AdaBoostRegressor(e_net, n_estimators=500, learning_rate=0.5)
# rms_score_cv(ada_boost)


# ### Gradient Boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(max_depth=5, n_estimators=15, learning_rate=0.6)
rms_score_cv(gbr)


# ### Stacking

# 1)

# In[ ]:


estimators = [ e_net, ridge, svm_poly_reg, rand_forest, extra_tree]
slice = int((0.8)*(len(train_prepared)))
train_d = train_prepared[:slice]
train_l = train_labels[:slice]
val_d = train_prepared[slice: ]
val_l = train_labels[slice:]


# In[ ]:


for estimator in estimators:
  estimator.fit(train_d, train_l)


# In[ ]:


val_preds = np.empty((len(val_d), len(estimators)), dtype=np.float32)
for index, estimator in enumerate(estimators):
    val_preds[:, index] = model.predict(val_d)
blender = RandomForestRegressor(n_estimators=200, oob_score=True, random_state=42)
blender.fit(val_preds, val_l)
blender.oob_score_


# 2)

# In[ ]:


from sklearn.base import RegressorMixin
from sklearn.base import clone
from sklearn.model_selection import KFold

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
    def fit(self, X, y=None):
        
        X = X
        y = y.values
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
    def get_metafeatures(self, X):
        return np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


# In[ ]:


stacked_averaged_models = StackingAveragedModels(base_models = (e_net, ridge, svm_poly_reg, rand_forest), meta_model = lasso)

# rms_score_cv(stacked_averaged_models)


# # 7)Predict Test Data

# In[ ]:


el_net = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
# stacked_averaged_models = StackingAveragedModels(base_models = (e_net, ridge, svm_poly_reg, rand_forest, extra_tree),meta_model = lasso)

el_net.fit(train_prepared, train_labels)


# In[ ]:


test_prepared = full_pipeline.transform(test_df)


# In[ ]:


pred = el_net.predict(test_prepared)
predSalePrice = [np.exp(x) for x in pred]


# In[ ]:



output = pd.DataFrame({
    'Id' : test_df['Id'],
    'SalePrice' : predSalePrice
})


# In[ ]:


output.head() 


# In[ ]:


output.to_csv('prediction.csv', index=False)

