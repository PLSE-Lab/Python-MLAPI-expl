import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

from scipy.stats import norm, skew


######################################################################
## Import
######################################################################
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

######################################################################
## Some Data Visualizations and Data Preprocessing
######################################################################
##### Detect some outliers on the most pertinent variable
# plt.scatter(x = train['GrLivArea'], y = train['SalePrice'])
# plt.show()

# Remove the 2 abnormal points
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

##### Log normalisation interest
# def plot_dist(col):
#     sns.distplot(col, fit=norm);

#     (mu, sigma) = norm.fit(col)

#     plt.legend(['($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)],
#                 loc='best')

#     fig = plt.figure()
#     res = stats.probplot(col, plot=plt)
#     plt.show()

# plot_dist(train['SalePrice'])
# plot_dist(np.log1p(train["SalePrice"]))


######################################################################
##### Feature Engineering
######################################################################
# Apply log norm to dependedant variable according to what we showed previously
train["SalePrice"] = np.log1p(train["SalePrice"])

# Concatenate both the train and test features for the preprocessing excluding the dependant variable
y_train = train['SalePrice'].values
data = pd.concat((train, test)).reset_index(drop=True)
data.drop(['SalePrice'], axis=1, inplace=True)

### Missing Data
# Here to elaborate an efficient missing data replacement we will have to handle the semantic behind each variable

# For these categories NA means no such entity or caracteristic in the corresponding house
# I think it is a fair choice to add another level to each of these variables
data[["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", 
      "GarageType", "GarageFinish", "GarageQual", "GarageCond",
      "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
      "BsmtFinType2", "MasVnrType", "MSSubClass"]] = data[["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", 
                                                           "GarageType", "GarageFinish", "GarageQual", "GarageCond",
                                                           "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
                                                           "BsmtFinType2", "MasVnrType", "MSSubClass"]].fillna("None")

# Same logic for numerical variables
data[["MasVnrArea", "GarageYrBlt", "GarageArea", "GarageCars",
      "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", 
      "BsmtFullBath", "BsmtHalfBath"]] = data[["MasVnrArea", "GarageYrBlt", "GarageArea", "GarageCars",
                                               "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", 
                                               "BsmtFullBath", "BsmtHalfBath"]].fillna(0)

# For these variables we need to handle real missing informations so I choose to fill with the most common value of each variable
for c in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType'):
    data[c] = data[c].fillna(data[c].mode()[0])

# According to data description
data["Functional"] = data["Functional"].fillna("Typ")

# LotFrontage is literally the linear feet of street connected to property, it seems clever to use the variable Neighborhood
# and get the median of all the LotFrontage related to this value
data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

# This variable contains only one value, it won't help the model
data = data.drop(['Utilities'], axis=1)


# Here we use a LabelEncoder in order to transform levels into much simpler values
# I had trouble in this part to choose whether or not to keep ordinality for some variables
# This specific process gives me the best results
from sklearn.preprocessing import LabelEncoder
for c in ('Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold', 'MoSold', 
        'FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street'):
    lbl = LabelEncoder() 
    lbl.fit(list(data[c].values)) 
    data[c] = lbl.transform(list(data[c].values))

### Normalization and final process
# Back to the log normalization property I showed previously
# Use skewness of each variable to log normalize the variables which don't have a normal enough distribution
# Get all the numerical variables
numerical = data.dtypes[data.dtypes != "object"].index

# Get the variables that have a skewness superior or inferior to 0.75
skewed = data[numerical].apply(lambda x: skew(x.dropna()))
skewed = skewed[(abs(skewed) > 0.75)].index

# Apply the log normalisation to these features
data[skewed] = np.log1p(data[skewed])

# Change categorical features into one-hot encoding
data = pd.get_dummies(data)

# Remap into classic split
X_train = data[:train.shape[0]]
X_test = data[train.shape[0]:]
y = train["SalePrice"]

small_X = X_train.sample(frac=0.1)
small_y = y[y.index.isin(small_X.index)]



#####################################################################
## Model Selection and Evaluation
#####################################################################
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import cross_val_score

##### Some built-in functions
def rmse_cv(model):
    '''
    Used this function to get quickly the rmse score over a cv
    '''
    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return rmse

def xgbfit(alg, X, y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    '''
    Used this function to get a full report of each xgb model I was training
    '''
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X, label=y)
        warnings.filterwarnings("ignore")
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round = alg.get_params()['n_estimators'], nfold = cv_folds, metrics = 'rmse', early_stopping_rounds = early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(X, y, eval_metric="rmse")
        
    #Predict training set:
    dtrain_predictions = alg.predict(X)
        
    #Print model report:
    print("XGB Score : {:.4f} ({:.4f})".format(rmse_cv(alg).mean(), rmse_cv(alg).std()))
    
    fig = plt.figure()
    fig.set_size_inches(15, 5)
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

###################
###### Ridge
###################
model_ridge = RidgeCV(alphas = [10, 1, 0.1, 0.001, 0.0005, 0.0001],
                      scoring = "neg_mean_squared_error").fit(X_train, y)

# print("Ridge score: {:.4f} ({:.4f})".format(rmse_cv(model_ridge).mean(), rmse_cv(model_ridge).std()))

###################
###### Lasso
###################
model_lasso = LassoCV(alphas = [10, 1, 0.1, 0.001, 0.0005, 0.0001]).fit(X_train, y)

# print("Lasso score: {:.4f} ({:.4f})".format(rmse_cv(model_lasso).mean(), rmse_cv(model_lasso).std()))

###################
##### ElasticNet
###################
model_Enet = ElasticNetCV(l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
                          alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)

# print("ElasticNet score: {:.4f} ({:.4f})".format(rmse_cv(model_Enet).mean(), rmse_cv(model_Enet).std()))

###################
##### XGBoost
###################
# Trying default model
model_xgb = xgb.XGBRegressor(seed=42)
# xgbfit(model_xgb, X_train, y)

## Grid used
# param_test1 = {
#  'max_depth': np.arange(2, 10, 2)
#  'learning_rate': np.arange(0.04, 0.07, 0.01)
# }
# gsearch1 = GridSearchCV(estimator = xgb.XGBRegressor(n_estimators=2000, objective= 'reg:linear', nthread=4, seed=42), 
#                         param_grid = param_test1, scoring='neg_mean_squared_error', n_jobs=4, iid=False, cv=5)
# gsearch1.fit(small_X, small_y)
# print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
## max_depth = 4
## learning_rate = 0.05

# param_test2 = {
#  'subsample': np.arange(0.5, 1, 0.1),
#  'min_child_weight': range(1, 6, 1)
# }
# gsearch2 = GridSearchCV(estimator = xgb.XGBRegressor(n_estimators=2000, max_depth = 4, learning_rate = 0.05, 
#                                                      objective= 'reg:linear', nthread=4, seed=42), 
#                         param_grid = param_test1, scoring='neg_mean_squared_error', n_jobs=4, iid=False, cv=5)
# gsearch2.fit(small_X, small_y)
# print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
# subsample = 0.5
# min_child_weight = 2

# Could have improved this but too long to execute for my computer


model_xgb = xgb.XGBRegressor(colsample_bytree=0.46, gamma=0.047, 
                             learning_rate=0.05, max_depth=4, 
                             min_child_weight=2, n_estimators=2000,
                             subsample=0.5, silent=1, seed = 42, nthread = 4)

# xgbfit(model_xgb, X_train, y)



model_xgb.fit(X_train, y)
xgb_preds = np.expm1(model_xgb.predict(X_test))

#######################
##### Mix model
#######################
class MixModel(BaseEstimator, RegressorMixin, TransformerMixin):
    '''
    Here I built a function that will get as parameter a set of models already trained and 
    will calculate the mean of the predictions for each model
    '''
    def __init__(self, algs):
        self.algs = algs
        
    # Define clones of parameters models
    def fit(self, X, y):
        self.algs_ = [clone(x) for x in self.algs]
        
        # Train cloned base models
        for alg in self.algs_:
            alg.fit(X, y)

        return self
    
    # Average predictions of all cloned models
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.algs_
        ])
        return np.mean(predictions, axis=1) 



model_avg = MixModel(algs = (model_ridge, model_lasso, model_Enet, model_xgb))
score = rmse_cv(model_avg)
# print("\nAveraged base algs score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

model_avg.fit(X_train, y)
preds = np.expm1(model_avg.predict(X_test))


# Final submit
solution = pd.DataFrame({"Id":test_ID, "SalePrice":preds})
solution.to_csv("pred.csv", index = False)