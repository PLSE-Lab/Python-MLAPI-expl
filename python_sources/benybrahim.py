import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score,train_test_split, learning_curve

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingClassifier

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
                      
# Month passed since 01/2006
all_data["Month"] = all_data.MoSold + 12*(all_data.YrSold - 2006) - 1
# convert the ordinal variable MSSubClass from int64 to object
all_data.loc[:,'MSSubClass'] = all_data.loc[:,'MSSubClass'].astype(str)
# price rounded to the nearset thousand, numerical stability
train.SalePrice = train.SalePrice/1000

all_data = pd.get_dummies(all_data)
#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())
#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

fit = SelectKBest(mutual_info_regression).fit(X_train, y)
scores = sorted(enumerate(fit.scores_), key= lambda x: x[1],reverse=True)
# find the indexes of the optimal features using the percentile
def indexes(p):
    return ([index for index,score in scores if score > np.nanpercentile(fit.scores_,p)])
    
# Create linear regression object
regr = LinearRegression()
def mse(p):
    xtrain, xtest, ytrain, ytest = train_test_split(X_train.iloc[:,indexes(p)], y, test_size=0.2, random_state=42)
    regr.fit(xtrain, np.log(ytrain+1))
    return (np.mean((regr.predict(xtest) - np.log(ytest+1)) ** 2))
mses = [mse(p) for p in range(40,95,5)]
optimal_indexes = [indexes(p) for p in range(40,95,5)][np.argmin(mses)]

X_train = X_train.iloc[:,optimal_indexes]
X_test = X_test.iloc[:,optimal_indexes]
ridge= LinearRegression(normalize= True)
ridge.fit(X_train, np.log(y+1))
# save to file to make a submission
p = np.expm1(ridge.predict(X_test))
solution = pd.DataFrame({"id":test.Id, "SalePrice":p*1000}, columns=['id', 'SalePrice'])
solution.to_csv("regression_sol.csv", index = False)