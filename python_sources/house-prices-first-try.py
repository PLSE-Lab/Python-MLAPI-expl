import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy import stats
from sklearn import linear_model
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings('ignore')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#Calculating results
def calc_results(train,labels):
    results={}
    
    def score_model(classifier):
        cv = KFold(n_splits=5,shuffle=True,random_state=45)
        scorer = make_scorer(mean_squared_error, greater_is_better = False)
        rmse_score = np.sqrt(-cross_val_score(classifier, train, labels, scoring=scorer, cv=cv))
        rmse_mean = rmse_score.mean()
        return rmse_mean
    
    classifier = linear_model.LinearRegression()
    results['LinearReg'] = score_model(classifier)
    
    classifier = linear_model.Ridge()
    results['Ridge'] = score_model(classifier)
    
    classifier = linear_model.BayesianRidge()
    results['BayesianRidge'] = score_model(classifier)
    
    classifier = linear_model.HuberRegressor()
    results['HuberRegressor'] = score_model(classifier)
    
    classifier = linear_model.Lasso(alpha=1e-4)
    results['Lasso'] = score_model(classifier)
    
    classifier = BaggingRegressor()
    results['BaggingReg'] = score_model(classifier)
    
    classifier = RandomForestRegressor()
    results['RandomForest'] = score_model(classifier)
    
    classifier = AdaBoostRegressor()
    results['AdaBoost'] = score_model(classifier)
    
    classifier = svm.SVR()
    results['SVM_R'] = score_model(classifier)
    
    classifier = svm.SVR(kernel='linear')
    results['SVM_Lin'] = score_model(classifier)
    
    results = pd.DataFrame.from_dict(results,orient='index')
    results.columns=["RMS"]
    results = results.sort(columns=["RMS"],ascending=False)
    print(results)
    return results

train = pd.read_csv('../input/train.csv')
labels = train['SalePrice']
test = pd.read_csv('../input/test.csv')
data = pd.concat([train,test],ignore_index=True)
data = data.drop('SalePrice',1)
ids= test['Id']

#Missing values
mv = pd.isnull(data).sum()

#Remove files with too many mv
data = data.drop('Id',1)
data = data.drop('Alley',1)
data = data.drop('Fence',1)
data = data.drop('FireplaceQu',1)
data = data.drop('MiscFeature',1)
data = data.drop('PoolQC',1)

all_col = data.columns.values
non_categ = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1','BsmtFinSF2',
            'BsmtUnfSF', 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea'
            'GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
            'ScreenPorch','PoolArea','MiscVal']
categ = [value for value in all_col if value not in non_categ]

#Convert
data = pd.get_dummies(data)

#Filling mv
imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
data = imp.fit_transform(data)

#Log Transformation
data = np.log(data)
labels = np.log(labels)
#non infinty
data[data==-np.inf] = 0

#splitting data
train = data[:1460]
test = data[1460:]
#Det best classifier
calc_results(train,labels)
#Prediction
classifier = linear_model.LinearRegression()
classifier.fit(train,labels)
pred_lasso = classifier.predict(test)
pred_lasso = np.exp(pred_lasso)
pred_lasso = np.round(pred_lasso,2)

sub = pd.DataFrame({'Id':ids,'SalePrice': pred_lasso})
sub.to_csv('sub.csv', index=False)