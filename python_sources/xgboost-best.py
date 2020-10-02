#Linear regression with L1 and L2 penalties
#Random Forest
#XGBoost tree or something similar (there's a lot of open source packages for this)

#Try a dimensionality reduction method on all, or a subset of your data (https://scikit-learn.org/stable/modules/unsupervised_reduction.html)
#Try a feature selection method.

#Try one regression method that we *have not* covered in class.

#Put these in Kaggle Kernels(can be one big one if you want) but don't make them public until after the competition is over.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.grid_search import GridSearchCV
import xgboost as xgb
#from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
#from scipy import stats

def printresult(y_test, filename):
	ind = np.arange(1,len(y_test)+1,1)
	dataset = pd.DataFrame({'id':ind,'OverallScore':y_test})
	dataset.to_csv(filename, sep=',', index=False)

def cleandata(df):
    for col in df.columns:
        if (df[col].isnull().sum())/len(df) > 0.9:
            df.drop(col,inplace=True,axis=1)
    for column in list(df.columns[df.isnull().sum() > 0]):
        df[column].fillna(df[column].mean(), inplace=True)
    df.drop({'ids'}, inplace = True, axis = 1)
    df['erkey'] = df['erkey'].str.slice_replace(0, 3, '')
    df['erkey'] = pd.to_numeric(df['erkey'])
    return df

trainF = pd.read_csv("../input/trainFeatures.csv")
trainL = pd.read_csv("../input/trainLabels.csv")
testF = pd.read_csv("../input/testFeatures.csv")


trainData = cleandata(pd.merge(trainF, trainL, on='ids'))
X_train=trainData.iloc[:,0:47]
Y_train=trainData.iloc[:,48:49]

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = cleandata(testF)
X_test = np.array(X_test)


dtrain = xgb.DMatrix(data=X_train,label=Y_train)
dtest = xgb.DMatrix(X_test)
xgb_model = xgb.XGBClassifier()

# use the following code to select the best parameter and print out
# parameters = {'n_estimators': stats.randint(100, 600),
#               'learning_rate': stats.uniform(0.01, 0.05),
#               'subsample': stats.uniform(0.1, 0.5),
#               'max_depth': [4, 5, 6, 7, 8],
#               'colsample_bytree': stats.uniform(0.5, 0.49),
#               'min_child_weight': [1, 2, 3, 4, 5],
#               'gamma': stats.uniform(0, 5),
#               'lambda': stats.uniform(0, 5),
#               }
# xgb = xgb.XGBRegressor()
# xgb.fit(X_train, Y_train)
# clf = RandomizedSearchCV(xgb, param_distributions = parameters, n_iter = 500, scoring = 'neg_mean_squared_error', error_score = 0, n_jobs = -1,cv=5,refit=True)
# clf.fit(X_train, Y_train)
# print(clf.best_score_)
# print(clf.best_params_)

# The following parameter set is selected by randomized search cross validation
bestpara = {'colsample_bytree': 0.8424184017314718, 'gamma': 2.3679458450768136, 'lambda': 2.341013664612634, 'learning_rate': 0.05512077462049304, 'max_depth': 8, 'min_child_weight': 3, 'n_estimators': 476, 'subsample': 0.5400055981855248}
#params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,'max_depth': 5, 'alpha': 10}
xgb = xgb.XGBRegressor().set_params(**bestpara)
xgb.fit(X_train, Y_train)
xg_reg = xgb.predict(X_test)

printresult(xg_reg, "XGBoost.csv")
