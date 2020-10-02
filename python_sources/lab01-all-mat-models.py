# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# 1) Import all Library that will be used
#%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import statsmodels.formula.api as smf

from scipy import stats

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import linear_model, svm, gaussian_process
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score# 1) Data treatment and cleaning

#df_train_original = pd.read_csv('train-House.csv')
df_train_original = pd.read_csv('../input/train.csv')
df_test_original = pd.read_csv('../input/test.csv')

df_train = df_train_original
df_test = df_test_original

all_data = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],
                      df_test.loc[:,'MSSubClass':'SaleCondition']))

# Get_Dummies para transformar categoricos em Numéricos 
all_data = pd.get_dummies(all_data)

# Substitui os campos nulos pelas médias da coluna em questão
all_data = all_data.fillna(all_data.mean())
#all_data = all_data.fillna(0)

#Cria Matriz X_train utilizando a Matriz com todos os dados all_data: do inicio da matriz (:) até o fim  da matriz df_train.shape[0]
X_train = all_data[:df_train.shape[0]]

#Cria Matriz X_test utilizando a Matriz com todos os dados all_data: a partir do último registro matriz df_train.shape[0], ou seja, todos os registros que não estiverem em df_train
X_test = all_data[df_train.shape[0]:]

# Cria o y, ou seja, o que será previsto, apenas com o campo "Survived"
y = df_train.SalePrice

#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(df_train.values)
    rmse= np.sqrt(-cross_val_score(model, df_train.values, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
    
# 2) Aplly Gradient Boost Model
from sklearn.ensemble import GradientBoostingRegressor
import statsmodels.formula.api as smf
from sklearn.preprocessing import scale
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y)

yhat_train = gbr.predict(X_train)
yhat_test = gbr.predict(X_test)

yhat_train_round = np.round(yhat_train)
yhat_train_round = yhat_train_round.astype(int)
print ('Accuracy: ', accuracy_score(y, yhat_train_round))
print(yhat_train_round)

yhat_test_round = np.round(yhat_test)
yhat_test_round = yhat_test_round.astype(int)

# Gerando um CSV para o resultado obtido com o Gradiente Descendente:
df_test_gbr = df_test
df_test_gbr['SalePrice'] = yhat_test_round
df_test_gbr = df_test_gbr.drop(df_test_gbr.columns[1:80], axis=1)
df_test_gbr.to_csv('House_GBR.csv', index = False)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y)

LR_yhat_train = logreg.predict(X_train)
LR_yhat_test = logreg.predict(X_test)

yhat_LR = LR_yhat_test
print (yhat_LR)

# Gerando um CSV para o resultado obtido com o Gradiente Descendente:
df_test_LR = df_test
df_test_LR['SalePrice'] = yhat_LR
df_test_LR = df_test_LR.drop(df_test_LR.columns[1:80], axis=1)
df_test_LR.to_csv('House_LR.csv', index = False)

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y)

random_forest_train = random_forest.predict(X_train)
random_forest_test = random_forest.predict(X_test)

print ('# # # # Esse é o yhat com o método Random Forest # # # #')
print (random_forest_test)

# Gerando um CSV para o resultado obtido com o Gradiente Descendente:
df_test_RF = df_test
df_test_RF['SalePrice'] = random_forest_test
df_test_RF = df_test_RF.drop(df_test_RF.columns[1:80], axis=1)
df_test_RF.to_csv('House_RF.csv', index = False)

from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train, y)

xgb_train = xgb.predict(X_train)
xgb_test = xgb.predict(X_test)

print ('# # # # Esse é o yhat com o método Xgboost # # # #')
print (xgb_test)

# Gerando um CSV para o resultado obtido com o Gradiente Descendente:
df_test_XGB = df_test
df_test_XGB['SalePrice'] = xgb_test
df_test_XGB = df_test_XGB.drop(df_test_XGB.columns[1:80], axis=1)
df_test_XGB.to_csv('House_XGB.csv', index = False)

knn = KNeighborsClassifier()
knn.fit(X_train, y)

knn_train = knn.predict(X_train)
knn_test = knn.predict(X_test)

print ('# # # # Esse é o yhat com o método KNeighbors # # # #')
print (knn_test)

# Gerando um CSV para o resultado obtido com o Gradiente Descendente:
df_test_KNN = df_test
df_test_KNN['SalePrice'] = knn_test
df_test_KNN = df_test_KNN.drop(df_test_KNN.columns[1:80], axis=1)
df_test_KNN.to_csv('House_KNN.csv', index = False)

# 7) Aplly SVC Model

svc = SVC(probability=True)
svc.fit(X_train, y)

svc_train = svc.predict(X_train)
svc_test = svc.predict(X_test)

print (svc_test)

# Gerando um CSV para o resultado obtido com o Gradiente Descendente:
df_test_SVC = df_test
df_test_SVC['SalePrice'] = svc_test
df_test_SVC = df_test_SVC.drop(df_test_SVC.columns[1:80], axis=1)
df_test_SVC.to_csv('House_SVC.csv', index = False)

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y)

dtc_train = dtc.predict(X_train)
dtc_test = dtc.predict(X_test)

print ('# # # # Esse é o yhat com o método DecisionTree # # # #')
print (dtc_test)

#Gerando um CSV para o resultado obtido com o Gradiente Descendente:
df_test_DTC = df_test
df_test_DTC['SalePrice'] = dtc_test
df_test_DTC = df_test_DTC.drop(df_test_DTC.columns[1:80], axis=1)
df_test_DTC.to_csv('House_DTC.csv', index = False)

nn = MLPClassifier(hidden_layer_sizes=(50,50,25))
nn.fit(X_train, y)

nn_train = nn.predict(X_train)
nn_test = nn.predict(X_test)

print ('# # # # Esse é o yhat com o método Neural # # # #')
print (nn_test)

#Gerando um CSV para o resultado obtido com o Gradiente Descendente:
df_test_NN = df_test
df_test_NN['SalePrice'] = nn_test
df_test_NN = df_test_NN.drop(df_test_NN.columns[1:80], axis=1)
df_test_NN.to_csv('House_NN.csv', index = False)