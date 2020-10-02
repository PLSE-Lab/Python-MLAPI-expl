import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import math as m
from sklearn.metrics import mean_squared_error
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train.AREA = train.AREA.replace({"Chrmpet": "Chrompt","Chormpet": "Chrompt","Chrompet": "Chrompt"})
train.AREA = train.AREA.replace({"T Nagar": "TNagar","Karapakam": "Karapakkam","KK Nagar": "KKNagar"})
train.AREA = train.AREA.replace({"Ann Nagar": "Anna Nagar", "Ana Nagar": "Anna Nagar", "Velchery": "Velachery","Adyr":"Adyar"})
#sns.countplot(x = 'AREA',data = train).set_title("plot")
#plt.show()
dummies = pd.get_dummies(train.AREA)
train = train.drop(['AREA'],axis=1)
train = train.join(dummies)

train["INT_SQFT"] = pd.DataFrame(np.log1p(train["INT_SQFT"]))

#train = train[~(train['DIST_MAINROAD']<1)]
#train = train[~(train['DIST_MAINROAD']>198.5)]


#Now changing the date columns 
#train.DATE_SALE.dtypes
train["DATE_SALE"] = pd.to_datetime(train["DATE_SALE"])
train["DATE_BUILD"] = pd.to_datetime(train["DATE_BUILD"])
train["time"] = train["DATE_SALE"]-train["DATE_BUILD"]
train["time"] = train["time"].dt.days
# now drop the date columns 

list1 = ['DATE_BUILD','DATE_SALE']
train = train.drop(list1,axis=1)


# Spelling error in the SALE_COND
train.SALE_COND = train.SALE_COND.replace({"Ab Normal": "AbNormal", "Adj Land" :"AdjLand", "PartiaLl" : "Partial","Partiall":"Partial"})
#sns.countplot(x = 'SALE_COND',data = train).set_title("plot")
#plt.show()

dummies = pd.get_dummies(train.SALE_COND)
train = train.drop(['SALE_COND'],axis=1)
train = train.join(dummies)

#encoding in parking facility 
train.PARK_FACIL = train.PARK_FACIL.replace({"Noo":"No"})

#sns.countplot(x = 'PARK_FACIL',data = train).set_title("plot")
#plt.show()
dummies = pd.get_dummies(train.PARK_FACIL)
train = train.drop(['PARK_FACIL'],axis=1)
train = train.join(dummies)


#Encoding in buildtype
train.BUILDTYPE = train.BUILDTYPE.replace({"Commercial":"Comercial", "Other":"Others"})
#sns.countplot(x = 'BUILDTYPE',data = train).set_title("plot")
#plt.show()
dummies = pd.get_dummies(train.BUILDTYPE)
train = train.drop(['BUILDTYPE'],axis=1)
train = train.join(dummies)


# label encoding in UTILITY_AVAIL
train.UTILITY_AVAIL = train.UTILITY_AVAIL.replace({"All Pub":"AllPub"})


train.UTILITY_AVAIL = train.UTILITY_AVAIL.replace({"AllPub": 4, "ELO": 1, "NoSeWa": 2, "NoSewr ": 3})



# encoding the STREET
train.STREET = train.STREET.replace({"Pavd": "Paved", "NoAccess": "No Access"})


train.STREET = train.STREET.replace({"Gravel": 3, "No Access": 1, "Paved": 2})


#MZZONE
dummies = pd.get_dummies(train.MZZONE)
train = train.drop(['MZZONE'],axis=1)
train = train.join(dummies)

# filling the null values 
train['N_BEDROOM'].fillna(0,inplace=True)
train['N_BATHROOM'].fillna(0,inplace=True)

train['QS_OVERALL'].fillna(0,inplace=True)



train['REG_FEE'] = (train['REG_FEE'] - train.REG_FEE.mean())/(train.REG_FEE.std())

train['COMMIS'] = (train['COMMIS'] - train.COMMIS.mean())/(train.COMMIS.std())




list1 = ['PRT_ID']
train = train.drop(list1,axis = 1)
featuers = train.columns[train.columns != 'SALES_PRICE']
#y_test = test
#y_test = y_test.drop(list1,axis =1)

X = train[featuers]
y = train['SALES_PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2)
params = {'subsample':[0.05,0.5,0.7,1],'learning_rate':[.05,0.1, 0.2,0.3],'max_depth': [2,3,4,5],'min_samples_leaf':[8,10,12,14],'min_samples_split':[2,4,6,10]}

from sklearn.ensemble import GradientBoostingRegressor

reg1 = GradientBoostingRegressor(max_features='sqrt',loss='huber',n_estimators=400)

'''GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='huber', max_depth=3,
             max_features=sqrt, max_leaf_nodes=None,
             min_impurity_split=1e-07, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=500, presort='auto', random_state=None,
             subsample=1.0, verbose=0, warm_start=False)'''



from sklearn import preprocessing
import xgboost 
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
'''xgb = xgboost.XGBRegressor(nthread=4, n_estimators=30000, verbose_eval=True)'''

'''params =  {base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.7, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=4, min_child_weight=4, missing=None, n_estimators=400,
       n_jobs=1, nthread=4, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=0.6, verbose_eval=True}'''

grid = GridSearchCV(reg1, params)
grid.fit(X_train, y_train)

print("R2:", m.sqrt(mean_squared_error(grid.best_estimator_.predict(X_test),y_test)))
print("best estimator:", grid.best_estimator_)



'''xgb.fit(X,y)
predicted = xgb.predict(y_test)

sub2 = pd.DataFrame({'PRT_ID':test.PRT_ID, 'SALES_PRICE':predicted})
sub2 = sub2[['PRT_ID', 'SALES_PRICE']]
sub2.to_csv('xgboost1_AV.csv', index=False)'''