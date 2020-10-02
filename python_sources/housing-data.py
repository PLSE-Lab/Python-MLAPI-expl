pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer as Imp
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor as xgb
from fancyimpute import KNN    


df=pd.read_csv("../input/train.csv")
df = df.drop(['Id'], axis=1)

X_test=pd.read_csv("../input/test.csv")
ID = X_test['Id']
X_test = X_test.drop(['Id'], axis= 1)




#extractinglabel, dividing cats and nums
y_train = df['SalePrice']
X_train = df.drop(['SalePrice'], axis=1)
Xtrain_num = X_train.select_dtypes(include = [np.number])
Xtrain_cat = X_train.select_dtypes(exclude = [np.number])

nullvalues = Xtrain_num.isnull().sum()/X_test.shape[0]
nullvalues.sort_values()
Xtrain_num.describe().transpose().sort_values(['count'])

#Overriding numeric imputer 
#Xtrain_num.loc[:,'LotFrontage'].fillna(Xtrain_num['LotFrontage'].median(), inplace=True)
#Xtrain_num.loc[:,'GarageYrBlt'].fillna(Xtrain_num['GarageYrBlt'].median(), inplace=True)
#Xtrain_num.loc[:,'MasVnrArea'].fillna(0, inplace=True)



#Imputing other num vars and preparing for 
Xtrain_num_imputed=pd.DataFrame(data=KNN(k=5).complete(Xtrain_num), columns=Xtrain_num.columns, index=Xtrain_num.index)
#scaling num vars 
sc = scaler()
ScalerFitted = sc.fit(Xtrain_num_imputed)
Xtrain_num_imputed_scaled = pd.DataFrame(sc.transform(Xtrain_num_imputed),columns = Xtrain_num.columns)
Xtrain_onehots = pd.get_dummies(Xtrain_cat, drop_first = True)
X_train_prepped = pd.concat([Xtrain_onehots, Xtrain_num_imputed_scaled], axis=1)
X_train_prepped.shape

#log of label
labelcol = np.log(y_train)


### testdata
Xtest_num = X_test.select_dtypes(include = [np.number])
Xtest_cat = X_test.select_dtypes(exclude = [np.number])
#Xtest_num.loc[:,'LotFrontage'].fillna(Xtrain_num['LotFrontage'].median(), inplace=True)
#Xtest_num.loc[:,'GarageYrBlt'].fillna(Xtrain_num['GarageYrBlt'].median(), inplace=True)
#Xtest_num.loc[:,'MasVnrArea'].fillna(0, inplace=True)

Xtest_num_imputed=pd.DataFrame(data=KNN(k=5).complete(Xtest_num), columns=Xtest_num.columns, index=Xtest_num.index)
Xtest_num_imputed = NumImputer.transform(Xtest_num)
Xtest_num_imputed_scaled = pd.DataFrame(sc.transform(Xtest_num_imputed),columns = Xtest_num.columns)
Xtest_onehots = pd.get_dummies(Xtest_cat, drop_first = True)
X_test_prepped = pd.concat([Xtest_onehots, Xtest_num_imputed_scaled], axis=1)
missing_cols = set( X_train_prepped) - set( X_test_prepped)
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    X_test_prepped[c] = 0
    
X_test_prepped.shape
X_test_prepped = X_test_prepped[X_train_prepped.columns]

#Split train data in test and train

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train_prepped, labelcol, test_size=0.3, random_state=42)


    

regressor = xgb(n_estimators=500)
regressor.fit(X_train1,np.ravel(y_train1))
y_pred = regressor.predict(X_test1)
print("Test mse= ", regressor.score(X_test1, y_test1))


sub = regressor.predict(X_test_prepped)
subexp = pd.DataFrame(np.exp(sub))

submission = pd.read_csv("../input/sample_submission.csv")

submission['SalePrice'] = subexp
submission
submission.to_csv('submission_data_G.csv', index =False)


print('FINI')





























