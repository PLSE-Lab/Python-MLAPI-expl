# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

### working from https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# Any results you write to the current directory are saved as output.

### Load data
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")

### Merge data to do some pre-processing
All_Data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))


### We want log prices for logarithmic mean squared errors
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist() 

train["SalePrice"] = np.log1p(train["SalePrice"])

###log transform skewed numeric features:
num_feats = All_Data.dtypes[All_Data.dtypes != "object"].index
#print(num_feats)
skew_feats = train[num_feats].apply(lambda x: skew(x.dropna())) #compute skewness
#print(skew_feats)
skew_feats = skew_feats[skew_feats > 0.75] # Normally distributed data has a skew of 0
#print(skew_feats)
skew_feats = skew_feats.index
#print(skew_feats)
#replace with log
All_Data[skew_feats] = np.log1p(All_Data[skew_feats])


### Convert categorical data to numerical
All_Data = pd.get_dummies(All_Data)

### Now we have unskewed datawe can look at replacing NULLS this is a simple way to do it
All_Data = All_Data.fillna(All_Data.mean())

### split into train and testing sets again
X_train = All_Data[:train.shape[0]]
#print(X_train)
X_test = All_Data[train.shape[0]:]
#print(X_test)
Y = train.SalePrice
#print(Y)

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, LinearRegression
from sklearn.model_selection import cross_val_score

### Define a function that calculates the Root Mean Square of Errors for a given model using a cross_validation 
def RMSE_CV(model):
    RMSE = np.sqrt(-cross_val_score(model, X_train, Y, scoring="neg_mean_squared_error", cv = 5))
    return(RMSE)

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, Y)
model_reg = LinearRegression().fit(X_train,Y)

print("Linear model error",RMSE_CV(model_reg).mean())
print("Lasso model error",RMSE_CV(model_lasso).mean())

coef_reg = pd.Series(model_reg.coef_,index = X_train.columns)
coef_lasso = pd.Series(model_lasso.coef_, index = X_train.columns)

print("Coefs reg = ",coef_reg)
print("Coefs lasso = ",coef_lasso)

print("Lasso picked " + str(sum(coef_lasso != 0)) + " variables and eliminated the other " +  str(sum(coef_lasso == 0)) + " variables")

pred_lasso = model_lasso.predict(X_test)
pred_reg = model_reg.predict(X_test)
sol_lasso = pd.DataFrame({"id":test.Id, "SalePrice":np.expm1(pred_lasso)})
sol_reg = pd.DataFrame({"id":test.Id, "SalePrice":np.expm1(pred_reg)})

sol_lasso.to_csv("Lasso.csv", index=False)
sol_reg.to_csv("Reg.csv", index=False)

### Look at most and least useful vars for that regression model
important_coef = pd.concat([coef_reg.sort_values().head(10),
                     coef_reg.sort_values().tail(10)])
                     