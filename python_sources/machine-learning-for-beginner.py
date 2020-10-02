#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # 1- Linear Regression Model
# Linear Regression is a linear approach to modeling the relationship between a scaler response and one or more explanatory variables. The case of one explanatory variable is called simple *linear regression* .

# In[ ]:



import pandas as pd
df = pd.read_csv("/kaggle/input/Advertising.csv")
df = df.iloc[:,1:len(df)]
df.head()


# In[ ]:


df.info


# In[ ]:


import seaborn as sns
sns.jointplot(x="TV", y="sales", data=df, kind="reg");


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


X = df[["TV"]]
X.head()


# In[ ]:


y = df[["sales"]]
y.head()


# In[ ]:


reg = LinearRegression()
model = reg.fit(X,y)
model


# In[ ]:


#Let's learn the model
str(model)


# In[ ]:


dir(model)


# In[ ]:


#beta 0 (n) in basic linear regression formul(mx+n)
model.intercept_  


# In[ ]:


#beta 1 (m) in basic linear regression formul(mx+n)
model.coef_


# In[ ]:


#r2 
#Percentage of change in dependent variable explained by independent variables
model.score(X,y)
#The change in the independent variable is about 60 percent explained.


# In[ ]:


#guess
import seaborn as sns
import matplotlib.pyplot as plt

g = sns.regplot(df["TV"],df["sales"], ci=None, scatter_kws= {"color":"r","s":9})

#set table
g.set_title("Models: Sales = 7.03 + TV * 0.05")  
g.set_ylabel("Sales Number")
g.set_xlabel("TV expenditures")

plt.xlim(-10,310)
plt.ylim(bottom=0);


# In[ ]:


#Real Value 
7.03+ 0.05*165


# In[ ]:


#Predict Value
model.predict([[165]])


# In[ ]:


new_data = [[5],[150],[300],[450]]


# In[ ]:


model.predict(new_data)


# In[ ]:


#Expected 2D array, got 1D array instead
#Value Error expected : model.predict([500])
model.predict([[500]])
#predict : ~30


# In[ ]:


y.head()


# **Residues and Their Importance in Machine Learning**
# *  MSE : Mean squared error
# * RMSE : Root mean squared error

# In[ ]:


y.head()


# In[ ]:


X.head()


# In[ ]:


model.predict(X)[0:6]


# In[ ]:


real_y = y[0:10]


# In[ ]:


predict_y = pd.DataFrame(model.predict(X)[0:10])


# In[ ]:


errors = pd.concat([real_y,predict_y], axis=1)
errors.columns = ["real_y","predict_y"]
errors


# In[ ]:


errors["error"] = errors["real_y"] - errors["predict_y"]
errors


# ### MSE
# **As you can see, the error values can be minus, and we will take the squaring process to  take care of it and we will take the average. In this way, we will have MSE**

# In[ ]:


errors["mean_squared"] = errors["error"]**2


# In[ ]:


errors


# In[ ]:


import numpy as np
MSE = np.mean(errors["mean_squared"])
print("MSE : ", MSE)


# # 2 - Multiple Linear Regression

# In[ ]:


#Model
import numpy as np
import pandas as pd
df = pd.read_csv("/kaggle/input/Advertising.csv")
df = df.iloc[:,1:len(df)]
df.head()


# In[ ]:


X = df.drop("sales",axis=1)
y = df[["sales"]]


# In[ ]:


y.head()


# In[ ]:


X.head()


# In[ ]:


#Model : wtih Sklearn 
from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[ ]:


model = lm.fit(X, y)


# In[ ]:


# mx+n --> intercept = n
model.intercept_


# In[ ]:


#mx+n --> coef = m
model.coef_


# In[ ]:





# ## Predict
# ** Sales = 2.94 + (TV * 0.04) + (radio * 0.19) -(newspaper*0.001) 
# * 30  TV , 10  radio , 40 newspaper

# In[ ]:


2.94 + (30 * 0.04) + (10 * 0.19) -(40 * 0.001)


# In[ ]:


new_data = [[30],[10],[50]]


# In[ ]:


import pandas as pd
new_data = pd.DataFrame(new_data).T
new_data


# In[ ]:


model.predict(new_data)


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


y.head()


# In[ ]:


model.predict(X)[0:10]


# In[ ]:


MSE = mean_squared_error(y,model.predict(X))
MSE


# In[ ]:


import numpy as np
RMSE = np.sqrt(MSE)
RMSE


# ## Model Tuning

# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state =99)


# In[ ]:


#X_test.ndim
X_test.shape


# In[ ]:


X_train.shape


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# In[ ]:


y_test.head()


# In[ ]:


lm = LinearRegression()
model = lm.fit(X_train, y_train)


# In[ ]:


#Error train value
y_predict_train = model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_predict_train))


# In[ ]:


#Error test value
y_predict_test = model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_predict_test))


# ### K-fold Cross Validation

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


model


# In[ ]:


# cv mse
cross_val_score(model, X_train, y_train, cv=10, scoring = "neg_mean_squared_error")


# In[ ]:


np.mean(-cross_val_score(model, X_train, y_train, cv=10, scoring = "neg_mean_squared_error"))


# In[ ]:


# cv root mse
import numpy as np
RMSE = np.sqrt(np.mean(-cross_val_score(model, X_train, y_train, cv=10, scoring = "neg_mean_squared_error")))
RMSE


# In[ ]:


P = np.sqrt(np.mean(-cross_val_score(model, X, y, cv=10, scoring = "neg_mean_squared_error")))


# In[ ]:


error =  P - RMSE 
error


# # 4- Ridge Regression

# ### Required libraries

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV


# In[ ]:


#Data Set
df = pd.read_csv("/kaggle/input/Hitters.csv")
df = df.dropna()

dms = pd.get_dummies(df[['League','Division','NewLeague']])

y = df["Salary"]
X_ = df.drop(['Salary','League','Division','NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N','Division_W','NewLeague_N']]],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.25,
                                                    random_state=42)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


ridge_model = Ridge(alpha = 5).fit(X_train, y_train)


# In[ ]:


ridge_model


# In[ ]:


ridge_model.coef_


# In[ ]:


ridge_model.intercept_


# In[ ]:


# generating random numbers (from 10 to 2)
np.linspace(10,2,100)


# In[ ]:


lambdas = 10** np.linspace(10,2,100)*0.5
lambdas


# In[ ]:


ridge_model = Ridge()
factor = []

for i in lambdas:
    ridge_model.set_params(alpha=i)
    ridge_model.fit(X_train, y_train)
    factor.append(ridge_model.coef_)


# In[ ]:


ax = plt.gca()
ax.plot(lambdas,factor)
ax.set_xscale("log")


# ## Predict

# In[ ]:


ridge_model = Ridge().fit(X_train, y_train)
y_pred = ridge_model.predict(X_train)


# In[ ]:


y_train[0:10]


# In[ ]:


y_pred[0:10]


# In[ ]:


RMSE = np.sqrt(mean_squared_error(y_train, y_pred))  
RMSE


# In[ ]:


#cv rmse
from sklearn.model_selection import cross_val_score 
np.sqrt(np.mean(-cross_val_score(ridge_model, X_train, y_train, cv=10, scoring = "neg_mean_squared_error")))


# In[ ]:


#test eror
y_pred = ridge_model.predict(X_test)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))  
RMSE


# # Model Tuning

# In[ ]:


ridge_model= Ridge(alpha=1).fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


np.random.randint(0,1000,100)


# In[ ]:


lambda1 = np.random.randint(0,1000,100)
lambda2 = 10** np.linspace(10,2,100)*0.5


# In[ ]:


# pick one lambda 1 or lambda 2(you should try ) :)
ridgecv = RidgeCV(alphas = lambda1, scoring = "neg_mean_squared_error", cv=10, normalize=True )
ridgecv.fit(X_train, y_train)


# In[ ]:


ridgecv.alpha_ #(i think optimuim alpha = 2 )


# In[ ]:


#final model
ridge_tuned = Ridge(alpha=ridgecv.alpha_).fit(X_train, y_train)
y_pred = ridge_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# #  4- Lasso Regresyon

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV


# In[ ]:


#Data Set
df = pd.read_csv("/kaggle/input/Hitters.csv")
df = df.dropna()

dms = pd.get_dummies(df[['League','Division','NewLeague']])

y = df["Salary"]
X_ = df.drop(['Salary','League','Division','NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N','Division_W','NewLeague_N']]],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.25,
                                                    random_state=42)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


lasso_model = Lasso().fit(X_train, y_train)


# In[ ]:


lasso_model


# In[ ]:


lasso_model.intercept_ 


# In[ ]:


lasso_model.coef_ 


# In[ ]:


lasso = Lasso()
coefs = []
#alphas = np.random.randint(0,100000,10) #lambdas
alphas = lambdalar = 10** np.linspace(10,2,100)*0.5
for a in alphas:
    lasso.set_params(alpha = a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)


# In[ ]:


ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale("log")


# ### Predict

# In[ ]:


lasso_model


# In[ ]:


lasso_model.predict(X_train)[0:5]


# In[ ]:


lasso_model.predict(X_test)[0:5]


# In[ ]:


y_pred = lasso_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


r2_score(y_test, y_pred)


# ## Model Tuning

# In[ ]:


alphas = lambdalar = 10** np.linspace(10,2,100)*0.5
lasso_cv_model = LassoCV(alphas = alphas, cv=10, max_iter = 100000).fit(X_train, y_train)
lasso_cv_model.alpha_


# In[ ]:


lasso_tuned = Lasso().set_params(alpha = lasso_cv_model.alpha_).fit(X_train, y_train)
lasso_tuned = Lasso(alpha = lasso_cv_model.alpha_).fit(X_train, y_train)
y_pred = lasso_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


pd.Series(lasso_tuned.coef_, index = X_train.columns)


# ## 5-ElasticNet Regresyon Modeli

# In[ ]:


#library
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV


# In[ ]:


#Data Set
df = pd.read_csv("/kaggle/input/Hitters.csv")
df = df.dropna()

dms = pd.get_dummies(df[['League','Division','NewLeague']])

y = df["Salary"]
X_ = df.drop(['Salary','League','Division','NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N','Division_W','NewLeague_N']]],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.25,
                                                    random_state=42)


# In[ ]:


enet_model = ElasticNet().fit(X_train, y_train)


# In[ ]:


enet_model.coef_


# In[ ]:


enet_model.intercept_


# ### Predict

# In[ ]:


enet_model.predict(X_train)[0:10]


# In[ ]:


enet_model.predict(X_test)[0:10]


# In[ ]:


y_pred = enet_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) 


# In[ ]:


r2_score(y_test, y_pred)


# ### Model Tuning

# In[ ]:


enet_cv_model = ElasticNetCV(cv=10).fit(X_train, y_train)


# In[ ]:


enet_cv_model.alpha_


# In[ ]:


enet_cv_model.intercept_


# In[ ]:


enet_cv_model.coef_


# In[ ]:


enet_tuned = ElasticNet(alpha = enet_cv_model.alpha_ ).fit(X_train, y_train)
y_pred = enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# # Dogrusal Olmayan Regresyon Modelleri

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR


# In[ ]:


from warnings import filterwarnings
filterwarnings('ignore')


# # KNN

# In[ ]:


#Data Set
df = pd.read_csv("/kaggle/input/Hitters.csv")
df = df.dropna()

dms = pd.get_dummies(df[['League','Division','NewLeague']])

y = df["Salary"]
X_ = df.drop(['Salary','League','Division','NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N','Division_W','NewLeague_N']]],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.25,
                                                    random_state=42)


# In[ ]:


X_train.head()


# ## Model & Predict

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




