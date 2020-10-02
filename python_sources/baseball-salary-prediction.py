#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.impute import KNNImputer
import missingno as msno
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.linear_model import LinearRegression


# # Data Understanding

# In[ ]:


hitters = pd.read_csv("../input/regularization-of-hitters/Hitters.csv")


# In[ ]:


df = hitters.copy()


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


df[df["Salary"].isnull()].head(10)


# In[ ]:


df[df["Hits"] < 70]


# In[ ]:


df[df["Hits"] > 70]


# In[ ]:


sns.lineplot(x = "Salary",y = "Years",data= df,hue = "League",style = "Division");


# In[ ]:


df.describe([0.01,0.25,0.75,0.99]).T


# In[ ]:


f, ax = plt.subplots(figsize= [20,15])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap = "coolwarm" )
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


# In[ ]:


df.groupby(["League"])["Salary"].mean()


# In[ ]:


df["Salary"].mean()


# # Data Preparation

# In[ ]:


msno.matrix(df);


# In[ ]:


for i in ["Hits","HmRun","Runs","RBI","Walks","Years","CAtBat","CHits","CHmRun","CRuns","CRBI","CWalks","PutOuts","Assists","Errors","Salary","AtBat"]:

    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3-Q1
    upper = Q3 + 1.5*IQR
    lower = Q1 - 1.5*IQR

    if df[(df[i] > upper) | (df[i] < lower)].any(axis=None):
        print(i,"yes")
        print(df[(df[i] > upper) | (df[i] < lower)].shape[0])
    else:
        print(i, "no")


# ## One hot encoding 

# In[ ]:


df = pd.get_dummies(df, columns =["League","Division","NewLeague"], drop_first = True)


# In[ ]:


cols = df.columns


# In[ ]:


cols


# In[ ]:


df.shape


# ## Missing values

# In[ ]:


#from sklearn.impute import SimpleImputer
#imputer = SimpleImputer()
#df_filled  = imputer.fit_transform(df)


# In[ ]:


#df_filled = pd.DataFrame(df_filled, columns = a)


# In[ ]:


imputer = KNNImputer(n_neighbors=6)
df_filled = imputer.fit_transform(df)


# In[ ]:


df.shape


# ## Outliers 

# In[ ]:


from sklearn.neighbors import LocalOutlierFactor


# In[ ]:


clf = LocalOutlierFactor(n_neighbors= 20,contamination= 0.1)


# In[ ]:


clf.fit_predict(df_filled)


# In[ ]:


df_scores = clf.negative_outlier_factor_


# In[ ]:


np.sort(df_scores)[0:30]


# In[ ]:


th = np.sort(df_scores)[8]
th


# In[ ]:


outlier = df_scores > th


# In[ ]:


dff = df_filled[df_scores > th]


# In[ ]:


dff = pd.DataFrame(dff,columns = cols)


# In[ ]:


dff.shape


# In[ ]:



from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
scaler = StandardScaler()
y = dff["Salary"]
X = dff.drop('Salary', axis=1)


# In[ ]:


df_ = pd.DataFrame(dff, columns = cols)


# In[ ]:


print(X.shape , y.shape)


# In[ ]:


X.head()


# In[ ]:


dummies = dff[["League_N","Division_W","NewLeague_N"]]
dummies


# In[ ]:


X = X.drop(["League_N","Division_W","NewLeague_N"],axis = 1)


# In[ ]:


X.head()


# In[ ]:


cols = X.columns


# In[ ]:


X = scaler.fit_transform(X)


# In[ ]:


X = pd.DataFrame(X, columns = cols)


# In[ ]:


X.head()


# In[ ]:


X.shape


# In[ ]:


dummies.shape


# In[ ]:


X_ = pd.concat([X,dummies],axis = 1)


# In[ ]:


X_.head()


# In[ ]:


print("X shape :", X_.shape,"Y shape",y.shape)


# # PREDICTION

# # Nonlinear Models

#  ## KNN 
#  
#  
#  
#  ![KNN-Algorithm-k3-edureka-437x300.png](attachment:KNN-Algorithm-k3-edureka-437x300.png)
#  
# >
#  
#  

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_, 
                                                    y, 
                                                    test_size = 0.20, random_state = 46)


# In[ ]:


knn_model = KNeighborsRegressor().fit(X_train,y_train)
y_pred = knn_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_pred))


# In[ ]:


knn_params = {"n_neighbors" : [3,4,5,6,7,8,9,10,12,20,23,24,30,32,35]}


# In[ ]:


knn_model = KNeighborsRegressor().fit(X_train,y_train)


# In[ ]:


#train

y_pred = knn_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_pred))


# In[ ]:


#test
y_pred = knn_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:


knn_cv_model = GridSearchCV(knn_model,knn_params, cv= 10,n_jobs =-1 ,verbose = 2).fit(X,y)


# In[ ]:


knn_cv_model.best_params_


# ## Tuned KNN

# In[ ]:


knn_tuned = KNeighborsRegressor(**knn_cv_model.best_params_).fit(X_train,y_train)
y_pred = knn_tuned.predict(X_test)
knn_sc = np.sqrt(mean_squared_error(y_test,y_pred))
knn_sc


# ## Support Vector Regressor
# 
# ![SVR.png](attachment:SVR.png)

# In[ ]:


from sklearn.svm import SVR


# In[ ]:


#train
svr_model = SVR().fit(X_train,y_train)
y_pred = svr_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_pred))


# In[ ]:


#test
y_pred = svr_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:


svr_params = {"C": [0.01,0.001, 0.2, 0.1,0.5,0.8,0.9,1]}


# In[ ]:


svr_model = SVR()


# In[ ]:


svr_cv_model = GridSearchCV(svr_model,svr_params,cv = 10,n_jobs = -1,verbose = 2).fit(X_train,y_train)


# In[ ]:


svr_cv_model.best_params_


# ## SVR Tuned

# In[ ]:


svr_tuned = SVR(**svr_cv_model.best_params_).fit(X_train,y_train)
y_pred = svr_tuned.predict(X_test)
svr_sc = np.sqrt(mean_squared_error(y_test,y_pred))
svr_sc


# ## CART
# 
# ![cart_model2.png](attachment:cart_model2.png)
# 
# **The aim is to transform the complex structures in the dataset into simple decision structures.**

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


cart_model = DecisionTreeRegressor().fit(X_train,y_train)


# In[ ]:


#train
y_pred = cart_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_pred))


# In[ ]:


cart_params = {"max_depth": [2,3,4,5,10,20, 100, 1000],
              "min_samples_split": [2,10,5,30,50,10]}


# In[ ]:


cart_cv_tuned = GridSearchCV(cart_model,cart_params,cv= 10,n_jobs = -1,verbose = 2).fit(X_train,y_train)


# In[ ]:


cart_cv_tuned.best_params_


# ## CART Tuned

# In[ ]:


cart_tuned = DecisionTreeRegressor(**cart_cv_tuned.best_params_).fit(X_train,y_train)


# In[ ]:


y_pred = cart_tuned.predict(X_test)
cart_sc = np.sqrt(mean_squared_error(y_test,y_pred))
cart_sc


# In[ ]:


Importance = pd.DataFrame({'Importance':cart_tuned.feature_importances_*100}, 
                          index = X_train.columns)


Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True).plot(kind = 'barh', 
                                              color = 'b', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None


# ## Random Forest
# 
# ![rand-forest-2.jpg](attachment:rand-forest-2.jpg)
# 
# 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf_model = RandomForestRegressor().fit(X_train,y_train)


# In[ ]:


#train
y_pred = rf_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_pred))


# In[ ]:


#test
y_pred = rf_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:


rf_params = {'max_depth': [10, 20, 30, None],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600,1000]}


# In[ ]:


rf_cv_model = GridSearchCV(rf_model,rf_params,cv = 10,n_jobs = -1,verbose = 2).fit(X_train,y_train)


# In[ ]:


rf_cv_model.best_params_


# In[ ]:


rf_tuned = RandomForestRegressor(**rf_cv_model.best_params_).fit(X_train,y_train)
y_pred = rf_tuned.predict(X_test)
rf_sc = np.sqrt(mean_squared_error(y_test,y_pred))
rf_sc


# In[ ]:


Importance = pd.DataFrame({'Importance':rf_tuned.feature_importances_*100}, 
                          index = X_train.columns)


Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True).plot(kind = 'barh', 
                                              color = 'b', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None


# ## GBM
# 
# ![GBM.png](attachment:GBM.png)
# 
# 

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


gbm_model = GradientBoostingRegressor().fit(X_train,y_train)


# In[ ]:


#train
y_pred = gbm_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_pred))


# In[ ]:


#test
y_pred = gbm_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:


gbm_params = {"learning_rate": [0.001,0.1,0.01, 0.05],
             "max_depth": [3,5,8,9,10],
             "n_estimators": [200,500,1000,1500],
             "subsample": [1,0.4,0.5,0.7],
             "loss": ["ls","lad","quantile"]}


# ## GBM Tuned

# In[ ]:


gbm_cv_model = GridSearchCV(gbm_model,gbm_params,cv = 10,n_jobs = -1,verbose = 2).fit(X_train,y_train)


# In[ ]:


gbm_cv_model.best_params_


# In[ ]:


gbm_tuned = GradientBoostingRegressor(**gbm_cv_model.best_params_).fit(X_train,y_train)
y_pred = gbm_tuned.predict(X_test)
gbm_sc = np.sqrt(mean_squared_error(y_test,y_pred))
gbm_sc


# In[ ]:


Importance = pd.DataFrame({'Importance':gbm_tuned.feature_importances_*100}, 
                          index = X_train.columns)


Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True).plot(kind = 'barh', 
                                              color = 'b', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None


# ## XGBOOST
# 
# ****XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.****
# 
# 

# In[ ]:


import xgboost
from xgboost import XGBRegressor


# In[ ]:


xgb_model = XGBRegressor().fit(X_train,y_train)


# In[ ]:


#train
y_pred = xgb_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_pred))


# In[ ]:


#test
y_pred = xgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:


xgb_params = {"learning_rate": [0.1,0.01,0.5],
             "max_depth": [2,3,4,5,8],
             "n_estimators": [100,200,500,1000],
             "colsample_bytree": [0.4,0.7,1]}


# In[ ]:


xgb_cv_model = GridSearchCV(xgb_model,xgb_params,cv = 10,n_jobs = -1,verbose= 2).fit(X_train,y_train)


# In[ ]:


xgb_cv_model.best_params_


# In[ ]:


xgb_tuned = XGBRegressor(**xgb_cv_model.best_params_).fit(X_train,y_train)


# In[ ]:


y_pred = xgb_tuned.predict(X_test)
xgb_sc = np.sqrt(mean_squared_error(y_test,y_pred))
xgb_sc


# In[ ]:


Importance = pd.DataFrame({'Importance':xgb_tuned.feature_importances_*100}, 
                          index = X_train.columns)


Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True).plot(kind = 'barh', 
                                              color = 'b', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None


# ## LightGBM
# 
# ****LightGBM**** is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:
# 
#  * Faster training speed and higher efficiency.
# 
#  * Lower memory usage.
# 
#  * Better accuracy.
# 
#  * Support of parallel and GPU learning.
# 
#  * Capable of handling large-scale data.

# In[ ]:


from lightgbm import LGBMRegressor


# In[ ]:


lgbm_model = LGBMRegressor().fit(X_train,y_train)


# In[ ]:


#train
y_pred = lgbm_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_pred))


# In[ ]:


#test
y_pred = lgbm_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:


lgbm_params = {"learning_rate": [0.01,0.001, 0.1, 0.5, 1],
              "n_estimators": [200,500,1000,5000],
              "max_depth": [2,4,6,7,10],
              "colsample_bytree": [1,0.8,0.5,0.4]}


# In[ ]:


lgbm_cv_model = GridSearchCV(lgbm_model,lgbm_params,cv =10,n_jobs= -1,verbose = 2).fit(X_train,y_train)


# In[ ]:


lgbm_cv_model.best_params_


# In[ ]:


lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train,y_train)


# ## LGBM Tuned

# In[ ]:


y_pred = lgbm_tuned.predict(X_test)
lgbm_sc = np.sqrt(mean_squared_error(y_test,y_pred))
lgbm_sc


# In[ ]:


Importance = pd.DataFrame({'Importance':lgbm_tuned.feature_importances_*100}, 
                          index = X_train.columns)


Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True).plot(kind = 'barh', 
                                              color = 'b', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None


# ## CatBoost

# In[ ]:


from catboost import CatBoostRegressor


# In[ ]:


catb_model = CatBoostRegressor(verbose = False).fit(X_train, y_train)


# In[ ]:


#train
y_pred = catb_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_pred))


# In[ ]:


#test
y_pred = catb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:


catb_params = {"iterations": [200,500,100],
              "learning_rate": [0.01,0.1],
              "depth": [3,6,8]}


# In[ ]:


catb_cv_model = GridSearchCV(catb_model,catb_params ,cv = 10,n_jobs = -1,verbose = 2).fit(X_train,y_train)


# In[ ]:


catb_cv_model.best_params_


# ## CatB Tuned

# In[ ]:


catb_tuned = CatBoostRegressor(**catb_cv_model.best_params_,verbose = False).fit(X_train,y_train)


# In[ ]:


y_pred = catb_tuned.predict(X_test)
catb_sc = np.sqrt(mean_squared_error(y_test,y_pred))
catb_sc


# In[ ]:


Importance = pd.DataFrame({'Importance':catb_tuned.feature_importances_*100}, 
                          index = X_train.columns)


Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True).plot(kind = 'barh', 
                                              color = 'b', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None


# # Reporting

# In[ ]:


models = pd.DataFrame({"Model" : ["KNN","SVR","CART","Random Forest","XGBoost","GBM","LightGBM","CatBoost"],
                     "Score" : [knn_sc,svr_sc,cart_sc,rf_sc,xgb_sc,gbm_sc,lgbm_sc,catb_sc]})


# In[ ]:


models.sort_values("Score")


# In[ ]:


plt.plot(models["Model"],models["Score"], 'ro');
plt.title("Regularization")
plt.xlabel("Models")
plt.ylabel("Scores")
plt.show()

