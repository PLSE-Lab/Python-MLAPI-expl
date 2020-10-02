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
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import seaborn as sns
from scipy.stats import spearmanr, chi2_contingency, mannwhitneyu, shapiro
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import missingno as msn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from warnings import filterwarnings
filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


test=pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train.head()


# In[ ]:


train["Survived"].value_counts()


# In[ ]:


test.head()


# # Data Union

# In[ ]:


dataset=pd.concat([train, test], ignore_index=True)


# In[ ]:


dataset.info()


# In[ ]:


dataset.Pclass.value_counts()


# In[ ]:


dataset.Sex.value_counts()


# In[ ]:


dataset.SibSp.value_counts()


# In[ ]:


dataset.Parch.value_counts()


# In[ ]:


dataset.Ticket.value_counts()


# In[ ]:


dataset.Cabin.value_counts()


# In[ ]:


dataset.Embarked.value_counts()


# # Train - Test Data missing value

# # Train Data

# In[ ]:


train.isnull().any()


# In[ ]:


msn.matrix(train);


# In[ ]:


msn.heatmap(train);


# In[ ]:


train.Age.isnull().sum()


# In[ ]:


print(train.Age.skew())
print(train.Age.kurtosis())


# In[ ]:


train.Age.fillna(train.Age.median(), inplace=True)


# In[ ]:


train.Embarked.value_counts()


# In[ ]:


train.Embarked.isnull().sum()


# In[ ]:


train.Embarked.fillna("S", inplace=True)


# In[ ]:


train.Cabin.isnull().sum()


# In[ ]:


train.Cabin.value_counts()


# In[ ]:


train.Cabin=np.where(train.Cabin.isnull(), "unknown", train.Cabin)


# In[ ]:


train.isnull().any()


# In[ ]:


train.head()


# In[ ]:





# # Test verisi

# In[ ]:


test.isnull().any()


# In[ ]:


msn.matrix(test);


# In[ ]:


msn.heatmap(test);


# In[ ]:


test.select_dtypes(exclude="object").describe().T


# In[ ]:


test.Age.isnull().sum()


# In[ ]:


print(test.Age.skew())
print(test.Age.kurtosis())


# In[ ]:


test.Age.fillna(test.Age.median(), inplace=True)


# In[ ]:


test.Fare.isnull().sum()


# In[ ]:


test.Fare.fillna(test.Fare.median(), inplace=True)


# In[ ]:


test.Cabin.isnull().sum()


# In[ ]:


test.Cabin=np.where(test.Cabin.isnull(), "unknown", test.Cabin)


# In[ ]:


test.isnull().any()


# In[ ]:


test.head()


# # Train Data Exploratory data analysis

# In[ ]:


train.head()


# In[ ]:


train.select_dtypes(exclude="object").describe().T


# In[ ]:


sns.set(rc={'figure.figsize': (8, 8)})
sns.boxplot(x=train.Fare,orient="v");


# # Outlier Detection for Variable "Fare"

# In[ ]:


Q1=train.Fare.quantile(0.25)
Q3=train.Fare.quantile(0.75)
IQR=Q3-Q1

lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR
IQR,lower_limit,upper_limit


# In[ ]:


freq_outliers = ((train.Fare < lower_limit) | (train.Fare > upper_limit)).value_counts()
freq_outliers


# In[ ]:


outliers_bool = ((train.Fare < lower_limit) | (train.Fare > upper_limit))
outliers_bool.head()


# In[ ]:


outliers=train.Fare[outliers_bool]


# In[ ]:


outliers.head()


# # Imputation of Fare with median

# In[ ]:


train.Fare[outliers_bool]=train.Fare.median()
train.Fare[outliers_bool]


# In[ ]:


train.describe().T


# # Questioning of multicollinearity

# In[ ]:


train[["Age", "SibSp", "Parch", "Fare"]].corr(method="spearman")


# In[ ]:


spearmanr(train.SibSp, train.Fare)


# In[ ]:


spearmanr(train.Fare, train.Parch)


# In[ ]:


spearmanr(train.SibSp, train.Parch)


# # Calculation of VIF

# In[ ]:


X=train.copy()


# In[ ]:


import time
from statsmodels.stats.outliers_influence import variance_inflation_factor    
from joblib import Parallel, delayed

# Defining the function that you will run later
def calculate_vif_(X, thresh=5.0):
    variables = [X.columns[i] for i in range(X.shape[1])]
    dropped=True
    while dropped:
        dropped=False
        print(len(variables))
        vif = Parallel(n_jobs=-1,verbose=0)(delayed(variance_inflation_factor)(X[variables].values, ix) for ix in range(len(variables)))

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print(time.ctime() + ' dropping \'' + X[variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables.pop(maxloc)
            dropped=True

    print('Remaining variables:')
    print([variables])
    return X[[i for i in variables]]

X = X[["Age", "SibSp", "Parch", "Fare"]] # Selecting your data

X2 = calculate_vif_(X,5) # Actually running the function


# # Distribution of dependent variables in variable "Survived" 

# # Pclass

# In[ ]:


pd.crosstab(train.Survived, train.Pclass, margins=True, margins_name="Total")


# In[ ]:


obs=np.matrix(pd.crosstab(train.Survived, train.Pclass))


# In[ ]:


chi2_contingency(obs)


# # Sex

# In[ ]:


pd.crosstab(train.Survived, train.Sex, margins=True, margins_name="Total")


# In[ ]:


obs=np.matrix(pd.crosstab(train.Survived, train.Sex))


# In[ ]:


chi2_contingency(obs)


# # SibSp

# In[ ]:


pd.crosstab(train.Survived, train.SibSp, margins=True, margins_name="Total")


# In[ ]:


obs=np.matrix(pd.crosstab(train.Survived, train.SibSp))


# In[ ]:


chi2_contingency(obs)


# # Parch

# In[ ]:


pd.crosstab(train.Survived, train.Parch, margins=True, margins_name="Total")


# In[ ]:


obs=np.matrix(pd.crosstab(train.Survived, train.Parch))


# In[ ]:


chi2_contingency(obs)


# # Cabin

# In[ ]:


train["Cabin"]=train["Cabin"].map(lambda x: str(x)[:1])


# In[ ]:


pd.crosstab(train.Survived, train.Cabin, margins=True, margins_name="Total")


# In[ ]:


obs=np.matrix(pd.crosstab(train.Survived, train.Cabin))


# In[ ]:


chi2_contingency(obs)


# # Cabin/unknown is higher within deads so Pclass 3rd maybe related with this?

# In[ ]:


pd.crosstab(train.Survived, [train.Pclass, train.Cabin], margins=True, margins_name="Total")


# # Embarked

# In[ ]:


pd.crosstab(train.Survived, train.Embarked, margins=True, margins_name="Total")


# In[ ]:


obs=np.matrix(pd.crosstab(train.Survived, train.Embarked))


# In[ ]:


chi2_contingency(obs)


# # Age

# In[ ]:


train.groupby("Survived")["Age"].describe()


# In[ ]:


plt.figure(figsize=(10,8))

sns.boxplot(x=train.Survived, y=train.Age);


# In[ ]:


Age_0=train[train["Survived"]==0]["Age"]


# In[ ]:


Age_1=train[train["Survived"]==1]["Age"]


# In[ ]:


shapiro(Age_0)


# In[ ]:


shapiro(Age_1)


# In[ ]:


mannwhitneyu(Age_0, Age_1)


# # Fare

# In[ ]:


train.groupby("Survived")["Fare"].describe()


# In[ ]:


plt.figure(figsize=(10,8))

sns.boxplot(x=train.Survived, y=train.Fare);


# In[ ]:


Fare_0=train[train["Survived"]==0]["Fare"]


# In[ ]:


Fare_1=train[train["Survived"]==1]["Fare"]


# In[ ]:


shapiro(Fare_0)


# In[ ]:


shapiro(Fare_1)


# In[ ]:


mannwhitneyu(Fare_0, Fare_0)

# Difference of categories of Pclass, Sex, Cabin, Embarked, Parch, Sibsp are statistically significant between dead and alive however Age and Fare are not statistically significant between dead and alive people.
# In[ ]:


train.head()


# In[ ]:


test.head()


# # Preprocess of whole dataset

# In[ ]:


# dataset is updated 

dataset=pd.concat([train, test])
dataset = dataset[[train, test][0].columns]


# In[ ]:


dataset.head()


# In[ ]:


dataset["Pclass"]=np.where(dataset["Pclass"]==1, "1st", dataset["Pclass"])


# In[ ]:


dataset["Pclass"]=np.where(dataset["Pclass"]=="2", "2nd", dataset["Pclass"])


# In[ ]:


dataset["Pclass"]=np.where(dataset["Pclass"]=="3", "3th", dataset["Pclass"])


# In[ ]:


dataset["Cabin"]=dataset["Cabin"].map(lambda x: str(x)[:1])


# In[ ]:


dataset.drop(["Name", "Ticket"], axis=1, inplace=True)


# In[ ]:


dataset.head(10)


# In[ ]:


df=dataset.copy()


# # OneHotEncoder for dataset

# In[ ]:


dms=pd.get_dummies(df[['Pclass', 'Sex', 'Cabin', 'Embarked']])


# In[ ]:


dms.head()


# In[ ]:


y=df["Survived"]


# In[ ]:


y.dropna(inplace=True)


# In[ ]:


y.isnull().sum()


# In[ ]:


y.shape


# In[ ]:


df.head()


# In[ ]:


X=df.drop(["Survived", "PassengerId", 'Pclass', 'Sex', 'Cabin','Embarked'], axis=1).astype("float64")


# In[ ]:


x = pd.concat([X, dms[['Pclass_2nd', "Pclass_3th", 'Sex_female',"Cabin_B","Cabin_C","Cabin_D", "Cabin_E","Cabin_F", "Cabin_G" ,"Cabin_T", "Cabin_u", 'Embarked_S', "Embarked_C"]]], axis=1)


# In[ ]:


x.head()


# In[ ]:


x_scaled=scale(x)


# In[ ]:


x_sc=pd.DataFrame(x_scaled, columns=['Age', 'SibSp', 'Parch', 'Fare',
       'Pclass_2nd', 'Pclass_3th', 'Sex_female', 'Cabin_B', 'Cabin_C',
       'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T', 'Cabin_u',
       'Embarked_S', 'Embarked_C'])


# In[ ]:


x_sc.head()


# In[ ]:


train_data=x_sc[0:len(train)]


# In[ ]:


test_data=x_sc[len(train):]


# In[ ]:


test_data.reset_index(drop=True, inplace=True)


# In[ ]:


train_data.tail()


# In[ ]:


test_data.head()


# # Machine Learning

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(train_data, y, 
                                                    test_size = 0.30, 
                                                    random_state = 42)


# # LOGISTIC REGRESSION

# In[ ]:


loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(x_train,y_train)
loj_model


# In[ ]:


y_pred=loj_model.predict(x_test)


# In[ ]:


accuracy_score(y_test, loj_model.predict(x_test))


# In[ ]:


cross_val_score(loj_model, x_test, y_test, cv = 10).mean()


# # NAIVE BAYES

# In[ ]:


nb = GaussianNB()
nb_model = nb.fit(x_train, y_train)
nb_model


# In[ ]:


y_pred = nb_model.predict(x_test)
print(accuracy_score(y_test, y_pred))
cross_val_score(nb_model, x_test, y_test, cv = 10).mean()


# # KNN

# knn_params = {"n_neighbors": np.arange(1,50)}

# knn = KNeighborsClassifier()
# knn_cv = GridSearchCV(knn, knn_params, cv=10)
# knn_cv.fit(x_train, y_train)

# print("best params: " + str(knn_cv.best_params_))

# In[ ]:


knn = KNeighborsClassifier(23)
knn_tuned = knn.fit(x_train, y_train)


# In[ ]:


knn_tuned


# In[ ]:


y_pred = knn_tuned.predict(x_test)
accuracy_score(y_test, y_pred)


# # SVC

# svc_params = {"C": np.arange(1,10)}
# 
# svc = SVC(kernel = "linear")
# 
# svc_cv_model = GridSearchCV(svc,svc_params, 
#                             cv = 10, 
#                             n_jobs = -1, 
#                             verbose = 2)
# 
# svc_cv_model.fit(x_train, y_train)

# print("best params: " + str(svc_cv_model.best_params_))

# In[ ]:


svc_tuned_linear=SVC(kernel = "linear", C = 1, probability=True, random_state=1)


# In[ ]:


svc_tuned_linear.fit(x_train, y_train)


# In[ ]:


y_pred = svc_tuned_linear.predict(x_test)
accuracy_score(y_test, y_pred)


# # RBF SVC

# svc_params = {"C": [0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100],
#              "gamma": [0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100]}

# svc = SVC()
# svc_cv_model = GridSearchCV(svc, svc_params, 
#                          cv = 10, 
#                          n_jobs = -1,
#                          verbose = 2)
# 
# svc_cv_model.fit(x_train, y_train)

# print("best params: " + str(svc_cv_model.best_params_))

# In[ ]:


svc_tuned=SVC(C = 1, gamma = 0.1, probability=True,random_state=1)


# In[ ]:


svc_tuned.fit(x_train, y_train)


# In[ ]:


y_pred=svc_tuned.predict(x_test)
accuracy_score(y_test, y_pred)


# # ANN MLP

# mlpc_params = {"alpha": [0.1, 0.01, 0.02, 0.005, 0.0001,0.00001],
#               "hidden_layer_sizes": [(10,10,10),
#                                      (100,100,100),
#                                      (100,100),
#                                      (3,5), 
#                                      (5, 3)],
#               "solver" : ["lbfgs","adam","sgd"],
#               "activation": ["relu","logistic"]}

# mlpc = MLPClassifier()
# mlpc_cv_model = GridSearchCV(mlpc, mlpc_params, 
#                          cv = 10, 
#                          n_jobs = -1,
#                          verbose = 2)
# 
# mlpc_cv_model.fit(x_train, y_train)

# print("best params: " + str(mlpc_cv_model.best_params_))

# In[ ]:


mlpc_tuned = MLPClassifier(activation = "relu", 
                           alpha = 0.0001, 
                           hidden_layer_sizes = (100, 100, 100),
                           solver = "sgd", random_state=1)


# In[ ]:


mlpc_tuned.fit(x_train, y_train)


# In[ ]:


y_pred = mlpc_tuned.predict(x_test)
accuracy_score(y_test, y_pred)


# # CART

# cart_grid = {"max_depth": range(1,10),
#             "min_samples_split" : list(range(2,50)) }

# cart = tree.DecisionTreeClassifier()
# cart_cv = GridSearchCV(cart, cart_grid, cv = 10, n_jobs = -1, verbose = 2)
# cart_cv_model = cart_cv.fit(x_train, y_train)

# print("best params: " + str(cart_cv_model.best_params_))

# In[ ]:


cart = tree.DecisionTreeClassifier(max_depth = 6, min_samples_split = 3, random_state=1)
cart_tuned = cart.fit(x_train, y_train)


# In[ ]:


cart_tuned


# In[ ]:


y_pred = cart_tuned.predict(x_test)
accuracy_score(y_test, y_pred)


# # Random Forests

# rf_params = {"max_depth": [2,5,8,10],
#             "max_features": [2,5,8],
#             "n_estimators": [10,500,1000],
#             "min_samples_split": [2,5,10]}

# rf_model = RandomForestClassifier()
# 
# rf_cv_model = GridSearchCV(rf_model, 
#                            rf_params, 
#                            cv = 10, 
#                            n_jobs = -1, 
#                            verbose = 2) 

# rf_cv_model.fit(x_train, y_train)

# print("best params: " + str(rf_cv_model.best_params_))

# In[ ]:


rf_tuned = RandomForestClassifier(max_depth = 8, 
                                  max_features = 5, 
                                  min_samples_split = 2,
                                  n_estimators = 1000, random_state=1)

rf_tuned.fit(x_train, y_train)


# In[ ]:


y_pred = rf_tuned.predict(x_test)
accuracy_score(y_test, y_pred)


# # GBM

# gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],
#              "n_estimators": [100,500,100],
#              "max_depth": [3,5,10],
#              "min_samples_split": [2,5,10]}

# gbm = GradientBoostingClassifier()
# 
# gbm_cv = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 2)

# gbm_cv.fit(x_train, y_train)

# print("best params: " + str(gbm_cv.best_params_))

# In[ ]:


gbm = GradientBoostingClassifier(learning_rate = 0.1, 
                                 max_depth = 3,
                                min_samples_split = 5,
                                n_estimators = 100, random_state=1)


# In[ ]:


gbm_tuned =  gbm.fit(x_train,y_train)


# In[ ]:


gbm_tuned


# In[ ]:


y_pred = gbm_tuned.predict(x_test)
accuracy_score(y_test, y_pred)


# # XGBoost

# xgb_params = {
#         'n_estimators': [100, 500, 1000, 2000],
#         'subsample': [0.6, 0.8, 1.0],
#         'max_depth': [3, 4, 5,6],
#         'learning_rate': [0.1,0.01,0.02,0.05],
#         "min_samples_split": [2,5,10]}

# xgb = XGBClassifier()
# 
# xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)

# xgb_cv_model.fit(x_train, y_train)

# xgb_cv_model.best_params_

# In[ ]:


xgb = XGBClassifier(learning_rate = 0.02, 
                    max_depth = 3,
                    min_samples_split = 2,
                    n_estimators = 2000,
                    subsample = 1)


# In[ ]:


xgb_tuned =  xgb.fit(x_train,y_train)


# In[ ]:


xgb_tuned


# In[ ]:


y_pred = xgb_tuned.predict(x_test)
accuracy_score(y_test, y_pred)


# # Light GBM

# lgbm_params = {
#         'n_estimators': [100, 500, 1000, 2000],
#         'subsample': [0.6, 0.8, 1.0],
#         'max_depth': [3, 4, 5,6],
#         'learning_rate': [0.1,0.01,0.02,0.05],
#         "min_child_samples": [5,10,20]}

# lgbm = LGBMClassifier()
# 
# lgbm_cv_model = GridSearchCV(lgbm, lgbm_params, 
#                              cv = 10, 
#                              n_jobs = -1, 
#                              verbose = 2)

# lgbm_cv_model.fit(x_train, y_train)

# lgbm_cv_model.best_params_

# In[ ]:


lgbm = LGBMClassifier(learning_rate = 0.02, 
                       max_depth = 4,
                       subsample = 0.6,
                       n_estimators = 500,
                       min_child_samples = 20,random_state=1)


# In[ ]:


lgbm_tuned = lgbm.fit(x_train,y_train)


# In[ ]:


lgbm_tuned


# In[ ]:


y_pred = lgbm_tuned.predict(x_test)
accuracy_score(y_test, y_pred)


# # CatBoost

# catb_params = {
#     'iterations': [200,500],
#     'learning_rate': [0.01,0.05, 0.1],
#     'depth': [3,5,8] }

# catb = CatBoostClassifier()
# catb_cv_model = GridSearchCV(catb, catb_params, cv=5, n_jobs = -1, verbose = 2)
# catb_cv_model.fit(x_train, y_train)

# catb_cv_model.best_params_

# In[ ]:


catb = CatBoostClassifier(iterations = 200, 
                          learning_rate = 0.1, 
                          depth = 5, verbose=False, random_seed=1)

catb_tuned = catb.fit(x_train, y_train)


# In[ ]:


catb_tuned


# In[ ]:


y_pred = catb_tuned.predict(x_test)
accuracy_score(y_test, y_pred)


# # Importance

# In[ ]:


models_importance = [
{
    'label': 'DecisionTreeClassifier',
    'model': cart_tuned,
},
{
    'label': 'RandomForestClassifier',
    'model': rf_tuned,
},
{
    'label': 'GradientBoostingClassifier',
    'model': gbm_tuned,
},
{
    'label': 'XGBClassifier',
    'model': xgb_tuned,
},
{
    'label': 'LGBMClassifier',
    'model': lgbm_tuned,
},
{
    'label': 'CatBoostClassifier',
    'model': catb_tuned,
}
]

for model in models_importance:
    names = model['label']
    model = model["model"]    
    model_imp=pd.DataFrame(model.feature_importances_, index = x_train.columns)
    model_imp.columns=["imp"]
    model_imp.sort_values(by = "imp", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "r")
    plt.title(names)
    plt.xlabel("Importance Values")


# In[ ]:


models_importance = [
{
    'label': 'LogisticRegression',
    'model': loj_model,
},
{
    'label': 'SVC_LINEAR',
    'model': svc_tuned_linear,
}
]


for model in models_importance:
    names = model['label']
    model = model["model"]
    coef=np.array(model.coef_)
    model_imp=pd.DataFrame(coef.flatten(), index = x_train.columns)
    model_imp.columns=["imp"]
    model_imp.sort_values(by = "imp", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "r")
    plt.title(names)
    plt.xlabel("Importance Values")


# # Comparison of All Models

# In[ ]:


models = [
{
    'label': 'KNeighborsClassifier',
    'model': knn_tuned,
},
{
    'label': 'LogisticRegression',
    'model': loj_model,
},
{
    'label': 'SVC_LINEAR',
    'model': svc_tuned_linear,
},
{
    'label': 'SVC_RBF',
    'model': svc_tuned,
},
{
    'label': 'GaussianNB',
    'model': nb_model,
},
{
    'label': 'MLPClassifier',
    'model': mlpc_tuned,
},
{
    'label': 'DecisionTreeClassifier',
    'model': cart_tuned,
},
{
    'label': 'RandomForestClassifier',
    'model': rf_tuned,
},
{
    'label': 'GradientBoostingClassifier',
    'model': gbm_tuned,
},
{
    'label': 'XGBClassifier',
    'model': xgb_tuned,
},
{
    'label': 'LGBMClassifier',
    'model': lgbm_tuned,
},
{
    'label': 'CatBoostClassifier',
    'model': catb_tuned,
}
]

for model in models:
    names = model['label']
    model = model["model"]
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusionmatrix=confusion_matrix(y_test, y_pred)
    (TN, FP, FN, TP) = confusionmatrix.ravel()
    TPR = TP/(TP+FN) 
    TNR = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    print("-"*28)
    print(names + ":" )
    print("Accuracy: {:.4%}".format(accuracy))
    print("TPR     : {:.4%}".format(TPR))
    print("TNR     : {:.4%}".format(TNR))
    print("PPV     : {:.4%}".format(PPV))
    print("NPV     : {:.4%}".format(NPV))
    print("FPR     : {:.4%}".format(FPR))
    print("FNR     : {:.4%}".format(FNR))


# In[ ]:


result = []

results = pd.DataFrame(columns= ["Models","Accuracy"])

for model in models:
    names = model['label']
    model = model["model"]
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)    
    result = pd.DataFrame([[names, accuracy*100]], columns= ["Models","Accuracy"])
    results = results.append(result)


    
results=results.sort_values(by="Accuracy", ascending=False).reset_index()
results.drop(["index"], axis=1, inplace=True)
g=sns.barplot(x= 'Accuracy', y = 'Models', data=results, color="r")
g.set_xlabel("Accuracy",fontsize=20)
g.set_ylabel("Models",fontsize=20)

for index, row in results.iterrows():
    g.text(row.Accuracy, row.name, round(row.Accuracy,2), color='black', horizontalalignment='left', fontsize=13)

plt.xlabel('Accuracy %')
plt.title('Accuracy Scores of Models');  


# # ROC Curve

# In[ ]:


from sklearn import metrics
import matplotlib.pyplot as plt

plt.figure()


# Below for loop iterates through your models list
for m in models:
    model = m['model'] # select the model
# Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(x_test)[:,1])
# Calculate Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_test,model.predict(x_test))
# Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (AUC = %0.4f)' % (m['label'], auc))
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5) , fontsize=15)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.show()   # Display


# In[ ]:


from sklearn import metrics

plt.figure()

# Add the models to the list that you want to view on the ROC plot
modelsROC = [
{
    'label': 'LogisticRegression',
    'model': loj_model,
},
{
    'label': 'GradientBoostingClassifier',
    'model': gbm_tuned,
}
]

# Below for loop iterates through your models list
for m in modelsROC:
    model = m['model'] # select the model
# Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(x_test)[:,1])
# Calculate Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_test,model.predict(x_test))
# Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (AUC = %0.4f)' % (m['label'], auc))
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='right corner', fontsize=15)
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.show()   # Display


# # Test Data Predictions

# In[ ]:


test_data_y_pred=gbm_tuned.predict(test_data)


# In[ ]:


survived=pd.concat([test["PassengerId"], (pd.DataFrame(test_data_y_pred, columns=["Survived"]))], axis=1)


# In[ ]:


survived["Survived"]=survived["Survived"].astype("int64")


# In[ ]:


survived


# In[ ]:


survived.to_csv("Survived_Prediction.csv")


# In[ ]:




