#!/usr/bin/env python
# coding: utf-8

# ## Introduction:
# 
# * The conventional process of testing the compressive strength of concrete involves casting several cubes for the respective grade (such as M5, M10, M15 etc.) and observing the strength of the concrete over a period of time ranging from 7 to 28 days. 
# 
# * Various combinations of the components of concrete are selected and cubes for each combination is casted and its test strength at 7, 14 and 28 days is noted dow.
# 
# * This is a time consuming and rather tedious process. 
# 
# #### This project aims to predict the compressive strength of concrete with maximum accuracy, for various quantities of constituent components as the input.
# 
# * The conrete cube exhibits behavioral differences in their compressive strengths for cubes that are cured/not cured. Curing is the process of maintaining the moisture to ensure uninterrupted hydration of concrete.
# 
# * The concrete strength increases if the concrete cubes are cured periodically. The rate of increase in strength is described here.
# 
# |Time|% Of Total Strength Achieved|
# |---|---|
# |1 day|16%|
# |3 days|40%|
# |7 days|65%|
# |14 days|90%|
# |28 days|99%|
# 
# * At 28 days, concrete achieves 99% of the strength. Thus usual measurements of strength are taken at 28 days.
# 

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


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_rows',500)


# In[ ]:


cc=pd.read_csv('/kaggle/input/concrete-compressive-strength-data-set/compresive_strength_concrete.csv')
cc.head()


# In[ ]:


cc.shape


# In[ ]:


cc.info()


# ## EDA

# In[ ]:


for i in cc.columns:
    sns.distplot(cc[i])
    plt.show()


# ## Univariate Analysis

# In[ ]:


cc.describe()


# * We can observe that there is a steady increase in the compressive strength of concrete with passing time.

# In[ ]:


cc.isnull().sum()


# ## Outlier Treatment

# We calculate the outliers in each columns.

# In[ ]:


q1=cc.quantile(0.25)
q3=cc.quantile(0.75)
IQR=q3-q1
cwo=((cc.iloc[:] <(q1-1.5*IQR))|(cc.iloc[:]>(q3+1.5*IQR))).sum(axis=0)
opdf=pd.DataFrame(cwo,index=cc.columns,columns=['No. of Outliers'])
opdf['Percentage Outliers']=round(opdf['No. of Outliers']*100/len(cc),2)
opdf


# Then we calculate the columns wise outliers. To determine in each row what is the presence of outliers.

# In[ ]:


rwo=(((cc[:]<(q1-1.5*IQR))|(cc[:]>(q3+1.5*IQR))).sum(axis=1))
ro005=(((rwo/len(cc.columns))<0.05).sum())*100/len(cc)
ro01=(((rwo/len(cc.columns))<0.1).sum())*100/len(cc)
ro015=(((rwo/len(cc.columns))<0.15).sum())*100/len(cc)
ro02=(((rwo/len(cc.columns))<0.2).sum())*100/len(cc)
ro025=(((rwo/len(cc.columns))<0.25).sum())*100/len(cc)
ro03=(((rwo/len(cc.columns))<0.30).sum())*100/len(cc)
ro035=(((rwo/len(cc.columns))<=0.35).sum())*100/len(cc)
ro04=(((rwo/len(cc.columns))<=0.4).sum())*100/len(cc)
ro045=(((rwo/len(cc.columns))<=0.45).sum())*100/len(cc)
ro05=(((rwo/len(cc.columns))<=0.50).sum())*100/len(cc)
ro055=(((rwo/len(cc.columns))<0.55).sum())*100/len(cc)
ro06=(((rwo/len(cc.columns))<0.6+0).sum())*100/len(cc)
ro=pd.DataFrame(np.round([ro005,ro01,ro015,ro02,ro025,ro03,ro035,ro04,ro045,ro05,ro055,ro06],2),
             index=['5%','10%','15%','20%','25%','30%','35%','40%','45%','50%','55%','60%'],
            columns=['% Data'])
ro.index.name='% Outlier'
ro


# More than 10% outliers of each row is not present. Hence the few outliers that will be treated using MICE (Multiple Imputation using Chained Equations) approach after these outliers are converted to NaN values.

# In[ ]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[ ]:


imp = IterativeImputer()
imp.fit(cc)
cc=pd.DataFrame(imp.transform(cc),columns=cc.columns)


# Thus we can see all outliers have been treated and removed by converting them to NaN values and imputing them using MICE.

# In[ ]:


cc.isnull().sum()


# ## Bivariate Analysis

# In[ ]:


g=cc.groupby('Age (day)')
g1=g.get_group(1)
g3=g.get_group(3)
g7=g.get_group(7)
g14=g.get_group(14)
g28=g.get_group(28)
pd.DataFrame(round(g28.iloc[:,-1].sort_values()).unique(),columns=['Comp Strength @ 28 days'])


# * Different types of concrete grades available and usually used are M7,M7.5,M10,M15,M20,M725,M30,M35,M40,M45,M50,M55,M60,M65,M70
# 
# * It essentially means at 28 days time the compressive strength should be 7MPa for M7 and 70MPa for M70.
# 
# * However we can see lots of grades of concrete. This could be due to variation in other contents of the concrete.

# In[ ]:


cp = cc.corr()
mask = np.zeros_like(cp)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(8,8))
with sns.axes_style("white"):
    sns.heatmap(cp,annot=True,linewidth=2,mask = mask,cmap="coolwarm")
plt.title("Correlation Plot")
plt.show()


# None of the features are highly inter correlated or correlated with the target variable.

# ## Linear Regression - OLS

# In[ ]:


import statsmodels.api as sm
X=cc.iloc[:,:8]
Y=cc.iloc[:,8]


# In[ ]:


ls=sm.OLS(Y,sm.add_constant(X))
results=ls.fit()
results.summary()


# Here we can see that the constant term is having P value greater than 0.05 viz. the assumed level of significance, thus we remove the constant term from modelling

# In[ ]:


ls=sm.OLS(Y,X)
results=ls.fit()
results.summary()


# Without the constant term we can observe that the R-squared value has increased drastically.

# ## SKLEARN - Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X, Y, random_state=150, test_size=0.3 )


# In[ ]:


lr=LinearRegression()
lr.fit(X_train,y_train)
print('Score: ',lr.score(X_train,y_train))
y_pred_lrtr=lr.predict(X_train)
y_pred_lrte=lr.predict(X_test)
from sklearn.metrics import r2_score
print('Train R2 score: ',r2_score(y_train,y_pred_lrtr))
print('Test R2 score: ',r2_score(y_test,y_pred_lrte))


# ## Polynomial Regression - Degree 2

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree = 2)
X_polytr = pf.fit_transform(X_train)
lr.fit(X_polytr,y_train)
y_pred_lr2tr = lr.predict(X_polytr)
print("Training R2 - degree 2 polynomial: ",r2_score(y_train, y_pred_lr2tr ))
X_polyte = pf.fit_transform(X_test)
y_pred_lr2te= lr.predict(X_polyte)
print("Test R2 - degree 2 polynomial: ",r2_score(y_test,y_pred_lr2te))


# ## Polynomial Regression - Degree 3

# In[ ]:


pf = PolynomialFeatures(degree = 3)
X_polytr = pf.fit_transform(X_train)
lr.fit(X_polytr,y_train)
y_pred_lr2tr = lr.predict(X_polytr)
print("Training R2 - degree 2 polynomial: ",r2_score(y_train, y_pred_lr2tr ))
X_polyte = pf.fit_transform(X_test)
y_pred_lr2te= lr.predict(X_polyte)
print("Test R2 - degree 2 polynomial: ",r2_score(y_test,y_pred_lr2te))


# ## Polynomial Regression - Degree 4

# In[ ]:


pf = PolynomialFeatures(degree = 4)
X_polytr = pf.fit_transform(X_train)
lr.fit(X_polytr,y_train)
y_pred_lr2tr = lr.predict(X_polytr)
print("Training R2 - degree 2 polynomial: ",r2_score(y_train, y_pred_lr2tr ))
X_polyte = pf.fit_transform(X_test)
y_pred_lr2te= lr.predict(X_polyte)
print("Test R2 - degree 2 polynomial: ",r2_score(y_test,y_pred_lr2te))


# Beyond this, the model does not perform well. From this it is clear that the model is non linear. Thus we proceed to other non-linear models.

# ## Decision Tree Regressor

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


dt=DecisionTreeRegressor()
dt.fit(X_train,y_train)
dt.score(X_train,y_train)
y_pred_dttr=dt.predict(X_train)
y_pred_dtte=dt.predict(X_test)
print('Train R2 score: ',r2_score(y_train,y_pred_dttr))
print('Test R2 score: ',r2_score(y_test,y_pred_dtte))


# The fully grown tree is overfitting. This can be controlled by pruning the tree. Using grid search we find the optimum depth and the impurity criterion and other hyper parameters.

# In[ ]:


from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': np.arange(3, 8),
             'criterion' : ['mse','mae'],
             'max_leaf_nodes': [5,10,20,100],
             'min_samples_split': [2, 5, 10, 20]}

grid_tree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv = 5, scoring= 'r2')
grid_tree.fit(X_train, y_train)
print(grid_tree.best_estimator_)
print(np.abs(grid_tree.best_score_))


# In[ ]:


dtpr=DecisionTreeRegressor(criterion='mse', max_depth=7, max_features=None,
                      max_leaf_nodes=100, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=10, min_weight_fraction_leaf=0.0,
                      presort=False, random_state=None, splitter='best')
dtpr.fit(X_train,y_train)
dtpr.score(X_train,y_train)
y_pred_dtprtr=dtpr.predict(X_train)
y_pred_dtprte=dtpr.predict(X_test)
print('Train R2 score: ',r2_score(y_train,y_pred_dtprtr))
print('Test R2 score: ',r2_score(y_test,y_pred_dtprte))


# It's severely overfit even now. We still have to prune it.

# In[ ]:


param_grid = {'max_depth': np.arange(3, 6),
             'criterion' : ['mse','mae'],
             'max_leaf_nodes': [100,105, 90,95],
             'min_samples_split': [6,7,8,9,10],
             'max_features':[2,3,4,5,6]}

grid_tree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv = 5, scoring= 'r2')
grid_tree.fit(X_train, y_train)
print(grid_tree.best_estimator_)
print(np.abs(grid_tree.best_score_))


# In[ ]:


dtpr=DecisionTreeRegressor(criterion='mae', max_depth=5, max_features=6,
                      max_leaf_nodes=95, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=8, min_weight_fraction_leaf=0.0,
                      presort=False, random_state=None, splitter='best')
dtpr.fit(X_train,y_train)
dtpr.score(X_train,y_train)
y_pred_dtprtr=dtpr.predict(X_train)
y_pred_dtprte=dtpr.predict(X_test)
print('Train R2 score: ',r2_score(y_train,y_pred_dtprtr))
print('Test R2 score: ',r2_score(y_test,y_pred_dtprte))


# The overfit has reduced but the model performance has nt imporoved on the test data. So we  now move onto other models.

# ## AdaBoost Regressor

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
abr = AdaBoostRegressor(random_state=0, n_estimators=100)
abr.fit(X_train, y_train)
abr.feature_importances_  
abr.fit(X_train,y_train)
abr.score(X_train,y_train)
y_pred_abrtr=abr.predict(X_train)
y_pred_abrte=abr.predict(X_test)
print('Train R2 score: ',r2_score(y_train,y_pred_abrtr))
print('Test R2 score: ',r2_score(y_test,y_pred_abrte))


# Adaboost has reduced the variance and improved the model performance as well.

# ## RandomForest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
rfr.fit(X_train,y_train)
rfr.score(X_train,y_train)
y_pred_rfrtr=rfr.predict(X_train)
y_pred_rfrte=rfr.predict(X_test)
print('Train R2 score: ',r2_score(y_train,y_pred_rfrtr))
print('Test R2 score: ',r2_score(y_test,y_pred_rfrte))


# The random forest is overfitting but has improved the model performance. So we now tune the hyper parameters to reduce the overfit.

# In[ ]:


param_grid = {'max_depth': np.arange(3, 8),
             'criterion' : ['mse','mae'],
             'max_leaf_nodes': [100,105, 90,95],
             'min_samples_split': [6,7,8,9,10],
             'max_features':['auto','sqrt','log2']}

grid_tree = GridSearchCV(RandomForestRegressor(), param_grid, cv = 5, scoring= 'r2')
grid_tree.fit(X_train, y_train)
print(grid_tree.best_estimator_)
print(np.abs(grid_tree.best_score_))


# In[ ]:


rfr=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=7,
                      max_features='auto', max_leaf_nodes=90,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=2, min_samples_split=7,
                      min_weight_fraction_leaf=0.0, n_estimators=100,
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)
rfr.fit(X_train,y_train)
rfr.score(X_train,y_train)
y_pred_rfrtr=rfr.predict(X_train)
y_pred_rfrte=rfr.predict(X_test)
print('Train R2 score: ',r2_score(y_train,y_pred_rfrtr))
print('Test R2 score: ',r2_score(y_test,y_pred_rfrte))


# ## Gradient Boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
gb=GradientBoostingRegressor()
gb.fit(X_train,y_train)
gb.score(X_train,y_train)
y_pred_gbtr=gb.predict(X_train)
y_pred_gbte=gb.predict(X_test)
print('Train R2 score: ',r2_score(y_train,y_pred_gbtr))
print('Test R2 score: ',r2_score(y_test,y_pred_gbte))


# In[ ]:


param_grid = {'n_estimators': [230],
              'max_depth': range(10,31,2), 
              'min_samples_split': range(50,501,10), 
              'learning_rate':[0.2]}
clf = GridSearchCV(GradientBoostingRegressor(random_state=1), 
                   param_grid = param_grid, scoring='r2', 
                   cv=5).fit(X_train, y_train)
print(clf.best_estimator_) 
print("R Squared:",clf.best_score_)


# In[ ]:


gb=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                          learning_rate=0.2, loss='ls', max_depth=14,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=150,
                          min_weight_fraction_leaf=0.0, n_estimators=230,
                          n_iter_no_change=None, presort='auto', random_state=1,
                          subsample=1.0, tol=0.0001, validation_fraction=0.1,
                          verbose=0, warm_start=False)
gb.fit(X_train,y_train)
gb.score(X_train,y_train)
y_pred_gbtr=gb.predict(X_train)
y_pred_gbte=gb.predict(X_test)
print('Train R2 score: ',r2_score(y_train,y_pred_gbtr))
print('Test R2 score: ',r2_score(y_test,y_pred_gbte))


# This model is slightly overfit. XGBoost may or may not reduce it. Trying out XGBoost in the next step.

# ## XGBoost Regressor

# In[ ]:


from xgboost import XGBRegressor

xgb=XGBRegressor()
xgb.fit(X_train,y_train)
print('Model Score: ', xgb.score(X_train,y_train))
y_pred_xgbtr=xgb.predict(X_train)
y_pred_xgbte=xgb.predict(X_test)
print('Train R2-Score: ', r2_score(y_train,y_pred_xgbtr))
print('Test R2-Score: ', r2_score(y_test,y_pred_xgbte))


# In[ ]:


xgb=XGBRegressor(base_score=0.7, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=0.65, colsample_bytree=1, gamma=0.3,
             importance_type='weight', learning_rate=0.2, max_delta_step=150,
             max_depth=4, min_child_weight=0.5, missing=None, n_estimators=200,
             n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,
             reg_alpha=0.001, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)
xgb.fit(X_train,y_train)
print('Model Score: ', xgb.score(X_train,y_train))
y_pred_xgbtr=xgb.predict(X_train)
y_pred_xgbte=xgb.predict(X_test)
print('Train R2-Score: ', r2_score(y_train,y_pred_xgbtr))
print('Test R2-Score: ', r2_score(y_test,y_pred_xgbte))


# Here we have achieved a model which performes well with both test and train data. This is very lightly overfit. It can be further adjusted. But this project will focus on the interpretability of the model.

# ## Interpreting Black Box Models

# In[ ]:


import shap


# In[ ]:


explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_train)


# In[ ]:


for i in X_train.columns:
    shap.dependence_plot(i,shap_values, X_train)


# 1. From the first plot we can see that Cement content and presence of super plasticizer has a linear impact on the model. When the cement content is less than 300 compressive strength decreases. As the cement content increases beyond 300, the compressive strength increases as well. The compressive strength increases with higher content of super plasticizers.
# 2. From the second plot we observe, when the blast furnace slag is greater than 50 kg/m3, the comrpessive strength increases. This feature in combination with age is responsible for the compressive strength. 
# 3. From the third plot we observe, when there is no fly ash present, but the mix with highest content of superplasticizers have a positive impact on the compressive strength. However the fly ash has an increasing followed by decreasing trend with the compressive strength. When the fly ash is in the range of 75-150 kg/m3 range, with super plasticizer content of 10 kg/m3 leads to highest compressive strength observed. Similary in the range of fly ash > 150 kg/m3 the least compressive strength was observed. 
# 4. From plot 4, we observe that water and blast furnace slag along with their interactive effect contributes to the compressive strength. Water content less than 150 kg/m3 with lower blast furnac slag provides the highest compressive strength. For water content greater than 150 kg/m3, higher blast furnace slag is preferred to have greater compressive strength.
# 5. From plot 5, superplasticizers in the range of 10-12kg/m3 along with higher blast furnace slag increases the compressive strength.
# 6. Plot 6 suggests that, with increasing coarse aggregate, keeping the coment content lower, has positive impact on the compressive strength.
# 7. For fine aggregates less than around 650 kg/m3, the water content should be greater than 180 kg/m3. Further, if the fine aggregate content is greater than 650 kg/m3, water content should be lesser than 170 kg/m3.
# 8. From plot 8, with increasing age, the compressive strength surely increases, however different amount of water content surely has an effect with age. For 28 days strength, having water content greater than 300kg/m3 is desired.

# In[ ]:


shap.summary_plot(shap_values, X_train)

