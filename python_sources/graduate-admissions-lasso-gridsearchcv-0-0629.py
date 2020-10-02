#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
from scipy import stats
from scipy.stats import norm, skew #for some statistics

from sklearn.model_selection import KFold, cross_val_score, train_test_split
print(os.listdir("../input"))


# # Data
# 
# **Context**
# This dataset is created for prediction of Graduate Admissions from an Indian perspective.
# 
# **Content**
# The dataset contains several parameters which are considered important during the application for Masters Programs. The parameters included are : 
# 1. GRE Scores ( out of 340 ) 
# 2. TOEFL Scores ( out of 120 ) 
# 3. University Rating ( out of 5 ) 
# 4. Statement of Purpose and Letter of Recommendation Strength ( out of 5 ) 
# 5. Undergraduate GPA ( out of 10 ) 
# 6. Research Experience ( either 0 or 1 ) 
# 7. Chance of Admit ( ranging from 0 to 1 )

# # Read in data

# In[ ]:


# Training data
data = pd.read_csv("../input/Admission_Predict.csv",sep = ",")
data.head()


# # Small exploratory analysis of the data

# Examine the number of rows and columns: 

# In[ ]:


print("Number of rows: ", data.shape[0])
print("Number of columns: ", data.shape[1])


# ## Variable Target

# ** Chance of Admit ** is the variable we need to predict. So, first let's make an analysis about this variable.

# In[ ]:


sns.distplot(data['Chance of Admit '] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(data['Chance of Admit '])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Chance of Admit distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(data['Chance of Admit '], plot=plt)
plt.show()


# **We can see that the target variable is slightly to the right.**

# # Lost data

# In[ ]:


print("Total Characteristics with NaN values = " + str(data.columns[data.isnull().sum() != 0].size))
if (data.columns[data.isnull().sum() != 0].size):
    print("Features with NaN => {}".format(list(data.columns[data.isnull().sum() != 0])))
    train_df[data.columns[train.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)


# # Data correlation

# In[ ]:


# Extract the EXT_SOURCE variables and show correlations
ExtractData= data
ExtractDataCorrs = ExtractData.corr()
ExtractDataCorrs


# In[ ]:


plt.figure(figsize = (8, 6))
# Heatmap of correlations
sns.heatmap(ExtractDataCorrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');


# # Elimination of characteristics

# In[ ]:


data.drop(["Serial No."],axis=1,inplace = True)


# In[ ]:


data.head()


# # Standardization

# In[ ]:


data=data.rename(columns = {'Chance of Admit ':'Chance of Admit'})


# In[ ]:


# we decided our data
y = data["Chance of Admit"].values
x = data.drop(["Chance of Admit"],axis=1)


# In[ ]:


x.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x[['GRE Score', 'TOEFL Score', 'University Rating','SOP',
       'LOR ', 'CGPA', 'Research']]=scaler.fit_transform(data[['GRE Score', 'TOEFL Score', 'University Rating','SOP',
       'LOR ', 'CGPA', 'Research']])


# In[ ]:


x.head()


# In[ ]:


# separating train (80%) and test (%20) sets
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)


# # Define a cross validation strategy
# 
# we will use the function ** cross_val_score ** of Sklearn. However, this function does not have a random attribute, so we added a line of code, to shuffle the data set before the cross-validation
# 

# In[ ]:


#Validation function
n_folds = 10

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# # Models

# Before adjusting our models we make a search of the best parameters for each one using GridSearchCV and then we will see what is your RMSE

# ## Grid Rigde

# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
parameters = {'alpha':[10,15,25,30,50,100,150,200,500,600,700,750,785,790,800,900,1000,1001]}
     
RGD =  Ridge(solver='lsqr',fit_intercept=False)
grid_search = GridSearchCV(RGD, parameters, cv=n_folds, scoring="neg_mean_squared_error")
grid_search.fit( X_train, y_train )  
grid_search.best_params_, grid_search.best_score_  


# **RMSE: **

# In[ ]:


model_rigd =  Ridge(solver='lsqr',fit_intercept=False,alpha=790)
score = rmsle_cv(model_rigd)
print("Rigde score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# ## LASSO Regression

# In[ ]:


from sklearn.linear_model import  Lasso
from sklearn.model_selection import GridSearchCV

parameters = {'alpha':[0.0005,0.0006,0.06,0.5,0.0001,0.01,1,2,3,4,4.4,4]}

lasso_grid = Lasso(random_state=1)
grid_search_lasso = GridSearchCV(lasso_grid, parameters, cv=n_folds, scoring="neg_mean_squared_error")
grid_search_lasso.fit(X_train, y_train )  
grid_search_lasso.best_params_, grid_search_lasso.best_score_  


# **RMSE:**

# In[ ]:


model_lasso =  Lasso(random_state=1,alpha=0.0006)
score = rmsle_cv(model_lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# ## Elastic Net Regression

# In[ ]:


from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

parameters = {'alpha':[0.0005,0.0006,0.06,0.5,0.0001,0.01,1,2,3,4,4.4,4]}

ENet_grid = ElasticNet(random_state=3,l1_ratio=.9)
grid_search_Elastic = GridSearchCV(ENet_grid, parameters, cv=n_folds, scoring="neg_mean_squared_error")
grid_search_Elastic.fit(X_train, y_train )  
grid_search_Elastic.best_params_, grid_search_Elastic.best_score_  


# **RMSE: **

# In[ ]:


model_ENet =  ElasticNet(random_state=3,l1_ratio=.9,alpha= 0.0006)
score = rmsle_cv(model_ENet)
print("Elastic Net Regression score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# # Other models: 

# ## XGBoost

# In[ ]:


import xgboost as xgb

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# **RMSE:**

# In[ ]:


score = rmsle_cv(model_xgb)
print("Model xgb score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# ## Linear Regression 

# In[ ]:


from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()


# **RMSE:**

# In[ ]:


score = rmsle_cv(model_lr)
print("Linear Regression  models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# ## Gradient increase regression

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

model_gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# **RMSE:**

# In[ ]:


score = rmsle_cv(model_gboost)
print("Models Gradient increase regression score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# ## LightGBM

# In[ ]:


import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# **RMSE:**

# In[ ]:


score = rmsle_cv(model_lgb)
print("Models LightGBM score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# **We will choose the Lasso model to make our predictions, we will also see the importance of our characteristics.
# First we will adjust it with our training data**

# In[ ]:


model_lasso_fit = Lasso(random_state=1,alpha=0.0006).fit(X_train, y_train)


# # Importance of characteristics

# In[ ]:


coefficients = pd.Series(model_lasso_fit.coef_, index = X_train.columns)


# In[ ]:


print("Lasso picked " + str(sum(coefficients != 0)) + " variables and eliminated the other " +  str(sum(coefficients == 0)) + " variables")


# In[ ]:


imp_coef = pd.concat([coefficients.sort_values().head(7)])

plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")


# **Note**
# Eliminating the SOP feature, we will see a slight improvement in the model, but it is too little, so we will leave with the model we have

# # Predictions

# In[ ]:


pred = model_lasso_fit.predict(X_test)


# In[ ]:


predictions = pd.DataFrame({"Predictions":pred, "Real value": y_test})


# In[ ]:


predictions.round(2).head()

