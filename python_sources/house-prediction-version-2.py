#!/usr/bin/env python
# coding: utf-8

# <h1>House price prediction</h1>
# 
# ![](https://ftrs.s3.amazonaws.com/images/all-about-classes/simpsons_re.gif)
# 
# 
# <p>
#         In this report I will describe data analysis techniques, which were applied to predict price in Boston region. First for all I start with EDA in section 1. Also here it included research dependency analysis between buildings prices and an object parameters and dependency analysis among attributes to find correlation and detect anomalies.
#     <br>
#     <br>
#      Section 2 describe choosing and building model. When model was chosen it validated using train and split method. XGBoost and Linear model was chosen for bulding models. Linear model was chosen using such libraries like sklearn and statsmodels.
# </p>

# <h2>Module imports (sklearn, scipy, statsmodel, pandas, numpy, seaborn, xgboost)</h2>

# In[423]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.pipeline import Pipeline
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from patsy import dmatrices

from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('ggplot')

def get_x():
    return data[list(filter(lambda x: x != 'MEDV', data.columns.values))].copy()

def get_y():
    return data['MEDV'].copy()

def get_norm_x():
    return MinMaxScaler().fit_transform(get_x())

def get_norm_y():
    return MinMaxScaler().fit_transform(get_y().values.reshape(-1, 1))


# <h1>Exploratory data analysis (EDA)</h1>

# First for all import data to dataset and glimpse on 10 first values.

# In[396]:


data = pd.read_csv('../input/data.csv')
data.head()


# In[ ]:


data.describe().drop(['count'])


# Let's check if NaN values in dataset. According to data, NaN values are absent:

# In[ ]:


data.info()


# A pairplot shows correlation between variables with MEDV. The most correlated variables with MEDV is TAX and RM.

# In[ ]:


sns.pairplot(data=data, y_vars=['MEDV'], x_vars=list(filter(lambda x: x != 'MEDV' and x != 'CHAS', data.columns)), kind='reg')
plt.draw()


# A box plot shows the distribution of quantitative data in a way that facilitates comparisons between features. According to picture CRIM and ZN features have so many outliers

# In[ ]:


v = MinMaxScaler()
d = v.fit_transform(data)
fig, ax = plt.subplots(1,1, figsize=(20, 10))
sns.boxplot(data=d, palette="Set2",dodge=False, ax=ax)
ax.set_xticklabels(data.columns)
ax.set_title('Features distributions')
plt.show()


# In[ ]:


fig, axes = plt.subplots(2, 7, figsize=(20, 10))
data_without_chas = data.drop(['CHAS'], axis=1)

for i, ax in zip(data_without_chas.columns, axes.flatten()):
    sns.kdeplot(data_without_chas[i], ax = ax)
    ax.set_title(i)
    ax.get_legend().remove()


# A kdeplot shows that many variables fit to normal distribution. Let check it.

# In[ ]:


normal_ad(data)


# None of the varibles have normal distribution.

# Data has one categorial variable CHAS. Let check proportion of the classes in this variable:

# In[ ]:


get_x()['CHAS'].value_counts().plot(kind='bar')


# Variable classes are not equal so it need to oversample or remove from model.

# In[ ]:


data.corr()


# In[ ]:





# <h2>Features-vector proposition and model pipelining. Model validation</h2>

# First for all I use linear model for predicting MEDV variables. Check if variables have multicollinearity. Use VIF method to check it:

# In[ ]:


independent_values = filter(lambda x: x != 'MEDV', data.columns.values) 
cols = filter(lambda x: x != i ,independent_values)
y, X = dmatrices(formula_like= 'MEDV' + " ~" + " + ".join(cols), data=data, return_type="dataframe")
vif_values = [vif(X.values, i) for i in range(X.shape[1])]
v = dict(zip(X.columns.values, vif_values))
pd.DataFrame({k: [v] for k,v in v.items() }).drop('Intercept', axis=1).T.plot(kind='bar')


# Fit full-variable model to get coefficients and p-values to choose better parameters for next model. 

# In[ ]:


X = get_x()
y = get_y()
linearModelFit = sm.OLS(y, X).fit()
linearModelFit.summary()


# The best parameters for linear model are room count and population percent. They have the best $R^2$ score. But $RM$ and $DIS$ have the best $AIC$ and $BIC$ information parameters. TAX and RAD feature is multicollinear to $MEDV$ that's why these parameters was removed from model. $INDUS$, $AGE$ and $NOX$ variables do not statistically significant for this linear model and has low $AIC$ and $BIC$ information parameters in model. $B$ and $ZN$ have so many outliers and it can decrease model accuracy. $CHAS$ is statistically significant but features need to oversample, but sample is so small to do it.

# In[401]:


X = get_x()[['RM', 'DIS']]
X = MinMaxScaler().fit_transform(X)
linearModelFit = sm.OLS(y, X).fit()
linearModelFit.summary()


# In[ ]:


X = get_x()[['RM', 'LSTAT']]
X.LSTAT = stats.boxcox(X.LSTAT)[0]
X = MinMaxScaler().fit_transform(X)
linearModelFit = sm.OLS(y, X).fit()
linearModelFit.summary()


# Check theory about normal resudials of linear regression

# In[ ]:


res = linearModelFit.resid
fig, axes = plt.subplots(1, 2, figsize=(20,5))
sm.qqplot(res, stats.t, fit=True, line='45', ax=axes[0])
sns.kdeplot(res)


# Try to use sklearn model. It shows worse results than StatsModel OLS model

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(get_x()[['RM', 'LSTAT']], y)
lm_pipe =  Pipeline([('scaler', MinMaxScaler()), ('lm', LinearRegression())])
fit = lm_pipe.fit(X_train,y_train)
v = fit.predict(X_test)
display(r2_score(v, y_test))


# Try to XGBReggressor. It gives quite nicer result than simple sklearn linear regression.

# In[415]:


get_ipython().run_cell_magic('time', '', 'import xgboost as xgb\nfrom sklearn.model_selection import GridSearchCV\nfrom xgboost import plot_tree\n\ndef get_xgboost_best_model(X_train, y_train):\n    \n    params =[ \\\n            {\'xgb__n_estimators\': [5, 50, 100], \\\n             \'xgb__max_depth\': [5, 10, 20], \\\n             \'xgb__learning_rate\': [0.01, 0.1, 1], \\\n             \'xgb__reg_alpha\': [0.01, 0.1, 1, 10], \\\n             \'xgb__reg_lambda\': [0.01, 0.1, 1, 10],\n             \'xgb__gamma\': [0.01, 0.1, 1, 10]}]\n    \n    pipe = Pipeline([(\'scaler\', MinMaxScaler()), (\'xgb\', xgb.XGBRegressor()) ]) \n    grid = GridSearchCV(pipe, param_grid=params)\n    fit = grid.fit(X_train, y_train)\n    return fit.best_estimator_\n\nX_train, X_test, y_train, y_test = train_test_split(get_x(), y)\n\n\n\nfit = get_xgboost_best_model(X_train, y_train)\ndisplay(fit)\ndisplay("R2: " + str(r2_score(fit.predict(X_test), y_test)))')


# Try to choose most significant variables using important_plot

# In[427]:


from sklearn.model_selection import cross_val_score
from xgboost import plot_tree

best_model = fit.steps[1][1]
fitted_model = best_model.fit(X_train, y_train)
xgb.plot_importance(fitted_model, importance_type='weight')


# In[417]:


xgb.plot_importance(fitted_model, importance_type='cover')


# In[418]:


xgb.plot_importance(fitted_model, importance_type='gain')


# In[ ]:


plot_tree(model.fit(X_train, y_train))
fig = plt.gcf()
fig.set_size_inches(10, 5)


# In[434]:


X_feature_names = ['RM', 'LSTAT', 'NOX', 'DIS']
X = pd.DataFrame(MinMaxScaler().fit_transform(get_x()[X_feature_names]))
X.columns = X_feature_names
y = get_y()

X_train, X_test, y_train, y_test = train_test_split(X, y)
fitted_model = best_model.fit(X_train, y_train)
display("R2: " + str(r2_score(fitted_model.predict(X_test), y_test)))
plot_tree(fitted_model)
fig = plt.gcf()
fig.set_size_inches(25, 10)


# In[426]:


fig, ax = plt.subplots(1,1,figsize=(10,5))
xgb.plot_importance(fitted_model, importance_type='gain', ax=ax)

