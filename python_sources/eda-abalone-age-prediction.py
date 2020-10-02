#!/usr/bin/env python
# coding: utf-8

# ##  Abalone Age Prediction
# Description- Predicting the age of abalone from physical measurements. The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task. Other measurements, which are easier to obtain, are used to predict the age. Further information, such as weather patterns and location (hence food availability) may be required to solve the problem. 

# In this article I have focussed on exploratory data analysis on Abalone Dataset. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import  train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import  RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.linear_model import  Ridge
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))


# In[ ]:


# Read the dataset 
data = pd.read_csv('../input/abalone.csv')


# From problem statement and feature discription, let's first compute the target varible of the problem ' Age' and assign it to the dataset. 
# Age = 1.5+Rings

# In[ ]:


data['age'] = data['Rings']+1.5
data.drop('Rings', axis = 1, inplace = True)


# ## Univariate analysis
# Understanding feature wise statistics using various inbuilt tools 

# In[ ]:


print('This dataset has {} observations with {} features.'.format(data.shape[0], data.shape[1]))


# In[ ]:


data.columns


# In[ ]:


data.info()


# In[ ]:


data.describe()


# Key insights : 
#             - No missing values in the dataset
#             - All numerical features but 'sex'
#             - Though features are not normaly distributed, are close to normality
#             - None of the features have minimum = 0 except Height (requires re-check)
#             - Each feature has difference scale range

# In[ ]:


data.hist(figsize=(20,10), grid=False, layout=(2, 4), bins = 30)


# In[ ]:


numerical_features = data.select_dtypes(include=[np.number]).columns
categorical_features = data.select_dtypes(include=[np.object]).columns


# In[ ]:


numerical_features


# In[ ]:


categorical_features


# In[ ]:


skew_values = skew(data[numerical_features], nan_policy = 'omit')
dummy = pd.concat([pd.DataFrame(list(numerical_features), columns=['Features']), 
           pd.DataFrame(list(skew_values), columns=['Skewness degree'])], axis = 1)
dummy.sort_values(by = 'Skewness degree' , ascending = False)


# For normally distributed data, the skewness should be about 0. For unimodal continuous distributions, a skewness value > 0 means that there is more weight in the right tail of the distribution. The function skewtest can be used to determine if the skewness value is close enough to 0, statistically speaking.
#         - Height has highest skewedness followed by age, Shucked weight (can be cross verified through histogram plot)

# In[ ]:


# Missing values
missing_values = data.isnull().sum().sort_values(ascending = False)
percentage_missing_values = (missing_values/len(data))*100
pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing'])


# No missing values as said before

# In[ ]:


sns.countplot(x = 'Sex', data = data, palette="Set3")


# In[ ]:


plt.figure(figsize = (20,7))
sns.swarmplot(x = 'Sex', y = 'age', data = data, hue = 'Sex')
sns.violinplot(x = 'Sex', y = 'age', data = data)


#         Male : age majority lies in between 7.5 years to 19 years
#         Female: age majority lies in between 8 years to 19 years
#         Immature: age majority lies in between 6 years to < 10 years

# In[ ]:


data.groupby('Sex')[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
       'Viscera weight', 'Shell weight', 'age']].mean().sort_values('age')


# ## Bivariate Analysis
# Bivariate analysis is vital part of data analysis process for, it gives clear picture on how each features are affected in presence of other features.  
# It also helps us understand and identify significance features, overcome multi-collinearity effect, inter-dependency and thus, provides insights on hidden data noise pattern.

# In[ ]:


sns.pairplot(data[numerical_features])


# key insights
#             length is linearly correlated with diameter while, non-linear relation with height, whole weight, shucked weight, viscera weight and shell weight
#         
#         

# In[ ]:


plt.figure(figsize=(20,7))
sns.heatmap(data[numerical_features].corr(), annot=True)


#         Whole Weight is almost linearly varying with all other features except age
#         Heigh has least linearity with remaining features
#         Age is most linearly proprtional with Shell Weight followed by Diameter and length
#         Age is least correlated with Shucked Weight
#         
#   Such high correlation coefficients among features can result into multi-collinearity. We need to check for that too, however, I have not done it here.

# ## Outliers handlings

# In[ ]:


data = pd.get_dummies(data)
dummy_data = data.copy()


# In[ ]:


data.boxplot( rot = 90, figsize=(20,5))


# In[ ]:


var = 'Viscera weight'
plt.scatter(x = data[var], y = data['age'],)
plt.grid(True)


# In[ ]:


# outliers removal
data.drop(data[(data['Viscera weight']> 0.5) & (data['age'] < 20)].index, inplace=True)
data.drop(data[(data['Viscera weight']<0.5) & (data['age'] > 25)].index, inplace=True)


# In[ ]:


var = 'Shell weight'
plt.scatter(x = data[var], y = data['age'],)
plt.grid(True)


# In[ ]:


data.drop(data[(data['Shell weight']> 0.6) & (data['age'] < 25)].index, inplace=True)
data.drop(data[(data['Shell weight']<0.8) & (data['age'] > 25)].index, inplace=True)


# In[ ]:


var = 'Shucked weight'
plt.scatter(x = data[var], y = data['age'],)
plt.grid(True)


# In[ ]:


data.drop(data[(data['Shucked weight']>= 1) & (data['age'] < 20)].index, inplace=True)
data.drop(data[(data['Shucked weight']<1) & (data['age'] > 20)].index, inplace=True)


# In[ ]:


var = 'Whole weight'
plt.scatter(x = data[var], y = data['age'],)
plt.grid(True)


# In[ ]:


data.drop(data[(data['Whole weight']>= 2.5) & (data['age'] < 25)].index, inplace=True)
data.drop(data[(data['Whole weight']<2.5) & (data['age'] > 25)].index, inplace=True)


# In[ ]:


var = 'Diameter'
plt.scatter(x = data[var], y = data['age'],)
plt.grid(True)


# In[ ]:


data.drop(data[(data['Diameter']<0.1) & (data['age'] < 5)].index, inplace=True)
data.drop(data[(data['Diameter']<0.6) & (data['age'] > 25)].index, inplace=True)
data.drop(data[(data['Diameter']>=0.6) & (data['age']< 25)].index, inplace=True)


# In[ ]:


var = 'Height'
plt.scatter(x = data[var], y = data['age'],)
plt.grid(True)


# In[ ]:


data.drop(data[(data['Height']>0.4) & (data['age'] < 15)].index, inplace=True)
data.drop(data[(data['Height']<0.4) & (data['age'] > 25)].index, inplace=True)


# In[ ]:


var = 'Length'
plt.scatter(x = data[var], y = data['age'],)
plt.grid(True)


# In[ ]:


data.drop(data[(data['Length']<0.1) & (data['age'] < 5)].index, inplace=True)
data.drop(data[(data['Length']<0.8) & (data['age'] > 25)].index, inplace=True)
data.drop(data[(data['Length']>=0.8) & (data['age']< 25)].index, inplace=True)


# ## Preprocessing, Modeling, Evaluation
# The base steps followed in any data modeling pipelines are:
#                - pre-processing 
#                - suitable model selection
#                - modeling
#                - hyperparamaters tunning using GridSearchCV
#                - evaluation

# In[ ]:


X = data.drop('age', axis = 1)
y = data['age']


# In[ ]:


standardScale = StandardScaler()
standardScale.fit_transform(X)

selectkBest = SelectKBest()
X_new = selectkBest.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.25)


# In[ ]:


np.random.seed(10)
def rmse_cv(model, X_train, y):
    rmse =- (cross_val_score(model, X_train, y, scoring='neg_mean_squared_error', cv=5))
    return(rmse*100)

models = [LinearRegression(),
             Ridge(),
             SVR(),
             RandomForestRegressor(),
             GradientBoostingRegressor(),
             KNeighborsRegressor(n_neighbors = 4),]

names = ['LR','Ridge','svm','GNB','RF','GB','KNN']

for model,name in zip(models,names):
    score = rmse_cv(model,X_train,y_train)
    print("{}    : {:.6f}, {:4f}".format(name,score.mean(),score.std()))


# You have seen the perofrmance of each one of above models.
# 
# So, according to you which model should we start or choose?
# Well the answer lies in Occam's razor principle from philosophy https://simple.wikipedia.org/wiki/Occam%27s_razor." Suppose there exist two explanations for an occurrence. In this case the simpler one is usually better. Another way of saying it is that the more assumptions you have to make, the more unlikely an explanation."
# Hence, starting with the simplest model Ridge, for various reasons:
#             - Feature Dimension is less
#             - No misisng values
#             - Few categorical features

# In[ ]:



def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['age'])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    #dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = -cross_val_score(alg, dtrain[predictors], dtrain['age'], cv=cv_folds, 
                                                    scoring='r2')
    
    #Print model report:
    print ("\nModel Report")
    print( "RMSE : %.4g" % mean_squared_error(dtrain['age'].values, dtrain_predictions))
    print( "R2 Score (Train): %f" % r2_score(dtrain['age'], dtrain_predictions))
    
    if performCV:
        print( "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),
                                                                                 np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.coef_, predictors).sort_values(ascending=False)
        plt.figure(figsize=(20,4))
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')


# In[ ]:


# Base Model
predictors = [x for x in data.columns if x not in ['age']]
lrm0 = Ridge(random_state=10)
modelfit(lrm0, data, predictors)


# ## Hyperparameter tunning using GrideSearchCV

# In[ ]:


# Let's do hyperparameter tunning using GrideSearchCV
from sklearn.model_selection import  GridSearchCV
param  = {'alpha':[0.01, 0.1, 1,10,100],
         'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
glrm0 = GridSearchCV(estimator = Ridge(random_state=10,),
param_grid = param,scoring= 'r2' ,cv = 5,  n_jobs = -1)
glrm0.fit(X_train, y_train)
glrm0.best_params_, glrm0.best_score_


# In[ ]:


modelfit(Ridge(alpha = 0.1,random_state=10,), data, predictors)


# CV score has improved slightly while, R2_score has decreased showing base model was overfitted.
# Using above process multiple options can be tried to could up with much more robust model.
# This process can also be tried on different models : RF, GB, etc.

# Hyperparameter tunning is an iterative process and it can go on. As this kernal primary focuses on EDA of Abalone dataset, modeling building will be taken into another kernal ["Modeling - Abalone Age Prediction" ]. Hope I have helped you getting insights of Abalone dataset through this kernal.

# Motivate me so that I will come up soon with  "Modeling - Abalone Age Prediction" . :)
