#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Importing Libraries

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb


# In[ ]:


df = pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")
test = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


#Dropping the irrelevant columns
df = df.drop(['Serial No.'],axis = 1)
df.head()


# In[ ]:


#Lets see the correlations of variables
sns.set(style="white")
corr = df.corr()
f , ax = plt.subplots(figsize=(11,9))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
colormap = sns.diverging_palette(220,20,as_cmap=True)
sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=mask)
plt.show()


# In[ ]:


#visualising distributions of variables
fig = sns.distplot(df['GRE Score'])
plt.title("Distribution of GRE Scores")
plt.show()


# In[ ]:





# In[ ]:


#Distribution of TOEFL score
fig = sns.distplot(df['TOEFL Score'])
plt.title("Distribution of TOEFL Scores ")
plt.show()

#Distribution of University Ranking
fig = sns.distplot(df['University Rating'])
plt.title("Distribution of University Rankings")
plt.show()

#Distribution of SOP 
fig = sns.distplot(df['SOP'])
plt.title("Distribution of SOP ")
plt.show()

#Distribution of CGPA
fig = sns.distplot(df['CGPA'])
plt.title("Distribution of CGPA")
plt.show()

#Distribution of Research
fig = sns.distplot(df['Research'])
plt.title("Distribution of Research")
plt.show()


# By Above Distributions it can be known that students with different number of ,CGPA,GRE ,TOEFL scores apply for admission. Also Students belonging to tier 2 University tend to apply more for Admission.

# Hence Finding relation between CGPA, TOEFL,GRE and University Ranking

# In[ ]:


sns.regplot(x='GRE Score',y='TOEFL Score',data = df)
plt.title("GRE vs TOEFL Scores")
plt.show()


# By Above Plot we conclude that people with more GRE Scores are more likely to have high TOEFL Scores.

# In[ ]:


sns.regplot(x='CGPA',y = 'GRE Score',data = df)
plt.title("CGPA vs GRE Score")
plt.show()

sns.regplot(x='CGPA',y='TOEFL Score',data = df)
plt.title("CGPA vs TOEFL Score")
plt.show()


# People with High CGPA are more likely to have High GRE and TOEFL Scores 

# In[ ]:


sns.catplot(x='University Rating',y = 'GRE Score',data = df,kind="boxen")
plt.title("University Rating vs GRE Scores")
plt.show()


# By Above plot we conclude that students which belong to University with Rating 4,5 score more in GRE test.Also Students from University with Rating 2,3 score in average region.

# In[ ]:


sns.catplot(x="University Rating",y="TOEFL Score",data = df)
plt.title("University Rating vs TOEFL Score")
plt.show()


# By Above plot we conclude that students which belong to University with Rating 4,5 score more in TOEFL

# Hence we can see a common trend here that a person with good CGPA  and from Universities whose rank are 4 or 5 are more likely to score high score in GRE and TOEFL.

# In[ ]:





# In[ ]:


x = df.iloc[:,:-1].values


# In[ ]:


y = df.iloc[:,-1].values


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0,shuffle=False)


# **Algorithm Selection**

# **XGBOOST**

# In[ ]:


clf1 = xgb.XGBRFRegressor()


# In[ ]:


n_estimators = [100,150,200,500,900,1000]
booster = ['gblinear','gbtree']
max_depth = [4,5,6,8,9]
learning_rate = [0.001,0.005,0.01,0.2,0.3]
base_score = [0.20,0.25,0.50,0.75,1.0]


# In[ ]:


hyperparameter_grid = {
    'n_estimators':n_estimators,
    'booster':booster,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'base_score':base_score
}

xgb_clf = RandomizedSearchCV(estimator=clf1,
                            param_distributions=hyperparameter_grid,
                            cv = 5,
                             n_iter = 25,
                            random_state=0,
                            n_jobs = -1,
                            scoring = 'neg_mean_absolute_error',
                             return_train_score=True
                             
                            )


# In[ ]:


xgb_clf.fit(x_train,y_train)


# In[ ]:


xgb_clf.best_estimator_


# In[ ]:


regressor_1 = xgb.XGBRFRegressor(base_score=0.25, booster='gblinear', colsample_bylevel=None,
               colsample_bytree=None, gamma=None, gpu_id=-1,
               importance_type='gain', interaction_constraints=None,
               learning_rate=0.3, max_delta_step=None, max_depth=6,
               min_child_weight=None, missing=None, monotone_constraints=None,
               n_estimators=900, n_jobs=0, num_parallel_tree=None,
               objective='reg:squarederror', random_state=0, reg_alpha=0,
               scale_pos_weight=1, tree_method=None, validate_parameters=1,
               verbosity=None)


# In[ ]:


regressor_1.fit(x_train,y_train)


# In[ ]:


y_pred = regressor_1.predict(x_test)
y_pred_r = np.round(y_pred,2)
y_pred_r


# In[ ]:


from sklearn.metrics import mean_squared_error
print("Error: ",np.sqrt(mean_squared_error(y_test,y_pred_r)))


# LIGHTGBM

# In[ ]:


clf_2 = lgb.LGBMRegressor()


# In[ ]:


num_leaves = [50,100,150,200,250,300,500]
min_child_weight = [0.01,0.02,0.05,0.09,0.10,0.15,0.20,0.3,0.5]
feature_fraction = [0.10,0.15,0.20,0.25,0.30,0.40]
boosting_type = ['gbdt'],
learning_rate = [0.01,0.05,0.09,0.10,0.15]
n_estimators = [100,200,300,350,450,500,700]
parameters = {
    'n_estimators':n_estimators,
    'min_child_weight': min_child_weight,
    'feature_fraction':feature_fraction,
    'learning_rate': learning_rate,
    'num_leaves':num_leaves
}


# In[ ]:


clf_lgb = RandomizedSearchCV(estimator=clf_2,
                            param_distributions=parameters,
                             random_state=0,
                            cv = 5,
                            n_iter = 25,
                            n_jobs = -1,
                            scoring='neg_mean_absolute_error')


# In[ ]:


clf_lgb.fit(x_train,y_train)


# In[ ]:


clf_lgb.best_estimator_


# In[ ]:


regressor_2 = lgb.LGBMRegressor(feature_fraction=0.15, learning_rate=0.05, min_child_weight=0.15,
              num_leaves=500)


# In[ ]:


regressor_2.fit(x_train,y_train)


# In[ ]:


y_pred2 = regressor_2.predict(x_test)
y_pred2_r = np.round(y_pred2,2)
y_pred2_r


# In[ ]:


y_test


# In[ ]:


print("Error: ",np.sqrt(mean_squared_error(y_test,y_pred2_r)))


# Hence we are getting better accuracy with LightGBM

# In[ ]:


Imp_Factors = pd.DataFrame()
factors = df.columns[:-1]
Imp_Factors['Factors'] = factors
Imp_Factors['Importance'] = regressor_2.feature_importances_
Imp_Factors = Imp_Factors.sort_values(by=['Importance'],ascending=True)


# In[ ]:


plt.figure(figsize=(11,9))
sns.barplot(x='Importance',y='Factors',data = Imp_Factors)
plt.show()


# Hence we conclude that CGPA is the most important factor or determining the Admission into Graduate Schools 

# ### **Testing Dataset**

# In[ ]:


test.head()


# In[ ]:


test.shape


# In[ ]:


test.info()


# In[ ]:


X = test.iloc[:,1:-1]
y = test.iloc[:,-1]


# In[ ]:


y_pred_test = regressor_2.predict(X)


# In[ ]:


print("Error: ",np.sqrt(mean_squared_error(y,y_pred_test)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




