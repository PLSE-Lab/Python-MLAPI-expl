#!/usr/bin/env python
# coding: utf-8

# This notebook is built to make a comparison between linear methods and ensemble methods in dealing with classification and regression problem. EDA would be made first, and then two specific problems would be introduced and solved.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest
from scipy.stats import shapiro

import sklearn
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import partial_dependence

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[ ]:


mat_data = pd.read_csv('../input/student-mat.csv')
print(mat_data.shape)
mat_data.head()


# **Exploratory Data Analysis**
# 
# First, let's take a brief look at what we are going to predict! EDA, classification and regression would all be based on G3(Final Grade), so here is the bar chart and kernel density plot.

# In[ ]:


sns.distplot(mat_data['G3'],kde=True)


# It is kind of symmetrical, with just a few students in the highest and lowest grade level. However, there are several students getting 0, which is not normal because nobody gets 0 in the first test(G1). So I believe these students might abandon the test. Shapiro-Wilk test for normality shows it is not normally distributed, with an emtremely low p-value (null hypothesis normality).

# In[ ]:


print("shapiro p-value:", str(shapiro(mat_data['G3'])[1]))


# Next, we get the overall alcohol comsumption amount by adding the amount during weekdays and amount during weekends. And let's define students 'making progress'. That is, if a student's final grade is equal to or higher than the average grade of previous tests, he or she does make a prgress. This is what we are going predict later in the classification problem.

# In[ ]:


###get the overall alcohol comsumption amount
mat_data['alc'] = mat_data['Dalc'] + mat_data['Walc']


# Students who abandoned the test are regarded as outliers. Instead of dropping them, I make a truncation, that is to say, I replace these zeros with 3. the lowest score in G1. This is called modified G3, or MG, which will be used in regression problem.

# In[ ]:


###deal with the outliers (truncation)
mat_data['MG']=mat_data['G3']
mat_data['MG'][mat_data['G3']<=3]=3


# The graph below is the bar chart and the kernel density of modified G3, which shows that it might not be normally distributed. Shapiro-Wilk test for normality also get this conclusion, with an emtremely low p-value.

# In[ ]:


sns.distplot(mat_data['MG'],kde=True)
print("shapiro p-value", str(shapiro(mat_data['MG'])[1]))


# The thermodynamic diagram of correlation matrix visualizes the linear relationship between the numeric features and grades.

# In[ ]:


plt.figure(figsize=(15,15))
data_cor=mat_data[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
       'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
       'freetime', 'goout', 'health', 'absences','alc', 'G1','MG']].corr()
sns.heatmap(data_cor,annot = True,fmt = ".2f",cbar = True)
plt.xticks(rotation=50)
plt.yticks(rotation = 0)


# There are quite a few interesting discoveries here in this graph, like the relationship between goout and alcohol, study time and alcohol and so on.
# The relationship between categorical features and modified G3 can be presented using box plot.

# In[ ]:


for cat in ["Mjob","Fjob","reason","guardian","schoolsup","romantic"]:
    plt.figure()
    sns.boxplot(x=cat, y="MG", data=mat_data)
    sns.stripplot(x=cat, y='MG', hue=None, data=mat_data, order=None, 
                 hue_order=None, jitter=True, split=False, 
                 orient=None, color=None, palette=None, 
                 size=5, edgecolor="gray", linewidth=0, 
                 ax=None)
    


# To deal  with the categorical features, one-hot encoding method is used. We also check whether the data is complete.

# In[ ]:


n_NA=mat_data.isnull().sum().sum()
print('There are %d NA(s) in the data set.'%n_NA)
mat_data.head()


# In[ ]:


###Deal with the categorical variables
mat_data["sex"] = mat_data["sex"].map({"F": 1, "M":0})
mat_data["school"] = mat_data["school"].map({"GP": 1, "MS":0})
mat_data["address"] = mat_data["address"].map({"U": 1, "R":0})
mat_data["famsize"] = mat_data["famsize"].map({"LE3": 1, "GT3":0})
mat_data["Pstatus"] = mat_data["Pstatus"].map({"A": 1, "T":0})
for col in ["schoolsup","famsup","paid","activities","nursery","higher","internet","romantic"]:
    mat_data[col] = mat_data[col].map({"yes": 1, "no":0})

dummy_col = ["Mjob","Fjob","reason","guardian"]
mat_data = pd.get_dummies(mat_data, columns = dummy_col, prefix=dummy_col)


# In[ ]:


mat_data.columns


# **Prediction of Modified G3**
# 
# Let's start modelling! In this section, a regression problem is to be solved. Features including students' background and the first period grade (G1) would be used to predict Modified G3. Since G3 are numeric values, this would be a regression problem.
# 
# A 5-fold cross validation MSE is used to assess the model accuracy and make tuning parameters. Plots of observations and fitted values are presented to see how well the model fit the training data. The closer the points are to the y=x line, the better the fitting model.

# In[ ]:


X = mat_data[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
       'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
       'freetime', 'goout', 'health', 'absences','alc', 'Mjob_at_home', 'Mjob_health',
       'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_at_home',
       'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher',
       'reason_course', 'reason_home', 'reason_other', 'reason_reputation',
       'guardian_father', 'guardian_mother', 'guardian_other','G1']]
y = mat_data['MG']
X.head(3)


# **Linear Regression (OLS)**

# In[ ]:


reg_lr = LinearRegression()
scores = cross_val_score(reg_lr,X, y, cv= 5,scoring="neg_mean_squared_error")


# In[ ]:


comp = dict({"OLS":-scores.mean()})
print('The cross validation MSE of linear regression is %2.2f .'%-scores.mean())


# Now let's see how it performs in fitting the training data.

# In[ ]:


reg_lr.fit(X,y)
lr_pred = reg_lr.predict(X)
plt.figure(figsize=(8,6))
plt.scatter(lr_pred, y,marker='.')
xx=np.arange(0,25,0.1)
yy=xx
plt.plot(xx,yy,'--')
plt.title("Predicted Results of OLS")
plt.xlabel("Predicted Values")
plt.ylabel("Observations")
print('The R^2 of this regression model is %2.2f.'%reg_lr.score(X,y))  ###R^2


# Not so bad, with R^2=0.77. But there might be some problems fitting the low grade students' G3. Let's see how other models perform.

# **Lasso**

# In[ ]:


param_grid = [{'alpha':[0.1,0.15,0.16,0.17,0.2,0.25],'normalize':[True,False],'fit_intercept':[True,False]}]
model = Lasso( )
reg_lasso = GridSearchCV(model, param_grid,cv=5,n_jobs=4,scoring="neg_mean_squared_error")
reg_lasso.fit(X, y)


# In[ ]:


comp.update({"Lasso":-reg_lasso.best_score_})
print('The cross validation MSE of Lasso is %2.2f .'%-reg_lasso.best_score_)


# In[ ]:


def model_result(model,label):
    model.fit(X,y)
    plt.figure(figsize=(8,6))
    pred = model.predict(X)
    model_plt=plt.scatter(pred, y,c='r',marker='.',label=label)
    olsplt=plt.scatter(lr_pred, y,c='b',marker='.',label="OLS")
    xx=np.arange(0,25,0.1)
    yy=xx
    plt.plot(xx,yy,'--')
    plt.title("Predicted Results of "+label)
    plt.xlabel("Predicted Values")
    plt.ylabel("Observations")
    plt.legend(handles=[model_plt,olsplt],loc=4)

model_result(reg_lasso.best_estimator_,label = "Lasso")


# Although Lasso does not perform better than OLS linear regression obviously in fitting the training data, it does make a lower CV MSE and a higher accuracy.

# **Ridge Regression**

# In[ ]:


param_grid = [{'alpha':[0.1,0.5,1,80,90,100,110,120,],'normalize':[True,False],'fit_intercept':[True,False]}]
model = Ridge( )
reg_rid = GridSearchCV(model, param_grid,cv=5,n_jobs=4,scoring="neg_mean_squared_error")
reg_rid.fit(X, y)


# In[ ]:


comp.update({"Ridge":-reg_rid.best_score_})
print('The cross validation MSE of Rdige Regression is %2.2f .'%-reg_rid.best_score_)


# In[ ]:


model_result(reg_rid.best_estimator_,label = "Ridge")


# Then it is time for ensemble learning method! Several decision tree based ensemble methods including Ramdom Forest and Gradient Boosting are carried out. Note that the parameters in the dictionary 'param_grid' in the code do not cover the entire range in tuning parameter. In order to save the running time, only a smaller range of parameters are covered here.
# 
# **Random Forest**

# In[ ]:


param_grid = [{'n_estimators':[50,100,200],'max_features':['auto'], 
               'max_depth':[5,10,None],'min_samples_split':[2,5,10],'min_samples_leaf':[7,9,11], 'oob_score':[True]}]
model = RandomForestRegressor(n_jobs=4)
reg_rf = GridSearchCV(model, param_grid,cv=5,n_jobs=4,scoring="neg_mean_squared_error")
reg_rf.fit(X, y)


# In[ ]:


comp.update({"Random Forest":-reg_rf.best_score_})
print('The cross validation MSE of Random Forest is %2.2f .'%-reg_rf.best_score_)


# In[ ]:


model_result(reg_rf.best_estimator_,label = "RandomForest")


# No matter in terms of CV MSE or fitting effect, Random Forest, although takes a longer time, performs better than all the linear methods above.
# 
# **Gradient Boosting**

# In[ ]:


param_grid = [{'loss':['ls'],'n_estimators':[100, 200,300], 
               'max_depth':[5,10,20],'min_samples_split':[15,20,25],'min_samples_leaf':[5,10,15], 
               'max_features':['auto',10,30,None],'learning_rate':[0.001,0.01,0.1]}]
model = GradientBoostingRegressor( )
reg_gb = GridSearchCV(model, param_grid,cv=5,n_jobs=4,scoring="neg_mean_squared_error")
reg_gb.fit(X, y)


# In[ ]:


comp.update({"Gradient Boosting":-reg_gb.best_score_})
print('The cross validation MSE of Gradient Boosting is %2.2f .'%-reg_gb.best_score_)


# In[ ]:


model_result(reg_gb.best_estimator_,label = "GradientBoosting")


# Gradient Boosting gets a lower CV MSE than Random Forest and the linear models above. We are going to make feature selection based on Gradient Boosting model to see whether this would further imporve the performance.

# In[ ]:


reg_gb.best_estimator_.fit(X,y)
X_gb=X[X.columns[np.abs(reg_gb.best_estimator_.feature_importances_)>0.005]]
X_gb.columns


# In[ ]:


#Feature Selection of gb
param_grid = [{'loss':['lad','ls'],'n_estimators':[50,100,200], 
               'max_depth':[5,10,20],'min_samples_split':[3,5,10],'min_samples_leaf':[5,10,15], 
               'max_features':['auto',5,None],'learning_rate':[0.01,0.1]}]
model = GradientBoostingRegressor()
reg_gb = GridSearchCV(model, param_grid,cv=5,n_jobs=4,scoring="neg_mean_squared_error")
reg_gb.fit(X_gb, y)
reg_gb.best_estimator_


# In[ ]:


comp.update({"GB after features selection":-reg_gb.best_score_})
print('The cross validation MSE of Gradient Boosting after feature selection is %2.2f .'%-reg_gb.best_score_)


# In[ ]:


reg_gb.best_estimator_.fit(X,y)
rank_gb=pd.Series(data=reg_gb.best_estimator_.feature_importances_, index=X.columns).rank(ascending=False)
sns.barplot(x=X.columns[np.abs(reg_gb.best_estimator_.feature_importances_)>0.015], 
                        y = reg_gb.best_estimator_.feature_importances_[np.abs(reg_gb.best_estimator_.feature_importances_)>0.015])


# In[ ]:


for s in comp.keys():
  print(s+":"+str(comp[s]))


# Glad to see it is really improved a little after feature selection. 
# 
# From the analysis above, we can see that Lasso and ridge regression perform better than OLS linear regression because of regularization, while ensemble methods are even better than linear methods in terms of fitting the training data and cross validation, and feature selection can further imporve the accuracy.

# 
