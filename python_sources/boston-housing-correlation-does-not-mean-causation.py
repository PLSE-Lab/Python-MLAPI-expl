#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.datasets import load_boston
house = load_boston()


# In[ ]:


print(house.get("DESCR"))


# In[ ]:


print(house.keys())


# In[ ]:


print(house.feature_names)


# In[ ]:


print(house.data.shape)


# In[ ]:


bos = pd.DataFrame(house.data)


# In[ ]:


bos.head()


# In[ ]:


bos.columns = house.feature_names
print(bos.head())


# In[ ]:


print(house.target.shape)


# In[ ]:


bos['Price']=house.target


# In[ ]:


print(bos.head())


# In[ ]:


bos.describe()


# In[ ]:


#Visualisation
import seaborn as sns
sns.jointplot("INDUS","Price",data = bos,kind = "reg", xlim=(0, 60), ylim=(0, 12), color="m", height=7)
#Indus - proportion of non-retail business acres per town and price are not related


# In[ ]:


sns.jointplot("B","Price",data = bos,kind = "reg", xlim=(0, 60), ylim=(0, 12), color="m", height=7)


# In[ ]:


sns.jointplot("PTRATIO","Price",data = bos,kind = "reg", xlim=(0, 60), ylim=(0, 12), color="m", height=7)


# In[ ]:


sns.jointplot("TAX","Price",data = bos,kind = "reg", xlim=(0, 60), ylim=(0, 12), color="m", height=7)


# In[ ]:


sns.jointplot("CRIM","Price",data = bos,kind ="reg",xlim=(0, 60), ylim=(0, 12), color="m", height=7)
#shows a clear negative correlation between criminals presence and prices


# In[ ]:


sns.jointplot("ZN","Price",data = bos,kind = "reg",xlim=(0, 60),ylim = (0, 12),color = "m",height = 7)


# In[ ]:


sns.jointplot("DIS","Price",data = bos,kind = "reg",xlim=(0, 60),ylim = (0, 12),color = "m",height = 7)


# In[ ]:


correl = bos.corr()
sns.heatmap(correl,xticklabels = correl.columns,yticklabels = correl.columns)
#Price is positively correlated with ZN,RM,CHAS,DIS & B.
#Correlation does not mean causation.


# In[ ]:


sns.lmplot("Price","CRIM",data = bos)
sns.lmplot("Price","INDUS",data = bos)
sns.lmplot("Price","NOX",data = bos)
sns.lmplot("Price","AGE",data = bos)
sns.lmplot("Price","RAD",data = bos)
sns.lmplot("Price","TAX",data = bos)
sns.lmplot("Price","PTRATIO",data = bos)
sns.lmplot("Price","LSTAT",data = bos)


# In[ ]:


#Visualising variables that are identified to be correalted
sns.lmplot("Price","ZN",data = bos)
sns.lmplot("Price","RM",data = bos)
sns.lmplot("Price","CHAS",data = bos)
sns.lmplot("Price","DIS",data = bos)
sns.lmplot("Price","B",data = bos)


# In[ ]:


sns.pairplot(bos,x_vars = ["CRIM","INDUS","NOX","ZN","RM","CHAS","DIS","B","AGE","RAD","TAX","PTRATIO","LSTAT"], y_vars='Price', size=12, aspect=0.8, kind='reg')


# In[ ]:


sns.pairplot(bos)


# In[ ]:


#test train split

X = bos.drop('Price',axis = 1)
Y = bos.Price


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 2)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[ ]:


from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


def get_best_score(grid):
    
    best_score = np.sqrt(-grid.best_score_)
    print(best_score)    
    print(grid.best_params_)
    print(grid.best_estimator_)
    
    return best_score


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
linreg = LinearRegression()
nr_cv = 5
score_calc = 'neg_mean_squared_error'

parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
grid_linear = GridSearchCV(linreg, parameters, cv=nr_cv, verbose=1 , scoring = score_calc)
grid_linear.fit(x_train, y_train)
get_best_score(grid_linear)


# In[ ]:


model = XGBClassifier()
model.fit(x_train, y_train)


# In[ ]:


print(model)


# In[ ]:


y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]


# In[ ]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,y_pred)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
# Fit model
tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(x_train,y_train)


# In[ ]:


r1 = y_train - tree_reg1.predict(x_train)


# In[ ]:


tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(x_train,r1)


# In[ ]:


# Compute errors/residuals on second tree
r2 = r1 - tree_reg2.predict(x_train)


# In[ ]:


# Fit third model
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(x_train,r2)


# In[ ]:


y__pred = sum(tree.predict(x_train) for tree in (tree_reg1, tree_reg2, tree_reg3))


# In[ ]:


y__pred[:10]


# In[ ]:


y_train[:10]


# In[ ]:


predictions = pd.DataFrame(tree_reg1.predict(x_train)[:10], columns=['Model_1'])
predictions['Model_2'] = pd.DataFrame(tree_reg2.predict(x_train)[:10])
predictions['Model_3'] = pd.DataFrame(tree_reg3.predict(x_train)[:10])
predictions['Ensemble'] = pd.DataFrame(y_pred[:10])
predictions['Actual'] = y_train.head(10).reset_index()['Price']

# Display predictions
predictions

