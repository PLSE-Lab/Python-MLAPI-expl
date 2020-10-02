#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

suicide = pd.read_csv('../input/master.csv',index_col=0,parse_dates=[0])

#Originally "Country" is the index column, this creates an index column
suicide = suicide.reset_index()


#we delete columns we dont need
del suicide['country-year']         #Redundant
del suicide['HDI for year']         #Dont know what it is
del suicide['suicides_no']          #Collapes into Suicides/pop
del suicide['population']
del suicide[' gdp_for_year ($) ']   #We keep gdp_per_capita

#We create a new categorical variable "Region" with all null values
header = ['country',
 'year',
 'sex',
 'age',
 'suicides/100k pop',
 'gdp_per_capita ($)',
 'generation',
 'region']

suicide = suicide.reindex(columns = header)        

#We manually put all the differnt countries into one of 6 regions
Europe = ["Albania","Russian Federation","France","Ukraine","Germany","Poland","United Kingdom",
         "Italy","Spain","Hungary","Romania","Belgium","Belarus","Netherlands","Austria",
         "Czech Republic","Sweden","Bulgaria","Finland","Lithuania","Switzerland","Serbia",
         "Portugal","Croatia","Norway","Denmark","Slovakia","Latvia","Greece","Slovenia",
         "Turkey","Estonia","Georgia","Albania","Luxembourg","Armenia","Iceland","Montenegro",
         "Cyprus","Bosnia and Herzegovina","San Marino","Malta","Ireland"]
NorthAmerica = ["United States","Mexico","Canada","Cuba","El Salvador","Puerto Rico",
                "Guatemala","Costa Rica","Nicaragua","Belize","Jamaica"]
SouthAmerica = ["Brazil","Colombia", "Chile","Ecuador","Uruguay","Paraguay","Argentina",
                "Panama","Guyana","Suriname"]
MiddleEast = ["Kazakhstan","Uzbekistan","Kyrgyzstan","Israel","Turkmenistan","Azerbaijan",
              "Kuwait","United Arab Emirates","Qatar","Bahrain","Oman"]
Asia = ["Japan","Republic of Korea", "Thailand", "Sri Lanka","Philippines","New Zealand",
        "Australia","Singapore","Macau","Mongolia"]

#if the country belongs to a region, we assign the observation with the region
for i in range(0,len(suicide)):
    if suicide.iloc[i,0] in Europe:
        suicide.iloc[i,7] = "Europe"
    elif suicide.iloc[i,0] in NorthAmerica:
        suicide.iloc[i,7] = "North America"
    elif suicide.iloc[i,0] in SouthAmerica:
        suicide.iloc[i,7] = "South America"
    elif suicide.iloc[i,0] in MiddleEast:
        suicide.iloc[i,7] = "Middle East"
    elif suicide.iloc[i,0] in Asia:
        suicide.iloc[i,7] = "Asia"
    else:
        suicide.iloc[i,7] = "Island Nation"

#Now that we dont need "country", we delete it. 
del suicide['country']

#We collect our categorial variables for OneHotEncoding
suicide_cat = suicide[['sex','age','generation','region']]
one_hot_data = pd.get_dummies(suicide_cat)

#We merge the data back together
year = suicide['year']
gdp_per_cap = suicide['gdp_per_capita ($)']
suicide_per_100k = suicide['suicides/100k pop']
data = pd.concat([year, gdp_per_cap, one_hot_data], axis=1)

#Now that the data is clean(er), we train some models.
    #We do a DecisionTreeRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, suicide_per_100k, test_size=0.4, random_state=42)

#A Cross-Validation set and a Test set. 
X_cv, X_test, y_cv, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

#We use a GridSearchCV to search for the best hyperparameters. In total we sampled 21000 different trees.
#I've truncated it for speed. 
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

params = {'max_leaf_nodes': list(range(93,95)), 'min_samples_split': list(range(6,8)), 'min_samples_leaf':list(range(2,4))}    
grid_search_cv = GridSearchCV(DecisionTreeRegressor(random_state=42),
                              params, n_jobs=-1, verbose=1, cv=3)

grid_search_cv.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error

y_pred = grid_search_cv.predict(X_cv)
tree_reg_mse = mean_squared_error(y_cv, y_pred)
tree_reg_rmse = np.sqrt(tree_reg_mse)
print("The Root-Mean-Squared Error for the CV set in a Decision Tree Regression model is :",tree_reg_rmse)


#Based on the scatter plots below, there doesnt look like a linear relationship,and thus Linear
#regression would most likely not produce a good model. 

from pandas.plotting import scatter_matrix
attributes = ['suicides/100k pop','year','gdp_per_capita ($)']
scatter_matrix(suicide[attributes], figsize=(12,8))

#We perform a linear regression just for fun. 
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

y_pred_lin_reg = lin_reg.predict(X_cv)
lin_reg_mse = mean_squared_error(y_cv, y_pred_lin_reg)
lin_reg_rmse = np.sqrt(lin_reg_mse)
print("The Root-Mean-Squared Error for the CV set in a linear regression model is :",lin_reg_rmse)

#The Decision Tree is better w.r.t. the Cross-Validation data. 
print("The Decision Tree is better w.r.t. the Cross-Validation data.")

#We get our RMSE for the test set.


y_pred = grid_search_cv.predict(X_test)
tree_reg_mse = mean_squared_error(y_test, y_pred)
tree_reg_rmse = np.sqrt(tree_reg_mse)
print("The Root-Mean-Squared Error for the Test set in a Decision Tree Regression model is :",tree_reg_rmse)






