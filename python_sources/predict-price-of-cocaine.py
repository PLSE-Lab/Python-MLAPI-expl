#!/usr/bin/env python
# coding: utf-8

# # Predicting the Price of Dark Market Cocaine

# ---

# In this project we used various machine learning models to predict the btc price of dark market cocaine.  
# 
# The dataset is composed of approximately 1,400 cleaned and standardized product listings from Dream Market's "Cocaine" category. It was collected with web-scraping and text extraction techniques in July 2017.  See the README.md in our Github repository for more information.

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Group-Name" data-toc-modified-id="Group-Name-1">Group Name</a></span></li><li><span><a href="#Student-Names" data-toc-modified-id="Student-Names-2">Student Names</a></span></li><li><span><a href="#Load-Data" data-toc-modified-id="Load-Data-3">Load Data</a></span></li><li><span><a href="#Data-Cleaning-and-Feature-Engineering" data-toc-modified-id="Data-Cleaning-and-Feature-Engineering-4">Data Cleaning and Feature Engineering</a></span></li><li><span><a href="#Initial-fit-for-linear-regression,-l1,-l2-and-random-forest-and-cross-validation" data-toc-modified-id="Initial-fit-for-linear-regression,-l1,-l2-and-random-forest-and-cross-validation-5">Initial fit for linear regression, l1, l2 and random forest and cross validation</a></span></li><li><span><a href="#Random-Forest-Random-Search-for-Hyperparameter-Tuning" data-toc-modified-id="Random-Forest-Random-Search-for-Hyperparameter-Tuning-6">Random Forest Random Search for Hyperparameter Tuning</a></span></li><li><span><a href="#Evaluation-Metrics" data-toc-modified-id="Evaluation-Metrics-7">Evaluation Metrics</a></span></li><li><span><a href="#Interpretation" data-toc-modified-id="Interpretation-8">Interpretation</a></span></li><li><span><a href="#Visualization" data-toc-modified-id="Visualization-9">Visualization</a></span></li></ul></div>

# Group Name
# -----

#  Random_Forest_Elves
# 

# Student Names
# ----

# 1. Shirley Li
# 2. Jingxian Li 
# 3. Michael Schulze
# 4. Mundy Reimer

# Load Data
# -----

# In[ ]:


reset -fs


# In[ ]:


import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor


from sklearn.metrics import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt 


# In[ ]:


df = pd.read_csv('fixed_cocaine_listings.csv')


# In[ ]:


df.head()


# Data Cleaning and Feature Engineering
# -----

# In[ ]:


# Split predictors & response variables, 
# do log transformation on grams, 
# and convert all True-False to 1s and 0s.

y = df.btc_price
df_b = df.drop("btc_price", axis=1)
df_b = df_b.drop(["cost_per_gram_pure", "Unnamed: 0", "product_title", "ships_from_to", "cost_per_gram",
                    "product_link", "vendor_link", "vendor_name", "ships_from", "ships_to", 
                    "ships_to_GR", "ships_from_GR", "ships_to_PL", "ships_from_PL", "ships_to_CO", 
                    "ships_from_CO", "ships_to_SE", "ships_from_SE", "ships_to_S. America", 
                    "ships_from_S. America", "ships_from_DK", "ships_to_DK", "ships_to_CN", "ships_to_CZ", 
                    "ships_to_BR", "ships_to_IT", "ships_from_SI", "ships_to_BE", "ships_from_N. America", 
                    "ships_to_ES", "ships_to_CH", "ships_from_CH", "ships_from_CZ", "ships_from_CN", 
                    "ships_to_WW", "ships_to_DE"], axis=1)

df_b = df_b * 1
df_b["grams"] = np.log(df_b["grams"])
y = np.log(y)

df_b.head()


# In[ ]:


# Extract key words from product title for feature engineering

my_regex = {"intro": "intro|sample",
           "columbia": "columbia",
           "peru": "peru",
           "bolivia": "bolivia",
           "free_ship": "freeship",
           "uncut": "uncut",
           "fishcut": "fish",
           "brick": "brick",
           "crack": "crack",
           "crystal": "crystal",
           "flake": "flake",
           "pure": "pure"}

for my_key in my_regex.keys():
    df_b.loc[df.product_title.str.replace("\s", "")                                .str.lower()
                                .str.contains(my_regex[my_key]), my_key] = 1
    df_b.loc[~df.product_title.str.replace("\s", "")                                .str.lower()
                                .str.contains(my_regex[my_key]), my_key] = 0


df_b["caps"] = df.product_title.str.findall(r'[A-Z]').str.len()/df.product_title.str.len()

df_b.head()


# Data Visualization
# ----

# In[ ]:


data = pd.read_csv('data_keywords.csv')


# In[ ]:


df['ship_from'] = df.ships_from_to.str[:2]
df.head()
ship_from = df.groupby('ship_from').agg(
    {'btc_price': ['mean', 'count']}).reset_index()
ship_from = pd.concat([ship_from['ship_from'], ship_from['btc_price']], axis=1)
ship_from.loc[ship_from['count'] < 5, 'count_less_than_5'] = 'Yes'
ship_from.loc[ship_from['count'] >= 5, 'count_less_than_5'] = 'No'
ship_from = ship_from.sort_values('mean', ascending=False)
ship_from


# In[ ]:


colors = {'Yes': 'coral', 'No': 'dodgerblue'}
ship_from_plt = ship_from.rename(columns={'mean': 'count_is_less_than_5'})
ax = ship_from_plt.plot(x='ship_from', y='count_is_less_than_5', kind='bar',
                        color=ship_from_plt['count_less_than_5'].apply(lambda x: colors[x]), figsize=(10, 5), width=0.8)
ax.set_xlabel('Ship From Which Country')
ax.set_ylabel('mean of price')
ax.set_title('Mean Cocaine Price for each Country Vender')
plt.show()


# In[ ]:


col_lst = list(data.columns[-13:])
col_lst.append('escrow')
col_lst.remove('caps')
col_lst


# In[ ]:


def plot_avg_price(col_name):
    new_dataset = data.groupby(col_name).mean()['btc_price'].reset_index()
    new_dataset.plot(x=col_name, y='btc_price', kind='bar', legend=False, figsize=(2,5))
    plt.show()


# In[ ]:


def plot_graphs(col_lst):
    fig, axs = plt.subplots(3, 5, figsize=(20, 20))
    i = 0
    for col in col_lst:
        row_index = int(i / 5)
        col_index = i % 5
        new_dataset = data.groupby(col).mean()['btc_price'].reset_index()
        axs[row_index, col_index].bar(
            new_dataset[col], new_dataset['btc_price'])
        axs[row_index, col_index].set_title(col)
        axs[row_index, col_index].set_title(col)
        axs[row_index, col_index].set_ylabel('btc_price')
        i += 1
plot_graphs(col_lst)


# In[ ]:


# Train test split
X_train, X_test, y_train, y_test = train_test_split(df_b, y, test_size=0.2,shuffle = True)


# Initial Fit of Models & Cross Validation Scores
# ----

# Models that were used and placed into pipeline include Linear Regression, L1, L2, and Random Forest.

# In[ ]:


print("Cross Validation Score")
pipelines = [LinearRegression(),
             Lasso(),
             Ridge(),
             RandomForestRegressor(criterion='mae')]

for pipe in pipelines:
    pipe.fit(X_train,y_train)
    name = pipe.__class__.__name__.split('.')[-1]
    cv_medae = cross_val_score(pipe, X_train,y_train, scoring = 'neg_median_absolute_error', cv =5)
    cv_score = cross_val_score(pipe, X_train,y_train, scoring = 'r2', cv =5)
    print(f"{name}")
    print(f"Average cross validation R^2: {cv_score.mean():.4}")
    print(f"{cv_score}")
    print(f"Average cross validation Medae: {cv_medae.mean():.4}")
    print(f"{cv_medae}",end = "\n\n")


# Random Forest Random Search for Hyperparameter Tuning
# ----

# In[ ]:


cv = 5
n_iter = 20


# In[ ]:


hyperparameters = dict(n_estimators=range(10, 200),
                       max_depth=range(3, 12))
rf_random = RandomizedSearchCV(RandomForestRegressor(
    criterion='mae', random_state=42), hyperparameters, cv=cv, n_iter=n_iter)


# In[ ]:


hyperparameters = dict(n_estimators=range(10, 200),
                       max_depth=range(3, 12))
rf_random = RandomizedSearchCV(RandomForestRegressor(
    criterion='mae', random_state=42), hyperparameters,n_iter = n_iter)


# In[ ]:


rf_random.fit(X_train, y_train)


# In[ ]:


cross_val_score(rf_random, X_train, y_train)


# Model Comparison & Evaluation Metrics
# ----

# Both RMSE and MEDAE were used as evaluation metrics with MEDAE being our North Star Metric.

# In[ ]:


y_test = np.exp(y_test)
y_train = np.exp(y_train)
X_test["grams"] = np.exp(X_test["grams"])
X_train["grams"] = np.exp(X_train["grams"])


# In[ ]:


for pipe in pipelines:
    name = pipe.__class__.__name__
    pred_test = pipe.predict(X_test)
    medae_value = median_absolute_error(y_test, pred_test)
    print(f"{medae_value:.4f} medae on {name} test set")
    mse_value = mean_squared_error(y_test, pred_test)
    rmse_value = np.sqrt(mse_value)
    print(f"{rmse_value:.4f} mse on {name} test set")


# In[ ]:


lm = LinearRegression() 
lm.fit(X_train, y_train) 
pred_train = np.exp(lm.predict(X_train))
pred_test = np.exp(lm.predict(X_test))
medae_value = median_absolute_error(y_train, pred_train)
print(f"{medae_value:.4f} medae on training set")
medae_value = median_absolute_error(y_test, pred_test)
print(f"{medae_value:.4f} medae on test set")


# In[ ]:


l1 = Lasso() 
l1.fit(X_train, y_train) 
pred_train = np.exp(l1.predict(X_train))
pred_test = np.exp(l1.predict(X_test))
medae_value = median_absolute_error(y_train, pred_train)
print(f"{medae_value:.4f} medae on training set")
medae_value = median_absolute_error(y_test, pred_test)
print(f"{medae_value:.4f} medae on test set")


# In[ ]:


l2 = Ridge()
l2.fit(X_train, y_train) 
pred_train = np.exp(l2.predict(X_train))
pred_test = np.exp(l2.predict(X_test))
medae_value = median_absolute_error(y_train, pred_train)
print(f"{medae_value:.4f} medae on training set")
medae_value = median_absolute_error(y_test, pred_test)
print(f"{medae_value:.4f} medae on test set")


# In[ ]:


rf = RandomForestRegressor(criterion='mae',n_estimators = 150,max_depth = 5 )
rf.fit(X_train, y_train) 
pred_train = np.exp(rf.predict(X_train))
pred_test = np.exp(rf.predict(X_test))
medae_value = median_absolute_error(y_train, pred_train)
print(f"{medae_value:.4f} medae on training set")
medae_value = median_absolute_error(y_test, pred_test)
print(f"{medae_value:.4f} medae on test set")


# As can be seen above, our Random Forest Regressor Model performed the best with the lowest MEDAE on our test set.  Initially our Linear Regression Model with the additional feature engineering done on it performed the best.  That model's interpretation is given below. 

# Model Interpretation
# -----

# In[ ]:


lm = LinearRegression()
lm.fit(X_train, y_train)


# In[ ]:


# Interpreting coefficients
ls = []
ls1 = []
for x,y in zip(list(df_b.columns), lm.coef_):
    if abs(y) < 0.05:
        print(x,y)
        ls.append(x)
        ls1.append(y)
fig, ax = plt.subplots(figsize=(20,5))
plt.bar(ls,ls1)
plt.show()


# In[ ]:


# Interpret coefficients
ls = []
ls1 = []
for x,y in zip(list(df_b.columns), lm.coef_):
    if abs(y) >= 0.5:
        print(x,y)
        ls.append(x)
        ls1.append(y)
fig, ax = plt.subplots(figsize=(20,5))
plt.bar(ls,ls1)
plt.show()


# Checking the relationship between our strongest predictor variable:

# In[ ]:


plt.scatter(y_train, X_train["grams"])
plt.show()


# In[ ]:


plt.scatter(y_test, X_test["grams"])
plt.show()


# # Conclusion & Summary of Results

# In this project we used various machine learning models to predict the btc price of dark market cocaine.  Our dataset was composed of approximately 1,400 cleaned and standardized product listings from Dream Market's "Cocaine" category.
# 
# Initial exploratory data analysis revealed that we could split up our *shipped from-to* column into their individual components from which we then one-hot encoded into their respective categorical variables.  Further feature engineering via string extraction and manipulation was done on our *product title* column as we noticed that not all amounts in the *price*  column were scraped correctly and that there were some mistakes in encoding price schemes resulting from commas and decimals not being uniform in their use across countries.
# 
# We then trained a variety of machine learning models ranging from simple linear regression, linear regression with added feature engineering, L1, L2, and a Random Forest Regressor model.  Our Random Forest model ended up undergoing further hyperparameter tuning using Random Search, and all models underwent Cross Validation to assess how well our models would generalize.  Our models were evaluated using RMSE and MEDAE, with MEDAE being our metric of choice due to the interpretability it lent via errors in price prediction.
# 
# Although we initially found our linear regression model with added feature engineering to be our strongest performing model (and from which we interpreted our coefficients and predictive variables), we later found our Random Forest Regressor model to have the strongest predictive power when comparing MEDAE across the board. 
# 
# In interpreting our model, we split our analysis between numeric and categorical variables.  For our numeric variables, we found that the grams variable had the strongest predictive power of btc price, which is consistent with our intuition.  However, in analyzing our categorical variables, we gleaned useful insights such as the place where this cocaine was shipped from was more important than where it was shipped to, probably because of strict drug laws in the home country posing an extra risk on the vendor's side which drove up price, as can be seen in cases like China.  Another interesting feature ended up being the type and cut of the product and whether or not it was in the form of "crystal" or not.  
# 
# Overall, our model performed with a median absolute error of about $44 USD, which as of this time is the best performing model on Kaggle.  
