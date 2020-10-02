#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, BayesianRidge
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import time
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
import pickle
from catboost import CatBoostRegressor
import catboost as cb
import lightgbm
import os 


# In[ ]:


df = pd.read_csv(r"../input/kickstarter-projects/ks-projects-201801.csv")
df.head()
df.info()
describe = df.describe()
# %%
df.isna().any()


# ## Calculating Missing values by column

# In[ ]:


def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Renaming the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# In[ ]:


missing_values_table(df)


# We drop usd_pledged since usd_pledged_real shows the same data but the conversion is done from the Fixer.io API instead of kickstarter. This column has no nans. We can also drop goal and pledged column since we are using Fixer.io API conversions

# In[ ]:


df.drop(columns=["goal", "usd pledged", "pledged", "ID"], inplace=True)


# In[ ]:


df[df["name"].isna()]
# We can also drop these rows as 4 rows won't make any statistical change in our analysis.


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


plt.figure(figsize=(10, 10))


# ## Histogram of the USD Pledged

# In[ ]:


fig = plt.figure(figsize = (14,8))
plt.style.use("seaborn")
plt.hist(df["usd_pledged_real"], bins=100, edgecolor="r")
plt.xlabel("USD Pledged")
plt.ylabel("Number of Campaigns")
plt.title("Kickstarter Campaigns Pledged Amount Distribution")
plt.show()


# This shows we have a problem: Outliers<br>
# We come to the conclusion that alot of the campaigns has raised none or very little money

# ## Histogram of the USD Pledged Above 10,000$

# In[ ]:


fig = plt.figure(figsize = (14,8))
plt.style.use("seaborn")
plt.hist(df["usd_pledged_real"], bins=1000, edgecolor="r", range=(10000, 200000))
plt.xlabel("USD Pledged")
plt.ylabel("Number of Campaigns")
plt.title("Kickstarter Campaigns Pledged Amount Distribution")


# In[ ]:


plt.show()


# The first plot will show the distribution of amounts pledged for categories<br>
# with more than 10,000 campaigns.

# Creating a list with categories that has more than 10,000 campaigns.

# In[ ]:


categories = df["category"].value_counts()
categories = list(categories[categories.values > 10000].index)


# ## Plotting Amount Pledged For Categories With > 10.0000 Observations

# In[ ]:


fig = plt.figure(figsize = (14,8))
for cat_type in categories:
    # Select the category type
    subset = df[df["category"] == cat_type]

    # Density plot of amount pledged
    sns.kdeplot(subset["usd_pledged_real"], label=cat_type, shade=False)
    plt.title("Kickstarter Campaigns Pledged Amount Distribution")
plt.xlim(0, 200000)
plt.ylim(0, 0.000008)
plt.show()


# From this graph we can see that categories have some effect on the amount pledged.<br>
# Especially short category has a big number of very low amounts pledged.

# ## Campaign Status Comparison Between Top Categories

# Now lets plot these categories to check if the campaigns failed or succeeded.

# In[ ]:


# Creating subplots for each category
sns.set_palette(sns.cubehelix_palette(8))
fig = plt.figure(figsize = (15,15))
fig.add_subplot(3, 3, 1)
plt.title("Product Design")
sns.countplot(df[df["category"] == "Product Design"]["state"])
fig.add_subplot(3, 3, 2)
plt.title("Tabletop Games")
sns.countplot(df[df["category"] == "Tabletop Games"]["state"])
fig.add_subplot(3, 3, 3)
plt.title("Shorts")
sns.countplot(df[df["category"] == "Shorts"]["state"])
fig.add_subplot(3, 3, 4)
plt.title("Video Games")
sns.countplot(df[df["category"] == "Video Games"]["state"])
fig.add_subplot(3, 3, 5)
plt.title("Food")
sns.countplot(df[df["category"] == "Food"]["state"])
fig.add_subplot(3, 3, 6)
plt.title("Film & Video")
sns.countplot(df[df["category"] == "Film & Video"]["state"])
fig.add_subplot(3, 3, 7)
plt.title("Documentary")
sns.countplot(df[df["category"] == "Documentary"]["state"])
fig.subplots_adjust(wspace = 0.4, hspace= 0.8, right= 0.9, left = 0.125)

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=50)
    
plt.show()


# We can see that there is a huge difference in pledge states between different categories

# ## Correlations

# [](http://)Find all correlations and sort

# In[ ]:


correlations_data = df.corr()["usd_pledged_real"].sort_values()


# Print the correlations

# In[ ]:


print(correlations_data.head(15))


# ## Feature Engineering

# We are formatting the date columns so that we can engineer a campaign duration column

# In[ ]:


df["deadline"] = pd.to_datetime(df["deadline"], format="%Y/%m/%d")
df["launched"] = pd.to_datetime(df["launched"], format="%Y/%m/%d")
df["deadline"] = df["deadline"].dt.date
df["launched"] = df["launched"].dt.date


# Creating campaign duration column

# In[ ]:


df["campaign duration"] = (df["deadline"] - df["launched"]).dt.days
df.drop(columns=["deadline", "launched"], inplace=True)


# 
# Now lets do some feature engineering to make some columns useful<br>
# We will make the name column to be the length of titles for the campaigns

# In[ ]:


df["name"] = df["name"].apply(lambda x: len(x))


# Select the categorical columns to one hot encode them so that we can add them to the correlations

# In[ ]:


categorical_subset = df.drop(columns=["usd_pledged_real", "backers", "usd_goal_real", "name", "campaign duration"])


# ## OneHotEncode

# In[ ]:


categorical_subset = pd.get_dummies(categorical_subset)


# Now lets create our numeric subset

# In[ ]:


numeric_subset = df[["name", "backers", "campaign duration", "usd_goal_real", "usd_pledged_real"]]


# We create a features dataframe which has our encoded categorical and numerical features

# In[ ]:


features = pd.concat([numeric_subset, categorical_subset], axis=1)


# We check correlations of the features with pledged amount

# In[ ]:


correlations = features.corr()["usd_pledged_real"].dropna().sort_values(ascending=False)


# ## Most positive correlations

# In[ ]:


correlations.head(15)


# ## Most Negative Correlations

# In[ ]:


correlations.tail(15)


# We can see that only backers column is strongly correlated with the amount pledged

# ## Plotting Relationship of USD Amount Pledged vs Backers

# Use seaborn to plot a scatterplot of USD Amount pledged vs Backers

# In[ ]:


plot_features = features.copy()


# Creating the category column again since our categories are encoded

# In[ ]:


plot_features["category"] = df["category"]


# Choosing the categories with >10000 observations from the mask we created earlier

# In[ ]:


sns.set()
plot_features = plot_features[plot_features["category"].isin(categories)]
sns.lmplot("backers", "usd_pledged_real", data=plot_features, hue="category",
               scatter_kws={'s': 60}, fit_reg=False,
               height=8, aspect=1)
plt.xlim((-500, 50000))
plt.ylim(0, 1000000)
plt.title("Backers vs Pledged USD", size=12)
plt.xlabel("Backers", size=10)
plt.ylabel("Pledged USD", size=10)
plt.tick_params(labelsize=10)
plt.show()


# From our plot, we can see that the relationship between backers and pledged amount isn't<br>
# completely linear. We can also see that some categories are doing significantly better<br>
# and higher pledged to backers ratio.

# ## Analyzing Cross Relationships Between Different Variables For Top Categories

# Finally, we will analyze relationships between many variables for the top categories with a pair plot.

# In[ ]:


pairplot_data = plot_features[["usd_pledged_real", "name", "backers",
                          "campaign duration", "usd_goal_real"]]


# Creating the category column again for the pairplot data

# In[ ]:


pairplot_data["category"] = df["category"]
pairplot_data.reset_index(drop= True, inplace= True)


# Choosing the categories with >10000 observations from the mask we created earlier

# In[ ]:


pairplot_data = pairplot_data[pairplot_data["category"].isin(categories)]


# In[ ]:


sns.set(font_scale = 1.1)
grid = sns.PairGrid(data=pairplot_data, hue="category")
grid = grid.map_offdiag(plt.scatter, linewidths=1, s=40)
grid = grid.map_diag(plt.hist, color= "darkred", edgecolor= "black", bins= 10)
grid = grid.add_legend(fontsize=14)
plt.show()


# ## Feature Reduction

# We will remove features that are collinear with eachother to reduce unneccesary features in our model.<br>
# Below is a function taken from stackoverflow to do this.<br>
# https://stackoverflow.com/a/43104383****

# In[ ]:


def corr_df(x, corr_val):
    """
    Obj: Drops features that are strongly correlated to other features.
          This lowers model complexity, and aids in generalizing the model.
    Inputs:
          df: features df (x)
          corr_val: Columns are dropped relative to the corr_val input (e.g. 0.8)
    Output: df that only includes uncorrelated features
    """

    # Creates Correlation Matrix and Instantiates
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterates through Correlation Matrix Table to find correlated columns
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            row = item.index
            val = item.values
            if val >= corr_val:
                # Prints the correlated feature set and the corr val
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(i)
    drops = sorted(set(drop_cols))[::-1]

    # Drops the correlated columns
    for i in drops:
        col = x.iloc[:, (i + 1):(i + 2)].columns.values
        x.drop(col, axis=1, inplace=True)
    return x


# Moving our dependent into another variable

# In[ ]:


y_features = features["usd_pledged_real"]


# Dropping the dependant variable so that the function does not test correlation between features and dependent variable

# In[ ]:


features.drop(columns=["usd_pledged_real"], inplace=True)


# Function will remove collinear features that are above 60%

# In[ ]:


features = corr_df(features, 0.6)


# Adding back our dependent variable

# In[ ]:


features = pd.concat([features, y_features], axis=1)


# ## Splitting Data Into Train and Test Sets

# Seperating dependent and independent variables

# In[ ]:


X = features.drop(columns=["usd_pledged_real"])
y = pd.DataFrame(features["usd_pledged_real"])


# Split the data into 80% 20% training and testing sets

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# 
# Checking the training and testing data from the EDA

# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# 
# ## Scaling the features

# Create the scaler object with a range of 0-1

# In[ ]:


scaler = MinMaxScaler(feature_range=(0, 1))


# Fit on the training data

# In[ ]:


scaler.fit(X_train)


# Transform both the training and testing data

# In[ ]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


y_train = np.array(y_train).reshape((-1,))
y_test = np.array(y_test).reshape((-1,))


# Function to calculate mean absolute error of the training algorithm

# In[ ]:


def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))


# 
# Function to train a model, and evaluates the model on the test set

# In[ ]:


def fit_and_evaluate(model):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)

    # Return the performance metric
    return model_mae


# ## Linear Regression

# In[ ]:


lr = LinearRegression()
lr_mae = fit_and_evaluate(lr)
print("Linear regression mean of error is {}.".format(lr_mae))


# ## XGBRegressor

# In[ ]:


gradient_boosted = XGBRegressor(random_state=1, silent = True)
gradient_boosted_mae = fit_and_evaluate(gradient_boosted)
print("Gradient boosted regression mean of error is {}.".format(gradient_boosted_mae))


# ## KNNRegressor

# In[ ]:


knn = KNeighborsRegressor(n_neighbors=10)
knn_mae = fit_and_evaluate(knn)
print("KNN regression mean of error is {}.".format(knn_mae))


# ## Gradient Boosting Regressor

# In[ ]:


sklearn_gradient_boosted = GradientBoostingRegressor()
sklearn_gradient_boosted_mae = fit_and_evaluate(sklearn_gradient_boosted)
print("sklearn Gradient boosted regression mean of error is {}.".format(sklearn_gradient_boosted_mae))


# ## SGDRegressor

# In[ ]:


sgd = SGDRegressor(random_state=1)
sgd_mae = fit_and_evaluate(sgd)
print("Stochastic gradiant descent regression mean of error is {}.".format(sgd_mae))


# ## Bayesian Ridge

# In[ ]:


br = BayesianRidge()
br_mae = fit_and_evaluate(br)
print("Bayesian ridge regression mean of error is {}.".format(br_mae))


# ## LightGBMRegressor

# In[ ]:


lgb = LGBMRegressor(objective ="regression", num_leaves = 35, random_state= 1)
lgb_mae = fit_and_evaluate(lgb)
print("LightGBM regression mean of error is {}.".format(lgb_mae))


# ## CatBoostRegressor

# In[ ]:



catboost = CatBoostRegressor(silent=True, n_estimators=300)
catboost_mae = fit_and_evaluate(catboost)
print("CatBoost regression mean of error is {}.".format(catboost_mae))


# In[ ]:


models_df = pd.DataFrame(
    {"model": ["Linear Regression","CatBoost Regression", "XGBoost Gradient Boosted Regression",
               "K-Nearest Neighbours", "SGD Regression", "Bayesian Ridge Regression",
               "LightGBM Regression"], "mae": [lr_mae,catboost_mae, gradient_boosted_mae,
                                            knn_mae, sgd_mae, br_mae, lgb_mae]})


# ## Plotting Model Performances

# In[ ]:


plt.figure(figsize=(13, 13))
sns.catplot(x="mae", y="model", kind="bar", data=models_df, height=5, aspect=3)
plt.show()


# It looks like top 3 performers are in order XGBoost, sklearnGBM, LightGBM

# ## Flowchart for Modeling

# <img src="https://go.gliffy.com/go/share/image/sue90o585kisdtlbmkvv.png?utm_medium=live-embed&utm_source=wordpress" width="1000px" align="middle">

# ## Feature selection for XGBoost, LightGBM and CatBoost Regression

# In[ ]:


# List of algorithms we will use

models = [XGBRegressor(silent=True), LGBMRegressor(), CatBoostRegressor(silent=True, n_estimators=300)]
sets = []


# Function to select features for each model.
def FeatureSelector(models,X_train, y_train,X_Test, y_test):


    
    for model in models:
        model_iter = model.fit(X_train, y_train)
        selection = SelectFromModel(model_iter, threshold=0.01, prefit=True)
        train = selection.transform(X_train)
        
        model_iter = model.fit(X_test, y_test)
        selection = SelectFromModel(model_iter, threshold=0.01, prefit=True)
        test = selection.transform(X_test)
        
        #Appending each selected feature to sets list for every model.
        sets.append(train)
        sets.append(test)
        

        
FeatureSelector(models, X_train, y_train, X_test, y_test)


# In[ ]:


# Assigning the models to variables in the sets list.
XGB_y_train = y_train.copy()
XGB_X_train = sets[0]
XGB_X_test = sets[1]

# Creating a validation set from our X_train and y_train sets. 
# We will use the validation set to train our level 1 model.
XGB_X_train, XGB_X_valid, XGB_y_train, XGB_y_valid = train_test_split(XGB_X_train, XGB_y_train, test_size = 0.2, random_state=42)

# Assigning the models to variables in the sets list.
LGBM_y_train = y_train.copy()
LGBM_X_train = sets[2]
LGBM_X_test = sets[3]

# Creating a validation set from our X_train and y_train sets. 
# We will use the validation set to train our level 1 model.
LGBM_X_train, LGBM_X_valid, LGBM_y_train, LGBM_y_valid = train_test_split(LGBM_X_train, LGBM_y_train, test_size = 0.2, random_state=42)

# Assigning the models to variables in the sets list.
catboost_y_train = y_train.copy()
catboost_X_train = sets[4]
catboost_X_test = sets[5]

# Creating a validation set from our X_train and y_train sets. 
# We will use the validation set to train our level 1 model.
catboost_X_train, catboost_X_valid, catboost_y_train, catboost_y_valid = train_test_split(catboost_X_train, catboost_y_train, test_size = 0.2, random_state=42)


# ## Tuning parameters for XGB algorithm through gridsearch.
# 

# In[ ]:


XGB_model = XGBRegressor(random_state=1, silent= True)
max_depth_xgb = [6, 7, 8, 9, 10]
n_estimators_xgb = [100]
min_child_weight_xgb = [1, 2, 4, 6]
gamma_xgb = [0, 0.1, 0.2, 0.3, 0.4]


hyperparameter_grid =  {"estimator__max_depth": max_depth_xgb, "estimator__n_estimators": n_estimators_xgb,
                        "estimator__min_child_weight": min_child_weight_xgb, "estimator__gamma": gamma_xgb}

gridsearch_cv = GridSearchCV(XGB_model, hyperparameter_grid, cv= 4, scoring="neg_mean_absolute_error",
                             n_jobs=-1, verbose=1, return_train_score=True)
gridsearch_cv.fit(XGB_X_train, XGB_y_train)
gridsearch_results = pd.DataFrame(gridsearch_cv.cv_results_).sort_values("mean_test_score", ascending=False)
gridsearch_results.head(10)


# Evaluate Model

# In[ ]:


print("Best score of the grid search: {}".format(gridsearch_cv.best_score_))
print("Best score parameters: {}".format(gridsearch_cv.best_params_))


# In[ ]:


# Tuning parameters for XGB algorithm through gridsearch.


XGB_model = XGBRegressor(random_state=1)
max_depth_xgb = [6]
n_estimators_xgb = [100]
min_child_weight_xgb = [1]
gamma_xgb = [0]

subsample_xgb = [0.6, 0.7, 0.8, 0.9, 1]
colsample_bytree_xgb = [0.6, 0.7, 0.8, 0.9, 1]


model = XGBRegressor(random_state=1, silent= True)

hyperparameter_grid =  {"estimator__max_depth": max_depth_xgb, "estimator__n_estimators": n_estimators_xgb,
                        "estimator__min_child_weight": min_child_weight_xgb, "estimator__gamma": gamma_xgb,
                        "estimator__subsample": subsample_xgb, "estimator__colsample_bytree": colsample_bytree_xgb}


gridsearch_cv = GridSearchCV(XGB_model, hyperparameter_grid, cv= 4, scoring="neg_mean_absolute_error",
                             n_jobs=-1, verbose=1, return_train_score=True)
gridsearch_cv.fit(XGB_X_train, XGB_y_train)
gridsearch_results = pd.DataFrame(gridsearch_cv.cv_results_).sort_values("mean_test_score", ascending=False)
gridsearch_results.head(10)


# In[ ]:


print("Best score of the grid search: {}".format(gridsearch_cv.best_score_))
print("Best score parameters: {}".format(gridsearch_cv.best_params_))


# In[ ]:



gamma_xgb = [0]
min_child_weight_xgb = [1]
subsample_xgb = [0.6]
colsample_bytree_xgb = [0.6]
max_depth_xgb = [6]
n_estimators_xgb = [100]

# Assigning selected parameters to our XGB algorithm.
XGB_model = XGBRegressor(max_depth= max_depth_xgb[0], n_estimators= n_estimators_xgb[0],gamma = gamma_xgb[0],
                    min_child_weight= min_child_weight_xgb[0], subsample = subsample_xgb[0], 
                     colsample_by_tree= colsample_bytree_xgb[0], silent= True)


# In[ ]:


#Function to fit and cross validate the algorithm with 5 folds.

def XGBmodelfit(alg, X_train, y_train, useTrainCV=True, cv_folds= 5, early_stopping_rounds=500):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        
        # Creating XGB matrix for the cross validation method.
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        global cvresult
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='mae', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric='mae')
    global XGB_pred
    # Predict training set:
    XGB_pred = alg.predict(X_train)


# In[ ]:


XGBmodelfit(XGB_model, XGB_X_train, XGB_y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=500)

# Print model report:
print("\nModel Report")
print("MAE (Train): {}".format(mean_absolute_error(XGB_y_train, XGB_pred)))


# ## Tuning parameters for LightGBM Algorithm Through Gridsearch.

# In[ ]:


LGBM_model = LGBMRegressor(metric = "l1", verbose =0)
num_leaves_LGBM = [ 75, 100, 125, 150]
learning_rate_LGBM = [0.01,0.02, 0.03, 0.04, 0.05]
n_estimators_LGBM = [100,200,300]
hyperparameter_grid = {"num_leaves": num_leaves_LGBM,
                       "learning_rate": learning_rate_LGBM, "n_estimators":n_estimators_LGBM}

gridsearch_cv = GridSearchCV(LGBM_model, hyperparameter_grid, cv= 4, scoring="neg_mean_absolute_error",
                             n_jobs=-1, verbose=1, return_train_score=True)

gridsearch_cv.fit(LGBM_X_train, LGBM_y_train)

gridsearch_results = pd.DataFrame(gridsearch_cv.cv_results_).sort_values("mean_test_score", ascending=False)
gridsearch_results.head(10)


# In[ ]:


print("Best score of the grid search: {}".format(gridsearch_cv.best_score_))
print("Reinitializing grid search with the following parameters: {}".format(gridsearch_cv.best_params_))


# In[ ]:


num_leaves_LGBM = [150]
learning_rate_LGBM = [0.05]
n_estimators_LGBM= [100]
min_data_leaves_LGBM = [1]


# In[ ]:


# Assigning the paramaters to our LightGBM algorithm.

LGBM_model = LGBMRegressor(metric = "l1", num_leaves= num_leaves_LGBM[0],
                           learning_rate= learning_rate_LGBM[0], n_estimators= n_estimators_LGBM[0],
                           min_data_leaves= min_data_leaves_LGBM[0], verbose =0)


# In[ ]:


#Function to fit and cross validate the algorithm with 5 folds.


def modelfitLGBM(alg, X_train, y_train, useTrainCV=True, nfolds= 5, early_stopping_rounds=500):
    if useTrainCV:
        LGBM_param = alg.get_params()
        
        # Creating LightGBM matrix to use in our CV method.
        lgtrain = lightgbm.Dataset(X_train, label=y_train)
        cvresult = lightgbm.cv(LGBM_param, lgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=nfolds,
                          metrics='mae', early_stopping_rounds=early_stopping_rounds, stratified=False)
        
    # Fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric='mae')

    # Predict training set:
    global lgbm_pred
    lgbm_pred = alg.predict(X_train)


# In[ ]:


modelfitLGBM(LGBM_model, LGBM_X_train, LGBM_y_train, useTrainCV=True, nfolds= 5, early_stopping_rounds=500 )

# Print model report:
print("\nModel Report")
print("MAE (Train): {}".format(mean_absolute_error(LGBM_y_train, lgbm_pred)))


# ## Adding in CatBoost Algorithm
# 

# In[ ]:


# Selecting second best parameters as the mean fit_time is drastically higher in the best parameters.
# The MAE isn't very different between first and second best score, which means our decision is justified.

max_depth_catboost = [10]
learning_rate_catboost = [0.05]
n_estimators_catboost =  [500]
catboost_model = CatBoostRegressor(silent=True, max_depth = max_depth_catboost[0],
                                   learning_rate= learning_rate_catboost[0], 
                                   n_estimators= n_estimators_catboost[0], 
                                   random_state = 42, loss_function = "MAE")


# In[ ]:


#Function to fit and cross validate the algorithm with 5 folds.


def modelfit_catboost(alg, X_train, y_train, useTrainCV=True, nfolds= 5, early_stopping_rounds=500):
    if useTrainCV:
        cat_param = alg.get_params()
        
        # Creating CatBoost matrix to use in our CV method.

        cattrain = cb.Pool(X_train, label=y_train)
        cvresult = cb.cv(cattrain, cat_param, 
                               num_boost_round=alg.get_params()['n_estimators'], 
                       fold_count=5, early_stopping_rounds=500, stratified=False,
                      logging_level = "Verbose")
   
    # Fit the algorithm on the data
    alg.fit(X_train, y_train)

    # Predict training set:
    global catboost_pred
    catboost_pred = alg.predict(X_train)


# In[ ]:


modelfit_catboost(catboost_model, catboost_X_train, catboost_y_train, useTrainCV=True, 
                  nfolds= 5, early_stopping_rounds=500)

# Print model report:
print("\nModel Report")
print("MAE (Train):{}".format(mean_absolute_error(catboost_y_train, catboost_pred)))
    


# ## Predicting Values for the Validation Dataset

# In[ ]:


# Reshaping our predictions so that we can merge it with the validation dataset and use it to train 
# our algorithm again.

level_0_catboost_pred = catboost_model.predict(catboost_X_valid)
level_0_catboost_pred = level_0_catboost_pred.reshape(60585,1)


level_0_XGB_pred = XGB_model.predict(XGB_X_valid)
level_0_XGB_pred = level_0_XGB_pred.reshape(60585,1)


level_0_LGBM_pred = LGBM_model.predict(LGBM_X_valid)
level_0_LGBM_pred = level_0_LGBM_pred.reshape(60585,1)


# In[ ]:


# Converting our Validation features dataset into a dataframe.
XGB_X_valid = pd.DataFrame(XGB_X_valid)

# Converting our predictions into dataframes.
level_0_XGB_pred = pd.DataFrame(level_0_XGB_pred)
level_0_LGBM_pred = pd.DataFrame(level_0_LGBM_pred)
level_0_catboost_pred = pd.DataFrame(level_0_catboost_pred)


# In[ ]:



# Merging the prediction dataframes to the Validation features dataframe as new columns.

XGB_X_valid["XGB_pred"] = level_0_XGB_pred
XGB_X_valid["LGBM_pred"] = level_0_LGBM_pred 
XGB_X_valid["catboost_pred"] = level_0_catboost_pred


# In[ ]:


# Converting the validation dataframe back to np.array so that we can use it to train another algorithm.

XGB_X_valid = XGB_X_valid.values


# In[ ]:


# Reassigning our level 1 algorithm

XGB_level_1_model = XGBRegressor(max_depth= max_depth_xgb[0], n_estimators= n_estimators_xgb[0],gamma = gamma_xgb[0],
                    min_child_weight= min_child_weight_xgb[0], subsample = subsample_xgb[0], 
                     colsample_by_tree= colsample_bytree_xgb[0], silent= True)


# ## Fitting the Level 1 Model

# In[ ]:


# Fitting new XGB model and doing 5 folds cross validation with the merged DataFrame

XGBmodelfit(XGB_level_1_model, XGB_X_valid, XGB_y_valid, useTrainCV=True, cv_folds=5, early_stopping_rounds=500)
print("\nModel Report")
print("MAE (Train): {}".format(mean_absolute_error(XGB_y_valid, XGB_pred)))


# > ## Testing Final Level 1 Model with Test Sample

# In[ ]:


# Testing our fitted model on the original test set we have split at the start.

XGBmodelfit(XGB_model, XGB_X_test, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=500)
print("\nModel Report")
print("MAE (Train): {}".format(mean_absolute_error(y_test, XGB_pred)))


# ## Calculating Residuals

# In[ ]:


residuals = XGB_pred - y_test

plt.hist(residuals, color="darkred", bins=50, range=(-10000, 10000))
plt.xlabel("MAE")
plt.ylabel("Count")
plt.title("Distribution of Residuals")
plt.show()

