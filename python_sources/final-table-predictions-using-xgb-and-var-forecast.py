#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# Created on Tue Jun  2 11:06:39 2020<br>
# author: jaketuricchi
# Generated in Spyder 4.0 using J2P for conversion from .py to .ipynb

# # Predicting the final EPL table: <br>
# Using a combination of machine learning model generated on game-level data and VAR forecasting to forecast features based on prior features to predict the final league table.
# 

# Aim 1: We want to predict match reuslt given match statistics (removing obvious stats such as goals); Supervised classification

# Aim 2: Forecast match features for the final 6 games of the season using VAR

# Aim 3: Predict the last 6 results for each team using the previously generated model.

# Aim 4: Predict final league table (ML/forecasting?)

# All meanings of variables can be found at: https://www.kaggle.com/idoyo92/epl-stats-20192020 along with the complete data set.

# 
# Import packages

# In[ ]:


import math
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings 
import seaborn as sns
import sklearn
from datetime import datetime
import calendar
get_ipython().run_line_magic('matplotlib', 'inline')
import shelve
import warnings
warnings.simplefilter('ignore')


#  Set wd and read in data

# In[ ]:





# In[ ]:


games = pd.read_csv('../input/epl-stats-20192020/epl2020.csv')


# # Sort games df 

# In[ ]:


print(games.columns)
print(games.isna().sum()) #no missing data


# In[ ]:


games = games.drop('Unnamed: 0', axis=1) # drop useless column
games = games.drop(['scored', 'missed', 'wins', 'draws', 'loses', 'pts',
                    'tot_points', 'tot_goal', 'tot_con'], axis=1) # drop columns which give away result


# Code target

# In[ ]:


games['target']=np.where(games['result']=='l', 0,
                         np.where(games['result']=='d', 1,2))


# Drop result string variable

# In[ ]:


games = games.drop('result', axis=1) # drop useless column


# Set categorical variables

# In[ ]:


games[['target', 'h_a', 'teamId', 'Referee.x', 'matchDay']] = games[['target', 'h_a', 'teamId', 'Referee.x', 'matchDay']].astype('category')


# # EDA

# Here we explore 'expected' variables (goals, concedes) given by the bookmakers, in relation to the target. We could do a more extensive visualisation with more time.

# In[ ]:


EDA_pairplot1=games.filter(items=['xG', 'xGA', 'npxG', 'npxGA', 'npxGD', 'h_a', 'target'])
sns.pairplot(EDA_pairplot1, hue='target')


# Here we explore HOME team match stats, in relation to the target

# In[ ]:


EDA_pairplot2=games.filter(items=['ppda_cal', 'allowed_ppda', 'HS.x', 'HST.x', 'HF.x', 'HC.x', 'target'])
sns.pairplot(EDA_pairplot2, hue='target')


# Here we explore AWAY team match stats, in relation to the target

# In[ ]:


EDA_pairplot3=games.filter(items=['ppda_cal', 'allowed_ppda', 'AS.x', 'AST.x', 'AF.x', 'AC.x', 'target'])
sns.pairplot(EDA_pairplot3, hue='target')


# Game outcomes

# In[ ]:


EDA_plot4=games.groupby('teamId')['target'].agg(counts='value_counts').reset_index()
EDA_plot4['target']=np.where(EDA_plot4['target']==0, 'loss',
                         np.where(EDA_plot4['target']==1, 'draw','win'))
EDA_plot4=EDA_plot4.reset_index() 
sns.catplot(y="teamId",x='counts', hue='target',data=EDA_plot4)


# # Modelling - Preprocessing

# Scale numeric data

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


nums= games.select_dtypes(include=['float', 'int64'])
other= games.select_dtypes(exclude=['float', 'int64']).drop(['date', 'target'], axis=1)


# In[ ]:


scaler = StandardScaler()
nums_scaled = scaler.fit_transform(nums)
games_scaled=pd.DataFrame(nums_scaled, columns=nums.columns)


# One hot encode categorical data

# In[ ]:


dummies=pd.get_dummies(other)


# Generate an analysis dataset

# In[ ]:


gameday_pred = pd.concat([games_scaled,dummies, games['target']], axis=1)  


# Split the data into features and labels

# In[ ]:


X=gameday_pred.drop('target', axis=1)
y=gameday_pred['target']


# In[ ]:


print(y.dtypes)
print(X.dtypes)


# Split the data intro train and test sets for model training

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# # Algorithm testing and initial selection

# Load packages

# In[ ]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,  QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, precision_score, recall_score, f1_score


# Select classification algorithms we'll test

# In[ ]:


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]


# Logging for Visual Comparison

# In[ ]:


log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)


# Run algo loop

# In[ ]:


for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    
    # calculate score
    precision = precision_score(y_test, train_predictions, average = 'macro') 
    recall = recall_score(y_test, train_predictions, average = 'macro') 
    f_score = f1_score(y_test, train_predictions, average = 'macro')
    
    
    print("Precision: {:.4%}".format(precision))
    print("Recall: {:.4%}".format(recall))
    print("F-score: {:.4%}".format(recall))
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)


# collinearity should be expected given the type of data
# 
# # Plot results of algo testing

# In[ ]:


sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")


# In[ ]:


sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")


# What are the most important features?

# In[ ]:


rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train, y_train);
feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
feat_importances.nlargest(10).plot(kind='barh')


# These results suggest that the 'expected' (i.e. betting agent prediction) variables provide more predictive value than the actual game statistics which is surprising.

# # Model selection and tuning<br>
# The initial results suggest that XGb and RF are the two best performing classifiers for this problem. Lets begin by tuning the best algorithm (XGb)

# In[ ]:


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import xgboost as xgb


# In[ ]:


xgb = XGBClassifier(objective='multi:softprob',silent=False)


# Generate a basic Xgb model and predict the test data

# In[ ]:


xgb.fit(X_train,y_train)
y_pred_xgb_basic=xgb.predict(X_test)


# Calculate score

# In[ ]:


precision = precision_score(y_test, y_pred_xgb_basic, average = 'macro') * 100
recall = recall_score(y_test, y_pred_xgb_basic, average = 'macro') * 100
f_score = f1_score(y_test, y_pred_xgb_basic, average = 'macro') * 100
a_score = accuracy_score(y_test, y_pred_xgb_basic) * 100


# In[ ]:


print('Precision: %.3f' % precision)
print('Recall: %.3f' % recall)
print('F-Measure: %.3f' % f_score)
print('Accuracy: %.3f' % a_score)


# Tuning the XGb using Grid search

# In[ ]:


parameters_xgb = {
        'learning_rate': [0.05, 0.1, 0.2, 0.3, 0.5], 
        'n_estimators': [200, 300, 400, 500, 600], 
        'max_depth': [1, 5, 10, 15, 20], 
        'gamma' :[0.1, 0.5, 1, 2, 5], 
        'subsample': [0.5, 0.75, 1], 
        'colsample_bytree': [0.01, 0.1, 0.5, 1], 
        }


# Instantiate the grid search model

# In[ ]:


grid_search_xgb = GridSearchCV(estimator = xgb, param_grid = parameters_xgb, 
                          cv = 3,n_jobs=-1, verbose = 2)


# Fit the grid search to the data

# In[ ]:


grid_search_xgb.fit(X_train,y_train)
print(grid_search_xgb.best_params_)
    # {'colsample_bytree': 0.1, 'gamma': 0.5, 'learning_rate': 0.3, 'max_depth': 15, 'n_estimators': 300, 'subsample': 1}


# Run best params

# In[ ]:


best_grid_xgb = grid_search_xgb.best_estimator_
best_grid_xgb.fit(X_train,y_train)


# Predict test data

# In[ ]:


y_pred_xgb = best_grid_xgb.predict(X_test)


# Calculate score

# In[ ]:


precision = precision_score(y_test, y_pred_xgb, average = 'macro') * 100
recall = recall_score(y_test, y_pred_xgb, average = 'macro') * 100
f_score = f1_score(y_test, y_pred_xgb, average = 'macro') * 100
a_score = accuracy_score(y_test, y_pred_xgb) * 100


# In[ ]:


print('Precision: %.3f' % precision)
print('Recall: %.3f' % recall)
print('F-Measure: %.3f' % f_score)
print('Accuracy: %.3f' % a_score)


# Hyperparameter tuning gives us a very small additional boost in XGB performance. We could improve the grid search but are limited by time and computation power currently.

# # Tune random forest

# Generate a basic model

# In[ ]:


rf = RandomForestClassifier(random_state = 1)
rf_model_basic = rf.fit(X_train, y_train)
y_pred_rf_basic = rf_model_basic.predict(X_test)


# Calculate score

# In[ ]:


precision = precision_score(y_test, y_pred_rf_basic, average = 'macro') * 100
recall = recall_score(y_test, y_pred_rf_basic, average = 'macro') * 100
f_score = f1_score(y_test, y_pred_rf_basic, average = 'macro') * 100
a_score = accuracy_score(y_test, y_pred_rf_basic) * 100


# In[ ]:


print('Precision: %.3f' % precision)
print('Recall: %.3f' % recall)
print('F-Measure: %.3f' % f_score)
print('Accuracy: %.3f' % a_score)


# Tuning the RF using Grid search

# In[ ]:


parameters_rf = {
    'bootstrap': [True],
    'n_estimators' : [100, 300, 500, 800, 1200],
    'max_depth' : [5, 8, 15, 25, 30],
    'min_samples_split' : [2, 5, 10, 15, 100],
    'min_samples_leaf' : [1, 2, 5, 10],
    'max_features': [2, 4]
}


# Instantiate the grid search model

# In[ ]:


grid_search_rf = GridSearchCV(estimator = rf, param_grid = parameters_rf, 
                          cv = 3,n_jobs=-1, verbose = 2)


# Fit the grid search to the data

# In[ ]:


grid_search_rf.fit(X_train,y_train)
print(grid_search_rf.best_params_)
    # {'bootstrap': True, 'max_depth': 15, 'max_features': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 800}


# Run best params

# In[ ]:


best_grid_rf = grid_search_rf.best_estimator_
best_grid_rf.fit(X_train,y_train)


# Predict test data

# In[ ]:


y_pred_rf = best_grid_rf.predict(X_test)


# Calculate score

# In[ ]:


precision = precision_score(y_test, y_pred_rf, average = 'macro') * 100
recall = recall_score(y_test, y_pred_rf, average = 'macro') * 100
f_score = f1_score(y_test, y_pred_rf, average = 'macro') * 100
a_score = accuracy_score(y_test, y_pred_rf) * 100


# In[ ]:


print('Precision: %.3f' % precision)
print('Recall: %.3f' % recall)
print('F-Measure: %.3f' % f_score)
print('Accuracy: %.3f' % a_score)


# Small increases in performance observed from tuning the RF, but a tuned XGb is still the best performing algorithm

# # PART 2: how will the league finish?

# Initial stage: calculate the current table. We need to reload the table to get points back.

# In[ ]:


games2 = pd.read_csv('../input/epl-stats-20192020/epl2020.csv')
table=games2.groupby('teamId')['pts'].agg(table_points='sum').reset_index().sort_values('table_points', ascending=False)
table['position']=range(0,len(table), 1)
table['position']=table['position']+1
print(table)


# Next, we'll use multivariate time series forecasting to forecast the match statistics used as features. Then, we'll use the XGb model we generated to predict the results of these matches<br>
# Last, we'll add the results to generate a final league table

# In[ ]:


fc_games=games
fc_games['date'] = pd.to_datetime(fc_games.date , format = '%Y-%m-%d %H:%M:%S')
fc_games.index = fc_games['date']
fc_games = fc_games.drop(['date'], axis=1)


# To forecast using VAR, categorical variables must be label encoded

# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# Seperate categories and numerics for preprocessing, and then rejoin.

# In[ ]:


categories= fc_games.select_dtypes(include=['category']).drop(['teamId','target'], axis=1).apply(LabelEncoder().fit_transform)
fc_games=fc_games.drop(categories.columns.values, axis=1)
fc_games=pd.concat([fc_games, categories], axis=1)


# We'll need to check stationarity for each participant. Perform ADFuller to test for Stationarity of given series and print report

# In[ ]:


from statsmodels.tsa.stattools import adfuller


# Function adapted from: https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html

# In[ ]:


def adfuller_test(series, signif=0.05, name='', verbose=False):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')
    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')
    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")   
    


# In[ ]:


x=fc_games[fc_games['teamId']=='Liverpool'] #lets take an example and see if we can test/produce stationarity
x=x.drop(['teamId','target'], axis=1) #dont want to forecast either of these.
fc_train = x[:int(0.8*(len(x)))] #split data
fc_valid = x[int(0.8*(len(x))):]


# In[ ]:


for name, column in fc_train.iteritems(): #run the ADF test of stationarity
    adfuller_test(column, name=column.name)
    print('\n')


# Most columns are non-stationary, lets difference and retry

# In[ ]:


fc_train_diff = fc_train.diff().dropna()


# NOTE: having tried differencing multiple times - this will not significantly change the stationarity of the series - this gives us less forecasting power but also impossible to avoid, probably due to<br>
# the small number of obs.

# # Validation of forecasting

# It is clear that forecasting all columns is given the number of observations we have is not possible. As such I decided to select the 1-5 and 6-10 most important variables<br>
# and forecast these.   

# In[ ]:


from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
 
feat_importances_1=feat_importances.nlargest(5).index.values
feat_importances_2=feat_importances.nlargest(10).index.values[5:]
feat_importances_3=feat_importances.nlargest(15).index.values[10:]


# In[ ]:


def validation_by_team(x):
    x1=x.filter(items=feat_importances_1)
    x2=x.filter(items=feat_importances_2)
    
    fc_train1 = x1[:int(0.8*(len(x1)))] #split data
    fc_valid1 = x1[int(0.8*(len(x1))):]
    
    fc_train2 = x2[:int(0.8*(len(x2)))] #split data
    fc_valid2 = x2[int(0.8*(len(x2))):]
    
    model1 = VAR(endog=fc_train1) #fit VAR model
    model_fit1 = model1.fit()
    
    model2 = VAR(endog=fc_train2) #fit VAR model
    model_fit2 = model2.fit()
    
    prediction1 = model_fit1.forecast(model_fit1.y, steps=len(fc_valid1)) #predict
    prediction1 = pd.DataFrame(data=prediction1, columns=x1.columns)
    
    prediction2 = model_fit2.forecast(model_fit2.y, steps=len(fc_valid2)) #predict
    prediction2 = pd.DataFrame(data=prediction2, columns=x2.columns)
       
    # Check the performance of the by serial correlation of errors using the Durbin Watson statistic.
    
    # The value of this statistic can vary between 0 and 4. The closer it is to the value 2, 
    # then there is no significant serial correlation. The closer to 0, there is a positive 
    # serial correlation, and the closer it is to 4 implies negative serial correlation.
    
    out1 = durbin_watson(model_fit1.resid)
    print(out1)
    out2 = durbin_watson(model_fit2.resid)
    print(out2)
    
    prediction_performance = pd.DataFrame([out1, out2]).T
    return(prediction_performance)


# In[ ]:


prediction_validations=fc_games.groupby('teamId').apply(validation_by_team)


# These values seem suitable. They are generally around 2. The distance between days doesnt matter so we can ignore the errors here.

# # Forecasting of match statistics

# Since we cannot forecast every one of the features due to limited obs, we'll forecast the top 10, and use mean imputation for the rest.

# There are 6 more games in the EPL season, so we'll predict 6 lags ahead.  It is important to note that with the amount of variables and previous observations we have, the predictions will be similar to an approximate mean imputation

# In[ ]:


import random


# In[ ]:


def forecast_by_team(x):
    target=x['target'].reset_index(drop=True)
    x=x.drop(['teamId','target'], axis=1) #dont want to forecast either of these.
    
    x1=x.filter(items=feat_importances_1)
    x2=x.filter(items=feat_importances_2)
    x3=x.filter(items=feat_importances_3)
    
    model1 = VAR(endog=x1) #fit VAR model1
    model_fit1 = model1.fit()
    
    model2 = VAR(endog=x2) #fit VAR model2
    model_fit2 = model2.fit()
    
    model3 = VAR(endog=x3) #fit VAR model3
    model_fit3 = model3.fit()
    
    prediction1 = model_fit1.forecast(model_fit1.y, steps=6) #predict
    prediction1 = pd.DataFrame(data=prediction1, columns=x1.columns)
    
    prediction2 = model_fit2.forecast(model_fit2.y, steps=6) #predict
    prediction2 = pd.DataFrame(data=prediction2, columns=x2.columns)
    
    prediction3 = model_fit3.forecast(model_fit3.y, steps=6) #predict
    prediction3 = pd.DataFrame(data=prediction3, columns=x3.columns)
    
    predictions=pd.concat([prediction1, prediction2, prediction3], axis=1)
    
    x_forecasted=pd.concat([x, predictions], axis=0).reset_index()
    
    # Lets randomly impute home/away games as 0 or 1
    # I start by generating random 0s and 1s and selecting the index where data is missing
    # Then we fill.
    na_loc =x_forecasted.index[x_forecasted['h_a'].isnull()]
    num_nas = len(na_loc)
    fill_values = pd.DataFrame({'h_a': [random.randint(0,1) for i in range(num_nas)]}, index = na_loc)
    
    x=pd.concat([x, fill_values], axis=0).reset_index(drop=True)
    predictions.index=fill_values.index
    x_forecasted2=x.combine_first(predictions)
    
    # Now lets mean (numeric) or mode (categorical) impute the other missing variables
    # Since these are of less modelling importance, a mean impute shouldn't make much difference/
    # Also, it is logical to assume consistency in Season performance, without any new data
    # to think otherwise.
    
    nums= x_forecasted2.select_dtypes(include=['float', 'int64']).apply(lambda x: x.fillna(x.mean()),axis=0).round()
    x_forecasted3=x_forecasted2.combine_first(nums) #join this imputation back in to fill missingness
    x_forecasted4=pd.concat([x_forecasted3, target], axis=1)
    
    return(x_forecasted4)
    
forecasted_data=fc_games.groupby('teamId').apply(forecast_by_team).reset_index().drop('level_1', axis=1) # run seperately for all teams
print(forecasted_data.isna().sum()) #no missing data but the target


# # Prediction of future games<br>
# First we must ensure that the new forecasted df is in the same format as the previous matchday prediction df<br>
# That includes changing categories back to categories and one-hot encoding as well as scaling.

# In[ ]:


forecasted_data[[
    'target', 'h_a', 'teamId', 'Referee.x', 'matchDay']] = forecasted_data[[
        'target', 'h_a', 'teamId', 'Referee.x', 'matchDay']].astype('category')
        
# Scale numeric data
from sklearn.preprocessing import StandardScaler


# In[ ]:


nums= forecasted_data.select_dtypes(include=['float', 'int64'])
other= forecasted_data.select_dtypes(exclude=['float', 'int64']).drop('target', axis=1)


# In[ ]:


scaler = StandardScaler()
nums_scaled = scaler.fit_transform(nums)
fc_scaled=pd.DataFrame(nums_scaled, columns=nums.columns)


# One hot encode categorical data

# In[ ]:


dummies=pd.get_dummies(other)


# Generate an analysis dataset

# In[ ]:


fc_pred = pd.concat([fc_scaled,dummies, forecasted_data['target']], axis=1)  


# Prediction data - since there is slight differences in the feature names, we'll retrain the best (tuned XGB) model on the new data<br>
# This shouldn't make much difference to results but just improve compatability.

# In[ ]:


train_data = fc_pred[fc_pred['target'].notnull()]
X_train=train_data.drop('target', axis=1)
y_train=train_data['target']


# In[ ]:


X_test=fc_pred[fc_pred['target'].isnull()].drop('target', axis=1).reset_index(drop=True)


# Fit to training data and predict. For the XGb parameters we'll insert the tuned hyperparameters

# In[ ]:


xgb = XGBClassifier(objective='multi:softprob',
                    colsample_bytree = 0.5, gamma = 2, learning_rate= 0.2, max_depth= 10, 
                    n_estimators= 500, subsample = 1)
xgb.fit(X_train,y_train)
final_predictions=pd.DataFrame({'target':xgb.predict(X_test)})


# Put the final df together

# In[ ]:


final_test=pd.concat([X_test, final_predictions], axis=1)
complete_season=pd.concat([train_data, final_test], axis=0).reset_index(drop=True)


# Backwards transform OneHotEncoded (dummy) variables (specifically, we need team)

# In[ ]:


team_cols=complete_season[complete_season.columns[complete_season.columns.to_series().str.contains('teamId')]]#[col==1].stack().reset_index().drop(0,1)
team_cols_long=team_cols[team_cols==1].stack().reset_index().drop(0,1).drop('level_0', axis=1)
team_cols_long['team']=team_cols_long['level_1'].str.rpartition('_')[2]
team_cols_long=team_cols_long.drop('level_1', axis=1)


# In[ ]:


complete_season=pd.concat([complete_season, team_cols_long], axis=1)


# # Rebuilding the final table

# In[ ]:


new_points=complete_season.filter(items=['team', 'target'])
new_points['pts_new']=np.where(new_points['target']==0, 0,
                           np.where(new_points['target']==1,1, 3))
table_final=new_points.groupby('team')['pts_new'].agg(table_points_new='sum').reset_index().sort_values('table_points_new', ascending=False)
table_final['position']=range(0,len(table_final), 1)
table_final['position']=table_final['position']+1


# In[ ]:


print(table_final)


# Table changes - did anyone overtake anyone in the last 6 games?

# In[ ]:


table_final.sort_values('team', inplace=True)
table.sort_values('teamId', inplace=True)
table_final['positition_change'] = table['position'] -  table_final['position']
table_final.sort_values('position', inplace=True, ascending=True)


# # Final table

# In[ ]:


print(table_final)


# # Complete

# After some preprocessing and EDA, I used a range of classification algorithms to test which would best<br>
# predict match result. I found that XGb and RF performed best. I tuned these both and found that<br>
#  RF improved in performance slightly, as did XGB though increased were small, this is probably due to the fact that<br>
# limited computation power did not allow for a full grid search so I limited the ability to tune<br>
# parameters to save time.

# Note: we could have developed individual team-level predictive models (or combined both) - but did not have time currently.

# I then forecasting the final 6 games to the end of the season using VAR. I had planned to forecast all<br>
# predictive variables but soon realised we did not have anywhere near enough observations for this.<br>
# I chose to forecast the 15 most importance variables (according to RF importance) in smaller batches of 5.<br>
# This is not ideal, and for this reason the forecasts are not too informative (though with more data may will<br>
# be).

# Lastly, I applied the XGb model generated earlier to predict the target (match result) in the final 6 games. <br>
# I then put the league table back together based on the results of the newly predicted games and calculated<br>
# a change in the table.

# The biggest climbers in the final 6 games will be Burnley!

# 

# In[ ]:


os.chdir(r"C:/Users/jaket")
get_ipython().system('jupyter nbconvert --to html EPL_2020_predictions.ipynb')


# In[ ]:




