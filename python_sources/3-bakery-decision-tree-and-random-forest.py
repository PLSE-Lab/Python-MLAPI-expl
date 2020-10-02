#!/usr/bin/env python
# coding: utf-8

# ### Random forest and decision tree

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


data_dummis = pd.read_pickle('../input/data-for-model/data_dummis_dataframe_with_events.pkl')


# In[ ]:


data_dummis.info()


# In[ ]:


X = data_dummis[['Week_Day_Friday', 'Week_Day_Monday',
       'Week_Day_Saturday', 'Week_Day_Sunday', 'Week_Day_Thursday',
       'Week_Day_Tuesday','is_local_event', 'is_bank_holiday']]
y = data_dummis['Item']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)

print(len(X_train), len(X_test))
print(len(y_train), len(y_test))


# ### Used a grid search to tuned the decision tree regressor
# - Need to have in mind that when tuning a decision tree we can create an over fitter model

# In[ ]:


estimator = DecisionTreeRegressor()

grid = GridSearchCV(estimator,
                    param_grid={'max_depth': range(1,10,1), 'min_samples_leaf': range(1,10,1)},
                    scoring='neg_mean_squared_error',
                    return_train_score=True,
                    cv=5, 
                   iid=True)

grid.fit(X_train,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_


# In[ ]:


dt_tuned = DecisionTreeRegressor(max_depth=4)
dt_tuned.fit(X_train, y_train)
#export_graphviz(dt_tuned, 'Data/Bakery_dt_tuned.dot', feature_names=X.columns)


# In[ ]:


dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
#export_graphviz(dt, 'Data/Bakery.dot', feature_names=X.columns)


# In[ ]:


dt_half_tuned = DecisionTreeRegressor(max_depth=3)
dt_half_tuned.fit(X_train, y_train)
#export_graphviz(dt_half_tuned, 'Data/Bakery_dt_half_tuned.dot', feature_names=X.columns)


# In[ ]:


# Compute the feature importances (the Gini index at each node).
dt_feature_importances = pd.DataFrame({'feature':X.columns, 'DT':dt.feature_importances_, 'DT_Tuned':dt_tuned.feature_importances_,'DT_half_Tuned':dt_half_tuned.feature_importances_}).sort_values(by='DT', ascending=False)
dt_feature_importances


# In[ ]:


dt_feature_importances.plot.bar()


# ### Let's also fit a model with Random Forest and compare

# In[ ]:


rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_train)


# ### Find the best parametors for the random forest

# In[ ]:


rf_grid = GridSearchCV(estimator=RandomForestRegressor(),
                    param_grid={"n_estimators": [1, 3, 5, 7 ,9, 10, 11, 13, 15, 17, 19, 21],
                                "min_samples_split": [2, 5, 10],
                                "min_samples_leaf": [1, 5, 10, 25],
                                "max_depth": [1, 3, 5, 7, 10, None],
                                #"max_features": [0, 10],
                                "bootstrap": [True, False]},
                    scoring="neg_mean_squared_error",
                    cv=5)

rf_grid.fit(X_train, y_train)


# In[ ]:


rf_grid.best_params_


# In[ ]:


rf_tuned = RandomForestRegressor(bootstrap=True,max_depth=5,min_samples_leaf=1,min_samples_split=5,n_estimators=17)
rf_tuned.fit(X_train, y_train)


# In[ ]:


y_pred_rf_tuned = rf_tuned.predict(X_train)
y_pred_dt = dt.predict(X_train)
y_pred_dt_tuned = dt_tuned.predict(X_train)


# ### Get the RMSE and scores for all model

# In[ ]:


#Get the RMSE and score of all models in a dataframe
RMSE_SCORES_results = pd.DataFrame(columns=['Model', 'Train_RMSE', 'Test_RMSE', 'Train_Score','Test_Score'], index=range(5))
model_list = [dt,dt_tuned,dt_half_tuned,rf,rf_tuned]
y_pred_mean_train = [y_train.mean()] * len(y_train)
y_pred_mean_test = [y_test.mean()] * len(y_test)

for i, item in enumerate(model_list):
    rmse_train = np.sqrt(mean_squared_error(y_train, item.predict(X_train)))
    rmse_test = np.sqrt(mean_squared_error(y_test, item.predict(X_test)))
    score_train = item.score(X_train, y_train)
    score_test = item.score(X_test, y_test)
    item_str = str(item)
    item_name = item_str[:16]
    if i == 1 or i == 4:
        item_name = item_name + '_Tuned'
    elif i == 2:
        item_name = item_name + '_half_Tuned'

    RMSE_SCORES_results.loc[i] = [item_name, rmse_train, rmse_test, score_train,score_test]

RMSE_SCORES_results.loc[5] = ['Baseline',  np.sqrt(metrics.mean_squared_error(y_train, y_pred_mean_train)), np.sqrt(metrics.mean_squared_error(y_test, y_pred_mean_test)),0.0, 0.0]

RMSE_SCORES_results.index = RMSE_SCORES_results.Model


# In[ ]:


RMSE_SCORES_results


# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(15,8), sharex=False, sharey=True, squeeze=False)

fig.suptitle('RMSE and Score', fontsize=12)
fig.text(0.06, 0.5, 'Total Item Sold', ha='center', va='center', rotation='vertical', )
#fig.text(0.5, 0.04, 'Hours', ha='center', va='center', rotation=45)

RMSE_train_graph = RMSE_SCORES_results['Train_RMSE']
RMSE_train_graph.plot(ax=axes[0][0], grid=True, kind='barh', title='RMSE for train')

RMSE_test_graph = RMSE_SCORES_results['Test_RMSE']
RMSE_test_graph.plot(ax=axes[0][1], grid=True, kind='barh', title='RMSE for test')

score_train_graph = RMSE_SCORES_results['Train_Score']
score_train_graph.plot(ax=axes[1][0], grid=True, kind='barh', title='Score for train')

score_test_graph = RMSE_SCORES_results['Test_Score']
score_test_graph.plot(ax=axes[1][1], grid=True, kind='barh', title='Score for test')


# ### Let run cross validation
# - All models have a little of a variance problem

# In[ ]:


def cross_validation_test(model_name, x_data, y_data, scoring_name, n):
    cv_scores = cross_val_score(model_name, x_data, y_data, scoring=scoring_name, cv=n)
    return np.sqrt(-cv_scores), np.sqrt(-cv_scores.mean())


# In[ ]:


RMSE_SCORES_results = pd.DataFrame(columns=['Model', 'RMSE', 'Average RMSE'], index=range(2))
model_list = [dt,dt_half_tuned, dt_tuned,rf,rf_tuned]
          
for item in model_list:
    model_string = str(item)
    print('Results for ' + model_string[:10])
    print(cross_validation_test(item,  X, y, 'neg_mean_squared_error', 5))


# ### Look at the results in graphs

# In[ ]:


results_dt_rf_pred_train = pd.DataFrame({'Actual': y_train, 'DT_Pred': y_pred_dt,'DT_Tuned_Pred': y_pred_dt_tuned,'RF_Pred': y_pred_rf, 'RF_Tuned_Pred': y_pred_rf_tuned})


# In[ ]:


results_dt_rf_pred_train[['Actual', 'DT_Pred', 'DT_Tuned_Pred','RF_Tuned_Pred', 'RF_Pred']].plot(figsize=(15,7), style={'Actual': '-or', 'DT_Pred': '-ob','DT_Tuned_Pred': '-oy','RF_Pred': '-og', 'RF_Tuned_Pred': '-om'})


# In[ ]:


results_dt_rf_pred_train.boxplot()


# In[ ]:


results_dt_rf_pred_test = pd.DataFrame({'Actual': y_test, 'DT_Pred': dt.predict(X_test),'DT_Tuned_Pred': dt_tuned.predict(X_test), 'RF_Pred': rf.predict(X_test), 'RF_Tuned_Pred': rf_tuned.predict(X_test)})


# In[ ]:


results_dt_rf_pred_test.describe()


# In[ ]:


results_dt_rf_pred_test.boxplot()


# In[ ]:


results_dt_rf_pred_test[['Actual', 'DT_Pred', 'RF_Pred', 'RF_Tuned_Pred']].plot(figsize=(15,7), style={'Actual': '-or', 'DT_Pred': '-ob', 'RF_Pred': '-oy', 'RF_Tuned_Pred': '-og'}, grid=True)


# In[ ]:


results_dt_rf_pred_train.to_pickle('results_dt_rf_pred_train.pkl')


# In[ ]:


results_dt_rf_pred_test.to_pickle('results_dt_rf_pred_test.pkl')


# ### Conclusion
# - The results for all the models are very similar
# - The decision tress and random forest tuned are able to predict correctly bank holidays
# - All models underestimates days with really high sales
# - In general results are very similar to the linear regression models

# In[ ]:




