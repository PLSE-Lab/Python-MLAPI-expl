#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[ ]:


# python general
import pandas as pd
import numpy as np
from collections import OrderedDict

#scikit learn

import sklearn
from sklearn.base import clone

# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# ML models
from sklearn.ensemble import RandomForestRegressor

# error metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error


# In[ ]:


def mape(y_true, y_pred):
    y_val = np.maximum(np.array(y_true), 1e-8)
    return (np.abs(y_true -y_pred)/y_val).mean()


# In[ ]:


metrics_dict_res = OrderedDict([ ('mean abs perc error', mape) ])


# In[ ]:


def regression_metrics_yin(y_train, y_train_pred, y_test, y_test_pred,
                           metrics_dict, format_digits=None):
    df_results = pd.DataFrame()
    for metric, v in metrics_dict.items():
        df_results.at[metric, 'train'] = v(y_train, y_train_pred)
        df_results.at[metric, 'test'] = v(y_test, y_test_pred)

    if format_digits is not None:
        df_results = df_results.applymap(('{:,.%df}' % format_digits).format)

    return df_results


# ## Define features

# In[ ]:


numeric_features = ['Curb_Weight','year']

all_numeric_features = list(numeric_features)

target = ['Price_USD']

target_name = 'Price_USD'


# # Global options

# In[ ]:


ml_model_type = 'Random Forest'

regression_metric = 'mean abs perc error'

do_grid_search_cv = False
scoring_greater_is_better = False  # THIS NEEDS TO BE SET CORRECTLY FOR CV GRID SEARCH

do_retrain_total = True
write_predictions_file = False

# relative size of test set
test_size = 0.3
random_state = 33


# # Load  data
# 

# ## Training data

# In[ ]:


df = pd.read_csv('/kaggle/input/ihsm-sample/train_sample.csv', index_col='vehicle_id')


# In[ ]:


df.head(5)


# ## Out of sample data (to predict)

# In[ ]:


df_oos = pd.read_csv('/kaggle/input/ihsm-sample/oos_sample.csv', index_col='vehicle_id')


# In[ ]:


df_oos.head()


# # Feature exploration

# ## Numerical features

# In[ ]:


# summary statistics
df[numeric_features + target].describe()


# In[ ]:


features = numeric_features
model_columns = features + [target_name]
len(model_columns)


# #  ML data preparation

# In[ ]:


#dataframe for further processing
df_proc = df[model_columns].copy()
df_proc.shape


# ## Train test split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df_proc[features], df_proc[target_name], 
                                                    test_size=test_size, random_state=random_state)

print(X_train.shape)
print(X_test.shape)


# ##  Model definition

# In[ ]:



if ml_model_type == 'Random Forest':

 model_hyper_parameters_dict = OrderedDict(n_estimators=10, 
                                           max_depth=4, 
                                           min_samples_split=2, 
                                           max_features='sqrt',
                                           min_samples_leaf=1, 
                                           random_state=random_state, 
                                           n_jobs=4)

 regressor = RandomForestRegressor(**model_hyper_parameters_dict)
     
base_regressor = clone(regressor)    


# ## ML model training

# In[ ]:


regressor.fit(X_train, y_train)


# # Model evaluation

# ## Train, test predictions

# In[ ]:


y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)


# In[ ]:


y_train_pred


# ## Metrics

# In[ ]:


df_regression_metrics = regression_metrics_yin(y_train, y_train_pred, y_test, y_test_pred,
                                               metrics_dict_res, format_digits=3)

df_output = df_regression_metrics.copy()
df_output.loc['Counts','train'] = len(y_train)
df_output.loc['Counts','test'] = len(y_test)
df_output


# # Apply model to OOS data

# In[ ]:


df_oos.head()


# ## Subset to relevant columns

# In[ ]:


df_proc_oos = df_oos[features].copy()
#df_proc_oos[target_name] = 1


# ## Apply model and produce output

# In[ ]:


y_oos_pred = regressor.predict(df_proc_oos)


# In[ ]:


df_proc_oos.head()


# In[ ]:


id_col = 'vehicle_id'
df_out = (pd.DataFrame(y_oos_pred, columns=[target_name], index=df_proc_oos.index)
            .reset_index()
            .rename({'index': id_col}, axis=1))


# In[ ]:


df_out.head()


# In[ ]:


df_out.shape


# In[ ]:


df_out.to_csv('submission.csv', index=False)

