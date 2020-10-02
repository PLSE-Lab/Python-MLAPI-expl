#!/usr/bin/env python
# coding: utf-8

# # Predicting Car Prices for VW Passat using CatBoost
# I've already added exploratory data analysis notebook for this dataset and built a regression model for the whole dataset. Now I would like to explore the hierarchy of features in the SHAP summary plot for different models. Based on my domain knowledge of the current subject (used cars) I suspect that there will be a difference in hierarchies. 

# In[ ]:


import pandas as pd
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns
import shap
import eli5
from collections import Counter

import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

shap.initjs()

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Load the dataset

# In[ ]:


df = pd.read_csv('/kaggle/input/usedcarscatalog/cars.csv')
df.shape


# ### Limit the dataset to a single car model
# Will start with VW Passat model as it is the most popular model in the catalog. It is considered to be the most reliable and practical

# In[ ]:


df = df.loc[df['model_name']=='Passat']
df.shape


# In[ ]:


df.price_usd.mean()


# ### Create a simple train-test split

# In[ ]:


from sklearn.model_selection import train_test_split 

X = df.drop('price_usd', axis=1)
y = df['price_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


print("Number of cars in X_train dataset: ", X_train.shape) 
print("Number of cars in y_train dataset: ", y_train.shape) 
print("Number of cars in X_test dataset: ", X_test.shape) 
print("Number of cars in y_test dataset: ", y_test.shape)


# In[ ]:


get_ipython().run_cell_magic('time', '', "# create train_pool object\nfrom catboost import CatBoostRegressor\nfrom catboost import Pool\nfrom catboost import MetricVisualizer\n\n\n\ncat_features=['manufacturer_name', \n              'model_name', \n              'transmission', \n              'color', \n              'engine_fuel',\n              'engine_has_gas',\n              'engine_type', \n              'body_type', \n              'has_warranty', \n              'state', \n              'drivetrain',\n              'is_exchangeable', \n              'location_region',\n              'feature_0',\n              'feature_1',\n              'feature_2',\n              'feature_3',\n              'feature_4',\n              'feature_5',\n              'feature_6',\n              'feature_7',\n              'feature_8',\n              'feature_9',]\n\ntrain_pool = Pool(\n    data=X_train, \n    label=y_train,\n    cat_features = cat_features\n)\n\n# create validation_pool object\nvalidation_pool = Pool(\n    data=X_test, \n    label=y_test,\n    cat_features = cat_features\n)")


# ### Train the model

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# pretty basic model, max_depth=10 give slightly better results\ncbs = CatBoostRegressor(iterations=4000,\n                         learning_rate=0.012,\n                         loss_function='MAE',\n                         max_depth=10, \n                         early_stopping_rounds=200,\n                         cat_features = cat_features)\n\n# we are passing categorical features as parameters here\ncbs.fit(\n    train_pool,\n    eval_set=validation_pool,\n    verbose=False,\n    plot=True \n);")


# In[ ]:


error = test_predictions - y_test
# print(type(error))

plt.figure(figsize=(10,10))
plt.scatter(y_test, 
            test_predictions, 
            c=error,
            s=2,
            cmap='hsv',
            )
plt.colorbar()
plt.xlabel('True Values [price_usd]')
plt.ylabel('Predictions [price_usd]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, 20000])
plt.ylim([0, 20000])
plt.show()


# In[ ]:


plt.figure(figsize=(16,7))
plt.hist(error, bins = 40, rwidth=0.9)
plt.xlabel('Predictions Error [price_usd]')
_ = plt.ylabel('Count')
plt.xlim([-6000, 6000])
plt.show()


# ### Explore feature importances

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nimportance_types = ['PredictionValuesChange',\n                    'LossFunctionChange'\n                   ]\n\n\nfor importance_type in importance_types:\n    print(importance_type)\n    print(cbs.get_feature_importance(data=train_pool, \n                                     type=importance_type))\n    print('\\n\\n\\n\\n')")


# ### Calculate SHAP values

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nimport shap\nshap.initjs()\n\nshap_values = cbs.get_feature_importance(Pool(X_test, \n                                              label=y_test,\n                                              cat_features=cat_features), \n                                         type="ShapValues")\nprint(type(shap_values))\n\nexpected_value = shap_values[0,-1]\nprint(expected_value)\n\nshap_values = shap_values[:,:-1]')


# In[ ]:


shap.summary_plot(shap_values, X_test, max_display=X_test.shape[1])


# The most important feature is, as expected, year_produced.

# In[ ]:


shap.dependence_plot(ind='year_produced', interaction_index='year_produced',
                     shap_values=shap_values, 
                     features=X_test,  
                     display_features=X_test)


# In[ ]:


shap.dependence_plot(ind='odometer_value', interaction_index='odometer_value',
                     shap_values=shap_values, 
                     features=X_test,  
                     display_features=X_test)


# In[ ]:


shap.dependence_plot(ind='engine_capacity', interaction_index='engine_capacity',
                     shap_values=shap_values, 
                     features=X_test,  
                     display_features=X_test)


# number_of_photos that a particular has in a catalog is an important feature too, because cheap and old cars rarely have lots of photos because there is just not too much to show.

# In[ ]:


shap.dependence_plot(ind='number_of_photos', interaction_index='number_of_photos',
                     shap_values=shap_values, 
                     features=X_test,  
                     display_features=X_test)


# In[ ]:



shap.force_plot(expected_value, shap_values[:1000,:], X_test.iloc[:1000,:])


# ### Explore predictions, SHAP values and listed prices for each model in the test set.

# In[ ]:


for i in range(50,70):
    print('Sample', i, 'from the test set:')
    display(shap.force_plot(expected_value, shap_values[i,:], X_test.iloc[i,:]))
    print('Listed_price -------------------------------------->', y_test.iloc[i])
    print('parameters:\n', X_test.iloc[i,:])
    print('\n\n\n\n\n\n\n')


# ## Conclusion
# I feel like CatBoost was the ideal choice for this particular dataset. The model performes reasonably well with 645 USD MAE, which is not bad at all given the 5100 USD mean price for the VW Passat. It also important to note that there are a lot of samples in the catalog that are listed for a very long periods of time, exceeding hundreds of days (check my EDA on this dataset). These samples may be potentially penalized. 
