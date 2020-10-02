#!/usr/bin/env python
# coding: utf-8

# # 1. Import Dataset

# In[ ]:


import pandas as pd
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()


# # 2. Install PyCaret

# In[ ]:


get_ipython().system('pip install pycaret')


# # 3. Iteration 1: Setup without Preprocessing

# In[ ]:


from pycaret.regression import *
reg1 = setup(train, target = 'SalePrice', session_id = 123, silent = True) #silent is set to True for unattended run during kernel execution


# ## 3.1 Compare Models

# In[ ]:


compare_models(blacklist = ['tr']) #blacklisted Thielsen Regressor due to longer training times


# ## 3.2 Create and Store Models in Variable

# In[ ]:


catboost = create_model('catboost', verbose = False) #verbose set to False to avoid printing score grid
gbr = create_model('gbr', verbose = False)
xgboost = create_model('xgboost', verbose = False)


# ## 3.3 Blend Models

# In[ ]:


blend_top_3 = blend_models(estimator_list = [catboost, gbr, xgboost])


# - No significant improvement after blending. Best individual model is Catboost with `0.1313` RMSLE. Blender RMSLE is `0.1364`.

# ## 3.4 Stack Models

# In[ ]:


stack1 = stack_models(estimator_list = [gbr, xgboost], meta_model = catboost, restack = True)


# - No improvement from stacking. Best model still Catboost Regressor with default hyperparameters with RMSLE `0.1313`.

# # 4. Iteration 2: Setup with Preprocessing

# In[ ]:


from pycaret.regression import *
reg1 = setup(train, target = 'SalePrice', session_id = 123, 
             normalize = True, normalize_method = 'zscore',
             transformation = True, transformation_method = 'yeo-johnson', transform_target = True,
             ignore_low_variance = True, combine_rare_levels = True,
             numeric_features=['OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 
                               'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 
                               'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'PoolArea'],
             silent = True #silent is set to True for unattended run during kernel execution
             )


# ## 4.1 Compare Models

# In[ ]:


compare_models(blacklist = ['tr']) #blacklisted Thielsen Regressor due to longer training times


# - Catboost Regressor RMSLE slightly improved from `0.1313` to `0.1275` 

# ## 4.2 Create and Store Models in Variable

# In[ ]:


gbr = create_model('gbr', verbose = False)
catboost = create_model('catboost', verbose = False)
svm = create_model('svm', verbose = False)
lightgbm = create_model('lightgbm', verbose = False)
xgboost = create_model('xgboost', verbose = False)


# ## 4.3 Blend Models

# In[ ]:


blend_top_5 = blend_models(estimator_list = [gbr,catboost,svm,lightgbm,xgboost])


# - Blending top models has slightly improved RMSLE from `0.1275` to `0.1265`.

# ## 4.4 Stack Models

# In[ ]:


stack2 = stack_models(estimator_list = [gbr,catboost,lightgbm,xgboost], meta_model = svm, restack = True)


# - No improvement after stacking.

# # 5. Iteration 3: Setup with Advance Preprocessing

# In[ ]:


from pycaret.regression import *
reg1 = setup(train, target = 'SalePrice', session_id = 123, 
             normalize = True, normalize_method = 'zscore',
             transformation = True, transformation_method = 'yeo-johnson', transform_target = True,
             numeric_features=['OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 
                               'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 
                               'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'PoolArea'],
             ordinal_features= {'ExterQual': ['Fa', 'TA', 'Gd', 'Ex'],
                                'ExterCond' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                'BsmtQual' : ['Fa', 'TA', 'Gd', 'Ex'], 
                                'BsmtCond' : ['Po', 'Fa', 'TA', 'Gd'],
                                'BsmtExposure' : ['No', 'Mn', 'Av', 'Gd'],
                                'HeatingQC' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                'KitchenQual' : ['Fa', 'TA', 'Gd', 'Ex'],
                                'FireplaceQu' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                'GarageQual' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                'GarageCond' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                'PoolQC' : ['Fa', 'Gd', 'Ex']},
             polynomial_features = True, trigonometry_features = True, remove_outliers = True, outliers_threshold = 0.01,
             silent = True #silent is set to True for unattended run during kernel execution
             )


# ## 5.1 Compare Models

# In[ ]:


compare_models(blacklist = ['tr']) #blacklisted Thielsen Regressor due to longer training times


# ## 5.2 Tune Models

# In[ ]:


huber = tune_model('huber', n_iter = 100)


# In[ ]:


omp = tune_model('omp', n_iter = 100)


# In[ ]:


ridge = tune_model('ridge', n_iter = 100)


# In[ ]:


br = tune_model('br', n_iter = 100)


# In[ ]:


lightgbm = tune_model('lightgbm', n_iter = 50)


# In[ ]:


par = tune_model('par', n_iter = 100)


# ## 5.3 Blend Models

# In[ ]:


blend_all = blend_models(estimator_list = [huber, omp, ridge, br])


# ## 5.4 Evaluate Bayesian Ridge Model

# In[ ]:


plot_model(br, plot = 'residuals')


# In[ ]:


plot_model(br, plot = 'error')


# In[ ]:


plot_model(br, plot = 'vc')


# In[ ]:


plot_model(br, plot = 'feature')


# ## 5.5 Interpret LightGBM Model

# In[ ]:


interpret_model(lightgbm)


# In[ ]:


interpret_model(lightgbm, plot = 'correlation', feature = 'TotalBsmtSF')


# In[ ]:


interpret_model(lightgbm, plot = 'reason', observation = 0)


# # 6. Finalize Blender and Predict test dataset

# In[ ]:


# check predictions on hold-out
predict_model(blend_all);


# - RMSLE on hold-out is `0.1061` vs. 10 fold CV is `0.1180`.

# In[ ]:


final_blender = finalize_model(blend_all)
print(final_blender)


# In[ ]:


predictions = predict_model(final_blender, data = test)
predictions.head()


# ## END OF NOTEBOOK - THANK YOU.
