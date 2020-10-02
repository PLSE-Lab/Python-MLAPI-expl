#!/usr/bin/env python
# coding: utf-8

# # Load the train dataset

# In[ ]:


import pandas as pd
train = pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')
train.head()


# # Install PyCaret

# In[ ]:


get_ipython().system('pip install pycaret')


# # Initializing Setup

# In[ ]:


from pycaret.classification import *
clf1 = setup(data = train, target = 'price_range', session_id = 786, silent = True)

#silent is True to perform unattended run when kernel is executed.


# # Compare Models

# In[ ]:


get_ipython().run_cell_magic('time', '', 'compare_models()')


# # Create Model

# In[ ]:


# create knn model
knn = create_model('knn')


# In[ ]:


# create catboost model
catboost = create_model('catboost')


# # Tune Model

# In[ ]:


# tune knn model
tuned_knn = tune_model('knn', optimize = 'Accuracy', n_iter = 100)


# In[ ]:


# parameters of tuned_knn
print(tuned_knn)


# In[ ]:


tuned_catboost = tune_model('catboost', optimize = 'Accuracy', n_iter = 100)


# In[ ]:


tuned_lightgbm = tune_model('lightgbm', optimize = 'Accuracy', n_iter = 100)


# In[ ]:


tuned_ada = tune_model('ada', optimize = 'Accuracy', n_iter = 100)


# In[ ]:


tuned_lr = tune_model('lr', optimize = 'Accuracy', n_iter = 100)


# # Ensemble Model

# In[ ]:


dt = create_model('dt')


# In[ ]:


bagged_dt = ensemble_model(dt, n_estimators = 100)


# # Plot Model

# In[ ]:


# auc
plot_model(bagged_dt)


# In[ ]:


# confusion matrix
plot_model(bagged_dt, plot = 'confusion_matrix')


# In[ ]:


# boundary
plot_model(bagged_dt, plot = 'boundary')


# In[ ]:


# vc
plot_model(bagged_dt, plot = 'dimension')


# # Predict on holdout set

# In[ ]:


pred_holdout = predict_model(bagged_dt)


# # Finalize Model 

# In[ ]:


final_dt = finalize_model(bagged_dt)


# # Predictions

# In[ ]:


test = pd.read_csv('/kaggle/input/mobile-price-classification/test.csv')
test.head()


# In[ ]:


predictions = predict_model(final_dt, data=test)


# In[ ]:


predictions.head()

