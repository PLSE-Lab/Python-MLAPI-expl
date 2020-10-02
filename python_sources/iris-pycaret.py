#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

iris = pd.read_csv("/kaggle/input/iris/Iris.csv") # the iris dataset is now a Pandas DataFrame


warnings.filterwarnings("ignore")


# In[ ]:


get_ipython().system('pip install pycaret')


# In[ ]:


iris.drop("Id",axis = 1, inplace = True)
iris.head()


# In[ ]:


from pycaret.classification import *


# In[ ]:


msk = np.random.rand(len(iris)) < 0.75
train = iris[msk]
test = iris[~msk]

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)


# In[ ]:


clf1 = setup(data = train, 
             target = "Species",
             silent = True,
             remove_outliers = True,
             feature_selection = True)


# In[ ]:


lgbm  = create_model('lightgbm') 


# In[ ]:


tuned_lightgbm = tune_model('lightgbm')


# In[ ]:


evaluate_model(tuned_lightgbm)


# In[ ]:


pred_lgbm = predict_model(tuned_lightgbm, data=test)
pred_lgbm['preds'] = pred_lgbm['Label']


# In[ ]:


pred_lgbm.head()


# In[ ]:


knn = create_model('knn')


# In[ ]:


tuned_knn = tune_model('knn')


# In[ ]:


plot_model(tuned_knn, plot = 'confusion_matrix')


# In[ ]:


plot_model(tuned_lightgbm, plot = 'confusion_matrix')


# In[ ]:


plot_model(tuned_knn, plot='boundary')


# In[ ]:


plot_model(tuned_knn, plot = 'error')


# In[ ]:


final_knn = finalize_model(tuned_knn)
final_lgbm = finalize_model(tuned_lightgbm)


# In[ ]:


predict_model(final_knn);


# In[ ]:


predict_model(final_lgbm);


# In[ ]:


plot_model(estimator = tuned_lightgbm, plot = 'learning')


# In[ ]:


plot_model(estimator = tuned_lightgbm, plot = 'auc')

