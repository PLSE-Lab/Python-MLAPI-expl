#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pycaret')


# In[ ]:


from pycaret.datasets import get_data
data=get_data('anomaly')
from pycaret.anomaly import *
setup=setup(data)


# In[ ]:


iforest=create_model('iforest')
plot_model(iforest)


# In[ ]:


knn=create_model('knn')
plot_model(knn)


# In[ ]:


knn_prediction=predict_model(knn, data=data)


# In[ ]:


knn_prediction

