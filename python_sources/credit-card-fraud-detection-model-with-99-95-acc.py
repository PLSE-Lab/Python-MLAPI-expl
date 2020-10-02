#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np # linear algebra




import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# This is Credit card fraud detection model using pycaret library.

# In[ ]:


import pandas as pd


# In[ ]:


df=pd.read_csv("../input/creditcardfraud/creditcard.csv")


# In[ ]:


df.head()


# In[ ]:


get_ipython().system('pip install pycaret')


# In[ ]:


from pycaret.classification import *


# In[ ]:


clf=setup(df,target="Class")


# In[ ]:


compare_models()


# In[ ]:


et=create_model("et")


# We are getting accuracy of 99.95%. This is really awesome.

# Let's print our model now

# In[ ]:


print(et)


# ROC Curve:

# In[ ]:


plot_model(et, plot = 'auc')


# Precision Curve

# In[ ]:


plot_model(et, plot = 'pr')


# In[ ]:


plot_model(et, plot='feature')


# Confusion Matrix

# In[ ]:


plot_model(et, plot = 'confusion_matrix')


# Model Evaluation:

# In[ ]:


evaluate_model(et)


# Prediction:

# In[ ]:


predict_model(et);


# Thank you
