#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Santander Customer Satisfaction Solution using h2o

# In[ ]:


import h2o
import pandas as pd
import numpy as np
from h2o.estimators.glm import H2OGeneralizedLinearEstimator


# ### Initialize h2o

# In[ ]:


h2o.init()


# ### Import train dataset

# In[ ]:


df = h2o.import_file("../input/santander-customer-satisfaction/train.csv")
df.summary()


# In[ ]:


df.col_names


# In[ ]:


y = 'TARGET'
x = df.col_names
x.remove(y)
x.remove('ID')


# In[ ]:


print("Response = " + y)
print("Pridictors = " + str(x))


# In[ ]:


df['TARGET'] = df['TARGET'].asfactor()
df['TARGET'].levels()


# In[ ]:


train, test = df.split_frame(ratios=[.7],seed = 2019)
print(df.shape)
print(train.shape)
#print(valid.shape)
print(test.shape)


# In[ ]:


glm_logistic = H2OGeneralizedLinearEstimator(family = "binomial")


# In[ ]:


glm_logistic.train(x=x, y= y, training_frame=train, 
                   validation_frame=test, model_id="glm_logistic")


# In[ ]:


print(glm_logistic.confusion_matrix() )
print(glm_logistic.auc())
print(glm_logistic.varimp_plot())


# ### Import test dataset

# In[ ]:


df_test_pd = pd.read_csv('../input/santander-customer-satisfaction/test.csv')


# In[ ]:


df_test = h2o.H2OFrame(df_test_pd)
df_test.col_names


# In[ ]:


x_test = df_test.col_names.remove('ID')
y_pred = glm_logistic.predict(test_data=df_test)
y_pred_df = y_pred.as_data_frame()


# In[ ]:


p1 = y_pred_df["p1"]
id = df_test_pd["ID"]


# ### Shutdown h2o

# In[ ]:


h2o.cluster().shutdown()


# In[ ]:


submit = pd.DataFrame({'ID':id, 'TARGET':p1})
submit.to_csv("submit_h2o_glm.csv",index=False)

