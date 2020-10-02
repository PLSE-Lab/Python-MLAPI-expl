#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


forcast_data_calendar= pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
forcast_data_calendar


# In[ ]:


forcast_sales_eval= pd.read_csv('../input/m5-forecasting-accuracy/sales_train_evaluation.csv')
forcast_sales_eval
#We take this data frame about evaluation


# In[ ]:


forcast_sales_val= pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
forcast_sales_val
#But the validation under evaluation


# In[ ]:


forcast_sample_sumb= pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')
forcast_sample_sumb
#The submission beside the eval and valid


# In[ ]:


forcast_sell_prices= pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
forcast_sell_prices
#join the sell price by item_id


# ## Frist but evaluation data under the validation

# In[ ]:


result4 = pd.concat([forcast_sales_eval,forcast_sales_val],join='outer')
result4


# ## Take the result and merge it with sample sumbition by item_id

# In[ ]:


result5 = pd.merge(result4,forcast_sample_sumb,how='outer',on='id')
result5


# ## Save the result as FEvalVal_concat.csv

# In[ ]:


result5.to_csv('FEvalVal_concat.csv')
result4= pd.read_csv('FEvalVal_concat.csv')


# ## Drop the duplication in forcast data and keep the last value
# 

# In[ ]:


forcast_sell_prices2=forcast_sell_prices.drop_duplicates(subset='item_id')
forcast_sell_prices2


# ## Merge the FEvalVal_concat.csv and sell price by item_id
# 

# In[ ]:


result5 = pd.merge(result4,forcast_sell_prices2, how='outer',on='item_id')
result5


# ## Save the result data for Regression model

# In[ ]:


result5.to_csv('finall.csv')

