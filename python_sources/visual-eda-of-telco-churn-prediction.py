#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Prediction 
# 
# ### What is Churn prediction?
# * Churn Prediction is knowing which users are going to stop using your platform/service in the future. 
# 
# 
# 
# ![](https://www.softwebdatascience.com/images/customer-churn.jpg)
# 
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from IPython.display import IFrame


# ### Dataset have imbalance classes ###

# ### Gender Churn Ratio
# 
# *  Female churn more
# 
# ### Senior Citized Churn Ration
# 
# * Senior Citized less like to churn
# 
# ### Having Partner Impact on Churn 
# 
# * as per partner to churn there is no great impact of partner parameter to churn
# 
# ### Dependents is trouble for operator?
# 
# * there is no much impact of dependents parameter on churn. and yeah no dependent less think to chunk

# In[ ]:


IFrame("https://public.tableau.com/views/ContractPeriodhaveimpactonchurn/ChurnAnalysis?:embed=y&:showVizHome=no", width=1200, height=600)


# ### Contract Period is major parameter for operator ?
# 
# * yes month to month contract customer are more who churns more so company have to plan some long contract tenure

# In[ ]:


IFrame("https://public.tableau.com/views/ContractPeriodhaveimpactonchurn/ContractPeriodhaveimpactonChurn?:embed=y&:showVizHome=no", width=1200, height=600)


# # Electronic Check is moderately accpeted?
# 
# * electronic check have 1071 churn and 1294 not churn(which is approax 45%/55%) so company have to encourage to opt for other payment methods to prevent chrun

# In[ ]:


IFrame("https://public.tableau.com/views/ContractPeriodhaveimpactonchurn/PaymentMethodhavechurn?:embed=y&:showVizHome=no", width=1200, height=300)


# ### Teleco customers likes company go green(paperless billing) conecpt !!!
# 
# * 469 customer have no paperless billing and churned so they might forget to pay bill and charged penalty 
# * 1400 cusotmer have paperless billing and churned so we can conculde that they actually have service issue

# In[ ]:


IFrame("https://public.tableau.com/views/ContractPeriodhaveimpactonchurn/IsPaperlessBillingMajorReasonofChrun?:embed=y&:showVizHome=no", width=1200, height=600)


# ### Multiple lines Leads to churn?
# 
# * there is no great impact of having more than connection on churn
# * if telco focus on 850 customer who have muliple liens means(yes) and churned means they are more service centric customer. telco prevent by some offering some great service

# In[ ]:


IFrame("https://public.tableau.com/views/IsMultipleLinesLeadstoChurn/IsMulitpleLinesLeadstochurn?:embed=y&:showVizHome=no", width=1200, height=600)


# ### Stay Connected For More EDA... 

# In[ ]:




