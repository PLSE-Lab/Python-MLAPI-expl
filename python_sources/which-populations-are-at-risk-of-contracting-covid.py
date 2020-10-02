#!/usr/bin/env python
# coding: utf-8

# # The below analysis is based upon a review/understanding of COVID-19 outbreaks using datasets available under UNCOVER *dir*

# >Loading useful libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


# >Checking Datasets which are related to popluation/health/risk in order to gather intel and collate results for populations which are at risk of contracting COVID-19?
# 

# In[ ]:


from collections import deque 
datasets=deque()
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #1st test of datasets containing health in filenames
        if "health" in filename:
#             print(os.path.join(dirname, filename))
            datasets.append(pd.read_csv(os.path.join(dirname, filename)))


# >Picking random data for EDA!

# In[ ]:


datasets[1].columns


# In[ ]:


#No. of address present in city ALAMOSA
#If we can get the data of age groups may be we can add more info for the populations which are at higher risk!
# test_data.groupby('city').count()[['state','address']]
#It seems this data contains the health services present in cities of CO State
#Let's tag this data and move to second data set 
test_data_eda=datasets[1]


# >As per our dataset maximum services are offered by City DENVER

# In[ ]:


max_operating_services=test_data_eda.groupby('city').count()['operating'].reset_index()
max_operating_services[max_operating_services['operating']==max_operating_services['operating'].max()]


# In[ ]:


plt.plot(max_operating_services.sort_values(by='operating').tail()['city'],max_operating_services.sort_values(by='operating').tail()['operating'])
plt.xticks(rotation='vertical')
plt.show()


# 

# In[ ]:




