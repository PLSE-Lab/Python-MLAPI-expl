#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime


# > <b>Loading dataset</b>

# In[ ]:


import pandas as pd
case = pd.read_csv("../input/coronavirusdataset/case.csv")
patient = pd.read_csv("../input/coronavirusdataset/patient.csv")
route = pd.read_csv("../input/coronavirusdataset/route.csv")
time = pd.read_csv("../input/coronavirusdataset/time.csv")
trend = pd.read_csv("../input/coronavirusdataset/trend.csv")


# In[ ]:


patient.head()


#  Changing table header of state = current_condition and sex = gender

# In[ ]:


patient.rename(columns = {'state': 'current_condition'}, inplace = True)
patient['current_condition'] = patient.current_condition.str.upper()

patient.rename(columns = {'sex': 'gender'}, inplace = True)
patient['gender'] = patient.gender.str.upper()



patient.head()


# grouping data on the basis of confirmed date

# In[ ]:


date = patient.groupby('confirmed_date').count()
date.head()


# In[ ]:


date_confirmed = date['patient_id']
date_confirmed.head()


# plotting on line graph

# In[ ]:


import matplotlib
matplotlib.rcParams['figure.figsize'] = (16,8)
date_confirmed.plot()


# showing patient id on line graph

# In[ ]:


date_confirmed.plot(marker = '.')
plt.grid(which = 'both')


# Let's group the data set on the basis of gender to see which gender is most effected.

# In[ ]:


patient.groupby('gender').count()


# Above we can see that more than 50% of gender data are missing and it is not a numeric data so we can't interpolate the data. 

# Now grouping on the basis of country

# In[ ]:


patient1 = patient.groupby('country').count()
patient1


# In[ ]:


exp_vals = patient1["patient_id"].values.tolist()


# In[ ]:


exp_vals


# We can see that only 3 country is present so we will try to plot it on pie chart

# In[ ]:


exp_labels =["china", "korea", "mongolia"]
plt.axis("equal")
plt.pie(exp_vals,labels= exp_labels, radius = 1.0, autopct = '%0.1f%%', explode = [1,0.1,3], startangle = 180)
plt.show


# Above pie chart shows that the most effected country is korea. China has a very low count followed by mongolia.

# Now to apply safety measures we will see the main reasons that transmits this disease.

# In[ ]:


patient.infection_reason.unique()


# In[ ]:


transmission_reason = patient.groupby('infection_reason').count()
transmission_reason


# In[ ]:


reason = transmission_reason['patient_id']
reason


# In[ ]:


reason.plot.bar()


# Here we can see that direct contact with the patient is the main reason for virus communication followed by visit to Daegu followed by visit to Wuhan. Minimizing these acts is the prime target to stay away from the virus.

# Now let's see the virus most effected regions.

# In[ ]:


affected_region = patient.groupby('region').count()
affected_region


# In[ ]:


affected_region1 = affected_region['patient_id']
affected_region1


# In[ ]:


affected_region1.plot.bar()


# Above we can see the most affected regions.

# Now let's see the current_condition of the patients who have corona virus.

# In[ ]:


condition = patient.groupby("current_condition").count()
condition 


# In[ ]:


current_condition =condition['patient_id']
current_condition


# In[ ]:


current_condition.plot.bar()


# We all know that many people are affected by the virus and everyone is afraid of COVID19. A simple visulization above shows that the virus is not as fatal as everyone says. People are recovering and many are in isolation and decease rate is very low.

# Simply washing your hands and staying away from infected person can prevent one from getting this virus.

# In[ ]:




