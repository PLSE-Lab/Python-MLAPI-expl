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
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


corona_virus_data =  pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")


# In[ ]:


corona_virus_data.head()


# In[ ]:


corona_virus_data['Last Update'] = corona_virus_data['Last Update'].apply(pd.to_datetime)
corona_virus_data.drop(['Sno'],axis=1,inplace=True)
corona_virus_data.head()


# In[ ]:


corona_virus_data.tail()


# In[ ]:


corona_virus_data.info()


# In[ ]:


country = corona_virus_data["Country"]
confirmed = corona_virus_data["Confirmed"]
deaths = corona_virus_data["Deaths"]
recovered = corona_virus_data["Recovered"]

total_confirmed_cases = confirmed.sum()
total_deathes_cases = deaths.sum()
total_recovered_cases = recovered.sum()
current_suffering_cases = int(total_confirmed_cases) - (int(total_deathes_cases) + int(total_recovered_cases))

print("Total confirmed cases: ",total_confirmed_cases)
print("Total death cases: ",total_deathes_cases)
print("Total recovered cases: ",total_recovered_cases)
print("Total suffering cases: ",current_suffering_cases)


# In[ ]:


labels = 'Total confirmed cases', 'Total death cases', 'Total recovered cases', 'Total suffering cases'
sizes = [total_confirmed_cases, total_deathes_cases, total_recovered_cases, current_suffering_cases]
explode = (0, 0.1, 0, 0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.show()


# In[ ]:


sns.pairplot(corona_virus_data)


# In[ ]:


plt.ylabel("Country")
plt.xlabel("Confirmed")
plt.title("Corona Virus Analysis")
plt.barh(country,confirmed,label="Country vs Confirmed")
plt.legend()


# In[ ]:


plt.ylabel("Country")
plt.xlabel("Confirmed")
plt.title("Corona Virus Analysis")
plt.barh(country,deaths,label="Country vs Deaths")
plt.legend()


# In[ ]:


plt.ylabel("Country")
plt.xlabel("Confirmed")
plt.title("Corona Virus Analysis")
plt.barh(country,recovered,label="Country vs Recovered")
plt.legend()


# In[ ]:


plt.ylabel("Country")
plt.xlabel("Confirmed")
plt.title("Corona Virus Analysis")
plt.barh(country,confirmed,label="Country vs Confirmed")
plt.barh(country,deaths,label="Country vs Deaths")
plt.barh(country,recovered,label="Country vs Recovered")
plt.legend()


# In[ ]:




