#!/usr/bin/env python
# coding: utf-8

# # Which populations have contracted COVID-19 and require ventilators?

# Task Details
# The Roche Data Science Coalition is a group of like-minded public and private organizations with a common mission and vision to bring actionable intelligence to patients, frontline healthcare providers, institutions, supply chains, and government. The tasks associated with this dataset were developed and evaluated by global frontline healthcare providers, hospitals, suppliers, and policy makers. They represent key research questions where insights developed by the Kaggle community can be most impactful in the areas of at-risk population evaluation and capacity management.

# In[ ]:


from IPython.display import Image
import os
get_ipython().system('ls ../input/')
Image("../input/iimage/nn.jpg")


# 
# # Neural Network

# Artificial neural networks are relatively crude electronic networks of neurons based on the neural structure of the brain. They process records one at a time, and learn by comparing their prediction of the record (largely arbitrary) with the known actual record. The errors from the initial prediction of the first record is fed back to the network and used to modify the network's algorithm for the second iteration. These steps are repeated multiple times.
# 
# A neuron in an artificial neural network is:
# 
# 1. A set of input values (xi) with associated weights (wi)
# 
# 2. A input function (g) that sums the weights and maps the results to an output function(y).

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as py
import seaborn as sb


# In[ ]:


df = pd.read_csv('../input/data-t/covid-19-tracker-canada.csv')
gf = df.drop(columns=['date','travel_history','id','source','province','city'])
x = gf.groupby(['age'])['confirmed_presumptive'].count()
x


# In[ ]:


py.figure(figsize=(14,5))
py.plot(x)
py.xticks(rotation=90)
py.title("Based on Canada Reports")
py.xlabel("Population infected")
py.ylabel("Age group")
py.show()


# ### Based on country

# In[ ]:


file = pd.read_csv("../input/data-t/johns-hopkins-covid-19-daily-dashboard-cases-by-country.csv")


# In[ ]:


py.figure(figsize=(14,5))
py.plot(file["country_region"][:60],file["confirmed"][:60],"orange")
py.xticks(rotation=90)
py.legend()
py.title("Infected in Country")
py.ylabel("No. of people")
py.xlabel("Country name")
py.show()


# In[ ]:


py.figure(figsize=(14,4))
py.plot(file["country_region"][61:120],file["confirmed"][61:120])
py.xticks(rotation=90)
py.legend()
py.title("Infected in Country")
py.ylabel("No. of people")
py.xlabel("Country name")
py.show()


# In[ ]:


py.figure(figsize=(14,5))
py.plot(file["country_region"][121:180],file["confirmed"][121:180],"red")
py.xticks(rotation=90)
py.legend()
py.title("Infected in Country")
py.ylabel("No. of people")
py.xlabel("Country name")
py.show()


# From above, we can say that following are hotspot countries in the world:
# 1. spain
# 2. Iran
# 3. China 
# 4. France
# 5. Germany
# 6. Italy
# 7. US

# ### Based on states of various countries

# In[ ]:


file1 = pd.read_csv("../input/uploads/johns-hopkins-covid-19-daily-dashboard-cases-by-states.csv")
file1['country_region'].unique()


# In[ ]:


index_us = []
index_cn = []
index_fr = []
index_uk = []
index_cd = []
index_nt = []
index_as = []
index_dm = []

for i in range(0,137):
    if file1['country_region'][i] == 'US':
        index_us.append(i)
    elif file1['country_region'][i] == 'France':
        index_fr.append(i)
    elif file1['country_region'][i] == 'China':
        index_cn.append(i)
    elif file1['country_region'][i] == 'Canada':
        index_cd.append(i)
    elif file1['country_region'][i] == 'Australia':
        index_as.append(i)
    elif file1['country_region'][i] == 'United Kingdom':
        index_uk.append(i)
    elif file1['country_region'][i] == 'Netherlands':
        index_nt.append(i)
    elif file1['country_region'][i] == 'Denmark':
        index_dm.append(i)
    
x = set(index_us)
index_us = list(x)
x = set(index_fr)
index_fr = list(x)
x = set(index_cn)
index_cn = list(x)
x = set(index_cd)
index_cd = list(x)
x = set(index_dm)
index_dm = list(x)
x = set(index_as)
index_as = list(x)
x = set(index_nt)
index_nt = list(x)
x = set(index_uk)
index_uk = list(x)


# In[ ]:


py.plot(file1["province_state"][index_fr],file["confirmed"][index_fr])
py.xticks(rotation=90)
py.legend()
py.title("Infected in State")
py.ylabel("No. of people")
py.xlabel("France states")
py.show()


# In[ ]:


py.figure(figsize=(9,5))
py.plot(file1["province_state"][index_cn],file["confirmed"][index_cn])
py.xticks(rotation=90)
py.title("Infected in State")
py.ylabel("No. of people")
py.xlabel("China states")
py.legend()
py.show()


# In[ ]:


py.figure(figsize=(14,5))
py.plot(file1["province_state"][index_us],file["confirmed"][index_us])
py.xticks(rotation=90)
py.legend()
py.title("Infected in State")
py.ylabel("No. of people")
py.xlabel("US states")
py.show()


# In[ ]:


py.plot(file1["province_state"][index_cd],file["confirmed"][index_cd])
py.xticks(rotation=90)
py.legend()
py.title("Infected in State")
py.ylabel("No. of people")
py.xlabel("Canada states")
py.show()


# In[ ]:


py.plot(file1["province_state"][index_nt],file["confirmed"][index_nt])
py.xticks(rotation=90)
py.legend()
py.title("Infected in State")
py.ylabel("No. of people")
py.xlabel("Netherlands states")
py.show()


# In[ ]:


py.plot(file1["province_state"][index_as],file["confirmed"][index_as])
py.xticks(rotation=90)
py.legend()
py.title("Infected in State")
py.ylabel("No. of people")
py.xlabel("Australia states")
py.show()


# In[ ]:


py.plot(file1["province_state"][index_dm],file["confirmed"][index_dm])
py.xticks(rotation=90)
py.title("Infected in State")
py.ylabel("No. of people")
py.xlabel("Denmark states")
py.legend()
py.show()


# In[ ]:


py.plot(file1["province_state"][index_uk],file["confirmed"][index_uk])
py.xticks(rotation=90)
py.legend()
py.show()


# # By using IBM SPSS Modeler
# IBM SPSS Modeler is a data mining and text analytics software application from IBM. It is used to build predictive models and conduct other analytic tasks. It has a visual interface which allows users to leverage statistical and data mining algorithms without programming.

# In[ ]:


Image("../input/iimage/task_7_1.png")


# In[ ]:


Image("../input/iimage/task_7_2.png")


# In[ ]:


Image("../input/iimage/task_7_3.png")


# In[ ]:


Image("../input/iimage/task_7_4.png")


# In[ ]:


Image("../input/iimage/task_7_5.png")


# In[ ]:


Image("../input/iimage/task_7_6.png")


# In[ ]:


Image("../input/iimage/task_7_7.png")


# In[ ]:




