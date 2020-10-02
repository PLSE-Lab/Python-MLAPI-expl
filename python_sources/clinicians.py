#!/usr/bin/env python
# coding: utf-8

# # Which populations of clinicians are most likely to contract COVID-19?

# Task Details
# The Roche Data Science Coalition is a group of like-minded public and private organizations with a common mission and vision to bring actionable intelligence to patients, frontline healthcare providers, institutions, supply chains, and government. The tasks associated with this dataset were developed and evaluated by global frontline healthcare providers, hospitals, suppliers, and policy makers. They represent key research questions where insights developed by the Kaggle community can be most impactful in the areas of at-risk population evaluation and capacity management.

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


df4 = pd.read_csv('../input/d-clinic/doctor_corona.csv')
df4.columns = ['state','no.of_staffe']
df4 = df4[:-2]


# In[ ]:


df4.plot(x='state',y='no.of_staffe',figsize=(20,4),kind='bar',title='No. Of Doctors')


# In[ ]:


city_wise = pd.read_csv('../input/d-clinic/city_wise_corona.csv')
city_wise.columns=['City','No.OfStafee']
city_wise = city_wise[:-1]
city_wise


# In[ ]:


city_wise.plot(x='City',y='No.OfStafee',kind='line')
py.show()


# # By using IBM SPSS Modeler
# IBM SPSS Modeler is a data mining and text analytics software application from IBM. It is used to build predictive models and conduct other analytic tasks. It has a visual interface which allows users to leverage statistical and data mining algorithms without programming.

# In[ ]:


from PIL import Image

img = Image.open("../input/ibm-img/Task_9_1.png")
img


# In[ ]:


img = Image.open("../input/ibm-img/Task_9_2.png")
img


# In[ ]:


img = Image.open("../input/ibm-img/Task_9_3.png")
img


# In[ ]:


img = Image.open("../input/ibm-img/Task_9_4.png")
img


# In[ ]:


img = Image.open("../input/ibm-img/Task_9_5.png")
img


# In[ ]:


img = Image.open("../input/ibm-img/Task_9_6.png")
img


# In[ ]:




