#!/usr/bin/env python
# coding: utf-8

# # Which patient populations pass away from COVID-19?

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


df1 = pd.read_csv('../input/filles/novel_corona.csv')


# In[ ]:


y = df1[['age','recovered','death']]
y


# In[ ]:


f= y.groupby(['age']).count()
f.drop(f.index[0:9],0,inplace=True)
f


# In[ ]:


novel =pd.DataFrame(f).reset_index()
type(novel)


# In[ ]:


novel.plot(x='age',y='death')
py.show()


# In[ ]:


novel.plot(x='age',y='death',kind='box')
py.show()


# In[ ]:


df = pd.read_csv('../input/data-t/covid-19-tracker-canada.csv')
gf = df.drop(columns=['date','travel_history','id','source','province','city'])
x = gf.groupby(['age'])['confirmed_presumptive'].count()
x = gf.groupby(['age']).count()
x.drop(x.index[9:24],0,inplace=True)
b =pd.DataFrame(x).reset_index()
b.drop(b.index[9:24],0,inplace=True)


# In[ ]:


new = pd.read_csv('../input/deaths/novel_deaths.csv')
new['confirmed_cases'] = b['confirmed_presumptive']
# new.drop(b.index[:-1],0,inplace=True)
new = new[:-1]
new['Deaths'][0] = 0.40
new


# In[ ]:


new.plot(x='Age',y=['Deaths','confirmed_cases'],kind='bar')
py.show()


# # By using IBM SPSS Modeler
# IBM SPSS Modeler is a data mining and text analytics software application from IBM. It is used to build predictive models and conduct other analytic tasks. It has a visual interface which allows users to leverage statistical and data mining algorithms without programming.

# In[ ]:


Image("../input/all-data/T_1.png")


# In[ ]:


Image("../input/all-data/T_2.png")


# In[ ]:


Image("../input/all-data/T_3.png")


# In[ ]:


Image("../input/all-data/T_4.png")


# In[ ]:


Image("../input/all-data/T_5.png")


# In[ ]:


Image("../input/all-data/TD_1.png")


# In[ ]:


Image("../input/all-data/TD_2.png")


# In[ ]:


Image("../input/all-data/TD_3.png")


# In[ ]:


Image("../input/all-data/TD_4.png")


# In[ ]:


Image("../input/all-data/TD_5.png")


# In[ ]:


Image("../input/all-data/TD_6.png")


# In[ ]:




