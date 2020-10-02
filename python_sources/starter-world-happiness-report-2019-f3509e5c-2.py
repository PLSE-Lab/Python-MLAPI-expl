#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as g; import numpy as n; import os; import pandas as p


# In[ ]:


feliz = p.read_csv('/kaggle/input/world-happiness-report-2019.csv', delimiter=',')
feliz.dataframeName = 'reporte-felicidad-mundial-2019'
obs, vars = feliz.shape
print(f'Existen {obs} observaciones y {vars} variables')


# In[ ]:


feliz.head(15)

