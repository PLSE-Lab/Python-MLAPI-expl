#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


x=np.arange(0,10)
y=np.arange(0,10)


# # **Scatter Plot**

# In[ ]:


plt.scatter(x,y, c='g')


# # **Adding Labels and Title**

# In[ ]:


plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Scatter Plot')
plt.scatter(x,y,c='b')


# **Saving the graph as png**

# In[ ]:


plt.savefig('filename.png')


# # **PLOT**

# In[ ]:


y = x*x
plt.plot(x,y)


# **Line Formatting**
# 

# In[ ]:


plt.plot(x,y, 'r--')
plt.plot(x,y, 'r*')


# # **SUBPLOT** 
# Many plots within a given area 
# 
# **Parameters:** No.of Rows, No.of cols, position
# 

# In[ ]:


plt.subplot(2,2, 1)
plt.plot(x,y,'r--')
plt.subplot(2,2, 2)
plt.plot(x,y, 'g')
plt.subplot(2,2,3)
plt.plot(x,y,'bo')


# **Sine and Cos waves**

# In[ ]:


x=np.arange(0, 5*np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.subplot(2,1,1)
plt.tight_layout(pad=3.0) 
plt.plot(x,y_sin)
plt.title('Sine')
plt.subplot(2,1,2)
plt.plot(x,y_cos)
plt.title('Cos')


# # **BAR PLOT**

# In[ ]:


x=[2,9,10]
y=[11,16,9]
x1=[3,9,11]
y1=[6,15,7]
plt.bar(x,y)
plt.bar(x1,y1, color='g')


# # **HISTOGRAM**

# In[ ]:


a=np.array([22,12,23,34,56,67,54,34,32,12])
plt.hist(a)
plt.title('histogram')


# # **PIE CHART**

# In[ ]:


labels='Python', 'C++', 'Ruby','Java'
sizes = [215, 120, 245, 210]
colors = ['gold', 'yellowgreen', 'lightcoral','lightskyblue']
explode=(0.2,0,0,0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, shadow=True, autopct='%1.1f%%')

