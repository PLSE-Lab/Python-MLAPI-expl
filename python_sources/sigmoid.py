#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy


# In[ ]:


def sigmoid(z):
    return 1.0 / (1.0 + numpy.exp(-z))


# In[ ]:


z = numpy.linspace(-10, 10)

y = sigmoid(z)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as pyplot

fig1 = pyplot.figure()
axes1 = pyplot.axes()
axes1.set_xlabel('z')
axes1.set_ylabel('y')
plot1 = axes1.plot(z, y, label='y = sigmoid(z)')
legend1 = axes1.legend()


# In[ ]:


z_small = numpy.linspace(-1, 1)
y_small = 0.25 * z_small + 0.5


# In[ ]:


axes1.plot(z_small, 0.25 * z_small + 0.5, label='y = 0.25z + 0.5')
axes1.legend()
fig1


# In[ ]:


y_step = sigmoid(100 * z)
axes1.plot(z, y_step, label='y = sigmoid(100 * z)')
axes1.legend()
fig1


# In[ ]:





# In[ ]:




