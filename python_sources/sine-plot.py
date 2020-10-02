#!/usr/bin/env python
# coding: utf-8

# > Plotting Sine Function in Matplotlib

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


y = np.arange(-3.14, 3.14, 0.1)
x = np.sin(y)
plt.plot(y,x)
plt.title('Sine Wave')
plt.xlabel('time')
plt.ylabel('sin function')
plt.show()

