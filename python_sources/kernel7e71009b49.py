#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[ ]:


x = np.array([0,113,236,354,485,622,732,847,949,1054,1162]).reshape(-1,1)
y = np.array([0,5.59,11.59,15.51,23.45,30.24,35.58,41.11,46.02,50.76,55.58]).reshape(-1,1)


# In[ ]:


reg = LinearRegression()


# In[ ]:


reg.fit(x,y)


# In[ ]:


m = reg.coef_
print('Slope : ',m[0][0])


# In[ ]:


c = reg.intercept_
print('Intercept : ',c[0])


# In[ ]:


plt.scatter(x,y,label='points')
plt.plot(x,reg.predict(x),color='red',label='line')
plt.xlabel('Magnetic Field')
plt.ylabel(r'$\Delta$R')
plt.title('Hall Effect',fontweight="bold")
#plt.legend()
plt.show()


# In[ ]:




