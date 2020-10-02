#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


A = np.array([[4, 2],[2, 4],[2, 3],[3, 6],[4, 4]])
B = np.array([[9,10],[6, 8],[9, 5],[8, 7],[10,8]])


# In[ ]:


fig = plt.figure()
ax = fig.gca()

plt.plot(A[:,0], A[:,1], 'ro')
plt.plot(B[:,0], B[:,1], 'bo')
plt.grid()


# $ S_{within} = S_A + S_B  $
# 
# $ S_{between} = ( \mu_A - \mu_B )*( \mu_A - \mu_B )^t $

# In[ ]:


muA = np.mean(A, axis=0)
muB = np.mean(B, axis=0)

sA  = np.cov(A.transpose())
sB  = np.cov(B.transpose())

# Within class
Sw  = sA + sB

# Between class
mu = (muA - muB)
mu = mu.reshape((2, 1))
Sb  = np.dot(mu, mu.transpose())


# $ w^* = S_{within}^{-1} ( \mu_A - \mu_B ) $

# In[ ]:


wStar = np.matmul( np.linalg.pinv(Sw), mu )


# ** Plot Projected Points **

# In[ ]:


projA = np.dot(A, wStar)
projB = np.dot(B, wStar)

fig = plt.figure()
ax = fig.gca()

plt.plot(projA, np.zeros(5), 'ro')
plt.plot(projB, np.zeros(5), 'bo')
plt.grid()
x=plt.title("Projected Points")


# In[ ]:





# In[ ]:





# In[ ]:




