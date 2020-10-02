#!/usr/bin/env python
# coding: utf-8

# # Generating random numbers with arbitrary distribution
# 
# <a href=http://www.inp.nsk.su/~petrenko/>A. Petrenko</a> (Novosibirsk, 2020)

# In[ ]:


import numpy as np
import holoviews as hv

hv.extension('matplotlib')


# Suppose we would like to generate random numbers from some finite interval with a specific distribution. For example,
# $$
# -5 < x < +5,\\
# f(x) = e^{-x/4}\cos^2 x
# $$

# In[ ]:


x0 = -5; x1 = +5
x = np.linspace(x0,x1,100)


# In[ ]:


def f(x):
    return np.exp(-x/4)*np.cos(x)*np.cos(x)


# In[ ]:


curve = hv.Curve( (x,f(x)) )


# In[ ]:


curve


# We just need to randomly fill the whole x-y plane with dots and then select only the ones below the curve:

# In[ ]:


N = 5000 # number of random dots in x-y plane
x = np.random.uniform(low=x0, high=x1, size=N)
y = np.random.uniform(low=0, high=np.max(f(x)), size=N)


# In[ ]:


get_ipython().run_line_magic('opts', 'Scatter (alpha=0.5 s=5)')

dots = hv.Scatter( (x,y) )


# In[ ]:


dots*curve


# Now selecting the $x$ values distributed according to the given distribution:

# In[ ]:


X = x[y < f(x)]


# In[ ]:


Y = y[y < f(x)]


# In[ ]:


selected_dots = hv.Scatter( (X,Y) )


# In[ ]:


dots*selected_dots*curve


# Resulting array X has values distributed according to the function f(x). The length of this array is less than the total number of dots N:

# In[ ]:


len(X)


# In[ ]:


print("The resulting efficiency (the yield of useful values) is %.1f %%" % (100*len(X)/N) )


# If one needs a specific number of random values (Np) it's enough to do the same procedure twice: once with some small number of dots to estimate the yield of useful values and then we can modify the N to produce the required number of useful values (with some safety margin).

# In[ ]:


efficiency = len(X)/N


# Suppose we need Np of resulting values:

# In[ ]:


Np = 10000


# The required N should be somewhat larger than Np/efficiency

# In[ ]:


N = int(1.1*(Np/efficiency))


# In[ ]:


N


# Let's repeat the same procedure again:

# In[ ]:


x = np.random.uniform(low=x0, high=x1, size=N)
y = np.random.uniform(low=0, high=np.max(f(x)), size=N)


# In[ ]:


X = x[y < f(x)]


# Now the number of values in the resulting array X should be slightly larger than Np:

# In[ ]:


len(X)


# So we can simply trim the resulting array to the required length:

# In[ ]:


X = X[0:Np]


# In[ ]:


len(X)


# ---

# In[ ]:


Y = y[y < f(x)]
Y = Y[0:Np]


# In[ ]:


hv.Scatter( (X,Y) )


# In[ ]:





# In[ ]:




