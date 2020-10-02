#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np, pandas as pd, datashader as ds
from datashader import transfer_functions as tf
from math import sin, cos, sqrt, fabs

@jit(nopython=True)
def Clifford(x, y, a, b, c, d, *o):
    return sin(a * y) + c * cos(a * x),            sin(b * x) + d * cos(b * y)


# In[ ]:


n=10000000

def trajectory_coords(fn, x0, y0, a, b=0, c=0, d=0, e=0, f=0, n=n):
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    for i in np.arange(n-1):
        x[i+1], y[i+1] = fn(x[i], y[i], a, b, c, d, e, f)
    return x,y

def trajectory(fn, x0, y0, a, b=0, c=0, d=0, e=0, f=0, n=n):
    x, y = trajectory_coords(fn, x0, y0, a, b, c, d, e, f, n)
    return pd.DataFrame(dict(x=x,y=y))


# In[ ]:


df = trajectory(Clifford, 0, 0, -1.3, -1.3, -1.8, -1.9)


# In[ ]:


df.tail()


# In[ ]:


cvs = ds.Canvas(plot_width = 700, plot_height = 700)
agg = cvs.points(df, 'x', 'y')
print(agg.values[190:195,190:195],"\n")


# In[ ]:


ds.transfer_functions.Image.border=0

tf.shade(agg, cmap = ["white", "black"])

