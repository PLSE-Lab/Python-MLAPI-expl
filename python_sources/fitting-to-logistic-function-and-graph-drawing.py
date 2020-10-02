#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook uses scipy's optimize.curve_fit function to perform curve fitting.
# 
# Logistic function (Population function):
# $$ P(t) = \frac{K}{1 + \left(\frac{K-P_0}{P_0}\right) \exp{(-rt)}} $$
# is the solution of the Verhulst equation:
# $$ \frac{\mathrm{d} P(t)}{\mathrm{d} t} = r P(t) \cdot \left( 1 - \frac{P(t)}{K} \right), $$
# where \\( P_0 \\) is the initial population, the constant \\( r \\) defines the growth rate and \\( K \\) is the carrying capacity.
# You can see that there are some differences.

# # import package

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab as pl

from scipy.optimize import curve_fit


# # Prepare data

# In[ ]:


# csv read
df1 = pd.read_csv("/kaggle/input/west-african-ebola-virus-epidemic-timeline/cases_and_deaths.csv", delimiter=',')
df1.dataframeName = 'cases_and_deaths.csv'

# df -> list
x_values = df1['Days'].values.tolist() 
y_values = df1['Case'].values.tolist()


# # Fitting

# In[ ]:


# fitting functions
def f(t, K, P0, r):
    return  (K / (1 + ((K-P0)/P0)*np.exp(-r*t)))

# fitting
popt, pcov = curve_fit(f, x_values, y_values, p0=[1, 1, 0.5], maxfev=300000)
print(f"Fitting parameters")
print(f"K: {popt[0]}, P0: {popt[1]}, r: {popt[2]}")


# # Plot

# In[ ]:


# init main graph
fig = pl.figure(figsize=(16, 9))
ax = pl.axes()

# main graph captions
pl.suptitle("2014 Ebola epidemic in West Africa ", fontweight="bold")
pl.ylabel('Cases')
pl.xlabel('Days')

# main fitting plot
xx = np.linspace(0, x_values[-1], 100)
yy = f(xx, popt[0], popt[1], popt[2])
pl.xlim(x_values[0], x_values[-1])
pl.ylim(y_values[0], y_values[-1])

pl.plot(x_values, y_values,'o', label='Cases')
pl.plot(xx, yy, label="Logistic Function")
pl.legend(loc='lower right')

# Any results you write to the current directory are saved as output.
pl.savefig("graph.png")


# # Plot(log scale)

# In[ ]:


# init main graph
fig = pl.figure(figsize=(16, 9))
ax = pl.axes()

# main graph captions
pl.suptitle("2014 Ebola epidemic in West Africa (log scale)", fontweight="bold")
pl.ylabel('Cases')
pl.xlabel('Days')

pl.yscale('Log')
pl.locator_params(axis='x',tight=True, nbins=5)
pl.plot(x_values, y_values,'o', label='Cases')
pl.plot(xx, yy, label="Logistic Function")
pl.legend(loc='lower right')

# Any results you write to the current directory are saved as output.
pl.savefig("graph_log.png")

