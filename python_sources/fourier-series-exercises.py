#!/usr/bin/env python
# coding: utf-8

# <h1>Fourier Series: An Exercise</h1>
# 
# I'm refamiliarizing myself with the properties of Fourier series and Fourier transforms to better understand how these tools can be used in data analysis.  Toward that end, I've been working through a variety of exercises and proofs, and wanted to memorialize this work for future reference.  I thought it might also be beneficial to incorporate some graphs to illustrate in places.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# <h2>Problem:</h2>
# 
# Expand the function \\(f(x) = x^3\\) in a Fourier cosine series on the interval \\( 0 \leq x \leq \pi \\).

# <h3><i>Answer:</i></h3>
# 
# We'll start with the calculation of the 0-th coefficient \\( a_0 \\) which is calculated using the equation
# 
# $$ a_0 = \frac{1}{\pi} \int_0^\pi x^3 dx = \frac{1}{\pi} \left( \frac{x^4}{4}\bigg|_0^\pi \right) = \frac{\pi^3}{4}$$
# 
# So the best horizontal line approximation for the function is given by \\( y = \frac{\pi^3}{4} \\).

# In[ ]:


x = pd.Series((np.arange(350)-175)/50*np.pi)
fx = abs(((x+np.pi) % (2*np.pi))-np.pi)**3
df = pd.DataFrame({'x': x, 'fx': fx})
df['k0'] = np.pi**3 / 4
df['c0'] = df['k0']

# Calculates l2 error for x, y, and an estimate e.
def l2_error(x, y, e):
    return sum(x.diff()[1:].reset_index(drop=True)
               .multiply(y[:-1].reset_index(drop=True)-e[:-1].reset_index(drop=True))**2)

print("L2 Error: " + str(l2_error(df['x'], df['fx'], df['c0'])))
df[['x', 'fx', 'k0']].plot.line(x='x')


# To calculate \\(a_k\\) for \\(k \ge 1\\), we need to solve
# 
# $$ a_k = \frac{2}{\pi} \int_0^\pi x^3 \cos(kx) dx $$
# 
# Using integration by parts, we get
# 
# $$ a_k = \frac{2}{\pi} \left( \frac{1}{k}x^3\sin(kx)\bigg|_0^\pi \right) - \frac{6}{\pi k} \int_0^\pi x^2 \sin(kx) dx$$
# 
# The term on the left evaluates to zero because \\(\sin(k\pi) = 0 \\) for all values of k.  Applying integration by parts a second time to the term on the right gets us
# 
# $$ a_k = -\frac{6}{\pi k} \left( -\frac{x^2}{k} \cos(kx) \bigg|_0^\pi \right) -\frac{12}{\pi k^2} \int_0^\pi x \cos(kx) dx $$
# 
# Note that \\( \cos(k \pi) \\) evaluates to 1 for even values of k and to -1 for odd values of k.  So for the term on the left, we have
# 
# $$ -\frac{6}{\pi k} \left( -\frac{x^2}{k} \cos(kx) \bigg|_0^\pi \right) = (-1)^k \frac{6 \pi}{k^2} $$
# 
# For the term on the right, we apply integration by parts one last time to get
# 
# $$-\frac{12}{\pi k^2} \int_0^\pi x \cos(kx) dx = -\frac{12x}{\pi k^3}\sin(kx)\bigg|_0^\pi +\frac{12}{\pi k^3} \int_0^\pi \sin(kx)dx $$
# 
# As above, the term on the left is identically zero for all values of k.  The term on the right evaluates to 
# 
# $$ -\frac{12}{\pi k^4} \cos(kx) \bigg|_0^\pi $$
# 
# As indicated earlier, \\( \cos(k\pi) \\) is 1 when k is even and is -1 when k is odd.  Thus we now have the full calculation for \\( a_k \\):
# 
# $$ a_k = (-1)^k \frac{6 \pi}{k^2} -\frac{12}{\pi k^4}\left( (-1)^k - 1 \right) $$

# In[ ]:


# Calculation of the Fourier coefficients for x^3
def get_x3_Fourier_Coefficient(k):
    return (-1)**k * 6.0 * np.pi / k**2 - 12.0 / (np.pi * k**4) * ((-1)**k - 1)


# Armed with this information, we can now demonstrate how to approximate our original function with successive refinements of the Fourier series.  We'll start with approximation with the constant \\(a_0\\) and the first series with \\(k=1\\).  The graph on the left shows the original function, the horizontal line, and the first Fourier series approximation individually.  The graph on the right combines \\(a_0\\) and function for \\(k=1\\).

# In[ ]:


df['k1'] = get_x3_Fourier_Coefficient(1) * np.cos(x)
df['c1'] = df['k0'] + df['k1']
print("L2 Error: " + str(l2_error(df['x'], df['fx'], df['c1'])))
fig1, ax = plt.subplots(1, 2, figsize=(14,8), sharex=True)
df[['x', 'fx', 'k0', 'k1']].plot.line(x='x', ax=ax[0])
df[['x', 'fx', 'k0', 'c1']].plot.line(x='x', ax=ax[1])


# Next, we add in the approximation for \\(k=2\\).

# In[ ]:


df['k2'] = get_x3_Fourier_Coefficient(2) * np.cos(2 * df['x'])
df['c2'] = df['c1'] + df['k2']
print("L2 Error: " + str(l2_error(df['x'], df['fx'], df['c2'])))
fig1, ax = plt.subplots(1, 2, figsize=(14,8), sharex=True)
df[['x', 'fx', 'k0', 'k1', 'k2']].plot.line(x='x', ax=ax[0])
df[['x', 'fx', 'k0', 'c1', 'c2']].plot.line(x='x', ax=ax[1])


# One more step, showing the inclusion of \\(k=3\\).

# In[ ]:


df['k3'] = get_x3_Fourier_Coefficient(3) * np.cos(3 * df['x'])
df['c3'] = df['c2'] + df['k3']
print("L2 Error: " + str(l2_error(df['x'], df['fx'], df['c3'])))
fig1, ax = plt.subplots(1, 2, figsize=(14,8), sharex=True)
df[['x', 'fx', 'k0', 'k1', 'k2', 'k3']].plot.line(x='x', ax=ax[0])
df[['x', 'fx', 'k0', 'c1', 'c2', 'c3']].plot.line(x='x', ax=ax[1])


# And now let's show the approximation when we include all of the approximations up to \\(k=32\\).  Aside from the neighborhoods around the peaks, the approximation is pretty close to indistinguishable from the original function.

# In[ ]:


df['c'] = df['k0']
for i in np.arange(32):
    df['c'] = df['c'] + get_x3_Fourier_Coefficient(i+1) * np.cos((i+1) * df['x'])
print("L2 Error: " + str(l2_error(df['x'], df['fx'], df['c'])))
fig2, ax2 = plt.subplots(1, 1, figsize=(14,8), sharex=True)
df[['x', 'fx', 'c']].plot.line(x='x', ax=ax2)

