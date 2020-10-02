#!/usr/bin/env python
# coding: utf-8

# This is a Kernel to call attention to a simple, but usually neglected 1st step in **any** data analysis:
# 
# **look up for well-known relations between different features of your data,  specially when it involves physical processes**.
# 
# In the past you would have to go to an university library, for this. Now, we have Google! 
# 
# In the present case, googling around you will find out that the number of meteorites impacts per year depends on the mass approximately as a power law. This is usually a good description for quantities that decay over many orders of magnitude. In a log-log plot it will look like a straight line. See the plot below.
# 
# Reference: [The impact rate on Earth, Philip A Bland 2005][1]
# 
# ![enter image description here][2]
# 
# Now, without even looking to the dataset in question, you can bet it will resemble some sort of power law, or maybe a broken power law. You can infer that the dataset will contain mostly light objects, thus the events will be scattered somewhere on the left side of the above plot.
# 
# 
#   [1]:  https://doi.org/10.1098/rsta.2005.1674
#   [2]: http://d29qn7q9z0j1p6.cloudfront.net/content/roypta/363/1837/2793/F2.large.jpg

# ## Preamble ##

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input/data"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Basic libraries
import numpy as np
import pandas as pd
from scipy import stats

# File related
import zipfile
from subprocess import check_output

# Regression and machine learning
import tensorflow as tf

# Plotting with matplotlib
import matplotlib
import matplotlib.pyplot as plt

from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import andrews_curves
from pandas.tools.plotting import radviz

plt.style.use('fivethirtyeight')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 14


# ## Read data ##

# In[ ]:


from subprocess import check_output
print(check_output(['ls', '../input/']).decode('utf8'))


# In[ ]:


meteorites = pd.read_csv('../input/meteorite-landings.csv')

meteorites.head()


# In[ ]:


rate_dict = {}

# Binning
mass_bins_edges = np.logspace(-1,
                              np.log10(meteorites['mass'].max()),
                              num=30, base=10.
                             )

for j in range(len(mass_bins_edges)-1):
    
    mass_bin_center = (mass_bins_edges[j+1] + mass_bins_edges[j]) / 2.
        
    rate_dict[mass_bin_center] = meteorites['mass'][
        (meteorites['mass'] >= mass_bins_edges[j]) &
        (meteorites['mass'] < mass_bins_edges[j+1])
        ].count()
    
rate = pd.Series(rate_dict)

rate


# In[ ]:


fig, axes = plt.subplots(figsize=(5.,6.))

plt.errorbar(
                rate.index.values,
                rate.values,
                yerr=np.sqrt(rate.values),
                fmt='o',
                ecolor='grey',
                capthick=0.5
                )

axes.set_xscale("log", nonposx='clip')
axes.set_yscale("log", nonposy='clip')

# Plot on the same scale as the plot shown in the introduction
axes.set_xlim([1.e-2, 1.e16])
axes.set_ylim([1.e-2, 2.e4])

axes.set_xlabel(r'$m$ (kg)')
axes.set_ylabel(r'$N$')

axes.set_title(r'Vertical bars are statistical errors: $\sigma=\sqrt{N}$', fontsize=14)

plt.show()
plt.close()


# There are two mass regimes, around m=10^2 kg, where the data behaves as a straight lines, on the log-log scale. Remember that a straight line in a log-log plot is a power law, i.e.
# 
# rate = q * m^(-p)

# ### Data analysis ###

# ### [1] Rescaling as a check ###

# A simple way of inferring the spectral index "b" of data following a power law of the form a*m^(-b) is to multiply it by m^(+b) to get a horizontal line on the log-log plot.

# In[ ]:


fig, axes = plt.subplots(ncols=2, sharex=True,  figsize=(10.,6.))

axes[0].errorbar(
                rate.index.values,
                rate.values * (rate.index.values ** 0.6),
                yerr=np.sqrt(rate.values) * (rate.index.values ** 0.6),
                fmt='o',
                ecolor='grey',
                capthick=0.5
                )

axes[0].set_xscale("log", nonposx='clip')
axes[0].set_yscale("log", nonposy='clip')

axes[0].set_xlabel(r'$m$ (kg)')
axes[0].set_ylabel(r'$N\times m^p$ (kg${}^{p}$)')

axes[0].set_title(r'$p=0.6$')

axes[1].errorbar(
                rate.index.values,
                rate.values * (rate.index.values ** (-0.6)),
                yerr=np.sqrt(rate.values) * (rate.index.values ** (-0.6)),
                fmt='o',
                ecolor='grey',
                capthick=0.5
                )

axes[1].set_xscale("log", nonposx='clip')
axes[1].set_yscale("log", nonposy='clip')

axes[1].set_xlabel(r'$m$ (kg)')

axes[1].set_title(r'$p=-0.6$')

plt.show()
plt.close()


# On the log-log scale, we can model the problem as a harmonic combination of two straight lines:
# 
# (1) y = -bx + c
# 
# (2) y = -dx + f
# 
# with 
# 
# y = log_10(N)
# 
# x = log_10(m)
# 
# In the linear scale, these are power laws:
# 
# (1') N = (10^c) * m^(-b)
# 
# (2') N = (10^f) * m^(-d)
# 
# The combination of straight lines becomes
# 
# y = [ 1/(-bx + c) + 1/(-dx + f) ]^(-1) = [ (-bx + c) * (-dx + f) ] / ( -bx + c - dx + f)
# 
# When x is small, y ~ (-dx + f). And, as we have seen, d should be close to -0.6.
# 
# When x is large, y ~ (-bx + c). And, as we have seen, b should be close to 0.6.

# **The lesson here is that we have been driven by previous knowledge**. We shouldn't be always in the dark when approaching difficult datasets, when people have already put a lot of thought on that.
# 
# Not thinking like a real data **scientist** may take you through dark paths such as using complicated machine learning to predict the movement of projectiles, which should be solved with high school Physics.

# ### [2] Data regression with TensorFlow ###

# The correct "cost function" for a regression of **number of rare events** is of the **Poisson** type.  Specially in a case where counts fall below 10, in some of the bins.
# 
# One begin writing a likelihood, L, for a Poisson distribution. The likelihood must be maximized in the regression. 
# 
# Equivalently, [-2*log(L)] must be minimized:
# 
# -2 * \log(L) = 2 * \sum_j [ N_{e,j} - N_{o,j} + N_{o,j} * log(N_{o,j} / N_{e,j}) ]
# 
# with 
# 
# N_e : expected number of events
# 
# N_o : observed number of events
# 
# **MUST READ Reference:**
# 
# [Clarification of the Use of Chi Square and Likelihood Functions in Fits to Histograms, Steve Baker, Robert D. Cousins 1983][1]
# 
# 
#   [1]: http://10.1016/0167-5087(84)90016-4
