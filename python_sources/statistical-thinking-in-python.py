#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# In[ ]:


iris_data = pd.read_csv('../input/iris.data.csv')


# **Graphical exploratory data analysis**

# In[ ]:


iris_data.head()
iris_data.columns


# Creating Histograms

# In[ ]:


sns.set()
plt.hist(iris_data['5.1'])
plt.xlabel('petal length (cm)')
plt.ylabel('count')
plt.show()


# Adjust number of bins in a histogram

# In[ ]:


n_data = len(iris_data['5.1'])
n_bins = np.sqrt(n_data)
n_bins = int(n_bins)
plt.hist(iris_data['5.1'], bins=n_bins)
plt.xlabel('petal length (cm)')
plt.ylabel('count')
plt.show()


# Bee swarm plots
# 
# Advantage of Bee plot over histogram: there is no **binning bias** 
# As users are bound to interpret histogram graphs differently with different sizes of bins
# To remediate from this bias, Bee swarm plot can be used.

# In[ ]:


_ = sns.swarmplot(x='Iris-setosa',y='5.1',data=iris_data)
_ = plt.xlabel('')
_ = plt.ylabel('')
plt.show()


# Plotting all of your data: Empirical cumulative distribution functions (ECDF)
#        
#        x axis data needs to be sorted in order to create ECDF plot for that we would use np.sort to sort the data
#        y axis data needs to be evenly spaced data points with maximum value 1, this can be generated using np.arange() function and dividing it with total number of data points
#      

# In[ ]:


x = np.sort(iris_data['5.1'])
y = np.arange(1,len(x)+1)/len(x)
_=plt.plot(x,y,marker='.',linestyle='none')
_=plt.xlabel('')
_=plt.ylabel('ECDF')
plt.margins(0.2)
plt.show()


# Creating a function to create ECDF

# In[ ]:


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y


# Quantitative data analysis
# 
# np.mean()
# np.median()
# np.percentile(df,[25,50,75])
# 

# In[ ]:


percentiles = np.array([2.5,25,50,75,97.5])
ptiles_vers = np.percentile(iris_data['5.1'],percentiles)
print(ptiles_vers)


# In[ ]:


# Plot the ECDF
_ = plt.plot(x, y, '.')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Overlay percentiles as red diamonds.
_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',
         linestyle='none')
# Show the plot
plt.show()


# In[ ]:


# Create box plot with Seaborn's default settings

sns.boxplot(x='Iris-setosa',y='5.1',data = iris_data)
# Label the axes

plt.xlabel('species')
plt.ylabel('petal length (cm)')

# Show the plot
plt.show()


# Variance and Standard Deviation
# 
# * np.var()
# * np.std()
# 
# Covariance and perason correlation cofficient
# 
# * np.cov()
# * np.corrcoef()
# 

# **Thinking Probalistically**
# 
# Probalisitic logic and statistical interference
# 
# Statistical inference involves taking your data to probabilistic conclusions about what you would expect if you took even more data, and you can make decisions based on these conclusions.
# 
# 
# * np.random.random() 
# * np.random.seed()

# In[ ]:


np.random.seed(42)
# draw 6 random numbers
random_numbers = np.random.random(size=6)


# Hacker statistics uses simulated repeated measurements to compute probabilities
# 
# A bank made 100 mortgage loans. It is possible that anywhere between 0 and 100 of the loans will be defaulted upon. You would like to know the probability of getting a given number of defaults, given that the probability of a default is p = 0.05. To investigate this, you will do a simulation. You will perform 100 Bernoulli trials using the **perform_bernoulli_trials() ** function 

# In[ ]:


def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0


    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()

        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success += 1

    return n_success


# In[ ]:


# Seed random number generator

np.random.seed(42)
# Initialize the number of defaults: n_defaults
n_defaults = np.empty(1000)

# Compute the number of defaults
for i in range(1000):
    n_defaults[i] = perform_bernoulli_trials(100,.05)


# Plot the histogram with default number of bins; label your axes
_ = plt.hist(n_defaults, normed=True)
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')

# Show the plot
plt.show()


# In[ ]:


np.random.binomial(4,.5)
np.random.binomial(4,.5,size=10)


# **Poisson Distribution **
# 
# occurance of event is independent of previous event 
# 
# The number of r of arrivals of a Poisson process in a given time interval with average rate of lambda arrivals per interval is poisson distributed
# 
# When we have rare events (low p, high n), the Binomial distribution is Poisson. This has a single parameter, the mean number of successes per time interval, in our case the mean number of no-hitters per season.
# 

# In[ ]:


np.random.poisson(6,size=100)


# Probability Density Function for continuous quantity 
# Mathematical description of the relative likelihood of observing a value of continuous variable
# 

# normal distribution : np.random.normal(mean,std,size=)

# In[ ]:


# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10

samples_std1 = np.random.normal(20,1,size=100000)
samples_std3 = np.random.normal(20,3,size=100000)
samples_std10 = np.random.normal(20,10,size=100000)

# Make histograms
_=plt.hist(samples_std1,bins=100,normed=True,histtype='step')
_=plt.hist(samples_std3,bins=100,normed=True,histtype='step')
_=plt.hist(samples_std10,bins=100,normed=True,histtype='step')
# Make a legend, set limits and show plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.show()


# In[ ]:


# Generate CDFs
x_std1,y_std1=ecdf(samples_std1)
x_std3,y_std3=ecdf(samples_std3)
x_std10,y_std10=ecdf(samples_std10)

# Plot CDFs
_=plt.plot(x_std1,y_std1,marker='.',linestyle='none')
_=plt.plot(x_std3,y_std3,marker='.',linestyle='none')
_=plt.plot(x_std10,y_std10,marker='.',linestyle='none')
# Make a legend and show the plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
plt.show()


# Exponential distribution
# mean = np.mean(sample)
# np.random.exponential(mean,size=)
# x,y=ecdf(sample)
# 

# In[ ]:





# In[ ]:




