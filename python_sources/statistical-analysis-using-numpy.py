#!/usr/bin/env python
# coding: utf-8

# # We all know very much about statistics. Let's just recall the important Descriptive Measures using NumPy: 
# * 1) Measures of Central Tendency:-
#              1.1) Mean. 
#              1.2) Median. (Quantitative Average)
#              1.3) Mode.
# * 2) Measures of Dispersion:-
#             1.1) Dispersion. 
#             1.2) Standard Deviation.
#             1.3) Variance.

# # 1) Measures of Central Tendency:-

# ## >Importing NumPy and creating a random dataset using np.random.normal:

# In[121]:


import numpy as np
''' Generating a random dataset which have been assigned to a memory location 
with an identifier as employment
Centered around 40,000 having a normal distribution and
Standard Deviation of 10,000, 
with 20,000 data points'''

employment = np.random.normal(40000, 10000, 20000)



# ## >Visualizing the data using Matplotlib: 

# In[122]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.title('Normal Distribution')
plt.hist(employment, 60,density=True, facecolor='black', alpha=0.9)
plt.show()


# ## >Mean:

# In[123]:


np.mean(employment)


# ## >Median:

# > The median of the incomes dataset which is pretty much an even distribution therefore the median ~ 40,000.

# In[124]:


np.median(employment)


# ## >Adding an outlier value "99999999" to understand Mean Vs Median effect on the dataset

# In[125]:


n_employment = np.append(employment, [99999999])
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.title('Normal distribution with an Outlier')
plt.hist(n_employment, 100,density=True, facecolor='black', alpha=0.9)

plt.show()


# In[126]:



print(r"Before adding an outlier Mean: {}".format(np.mean(employment)))
print(r"After adding an outlier Mean: {}".format(np.mean(n_employment)))

print("___________________________________________________________________")

print(r"Before adding an outlier Median: {}".format(np.median(employment)))
print(r"After adding an outlier Median: {}".format(np.median(n_employment)))


# ## Woah! Seams MEAN is so mean while MEDIAN is decent!
# 
# ## By this we can conclude:-
#     1. Median is less prone to outliers than Mean.
#     2. Median well represents the data points.
#     
#    Let's continue to calculate Mode.

# ## >Mode:

# The below line of code creates a random dataset using [np.random.randint](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randint.html#numpy-random-randint) method:

# In[127]:


persons = np.random.randint(10, high=100, size=500) 
#Lowest (signed) integer=10 & Highest (signed) integer =100
persons


# In[128]:


from scipy import stats
stats.mode(persons) # To Calculate Mode


# ## In output 1st array tells us which values is the mode say [12]  & second array tells us how many times it have occured say [15] times.
# 
# To be continued...
