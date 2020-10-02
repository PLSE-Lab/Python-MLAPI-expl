#!/usr/bin/env python
# coding: utf-8

# This notebook demonstrates what happens to the standard deviation of the weight of marbles in a bag with a changing distribution of heavy/light marbles. The bag contains a combination of heavy and light marbles, initially containing mostly heavy marbles (say 10) with fewer light marbles (say one).
# 
# ![Marbles](https://www.eduplace.com/math/mw/background/4/09/graphics/ts_4_9_wi-5.gif)
# 
# What happens to the standard deviation of the weights as we add one more light marble at a time? When will the standard deviation be at a maximum, when at a minimum?
# 
# What happens if you change the relative weights, or switch the roles of heavy/light marbles?

# In[87]:


# Library used for standard deviation calculation
import statistics

# Relative weight of different marbles
marbleWeight1 = 1
marbleWeight2 = 2

# Initial number of each marble type in the bag
numberMarbles1 = 1
numberMarbles2 = 10

# Initiate an empty list for the bag
marbleBag = []

# then add the appropriate number of each marble type
for i in range(numberMarbles1):
    marbleBag.append(marbleWeight1)

for i in range(numberMarbles2):
    marbleBag.append(marbleWeight2)


# Now we have created our initial bag, we can take a look at it (it is being represented as a Python list).

# In[88]:


import collections

print("Bag: ",marbleBag)
print("Counts: ",collections.Counter(marbleBag))


# And get some sample statistics for the weight distribution. Note, we are using the population standard deviation, since the marble bag is not a sample.

# In[89]:


# Rounding statistics to 4 decimal places
print("Mean: ",round(statistics.mean(marbleBag),4))
print("Median: ",statistics.median(marbleBag))
print("Mode: ",statistics.mode(marbleBag))
print("Std dev: ",round(statistics.pstdev(marbleBag),4))


# Next we will add more of the type 1 (light) marbles to the bag and record the standard deviation of the weight distribution at each step.

# In[90]:


# We create a copy of the initial marble bag first
# allowing this section of code to be re-run and keeping the initial bag conditions
marbleBagUpdated = list(marbleBag)

# Record the initial std dev of the weight distribution
stdevBag = [statistics.pstdev(marbleBagUpdated)]

# Define how many marbles to track
numMarblesToAdd = 100

# Then add one at a time and record the new std dev of the weight distribution
for i in range(numMarblesToAdd):
    marbleBagUpdated.append(marbleWeight1)
    stdevBag.append(statistics.pstdev(marbleBagUpdated))


# We can check the bag again to see the end state.

# In[91]:


print("Bag: ",marbleBagUpdated)
print("Counts: ",collections.Counter(marbleBagUpdated))


# Now let's plot the results and find the min/max values and positions.

# In[92]:


# Libraries used for plotting and finding the position of the min/max standard deviation
import matplotlib.pyplot as plt
import numpy as np

# Create the plot
plt.xlabel('Number of type 1 marbles in the bag')
plt.ylabel('Std dev of weight distribution')
plt.plot(stdevBag)
plt.show()

# Get the min/max statistics
# (note Python lists are indexed from zero, so 0 = 1 marble etc, hence the +1 below)
print("Max Std dev: ",round(max(stdevBag),4))
print("Max Std dev position: ",np.argmax(stdevBag)+1)
print("Min Std dev: ",round(min(stdevBag),4))
print("Min Std dev position: ",np.argmin(stdevBag)+1)


# **Conclusion**
# 
# The marble distribution is initially quite homogenoues, with mostly heavy marbles. Adding more light marbles makes the bag less homogeneous and so increases the standard deviation of the weight. When the number of light marbles equals the number of heavy marbles (ten each) the standard deviation is at a maximum. Adding more light marbles then makes the bag more homogeneous, in favour of light marbles, and so reduces the standard deviation.
# 
# Switching the role of light/heavy marbles should yield the same pattern. This can be tested by reversing the values of marbleWeight1 and marbleWeight2 in the first code snippet.
