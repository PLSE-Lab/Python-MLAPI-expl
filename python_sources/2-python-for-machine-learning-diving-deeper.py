#!/usr/bin/env python
# coding: utf-8

# **Hi ! Welcome to my Kernel**
# 
# > **** This the second part of a series of kernels that wil familiarize you with the naunces of Machine Learning using Python
# 
# * I created this kernel so that beginners can use this to learn Python and Machine Learning.
# 
# * Others may just use this code as a refresher !!
# 
# **We will look at the following topics in this kernel**
# 
# 1. Pandas and Matplotlib Basics
# 2. Dictionaries & Pandas
# 3. Logic, Control Flow and Filtering
# 4. Loops
# 5. Applying these concepts for a real world data set
# 

# In[ ]:


# import pandas as pd 
import pandas as pd 
  
# Creating empty series 
ser = pd.Series() 
  
print(ser) 



# In[ ]:


# import pandas and numpy  
import pandas as pd  
import numpy as np  
    
# series with numpy linspace()   
series = pd.Series(np.linspace(1950, 2049, 100))
print(series)

    


# In[ ]:


years = [ i for i in range(1900,2100,4)]
print(years)


# In[ ]:


# generate random integer values
from random import seed
from random import randint
# seed random number generator
seed(1)
# generate some integers
for _ in range(10):
	value = randint(0, 10)
	print(value)


# In[ ]:


from random import seed
from random import randint
seed(2)
crime = [ randint(134587,9857630) for x in range(0,len(years),1)]

print(crime)


# In[ ]:


print(len(years))
print(len(crime))


# In[ ]:


len(years)==len(crime)


# In[ ]:



# Print the last item from year and pop
print(years[-1])
print(crime[-1])

get_ipython().run_line_magic('matplotlib', 'inline')
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Make a line plot: year on the x-axis, pop on the y-axis
plt.plot(years,pop)
plt.xlabel("Years")
plt.ylabel("Crime")
plt.title("Crime over Years")

# Display the plot with plt.show()
plt.show()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()


# In[ ]:


data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('Entry a')
plt.ylabel('Entry b')
plt.show()


# In[ ]:


names = ['Group A', 'Group B', 'Group C']
values = [1, 10, 100]

plt.figure(figsize=(18, 3))

plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
from numpy.random import rand
from numpy.random import randint
from numpy.random import seed

seed(3)


gdp_cap = [randint(223,4510) for i in range(0,100,1)]


life_exp = [randint(60,80) for i in range(0,100,1)]


# In[ ]:


len(gdp_cap)


# In[ ]:


len(life_exp)


# In[ ]:


# Print the last item of gdp_cap and life_exp

print(gdp_cap[-1])

print(life_exp[-1])



# In[ ]:


# Make a line plot, gdp_cap on the x-axis, life_exp on the y-axis
plt.plot(gdp_cap,life_exp)

# Display the plot
plt.show()

