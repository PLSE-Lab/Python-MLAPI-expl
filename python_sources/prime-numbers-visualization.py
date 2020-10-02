#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[ ]:


import matplotlib.pyplot as plt
import time


# > I watched the following YouTube channeland decided to prepare a Kernel here. I think it is really beautiful.
# The YouTube channel name **3Blue1Brown**
# https://www.youtube.com/watch?v=EK32jo7i5LQ&t=381s
# 
# ![Capture.PNG](attachment:Capture.PNG)

# # Generate Prime Number and plot funtion

# In[ ]:


def pattern_1(end):
    start1 = time.time()
    start = 1
    #end = 10000
    xs = []
    ys = []
    for val in range(start, end + 1): 
        if val > 1:
            for n in range(2, val): 
                if (val % n) == 0: 
                       break
            else:
                xs.append(val)
                ys.append(val)

    #xs = list(range(1, len(ys)))

    fig = plt.figure(figsize=(15, 15))
    
    for x, y in zip(xs, ys):
        plt.polar(x, y, 'b.')
    end1 = time.time()
    print('it was run for {} seconds'.format(end1-start1))


# # Run program for 100 iteratoins

# In[ ]:


pattern_1(100)


# # Run program for 1000 iteratoins

# In[ ]:


pattern_1(1000)


# # Run program for 10000 iteratoins

# In[ ]:


pattern_1(10000)


# # Run program for 100000 iteratoins

# In[ ]:


pattern_1(100000)


# # Run program for 1000000 iteratoins

# In[ ]:


pattern_1(1000000)


# # Try something different

# In[ ]:


start = 1
end = 15000
ys = []
for val in range(start, end + 1): 
    if val > 1:
        for n in range(2, val): 
            if (val % n) == 0: 
                   break
        else:
            ys.append(val)

xs = list(range(1, len(ys)))

fig = plt.figure(figsize=(10, 10))
for x, y in zip(xs, ys):
    plt.polar(x, y, 'r.')

