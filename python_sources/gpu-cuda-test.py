#!/usr/bin/env python
# coding: utf-8

#  **Compare speed of GPU vs CPU**
#  
#  In this example, we create array of 100 Million of random number and compute the power of it and printlout the duration taken/
#  
# >  Please enable GPU before run this program

# In[13]:


# Please enable GPU before run this program


import numpy as np
from timeit import default_timer as timer
from numba import vectorize

@vectorize(['float32(float32, float32)'], target='cuda')
def pow(a, b):
    return a ** b

def main():
    vec_size = 100000000

    a = b = np.array(np.random.sample(vec_size), dtype=np.float32)
    c = np.zeros(vec_size, dtype=np.float32)

    start = timer()

    c = pow(a, b)

    duration = timer() - start

    print("duration taken :" , duration)

if __name__ == '__main__':
    main()


# In[5]:


# Same program run on CPU

import numpy as np
from timeit import default_timer as timer

def pow(a, b, c):
    for i in range(a.size):
         c[i] = a[i] ** b[i]

def main():
    vec_size = 100000000

    a = b = np.array(np.random.sample(vec_size), dtype=np.float32)
    c = np.zeros(vec_size, dtype=np.float32)

    start = timer()
    pow(a, b, c)
    duration = timer() - start

    print("duration taken :" , duration)

if __name__ == '__main__':
    main()

