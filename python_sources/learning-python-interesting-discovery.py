#!/usr/bin/env python
# coding: utf-8

# This is my first Kernel!
# 
# Typically, I use C# for all of my programming needs, however, I have decided to learn Python to expand my tool belt. I am going through the free tutorial here on kaggle to learn python and the below exercise was implemented. 
# 
# 

# In[ ]:


from time import time, sleep

def time_call(fn, arg):
    """Return the amount of time the given function takes (in seconds) when called with the given argument.
    """
    t1 = time()
    fn(arg)
    t2 = time()
    return t2 - t1


# So, I expanded the above definition to test the inline code verses the function call. As shown when the second code is ran, something interesting results. The def call operation is faster than the inline. *Calling on all python gurus*: How is this the case. Like I said, I am in the discovery phase of python, so have I made and error or is a def call faster than the inline code?

# In[ ]:


def time_call(fn, arg):
    """Return the amount of time the given function takes (in seconds) when called with the given argument.
    """
    t1 = time()
    fn(arg)
    t2 = time()
    return t2 - t1

timeforfunction = time_call(sleep, 5)

t1 = time()
sleep(5)
t2 = time()
timetaken = t2 - t1

print("there is a loss of" , timeforfunction - timetaken, "seconds")

