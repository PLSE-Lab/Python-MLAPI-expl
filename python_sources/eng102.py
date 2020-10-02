#!/usr/bin/env python
# coding: utf-8

#    # What Why and How's of the Disceret Fourier Transform

# ### This is a notebook which will go through a Discrete Fourier Transform and explain how it can be useful and what how it functions. This is all done in a programming language called python. 

# In[1]:


import numpy as np
import numpy.fft as nf
import matplotlib.pyplot as plt


# This first section are just pachages of code used to help perform complex tasks, these packages are just prebuilt libraries that have functions that you can easily use, instead of having to make your own from scratch

# ### What is a Fourier Transform?

# Before we can talk about Fourier Transfomrs first its important to understand sin and cos functions. The plot bellow shows a waveform that has a period of 50 seconds, meaning it takes 50 seconds for the wave to make a complete cycle. The wave can also be described in terms of its frequency, or how many cycles it completes in seconds, this would be 1 cycle every 50 seconds 

# In[2]:


def waveform(n, p, color = "b"):
    t = np.linspace(0, n, n*20)
    plt.xlabel("seconds")
    plt.plot( t,np.sin(2*np.pi*(t/p)), color ) #creates a sine wave with a period of 1 cycle every 50 seconds


# In[3]:


color = ""
waveform(50, 50, color)
plt.show()


# In[4]:


waveform(100, 50, "g")
waveform(100, 25, "y--")
plt.legend(["freq = 50", "freq = 25"])
plt.show()


# Here are two waves that are shown over a time frame of 100 seconds, one completes a cycle every 50 seconds (green), thet other completes a cycle every 25 seconds (yellow). We see on the graph that green has 2 complete cycles and yellow has 4.
# 

# ### The problem

# In[11]:


def signal(numWaves):
    t = np.linspace(0,100,1000)
    wave = np.zeros(1000)
    for i in range(numWaves):
        wave += np.sin(t*np.random.randint(1,10)/10*(2*np.pi))
    return [t,wave]
        


# In[12]:


t,sgnl = signal(3)
plt.plot(t,sgnl)
plt.show()


# Now can you tell me what cose and sine wave makes up this? No? This signal is a composite signal made up of smaller waves, the goal is to find those smaller waves that make up this larger wave, but how can we do that? That's when the Discrete Fourier Transfrom comes to hand.
