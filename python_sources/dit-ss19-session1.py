#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# importing the math library
import math as m

# plot lib
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.


# In[2]:


2+1


# In[3]:


3+4


# In[4]:


6+6


# In[5]:


number = 0
number


# In[6]:


number = 5
print(number)

# Add 2 to the current value of number
number = number + 2
print(number)

#number = 0

if number > 0:
    print("Number greater than 0")
    
print("always printed")

number


# In[7]:


print(number)


# In[8]:


amount = 4

text_info = "Hello " * amount
print(text_info)


# In[9]:


number_b = 5

type(number_b)


# In[10]:


number_b = 19.666
type(number_b)


# In[11]:


print(1+2)

print(8-3)

print(3*4)

## true division ##
print(8/2)
print(8/3)

## floor division ##
print(8//2) # 4
print(8//3) # 2


# In[12]:


8 - 3 +2


# **PEMDAS - Parentheses, Exponents, Multiplication/Division, Addition/Subtraction.**

# In[13]:


-3 + 4 * 2


# In[14]:


(-3 + 4) * 4


# In[15]:


hat_height_cm = 25
my_height_cm = 206

total_height_meters = hat_height_cm + my_height_cm / 100
print("Height in meters = ", total_height_meters, "?")


# In[16]:


total_height_meters = (hat_height_cm + my_height_cm) / 100
print("Height in meters = ", total_height_meters)


# In[17]:


#minimum
print(min(1,2,3))

#maximum
print(max(1,2,3))


# **c - copy
# v - paste
# x - cut**

# In[18]:


#minimum
print("minimum of 1,2,3 = ", min(1,2,3))

#maximum
print("maximum of 1,2,3 = ", max(1,2,3))


# In[19]:


print(abs(32))
print(abs(-32))


# In[20]:


print(float(10))


# In[21]:


print(int(3.33))


# In[22]:


print(int("345") + 3)


# In[23]:


text = "123"

print (int(text) * 2)


# In[24]:


#text = "123a"

print (int(text) * 2)


# In[25]:


5 % 2


# **modulus of 2 can only be 1 or 0**

# In[26]:


if 3 % 2 == 0:
    print("even")
else:
    print("odd")
    
    


# In[27]:


number = 3
if number % 2 == 0:
    print(number, " is even")
else:
    print(number, " is odd")
    
    


# In[28]:


for i in range(10):
    print(i+1)


# In[29]:


for i in range(10):
    if (i % 2 == 0):
        print(i, " is even")
    else:
        print(i, " is odd")


# In[30]:


math.pi


# In[ ]:


number = 3.56
math.ceil(number)


# In[ ]:


math.floor(number)


# In[ ]:


math.sqrt(number)


# In[ ]:


math.pow(3, 4)


# In[ ]:


m.pi


# In[ ]:





# In[ ]:


plt.plot([1,2,3,4])


# In[ ]:


plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.xlabel('numbers')


# In[ ]:


plt.plot([1, 2, 3, 4], [1, 4, 9, 16])


# In[ ]:


plt.plot([1,2,3,4], [1,4,9,16], 'go')
plt.axis([0, 6, -10, 20])


# [plot](https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot)

# In[ ]:


np.arange(0., 5., 0.2)


# In[ ]:


t = np.arange(0., 5., 0.2)

plt.plot(t, t, 'r--', t, t**2, 'bs')


# In[ ]:


t = np.arange(0., 5., 0.2)

plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')


# In[ ]:


t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

plt.figure(1)
plt.subplot(211) # 2, 1, 1 - numrows, numcols, fignum
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212) # 2, 1, 2 - numrows, numcols, fignum
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')


# In[ ]:


t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

plt.figure(1)
plt.subplot(211) # 2, 1, 1 - numrows, numcols, fignum
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212) # 2, 1, 2 - numrows, numcols, fignum
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')


# In[ ]:


plt.figure(1)
plt.subplot(211)
plt.plot([1,2,3])

plt.subplot(2,1,2)
plt.plot([4,5,6])

plt.figure(2)
plt.plot([4,5,6])

plt.title('Easy as 1, 2, 3') 


# In[ ]:


plt.figure(1)
plt.subplot(211)
plt.plot([1,2,3])
plt.title('Easy as 1, 2, 3') 

plt.subplot(2,1,2)
plt.plot([4,5,6])

plt.figure(2)
plt.plot([4,5,6])

#plt.figure(1)
#plt.subplot(211)
#plt.title('Easy as 1, 2, 3') 


# In[ ]:


mu = 100
sigma = 15

x = mu + sigma * np.random.randn(10000)

plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)

plt.xlabel("Smarts")
plt.ylabel('Probability')
plt.title('Histogram Test')
plt.text(120, .025, r'$\mu=100,\ \sigma=15$')
plt.grid(True)

plt.show()


# In[36]:


df = pd.read_csv("../input/test_Y3wMUE5_7gLdaTN.txt")

# head first 10 rows of file
df.head(10)

# tail - last 10 rows of file
#df.tail(10)


# In[37]:


df.describe()


# In[38]:


df['Property_Area'].value_counts()

