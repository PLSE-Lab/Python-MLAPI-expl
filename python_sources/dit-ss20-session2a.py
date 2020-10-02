#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


2+1


# In[ ]:


6+6


# In[ ]:


9-4


# In[ ]:


number = 0
number


# In[ ]:


number = 43
number


# In[ ]:


number = 5
print(number)

# Add 2 to 5 and print the value of number
number = 5 + 2
print(number)



# In[ ]:


number


# In[ ]:


number = 5

if number >= 5:
    print("Number is greater than 5")
    
print("always printed")


# In[ ]:


amount = 15

text_info = "hello" * amount

print(text_info)


# In[ ]:


number_int = 5

type(number_int)


# In[ ]:


number_float = 19.666

type(number_float)


# In[ ]:


print(1+2)

print(8-3)

print (3*4)

#division
print(8/2)
print(8/3)

print(8//3)


# In[ ]:


8-3+2


# In[ ]:


-3 + 4 * 2


# **PEMDAS - Parentheses, Exponents, Multiplication/Division, Addition/Subtraction.**

# In[ ]:


(-3 + 4) * 4


# In[ ]:


hat_height_cm = 25
my_height_cm = 206

total_height_meters = (hat_height_cm + my_height_cm) / 100

print("Height in meters = ", total_height_meters, "!")


# In[ ]:


# minimum
print(min(1,2,3))


# In[ ]:


#maximum
print(max(4,5,6))


# In[ ]:


print("minimum of 1,2,3 is ", min(1,2,3))


# In[ ]:


print(abs(32))
print(abs(-32))


# In[ ]:


print(float(10))
print(int(3.334))


# In[ ]:


print(int("345"))
#print(int("3a4"))


# In[ ]:


#print(int("345") + 3)
#print("345" + 3)


# In[ ]:


# 5 / 2 1R
155643645 % 3


# In[ ]:


5 / 2


# In[ ]:


print(15 % 2)
print(14 % 2)


# In[ ]:


if 3 % 2 == 0:
    print("even")
else:
    print("odd")


# In[ ]:


print(7 % 3)
print(8 % 3)
print(9 % 3)
print(10 % 3)
print(11 % 3)
print(12 % 3)


# In[ ]:


for i in range(10):
    print(i)


# In[ ]:


for i in range(10):
    print(i+1)


# In[ ]:


for i in range(5,10):
    print(i+1)


# In[ ]:


for i in range(1,100, 3):
    print(i+1)


# In[ ]:


for number in range(100):
    print(number)


# Excercise #1
# Print alle Numbers from 0 to 20 and information if number is even or odd
# 
# 0 is even
# 1 is odd
# 2 is even
# ...

# In[ ]:


for i in range(21):
    if (i % 2 == 0): 
        print(i, " is even")
    else:
        print(i, " is odd")
print("end")


# In[ ]:


# importing the math library
import math as m


# In[ ]:


m.pi


# In[ ]:


number = 3.56
m.ceil(number)


# In[ ]:


m.floor(number)


# In[ ]:


m.sqrt(4)


# In[ ]:


m.sqrt(9)


# In[ ]:


m.pow(3,2)


# In[ ]:


m.pow(3,3)


# In[ ]:


# plot lib
import matplotlib.pyplot as plt


# In[ ]:


plt.plot([1,2,3,4])
plt.xlabel("numbers")
plt.ylabel("some numbers")


# In[ ]:


plt.plot([1, 2, 3, 4], [1, 4, 9, 16])


# In[ ]:


plt.plot([1,2,3,4], [1,4,9,16], 'go')
plt.axis([0, 6, -10, 20])


# In[ ]:


np.arange(0., 5., 0.2)


# In[ ]:


t = np.arange(0., 5., 0.2)
plt.plot(t, t, 'r--', t, t**2, 'bs')


# In[ ]:


plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')


# In[ ]:


t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

plt.figure(1)
plt.subplot(211) # 2, 1, 1 - numrows, numcols, fignum
plt.plot(t1, f(t1), 'bo')

plt.subplot(212) # 2, 1, 2 - numrows, numcols, fignum
plt.plot(t2, f(t2), 'k')


# In[ ]:


plt.figure(1)
plt.subplot(211)
plt.plot([1,2,3])
plt.title('Easy as 1, 2, 3') 

plt.subplot(2,1,2)
plt.plot([4,5,6])

plt.figure(2)
plt.plot([4,5,6])


# In[ ]:


mu = 100
sigma = 15

x = mu + sigma * np.random.randn(10000)

plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)
#plt.hist(x, 50, normed=0, facecolor='g', alpha=0.75)

plt.xlabel("Smarts")
plt.ylabel('Probability')
plt.title('Histogram Test')
plt.text(120, .025, r'$\sigma=15$')
plt.text(160, .025, r'$\mu=100$')

plt.grid(True)

plt.show()

