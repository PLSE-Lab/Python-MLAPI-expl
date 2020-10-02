#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
import time
import math


# In[ ]:


sys.version


# In[ ]:


sys.platform


# In[ ]:


os.getcwd()


# In[ ]:


os.listdir()


# In[ ]:


time.strftime("%d-%B-%Y %H:%M:%S %p",time.localtime())


# In[ ]:


2 + 3


# In[ ]:


2.5 + 6.5


# In[ ]:


10 * 3


# In[ ]:


50 / 6


# In[ ]:


50 // 6


# In[ ]:


50 % 6


# In[ ]:


4 ** 3


# In[ ]:


math.pi


# In[ ]:


math.sqrt(3 ** 2 + 4 ** 2)


# In[ ]:


math.pow(9, 1/2)


# In[ ]:


math.pow(5,3)


# In[ ]:


math.log10(100)


# In[ ]:


math.log2(32)


# In[ ]:


math.log(50)


# In[ ]:


math.e


# In[ ]:


print('Hello World')


# In[ ]:


math.e * 3


# In[ ]:


fname = 'Johnny'
lname = 'Depp'


# In[ ]:


fname


# In[ ]:


lname


# In[ ]:


print(fname, lname)


# In[ ]:


len(fname)


# In[ ]:


len(lname)


# In[ ]:


fname = 'Michael'


# In[ ]:


print(fname, lname)


# In[ ]:


iphone_purchased = 3
iphone_unit_price = 749.00


# In[ ]:


total_price = iphone_purchased * iphone_unit_price


# In[ ]:


total_price


# In[ ]:


str = 'United States'


# In[ ]:


len(str)


# In[ ]:


str[0]


# In[ ]:


str[-1]


# In[ ]:


str[0:5]


# In[ ]:


str[0:6]


# In[ ]:


str[0:4]


# In[ ]:


str[7:]


# In[ ]:


str[-6:]


# In[ ]:


str[::-1]


# In[ ]:


str[::2]


# In[ ]:


str * 2


# In[ ]:


str + ' of America'


# In[ ]:


str.capitalize()


# In[ ]:


str.lower()


# In[ ]:


str.upper()


# In[ ]:


s = str.split()
s


# In[ ]:


type(s)


# In[ ]:


s[0]


# In[ ]:


s[1]


# In[ ]:


str = 'Be the change you want to see in the world'


# In[ ]:


len(str)


# In[ ]:


for word in str.split():
  print(word)


# In[ ]:


str_split = str.split(sep=None)


# In[ ]:


str_split


# In[ ]:


str.isdigit()


# In[ ]:


str.isalpha()


# In[ ]:


str.isalnum()


# In[ ]:


str.isnumeric()


# In[ ]:


str.isprintable()


# In[ ]:


str.isupper()


# In[ ]:


str.islower()


# In[ ]:


str.count('the')


# In[ ]:


str.count('be')


# In[ ]:


str.find('an')


# In[ ]:


str.find('Be')


# In[ ]:


str_copy = str


# In[ ]:


str_copy


# In[ ]:


del str, str_copy

