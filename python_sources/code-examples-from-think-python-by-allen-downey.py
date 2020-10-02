#!/usr/bin/env python
# coding: utf-8

# #### Chapter 1 - The Way of the Program
# 
# E-book available here : http://greenteapress.com/thinkpython2/html/index.html

# The first program

# In[ ]:


print ('Hello World!')


# Basic Arithmetic operations

# In[ ]:


84 * -2


# Exponentiation

# In[ ]:


6 ** 2


# Bitwise operations

# In[ ]:


6 ^ 2


# Types of values

# In[ ]:


type('1,00,000')


# ### Chapter 2 - Variables, Expressions and Statements

# In[ ]:


x = y = 1
print(x)
print(y)


# In[ ]:


z=3,
print(z)


# In[ ]:


(4/3) * 3 * x**3


# ### Chapter - 5 Conditionals and Recursion

# In[ ]:


minutes = 105
hours = minutes / 60
print (hours)


# In[ ]:


hours = minutes // 60
print(hours)


# In[ ]:


rem = minutes % 60
print(rem)


# In[ ]:


5 != 6


# In[ ]:


2 and True


# In[ ]:


x = 5
if x%2 == 0 :
    print("even")
else :
    print("odd")


# ##### Recursive function call

# In[ ]:


def countdown(n):
    if n <= 0:
        print('Blastoff!')
    else:
        print(n)
        countdown(n-1)


# In[ ]:


countdown(5)


# In[ ]:


def print_n(s, n):
    if n <= 0:
        return
    print(s)
    print_n(s, n-1)


# In[ ]:


print_n("hello world", 5)


# In[ ]:


#### Infinite Recursion


# In[ ]:


def recurse():
    recurse()


# In[ ]:


recurse()


# In[ ]:




