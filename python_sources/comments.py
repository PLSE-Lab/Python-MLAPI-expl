#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# this is a comment, nothing will happen

#spyder block comment: ctrl+4

# ctrl + / jupyter


# In[ ]:


"""
multiple lines 
comments

"""


# In[ ]:


"""
multiple lines 
comments

"""
print("hello world")


# In[ ]:


#BAD comments:

#we will print answers based on if and elif statements.

year = 2018

if year % 400 == 0:#if the rest is 0 when divided by 400
    bisextile = True #then bisextile is true
elif year % 100 == 0:
    bisextile = False
elif year % 4 == 0:
    bisextile = True
else:
    bisextile = False
    
if bisextile:
    print("yes, this year is bisextile")#printing that this year is bisextile if it is bisextile
else:
    print("no, this year isn't bisextile")


# In[ ]:


#Better comment:

#tells if a year is bisextile or not

year = 2018

#a year is bisextile when it is exactly divisible by 400, or divisible by 4 (only if not divisible by 100)
if year % 400 == 0:
    bisextile = True 
elif year % 100 == 0:
    bisextile = False
elif year % 4 == 0:
    bisextile = True
else:
    bisextile = False
    
if bisextile:
    print("yes, this year is bisextile")
else:
    print("no, this year isn't bisextile")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




