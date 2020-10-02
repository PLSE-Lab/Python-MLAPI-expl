#!/usr/bin/env python
# coding: utf-8

# # Comparison Operators

# \>  greater than
# 
# <  less than
# 
# <= less than or equal to
# 
# \>= greater than or equal to
# 
# == equal to
# 
# != not equal to
# 

# In[ ]:


2<3


# In[ ]:


3<2


# In[ ]:


3>3


# In[ ]:


a = 10
b = 20


# In[ ]:


b<a


# In[ ]:


a>=10


# In[ ]:


b<=20


# In[ ]:


a == b


# In[ ]:


a != b


# In[ ]:


True and True


# In[ ]:


True and False


# In[ ]:


False and False


# In[ ]:


True or True


# In[ ]:


True or False


# In[ ]:


False or False


# In[ ]:


10 < 20 and 2 > 1


# In[ ]:


10 > 20 and 2 > 1


# In[ ]:


10 > 20 or 2 > 1


# In[ ]:


10 > 20 or 2 < 1


# In[ ]:


10 > 20 & 2 > 1


# In[ ]:


(10 > 20) | (2 > 1)


# In[ ]:


(10 > 20) & (2 > 1)


# In[ ]:


(1 == 1) & (2 > 1)


# In[ ]:


myList = ["h", "i", "p"]


# In[ ]:


"p" in myList


# In[ ]:


"p" not in myList


# # If

# In[ ]:


name = "John"
if name == "Bob":
    print('your name is Bob')
    print("hello")


# In[ ]:


name = "Bob"
if name == "Bob":
    print('your name is Bob')
    print("hello")


# In[ ]:


if True:
    print("when if is True what is inside the if is done")


# # Else

# In[ ]:


if name == "Bob":
    print("Hi Bob!")
else:
    print("It seems you are not Bob")


# In[ ]:


name = "Yogesh"


# In[ ]:


if name == "Bob":
    print("Hi Bob!")
else:
    print("It seems you are not Bob")


# # Elif

# In[ ]:


IQ = 300

if IQ < 10:
    print("do you understand what is written?")
elif IQ >= 10 and IQ < 80:
    print("who cares about the IQ...")
elif IQ >= 80 and IQ < 100:
    print("not too great, not too bad")
elif IQ >= 100 and IQ < 120:
    print("you are normal!")
else:
    print("you are not normal!")


# In[ ]:


IQ = 10

if IQ < 10:
    print("do you understand what is written?")

if IQ >= 10 and IQ < 80:
    print("who cares about the IQ...")

if IQ >= 80 and IQ < 100:
    print("not too great, not too bad")
elif IQ >= 100 and IQ < 120:
    print("you are normal!")
else:
    print("you are not normal!")


# # Project
# Create a code that would be able to tell you if a year is bisextile (= leap year) or not.
# 
# "Every year that is exactly divisible by four is a leap year, except for years that are exactly divisible by 100, but these centurial years are leap years if they are exactly divisible by 400. For example, the years 1700, 1800, and 1900 were not leap years, but the years 1600 and 2000 were."
# 
# Hint: to be bisextile a year must be:
# - multiple of 4 (the rest of dividing by 4 is 0), but not a multiple of 100 (ex: year 1800 wasn't bissextile (multiple of 100), but year 1804 was)
# - multiple of 400 (year 2000 was bisextile)
# 
# good luck!
# 
# 
# # Correction
# check the video!
# 

# In[ ]:





# In[ ]:




