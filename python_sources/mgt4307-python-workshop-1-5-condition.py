#!/usr/bin/env python
# coding: utf-8

# **Condition**

# Simple If Statement. If the condition is met, do the following statments

# In[12]:


age = 2


# In[2]:


age


# In[13]:


if age > 18:
    print("You are adult")
    
    print("You are adult")
    
print("hello")


# If-else statement. If the condition is met, do the following statement, if not met, do the statement after the else:

# In[19]:


age = 23

if age > 18:
    print("You are adult")
    print("hello")
else:
    print("You are not adult")
print("Hey")


# If-elif-else statement. multiple condition

# In[ ]:


bmi = 27


# In[24]:


bmi = 50
age = 20

if age > 18:
    if bmi < 15:
        print("Very severely underweight")
    elif bmi < 16:
        print("Severely underweight")
    elif bmi < 18.5:
        print("Underweight")
    elif bmi < 25:
        print("Normal")
    elif bmi < 30:
        print("Overweight")
    elif bmi < 35:
        print("Moderately obese")
    elif bmi < 40:
        print("Severely obese")
    else:
        print("Very severely obese")
else:
    print("the bmi reference is only for adult")


# In[31]:


age = 23
bmi = 40

(age > 50 or bmi > 23) and bmi < 50


# In[ ]:


if bmi < 15 and age > 18:
    print("Very severely underweight")
elif bmi < 16 and age > 18:
    print("Severely underweight")
elif bmi < 18.5 and age > 18:
    print("Underweight")
elif bmi < 25 and age > 18:
    print("Normal")
elif bmi < 30 and age > 18:
    print("Overweight")
elif bmi < 35 and age > 18:
    print("Moderately obese")
elif bmi < 40 and age > 18:
    print("Severely obese")
elif age > 18:
    print("Very severely obese")
else:
    print("the bmi reference is only applicable for adult")


# In[ ]:




