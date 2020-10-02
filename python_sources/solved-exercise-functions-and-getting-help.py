#!/usr/bin/env python
# coding: utf-8

# **[Python Micro-Course Home Page](https://www.kaggle.com/learn/python)**
# 
# ---
# 

# These exercises accompany the tutorial on [functions and getting help](https://www.kaggle.com/colinmorris/functions-and-getting-help).
# 
# As before, don't forget to run the setup code below before jumping into question 1.

# In[ ]:


# SETUP. You don't need to worry for now about what this code does or how it works.
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex2 import *
print('Setup complete.')


# # Exercises

# ## 1.
# 
# Complete the body of the following function according to its docstring.
# 
# HINT: Python has a builtin function `round`

# In[ ]:


def round_to_two_places(num):
    """Return the given number rounded to two decimal places. 
    
    >>> round_to_two_places(3.14159)
    3.14
    """
    # Replace this body with your own code.
    # ("pass" is a keyword that does literally nothing. We used it as a placeholder
    # because after we begin a code block, Python requires at least one line of code)
    return round(num, 2)
    
    pass

q1.check()


# In[ ]:


# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
#q1.solution()


# ## 2.
# The help for `round` says that `ndigits` (the second argument) may be negative.
# What do you think will happen when it is? Try some examples in the following cell?
# 
# Can you think of a case where this would be useful?

# In[ ]:


# Put your test code here
def round_test(num):
    return round(num, -4)

print (round_test(409247027))


# In[ ]:


#q2.solution()


# ## 3.
# 
# In a previous programming problem, the candy-sharing friends Alice, Bob and Carol tried to split candies evenly. For the sake of their friendship, any candies left over would be smashed. For example, if they collectively bring home 91 candies, they'll take 30 each and smash 1.
# 
# Below is a simple function that will calculate the number of candies to smash for *any* number of total candies.
# 
# Modify it so that it optionally takes a second argument representing the number of friends the candies are being split between. If no second argument is provided, it should assume 3 friends, as before.
# 
# Update the docstring to reflect this new behaviour.

# In[ ]:


def to_smash(total_candies, n = 3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between n-friends, where the default number of friends is 3.
    
    >>> to_smash(91)
    1
    
    >>> to_smash(91, 4)
    3
    """
    return total_candies % n
print (to_smash (800, 9))
print (to_smash (800))
q3.check()


# In[ ]:


#q3.hint()


# In[ ]:


#q3.solution()


# ## 4.
# 
# It may not be fun, but reading and understanding error messages will be an important part of your Python career.
# 
# Each code cell below contains some commented-out buggy code. For each cell...
# 
# 1. Read the code and predict what you think will happen when it's run.
# 2. Then uncomment the code and run it to see what happens. (**Tip**: In the kernel editor, you can highlight several lines and press `ctrl`+`/` to toggle commenting.)
# 3. Fix the code (so that it accomplishes its intended purpose without throwing an exception)
# 
# <!-- TODO: should this be autochecked? Delta is probably pretty small. -->

# In[ ]:


round_to_two_places(9.9999)


# In[ ]:


x = -10
y = 5
# Which of the two variables above has the smallest absolute value?
smallest_abs = min(abs(x), abs(y))


# In[ ]:


def f(x):
    y = abs(x)
    return y

print(f(5))


# # Keep Going
# 
# You are ready for **[booleans and conditionals](https://www.kaggle.com/colinmorris/booleans-and-conditionals).**
# 

# ---
# **[Python Micro-Course Home Page](https://www.kaggle.com/learn/python)**
# 
# 
