#!/usr/bin/env python
# coding: utf-8

# These exercises accompany the tutorial on [functions and getting help](https://www.kaggle.com/colinmorris/functions-and-getting-help).
# 
# As before, don't forget to run the setup code below before jumping into question 1.

# In[ ]:


# SETUP. You don't need to worry for now about what this code does or how it works.
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex2 import *
print('Setup complete.')


# <h1> Getting Help </h1>
# 
# for getting help in any function just add question mark in front of it for example run block given below for help in round function

# In[ ]:


get_ipython().run_line_magic('pinfo', 'round')


# # Solved Exercises from python tutorial.

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
    return round(num,2)
    pass

q1.check()


# In[ ]:


# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
q1.solution()


# ## 2.
# The help for `round` says that `ndigits` (the second argument) may be negative.
# What do you think will happen when it is? Try some examples in the following cell?
# 
# Can you think of a case where this would be useful?

# In[ ]:


# Put your test code here
round(559,ndigits=-3)


# In[ ]:


q2.solution()


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


def to_smash(total_candies,num_of_friends=3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between 3 friends.
    
    >>> to_smash(91)
    1
    """
    return total_candies % num_of_friends

q3.check()


# In[ ]:


#q3.hint()


# In[ ]:


q3.solution()


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
smallest_abs = min(abs(x),abs(y))
smallest_abs


# In[ ]:


def f(x):
    y = abs(x)
    return y

print(f(5))


# <h1> function Calling </h1> 
# Calling function inside another function to learn how operation work inside a function.

# In[ ]:


def funct1(A):
    A.append(1)

    
def funct3(B):
    B = 3


# In[ ]:



def funct2():
    c = [1,2,3]
    funct1(c)
    funct3(c)
    
    return c
    

print(funct2())


# funct3 declaration doesnot have any effect , As we can see we can manipulate the values through our function but declaration are ignored from called function.
