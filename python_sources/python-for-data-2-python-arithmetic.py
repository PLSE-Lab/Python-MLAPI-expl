#!/usr/bin/env python
# coding: utf-8

# # Python for Data 2: Python Arithmetic
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)

# In Part 1 of this series, we discussed the basics of what Python is and how we will interact with Python using the Kaggle Jupyter notebook environment. With those preliminaries out of the way, we can jump in and start learning Python. In this short lesson, we'll learn how to use Python to do arithmetic operations.
# 
# Python allows you to perform all of the basic mathematical computations that you can perform on a calculator, using syntax similar to what you'd expect if you were using a calculator. To perform arithmetic with Python, simply type the operation you wish to perform into a code cell and then run the cell to view the result. (Remember, if you like you can fork this notebook using the blue button at the top of the screen and then run code cells yourself by selecting the cell you wish to run and pressing control + enter.).
# 
# Use the + sign to perform addition:

# In[ ]:


10 + 5


# Use the - sign to perform subtraction:
# 
# Note: putting a - sign in front of a number makes it negative.

# In[ ]:


10 - 5


# Use * for multiplication:

# In[ ]:


10 * 5


# Use / for decimal division:

# In[ ]:


10 / 3


# Use // for floor division (Round decimal remainders down):

# In[ ]:


10 // 3


# Use ** for exponentiation:

# In[ ]:


10 ** 3


# Math expressions in Python follow the normal arithmetic order of operations so * and / are executed before + and - and ** is executed before multiplication and division.
# 
# Note: Lines within Python code that begin with "#" are text comments. Comments are typically used to make notes and describe what code does and do not affect the code when it is run.

# In[ ]:


# These operations are executed in reverse order of appearance due to the order of operations.

2 + 3 * 5 ** 2


# You can use parentheses in your math expressions to ensure that operations are carried out on the correct order. Operations within parentheses are carried out before operations that are external to the parentheses, just like you'd expect.

# In[ ]:


# This time, the addition comes first and the exponentiation comes last.

((2 + 3) * 5 ) ** 2


# If you're new to programming, you may not be familiar with the modulus operator. The modulus produces the remainder you'd get when dividing two numbers. Use the % sign to take the modulus in Python:

# In[ ]:


100 % 75


# Beyond symbolic operators, Python contains a variety of named math functions available in the "math" module. To load a library into Python, "import" followed by the name of the library.

# In[ ]:


import math     # Load the math module


# In[ ]:


# math.log() takes the natural logarithm of its argument:

math.log(2.7182)


# In[ ]:


# Add a second argument to specify the log base:

math.log(100, 10)       # Take the log base 10 of 100


# In[ ]:


# math.exp() raises e to the power of its argument

math.exp(10) 


# In[ ]:


# Use math.sqrt() to take the square root of a number:

math.sqrt(64)


# In[ ]:


# Use abs() to get the absolute value of a number. Note abs() is a base Python function so you do not need to load the math package to use it.

abs(-30)


# In[ ]:


math.pi   # Get the constant pi


# # Rounding Numbers

# Base Python contains a round() function that lets you round numbers to the nearest whole number. You can also round up or down with math.ceil and math.floor respectively.

# In[ ]:


# Use round() to round a number to the nearest whole number:

round(233.234)


# In[ ]:


# Add a second argument to round to a specified decimal place

round(233.234, 1)   # round to 1 decimal place


# In[ ]:


# Enter a negative number to round to the left of the decimal

round(233.234, -1)   # round to the 10's place


# In[ ]:


# Round down to the nearest whole number with math.floor()

math.floor(2.8) 


# In[ ]:


# Round up with math.ciel()

math.ceil(2.2)


# # Wrap Up

# In this lesson, we learned that Python is a powerful calculator. Any time you need to perform a common mathematical operation in Python, chances are there is a function for it available in one of Python's many libraries. When in doubt, try searching Google. Helpful blog posts and answers posted on programming sites like Stack Overflow can often save you a lot time and help you learn better ways of doing things without reinventing the wheel.
# 
# In the next lesson, we'll learn about the basic data types you'll encounter in Python.

# # Next Lesson: [Python for Data 3: Basic Data Types](https://www.kaggle.com/hamelg/python-for-data-3-basic-data-types)
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)
