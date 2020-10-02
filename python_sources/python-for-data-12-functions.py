#!/usr/bin/env python
# coding: utf-8

# # Python for Data 12: Functions
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)

# The functions built into Python and its libraries can take you a long way, but general-purpose functions aren't always applicable to the specific tasks you face when analyzing data. The ability to create user-defined functions gives you the flexibility to handle situations where pre-made tools don't cut it.

# ## Defining Functions

# Define a function using the "def" keyword followed by the function's name, a tuple of function arguments and then a colon:

# In[ ]:


def my_function(arg1, arg2):   # Defines a new function
    return arg1 + arg2           # Function body (code to execute)


# After defining a function, you can call it using the name you assigned to it, just like you would with a built in function. The "return" keyword specifies what the function produces as its output. When a function reaches a return statement, it immediately exits and returns the specified value. The function we defined above takes two arguments and then returns their sum:

# In[ ]:


my_function(5, 10)


# You can give function arguments a default value that is used automatically unless you override it. Set a default value with the argument_name = argument_value syntax:

# In[ ]:


def sum_3_items(x, y, z, print_args = False): 
    if print_args:                        
        print(x,y,z)
    return x + y + z


# In[ ]:


sum_3_items(5,10,20)        # By default the arguments are not printed


# In[ ]:


sum_3_items(5,10,20,True)   # Changing the default prints the arguments


# A function can be set up to accept any number of named or unnamed arguments. Accept extra unnamed arguments by including *args in the argument list. The unnamed arguments are accessible within the function body as a tuple:

# In[ ]:


def sum_many_args(*args):
    print (type (args))
    return (sum(args))

    
sum_many_args(1, 2, 3, 4, 5)


# Accept additional keyword arguments by putting \*\*kwargs in the argument list. The keyword arguments are accessible as a dictionary:

# In[ ]:


def sum_keywords(**kwargs):
    print (type (kwargs))
    return (sum(kwargs.values()))

    
sum_keywords(mynum=100, yournum=200)


# ## Function Documentation

# If you are writing a function that you or someone else is going to use in the future, it can be useful to supply some documentation that explains how the function works. You can include documentation below the function definition statement as a multi-line string. Documentation typically includes a short description of what the function does, a summary of the arguments and a description of the value the function returns:

# In[ ]:


import numpy as np

def rmse(predicted, targets):
    """
    Computes root mean squared error of two numpy ndarrays
    
    Args:
        predicted: an ndarray of predictions
        targets: an ndarray of target values
    
    Returns:
        The root mean squared error as a float
    """
    return (np.sqrt(np.mean((targets-predicted)**2)))


# *Note: root mean squared error (rmse) is a common evaluation metric in predictive modeling.*
# 
# Documentation should provide enough information that the user doesn't have to read the code in the body of the function to use the function.

# ## Lambda Functions

# Named functions are great for code that you are going to reuse several times, but sometimes you only need to use a simple function once. Python provides a shorthand for creating functions that let you define unnamed (anonymous) functions called Lambda functions, which are typically used in situations where you only plan to use a function in one part of your code.
# 
# The syntax for creating lambda functions looks like this:

# In[ ]:


lambda x, y: x + y


# In the function above, the keyword "lambda" is similar to "def" in that it signals the definition of a new lambda function. The values x, y are the arguments of the function and the code after the colon is the value that the function returns.
# 
# You can assign a lambda function a variable name and use it just like a normal function:

# In[ ]:


my_function2 = lambda x, y: x + y

my_function2(5,10)


# Although you can assign a name to lambda function, their main purpose is for use in situations where you need to create an unnamed function on the fly, such as when using functions that take other functions as input. For example, consider the Python built in function map(). map() takes a function and an iterable like a list as arguments and applies the function to each item in the iterable. Instead of defining a function and then passing that function to map(), we can define a lambda function right in the call to map():

# In[ ]:


# Example of using map() without a lambda function

def square(x):    # Define a function
    return x**2

my_map = map(square, [1, 2, 3, 4, 5])  # Pass the function to map()

for item in my_map:
    print(item)


# In[ ]:


# Example of using map() with a lambda function

my_map = map(lambda x: x**2, [1, 2, 3, 4, 5]) 

for item in my_map:
    print(item)


# The lambda function version of this code is shorter and more importantly, it avoids creating a named function that won't be used anywhere else in our code.

# ## Wrap Up

# Python's built in functions can take you a long way in data analysis, but sometimes you need to define your own functions to perform project-specific tasks.
# 
# In our final lesson on Python programming constructs, we'll cover several miscellaneous Python functions and conveniences including the ability to construct lists and dictionaries quickly and easily with list and dict comprehensions.

# ## Next Lesson: [Python for Data 13: List Comprehensions](https://www.kaggle.com/hamelg/python-for-data-13-list-comprehensions)
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)
