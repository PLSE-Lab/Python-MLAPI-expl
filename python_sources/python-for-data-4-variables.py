#!/usr/bin/env python
# coding: utf-8

# # Python for Data 4: Variables
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)

# In this short lesson, we'll learn how to define variables in Python. A variable is a name you assign a value or object, such as one of the basic data types that we learned about last time. After assigning a variable, you can access its associated value or object using the variable's name. Variables are a convenient way of storing values with names that are meaningful.
# 
# In Python, assign variables using "=":

# In[ ]:


x = 10                 
y = "Python is fun"         
z = 144**0.5 == 12

print(x)
print(y)
print(z)


# *Note: assigning a variable does not produce any output.*
# 
# When assigning a variable, it is good practice to put a space between the variable name, the assignment operator and the variable value for clarity:

# In[ ]:


p=8          # This works, but it looks messy.
print(p)   

p = 10       # Use spaces instead
print(p)


# As shown above, you can reassign a variable after creating it by assigning the variable name a new value. After assigning variables, you can perform operations on the objects assigned to them using their names:

# In[ ]:


x + z + p


# You can assign the same object to multiple variables with a multiple assignment statement.

# In[ ]:


n = m = 4
print(n)
print(m)


# You can also assign several different variables at the same time using a comma separated sequence of variable names followed by the assignment operator and a comma separated sequence of values inside parentheses:

# In[ ]:


# Assign 3 variables at the same time:

x, y, z = (10, 20 ,30)

print(x)
print(y)
print(z)


# This method of extracting variables from a comma separated sequence is known as "tuple unpacking."
# 
# You can swap the values of two variables using a similar syntax:

# In[ ]:


(x, y) = (y, x)

print(x)
print(y)


# We'll learn more about tuples in the next lesson, but these are very common and convenient methods for of assigning and altering variables in Python.
# 
# When you assign a variable in Python, the variable is a reference to a specific object in the computer's memory. Reassigning a variable simply switches the reference to a different object in memory. If the object a variable refers to in memory is altered in some way, the value of the variable corresponding to the altered object will also change. All of the basic data types we've seen thus far are immutable, meaning they cannot be changed after they are created. If you perform some operation that appears to alter an immutable object, it is actually creating a totally new object in memory, rather than changing the original immutable object.
# 
# Consider the following example:

# In[ ]:


x = "Hello"     # Create a new string
y = x           # Assign y the same object as x
y = y.lower()   # Assign y the result of y.lower()
print(x)
print(y)


# 
# In the case above, we first assign x the value "Hello", a string object stored somewhere in memory. Next we use the string method lower() to make the string assigned to y lowercase. Since strings are immutable, Python creates an entirely new string, "hello" and stores it somewhere in memory separate from the original "Hello" object. As a result, x and y refer to different objects in memory and produce different results when printed to the console.
# 
# By contrast, lists are a mutable data structure that can hold multiple objects. If you alter a list, Python doesn't make an entirely new list in memory: it changes the actual list object itself. This can lead to seemingly inconsistent and confusing behavior:

# In[ ]:


x = [1,2,3]    # Create a new list
y = x          # Assign y the same object as x
y.append(4)    # Add 4 to the end of list y
print(x)
print(y)


# In this case, x and y still both refer to the same original list, so both x and y have the same value, even though it may appear that the code only added the number 4 to list y.

# # Wrap Up

# Variables are a basic coding construct used across all programming languages. Many data applications involve assigning data to some variables and then passing those variables on to functions that perform various operations on the data.
# 
# This lesson briefly introduced the concept of tuples and lists, which are sequence data types that can hold several values. In the next lesson, we will dig deeper into these sorts of compound data types.

# # Next Lesson: [Python for Data 5: Lists](https://www.kaggle.com/hamelg/python-for-data-5-lists)
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)
