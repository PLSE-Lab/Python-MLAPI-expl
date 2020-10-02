#!/usr/bin/env python
# coding: utf-8

# # 01 Basics
# - Hello World
# - Comments
# - Errors
# - Types
# - Expressions
# - Variables
# - Strings

# In[ ]:


print("Hi")


# In[ ]:


# Print Hello Python 101
print("Hello Python 101")


# In[ ]:


# Check the Python Version

import sys
print(sys.version)


# # Comments

# In[ ]:


# This is a single line comment
print("Comment") # This is also a single line comment

# These are multiline commentsII
# that spread on more than one line


# # Errors

# In[ ]:


# Print Hello Python 101
Print("Hello Python 101")


# In[ ]:


# Print Hello Python 101
print("Hello Python 102")


# # Types

# In[ ]:


match = "Hello Python 101"
print(f'Match {match} \nMatch type {type(match)}')


# In[ ]:


match = 101
print(f'Match {match} \nMatch type {type(match)}')


# In[ ]:


match = "101"
print(f'Match {match} \nMatch type {type(match)}')


# In[ ]:


match = 10.1
print(f'Match {match} \nMatch type {type(match)}')


# In[ ]:


match = True
print(f'Match {match} \nMatch type {type(match)}')


# In[ ]:


match = int(True)
print(f'Match {match} \nMatch type {type(match)}')


# In[ ]:


match = bool(0)
print(f'Match {match} \nMatch type {type(match)}')


# # Expressions 

# In[ ]:


1+2+3


# In[ ]:


2*2


# In[ ]:


1-2


# In[ ]:


1/2


# In[ ]:


1//2


# In[ ]:


2**3


# In[ ]:


(2+3)*4


# # Variables

# In[ ]:


variable = 1
variable


# In[ ]:


print(f'variable: {variable}')


# In[ ]:


print(f'variable: {variable}')
print(f'variable type: {type(variable)}')


# In[ ]:


minutes = 60 # number of seconds per minute
hours = 60 # number of minutes per hour


# In[ ]:


# number of seconds per one hour
seconds = minutes * hours


# In[ ]:


print(f'Total seconds per one hour: {seconds}')
print(f'seconds type: {type(seconds)}')


# # Strings

# In[ ]:


"John Smith"


# In[ ]:


'0 1 2 3 4 5 6 7 8 9'


# In[ ]:


Name = "John Smith"
Name


# ## Indexing

# In[ ]:


len(Name)


# In[ ]:


Name[0]


# In[ ]:


Name[1]


# In[ ]:


# Take care
Name[10]


# In[ ]:


Name[9]


# In[ ]:


Name[-1]


# In[ ]:


Name[-2]


# ## Slicing

# In[ ]:


Name[0:9]


# In[ ]:


Name[0:]


# In[ ]:


Name[:]


# In[ ]:


Name[1:]


# In[ ]:


Name[2:8]


# ## Stride

# In[ ]:


Name[::2]


# In[ ]:


Name[:5:2]


# In[ ]:


Number = '0 1 2 3 4 5 6 7 8 9'
Number[::2]


# ## Concatenation

# In[ ]:


Name = 'John Smith'
Name


# In[ ]:


Statement = Name + " is a Person"
Statement


# ## Escape Sequences

# In[ ]:


print("John Smith \nis a Person")


# In[ ]:


print("John Smith \\is a Person")


# In[ ]:


print("John Smith \tis a Person")


# In[ ]:


print("John Smith \ is a Person")


# In[ ]:


print(r"John Smith \ is a Person")


# ## String Operations

# In[ ]:


A = "John Smith is a Person"
print(f'A Before upper: {A}')
B = A.upper()
print(f'B After upper: {B}')


# In[ ]:


A = "John Smith is a Person"
print(f'A Before upper: {A}')
B = A.upper()
print(f'A After upper: {A}')


# In[ ]:


A = "John Smith is a Person"
B = A.replace('John','Jane')
print(f'A Before replace: {A}')
print(f'B After replace: {B}')


# In[ ]:


A = "John Smith is a Person"
B = A.replace('John','Jane')
print(f'A Before replace: {A}')
print(f'A After replace: {A}')


# In[ ]:


Name = 'John Smith'
Name


# In[ ]:


Name.find('hn')


# In[ ]:


Name.find('Smith')


# In[ ]:


Name.find('Jasdfasdasdf')

