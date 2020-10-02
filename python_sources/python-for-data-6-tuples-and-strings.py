#!/usr/bin/env python
# coding: utf-8

# # Python for Data 6: Tuples and Strings
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)

# In the last lesson we learned about lists, Python's jack-of-all trades sequence data type. In this lesson we'll take a look at 2 more Python sequences: tuples and strings.

# ## Tuples

# Tuples are an immutable sequence data type that are commonly used to hold short collections of related data. For instance, if you wanted to store latitude and longitude coordinates for cities, tuples might be a good choice, because the values are related and not likely to change. Like lists, tuples can store objects of different types.
# 
# Construct a tuple with a comma separated sequence of objects within parentheses:

# In[1]:


my_tuple = (1,3,5)

print(my_tuple)


# Alternatively, you can construct a tuple by passing an iterable into the tuple() function:

# In[2]:


my_list = [2,3,1,4]

another_tuple = tuple(my_list)

another_tuple


# Tuples generally support the same indexing and slicing operations as lists and they also support some of the same functions, with the caveat that tuples cannot be changed after they are created. This means we can do things like find the length, max or min of a tuple, but we can't append new values to them or remove values from them:

# In[3]:


another_tuple[2]     # You can index into tuples


# In[4]:


another_tuple[2:4]   # You can slice tuples


# In[5]:


# You can use common sequence functions on tuples:

print( len(another_tuple))   
print( min(another_tuple))  
print( max(another_tuple))  
print( sum(another_tuple))  


# In[6]:


another_tuple.append(1)    # You can't append to a tuple


# In[7]:


del another_tuple[1]      # You can't delete from a tuple


# You can sort the objects in tuple using the sorted() function, but doing so creates a new list containing the result rather than sorting the original tuple itself like the list.sort() function does with lists:

# In[8]:


sorted(another_tuple)


# In[9]:


list1 = [1,2,3]

tuple1 = ("Tuples are Immutable", list1)

tuple2 = tuple1[:]                       # Make a shallow copy

list1.append("But lists are mutable")

print( tuple2 )                          # Print the copy


# To avoid this behavior, make a deepcopy using the copy library:

# In[10]:


import copy

list1 = [1,2,3]

tuple1 = ("Tuples are Immutable", list1)

tuple2 = copy.deepcopy(tuple1)           # Make a deep copy

list1.append("But lists are mutable")

print( tuple2 )                          # Print the copy


# ## Strings

# We already learned a little bit about strings in the lesson on basic data types, but strings are technically sequences: immutable sequences of text characters. As sequences, they support indexing operations where the first character of a string is index 0. This means we can get individual letters or slices of letters with indexing:

# In[11]:


my_string = "Hello world"

my_string[3]    # Get the character at index 3


# In[12]:



my_string[3:]   # Slice from the third index to the end


# In[13]:



my_string[::-1]  # Reverse the string


# In addition, certain sequence functions like len() and count() work on strings:

# In[14]:


len(my_string)


# In[15]:


my_string.count("l")  # Count the l's in the string


# As immutable objects, you can't change a string itself: every time you transform a string with a function, Python makes a new string object, rather than actually altering the original string that exists in your computer's memory.
# 
# Strings have many associated functions. Some basic string functions include:

# In[16]:


# str.lower()     

my_string.lower()   # Make all characters lowercase


# In[17]:


# str.upper()     

my_string.upper()   # Make all characters uppercase


# In[18]:


# str.title()

my_string.title()   # Make the first letter of each word uppercase


# Find the index of the first appearing substring within a string using str.find(). If the substring does not appear, find() returns -1:

# In[19]:


my_string.find("W")


# Notice that since strings are immutable, we never actually changed the original value of my_string with any of the code above, but instead generated new strings that were printed to the console. This means "W" does not exist in my_string even though our call to str.title() produced the output 'Hello World'. The original lowercase "w" still exists at index position 6:

# In[20]:


my_string.find("w")


# Find and replace a target substring within a string using str.replace():

# In[21]:


my_string.replace("world",    # Substring to replace
                  "friend")   # New substring


# Split a string into a list of substrings based on a given separating character with str.split():

# In[22]:


my_string.split()     # str.split() splits on spaces by default


# In[23]:


my_string.split("l")  # Supply a substring to split on other values


# Split a multi-line string into a list of lines using str.splitlines():

# In[24]:


multiline_string = """I am
a multiline 
string!
"""

multiline_string.splitlines()


# Strip leading and trailing characters from both ends of a string with str.strip().

# In[26]:


# str.strip() removes whitespace by default

"    strip white space!   ".strip() 


# Override the default by supplying a string containing all characters you'd like to strip as an argument to the function:

# In[27]:


"xXxxBuyNOWxxXx".strip("xX")


# You can strip characters from the left or right sides only with str.lstrip() and str.rstrip() respectively.
# 
# To join or concatenate two strings together, you can us the plus (+) operator:

# In[28]:


"Hello " + "World"


# Convert the a list of strings into a single string separated by a given delimiter with str.join():

# In[29]:


" ".join(["Hello", "World!", "Join", "Me!"])


# Although the + operator works for string concatenation, things can get messy if you start trying to join more than a couple values together with pluses.

# In[30]:


name = "Joe"
age = 10
city = "Paris"

"My name is " + name + " I am " + str(age) + " and I live in " + "Paris"


# For complex string operations of this sort is preferable to use the str.format() function or formatted strings. str.format() takes in a template string with curly braces as placeholders for values you provide to the function as the arguments. The arguments are then filled into the appropriate placeholders in the string:

# In[31]:


template_string = "My name is {} I am {} and I live in {}"

template_string.format(name, age, city)


# Formatted strings or f-strings for short are an alternative, relatively new (as of Python version 3.6) method for string formatting. F-strings are strings prefixed with "f" (or "F") that allow you to insert existing variables into string by name by placing them within curly braces:

# In[32]:


# Remaking the example above using an f-string

f"My name is {name} I am {age} and I live in {city}"


# As you can see, being able to directly access variable values via variable names can make f-strings more interpretable and intuitive than older formatting methods. For more on f-strings, see the [official documentation](https://docs.python.org/3/reference/lexical_analysis.html#f-strings).

# ## Wrap Up

# Basic sequences like lists, tuples and strings appear everywhere in Python code, so it is essential to understand the basics of how they work before we can start using Python for data analysis. We're almost ready to dive into data structures designed specifically data analysis, but before we do, we need to cover two more useful built in Python data structures: dictionaries and sets.

# ## Next Lesson: [Python for Data 7: Dictionaries and Sets](https://www.kaggle.com/hamelg/python-for-data-7-dictionaries-and-sets)
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)
