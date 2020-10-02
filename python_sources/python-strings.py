#!/usr/bin/env python
# coding: utf-8

# # Python Strings

# In this tutorial you will learn to create, format, modify and delete strings in Python. Also, you will be introduced to various string operations and functions.

# How to create a string in Python?

# In[ ]:


#Some Ways to create Strings.
my_string = 'Hello'
print(my_string)

my_string = "Hello"
print(my_string)

my_string = '''Hello'''
print(my_string)


# In[ ]:


#String with multiple Line
my_string = """
Hello, welcome to
the world of Python"""
print(my_string)


# You can check if any variable is string or not using "type()".

# In[ ]:


my_string="Jovian"
print(my_string,type(my_string))


# **How to access characters in a string?**

# We can access individual characters using indexing and a range of characters using slicing. 
# Index starts from 0. Trying to access a character out of index range will raise an IndexError. 
# The index must be an integer. We can't use float or other types, this will result into TypeError.
# 
# Python allows negative indexing for its sequences.
# 
# The index of -1 refers to the last item, -2 to the second last item and so on. We can access a range of items in a string by using the slicing operator (colon).

# In[ ]:


str = 'Jovian'
print('str = ', str)

#first character
print('str[0] = ', str[0])

#last character
print('str[-1] = ', str[-1])

#slicing 2nd to 5th character
print('str[1:5] = ', str[1:5])

#slicing 2nd to 2nd last character
print('str[1:-1] = ', str[1:-1])


# **How to change or delete a string?**

# Strings are immutable. This means that elements of a string cannot be changed once it has been assigned. We can simply reassign different strings to the same name.
# 
# We cannot delete or remove characters from a string. But deleting the string entirely is possible using the keyword del.

# # Python String Operations

# Concatenating Two or More Strings

# Joining of two or more strings into a single one is called concatenation.
# 
# The + operator does this in Python. Simply writing two string literals together also concatenates them.
# 
# The * operator can be used to repeat the string for a given number of times.

# In[ ]:


str1 = 'Hello'
str2 ='World!'

# using +
print('str1 + str2 = ', str1 + str2)

# using *
print('str1 * 3 =', str1 * 3)


# Iterating Through String
# Using for loop we can iterate through a string. Here is an example to count the number of 'l' in a string.

# In[ ]:


count = 0
for letter in 'Hello World':
    if(letter == 'l'):
        count += 1
print(count,'letters found')


# String Membership Test
# We can Check whether a sub string exists within a string or not, using the keyword **in**.

# In[ ]:


'a' in 'Jovian'


# In[ ]:


'at' not in 'battle'


# Built-in functions to Work with Python Strings

# The enumerate() function returns an enumerate object. It contains the index and value of all the items in the string as pairs. This can be useful for iteration.
# 
# Similarly, len() returns the length (number of characters) of the string.

# In[ ]:


str = 'jovian'

# enumerate()
list_enumerate = list(enumerate(str))
print('list[(enumerate(str)] = ', list_enumerate)

#character count
print('len(str) = ', len(str))


# Escape Sequence
# If we want to print a text like -He said, "What's there?"- we can neither use single quote or double quotes. This will result into SyntaxError as the text itself contains both single and double quotes.

# In[ ]:


# using triple quotes
print('''He said, "What's there?"''')

# escaping single quotes
print('He said, "What\'s there?"')

# escaping double quotes
print("He said, \"What's there?\"")


# # The format() Method for Formatting Strings

# The format() method that is available with the string object is very versatile and powerful in formatting strings. Format strings contains curly braces {} as placeholders or replacement fields which gets replaced.

# In[ ]:


# default(implicit) order
default_order = "{}, {} and {}".format('Amit','Bhavesh','Nilesh')
print('\n-   Using Default Order   -')
print(default_order)

# order using positional argument
positional_order = "{1}, {0} and {2}".format('Amit','Bhavesh','Nilesh')
print('\n-  Using Positional Order   -')
print(positional_order)

# order using keyword argument
keyword_order = "{a}, {b} and {n}".format(a='Amit',b='Bhavesh',n='Nilesh')
print('\n-  Using Keyword Order   -')
print(keyword_order)


# In Python 3.6 a new string formatting method is introduced which is called **f-string**.
# We can use f-string to print variables inside the string in a easy way.

# In[ ]:


name="John"
age=25
print(f"My name is {name},and I am {age} years old.")


# # Common Python String Methods

# lower() :This mehod converts the string into lowercase.

# In[ ]:


str.lower()


# upper() :This mehod converts the string into uppercase.

# In[ ]:


str.upper()


# split() :This will split all words into a list

# In[ ]:


str="Hello World"
str.split()


# join() :This will join all words into a string

# In[ ]:


l=['Happy','Birthday']
''.join(l)


# find() :This will find the given substring and returns the indexvalue of its first occurence

# In[ ]:


str.find('o')


# replace:('first','second') :This will replace first word with the second word

# In[ ]:


str.replace('Hello','Happy')

