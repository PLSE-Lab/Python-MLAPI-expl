#!/usr/bin/env python
# coding: utf-8

# # Python for Data 3: Basic Data Types
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)

# In the [last lesson](https://www.kaggle.com/hamelg/python-for-data-2-python-arithmetic) we learned that Python can act as a powerful calculator, but numbers are just one of many basic data types you'll encounter using Python and conducting data analysis. A solid understanding of basic data types is essential for working with data in Python.

# # Integers

# Integers or "ints" for short, are whole-numbered numeric values. Any positive or negative number (or 0) without a decimal is an integer in Python. Integer values have unlimited precision, meaning an integer is exact. You can check the type of a Python object with the type() function. Let's run type() on an integer:

# In[ ]:


type(12)


# Above we see that the type of 12 is of type "int". You can also use the function isinstance() to check whether an object is an instance of a given type:

# In[ ]:


# Check if 12 is an instance of type "int"

isinstance(12, int)


# The code output True confirms that 12 is an int.
# 
# Integers support all the basic math operations we covered last time. If a math operation involving integers would result in a non-integer (decimal) value, the result is becomes a float:

# In[ ]:


1/3  # A third is not a whole number*


# In[ ]:


type(1/3)  # So the type of the result is not an int


# *Note: In Python 2, integer division performs floor division instead of converting the ints to floats as we see here in Python 3.*

# # Floats

# Floating point numbers or "floats" are numbers with decimal values. Unlike integers, floating point numbers don't have unlimited precision because irrational decimal numbers are infinitely long and therefore can't be stored in memory. Instead, the computer approximates the value of long decimals, so there can be small rounding errors in long floats. This error is so minuscule it usually isn't of concern to us, but it can add up in certain cases when making many repeated calculations.
# 
# Every number in Python with a decimal point is a float, even if there are no non-zero numbers after the decimal:

# In[ ]:


type(1.0)


# In[ ]:


isinstance(0.33333, float)


# The arithmetic operations we learned last time work on floats as well as ints. If you use both floats and ints in the same math expression the result is a float:

# In[ ]:


5 + 1.0


# You can convert a float to an integer using the int() function:

# In[ ]:


int(6.0)


# You can convert an integer to a float with the float() function:

# In[ ]:


float(6)


# Floats can also take on a few special values: Inf, -Inf and NaN. Inf and -Inf stand for infinity and negative infinity respectively and NaN stands for "not a number", which is sometimes used as a placeholder for missing or erroneous numerical values.

# In[ ]:


type ( float ("Inf") )


# In[ ]:


type ( float ("NaN") )


# *Note: Python contains a third, uncommon numeric data type "complex" which is used to store complex numbers.*

# # Booleans

# Booleans or "bools" are true/false values that result from logical statements. In Python, booleans start with the first letter capitalized so True and False are recognized as bools but true and false are not. We've already seen an example of booleans when we used the isinstance() function above.

# In[ ]:


type(True)


# In[ ]:


isinstance(False, bool)  # Check if False is of type bool


# You can create boolean values with logical expressions. Python supports all of the standard logic operators you'd expect:

# In[ ]:


# Use >  and  < for greater than and less than:
    
20 > 10 


# In[ ]:


20 < 5


# In[ ]:


# Use >= and  <= for greater than or equal and less than or equal:

20 >= 20


# In[ ]:


# Use == (two equal signs in a row) to check equality:

10 == 10


# In[ ]:


40 == 40.0  # Equivalent ints and floats are considered equal


# In[ ]:


# Use != to check inequality. (think of != as "not equal to")

1 != 2


# In[ ]:


# Use the keyword "not" for negation:

not False


# In[ ]:


# Use the keyword "and" for logical and:

(2 > 1) and (10 > 11)


# In[ ]:


# Use the keyword "or" for logical or:

(2 > 1) or (10 > 11)


# Similar to math expressions, logical expressions have a fixed order of operations. In a logical statement, comparisons like >, < and == are executed first, followed by "not", then "and" and finally "or". See the following link to learn more: [Python operator precedence](https://docs.python.org/3/reference/expressions.html#operator-precedence). Use parentheses to enforce the desired order of operations.

# In[ ]:


2 > 1 or 10 < 8 and not True


# In[ ]:


((2 > 1) or (10 < 8)) and (not True)


# You can convert numbers into boolean values using the bool() function. All numbers other than 0 convert to True:

# In[ ]:


bool(1)


# In[ ]:


bool(0)


# # Strings

# Text data in Python is known as a string or "str". Surround text with single or double quotation marks to create a string:

# In[ ]:


type("cat")


# In[ ]:


type('1')


# Two quotation marks right next to each other (such as '' or "") without anything in between them is known as the empty string. The empty string often represents a missing text value.
# 
# Numeric data and logical data are generally well-behaved, but strings of text data can be very messy and difficult to work with. Cleaning text data is often one of the most laborious steps in preparing real data sets for analysis. We will revisit strings and functions to help you clean text data in future lesson.

# # None

# In Python, "None" is a special data type that is often used to represent a missing value. For example, if you define a function that doesn't return anything (does not give you back some resulting value) it will return "None" by default.

# In[ ]:


type(None)  


# In[ ]:


# Define a function that prints the input but returns nothing*

def my_function(x):
    print(x)
    
my_function("hello") == None  # The output of my_function equals None


# *Note: We will cover defining custom functions in detail in a future lesson.*

# # Wrap Up

# This lesson covered the most common basic data types in Python, but it is not an exhaustive list of Python data objects or the functions. The Python's [official documentation](https://docs.python.org/3/library/stdtypes.html) has a more thorough summary of built-in types, but it is a bit more verbose and detailed than is necessary when you are first getting started with the language.
# 
# Now that we know about the basic data types, it would be nice to know how to save values to use them later. We'll cover how to do that in the next lesson on variables.

# # Next Lesson: [Python for Data 4: Variables](https://www.kaggle.com/hamelg/python-for-data-4-variables)
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)
