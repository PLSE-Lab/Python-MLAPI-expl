#!/usr/bin/env python
# coding: utf-8

# # Python Basics Example
# 
# ## Indruction: In this article, we will make examples of general python basics, I would like to get new trainings on these issues and reinforce them, and I wanted to create such an article. <u>Over time, all data will continue to be updated daily.</u>
# 
# <font color='blue'>
# # Content
# 
# 1. [Introduction to Variables](#1)
#     * [Integer(int)](#2)
#     * [Float(float)](#3)
#     * [String(str)](#4)
# 1. [Most Used Methods](#5)
#     * [dir() method](#6)
#     * [type() method](#7)
#     * [len() method](#8)
#     * [String Methods](#9)
#         * [replace() method](#10)
#         * [capitalize() method](#11)
#         * [casefold() method](#12)
#         * [endswith() method](#13)
#         * [index() method](#14)
#         * [islower() method](#15)
#         * [isupper() method](#16)
#         * [lower() method](#17)
#         * [upper() method](#18)
#         * [strip() method](#19)
#         * [other methods](#20)
#     * [Lists and Methods](#21)
#     * [Dictionaries and Methods](#25)
#     * [Tuples and Methods](#26)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a id='1'></a>
# # Introduction to Variables

# <a id=2></a>
# ## Integer(int)
# We use it in definitions of numbers that are external to certain (decimal numbers).

# In[ ]:


#integer(int): We use it in definitions of numbers that are external to certain (decimal numbers).

a = 5 #integer
b = 7 #integer
c = 4 #integer
d = 10 #integer

var_sum = a + b #Adds our values a and b
var_subtracts = d - c #Subtracts our values d and c

#we use the print () command to print our values

print(var_sum)
print(var_subtracts)


# In[ ]:


#integer(int): We use it in definitions of numbers that are external to certain (decimal numbers).

#EXAMPLE 1:
a = 100
x = 500
d = 400
e = 600

var_transactions = (d+e) - (a+x)

print(var_transactions)


# In[ ]:


#integer(int): We use it in definitions of numbers that are external to certain (decimal numbers).

#EXAMPLE 2:
s = 10
d = 40
c = 50
a = 20

var_math = ((s*d) / c) * a

print(var_math)


# In[ ]:


#integer(int): We use it in definitions of numbers that are external to certain (decimal numbers).

#EXAMPLE 3:
x = 4
y = 5
z = 8

var_math = (x+y+z)**2 #'**2'it takes the square of the released value.

print(var_math)


# <a id='3'></a>
# ## Float
# We can say that it is a data variable containing decimal numbers other than normal numbers (example; 10.2)

# In[ ]:


#float: We can say that it is a data variable containing decimal numbers other than normal numbers (example; 10.2)

a = 10.2 #float
b = 9.8 #float
c = 20.5 #float
d = 34.2 #float

var_sum = a + b #Adds our values a and b
var_subtracts = d - c #Subtracts our values d and c

#we use the print () command to print our values

print(var_sum)
print(var_subtracts)


# In[ ]:


#float: We can say that it is a data variable containing decimal numbers other than normal numbers (example; 10.2)
#integer(int): We use it in definitions of numbers that are external to certain (decimal numbers).

#EXAMPLE 1

a = 10 #integer
b = 2.2 #float
c = 100 #integer
d = 14.8 #float

var_math1 = (a*b) + (a*c) - (c*d)

print(var_math1)


# In[ ]:


#float: We can say that it is a data variable containing decimal numbers other than normal numbers (example; 10.2)
#integer(int): We use it in definitions of numbers that are external to certain (decimal numbers).

#EXAMPLE 2

a = 10 #integer
b = 2.2 #float
c = 100 #integer
d = 14.8 #float

var_math2 = (a+b)**2

print(var_math2)


# In[ ]:


#float: We can say that it is a data variable containing decimal numbers other than normal numbers (example; 10.2)
#integer(int): We use it in definitions of numbers that are external to certain (decimal numbers).

#EXAMPLE 3

x = 1000
c = 100
d = 10
e = 10.5

var_math3 = ((x/c) * d) * e + 1000

print(var_math3)


# <a id='4'></a>
# ## String(str)
# We use not as a number, but as a variable we assign to our texts or texts.

# In[ ]:


#string: We use not as a number, but as a variable we assign to our texts or texts.

name = "john" #string
surname = "jackson" #string
job = "doctor" #string

print(name, surname, "is a", job)


# In[ ]:


#string: We use not as a number, but as a variable we assign to our texts or texts.

#EXAMPLE 1

number1 = "100"
number2 = "300"

print(number1,number2)
print(number1 + number2)


# In[ ]:


#string: We use not as a number, but as a variable we assign to our texts or texts.

#INCORRECT EXAMPLE

number3 = "100"
number4 = 300

print(number3 + number4) #We can't do the aggregation of the post with an integer value.
# TypeError: must be str, not int


# In[ ]:


#string: We use not as a number, but as a variable we assign to our texts or texts.

#EXAMPLE 2

name1 = "Alex"
surname2 = "John"
age = 19 #integer
job = "Student"

print(name1,surname2, age, "years old and", "He is a", job)


# <a id='5'></a>
# # Most Used Methods

# <a id='6'></a>
# ## dir() method
# It is a method that allows us to see what methods are created by Python.

# In[ ]:


#usage: dir(class) -- we write whatever class we want to call the class

dir(str)


# In[ ]:


#usage: dir(class) -- we write whatever class we want to call the class

dir(int)


# <a id='7'></a>
# ## type() method
# It shows which variable we use our data.

# In[ ]:


# It shows which variable we use our data. usage: type(class)

a = 10 #integer
b = 50 #integer

x = a + b
print(x)
type(x)


# In[ ]:


# It shows which variable we use our data. usage: type(class)

name4 = "angelica"
surname4 = "john"

x = name4 + surname4
print(x)
type(x)


# <a id='8'></a>
# ## len() method
# It gives us information about the text length (s) of the variable we have defined.

# In[ ]:


# It gives us information about the text length (s) of the variable we have defined. - len(class)

x = "100"
c = "he is 20 years old"

len(x) # "100" converted us to 3 because its value is 3 characters


# In[ ]:


# It gives us information about the text length (s) of the variable we have defined. - len(class)

#EXAMPLE 1:

x = 100 #int
y = "20000" #str
z = x + len(y) #int + len(str)

print(z)


# <a id='9'></a>
# # String Methods
# We will apply useful and necessary methods for the articles.

# <a id='10'></a>
# # replace() method
# As the name suggests, a method that allows us to change a certain value.

# In[ ]:


# As the name suggests, a method that allows us to change a certain value. -- variable.replace("value","new_value")

txt = "He is 10 years old"
x = txt.replace("10","20")

print(txt) #default
print(x) #replace


# <a id='11'></a>
# # capitalize() method
# Converts the first character to upper case

# In[ ]:


# Converts the first character to upper case -- variable.capitalize()

high = "hello world"
high.capitalize()


# <a id='12'></a>
# # casefold() method
# Converts string into lower case

# In[ ]:


# Converts string into lower case - variable.casefold()

high2 = "HELLO WORLD"
high2.casefold()


# <a id='13'></a>
# # endswith() method
# Returns true if the string ends with the specified value.

# In[ ]:


#Returns true if the string ends with the specified value. -- variable.endswith()

variable = "hello world"
variable.endswith("d")


# In[ ]:


#Returns true if the string ends with the specified value. -- variable.endswith()

variable = "hello world"
variable.endswith("a")


# <a id='14'></a>
# # index() method
# Searches the string for a specified value and returns the position of where it was found

# In[ ]:


# Searches the string for a specified value and returns the position of where it was found -- variable.index()

variables = "hello world"
variables.index("d")


# In[ ]:


# Searches the string for a specified value and returns the position of where it was found -- variable.index()

# EXAMPLE 1:

variable2 = 10
variables = "hello world"
variables.index("d")

x = variable2 + variables.index("d")
print(x)


# <a id='15'></a>
# # islower() method
# Returns True if all characters in the string are lower case

# In[ ]:


#islower: Returns True if all characters in the string are lower case

is_lower = "this a text"
x = is_lower.islower()

print(x)


# <a id='16'></a>
# # isupper() method
# Returns True if all characters in the string are upper case

# In[ ]:


#isupper: Returns True if all characters in the string are upper case

is_upper = "THIS A TEXT"
x = is_upper.isupper()

print(x)


# <a id='17'></a>
# # lower() method
# Converts a string into lower case

# In[ ]:


#lower: Converts a string into lower case

lower_text = "LOWER TEXT"
z = lower_text.lower()

print(z)


# <a id='18'></a>
# # upper() method
# Converts a string into upper case

# In[ ]:


#upper: Converts a string into upper case

upper_text = "upper text"
z = upper_text.upper()

print(z)


# <a id='19'></a>
# # strip() method
# Returns a trimmed version of the string

# In[ ]:


#strip: Returns a trimmed version of the string

strip_text = "xx ddarkk007 xx"
s = strip_text.strip("xx")

print(s)

#lstrip: Returns a left trim version of the string

lstrip_text = "yy ddarkk007 yy"
l = lstrip_text.lstrip("yy")

print(l)

#rstrip: Returns a right trim version of the string

rstrip_text = "zz ddarkk007 zz"
r = rstrip_text.lstrip("zz")

print(r)


# <a id='20'></a>
# ## Other Methods
# [Different and other methods](https://www.w3schools.com/python/python_strings.asp)

# In[ ]:


#center: Returns a centered string

center_text = "centered string"
x = center_text.center(50)

print(x)


# <a id='21'></a>
# # Lists and Methods
# In this section we will look at lists and methods.
# * Features of the lists; Ordered, data can be added,removed and changed

# In[ ]:


#A list is a collection which is ordered and changeable. In Python lists are written with square brackets.

list1 = ["orange","apple","watermelon"] #create a list
print(list1)


# In[ ]:


#Is list1 really a list?

print(type(list1)) #ok this a list


# In[ ]:


#You access the list items by referring to the index number:

list2 = ["first","second","third"]
x = list2[0] #we called element 0 of the list

print(x)


# In[ ]:


# Let's call the element at the bottom of the list

list3 = ["first","second","third"]
y = list3[-1]

print(y)


# In[ ]:


# You can specify a range of indexes by specifying where to start and where to end the range.

thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "strawberry"]
x = thislist[1:4] # 1.banana 2.cherry 3.orange 4.kiwi(not included)

print(x)


# In[ ]:


# Reverse the list

thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "strawberry"]
a = thislist[::-1]

print(a)


# In[ ]:


# To change the value of a specific item, refer to the index number:
# default list = thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "strawberry"]

thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "strawberry"]
thislist[0] = "watermelon"
thislist[1] = "apple"

print(thislist)


# In[ ]:


# EXAMPLES

# EXAMPLE 1: Create a list!
create_list = ["1","2","3","4","5"]

# EXAMPLE 2: Select the 1st element by synchronizing the list values to c
c = create_list[0]
print(c)

# EXAMPLE 3: Print from 1 to 3
print(create_list[0:3])

# EXAMPLE 4: create_list query your type
print(type(create_list))

# EXAMPLE 5: Reverse a list
print(create_list[::-1])


# <a id='22'></a>
# # append() methods
# To add an item to the end of the list, use the append() method:

# In[ ]:


# To add an item to the end of the list, use the append() method:

x = secondlist = ["ddarkk","yellow","green","world"] #default list
print(x)

secondlist.append("hello") #append hello list
print(secondlist)


# <a id='23'></a>
# # insert() methods
# To add an item at the specified index, use the insert() method:

# In[ ]:


# To add an item at the specified index, use the insert() method:

thislist = ["apple", "banana", "cherry"]
thislist.insert(1, "orange")

print(thislist)


# <a id='24'></a>
# # remove() methods
# The remove() method removes the specified item:

# In[ ]:


# The remove() method removes the specified item:

thislist = ["apple", "banana", "cherry"]
thislist.remove("banana")
print(thislist)


# <a id='25'></a>
# # pop() methods
# The pop() method removes the specified index, (or the last item if index is not specified):

# In[ ]:


# The pop() method removes the specified index, (or the last item if index is not specified):

thislist = ["apple", "banana", "cherry"]
thislist.pop(1)
print(thislist)


# <a id='26'></a>
# # del methods
# The del keyword removes the specified index:

# In[ ]:


# The del keyword removes the specified index:

thislist = ["apple", "banana", "cherry"]
del thislist[0]
print(thislist)


# In[ ]:


# The list failed because it was deleted

thislist = ["apple", "banana", "cherry"]
del thislist

print(thislist)


# <a id='27'></a>
# # clear() methods
# The clear() method empties the list:

# In[ ]:


#The clear() method empties the list:

new_list = ["green","yellow","black"]
new_list.clear()
print(new_list)


# <a id='28'></a>
# # copy() methods
# There are ways to make a copy, one way is to use the built-in List method copy().

# In[ ]:


#There are ways to make a copy, one way is to use the built-in List method copy().

new_list = ["green","yellow","black"]
copy_list = new_list.copy()
print(copy_list)


# <a id='29'></a>
# # Liste Merge
# There are severalaways to join, or concatenate, two or more lists in Python.

# In[ ]:


# There are several ways to join, or concatenate, two or more lists in Python.

list1 = ["a", "b" , "c"]
list2 = [1, 2, 3]

list3 = list1 + list2
print(list3) 


# <a id='30'></a>
# # extend() method
# Use the extend() method to add list2 at the end of list1:

# In[ ]:


# Use the extend() method to add list2 at the end of list1:

list1 = ["a", "b" , "c"]
list2 = [1, 2, 3]

list1.extend(list2)
print(list1)


# <a id='31'></a>
# # Other lists methods

# In[ ]:


dir(list)


# <a id='25'></a>
# # Dictionaries and Methods
# A dictionary is a collection which is unordered, changeable and indexed. In Python dictionaries are written with curly brackets, and they have keys and values.
# 
#     *Can change, new data can be added and unordered

# In[ ]:


# Create and print a dictionary:

new_dict = {
    "brands": "Python",
    "model": "Dictionaries",
    "year": 2020
}

new_dict


# In[ ]:


#Is new_dict really a dictionaries?

new_dict = {
    "brands": "Python",
    "model": "Dictionaries",
    "year": 2020
}

print(type(new_dict))


# In[ ]:


# You can access the items of a dictionary by referring to its key name, inside square brackets:

new_dict = {
    "brands": "Python",
    "model": "Dictionaries",
    "year": 2020
}

x = new_dict["model"]
print(x)

# There is also a method called get() that will give you the same result:

y = new_dict.get("year")
print(y)


# In[ ]:


# You can change the value of a specific item by referring to its key name:

new_dict = {
    "brands": "Python",
    "model": "Dictionaries",
    "year": 2020
}

z = new_dict["model"] = "New_Dictionaries"
print(new_dict)


# In[ ]:


# To determine how many items (key-value pairs) a dictionary has, use the len() method.

new_dict = {
    "brands": "Python",
    "model": "Dictionaries",
    "year": 2020
}

print(len(new_dict))


# In[ ]:


# Adding an item to the dictionary is done by using a new index key and assigning a value to it:

new_dict = {
    "brands": "Python",
    "model": "Dictionaries",
    "year": 2020
}

new_variable = new_dict["color"] = "red"

print(new_variable)
print(new_dict)


# In[ ]:


# There are several methods to remove items from a dictionary:

new_dict = {
    "brands": "Python",
    "model": "Dictionaries",
    "year": 2020
}

new_variable = new_dict["color"] = "red"
remove_variable = new_dict.pop("color")

print(remove_variable)
print(new_dict)


# In[ ]:


# The del keyword removes the item with the specified key name:

new_dict = {
    "brands": "Python",
    "model": "Dictionaries",
    "year": 2020
}

del new_dict["year"]

print(new_dict)


# In[ ]:


# Convert a dictionary to a list

z = list(new_dict)
print(z)


# In[ ]:


# It is also possible to use the dict() constructor to make a new dictionary:

thisdict = dict(brand="Ford", model="Mustang", year=1964)
thisdict


# In[ ]:


new1_dict = {
    "brands": "Python",
    "model": "Dictionaries",
    "year": 2020
}

my_dict = new1_dict.copy()

print(my_dict)


# <a id='26'></a>
# # Tuples and Methods
# A tuple is a collection which is ordered and unchangeable. In Python tuples are written with round brackets.

# In[ ]:


# Create a Tuple:

tuple2 = ("python","ddarkk","green")
print(tuple2)


# In[ ]:


# You can access tuple items by referring to the index number, inside square brackets:

tuple2 = ("python","ddarkk","green")
print(tuple2[1])


# In[ ]:


# Negative indexing means beginning from the end, -1 refers to the last item, -2 refers to the second last item etc.

tuple2 = ("python","ddarkk","green")
print(tuple2[-1])


# In[ ]:


# You can specify a range of indexes by specifying where to start and where to end the range.
# When specifying a range, the return value will be a new tuple with the specified items.

tuple2 = ("python","ddarkk","green")
print(tuple2[0:2])


# In[ ]:


# Specify negative indexes if you want to start the search from the end of the tuple:

tuple2 = ("python","ddarkk","green")
print(tuple2[-2:-1])


# In[ ]:


"""
Once a tuple is created, you cannot change its values. Tuples are unchangeable, or immutable as it also is called.
But there is a workaround. You can convert the tuple into a list, change the list, and convert the list back into a tuple.
"""

x = ("apple", "banana", "cherry")
y = list(x) #x convert to list
y[1] = "kiwi"
x = tuple(y)

print(x)


# In[ ]:




