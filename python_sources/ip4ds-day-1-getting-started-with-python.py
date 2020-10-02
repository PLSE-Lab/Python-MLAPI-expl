#!/usr/bin/env python
# coding: utf-8

# * # Table of Contents
# - **<a href='#intro'>Introduction</a>**<br>
# - **<a href='#whoarewe'>Who Are We?</a>**<br>
# - **<a href='#helloworld'>Hello World - Getting Started</a>**<br>
# - **<a href='#if'>If Statements</a>**<br>
# - **<a href='#containers'>Lists, Dictionaries, and more...</a>**<br>
# - **<a href='#for'>For Loops</a>**<br>
# - **<a href='#methods'>Methods and Functions</a>**<br>
# - **<a href='#exceptions'>Errors, Exceptions, and Try/Catch</a>**<br>
# - **<a href='#import'>Importing and Numpy</a>**<br>
# - **<a href='#shakespeare'>Exercise I: All of Shakespeare's Works</a>**<br>
# - **<a href='#boolean'>Boolean Arrays and Numpy indexing</a>**<br>
# - **<a href='#classes'>Intro to Classes</a>**<br>
# - **<a href='#afternoon1'>Afternoon Project I: Seattle Weather</a>**<br>
# - **<a href='#afternoon2'>Afternoon Project 2: SF Crime</a>**<br>
# - **<a href='#afternoon3'>Afternoon Group Project 3</a>**<br>
# 
# 

# #What is Data Science?
# ![image.png](attachment:image.png)

# # Why Python? <a id='intro'></a>

# ### A killer combination of usability, versatility, and popularity.

# ![image.png](attachment:image.png)

# # Who Are We? <a id='whoarewe'></a>

# ## Instructors:
# ![image.png](attachment:image.png)
# ### Dillon Brout
# - @DillonBrout
# 
# - www.dillonbrout.com
# 
# - Ph.D. in Astrophysics/Cosmology this summer. 
# 
# - NASA Einstein Fellow
# 
# #### My Research: Measuring Dark Energy and the acceleration of the universe.
# 
# #### Data sciency stuff in my research: Search optimization, image analysis, parallel computing, statistical inference, machine learning, and more...
#  
# #### Outside of research: 
# 
# - 3rd Place at PennApps 2016! https://devpost.com/software/eyehud
# - Every year I run machine learning on March Madness data to predict the winner of the NCAA tournament. In the last 3 years I've come in the 100th, 98th and 100th percentiles 
# - I have worked for Bentley Motorsport as the race strategist.
# 
# #### Fun Fact: I like playing the saxophone!

# ![image.png](attachment:image.png) 
# ### Cyrille Doux
#  - UPenn Physics PostDoc working with Bhuvnesh Jain
#  - French
#  - Machine Learning Guru - works on ML blending of galaxies.

# ## TAs:
# ![image.png](attachment:image.png)
# ### Jeni Stiso
#  - Jeni is a fourth year grad student in Dani Bassett's lab studying computational neuroscience. 
#  - She majored in Molecular Biology as an undergraduate, and taught herself coding in mostly informal settings (like this one), and self guided online courses.
#  - You can reach her at jeni.stiso@gmail.com
# 

# ![image.png](attachment:image.png)
# ### Sara Casella
# - Phd Student Dept. of Economics

# # Distribution of specialties in this class.

# ![What%20is%20your%20department%20at%20Penn_.png](attachment:What%20is%20your%20department%20at%20Penn_.png)

# ## References
# - Official Jupyter documentation: https://jupyter-notebook.readthedocs.io/en/stable/
# - A nice tutorial of Jupyter: https://hub.packtpub.com/basics-jupyter-notebook-python/
# - Another nice tutorial: https://realpython.com/jupyter-notebook-introduction/
# - Markdown language guide: https://markdown-guide.readthedocs.io/en/latest/basics.html
# - **Markdown language cheat sheet: https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet

# Using the Jupyter notebook
# The Jupyter notebook allows you to write notebooks similar to e.g. Mathematica, SageMath, etc. You can include text, code, and plots in the same document and share it with others.
# 
# There are two main types of cells we'll use: code and markdown. Code cells are for using Python to do things, whereas markdown cells are used for styling text. Jupyter Notebook supports Markdown, which is a markup language that is a superset of HTML, and you can learn more about it through the references above.

# ## General Tips
# 
# - Google is your friend
# - Debugging may not seem intuitive at first, but stick with it and you will eventually 

# # Hello World <a id='helloworld'></a>

# In[ ]:


print('Hello World!')


# In[ ]:


x = 2
y = 4
print(x+y)


# ### Our first function!

# In[ ]:


#notation: INDENTATION MATTERS
def say_hello(recipient):
    print('Hello, ' + recipient)


# In[ ]:


say_hello('Jane')
say_hello('John')


# In[ ]:


#you can specify d
def say_hello(recipient='Default Name'):
    print('Hello, ' + recipient)


# In[ ]:


say_hello('Jane')
say_hello()


# In[ ]:


def double_me(some_input):
    the_double = some_input*2
    return the_double


# In[ ]:


result = double_me(10)
print(result)


# ### Python objects, basic types, and variables 
# 
# Everything in Python is an **object** and every object in Python has a **type**. Some of the basic types include:
# 
# - **`int`** (integer; a whole number with no decimal place)
#   - `10`
#   - `-3`
# - **`float`** (float; a number that has a decimal place)
#   - `7.41`
#   - `-0.006`
# - **`str`** (string; a sequence of characters enclosed in single quotes, double quotes, or triple quotes)
#   - `'this is a string using single quotes'`
#   - `"this is a string using double quotes"`
#   - `'''this is a triple quoted string using single quotes'''`
#   - `"""this is a triple quoted string using double quotes"""`
# - **`bool`** (boolean; a binary value that is either true or false)
#   - `True`
#   - `False`
# - **`NoneType`** (a special type representing the absence of a value)
#   - `None`
# 
# In Python, a **variable** is a name you specify in your code that maps to a particular **object**, object **instance**, or value.
# 
# By defining variables, we can refer to things by names that make sense to us. Names for variables can only contain letters, underscores (`_`), or numbers (no spaces, dashes, or other characters). Variable names must start with a letter or underscore.

# ### Basic operators
# 
# In Python, there are different types of **operators** (special symbols) that operate on different values. Some of the basic operators include:
# 
# - arithmetic operators
#   - **`+`** (addition)
#   - **`-`** (subtraction)
#   - **`*`** (multiplication)
#   - **`/`** (division)
#   - __`**`__ (exponent)
# - assignment operators
#   - **`=`** (assign a value)
#   - **`+=`** (add and re-assign; increment)
#   - **`-=`** (subtract and re-assign; decrement)
#   - **`*=`** (multiply and re-assign)
# - comparison operators (return either `True` or `False`)
#   - **`==`** (equal to)
#   - **`!=`** (not equal to)
#   - **`<`** (less than)
#   - **`<=`** (less than or equal to)
#   - **`>`** (greater than)
#   - **`>=`** (greater than or equal to)
# 
# When multiple operators are used in a single expression, **operator precedence** determines which parts of the expression are evaluated in which order. Operators with higher precedence are evaluated first (like PEMDAS in math). Operators with the same precedence are evaluated from left to right.
# 
# - `()` parentheses, for grouping
# - `**` exponent
# - `*`, `/` multiplication and division
# - `+`, `-` addition and subtraction
# - `==`, `!=`, `<`, `<=`, `>`, `>=` comparisons
# 
# > See https://docs.python.org/3/reference/expressions.html#operator-precedence
# 
# Examples:

# In[ ]:


x = 2
y = 4


# In[ ]:


x==3


# In[ ]:


# Assigning some numbers to different variables
num1 = 10
num2 = -3
num3 = 7.41
num4 = -.6
num5 = 7
num6 = 3
num7 = 11.11


# In[ ]:


# Addition
num1 + num2


# In[ ]:


# Subtraction
num2 - num3


# In[ ]:


# Multiplication
num3 * num4


# In[ ]:


# Division
num4 / num5


# In[ ]:


# Exponent
num5 ** num6


# In[ ]:


# Increment existing variable
print(num7)
num7 += 4
num7


# In[ ]:


# Decrement existing variable
num7 -= 2
num7


# In[ ]:


# Multiply & re-assign
num7 *= 5
num7


# In[ ]:


# Assign the value of an expression to a variable
num8 = num1 + num2 * num3
num8


# In[ ]:


# Are these two expressions equal to each other?
num1 + num2 == num5
print(num1 + num2)
num5


# In[ ]:


# Are these two expressions not equal to each other?
print(num3 != num4)
print(num3,num4)


# In[ ]:


# Is the first expression less than the second expression?
num5 < num6


# In[ ]:





# ### Python "if statements" <a id='if'></a>
# 
# The **if statement** allows you to test a condition and perform some actions if the condition evaluates to `True`. You can also provide `elif` and/or `else` clauses to an if statement to take alternative actions if the condition evaluates to `False`. 

# > **Example:** An `if` statement simple program. There can be zero or more `elif` parts, and the `else` part is optional. The keyword "`elif`" is short for "else if", and is useful to avoid excessive indentation.

# In[ ]:


x = -1
if x < 0:
    print('You entered a negative integer')
elif x == 0:
    print('You entered 0')
elif x == 1:
    print('You entered the integer 1')
else:
    print('You entered an integer greater than 1')


# In[ ]:


mystring = 'I love python data science bootcamp'
if 'python' in mystring:
    print('yeahhhh python')


# In[ ]:


'python' in mystring


# ## Exercise 1:
# Write a function called exercise1(miles) that converts miles to feet. There are 5280 feet in each mile. Make the program print out the following statement: "There are 10560 feet in 2 miles."

# In[ ]:


def exercise1(miles):
    nf = miles * 5280
    return 'There are ' + str(nf) + ' feet in ' + str(miles) + ' miles!'
    #do something and then return the result!


# In[ ]:


thenumber = exercise1(10)
print(thenumber)


# ## Exercise 2:
# Write a function called exercise2(age). This function should use if-elif-else statement to print out "Have a glass of milk." for anyone under 7; "Have a coke." for anyone under 21, and "Have a martini." for anyone 21 or older.
# 
# Tip: Be careful about the ages 7 (a seven year old is not under 7) and 21.

# In[ ]:


def exercise2(age):
    if age < 7:
        out = 'Have a glass of milk.'
    elif age < 21:
        out = 'Have a coke.'
    else:
        out = 'Have a martini.'
    
    return out


# In[ ]:


out = exercise2(20)


# In[ ]:


exercise2(50)


# ### Basic containers <a id='containers'></a>
# 
# > Note: **mutable** objects can be modified after creation and **immutable** objects cannot.
# 
# Containers are objects that can be used to group other objects together. The basic container types include:
# 
# - **`list`** (list: mutable; indexed by integers; items are stored in the order they were added)
#   - `[3, 5, 6, 3, 'dog', 'cat', False]`
# - **`tuple`** (tuple: immutable; indexed by integers; items are stored in the order they were added, **more memory efficient**)
#   - `(3, 5, 6, 3, 'dog', 'cat', False)`
# - **`set`** (set: mutable; not indexed at all; items are NOT stored in the order they were added; can only contain immutable objects; does NOT contain duplicate objects)
#   - `{3, 5, 6, 3, 'dog', 'cat', False}`
# - **`dict`** (dictionary: mutable; key-value pairs are indexed by immutable keys; items are NOT stored in the order they were added)
#   - `{'name': 'Jane', 'age': 23, 'fav_foods': ['pizza', 'fruit', 'fish']}`
# 
# When defining lists, tuples, or sets, use commas (,) to separate the individual items. When defining dicts, use a colon (:) to separate keys from values and commas (,) to separate the key-value pairs.
# 
# Strings, lists, and tuples are all **sequence types** that can use the `+`, `*`, `+=`, and `*=` operators.

# In[ ]:


# Assign some containers to different variables
list1 = [3, 5, 6, 3, 'dog', 'cat', False]
tuple1 = (3, 5, 6, 3, 'dog', 'cat', False)
set1 = {3, 5, 6, 3, 'dog', 'cat', False}
dict1 = {'name': 'Jane', 'age': 23, 'fav_foods': ['pizza', 'fruit', 'fish']}


# In[ ]:


# Items in the list object are stored in the order they were added
list1


# In[ ]:


# Items in the tuple object are stored in the order they were added
tuple1


# In[ ]:


# Items in the set object are not stored in the order they were added
# Also, notice that the value 3 only appears once in this set object
set1


# In[ ]:


# Items in the dict object are not stored in the order they were added
dict1
dict1['name']


# In[ ]:


#lists
mylist = [1, 4, 3, 2, 99, 100]


# In[ ]:


type(mylist)


# In[ ]:


mylist[3]


# In[ ]:


mylist[0:3]


# In[ ]:


mylist[:3]


# In[ ]:


mylist[0:-2]


# In[ ]:


mylist[::-1] #reverse a list


# In[ ]:


mylist[::2] #every other element


# ### Python "for loops" <a id='for'></a>
# 
# It is easy to **iterate** over a collection of items using a **for loop**. The strings, lists, tuples, sets, and dictionaries we defined are all **iterable** containers.
# 
# The for loop will go through the specified container, one item at a time, and provide a temporary variable for the current item. You can use this temporary variable like a normal variable.

# **Example:** A `for` loop program which prints each string in a list along with the number of letters in each string.

# In[ ]:


# Measure some strings:
words = ['dog', 'pennsylvania', 'summer'] # a container (list) named words consisting of 3 strings
for  word in words:  # i is each string in words in indexed order; cat is 0, window is 1, defenestrate is 2
    print(word, len(word))  # print each string and the number of letters in each string


# In[ ]:


names = ['Sam', 'Erin', 'Jane'] 
ages = [18, 21, 33]
for  name,age in zip(names,ages): # what is zip?
    print(name,'is',age,'years old') 


# **Example 2:** If you do need to iterate over a sequence of numbers, the built-in function `range()` comes in handy. It generates arithmetic progressions. Note: the given end point is **never** part of the generated sequence.

# In[ ]:


for i in range(5):  # from index 0 to 4
    print(i)


# In[ ]:


for myvar in mylist:
    print(myvar)


# In[ ]:


mylist = [1,2,3]
mylist.append(4)
print(mylist)


# In[ ]:


#tuples are immutabe
a = (1,2)
for i in a:
    a.append(1)


# ## Exercise 3:
# Write a function called exercise3(n) that adds up the numbers 1 through n and prints out the result. Be sure that you check your answer on several numbers n, taking care that your loop steps through all the numbers from 1 through and including n.

# In[ ]:


def exercise3(n):
    print(sum(range(n+1)))
    #do something
exercise3(14)


# In[ ]:


def exercise3_alt(n):
    out = 0
    for i in range(n):
        out += (i+1)
    
    return out

exercise3_alt(14)


# In[ ]:





# ## Exercise 4:
# Write a function called exercise4(month, day, year) that will convert a date such as June 3, 2019 into the format 6/3/2019. Use a dictionary to convert from the name of the month to the number of the month.

# In[ ]:


def exercise4(month,day,year):
    months = ('January','February','March','April','May','June','July','August','September','October','November','December')
    print(str(months.index(month)+1) + '/' + str(day) + '/' + str(year))
    #do something
exercise4('August',12, 2019)


# In[ ]:


def exercise4_alt(month,day,year):
    months = {'january':1,
              'february':2,
              'march':3,
              'april':4,
              'may':5,
              'june':6,
              'july':7,
              'august':8,
              'september':9,
              'october':10,
              'november':11,
              'december':12}
    print(str(months.lower(month)) + '/' + str(day) + '/' + str(year))
    #do something
exercise4('August',12, 2019)


# In[ ]:


# Iterate a dict
months = {'january':1,
              'february':2,
              'march':3,
              'april':4,
              'may':5,
              'june':6,
              'july':7,
              'august':8,
              'september':9,
              'october':10,
              'november':11,
              'december':12}
for key,value in months.items(): # months.items has two items (key and value), so we need two iterators to print each item
    print(key,value)


# In[ ]:


months['january']
#months.keys()
#months.values()


# ## Exercise 5:
# Write a function called exercise5(appendfruit,replacefruit) that will do the following:
# - print the whole fruitlist
# - print the first elemeent of the fruitlist
# - print the last element of the fruitlist
# - print the third through fifth elements of the fruitlist
# - print the first 3 elements of the fruitlist
# - replaces 'apple' by replacefruit
# - append appendfruit to the fruitlist. Print the new fruitlist. And return the new fruitlist

# In[ ]:


def exercise5(appendfruit, replacefruit):
    fruitlist = ['apple','bannana','orange','mango','strawberry','peach']
    print('First fruit: ' + str(fruitlist[0]))
    print('Last fruit: ' + str(fruitlist[-1]))
    print('3:5 fruits: ' + str(fruitlist[2:5]))
    print('First 3 fruits' + str(fruitlist[0:3]))
    
    if replacefruit:
        fruitlist[fruitlist.index('apple')] = replacefruit
    if appendfruit:
        fruitlist.append(appendfruit)
        
    print('New fruitlist: ' + str(fruitlist))
    
    
    #do something


# In[ ]:


fruitlist = ['apple','bannana','orange','mango','strawberry','peach']
fruitlist[-1]

fruitlist.index('apple')


# In[ ]:


mynewfruit = 'durian'
myreplacefruit = 'papaya'
exercise5(mynewfruit,myreplacefruit)


# In[ ]:


# add a single item
hd = [2,'grilled','relish','chili']
hd.append('ketchup')
print(hd)

# add multiple items
hd.extend(['ketchup','mayo','onion'])
print(hd)


# ### Python built-in functions and callables <a id='methods'></a>
# 
# A **function** is a Python object that you can "call" to **perform an action** or compute and **return another object**. You call a function by placing parentheses to the right of the function name. Some functions allow you to pass **arguments** inside the parentheses (separating multiple arguments with a comma). Internal to the function, these arguments are treated like variables.
# 
# Python has several useful built-in functions to help you work with different objects and/or your environment. Here is a small sample of them:
# 
# - **`type(obj)`** to determine the type of an object
# - **`len(container)`** to determine how many items are in a container
# - **`sum(container)`** to compute the sum of a container of numbers
# - **`min(container)`** to determine the smallest item in a container
# - **`max(container)`** to determine the largest item in a container
# - **`abs(number)`** to determine the absolute value of a number
# 
# > Complete list of built-in functions: https://docs.python.org/3/library/functions.html

# ### Python object attributes (methods and properties)
# 
# Different types of objects in Python have different **attributes** that can be referred to by name (similar to a variable). To access an attribute of an object, use a dot (`.`) after the object, then specify the attribute (i.e. `obj.attribute`)
# 
# When an attribute of an object is a callable, that attribute is called a **method**. It is the same as a function, only this function is bound to a particular object.
# 
# When an attribute of an object is not a callable, that attribute is called a **property**. It is just a piece of data about the object, that is itself another object.
# 
# The built-in `dir()` function can be used to return a list of an object's attributes.

# ### Some methods on string objects
# - **`.upper()`** to return an uppercase version of the string (all chars uppercase)
# - **`.lower()`** to return an lowercase version of the string (all chars lowercase)
# - **`.count(substring)`** to return the number of occurences of the substring in the string
# - **`.startswith(substring)`** to determine if the string starts with the substring
# - **`.endswith(substring)`** to determine if the string ends with the substring
# - **`.replace(old, new)`** to return a copy of the string with occurences of the "old" replaced by "new"
# - **`.split()`** to split the string into individual strings (delimited by spaces)

# In[ ]:


# Assign a string to a variable
a_string = 'tHis is a sTriNg'


# In[ ]:


# Return an uppercase version of the string
a_string.upper()


# In[ ]:


# Return a lowercase version of the string
a_string.lower()


# In[ ]:


# Notice that the methods called have not actually modified the string
a_string


# In[ ]:


# Count number of occurences of a substring in the string
a_string.count('i')


# In[ ]:


# Count number of occurences of a substring in the string
a_string.count('is')


# In[ ]:


# Does the string start with 'this'?
a_string.startswith('this')


# In[ ]:


# Does the lowercase string start with 'this'?
a_string.lower().startswith('this')


# In[ ]:


# Return a version of the string with a substring replaced with something else
a_string.replace('is', 'XYZ')


# In[ ]:


# Split the string into individual strings (delimited by spaces)
a_string.split()


# ### Some methods on list objects
# 
# - **`.append(item)`** to add a single item to the list
# - **`.extend([item1, item2, ...])`** to add multiple items to the list
# - **`.remove(item)`** to remove a single item from the list
# - **`.pop()`** to remove and return the item at the end of the list
# - **`.pop(index)`** to remove and return an item at an index

# In[ ]:


# Assign a list of common hotdog properties and ingredients to the variable hd
hd = [2, 'grilled', 'relish', 'chili']


# In[ ]:


# Add a single item to the list
hd.append('mustard')
hd


# In[ ]:


# Add multiple items to the list
hd.extend(['ketchup', 'mayo', 'onions'])
hd


# In[ ]:


# Remove a single item from the list
hd.remove('mayo')
hd


# In[ ]:


# Remove and return the last item in the list
hd.pop()
hd


# In[ ]:


# Remove and return an item at an index
hd.pop(4)
hd


# ### Some methods on dict objects
# 
# - **`.update([(key1, val1), (key2, val2), ...])`** to add multiple key-value pairs to the dict
# - **`.update(dict2)`** to add all keys and values from another dict to the dict
# - **`.pop(key)`** to remove key and return its value from the dict (error if key not found)
# - **`.get(key)`** to return the value at a specified key in the dict (or None if key not found)
# - **`.keys()`** to return a list of keys in the dict
# - **`.values()`** to return a list of values in the dict
# - **`.items()`** to return a list of key-value pairs (tuples) in the dict

# In[ ]:


# Assign keywords and their values to a dictionary called dict2
dict2 = {'name': 'Francie', 'age': 43, 'fav_foods': ['veggies', 'rice', 'pasta']}
dict2


# In[ ]:


# Add a single key-value pair fav_color purple to the dictionary dict2
dict2['fav_color'] = 'purple'
dict2


# In[ ]:


# Add all keys and values from another dict to dict2
dict3 = {'fav_animal': 'cat', 'phone': 'iPhone'}
dict2.update(dict3)
dict2


# In[ ]:


# Return a list of keys in the dict
dict2.keys()


# In[ ]:


# Return a list of values in the dict
dict2.values()


# In[ ]:


# Return a list of key-value pairs (tuples) in the dict 
dict2.items()


# In[ ]:


# Return a specific value of a key in dict2 that has multiple values
dict2['fav_foods'][2]  # veggies is 0, rice is 1, pasta is 2


# In[ ]:


# Remove key and return its value from dict2 (error if key not found)
dict2.pop('phone')
dict2


# In[ ]:





# ## Errors, Exceptions, and Try/Catch <a id='exceptions'></a>

# In[ ]:


print(newvariable)


# In[ ]:


a=10
b=5
c=2
print(a/(a-b*c))


# **Exceptions:**
# Exceptions occur during run-time. Your code may be syntactically correct but it may happen that during run-time Python encounters something which it can't handle, then it raises an exception. For example, dividing a number by zero or trying to write to a file which is read-only.
# When a Python script raises exception, it creates an Exception object. If the script doesn't handle exception the program will terminate abruptly.
# 
# - **Python Built-in Exceptions:** Python provides us some basic exception classes which are already defined and can be used in generic cases.
# 
#     + **Exception Class: Event**
# 
#     `Exception`: Base class for all exceptions
# 
#     `ArithmeticError`: Raised when numeric calculations fails
# 
#     `FloatingPointError`: Raised when a floating point calculation fails
# 
#     `ZeroDivisionError`: Raised when division or modulo by zero takes place for all numeric types
# 
#     `AssertionError`: Raised when Assert statement fails
# 
#     `OverflowError`: Raised when result of an arithmetic operation is too large to be represented
# 
#     `ImportError`: Raised when the imported module is not found
# 
#     `IndexError`: Raised when index of a sequence is out of range
# 
#     `KeyboardInterrupt`: Raised when the user interrupts program execution, generally by pressing Ctrl+c
# 
#     `IndentationError`: Raised when there is incorrect indentation
# 
#     `SyntaxError`: Raised by parser when syntax error is encountered
# 
#     `KeyError`: Raised when the specified key is not found in the dictionary
# 
#     `NameError`: Raised when an identifier is not found in the local or global namespace
# 
#     `TypeError`: Raised when a function or operation is applied to an object of incorrect type
# 
#     `ValueError`: Raised when a function gets argument of correct type but improper value
# 
#     `IOError`: Raised when an input/ output operation fails
# 
#     `RuntimeError`: Raised when a generated error does not fall into any category

# ### We should *"try"* to handle (*"except"*) these bad situations whenever possible

# In[ ]:


try:
    a=10
    b=5
    c=2
    result = a/(a-b*c)
    #print(a/(a-b*c))
except ZeroDivisionError:
    result = 0
    print('Whoops, you tried to divide by zero')

print(result)


# ## We can also *raise* exceptions when we think the user is doing something bad.

# In[ ]:


something_bad_is_happening = True
if something_bad_is_happening:
    raise ValueError("are you sure you want to be doing this?")


# ## Exercise 6:
# Write a function called exercise6() that computes the area of a trapezoid. Here is the formula: A = (1/2)(b1+b2)h. In the formula b1 is the length of one of the bases, b2 the other. The height is h and the area is A. Use input statements to ask for the bases and the height. Convert these input strings to real numbers using float().
# `
# 
# - Add try/except that will prevent the user from entering negative numbers.
# 
# - Raise an error if the user tries to make an empty trapezoid (b1 or b2 or h are zero).
# 
# - Use an output statement like this:
# 
# The area of a trapezoid with bases and and height is

# In[ ]:


def exercise6(b1,b2,h):
    
    if (b1 <= 0) or (b2 <= 0) or (h <= 0):
        raise ValueError('One of the inputs is <= 0, try again')
    else:
        A = .5*(b1+b2)*h
        return(A)
        
        
    #do something


# In[ ]:


print(exercise6(15,5,10))


# In[ ]:


exercise6(12.4,13.5,.01)


# In[ ]:


exercise6(14,-99,10)


# ### List Comprehension

# In[ ]:


#you've already learned for loops
squares = []             # creates an empty container (list) named "squares"
for x in range(10):      # for every number in the range from 0 to 9
    squares.append(x**2) # square that number and append it to the list called squares
squares                  # print the final form of the list

#joining strings from a list


# In[ ]:


#here is a more consice way of executing the same loop using list comprehension
squares = [x**2 for x in range(10)]  # define a list whose entries are the numbers 0 to 9 squared
squares


# ### Dictionary comprehension

# In[ ]:


# You can do the same thing with dictionaries for example
squares = {x: x**2 for x in (2, 4, 6)}
squares


# In[ ]:


list1 = [1,2,3,4]
list2 = ['a','b','c','d']
a = zip(list1,list2)

for x,y in a:
    print(x,y)


# In[ ]:


a = zip(list1,list2)
print(a.__next__())


# In[ ]:


print(a.__next__())


# ## Importing Libraries <a id='import'></a>

# In[ ]:


import numpy as np
import scipy
import matplotlib.pyplot as plt


# In[ ]:


#lets imagine you've done some internet research and 
#you find that there is a package called lmfit that solves your problem
import lmfit


# # pip: The Defacto Python Package Management sytem.

# In[ ]:


#the exclamation point means you're actually running shell code, we wont worry about that for now.
get_ipython().system('pip install lmfit ')


# In[ ]:


#Trying again...
import lmfit


# ### Lets focus on numpy
# 

# In[ ]:


#numpy is almost always imported like so (for convenience)
import numpy as np


# In[ ]:


mylist = ['apple',16,'banana']
myarray = np.array(mylist)
myarray


# ### Numpy arrays often have significant computation advantage over lists

# In[ ]:


[print(type(l)) for l in mylist]


# ![image.png](attachment:image.png)

# In[ ]:


def f1(x):
    return x**2 # x**2 is x"squared"

x = np.array([1,2,3,4]) #x data to integrate with
y = f1(x)
print(x)
print(y)


# In[ ]:


x = np.array([1,2,3,-3,-2,-1])
x[x<0]


# In[ ]:


# index using predefined index
I = np.array([0,1,2,3])
x[I]


# In[ ]:


str = np.array(['hello','my','name','is','chris'])
print([str=='chris'])


# In[ ]:


ages = np.array([15,92,2,19,33])
names = np.array(['Sally','Grandpa George','Baby Jack','Jane','Billy'])
print('Ages',np.sort(ages))
print('Indices',np.argsort(ages))
print('Names',names[np.argsort(ages)])


# ### Where statements

# In[ ]:


shakespeare = open('../input/shakespeare-plays/alllines.txt').readlines()


# In[ ]:


shakespeare


# In[ ]:


get_ipython().system('ls ../input/shakespeare-plays/')


# In[ ]:


len(shakespeare)


# In[ ]:


print(shakespeare[0])
print(shakespeare[1].strip())
shakespeare[0]


# In[ ]:


shakespeare[1].lower().strip().split()


# <a id='shakespeare'></a>

# ## Exercise I: Shakespeare 
# ### Determine the total number of words and the frequency of each word in the whole of Shakespeare's works.

# In[ ]:


# It will be useful to define a function that cleans up any string you pass it.
def split_and_clean_line(mystring):
    #input: string
    #output: a python list of lowercase words without punctuation
    
    a = mystring.lower().strip()
    
    # punctuation flags
    # punct = '!,.-_~`?@#$%^&*()[]{};:+\"'
    punct = ["'",'"','~','!','--','.',",",'-','?','`','@','#','$','%','^','&','(',')',"[","]",'{','}']
    
    # for each character, check against punctuation
    out = []
    for i in a:
        if i not in punct:
            out.append(i)
            
    # join the rest as a string
    alist = ''.join(out).split()
    
    return alist

print(split_and_clean_line(shakespeare[1]))
shakespeare[1]


# In[ ]:


words = {}

# for each line
for i in shakespeare:
    
    # split and clean that line, then for each word in the line:
    for w in split_and_clean_line(i):
        
        if w in words:
            # if in the dictionary already, iterate the count for that key
            words[w] += 1
        else:
            # if not, initialize the key for that word
            words[w] = 1

print(words)

#my_dict{line[1]:1}
#for i in line:
#    if i not in my_dict.keys():
#        my_dict(line[i]:1)


# In[ ]:


# now sort it
import operator
sorted_d = sorted(words.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_d)


# In[ ]:


print('This many different words!',len(words.keys()))


# In[ ]:


#Please print the 10 most frequent words and their respective counts
sorted_d[0:10]


# ### Now lets put it all into a single function that we can call later...

# In[ ]:


import operator
import numpy as np
import scipy
import matplotlib.pyplot as plt

def strip_shakespeare(fn):
    
    def split_and_clean_line(mystring):
        #input: string
        #output: a python list of lowercase words without punctuation

        a = mystring.lower().strip()

        # punctuation flags
        # punct = '!,.-_~`?@#$%^&*()[]{};:+\"'
        punct = ["'",'"','~','!','--','.',",",'-','?','`','@','#','$','%','^','&','(',')',"[","]",'{','}',':',';']

        # for each character, check against punctuation
        out = []
        for i in a:
            if i not in punct:
                out.append(i)
        
        # alternative
        # for p in punct:
        #     a = a.replace(p,'')
        # return a.lower().split()

        # join the rest as a string
        alist = ''.join(out).split()

        return alist
    
    shakespeare = open(fn).readlines()
    
    # dictionary
    words = {}

    # for each line
    for i in shakespeare:

        # split and clean that line, then for each word in the line:
        for w in split_and_clean_line(i):
            if len(w) > 5:
                if w in words:
                    # if in the dictionary already, iterate the count for that key
                    words[w] += 1
                else:
                    # if not, initialize the key for that word
                    words[w] = 1
                
    print('This many different words!',len(words.keys()))
    #sorted_d = sorted(words.items(), key=operator.itemgetter(1), reverse=True)
    #print(sorted_d[0:10])
    
    I = np.argsort(np.array(list(words.values())))[::-1]
    wList = np.array(list(words.keys()))[I]
    cList = np.array(list(words.values()))[I]
    print('-----')

    for i in range(10):
        print(wList[i],cList[i])
    
    #return words
    

# call fcn
fn = '../input/shakespeare-plays/alllines.txt'
strip_shakespeare(fn)
    
    


# # Exercise 2: Election Day Tweets

# In[ ]:


#tweets = open('../input/election-day-tweets/election_day_tweets.csv').readlines()
tweets = open('../input/election-day-tweets-txt/election_day_tweets.txt').readlines()
print(len(tweets))


# In[ ]:


fn = '../input/election-day-tweets-txt/election_day_tweets.txt'
strip_shakespeare(fn)


# # Boolean Arrays and Numpy Where

# In[ ]:


x = np.arange(8)
x < 5


# In[ ]:


A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
A | B


# In[ ]:


x = np.linspace(0,1,20)
print('original x',x)
condition = (x > .3) & (x < .7)
print('boolean array',condition)
print('just what we want',x[condition])


# <a id='classes'></a>

# # Classes
# ### This will be brief because you need to know about them but for Data Science and this course you most likely wont be making any of your own (but you'll be using lots of them!)

# What is a class?
# 
# Instance = class(arguments)

# snake = animal(length=10,width=1)
# 
# bunny = animal(length=2,width=2)

# In[ ]:


class animal:

    def __init__(self,l,w,h,name):
        self.name = name #PROPERITES
        self.belly = []  #PROPERITES
        self.length = l
        self.width = w
        self.heigh = h
    
    def eat(self, otheranimal): #FUNCTIONS
        self.belly.append(otheranimal.name)
        print('Yummmm',otheranimal.name,'was delicious')


# In[ ]:


snakey = animal(10,1,1,'sankeyMcSnakes')
mickeymouse = animal(2,3,1,'Mickey')
buggsbunny = animal(5,5,5,'BuggsBunny')


# In[ ]:


print('Animal Name:',snakey.name)
print('Stomach:',snakey.belly)


# In[ ]:


snakey.eat(mickeymouse)
print(snakey.belly)


# In[ ]:


snakey.eat(buggsbunny)
print(snakey.belly)


# <a id='afternoon1'></a>

# # Afternoon Project 1: Seattle Weather in 2014
# ### What are?
# - Number days without rain
# - Number days with rain
# - Days with more than 0.5 inches
# - Median precip on rainy days
# - Median precip on summer days
# - Maximum precip on summer days
# - Median precip on non-summer rainy days

# In[ ]:


data = np.genfromtxt('../input/c2numpy/Seattle2014.csv',names=True,delimiter=',')
print(data.dtype.names)
rainfall = data['PRCP']
inches = rainfall / 254.0  # 1/10mm -> inches
inches.shape


# In[ ]:


print(sum(data['PRCP']==0))
print(sum(data['PRCP']>0))
print(sum(data['PRCP']>.5))
print(np.median(data['PRCP'][data['PRCP']>0]))
plt.hist(data['PRCP'][data['PRCP']>0])


# In[ ]:


summer = []
for date in data['DATE']:
    month = int(np.array2string(date)[4:6])
    #day = int(np.array2string(date)[-1]
    summer.append((month >= 6) & (month <= 9))
print('ndays with no rain: ',sum(inches==0))
print('ndays with rain: ',sum(inches>0))
print('ndays with more than .5: ',sum(inches>.5))
print('median ran on rainy days: ',np.median(inches[inches>0]))
print('Max summer rain: ', np.max(inches[summer]))
print('Median summer rain: ',np.median(inches[summer]))
print('Median non-summer rain: ',np.median(inches[np.logical_not(summer)]))
plt.hist(inches[summer])
int(np.array2string(date)[-2])


# <a id='afternoon2'></a>

# # Afternoon Project 2: San Francisco Police Incidents
# 
# ### What is the total number of incidents for each year?
# ### What is the relative percentage of all incidents occurring on each day of the week?
# ### Which day of the week has the highest variance ('standard deviation') in number of incidents reported.
# ### Which hour of the day has the most incidents?
# ### Which hour of the day has the most "BURGLARY or ROBBERY" indicents (use *Category*)?
# ### Write a function that will take any arbitrary keyword and tell you the hour of the day that has the most indicents that match the keyword (use *Descript*).
# 
# 

# In[ ]:


get_ipython().system('ls ../input/sfpolice2016through2018')
get_ipython().system('head -10 ../input/sfpolice2016through2018/sf-police-incidents-short.csv')


# In[ ]:


import numpy as np
data = np.genfromtxt('../input/sfpolice2016through2018/sf-police-incidents-short.csv',names=True,delimiter=',',dtype=(int,int,'U30','U30','U30','U30','U10')) #NEED TO SPECIFY THE DATA TYPE (NOT SMART UNFORTUNATELY...)


# In[ ]:


data.dtype.names


# In[ ]:


data['Category']
np.shape(data)


# In[ ]:


from datetime import datetime
a = datetime.strptime(data['Date'][0],'%Y-%m-%dT%H:%M:%S.%f')


# In[ ]:


## incidents per year
years = []
date = []
for i in data['Date']:
    years.append(i[0:4])
    date.append(i[0:10])

years = np.array(years)
for year in np.unique(years):
    print('In ' + year + ' there were ' + str(sum(years == year)) + ' incedents')


# In[ ]:


## prct incidents per day
days = np.array(list(data['DayOfWeek']))
dayCount = []
for day in np.unique(days):
    dayCount.append(sum(days==day))
print(np.array(dayCount)/len(days))


# In[ ]:


## variability from week to week per day (need to find the number of events on a given day)
date = np.array(date)

#dates = np.unique(date[days == 'Friday'])

var = []
# for each weekday
for i in np.unique(days):
    
    print(i,len(dates))
    
    sumVar = []
    # for each unique date
    for j in np.unique(date[days == i]):
        
        # make a sum
        sumVar.append(sum(date == j))
        print(j,sum(date == j))
        
    var.append(np.std(sumVar))

    
        


# In[ ]:


print(var)


# <a id='afternoon3'></a>

# # Afternoon Project 3:
# ### Get into groups of 3 people
# - Take the person's code with the longest number of lines for Afternoon Project 2 
# - As a group, try to improve the code to be written in the fewest lines possible while still answering all of the questions
# - The group at the end with the fewest number of lines is the winner!

# In[ ]:


#new and improved code here


# In[ ]:





# ## Still want more? lets go through these
# https://www.datacamp.com/community/tutorials/18-most-common-python-list-questions-learn-python
