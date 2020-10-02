#!/usr/bin/env python
# coding: utf-8

# # Larn Yersel Python
# ## Part 1 Variables and Data Structures

# Links to the full course:
# 
# [Part 1 Variables and Data Structures](https://www.kaggle.com/markthornber/lyp-part-1)
# 
# [Exercise 1](https://www.kaggle.com/markthornber/lyp-ex1)
# 
# [Part 2 Programming Constructs](https://www.kaggle.com/markthornber/lyp-part-2)
# 
# [Exercise 2](https://www.kaggle.com/markthornber/lyp-ex-2)
# 
# [Part 3 Functions and Input/Output](https://www.kaggle.com/markthornber/lyp-part-3)
# 
# [Exercise 3](https://www.kaggle.com/markthornber/lyp-ex3)
# 
# [Part 4 External Libraries](https://www.kaggle.com/markthornber/lyp-part-4)
# 
# [Notebook for your own graphs](https://www.kaggle.com/markthornber/lyp-exp)

# This course provides a brief introduction to programming in Python. It goes through the basic constructions and illustrates with simle examples. No attempt is made to give extensive lists of features, functions or external libraries. Interested readers can find lots of tutorials and references online. The aim is to get you started if you've never coded before or to get you up to speed with Python if you have coded in another language.

# In this notebook you should read all of the text boxes and then run the code in the following box by clicking in the box and then pressing the blue arrow that appears at the left. You can experiment by changing the code.

# 
# 
# Python is a programming language. It allows you to store data into variables (think of these as being like memories in your calculator). You can then do stuff with this data. You can use any name you like for a variable, with a few exceptions. Your name must not start with a number and it must not be one of Python's commands - you can't create a variable called "print" for example. 
# 
# To store some data in a variable you put the name first, then "=", then the data to store.
# 
# The simplest way to see the result of a piece of code is to use the `print` function:

# In[ ]:


number1 = 25
number2 = 17
answer = number1+number2
print(answer)


# Python is case sensitive and gives detailed error messages (EOF means "End Of File"):

# In[ ]:


number1 = 25
number2 = 17
answer = number1+number2
print(Answer)


# In[ ]:


number1 = 25
number2 = 17
answer = number1+number2
print(answer


# ![](http://)Python is more forgiving than most languages about the type of data stored in a variable, but it does distinguish between whole numbers and decimals, called "floats". Here are some simple examples with the `type` function used to reveal the type of variable.

# In[ ]:


number1 = 25
number2 = 12.3
answer1 = number1+3
print(answer1, type(answer1))
answer2 = number1+number2
print(answer2, type(answer2))
answer3 =number1+3.0
print(answer3, type(answer3))


# Notice that `number1` is an integer, but `number2` is a float. You can add different types together and Python automatically converts to the correct type for the answer. Can you see why `answer3` has to be a float?
# 
# Python has the usual arithmetical operations:
# addition, subtraction, multiplication and division: `+ , -, *, /`
# There is also raising to a power: `**`
# 
# Finally, there are a couple of useful functions that work on integers and give integer answers. These become very useful when we combine with other programming features.
# 
# Integer division, `//`, gives the whole number part of a division (the quotient)
# 
# Remainder, `%`, gives the remainder from a division)
# 
# Here are a few simple examples:

# In[ ]:


print('3+4*5=',3+4*5)
print('2**3=',2**3)
print('23/7=',23/7)
print('23//7=',23//7)
print('23%7=',23%7)


# Python obeys the usual rules of precedence and brackets. It doesn't understand implied multiplication, where two items are placed next to each other:

# In[ ]:


print((2+3)*(5-1))
print((2+3)(5-1))


# This error message looks a bit opaque! Python always tries to do what it is told. 
# 
# `(2+3)` is calculated and the result is an integer. The next thing Python sees is `(`. This is normally what is seen when using a function (like `print`). Functions are "callable" since Python "calls" the function and sends it the data in the brackets to do stuff with. 
# 
# The result of all of this is that Python thinks you want to use the integer`5` as a function and send it the result of the second bracket, `4`, to do stuff with. This causes the problem!

# Programming languages can also be used to process text - called *strings*. Text must be enclosed in quotes so that Python knows it is not a variable name. You can use single or double quotes, as long as you are consistent.

# In[ ]:


text1 = 'Hello'
text2 = "World"
print(text1)
print(text2)


# We can use `+` to combine text variables. Python will interpret this as concatenation, i.e. create a longer piece of text by putting the originals together one after the other. This is commonly used to construct different types of messages for a program to print out. Notice how the third variable contains an extra space at the end so the printout looks correct.

# In[ ]:


text1 = 'correct'
text2 = 'wrong'
messageStart = 'Your answer is '
print(messageStart+text1)
print(messageStart+text2)


# If you want to construct a message that includes a numerical variable you must first convert it to a string with the `str` function

# In[ ]:


number = (2+3**5)*(3-5)
message = messageStart+str(number)
print(message)


# In a long program you should give some thought to variable names. Just like a calculator memory, Python will remember the data long after you stopped using it. At the start of this Notebook I defined a variable `answer`. It hasn't been used since then. Do you remember what it contains? Python does:

# In[ ]:


print(answer)


# One consequence of this is that you should use easy to remember names for variables that will be used in multiple places. They should describe what is being stored. Since they must be a single word we often use upper case to make a phrase like `messageStart`.
# 
# On the other hand, if you're doing a quick calculation and will never use that data again, you can use single letter names for readability.

# In 1976 the computing pioneer Niklaus Wirth wrote a very influential book with the title *Algorithms + Data Structure = Programs*. As we will see, later, there are only a few simple types of programming constructs. The key to interesting results is how you combine them (the algorithms) and how the data is structured in memory. In most cases the structure of the data comes first. Getting this right makes it easier to process the data, making it easier to write a program. Python is designed to use external libraries, for example `pandas`. These provide data structures and functions that are constructed to be useful in particular areas. `pandas` is designed for handling data in tables called datasets. These can be imported from spreadsheets and other files or applications.
# 
# Python comes with some built-in data structures that can be used in all situations. The simplest of these is a list. These are written in square brackets. The individual data items can be of any type. They are separated by commas:

# In[ ]:


list1 = [1,2,3,4,5]
list2 = ['red','green','yellow']
print(list1)
print(list2)
print(list1+list2)


# As you can see, lists are added by concatenating, like strings. Python will interpret `+` by looking at the data objects on either side and acting accordingly. This ability to use the same symbol to mean different things in different contexts is called *overloading*. When reading a program you cannot try to understand individual parts in isolation. Context is everything.
# 
# To get at individual items in a list we use the position from the left (called the *index*). We start counting at `0`. Square brackets are used. For example, `mylist[3]` is the item in `mylist` at index number 3. This is the fourth one in from the start.

# In[ ]:


print(list2[0])
print(list2[2])
list2[1] = 'brown'
print(list2)


# Another common data structure is a tuple. These are very similar to lists, but cannot be changed (they are "immutable"). Python uses tuples a lot in internal stuff like sending a set of data values to a function. They are defined using round brackets. Indexing to get at individual items still uses square brackets.

# In[ ]:


tuple1 = ('red','green','blue')
print(tuple1)
print(tuple1[1])
tuple1[1] = 'brown'


# We will not use tuples much, but it's useful to know they exist. For example, Python interprets `print('Hello','World')` as "send the tuple `('Hello','World')` to the `print` function. The `print` function takes a tuple of data values and prints them out, one after the other, with spaces between them.

# In[ ]:


print('Hello','World')


# A final data structure that we see in lots of applications is a dictionary (called an "associative array" or "map" in some languages). This is a lookup table that lets you look up one item and replace it with another. Curly brackets are used and the data items are stored in the format "key:result".

# In[ ]:


dictionary1 = {'blue':14,'red':12,'green':7,'yellow':8}
print(dictionary1['green'])


# In this code, `dictionary1['green']` is an instruction to look for the key `'green'`in the dictionary. The result is `7` so that's what is printed out. 
# 
# Keys can be any type variable and so can the answers!

# In[ ]:


dictionary2 = {'red':1,'green':2,3:'yellow'}
print(dictionary2[3])


# In this case the keys are `'red','green',3` and the respective answers are `1,2,'yellow'`.
# 
# Things get interesting when we build a more complicated structure by combining simple ones. A basic example is to make a table as a list of lists:

# In[ ]:


table1 = [['First Name','Surname','Age','Height'], ['John','Smith',25,1.78],['Jane','Doe',19,1.72]]
row = 0
print(table1[row])
row = 2
print(table1[row])
column = 1
print(table1[row][column])


# Notice that `table1` is a actually a list. `table1[2]` gives the item in position 2 in the list. This is the sublist `['Jane','Doe',19,1.72]`, in other words, the second row of the table (row 0 is the title row). `table1[2][1]` first gets the sublist in position 2, i.e. row 2 of the table, and then looks for the item in position 1 within the sublist. This is column 1, the Surname column. The descriptive variable names help to make this clear.

# A final data structure that we have already met is a string. Any string variable can be thought of as a list of characters. Be careful! spaces and punctuation marks are all characters, they all count when using indexing to get at individual letters.

# In[ ]:


text = 'Hello World 42'
print(text[0])
print(text[6])
print(text[12])


# Notice that `text[12]` is the item in position 12. This looks like the number 4, but it's actually stored as a string. Arithmetic with numbers will lead to errors.

# In[ ]:


text[12]+3

