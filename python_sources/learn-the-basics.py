#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# * Python is open source
# * Python community
# * How to install python
# * What is foss
# * Notebook: Markdown vs. Code
# * Markdown syntax: https://support.squarespace.com/hc/en-us/articles/206543587-Markdown-cheat-sheet
# 

# # **Hello World**
# 
# Python is a very simple language, and has a very straightforward syntax. It encourages programmers to program without boilerplate (prepared) code. The simplest directive in Python is the **<span style="color:purple">print</span>** directive - it simply prints out a line (and also includes a newline, unlike in C).
# 

# ![](https://i.imgur.com/Ccy4Nll.jpg?1)

# There are two major Python versions, Python 2 and Python 3. Python 2 and 3 are quite different. This tutorial uses Python 3, because it more semantically correct and supports newer features.
# 
# For example, one difference between Python 2 and 3 is the **<span style="color:purple">print</span>** statement. In Python 2, the **<span style="color:purple">print</span>** statement is not a function, and therefore it is invoked without parentheses. However, in Python 3, it is a function, and must be invoked with parentheses.
# 
# https://www.learnpython.org/en/Hello%2C_World%21
# 
# To print a string in Python 3, just write:

# In[ ]:


print("This line will be printed.")


# But what about Hello World?

# In[ ]:


print("Hello World")


# ![](https://i.imgur.com/5RuCpWd.jpg)

# ## **Indentation**
# 
# Python uses indentation for blocks, instead of curly braces. Both tabs and spaces are supported, but the standard indentation requires standard Python code to use four spaces. For example:

# In[ ]:


x = 1
if x == 1:
    # Indented four spaces
    print("x is 1.")


# **Exercise**

# In[ ]:


# Modify below to print "Hello, World"
print("Goodbye, World!")


# ## **Comments**

# In[ ]:


# Comments are added to code to explain or add content to your code. They are ignored by the complier.
print ('Hello World!') # You can even add them to the end of a line. The code before the comment will still be executed.

"""
You
can
even
do
multi-line
docstrings
"""
# These are compiled but are just printed out as strings to the console


# # **Variables and Types**
# 
# Python is completely object oriented, and not "statically typed". You do not need to declare variables before using them, or declare their type. Every variable in Python is an object.
# 
# This tutorial will go over a few basic types of variables.

# ![image.png](attachment:image.png)

# ## **Numbers**

# Python supports two types of numbers - integers and floating point numbers. (It also supports complex numbers, which will not be explained in this tutorial).
# 
# To define an integer, use the following syntax:

# In[ ]:


myint = 7
print(myint)


# To define a floating point number, you may use one of the following notations:

# In[ ]:


myfloat = 7.0
print(myfloat)
myfloat = float(7)
print(myfloat)


# ## **Strings**
# 
# Strings are defined either with a single quote or a double quotes.

# In[ ]:


mystring = 'hello'
print(mystring)
mystring = "hello"
print(mystring)


# The difference between the two is that using double quotes makes it easy to include apostrophes (whereas these would terminate the string if using single quotes)

# In[ ]:


mystring = "Don't worry about apostrophes"
print(mystring)


# There are additional variations on defining strings that make it easier to include things such as carriage returns, backslashes and Unicode characters. These are beyond the scope of this tutorial, but are covered in the [Python documentation](https://docs.python.org/tutorial/introduction.html#strings).
# 
# Simple operators can be executed on numbers and strings:

# In[ ]:


one = 1
two = 2
three = one + two
print(three)

hello = "hello"
world = "world"
helloworld = hello + " " + world
print(helloworld)


# Assignments can be done on more than one variable "simultaneously" on the same line like this

# In[ ]:


a, b = 3, 4
print(a, b)


# Mixing operators between numbers and strings is not supported:

# In[ ]:


# This will not work!
one = 1
two = 2
hello = "hello"

print(one + two + hello)


# ### **Exercise**
# 
# The target of this exercise is to create a string, an integer, and a floating point number. The string should be named <span style="color:orangered">mystring</span> and should contain the word "hello". The floating point number should be named <span style="color:orangered">myfloat</span> and should contain the number 10.0, and the integer should be named <span style="color:orangered">myint</span> and should contain the number 20.

# In[ ]:


# Change this code
mystring = None
myfloat = None
myint = None

# Testing code
if mystring == "hello":
    print("String: %s" % mystring)
if isinstance(myfloat, float) and myfloat == 10.0:
    print("Float: %f" % myfloat)
if isinstance(myint, int) and myint == 20:
    print("Integer: %d" % myint)


# # **Lists**
# 
# Lists are very similar to arrays. They can contain any type of variable, and they can contain as many variables as you wish. Lists can also be iterated over in a very simple manner. Here is an example of how to build a list.

# ![](https://i.imgur.com/IfdEwo7.jpg)

# In[ ]:


mylist = []
mylist.append(1)
mylist.append(2)
mylist.append(3)
print(mylist[0]) # prints 1
print(mylist[1]) # prints 2
print(mylist[2]) # prints 3

# Prints out 1, 2, 3
for x in mylist:
    print(x)


# Accessing an index which does not exist generates an exception (an error).

# In[ ]:


mylist = [1, 2, 3]
print(mylist[10])


# ### **Useful list functions and methods**
# 
# There are some popular functions and methods used with lists.

# Function **<span style="color:purple">len</span>** outputs the lenght of a list.

# In[ ]:


list_len = len(mylist)
print(list_len)


# If a list consists of integer or real numbers, functions **<span style="color:purple">min</span>** and **<span style="color:purple">max</span>** find the smallest and largest numbers in the list, respectively.

# In[ ]:


list_min = min(mylist)
list_max = max(mylist)
print(list_min)
print(list_max)


# Function **<span style="color:purple">insert</span>** inserts an element at specified position.

# In[ ]:


mylist.insert(3, 2.5)
print(mylist)


# Function **<span style="color:purple">pop</span>** deletes one or more elements by using their indices.

# In[ ]:


mylist.pop(3)
print(mylist)


# Function **<span style="color:purple">remove</span>** deletes a specified element.

# In[ ]:


mylist.remove(2)
print(mylist)


# Function **<span style="color:purple">extend</span>** adds contents of the second list to the end of the first list.

# In[ ]:


mylist.extend([6, 7, 8])
print(mylist)


# Function **<span style="color:purple">sum</span>** calculates the sum of all the elements of a list.

# In[ ]:


list_sum = sum(mylist)
print(list_sum)


# Function **<span style="color:purple">sort</span>** sorts a list in either accending or descending order.

# In[ ]:


print(mylist)
mylist.sort(reverse=True)
print(mylist)
mylist.sort(reverse=False)
print(mylist)


# ### **Excercise**
# 
# In this exercise, you will need to add numbers and strings to the correct lists using the **<span style="color:purple">append</span>** list method. You must add the numbers 1,2, and 3 to the <span style="color:orangered">numbers</span> list, and the words <span style="color:green">'hello'</span> and <span style="color:green">'world'</span> to the strings variable.
# 
# You will also have to fill in the variable <span style="color:orangered">second_name</span> with the second name in the names list, using the brackets operator **<span style="color:purple">[]</span>**. Note that the index is zero-based, so if you want to access the second item in the list, its index will be 1.

# In[ ]:


numbers = []
strings = []
names = ["John", "Eric", "Jessica"]

# Write your code here
second_name = None


# This code should write out the filled arrays and the second name in the names list (Eric).
print(numbers)
print(strings)
print("The second name on the names list is %s" % second_name)


# # **Basic Operators**
# 
# This section explains how to use basic operators in Python.

# ![](https://i.imgur.com/r0fay3y.jpg)

# ## **Arithmetic Operators**
# 
# Just as any other programming languages, the addition, subtraction, multiplication, and division operators can be used with numbers.

# In[ ]:


number = 1 + 2 * 3 / 4.0
print(number)


# Be careful while using the division **<span style="color:purple">/</span>** operator . Some versions of Python implement integer division when applied to two integer numbers. In order to perform real division, specify one of the numbers as float. If you want to make sure integer division is performed, use **<span style="color:purple">//</span>** instead of **<span style="color:purple">/</span>**.

# In[ ]:


int_div = 5 // 3
real_div = 5.0 / 3
print(int_div)
print(real_div)


# Try to predict what the answer will be. Does python follow order of operations?
# 
# Another operator available is the modulo **<span style="color:purple">%</span>** operator, which returns the integer remainder of the division. Recall that <span style="color:red">dividend % divisor = remainder</span>.

# In[ ]:


remainder = 11 % 3
print(remainder)


# Using two multiplication symbols makes a power relationship.

# In[ ]:


squared = 7 ** 2
cubed = 2 ** 3
print(squared)
print(cubed)


# ## **Using Operators with Strings**
# 
# Python supports concatenating strings using the addition operator **<span style="color:purple">+</span>**:

# In[ ]:


helloworld = "hello" + " " + "world"
print(helloworld)


# Python also supports multiplying strings to form a string with a repeating sequence:

# In[ ]:


lotsofhellos = "hello" * 10
print(lotsofhellos)


# ## **Using Operators with Lists**
# 
# Lists can be joined with the addition operators:

# In[ ]:


even_numbers = [2, 4, 6, 8]
odd_numbers = [1, 3, 5, 7]
all_numbers = odd_numbers + even_numbers
print(all_numbers)


# Just as in strings, Python supports forming new lists with a repeating sequence using the multiplication operator:

# In[ ]:


print([1, 2, 3] * 3)


# ### **Exercise**
# 
# The target of this exercise is to create two lists called <span style="color:orangered">x_list</span> and <span style="color:orangered">y_list</span>, which contain 10 instances of the variables <span style="color:orangered">x</span> and <span style="color:orangered">y</span>, respectively. You are also required to create a list called <span style="color:orangered">big_list</span>, which contains the variables <span style="color:orangered">x</span> and <span style="color:orangered">y</span>, 10 times each, by concatenating the two lists you have created.

# In[ ]:


x = object()
y = object()

# TODO: change this code
x_list = [x]
y_list = [y]
big_list = []

print("x_list contains %d objects" % len(x_list))
print("y_list contains %d objects" % len(y_list))
print("big_list contains %d objects" % len(big_list))

# Testing code
if x_list.count(x) == 10 and y_list.count(y) == 10:
    print("Almost there...")
if big_list.count(x) == 10 and big_list.count(y) == 10:
    print("Great!")


# # **String Formatting**
# 
# Python uses C-style string formatting to create new, formatted strings. The **<span style="color:purple">%</span>** operator is used to format a set of variables enclosed in a <span style="color:red">tuple</span> (a fixed size list), together with a format string, which contains normal text together with "argument specifiers", special symbols like **<span style="color:purple">%s</span>** and **<span style="color:purple">%d</span>**.

# ![](https://i.imgur.com/CWrPWdb.jpg)

# Let's say you have a variable called <span style="color:orangered">name</span> with your user name in it, and you would then like to print out a greeting to that user.

# In[ ]:


# This prints out "Hello, John!"
name = "John"
print("Hello, %s!" % name)


# To use two or more argument specifiers, use a tuple (parentheses):

# In[ ]:


# This prints out "John is 23 years old."
name = "John"
age = 23
print("%s is %d years old." % (name, age))
print("%s is %03d years old." % (name, age))
print("%s is %3d years old." % (name, age))
print("%s is %2.2f years old." % (name, age))


# Any object which is not a string can be formatted using the **<span style="color:purple">%s</span>** operator as well. The string which returns from the **<span style="color:purple">repr</span>** method of that object is formatted as the string. For example:

# In[ ]:


# This prints out: A list: [1, 2, 3]
mylist = [1, 2, 3]
print("A list: %s" % mylist)


# Here are some basic argument specifiers you should know:
# 
# <span style="color:purple">%s - string (or any object with a string representation, like numbers)</span>
# 
# <span style="color:purple">%d - integers</span>
# 
# <span style="color:purple">%f - floating point numbers</span>
# 
# <span style="color:purple">%.[number of digits]f - floating point numbers with a fixed amount of digits to the right of the dot.</span>
# 
# <span style="color:purple">%x/%X - integers in hex representation (lowercase/uppercase)</span>

# There is another way to format strings in Python. Using the **<span style="color:purple">format</span>** method of the string class is an alternative to the **<span style="color:purple">%</span>** operator. Below you can find out how the same tasks can be implemented using the **<span style="color:purple">format</span>** method.

# ![](https://i.imgur.com/XpVb37e.jpg)

# In[ ]:


# This prints out "Hello, John!"
name = "John"
print("Hello, {}!".format(name))


# In[ ]:


# This prints out "John is 23 years old."
name = "John"
age = 23
print("{0} is {1:0=2d} years old.".format(name, age))
print("{0} is {1:0=3d} years old.".format(name, age))
print("{0} is {1:=3d} years old.".format(name, age))
print("{0} is {1:=2.2f} years old.".format(name, age))


# In[ ]:


# This prints out: A list: [1, 2, 3]
mylist = [1, 2, 3]
print("A list: {}".format(mylist))


# ### **Exercise**
# 
# You will need to write a format string which prints out the data using the following syntax: <span style="color:red">Hello John Doe. Your current balance is $53.44</span>.

# In[ ]:


data = ("John", "Doe", 53.44)
format_string = "Hello"

print(format_string % data)


# # **Basic String Operations**

# ![](https://i.imgur.com/ToBWS0Q.jpg)

# Strings are bits of text. They can be defined as anything between quotes:

# In[ ]:


astring = "Hello world!"
astring2 = 'Hello world!'


# As you can see, the first thing you learned was printing a simple sentence. This sentence was stored by Python as a string. However, instead of immediately printing strings out, we will explore the various things you can do to them. You can also use single quotes to assign a string. However, you will face problems if the value to be assigned itself contains single quotes. For example, to assign the string in these bracket (single quotes are <span style="color:purple">**' '**</span>) you need to use double quotes only like this

# In[ ]:


astring = "Hello world!"
print("single quotes are ' '")

print(len(astring))


# That prints out 12, because <span style="color:green">"Hello world!"</span> is 12 characters long, including punctuation and spaces.

# In[ ]:


astring = "Hello world!"
print(astring.index("o"))


# That prints out 4, because the location of the first occurrence of the letter <span style="color:green">o</span> is 4 characters away from the first character. Notice how there are actually two <span style="color:green">o</span>'s in the phrase - this method only recognizes the first.
# 
# But why didn't it print out 5? Isn't <span style="color:green">o</span> the fifth character in the string? To make things more simple, Python (and most other programming languages) start things at 0 instead of 1. So the index of <span style="color:green">o</span> is 4.

# In[ ]:


astring = "Hello world!"
print(astring.count("l"))


# For those of you using silly fonts, that is a lowercase <span style="color:green">L</span>, not a number one. This counts the number of <span style="color:green">l</span>'s in the string. Therefore, it should print 3.

# In[ ]:


astring = "Hello world!"
print(astring[3:7])


# This prints a slice of the string, starting at index 3, and ending at index 6. But why 6 and not 7? Again, most programming languages do this - it makes doing math inside those brackets easier.
# 
# If you just have one number in the brackets, it will give you the single character at that index. If you leave out the first number but keep the colon, it will give you a slice from the start to the number you left in. If you leave out the second number, it will give you a slice from the first number to the end.
# 
# You can even put negative numbers inside the brackets. They are an easy way of starting at the end of the string instead of the beginning. This way, -3 means "3rd character from the end".

# In[ ]:


astring = "Hello world!"
print(astring[3:7:2])


# This prints the characters of string from 3 to 7 skipping one character. This is extended slice syntax. The general form is <span style="color:orangered">[start:stop:step]</span>.

# In[ ]:


astring = "Hello world!"
print(astring[3:7])
print(astring[3:7:1])


# Note that both of them produce same output
# 
# There is no function like **<span style="color:purple">strrev</span>** in C to reverse a string. But with the above mentioned type of slice syntax you can easily reverse a string like this

# In[ ]:


astring = "Hello world!"
print(astring[::-1])


# These make a new string with all letters converted to uppercase and lowercase, respectively.

# In[ ]:


astring = "Hello world!"
print(astring.upper())
print(astring.lower())


# This is used to determine whether the string starts with something or ends with something, respectively. The first one will print <span style="color:blue">True</span>, as the string starts with <span style="color:green">"Hello"</span>. The second one will print <span style="color:blue">False</span>, as the string certainly does not end with <span style="color:green">"asdfasdfasdf"</span>.

# In[ ]:


astring = "Hello world!"
print(astring.startswith("Hello"))
print(astring.endswith("asdfasdfasdf"))


# This splits the string into a bunch of strings grouped together in a list. Since this example splits at a space, the first item in the list will be <span style="color:green">"Hello"</span>, and the second will be <span style="color:green">"world!"</span>.

# In[ ]:


astring = "Hello world!"
afewwords = astring.split(" ")


# ### **Exercise**
# 
# Try to fix the code to print out the correct information by changing the string.

# In[ ]:


s = "Hey there! what should this string be?"
# Length should be 20
print("Length of s = %d" % len(s))

# First occurrence of "a" should be at index 8
print("The first occurrence of the letter a = %d" % s.index("a"))

# Number of a's should be 2
print("a occurs %d times" % s.count("a"))

# Slicing the string into bits
print("The first five characters are '%s'" % s[:5]) # Start to 5
print("The next five characters are '%s'" % s[5:10]) # 5 to 10
print("The thirteenth character is '%s'" % s[12]) # Just number 12
print("The characters with odd index are '%s'" % s[1::2]) #(0-based indexing)
print("The last five characters are '%s'" % s[-5:]) # 5th-from-last to end

# Convert everything to uppercase
print("String in uppercase: %s" % s.upper())

# Convert everything to lowercase
print("String in lowercase: %s" % s.lower())

# Check how a string starts
if s.startswith("Str"):
    print("String starts with 'Str'. Good!")

# Check how a string ends
if s.endswith("ome!"):
    print("String ends with 'ome!'. Good!")

# Split the string into three separate strings, each containing only a word
print("Split the words of the string: %s" % s.split(" "))

# Reverse the string
s_list = list(s)
s_reversed = ''.join(s_list[::-1])
print("Reversed string: %s" % s_reversed)


# # **Conditions**
# 

# ![](https://i.imgur.com/RsXL5vw.jpg)

# Python uses boolean variables to evaluate conditions. The boolean values <span style="color:blue">True</span> and <span style="color:blue">False</span> are returned when an expression is compared or evaluated. For example:

# In[ ]:


x = 2
print(x == 2) # prints out True
print(x == 3) # prints out False
print(x < 3) # prints out True


# Notice that variable assignment is done using a single equals operator **<span style="color:purple">=</span>**, whereas comparison between two variables is done using the double equals operator **<span style="color:purple">==</span>**. The "not equals" operator is marked as **<span style="color:purple">!=</span>**.
# 
# ## **Boolean operators**
# 
# The **<span style="color:purple">and</span>** and **<span style="color:purple">or</span>** boolean operators allow building complex boolean expressions, for example:

# In[ ]:


name = "John"
age = 23
if name == "John" and age == 23:
    print("Your name is John, and you are also 23 years old.")

if name == "John" or name == "Rick":
    print("Your name is either John or Rick.")
    
if type(name) == str:
    print ("It is a string!")
    
if isinstance(age,int):
    print ("It is an int!")


# ## **The <span style="color:purple">in</span> operator**
# 
# The **<span style="color:purple">in</span>** operator could be used to check if a specified object exists within an iterable object container, such as a list:

# In[ ]:


name = "John"
if name in ["John", "Rick"]:
    print("Your name is either John or Rick.")


# Python uses indentation to define code blocks, instead of brackets. The standard Python indentation is 4 spaces, although tabs and any other space size will work, as long as it is consistent. Notice that code blocks do not need any termination.
# 
# Here is an example for using Python's **<span style="color:purple">if</span>** statement using code blocks:

# In[ ]:


statement = False
another_statement = True
if statement is True:
    # Do something
    pass
elif another_statement is True: # else if
    # Do something else
    pass
else:
    # Do another thing
    pass


# For example:

# In[ ]:


x = 2
if x == 2:
    print("x equals two!")
else:
    print("x does not equal to two.")


# A statement is evaluated as true if one of the following is correct: 
# 1. The <span style="color:blue">True</span> boolean variable is given, or calculated using an expression, such as an arithmetic comparison. 
# 2. An object which is not considered "empty" is passed.
# 
# Here are some examples for objects which are considered as empty: 
# 1. An empty string: <span style="color:green">""</span> 
# 2. An empty list: <span style="color:orangered">[]</span> 
# 3. The number zero: <span style="color:orangered">0</span> 
# 4. The false boolean variable: <span style="color:blue">False</span>
# 
# ## **The <span style="color:purple">is</span> operator**
# 
# Unlike the double equals operator **<span style="color:purple">==</span>**, the **<span style="color:purple">is</span>** operator does not match the values of the variables, but the instances themselves. For example:

# In[ ]:


x = [1, 2, 3]
y = [1, 2, 3]
print(x == y) # prints out True
print(x is y) # prints out False


# ## **The <span style="color:purple">not</span> operator**
# 
# Using **<span style="color:purple">not</span>** before a boolean expression inverts it:

# In[ ]:


print(not False) # prints out True
print((not False) == (False)) # prints out False


# ### **Exercise**
# 
# Change the variables in the first section, so that each **<span style="color:purple">if</span>** statement resolves as <span style="color:blue">True</span>.

# In[ ]:


# Change this code
number = 10
second_number = 10
first_array = []
second_array = [1, 2, 3]

if number > 15:
    print("1")

if first_array:
    print("2")

if len(second_array) == 2:
    print("3")

if len(first_array) + len(second_array) == 5:
    print("4")

if first_array and first_array[0] == 1:
    print("5")

if not second_number:
    print("6")


# # **Loops**
# 
# There are two types of loops in Python, **<span style="color:purple">for</span>** and **<span style="color:purple">while</span>**.

# ## **The **<span style="color:purple">for</span>** loop**

# ![](https://i.imgur.com/CBvBkQm.jpg)

# The **<span style="color:purple">for</span>** loops iterate over a given sequence. Here is an example:

# In[ ]:


primes = [2, 3, 5, 7]
for prime in primes:
    print(prime)


# The **<span style="color:purple">for</span>** loops can iterate over a sequence of numbers using the **<span style="color:purple">range</span>** and **<span style="color:purple">xrange</span>** functions. The difference between **<span style="color:purple">range</span>** and **<span style="color:purple">xrange</span>** is that the **<span style="color:purple">range</span>** function returns a new list with numbers of that specified range, whereas **<span style="color:purple">xrange</span>** returns an iterator, which is more efficient (Python 3 uses the **<span style="color:purple">range</span>** function, which acts like **<span style="color:purple">xrange</span>**). Note that the **<span style="color:purple">range</span>** function is zero based.

# In[ ]:


# Prints out the numbers 0, 1, 2, 3, 4
for x in range(5):
    print(x)

# Prints out 3, 4, 5
for x in range(3, 6):
    print(x)

# Prints out 3, 5, 7
for x in range(3, 8, 2):
    print(x)


# ## **The <span style="color:purple">while</span> loop**

# ![](https://i.imgur.com/um3q7oa.jpg)

# A **<span style="color:purple">while</span>** loop repeats as long as a certain boolean condition is met. For example:

# In[ ]:


# Prints out 0, 1, 2, 3, 4

count = 0
while count < 5:
    print(count)
    count += 1  # this is the same as count = count + 1


# ## **The <span style="color:purple">break</span> and <span style="color:purple">continue</span> statements**
# 
# The **<span style="color:purple">break</span>** command is used to exit a **<span style="color:purple">for</span>** loop or a **<span style="color:purple">while</span>** loop, whereas **<span style="color:purple">continue</span>** is used to skip the current block, and return to the **<span style="color:purple">for</span>** or **<span style="color:purple">while</span>** statement. A few examples:

# In[ ]:


# Prints out 0, 1, 2, 3, 4

count = 0
while True:
    print(count)
    count += 1
    if count >= 5:
        break

# Prints out only odd numbers - 1, 3, 5, 7, 9
for x in range(10):
    # Check if x is even
    if x % 2 == 0:
        continue
    print(x)


# ## **Can we use an <span style="color:purple">else</span> clause for loops?**
# 
# Unlike languages like C, CPP, etc. we can use **<span style="color:purple">else</span>** for loops. When the loop condition of a **<span style="color:purple">for</span>** or **<span style="color:purple">while</span>** statement fails then code part in **<span style="color:purple">else</span>** is executed. If a **<span style="color:purple">break</span>** statement is executed inside the **<span style="color:purple">for</span>** loop then the **<span style="color:purple">else</span>** part is skipped. Note that the **<span style="color:purple">else</span>** part is executed even if there is a **<span style="color:purple">continue</span>** statement.
# 
# Here are a few examples:

# In[ ]:


# Prints out 0, 1, 2, 3, 4 and then it prints "count value reached 5"

count = 0
while(count < 5):
    print(count)
    count +=1
else:
    print("count value reached %d" % count)

# Prints out 1, 2, 3, 4
for i in range(1, 10):
    if(i % 5 == 0):
        break
    print(i)
else:
    print("this is not printed because for loop is terminated because of break but not due to fail in condition")


# ### **Exercise**
# 
# Loop through and print out all even numbers from the <span style="color:orangered">numbers</span> list in the same order they are received. Don't print any numbers that come after <span style="color:orangered">237</span> in the sequence.

# In[ ]:


numbers = [
    951, 402, 984, 651, 360, 69, 408, 319, 601, 485, 980, 507, 725, 547, 544,
    615, 83, 165, 141, 501, 263, 617, 865, 575, 219, 390, 984, 592, 236, 105, 942, 941,
    386, 462, 47, 418, 907, 344, 236, 375, 823, 566, 597, 978, 328, 615, 953, 345,
    399, 162, 758, 219, 918, 237, 412, 566, 826, 248, 866, 950, 626, 949, 687, 217,
    815, 67, 104, 58, 512, 24, 892, 894, 767, 553, 81, 379, 843, 831, 445, 742, 717,
    958, 609, 842, 451, 688, 753, 854, 685, 93, 857, 440, 380, 126, 721, 328, 753, 470,
    743, 527
]

# Your code goes here


# # **Functions**
# 
# ## **What are Functions?**
# 
# Functions are a convenient way to divide your code into useful blocks, allowing us to order our code, make it more readable, reuse it and save some time. Also functions are a key way to define interfaces, so programmers can share their code.

# ![](https://i.imgur.com/LTMEw7x.jpg)

# ## **How do you write functions in Python?**
# 
# As we have seen on previous tutorials, Python makes use of blocks.
# 
# A block is an area of code written in the format of:

# In[ ]:


block_head:
    1st block line
    2nd block line
    ...


# Every block line consists of more Python code (even another block), and the block head is of the following format: <span style="color:red">block_keyword block_name(argument1, argument2, ...)</span>. Block keywords you already know are **<span style="color:purple">if</span>**, **<span style="color:purple">for</span>**, and **<span style="color:purple">while</span>**.
# 
# Functions in python are defined using the block keyword **<span style="color:purple">def</span>**, followed with the function's name as the block's name. For example:

# In[ ]:


def my_function():
    print("Hello From My Function!")


# Functions may also receive arguments (variables passed from the caller to the function). For example:

# In[ ]:


def my_function_with_args(username, greeting):
    print("Hello, %s , From My Function!, I wish you %s" % (username, greeting))


# Functions may return a value to the caller, using the keyword **<span style="color:purple">return</span>** . For example:

# In[ ]:


def sum_two_numbers(a, b):
    return a + b


# ## **How do you call functions in Python?**
# 
# Simply write the function's name followed by **<span style="color:purple">()</span>**, placing any required arguments within the brackets. For example, let's call the functions written above (in the previous example):

# In[ ]:


# Define our 3 functions
def my_function():
    print("Hello From My Function!")

def my_function_with_args(username, greeting):
    print("Hello, %s, From My Function! I wish you %s" % (username, greeting))

def sum_two_numbers(a, b):
    return a + b

# Prints a simple greeting
my_function()

# Prints "Hello, John Doe, From My Function!, I wish you a great year!"
my_function_with_args("John Doe", "a great year!")

# After this line x will hold the value 3!
x = sum_two_numbers(1, 2)


# ### **Exercise**
# 
# In this exercise you'll use an existing function, and while adding your own to create a fully functional program.
# 
# Add a function named <span style="color:orangered">list_benefits</span> that returns the following list of strings: <span style="color:green">"More organized code"</span>, <span style="color:green">"More readable code"</span>, <span style="color:green">"Easier code reuse"</span>, <span style="color:green">"Allowing programmers to share and connect code together"</span>.
# 
# Add a function named <span style="color:orangered">build_sentence</span> which receives a single argument containing a string and returns a sentence starting with the given string and ending with the string <span style="color:green">" is a benefit of functions!"</span>.
# 
# Run and see all the functions work together!

# In[ ]:


# Modify this function to return a list of strings as defined above
def list_benefits():
    pass

# Modify this function to concatenate to each benefit - " is a benefit of functions!"
def build_sentence(benefit):
    pass

def name_the_benefits_of_functions():
    list_of_benefits = list_benefits()
    for benefit in list_of_benefits:
        print(build_sentence(benefit))

name_the_benefits_of_functions()


# # **Classes and Objects**
# 
# Objects are an encapsulation of variables and functions into a single entity. Objects get their variables and functions from classes. Classes are essentially a template to create your objects.
# 

# ![](https://i.imgur.com/sVpMK95.jpg)

# A very basic class would look something like this:

# In[ ]:


class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")


# We'll explain why you have to include that **<span style="color:purple">self</span>** as a parameter a little bit later. First, to assign the above class (template) to an object you would do the following:

# In[ ]:


class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()


# Now the variable <span style="color:orangered">myobjectx</span> holds an object of the class <span style="color:navy">MyClass</span> that contains the variable and the function defined within the class called <span style="color:navy">MyClass</span>.
# 
# ## **Accessing Object Variables**
# 
# To access the variable inside of the newly created object <span style="color:orangered">myobjectx</span> you would do the following:

# In[ ]:


class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()

myobjectx.variable


# So for instance the below would output the string <span style="color:green">"blah"</span>:

# In[ ]:


class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()

print(myobjectx.variable)


# You can create multiple different objects that are of the same class (have the same variables and functions defined). However, each object contains independent copies of the variables defined in the class. For instance, if we were to define another object with the <span style="color:navy">MyClass</span> class and then change the string in the variable above:

# In[ ]:


class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()
myobjecty = MyClass()

myobjecty.variable = "yackity"

# Then print out both values
print(myobjectx.variable)
print(myobjecty.variable)


# ## **Accessing Object Functions**
# 
# To access a function inside of an object you use notation similar to accessing a variable:

# In[ ]:


class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()

myobjectx.function()


# The above would print out the message <span style="color:green">"This is a message inside the class."</span>
# 
# ### **Exercise**
# 
# We have a class defined for vehicles. Create two new vehicles called <span style="color:orangered">car1</span> and <span style="color:orangered">car2</span>. Set <span style="color:orangered">car1</span> to be a red convertible worth <span style="color:red">\$60,000.00</span> with a name of <span style="color:red">Fer</span>, and <span style="color:orangered">car2</span> to be a blue van named <span style="color:blue">Jump</span> worth <span style="color:blue">\$10,000.00</span>.

# In[ ]:


# Define the Vehicle class
class Vehicle:
    name = ""
    kind = "car"
    color = ""
    value = 100.00
    def description(self):
        desc_str = "%s is a %s %s worth $%.2f." % (self.name, self.color, self.kind, self.value)
        return desc_str
# Your code goes here

# Test code
print(car1.description())
print(car2.description())


# # **Dictionaries**
# 
# A dictionary is a data type similar to arrays, but works with keys and values instead of indices. Each value stored in a dictionary can be accessed using a key, which is any type of object (a string, a number, a list, etc.) instead of using its index to address it.

# ![](https://i.imgur.com/DgFQmF3.jpg)

# For example, a database of phone numbers could be stored using a dictionary like this:

# In[ ]:


phonebook = {}
phonebook["John"] = 938477566
phonebook["Jack"] = 938377264
phonebook["Jill"] = 947662781
print(phonebook)


# Alternatively, a dictionary can be initialized with the same values in the following notation:

# In[ ]:


phonebook = {
    "John" : 938477566,
    "Jack" : 938377264,
    "Jill" : 947662781
}
print(phonebook)


# ## **Iterating over dictionaries**
# 
# Dictionaries can be iterated over, just like lists. However, a dictionary, unlike a list, does not keep the order of the values stored in it. To iterate over key-value pairs, use the following syntax:

# In[ ]:


phonebook = {"John" : 938477566,"Jack" : 938377264,"Jill" : 947662781}
for name, number in phonebook.items():
    print("Phone number of %s is %d" % (name, number))


# ## **Removing a value**
# 
# To remove a specified index, use either one of the following notations:

# In[ ]:


phonebook = {
   "John" : 938477566,
   "Jack" : 938377264,
   "Jill" : 947662781
}
del phonebook["John"]
print(phonebook)


# or:

# In[ ]:


phonebook = {
   "John" : 938477566,
   "Jack" : 938377264,
   "Jill" : 947662781
}
phonebook.pop("John")
print(phonebook)


# ### **Exercise**
# 
# Add <span style="color:green">"Jake"</span> to the phonebook with the phone number <span style="color:orangered">938273443</span>, and remove <span style="color:green">"Jill"</span> from the phonebook.

# In[ ]:


phonebook = {
    "John" : 938477566,
    "Jack" : 938377264,
    "Jill" : 947662781
}

# Write your code here


# Testing code
if "Jake" in phonebook:
    print("Jake is listed in the phonebook.")
if "Jill" not in phonebook:
    print("Jill is not listed in the phonebook.")


# # **Modules and Packages**
# 
# In programming, a module is a piece of software that has a specific functionality. For example, when building a ping pong game, one module would be responsible for the game logic, and
# another module would be responsible for drawing the game on the screen. Each module is a different file, which can be edited separately.

# ![](https://i.imgur.com/D9rWjEs.jpg)

# ## **Writing modules**
# 
# Modules in Python are simply Python files with a <span style="color:red">.py</span> extension. The name of the module will be the name of the file. A Python module can have a set of functions, classes or variables defined and implemented. In the example above, we will have two files:

# In[ ]:


mygame/
mygame/game.py
mygame/draw.py


# The Python script <span style="color:red">game.py</span> will implement the game. It will use the function <span style="color:orangered">draw_game</span> from the file <span style="color:red">draw.py</span>, or in other words, the <span style="color:magenta">draw</span> module, that implements the logic for drawing the game on the screen.
# 
# Modules are imported from other modules using the **<span style="color:purple">import</span>** command. In this example, the <span style="color:red">game.py</span> script may look something like this:

# In[ ]:


# game.py

# Import the draw module
import draw

def play_game():
    ...

def main():
    result = play_game()
    draw.draw_game(result)

# This means that if this script is executed, then main() will be executed
if __name__ == '__main__':
    main()


# The <span style="color:magenta">draw</span> module may look something like this:

# In[ ]:


# draw.py

def draw_game():
    ...

def clear_screen(screen):
    ...


# In this example, the <span style="color:magenta">game</span> module imports the <span style="color:magenta">draw</span> module, which enables it to use functions implemented in that module. The <span style="color:orangered">main</span> function would use the local function <span style="color:orangered">play_game</span> to run the game, and then draw the result of the game using a function implemented in the <span style="color:magenta">draw</span> module called <span style="color:orangered">draw_game</span>. To use the function <span style="color:orangered">draw_game</span> from the <span style="color:magenta">draw</span> module, we would need to specify in which module the function is implemented, using the dot operator. To reference the <span style="color:orangered">draw_game</span> function from the <span style="color:magenta">game</span> module, we would need to import the <span style="color:magenta">draw</span> module and only then call <span style="color:orangered">draw.draw_game()</span>.
# 
# When the **<span style="color:purple">import draw</span>** directive will run, the Python interpreter will look for a file in the directory which the script was executed from, by the name of the module with a <span style="color:red">.py</span> prefix, so in our case it will try to look for <span style="color:red">draw.py</span>. If it will find one, it will import it. If not, it will continue to look for built-in modules.
# 
# You may have noticed that when importing a module, a <span style="color:red">.pyc</span> file appears, which is a compiled Python file. Python compiles files into Python bytecode so that it won't have to parse the files each time modules are loaded. If a <span style="color:red">.pyc</span> file exists, it gets loaded instead of the <span style="color:red">.py</span> file, but this process is transparent to the user.
# 
# ## **Importing module objects to the current namespace**
# 
# We may also import the function <span style="color:orangered">draw_game</span> directly into the main script's namespace, by using the **<span style="color:purple">from</span>** command.

# In[ ]:


# game.py

# Import the draw module
from draw import draw_game

def main():
    result = play_game()
    draw_game(result)


# You may have noticed that in this example, <span style="color:orangered">draw_game</span> does not precede with the name of the module it is imported from, because we've specified the module name in the **<span style="color:purple">import</span>** command.
# 
# The advantages of using this notation is that it is easier to use the functions inside the current module because you don't need to specify which module the function comes from. However, any namespace cannot have two objects with the exact same name, so the **<span style="color:purple">import</span>** command may replace an existing object in the namespace.
# 
# ## **Importing all objects from a module**
# 
# We may also use the **<span style="color:purple">import *</span>** command to import all objects from a specific module, like this:

# In[ ]:


# game.py

# Import the draw module
from draw import *

def main():
    result = play_game()
    draw_game(result)


# This might be a bit risky as changes in the module might affect the module which imports it, but it is shorter and also does not require you to specify which objects you wish to import from the module.
# 
# ## **Custom import name**
# 
# We may also load modules under any name we want. This is useful when we want to import a module conditionally to use the same name in the rest of the code.
# 
# For example, if you have two <span style="color:orangered">draw</span> modules with slighty different names - you may do the following:

# In[ ]:


# game.py

# Import the draw module
if visual_mode:
    # In visual mode, we draw using graphics
    import draw_visual as draw
else:
    # In textual mode, we print out text
    import draw_textual as draw

def main():
    result = play_game()
    # This can either be visual or textual depending on visual_mode
    draw.draw_game(result)


# ## **Module initialization**
# 
# The first time a module is loaded into a running Python script, it is initialized by executing the code in the module once. If another module in your code imports the same module again, it will not be loaded twice but once only - so local variables inside the module act as a "singleton" - they are initialized only once.
# 
# This is useful to know, because this means that you can rely on this behavior for initializing objects.

# ## **Extending module load path**
# 
# There are a couple of ways we could tell the Python interpreter where to look for modules, aside from the default, which is the local directory and the built-in modules. You could either use the environment variable **<span style="color:purple">PYTHONPATH</span>** to specify additional directories to look for modules in, like this:

# In[ ]:


PYTHONPATH=/foo python game.py


# or

# In[ ]:


import os
os.environ['PYTHONPATH']='/foo'
import foo


# This will execute <span style="color:red">game.py</span>, and will enable the script to load modules from the <span style="color:red">foo</span> directory as well as the local directory.
# 
# Another method is the **<span style="color:purple">sys.path.append</span>** function. You may execute it before running an **<span style="color:purple">import</span>** command:

# In[ ]:


import sys
sys.path.append('/foo')


# This will add the <span style="color:red">foo</span> directory to the list of paths to look for modules in as well.
# 
# ## **Exploring built-in modules**
# 
# Check out the full list of built-in modules in the Python standard library here.
# 
# Two very important functions come in handy when exploring modules in Python - the **<span style="color:purple">dir</span>** and **<span style="color:purple">help</span>** functions.
# 
# If we want to import the module **<span style="color:purple">urllib</span>**, which enables us to read data from URLs, we simply import the module:

# In[ ]:


# Import the library
import urllib

# Use it
x = urllib.request.urlopen('https://www.google.com/')
print(x.read())


# We can look for which functions are implemented in each module by using the **<span style="color:purple">dir</span>** function:

# In[ ]:


import urllib
dir(urllib)


# When we find the function in the module we want to use, we can read about it more using the **<span style="color:purple">help</span>** function, inside the Python interpreter:

# In[ ]:


help(urllib.error)


# ## **Writing packages**
# 
# Packages are namespaces which contain multiple packages and modules themselves. They are simply directories, but with a twist.
# 
# Each package in Python is a directory which MUST contain a special file called <span style="color:red">\_\_init\_\_.py</span>. This file can be empty, and it indicates that the directory it contains is a Python package, so it can be imported the same way a module can be imported.
# 
# If we create a directory called <span style="color:red">foo</span>, which marks the package name, we can then create a module inside that package called <span style="color:magenta">bar</span>. We also must not forget to add the <span style="color:red">\_\_init\_\_.py</span> file inside the <span style="color:red">foo</span> directory.
# 
# To use the module <span style="color:magenta">bar</span>, we can import it in two ways:

# In[ ]:


import foo.bar


# or:

# In[ ]:


from foo import bar


# In the first method, we must use the <span style="color:orangered">foo</span> prefix whenever we access the module <span style="color:magenta">bar</span>. In the second method, we don't, because we import the module to our module's namespace.
# 
# The <span style="color:red">\_\_init\_\_.py</span> file can also decide which modules the package exports as the API, while keeping other modules internal, by overriding the <span style="color:red">\_\_all\_\_</span> variable, like so:

# In[ ]:


__init__.py:

__all__ = ["bar"]


# ## **Exercise**
# 
# In this exercise, you will need to print an alphabetically sorted list of all functions in the <span style="color:magenta">re</span> module, which contain the word <span style="color:green">"find"</span>.

# In[ ]:


import re

# Your code goes here

