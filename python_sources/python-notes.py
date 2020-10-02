#!/usr/bin/env python
# coding: utf-8

# # Python - Introduciton
# 
# Python is a high level, interpretable language.
# 
# **Keywords**
# - `in` operator, returns boolean, can be used to find an object in list. For eg, char in string.
# - `True` and `False` are constants to hold boolean values.
# - `not and or` used for boolean operations.
# - `if elif else` conditional statements.
# - `map() lambda apply` used to define functions and apply it to a list.
# - `for range break continue while with` iterating words.
# - `file open() close()` used for file handling.
# - `try except finally` used for exception handling.
# - **Note:** We can use `else` with `for`. eg, for this, do this, else this.
# 
# **Data Types**
# - `str()` string class, has many string methods, also converts other data type to string.
# - `int()` integer data type class, also converts variable passed to integer.
# - `float()` float class, has functions to work on decimal numbers.
# - `chr()` converts integer to string.
# - `bool()` boolean class, converts to boolean.
# 
# **In Built Data Structures**
# - `list()` class for DS list or array. has many functions related to list operations. Defined in [,]
# - `dict()` class for DS dictionaries, has functions like .keys, .values etc. Defined in {:,}
# - `tuple()` class for DS tuple, has function index and count.Value cannot be changed. Defined in (,).
# - `set()` class for DS set. Has unique values. Defined in {,}.
# - [NumPy](numpy-notes/) and [Pandas](padas-notes/) are external packages that offers other data structures to work on large datasets.
# 
# **General Functions**
# 
# Following fuctions can be directly called in python:
# - `type()` method, returns string, type of a variable.
# - `len()` method, returns int, length of object passed. For eg, `len(s1)` length of string
# - `max()` and `min()` works on list, string etc.
# - `sum()` adds numbers in list.
# - `zip()` combines lists element to element into a tuple. [(x1,y1),(x2,y2)]
# - `sorted()` sorts any list using a key. default is alpanum.
# - `file()` is a class for file operations. Obejct created when we open.
# 
# 

# **Run external commands**
# 
# % can be used to run linux commands from notebook. It is called magic commands. For eg, `%mkdir, %ls, %pwd` etc.

# In[ ]:


get_ipython().run_line_magic('run', "'~/code/py/life.py'")


# In[ ]:


get_ipython().run_line_magic('history', '# Gives history of commands run in notebook')


# ## Strings
# - A built in class `str` has many string related functions.
# - Any String is by default a list of characters.
# - String can be sliced using [:] operator. Starts from 0 and -1 from end. Also works in list or array.
# - Operator `+` for concatenation and `*` for repetition.
# - We can call `str.function(my_string)` or `my_string.function(args)`
# - `len(s1)` length of string
# - `in` operator can be used to find an object in list, char in string.

# In[ ]:


s1 = 'Vaibhav'
print (s1[4])   # prints char at 4th index
print (s1[:4])  # prints chars before 4th index, 4th index excluded.

# Same for -ve index
print (s1[-2])   # prints char at 2nd last index
print (s1[:-2])  # prints chars before 2nd last index, 2nd last index excluded.

print ( len(s1) ) # prints length, number of characters in string.

str.upper(s1)    # same as below 
s1.upper()       # same as above

strList = list(s1) # converts to list data type, array of each char in string
print (strList)

# Formatting
myFormat = '%.2f %% this as fraction, and string %s and value INR%d'
myFormat %(2,1,4.8) # takes 3 args, convers and prints as per definition.


# In[ ]:


# find in string
'bh' in s1


# In[ ]:


type(float('23'))


# ## Data Structures
# 
# Python's in built data sets are list, dict, tuple and sets.
# - All DS can be accessed using [] and index.
# - Index starts from 0 and end is not included.
# - [0:3] gives 0,1,2 and not 3
# - [0:-2] all but last two.

# In[ ]:


my_list = [1,2,3,'hi',[2,3,4]]
# mix data type list
# list is collection of objects
print (my_list)

my_dict = {'k1':'v1' , 'k2':'v2'}
# Key value pair
print (my_dict)


my_tuple = (1,2,3)
# tuples are fixed values and cannot be assigned
print (my_tuple)

my_set = {1,2,3,1,2,1,2,1,2,1,2,1,3}
# set can have only unique values
# for above result is only 1,2,3
print (my_set)


# ### Lists
# - List is a built in data structure.
# - It can hold list of any object.
# - Can have mixed data types.
# - Can hold list in list.
# - declared in [], separated by ,
# - it is similar to array, index works from 0 and -1 from end
# - \* to repeat, + to concatenate, it combines lists to single list.
# - has functions like, `append(), pop(), remove(), index()` etc.

# In[ ]:


a_list = ['amit', 'pranav', 'ramesh', 123, 22.4]
a_list.append(5)
print (a_list)

a_list.reverse() # reverses in place, returns nothing
print (a_list)

# a_list.sort()
print (a_list)


# In[ ]:


sorted('hello World 4')


# #### List Comprehension
# - A way to define list in one line.
# - Similar to the way we define sets in maths, eg, x such that x = x*2.
# - use [] for list
# - use {} for dict or set
# - use () for generators, this does not allocate memory at once. Hence, memory efficient.

# In[ ]:


sqr = [x**2 for x in range(5)] # This gives list with squares of numbers 0 to 4
print (sqr)


# In[ ]:


# With Condition
input_list = [1,2,3,4,5,6,7,8,9,8,7,56,4,32]
even = [var for var in input_list if var % 2 == 0]
print (even)


# In[ ]:


# list comp dict

state = ['Gujarat', 'Maharashtra', 'Rajasthan'] 
capital = ['Gandhinagar', 'Mumbai', 'Jaipur'] 
  
dict_using_comp = {key:value for (key, value) in zip(state, capital)} 
  
print(dict_using_comp) 


# #### Map
# - Map is used to apply a function to all elements in a list.
# - eg, `map(fun, iter)`
# - like, map this function on this list. map my_square on my_numbers.

# In[ ]:


def my_sqr(num):
  return num**2

seq = [1,2,3,4,5]

map(my_sqr,seq) # It computes my_sqr for all items in seq.


# #### Lambda
# - This is used to define a function in a line.
# - `lambda args: expression` - this means passing variable : return statement.
# - Useful for use in map.

# In[ ]:


lambda num: num*2
map(lambda num: num*2, range(1,5)) # square for list


# In[ ]:


map(lambda x, y: x + y,[1,20],[4,50])


# #### Filters
# - Filter is used to filter items in list based on function/lambda
# - eg: to get even values `filter(lambda num: num%2 == 0,seq)`

# In[ ]:


filter(lambda num: num%2 == 0,seq)


# ### Dictionary
# - A built in data structure
# - Consists of key-value pair
# - Key can be anything but mostly num or string
# - values can be any object
# - declared using {} and :
# - accessed using key in []

# In[ ]:


my_dict = {
    'k1': 'value1',
    'k2': 'value2',
    'k3': 'value3'
}
my_dict.items() # returns list of tuples

print (type(my_dict))               # Dict is list of tuples of strings
print (my_dict.items())             # list of tuples of strings
print (type(my_dict.items()))
# print (type(my_dict.items()[0]))    # tuple of strings
# print (type(my_dict.items()[0][1])) # String


# In[ ]:


sorted(my_dict.values(), reverse=True)


# ## Functions
# - The are also called methods, property.
# - `def` is used to define a new function.
# - Similar to java, except it can accept any number of arguments with or without keys.
# - eg, `def my_func(a=5, *numbers, **keywords):`

# In[ ]:


def my_square(num):
  """
  This is DocString, shows up in help using ? or shift tab.
  Can be multiline
  This func squares a number
  """
  return num**2

my_square(4)


# In[ ]:


def my_sum(a=5, b=4, *args, **kwargs):
    print ('Arguments passed:', args)
    print ('Key With Args passed', kwargs)
    sum = a + b
    for n in args:
        sum += n
    return sum

my_sum(2, 4, 3, 1, key0='val0', key1='val1')


# In[ ]:


bool(1.2) #returns boolean or not


# In[ ]:


x=2
x**4 # to the power of


# In[ ]:


True and not False # keywords and not


# In[ ]:


type(chr(97) ) # keywork chr


# In[ ]:


sum(range(0,10)) # last, that is 10th index, is not included


# ## File Operations
# - `open` passes filename and mode to open file.

# In[ ]:


get_ipython().run_line_magic('echo', '"hello world" > new.txt')
handle = open('./new.txt', 'r')

for line in handle:
    print (line)
handle.close()


# In[ ]:


type(handle)


# In[ ]:


handle.name


# In[ ]:


handle.mode


# In[ ]:


new_file = open('new.txt', 'w')


# In[ ]:


new_file.write('some text')


# In[ ]:


new_file.close()


# In[ ]:


get_ipython().run_line_magic('cat', 'new.txt')


# **Modules**
# - python code file having functions
# - can be imported in another files.
# - `import filename as alias` to use module.
# 

# **Exception Handling**
# - use `try except finally` block to handle exceptions.

# In[ ]:


def average( numList ):
    # Raises TypeError or ZeroDivisionError exceptions.
    sum= 0
    for v in numList:
        sum = sum + v
    return float(sum)/len(numList)

def averageReport( numList ):
    try:
        print ("Start averageReport")
        m = average(numList)
        print ("Average = ", m)
    except (TypeError, ex):
        print ("TypeError: ", ex)
    except (ZeroDivisionError, ex):
        print ("ZeroDivisionError: ", ex)
    finally:
         print ("Finish block always runs")

list1 = [10,20,30,40]
list2 = []
list3 = [10,20,30,'abc']

averageReport(list1)


# In[ ]:


state = ['Gujarat', 'Maharashtra', 'Rajasthan'] 
capital = ['Gandhinagar', 'Mumbai', 'Jaipur'] 
  
output_dict = {} 
  
# Using loop for constructing output dictionary 
zip(state, capital)

