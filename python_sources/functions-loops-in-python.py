#!/usr/bin/env python
# coding: utf-8

# ##  Defining Functions

# **Define a function with no arguments and no return values:**

# In[ ]:


def print_text():
    print('this is text')


# In[ ]:


# call the function
print_text()


# **Define a function with one argument and no return values:**

# In[ ]:


def print_this(x):
    print(x)


# In[ ]:


# call the function
print_this(3)


# In[ ]:


# prints 3, but doesn't assign 3 to n because the function has no return statement
n = print_this(3)


# **Define a function with one argument and one return value:**

# In[ ]:


def square_this(x):
    return x**2
square_this(5)


# In[ ]:


# include an optional docstring to describe the effect of a function
def square_this(x):
    """Return the square of a number."""
    return x**2
print(square_this.__doc__)    # .__doc__  returns the docstring


# In[ ]:


# call the function
square_this(3)


# In[ ]:


# assigns 9 to var, but does not print 9
var = square_this(3)


# **Define a function with two 'positional arguments' (no default values) and one 'keyword argument' (has a default value):**

# In[ ]:


def calc(a, b, op='add'):
    if op == 'add':
        return a + b
    elif op == 'sub':
        return a - b
    else:
        print('valid operations are add and sub')


# In[ ]:


# call the function
calc(10, 4, op='add')


# In[ ]:


# unnamed arguments are inferred by position
calc(10, 4, 'add')


# In[ ]:


# default for 'op' is 'add'
calc(10, 4)


# In[ ]:


calc(10, 4, 'sub')


# In[ ]:


calc(10, 4, 'div')


# **Use `pass` as a placeholder if you haven't written the function body:**

# In[ ]:


def stub():
    pass


# **Return two values from a single function:**

# In[ ]:


def min_max(nums):
    return min(nums), max(nums)


# In[ ]:


# return values can be assigned to a single variable as a tuple
nums = [1, 2, 3]
min_max_num = min_max(nums)
min_max_num


# In[ ]:


# return values can be assigned into multiple variables using tuple unpacking
min_num, max_num = min_max(nums)
print(min_num)
print(max_num)


# ## Anonymous (Lambda) Functions
# 
# - Primarily used to temporarily define a function for use by another function

# In[ ]:


# define a function the "usual" way
def squared(x):
    return x**2


# In[ ]:


# define an identical function using lambda
squared = lambda x: x**2


# **Sort a list of strings by the last letter:**

# In[ ]:


# without using lambda
simpsons = ['homer', 'marge', 'bart']
def last_letter(word):
    return word[-1]
sorted(simpsons, key=last_letter)


# In[ ]:


# using lambda
sorted(simpsons, key=lambda word: word[-1])


# In[ ]:


##  For Loops and While Loops

# includes the start value but excludes the stop value
range(0, 3)


# In[ ]:


# default start value is 0
range(3)


# In[ ]:


# third argument is the step value
range(0, 5, 2)


# **`for` loops:**

# In[ ]:


# not the recommended style
fruits = ['apple', 'banana', 'cherry']
for i in range(len(fruits)):
    print(fruits[i].upper())


# In[ ]:


# recommended style
for fruit in fruits:
    print(fruit.upper())


# In[ ]:


# iterate through two things at once (using tuple unpacking)
family = {'dad':'homer', 'mom':'marge', 'size':6}
for key, value in family.items():
    print(key, value)


# In[ ]:


# use enumerate if you need to access the index value within the loop
for index, fruit in enumerate(fruits):
    print(index, fruit)


# **`for`/`else` loop:**

# In[ ]:


for fruit in fruits:
    if fruit == 'banana':
        print('Found the banana!')
        break    # exit the loop and skip the 'else' block
else:
    # this block executes ONLY if the for loop completes without hitting 'break'
    print("Can't find the banana")


# **`while` loop:**

# In[ ]:


count = 0
while count < 5:
    print('This will print 5 times')
    count += 1    # equivalent to 'count = count + 1'


# In[ ]:


## Comprehensions


# **List comprehension:**

# In[ ]:


# for loop to create a list of cubes
nums = [1, 2, 3, 4, 5]
cubes = []
for num in nums:
    cubes.append(num**3)
cubes


# In[ ]:


# equivalent list comprehension
cubes = [num**3 for num in nums]
cubes


# In[ ]:


# for loop to create a list of cubes of even numbers
cubes_of_even = []
for num in nums:
    if num % 2 == 0:
        cubes_of_even.append(num**3)
cubes_of_even


# In[ ]:


# equivalent list comprehension
# syntax: [expression for variable in iterable if condition]
cubes_of_even = [num**3 for num in nums if num % 2 == 0]
cubes_of_even


# In[ ]:


# for loop to cube even numbers and square odd numbers
cubes_and_squares = []
for num in nums:
    if num % 2 == 0:
        cubes_and_squares.append(num**3)
    else:
        cubes_and_squares.append(num**2)
cubes_and_squares


# In[ ]:


# equivalent list comprehension (using a ternary expression)
# syntax: [true_condition if condition else false_condition for variable in iterable]
cubes_and_squares = [num**3 if num % 2 == 0 else num**2 for num in nums]
cubes_and_squares


# In[ ]:


# for loop to flatten a 2d-matrix
matrix = [[1, 2], [3, 4]]
items = []
for row in matrix:
    for item in row:
        items.append(item)
items


# In[ ]:


# equivalent list comprehension
items = [item for row in matrix
              for item in row]
items


# **Set comprehension:**

# In[ ]:


fruits = ['apple', 'banana', 'cherry']
unique_lengths = {len(fruit) for fruit in fruits}
unique_lengths


# **Dictionary comprehension:**

# In[ ]:


fruit_lengths = {fruit:len(fruit) for fruit in fruits}
fruit_lengths

fruit_indices = {fruit:index for index, fruit in enumerate(fruits)}
fruit_indices


# ##  Map, Filter and Reduce

# **`map` applies a function to every element of a sequence and returns a list (Python 2) or iterator (Python 3):**

# In[ ]:


simpsons = ['homer', 'marge', 'bart']
mapper = map(len, simpsons)
mapper


# In[ ]:


for i in mapper:
    print(i)


# In[ ]:


# equivalent list comprehension
[len(word) for word in simpsons]


# In[ ]:


mapper = map(lambda word: word[-1], simpsons)
for i in mapper:
    print(i)


# In[ ]:


# equivalent list comprehension
[word[-1] for word in simpsons]


# **`filter` returns a list (Python 2) or iterator (Python 3) containing the elements from a sequence for which a condition is `True`:**

# In[ ]:


nums = range(5)
iterator = filter(lambda x: x % 2 == 0, nums) 
for element in iterator:
    print(element)


# In[ ]:


# equivalent list comprehension
[num for num in nums if num % 2 == 0]


# **`Reduce` returns a single value after applying a function on an iterable**
# example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates
# ((((1+2)+3)+4)+5). 

# In[ ]:


from functools import reduce

data = [1,2,3,4]
reduce(lambda x,y: x * y, data)


# ## Loops

# **for-each loop**
# - In python, the for loop is called 'for each loop' ( that is the way we read it. "for each element in iterable")
# - We use it to iterate over a series, list, string etc any iterable

# In[ ]:


# print each elemt of a list using for loop
a = [1,2,3,4,5,6]  

for element in a:
    print(element)


# In[ ]:


# square each element of a list and append to a new list
b = []  # empty list
for element in a:
    b.append(element*element)
    
print(b)    


# In[ ]:


# populate a dictionary using a for loop
name = ['joe', 'jonas', 'Kit', 'Harrington', 'Liam', 'Neeson']
weight = [15, 18, 17, 22, 25, 23]
dictionary = {}

for i in range(0,6):
    dictionary[name[i]] = weight[i]

dictionary


# In[ ]:


# Lets print some shapes to get comfortable with loops

# pyramid
for i in range(10):
    print((10-i)*' ' + i*'* ')

# half diamond
for i in range(10):
    if i < 6:
        print(i*' *')
    else:
        print((10 - i)*' *')

# full diamond
for i in range(10):
    if i < 6:
        print((10-i)*' ' + i*'* ')
    else:
        print((i*' ' + (10 - i)*'* '))
    

