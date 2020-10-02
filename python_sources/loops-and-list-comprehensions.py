#!/usr/bin/env python
# coding: utf-8

# **[Python Course Home Page](https://www.kaggle.com/learn/python)**
# 
# ---
# 

# # Loops
# 
# Loops are a way to repeatedly execute some code statement.

# In[ ]:


planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
for planet in planets:
    print(planet, end=' ') # print all on same line


# In[ ]:


planets = ['Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune']
for planet in planets:
    print(planet,end=' ')


# Notice the simplicity of the ``for`` loop: we specify the variable we want to use, the sequence we want to loop over, and use the "``in``" keyword to link them together in an intuitive and readable way.
# 
# The object to the right of the "``in``" can be any object that supports iteration. Basically, if it can be thought of as a sequence or collection of things, you can probably loop over it. In addition to lists, we can iterate over the elements of a tuple:

# In[ ]:


multiplicands = (2, 2, 2, 3, 3, 5)
product = 1
for mult in multiplicands:
    product = product * mult
product


# In[ ]:


multiplicands = 2,2,2,3,3,5
product = 1
for mult in multiplicands:
    product = product * mult
product


# And even iterate over each character in a string:

# In[ ]:


s = 'steganograpHy is the practicE of conceaLing a file, message, image, or video within another fiLe, message, image, Or video.'
msg = ''
# print all the uppercase letters in s, one at a time
for char in s:
    if char.isupper():
        print(char, end='')        


# In[ ]:


s = 'steganograpHy is the practicE of conceaLing a file, message, image, or video within another fiLe, message, image, Or video.'
for char in s:
    if char.isupper():
        print(char,end='')


# ### range()
# 
# `range()` is a function that returns a sequence of numbers. It turns out to be very useful for writing loops.
# 
# For example, if we want to repeat some action 5 times:

# In[ ]:


for i in range(5):
    print("Doing important work. i =", i)


# You might assume that `range(5)` returns the list `[0, 1, 2, 3, 4]`. The truth is a little bit more complicated:

# In[ ]:


r = range(5)
r


# `range` returns a "range object". It acts a lot like a list (it's iterable), but doesn't have all the same capabilities. As we saw in the [previous tutorial](https://www.kaggle.com/colinmorris/lists), we can call `help()` on an object like `r` to see Python's documentation on that object, including all of its methods. Click the 'output' button if you're curious about what the help page for a range object looks like.

# In[ ]:


help(range)


# Just as we can use `int()`, `float()`, and `bool()` to convert objects to another type, we can use `list()` to convert a list-like thing into a list, which shows a more familiar (and useful) representation:

# In[ ]:


list(range(5))


# Note that the range starts at zero, and that by convention the top of the range is not included in the output. `range(5)` gives the numbers from 0 up to *but not including* 5. 
# 
# This may seem like a strange way to do things, but the documentation (accessed via `help(range)`) alludes to the reasoning when it says:
# 
# > `range(4)` produces 0, 1, 2, 3.  These are exactly the valid indices for a list of 4 elements.  
# 
# So for any list `L`, `for i in range(len(L)):` will iterate over all its valid indices.

# In[ ]:


nums = [1, 2, 4, 8, 16]
for i in range(len(nums)):
    nums[i] = nums[i] * 2
nums


# In[ ]:


nums = [1,2,4,8,16]
for i in range(len(nums)):
    nums[i] = nums[i] * 2
nums


# This is the classic way of iterating over the indices of a list or other sequence.
# 
# > **Aside**: `for i in range(len(L)):` is analogous to constructs like `for (int i = 0; i < L.length; i++)` in other languages.

# ### `enumerate`
# 
# `for foo in x` loops over the elements of a list and `for i in range(len(x))` loops over the indices of a list. What if you want to do both?
# 
# Enter the `enumerate` function, one of Python's hidden gems:

# In[ ]:


def double_odds(nums):
    for i, num in enumerate(nums):
        if num % 2 == 1:
            nums[i] = num * 2

x = list(range(10))
double_odds(x)
x


# Given a list, `enumerate` returns an object which iterates over the indices *and* the values of the list.
# 
# (Like the `range()` function, it returns an iterable object. To see its contents as a list, we can call `list()` on it.)

# In[ ]:


list(enumerate(['a', 'b']))


# We can see that that the things we were iterating over are tuples. This helps explain that `for i, num` syntax. We're "unpacking" the tuple, just like in this example from the previous tutorial:

# In[ ]:


x = 0.125
numerator, denominator = x.as_integer_ratio()


# In[ ]:


x = 0.125
numerator,denominator = x.as_integer_ratio()


# We can use this unpacking syntax any time we iterate over a collection of tuples.

# In[ ]:


nums = [
    ('one', 1, 'I'),
    ('two', 2, 'II'),
    ('three', 3, 'III'),
    ('four', 4, 'IV'),
]

for word, integer, roman_numeral in nums:
    print(integer, word, roman_numeral, sep=' = ', end='; ')


# In[ ]:


nums = [
    ('one',1,'I'),
    ('two',2,'II'),
    ('three',3,'III'),
    ('four',4,'IV'),
]
for word,integer,roman_numeral in nums:
    print(integer,word,roman_numeral, sep=' = ', end=' ; ')


# This is equivalent to the following (more tedious) code:

# In[ ]:


for tup in nums:
    word = tup[0]
    integer = tup[1]
    roman_numeral = tup[2]
    print(integer, word, roman_numeral, sep=' = ', end='; ')


# In[ ]:


for tup in nums:
    word = tup[0]
    integer = tup[1]
    roman_numeral = tup[2]
    print(integer,word,roman_numeral, sep=' = ', end='; ')


# ## ``while`` loops
# The other type of loop in Python is a ``while`` loop, which iterates until some condition is met:

# In[ ]:


i = 0
while i < 10:
    print(i, end=' ')
    i += 1


# The argument of the ``while`` loop is evaluated as a boolean statement, and the loop is executed until the statement evaluates to False.

# ## List comprehensions
# 
# List comprehensions are one of Python's most beloved and unique features. The easiest way to understand them is probably to just look at a few examples:

# In[ ]:


squares = [n**2 for n in range(10)]
squares


# Here's how we would do the same thing without a list comprehension:

# In[ ]:


squares = []
for n in range(10):
    squares.append(n**2)
squares


# We can also add an `if` condition:

# In[ ]:


short_planets = [planet for planet in planets if len(planet) < 6]
short_planets


# (If you're familiar with SQL, you might think of this as being like a "WHERE" clause)
# 
# Here's an example of filtering with an `if` condition *and* applying some transformation to the loop variable:

# In[ ]:


# str.upper() returns an all-caps version of a string
loud_short_planets = [planet.upper() + '!' for planet in planets if len(planet) < 6]
loud_short_planets


# People usually write these on a single line, but you might find the structure clearer when it's split up over 3 lines:

# In[ ]:


[
    planet.upper() + '!' 
    for planet in planets 
    if len(planet) < 6
]


# (Continuing the SQL analogy, you could think of these three lines as SELECT, FROM, and WHERE)
# 
# The expression on the left doesn't technically have to involve the loop variable (though it'd be pretty unusual for it not to). What do you think the expression below will evaluate to? Press the 'output' button to check. 

# In[ ]:


[32 for planet in planets]


# List comprehensions combined with some of the functions we've seen like `min`, `max`, `sum`, `len`, and `sorted`, can lead to some pretty impressive one-line solutions for problems that would otherwise require several lines of code. 
# 
# For example, [the last exercise](https://www.kaggle.com/kernels/fork/1275173) included a brainteaser asking you to write a function to count the number of negative numbers in a list *without using loops* (or any other syntax we hadn't seen). Here's how we might solve the problem now that we have loops in our arsenal:
# 

# In[ ]:


def count_negatives(nums):
    """Return the number of negative numbers in the given list.
    
    >>> count_negatives([5, -1, -2, 0, 3])
    2
    """
    n_negative = 0
    for num in nums:
        if num < 0:
            n_negative = n_negative + 1
    return n_negative


# Here's a solution using a list comprehension:

# In[ ]:


def count_negatives(nums):
    return len([num for num in nums if num < 0])


# Much better, right?
# 
# Well if all we care about is minimizing the length of our code, this third solution is better still!

# In[ ]:


def count_negatives(nums):
    # Reminder: in the "booleans and conditionals" exercises, we learned about a quirk of 
    # Python where it calculates something like True + True + False + True to be equal to 3.
    return sum([num < 0 for num in nums])


# Which of these solutions is the "best" is entirely subjective. Solving a problem with less code is always nice, but it's worth keeping in mind the following lines from [The Zen of Python](https://en.wikipedia.org/wiki/Zen_of_Python):
# 
# > Readability counts.  
# > Explicit is better than implicit.
# 
# The last definition of `count_negatives` might be the shortest, but will other people reading your code understand how it works? 
# 
# Writing Pythonic code doesn't mean never using for loops!

# # Your Turn
# 
# Try the [hands-on exercise](https://www.kaggle.com/kernels/fork/1275177) with loops and list comprehensions
# 

# ---
# **[Python Course Home Page](https://www.kaggle.com/learn/python)**
# 
# 
