#!/usr/bin/env python
# coding: utf-8

# $$
# \def\CC{\bf C}
# \def\QQ{\bf Q}
# \def\RR{\bf R}
# \def\ZZ{\bf Z}
# \def\NN{\bf N}
# $$
# # Control Flow Tools

# ## `!while` Statements
# 
# With the while loop we can execute a set of statements as long as a condition is true.

# In[ ]:


i = 1
while i < 6:
    print(i)
    i += 1


# ## `!if` Statements
# 
# Perhaps the most well-known statement type is the `if` statement. For
# example:

# In[ ]:


if x < 0:
    x = 0
    print('Negative changed to zero')
elif x == 0:
    print('Zero')
elif x == 1:
    print('Single')
else:
    print('More')


# There can be zero or more `elif` parts, and the `else` part is optional.
# The keyword '`elif`' is short for 'else if', and is useful to avoid
# excessive indentation. An `if` ... `elif` ... `elif` ... sequence is
# a substitute for the `switch` or `case` statements found in other
# languages.

# ## `!for` Statements
# 
# 
# The `for` statement in Python differs a bit from what you may be used to
# in C or Pascal. Rather than always iterating over an arithmetic
# progression of numbers (like in Pascal), or giving the user the ability
# to define both the iteration step and halting condition (as C), Python's
# `for` statement iterates over the items of any sequence (a list or a
# string), in the order that they appear in the sequence.

# In[ ]:


# Measure some strings:
animals = ['cat', 'dog', 'rabbit', 'tiger', 'lion']
for animal in animals:
    print(animal, len(animal))


# Code that modifies a list while iterating over that same
# list can be tricky to get right. Instead, it is usually more
# straight-forward to loop over a copy of the list or to create a
# new list:

# In[ ]:


my_list = [0, 1, 2, 3, 4]
for elem in my_list:
    #if elem >= 4:
        #my_list.append(len(my_list))
    print("Length is: {}".format(len(my_list)))


# In[ ]:


animals = ['cat', 'dog', 'rabbit', 'tiger', 'lion']
print("list before loop: {}".format(animals))
# Strategy:  Iterate over a copy
for animal in animals.copy():
    if animal == 'rabbit':
        animals.remove(animal)
print("list after loop: {}".format(animals))


# In[ ]:


# Strategy:  Create a new list
animals = ['cat', 'dog', 'rabbit', 'tiger', 'lion']
print("list before loop: {}".format(animals))
# Strategy:  Iterate over a copy
for animal in list(animals):
    if animal == 'rabbit':
        animals.remove(animal)
print("list after loop: {}".format(animals))


# 
# ## `!break` and `!continue` Statements, and `!else` Clauses on Loops
# 
# The `break` statement, like in C, breaks out of the innermost enclosing
# `for` or `while` loop.
# 
# Loop statements may have an `else` clause; it is executed when the loop
# terminates through exhaustion of the iterable (with `for`) or when the
# condition becomes false (with `while`), but not when the loop is
# terminated by a `break` statement. This is exemplified by the following
# loop, which searches for prime numbers:

# In[ ]:


for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print(n, 'equals', x, '*', n//x)
            break
    else:
        # loop fell through without finding a factor
        print(n, 'is a prime number')


# (Yes, this is the correct code. Look closely: the `else` clause belongs
# to the `for` loop, **not** the `if` statement.)
# 
# When used with a loop, the `else` clause has more in common with the
# `else` clause of a `try` statement than it does with that of `if`
# statements: a `try` statement's `else` clause runs when no exception
# occurs, and a loop's `else` clause runs when no `break` occurs.
# 
# The `continue` statement, also borrowed from C, continues with the next
# iteration of the loop:

# In[ ]:


for num in range(2, 10):
    if num % 2 == 0:
        print("Found an even number", num)
        continue
    print("Found a number", num)

