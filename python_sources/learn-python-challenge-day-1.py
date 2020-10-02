#!/usr/bin/env python
# coding: utf-8

# Welcome to day 1 of the Python Challenge! 
# 
# If you're here to learn Python, you've come to the right place.
# 
# If you already have some Python experience, I hope I can teach you at least a few new tricks, or throw you a few challenging problems.
# 
# If you're here to learn how to use Python for machine learning, data analysis, or visualization, then bad news: I won't be teaching any of that. Good news: we have [several tracks on Kaggle Learn](https://www.kaggle.com/learn/overview) for just that purpose. The goal of this challenge is to quickly cover the fundamentals of the language - which will hopefully stand you in good stead whether you use Python for deep learning, data cleaning, or cheating at Scrabble.
# 
# Each day of the Python Challenge will consist of two parts:
# - A **tutorial** notebook explaining Python concepts and showing example code. That's where you are now.
# - An [**exercise**](https://www.kaggle.com/kernels/fork/969424) notebook with questions and coding problems to test your new knowledge.
# 
# Today's tutorial includes a brief overview of Python syntax, variable assignment, and arithmetic operators. If you have previous Python experience, day 1 might not hold anything new for you, so feel free to [skip straight to the exercise](https://www.kaggle.com/kernels/fork/969424).
# 
# > These lessons borrow and adapt some content from [A Whirlwind Tour of Python](https://www.kaggle.com/sohier/whirlwind-tour-of-python-index). Thanks to the author, Jake VanderPlas, for releasing it under a permissive license. If you're interested in a more thorough, theoretical grounding in Python, I encourage you to check out WTOP.
# 

# # Hello, Python!
# 
# Python was named for the British comedy troupe [Monty Python](https://en.wikipedia.org/wiki/Monty_Python), so why not make our first Python program an homage to their famous [Spam](https://en.wikipedia.org/wiki/Spam_(Monty_Python%29) skit?
# 
# <!-- todo: maybe a little side note on spam metasyntactic variables in Python -->
# 
# Just for fun, try reading over the code below and predicting what it's going to do when run. (If you have no idea, that's fine!)
# 
# Then click the "output" button to see the results of our program.

# In[ ]:


spam_amount = 0
print(spam_amount)

# Ordering Spam, egg, Spam, Spam, bacon and Spam (4 more servings of Spam)
spam_amount = spam_amount + 4

if spam_amount > 0:
    print("But I don't want ANY spam!")

viking_song = "Spam " * spam_amount
print(viking_song)


# There's a lot to unpack here! This silly program demonstrates many important aspects of what Python code looks like (its *syntax*) and how it works (its *semantics*). Let's run down the code from top to bottom.

# In[ ]:


spam_amount = 0


# **Variable assignment!** Here we create a variable called `spam_amount` and assign it the value of 0 using `=`, Python's assignment operator.
# 
# > **Aside**: If you've programmed in certain other languages (like Java or C++), you might be noticing some things Python *doesn't* require us to do here:  
# - we don't need to "declare" `spam_amount` before assigning to it
# - we don't need to tell Python what type of value `spam_amount` is going to refer to. In fact, we can even go on to reassign `spam_amount` to refer to a different sort of thing like a string or a boolean.

# In[ ]:


print(spam_amount)


# A **function call**. `print` is an extremely useful builtin Python function that displays the value passed to it on the screen. We call functions by putting parentheses after their name, with the inputs to the function (or *arguments*) in between.

# In[ ]:


# Ordering Spam, egg, Spam, Spam, bacon and Spam (4 more servings of Spam)
spam_amount = spam_amount + 4


# The first line above is a **comment**. In Python, comments begin with the `#` symbol.
# 
# Next we see an example of reassignment. Reassigning the value of an existing variable looks just the same as creating a variable - it still uses the `=` assignment operator.
# 
# In this case, the value we're assigning to `spam_amount` involves a little simple arithmetic on its previous value. When it encounters this line, Python evaluates the expression on the right-hand-side of the `=` (0 + 4 = 4), and then assigns that value to the variable on the left-hand-side.

# In[ ]:


if spam_amount > 0:
    print("But I don't want ANY spam!")


# We won't talk much about conditionals until later, but, even if you've never coded before, you can probably guess what this does. Python is prized for its readability and the simplicity of its syntax (with some going as far as to call it "executable pseudocode"). 
# 
# Note how we indicated which code belongs to the `if`. `"But I don't want ANY spam!"` is only supposed to be printed if `spam_amount` is positive. But the later code (like `print(viking_song)`) should be executed no matter what. How do we (and Python) know that?
# 
# The colon (`:`) at the end of the `if` line indicates that a new "code block" is coming up. Subsequent lines which are **indented** (beginning with an extra 4 spaces) are part of that code block. You may be familiar with other languages which use `{`curly braces`}` to mark the beginning and end of code blocks. Python's use of meaningful whitespace often is surprising to programmers who are accustomed to other languages, but in practice it can lead to more consistent and readable code than languages that do not enforce indentation of code blocks. 
# 
# The later lines dealing with `viking_song` are not indented with an extra 4 spaces, so they're not a part of the `if`'s code block. We'll see more examples of indented code blocks later when we start defining functions and using loops.
# 
# This code snippet is also our first sighting of a **string** in Python:
# 
# ```python
# "But I don't want ANY spam!"
# ```
# 
# Strings can be marked either by double or single quotation marks. (But because this particular string *contains* a single-quote character, we might confuse Python by trying to surround it with single-quotes, unless we're careful.)

# In[ ]:


viking_song = "Spam " * spam_amount
print(viking_song)


# The `*` operator can be used to multiply two numbers (`3 * 3` evaluates to 9), but amusingly enough, we can also multiply a string by a number, to get a version that's been repeated that many times. Python offers a number of cheeky little time-saving tricks like this where operators like `*` and `+` have a different meaning depending on what kind of thing they're applied to. (The technical term for this is [operator overloading](https://en.wikipedia.org/wiki/Operator_overloading))

# <hr>

# ## Numbers and arithmetic in Python
# 
# We've already seen an example of a variable containing a number above:

# In[ ]:


spam_amount = 0


# "Number" is a fine informal name for the kind of thing, but if we wanted to be more technical, we could ask Python how it would describe the type of thing that `spam_amount` is:

# In[ ]:


type(spam_amount)


# It's an `int` - short for integer. There's another sort of number we commonly encounter in Python:

# In[ ]:


type(19.95)


# A `float` is a number with a decimal place - very useful for representing things like weights or proportions.
# 
# `type()` is the second built-in function we've seen (after `print()`), and it's another good one to remember. It's very useful to be able to ask Python "what kind of thing is this?". 

# A natural thing to want to do with numbers is perform arithmetic. We've seen the `+` operator for addition, and the `*` operator for multiplication (of a sort). Python also has us covered for the rest of the basic buttons on your calculator:
# 
# | Operator     | Name           | Description                                            |
# |--------------|----------------|--------------------------------------------------------|
# | ``a + b``    | Addition       | Sum of ``a`` and ``b``                                 |
# | ``a - b``    | Subtraction    | Difference of ``a`` and ``b``                          |
# | ``a * b``    | Multiplication | Product of ``a`` and ``b``                             |
# | ``a / b``    | True division  | Quotient of ``a`` and ``b``                            |
# | ``a // b``   | Floor division | Quotient of ``a`` and ``b``, removing fractional parts |
# | ``a % b``    | Modulus        | Integer remainder after division of ``a`` by ``b``     |
# | ``a ** b``   | Exponentiation | ``a`` raised to the power of ``b``                     |
# | ``-a``       | Negation       | The negative of ``a``                                  |
# 
# <span style="display:none">hack</span>
# 
# One interesting observation here is that, whereas your calculator probably just has one button for division, Python can do two kinds. "True division" is basically what your calculator does:

# In[ ]:


print(5 / 2)
print(6 / 2)


# It always gives us a `float`. 
# 
# The `//` operator gives us a result that's rounded down to the next integer.

# In[ ]:


print(5 // 2)
print(6 // 2)


# Can you think of where this would be useful? You may see an example soon in the coding problems.

# ### Order of operations
# 
# The arithmetic we learned in primary school has conventions about the order in which operations are evaluated. Some remember these by a mnemonic such as **PEMDAS** - **P**arentheses, **E**xponents, **M**ultiplication/**D**ivision, **A**ddition/**S**ubtraction.
# 
# Python follows similar rules about which calculations to perform first. They're mostly pretty intuitive.

# In[ ]:


8 - 3 + 2


# In[ ]:


-3 + 4 * 2


# Sometimes the default order of operations isn't what we want:

# In[ ]:


hat_height_cm = 25
my_height_cm = 190
# How tall am I, in meters, when wearing my hat?
total_height_meters = hat_height_cm + my_height_cm / 100
print("Height in meters =", total_height_meters, "?")


# Parentheses are your trump card. You can add them to force Python to evaluate sub-expressions in a different order (or just to make your code easier to read).

# In[ ]:


total_height_meters = (hat_height_cm + my_height_cm) / 100
print("Height in meters =", total_height_meters)


# ### Builtin functions for working with numbers
# 
# `min` and `max` return the minimum and maximum of their arguments, respectively...

# In[ ]:


print(min(1, 2, 3))
print(max(1, 2, 3))


# `abs` returns the absolute value of it argument:

# In[ ]:


print(abs(32))
print(abs(-32))


# In addition to being the names of Python's two main numerical types, `int` and `float` can also be called as functions which convert their arguments to the corresponding type:

# In[ ]:


print(float(10))
print(int(3.33))
# They can even be called on strings!
print(int('807') + 1)


# ## Your turn!
# 
# Head over to the [Exercises](https://www.kaggle.com/kernels/fork/969424) notebook to get some hands-on practice writing Python.
