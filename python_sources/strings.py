#!/usr/bin/env python
# coding: utf-8

# # Strings
# 
# Python strings are just pieces of text.

# In[ ]:


our_string = "Hello World!"


# In[ ]:


our_string


# So far we know how to add them together.

# In[ ]:


"I said: " + our_string


# We also know how to repeat them multiple times.

# In[ ]:


our_string * 3


# Python strings are [immutable](https://docs.python.org/3/glossary.html#term-immutable).
# That's just a fancy way to say that
# they cannot be changed in-place, and we need to create a new string to
# change them. Even `some_string += another_string` creates a new string.
# Python will treat that as `some_string = some_string + another_string`,
# so it creates a new string but it puts it back to the same variable.
# 
# `+` and `*` are nice, but what else can we do with strings?
# 
# ## Slicing
# 
# Slicing is really simple. It just means getting a part of the string.
# For example, to get all characters between the second place between the
# characters and the fifth place between the characters, we can do this:

# In[ ]:


our_string[2:5]


# So the syntax is like `some_string[start:end]`.
# 
# This picture explains how the slicing works:
# 
# ![image.png](attachment:image.png)
# 
# But what happens if we slice with negative values?

# In[ ]:


our_string[-5:-2]


# It turns out that slicing with negative values simply starts counting
# from the end of the string.
# 
# ![Slicing with negative values](../images/slicing2.png)
# 
# If we don't specify the beginning it defaults to 0, and if we don't
# specify the end it defaults to the length of the string. For example, we
# can get everything except the first or last character like this:

# In[ ]:


our_string[1:]


# In[ ]:


our_string[:-1]


# Remember that strings can't be changed in-place.

# In[ ]:


our_string[:5] = 'Howdy'


# There's also a step argument we can give to our slices, but I'm not
# going to talk about it now.
# 
# ## Indexing
# 
# So now we know how slicing works. But what happens if we forget the `:`?

# In[ ]:


our_string[1]


# That's interesting. We got a string that is only one character long. But
# the first character of `Hello World!` should be `H`, not `e`, so why did
# we get an e?
# 
# Programming starts at zero. Indexing strings also starts at zero. The
# first character is `our_string[0]`, the second character is
# `our_string[1]`, and so on.

# In[ ]:


our_string[0]


# In[ ]:


our_string[1]


# In[ ]:


our_string[2]


# In[ ]:


our_string[3]


# In[ ]:


our_string[4]


# So string indexes work like this:
# 
# ![image.png](attachment:image.png)
# 
# How about negative values?

# In[ ]:


our_string[-1]


# We got the last character.
# 
# But why didn't that start at zero? `our_string[-1]` is the last
# character, but `our_string[1]` is not the first character!
# 
# That's because 0 and -0 are equal, so indexing with -0 would do the same
# thing as indexing with 0.
# 
# Indexing with negative values works like this:
# 
# ![Indexing with negative values](../images/indexing2.png)
# 
# ## String methods
# 
# Python's strings have many useful methods.
# [The official documentation](https://docs.python.org/3/library/stdtypes.html#string-methods)
# covers them all, but I'm going to just show some of the most commonly
# used ones briefly. Python also comes with built-in documentation about
# the string methods and we can run `help(str)` to read it. We can also
# get help about one string method at a time, like `help(str.upper)`.
# 
# Again, nothing can modify strings in-place. Most string methods
# return a new string, but things like `our_string = our_string.upper()`
# still work because the new string is assigned to the old variable.
# 
# Also note that all of these methods are used like `our_string.stuff()`,
# not like `stuff(our_string)`. The idea with that is that our string
# knows how to do all these things, like `our_string.stuff()`, we don't
# need a separate function that does these things like `stuff(our_string)`.
# We'll learn more about methods [later](classes.md).
# 
# Here's an example with some of the most commonly used string methods:

# In[ ]:


our_string.upper()


# In[ ]:


our_string.lower()


# In[ ]:


our_string.startswith('Hello')


# In[ ]:


our_string.endswith('World!')


# In[ ]:


our_string.endswith('world!')  # Python is case-sensitive


# In[ ]:


our_string.replace('World', 'there')


# In[ ]:


our_string.replace('o', '@', 1)   # only replace one o


# In[ ]:


'  hello 123  '.lstrip()    # left strip


# In[ ]:


'  hello 123  '.rstrip()    # right strip


# In[ ]:


'  hello 123  '.strip()     # strip from both sides


# In[ ]:


'  hello abc'.rstrip('cb')  # strip c's and b's from right


# In[ ]:


our_string.ljust(30, '-')


# In[ ]:


our_string.rjust(30, '-')


# In[ ]:


our_string.center(30, '-')


# In[ ]:


our_string.count('o')   # it contains two o's


# In[ ]:


our_string.index('o')   # the first o is our_string[4]


# In[ ]:


our_string.rindex('o')  # the last o is our_string[7]


# In[ ]:


'-'.join(['hello', 'world', 'test'])


# In[ ]:


'hello-world-test'.split('-')


# In[ ]:


our_string.upper()[3:].startswith('LO WOR')  # combining multiple things

