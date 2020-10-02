#!/usr/bin/env python
# coding: utf-8

# ### What Are Regular Expressions?
# Regular expressions (regex) allow us to perform find and edit substrings within text with great precision. They are useful for a wide range of tasks where you're starting with messy text. Need to standardize the entries users made to a webform? Identify the authors from a collection of newspaper articles? Remove all emoji from dataset? Regex can do all of those and more.
# 
# The good news is that regular expressions are incredibly powerful. The bad news is that this power means regular expressions can be complicated: the first book on regex I checked is over 500 pages long. Fortunately, even the most basic regular expressions enable you to do things that would be frustrating or impossible with basic string processing tools. 
# 
# With that in mind, this tutorial is shows you what regex can do and where to look for new commands in the [documentation](https://docs.python.org/3/library/re.html) or a [cheat sheet](https://www.debuggex.com/cheatsheet/regex/python). For each of the concepts we cover I'll provide some example use cases and then provide questions that will require you to look up new commands.
# 
# ### Core Tools
# There are a few tools and concepts we need to cover before we start matching patterns. These web pages will be helpful, though you don't need to read through them just yet:
# - [Regex cheat sheet](https://www.debuggex.com/cheatsheet/regex/python): defines the regex commands we'll be working through.
# - [Pythex](https://pythex.org/): this regex tester shows what regex pattern matches. This is especially useful when you get a different number of matches than you expected.
# - [Practice Problems](https://www.kaggle.com/sohier/exercises-for-intro-to-regular-expressions/): this kernel has both warmup and advanced exercises.
# 
# Python uses regular expressions through the `re` library. It has several regex functions, but you only need two to get through most use cases:
# - `re.findall`: returns a list of all matching strings.
# - `re.sub`: substitutes matching strings.
# 
# ### The Basics
# The simplest type of regex matches complete substrings in the same way as normal Python string processing.

# In[ ]:


import re


# In[ ]:


print('I would like some vegetables.'.replace('vegetables', 'pie'))
print(re.sub('vegetables', 'pie', 'I would like some vegetables.'))


# The advantages of regex start to become more clear if we need to make more than one replacement:

# In[ ]:


veggie_request = 'I would like some vegetables, vitamins, and water.'
print(veggie_request.replace('vegetables', 'pie')
    .replace('vitamins', 'pie')
    .replace('water', 'pie'))
print(re.sub('vegetables|vitamins|water', 'pie', veggie_request))


# I used the metacharacter `|`, the regex "or" operator, to shorten the command. Metacharacters signify a special regex command and don't match themselves unless escaped with `\`. We won't go over the other metacharacters here, so I highly recommend looking at the basics section of [the cheat sheet](https://www.debuggex.com/cheatsheet/regex/python) when you tackle the exercises.
# 
# ### Character Classes
# Suppose we want to match a specific set of characters. `Re` offers several built in sets, plus the ability to build our own custom version. For example, the special character `\D` matches all non-digit characters and makes it trivial to do do basic phone number cleanup:

# In[ ]:


messy_phone_number = '(123) 456-7890'
print(re.sub(r'\D', '', messy_phone_number))


# You may have noticed that I the added raw string prefix `r` before my pattern. This allows us to specify special characters with a single `\` rather than `\\`. [Raw string notation (r"text") keeps regular expressions sane](https://docs.python.org/3/library/re.html#raw-string-notation); use them by default.
# 
# If we take a second look at the example above, you'll notice that it strips out too much data for some use cases. If a user entered some letters into the phone number, we might want to raise an error for that entry rather than try to clean it up. A better option is to define a custom character set to narrow down what we delete. 

# In[ ]:


really_messy_number = messy_phone_number + ' this is not a valid phone number'
print(re.sub(r'\D', '', really_messy_number))
print(re.sub(r'[-.() ]', '', really_messy_number))


# That pattern means 'delete any character found between the brackets'. Everything within the brackets is treated as if they were `|` delimited, and we wouldn't have to escape special characters.
# 
# If you need to build custom classes, it's worth taking a look at [the detailed explanation in the documentation](https://docs.python.org/3/library/re.html#regular-expression-syntax) as there are some special ordering rules that only apply within the `[]`.
# 
# ### Quantifiers
# In many cases, we only want to match a specific number of occurrences. A full US phone number including the area code but  country code and no extension will always have 10 digits. If we're searching a text for phone numbers, we'll want to match strings of digits with no more or less than that.

# In[ ]:


buried_phone_number = 'You are the 987th caller in line for 1234567890. Please continue to hold.'
re.findall(r'\d{10}', buried_phone_number)


# ### Lookarounds
# In other cases we may only want a portion of the item we're matching. Let's say that we just need the area code from a phone number. This is where lookarounds come in handy.

# In[ ]:


re.findall(r'\d{3}(?=\d{7})', buried_phone_number)


# That pattern matches three numbers if and only if they're followed by seven more numbers, and only returns the first three. The relevant special characters are in the `Regular Expression Assertions` section of [the cheat sheet](https://www.debuggex.com/cheatsheet/regex/python). 
# 
# ### Flags
# It's often helpful to adjust a pattern's 'settings'. Flags allow us to do that. My personal favorite makes a pattern case insensitive:

# In[ ]:


wordy_tom = """Tom. Let's talk about him. He often forgets to capitalize tom, his name. Oh, and don't match tomorrow."""
re.findall(r'(?i)\bTom\b', wordy_tom)


# That pattern will match any occurrence of Tom, upper or lower case, that starts and ends with word boundaries.
# 
# Congratulations! You now know enough to be dangerous. [Head on over to the exercises](https://www.kaggle.com/sohier/exercises-for-intro-to-regular-expressions/) to practice writing patterns and looking up new commands.

# In[ ]:




