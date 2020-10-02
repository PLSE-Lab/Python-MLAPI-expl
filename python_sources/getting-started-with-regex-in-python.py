#!/usr/bin/env python
# coding: utf-8

# Regular expression offers a powerful too to match and search for strings we want in text data and various corpuses.

# In[ ]:


import re # regex library in python


# ### Find and Search

# In[ ]:


mystring = "preposterous"

# Gives you the index number of the start of the match
mystring.find("rous")


# In[ ]:


# If it doesn't find the matching string, it returns None
re.search("rous", mystring)


# In[ ]:


f = open("../input/alice-in-wonderland.txt") 
alice_lines = f.readlines() # Read all the lines
alice_lines = [l.rstrip() for l in alice_lines] # strip whitespace in each line and then add it in list (list comprehension)
f.close()

for line in alice_lines:
     if re.search("Hatter", line): print( line ) # Look through each line and if a line has word "Hatter" in it, print that line


# In[ ]:


for line in alice_lines:
    if re.search("riddle", line): print(line)


# ### More than Exact Matches

# In[ ]:


# Find word Hatter but also match not just "H"atter but also "h"atter

for line in alice_lines:
    if re.search("[Hh]atter", line): print(line)


# - [Hh] matches a single letter which can be either H or h.
# - [aeiuo] matches any lowercase vowel.
# - [1234567890_abc] matches a digit or an underscore or a or b or c
# - [....] always matches a single character, and you are specifying all the possibilities of what it could be.

# Some shortcuts include:
# 
# - [A-Z] matches any uppercase letter
# - [a-z] matches any lowercase letter
# - [0-9] matches any digit
# - You can combine them, for example in [A-Za-z0-9]

# Exercise) How do you match 3 sequences of numbers?

# In[ ]:


for line in alice_lines:

     if re.search("[0-9][0-9][0-9]", line): print(line)


# Caret(^) acts as a negation if used within a bracket
# 
# - [^aeiou] matches any character that is not a lowercase vowel (what does that encompass?)
# - [^A-Za-z] matches anything but a letter

# ### More methodds of matching single characters

# - \d matches a single digit, equivalent to [0-9]
# - \D matches a single character that is not a digit, equivalent to [^0-9]
# - \s matches a whitespace, equivalent to [\t\n\r\f\v]
# - \S matches a non-whitespace
# - \w matches an alphanumeric character, equivalent o [A-Za-z0-9_]
# - \W matches a non-alphanumeric character

# Exercise) What does the following regex match??

# In[ ]:


for line in alice_lines:
    if re.search("b\w\w\wed", line): print( line )


# It matches any words that have the format "bxxxed" where x represents some single alphanumeric character. For example, the word "bombed" will be matched.

# ### More matching symbols

# - period (.) matches any single character (e.g. letter digit punctuation whitespace etc.)
# - You need backslash (\) if you want to match a literal period

# In[ ]:


# ... : Three consecutive sequences of any single character
for line in alice_lines:
    if re.search("b...ed", line): print(line)


# - plus (+) matchces characters one or more times
# - star (*) matches characters zero or more times

# Exercise) What does \(.+\) match?

# The above mathces "an opening parenthesis (note there is backslash in front of the opening parenthesis), then one or more arbitrary characters, then a closing parenthesis"

# - verticle line (|) in paranthesis is equivalent to "or" when matching words. For example, mov(es|ing|e|ed) matches "moves", "moving", "move", and "moved".
# - "?" means optionality. For instance, sings? will match "sing" as well as "sings".[](http://)

# Example

# In[ ]:


# Lines with words with at least 7 characters
for line in alice_lines:
    if re.search(".......+", line): print(line)


# ### Anchors

# Anchors don't match any characters, they mark special places in a string: at the beginning and end of the string, and at the boundaries of words!

# - Caret ("^"), when not used within a bracket, matches wordss at the beginning of a string. So "^123" will only match strings that begin with "123".

# In[ ]:


# look for lines in Alice in Wonderland that start with "The"
for line in alice_lines:
    if re.search("^The", line): print(line)


# - dollar sign matches at the end of a string. 
# - Hence, "123$" will match strings that end with "123".

# In[ ]:


#  Look for lines that have the word "Alice" occurring in the end of a line
for line in alice_lines:
    if re.search("Alice$", line): print(line)


# - \b matches a word boundary
# - \B matches anywhere but at a word boundary
# - word boundary includes "punctuation" as well (as you can see from the example below)

# In[ ]:


for line in alice_lines:
    if re.search(r"\bsing\b", line): print(line)


# ### Other Functions

# - re.split(): split on regular expressions rather than just strings
# - re.findall(): find all non-overlapping matches in a string and return them in a list
# - re.sub(): substitutes matches of a regex pattern by a string

# ### If you found this notebook helpful, consider upvoting it! Thank you!
