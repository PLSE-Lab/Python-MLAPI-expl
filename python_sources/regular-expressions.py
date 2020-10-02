#!/usr/bin/env python
# coding: utf-8

# # Regular Expressions 

# ## Introduction

# - A sequence of characters that defines a search pattern.
# - '\d' means any character between 0 and 9 where d is digit (Meta character)
# - Literal characters are characters that are specified and will always occur.
# - Meta characters are characters that are more generalized not specific.
# - E.g. to match a number in the format "917-55-1234", the regular expression will be "\d\d\d-\d\d\d-\d\d\d\d"

# In[ ]:


# For example matching a time format

import re

line = "Jan  3 07:57:39 Kali sshd[1397]: Failed password 02:12:36 for root from 172.16.12.55 port 34380 ssh2"
regex = "\d+"
result = re.findall(regex, line) # returns all of the digit matches as a list
first_result = re.findall(regex, line)[0] # returns first match
print(result)
print(first_result)


# - '.' means any character, and '*' means 0 or more 
# - For example, if we write a regular expression "rainbow.*", it means all data that begins with rainbow, could be rainbow123, rainbow boy, rainbow city etc.
# - '.*' is a wildcard that matches the universe

# ## Meta characters

#  ### Single Characters

# - '\d' matches any character between 0 and 9 where d means digit
# - '\w' matches any character A-Za-z0-9 where w means word
# - '\s' matches any whitespace (can match a space, a tab etc.)
# - '.' matches any character whatsoever
# - Capitalizing 'd' or 'w' or 's' makes the expression the opposite
# 

# ### Quantifiers

# - They are meta characters that modify the previous character in a regular expression. (e.g. how many of those things you want to match in a row)
# - '*' matches 0 or more 
# - '+' matches 1 or more 
# - '?' matches 0 or 1 (optional)
# - {n} matches all n
# - {min, max}
# 
# For example, '\w\w\w' means match all first 3 words. Also, '\w{3}' does the same thing

# In[ ]:


word = "I just realized how interesting coding is"
regex = "\w+"
result = re.findall(regex, word) # returns each word as a list
print(result)


# In[ ]:


word = "The colors of the rainbow has many colours and the rainbow does not have a single colour"
regex = "colou?rs?" # ? before a string signifies that the string is optional
result = re.findall(regex, word) # returns all of the matches as a list
print(result)


# In[ ]:


word = "I just realized how interesting coding is"
regex = "\w{3}"
result = re.findall(regex, word) # returns the first three character of each word as a list
print(result)


# - In the example below, we get to see that '.*' is greedy by default. It will continue to match until it can no more match 

# In[ ]:


word = "[Google](http://google.com), [test] \n [itp](http:itp.nyu.edu)"
regex = "\[.*\]" # ? before a string signifies that the string is optional
result = re.findall(regex, word) # return s all of the matches as a list
print(result)


# - Note that '?' paired with a quantifier makes '.*' not greedy

# In[ ]:


word = "[Google](http://google.com), [test] \n [itp](http:itp.nyu.edu)"
regex = "\[.*?\]" # ? before a string signifies that the string is optional
result = re.findall(regex, word) # returns all of the matches as a list
print(result)


# ### Position

# - They are meta characters that matches the position of a character in the string itself. 
# - '^' means beginning
# - '$' means end
# - '\b' word boundary (it is advisable to use escape before it (i.e. "\\b") otherwise \b means a backspace characters

# In[ ]:


word = "The colors of the rainbow has many colours and the rainbow does not have a single colour"
# regex = "\w+$" # means 1 or more word characters at the end of a line.
# regex = "^\w+$" # means 1 or more word characters at the beginning and end of a line (equally just a line with just one word).
regex = "^\w+" # means the beginning of a line followed by 1 or more word characters
result = re.findall(regex, word) # returns all of the matches as a list
print(result)


# In[ ]:


word = "The colors of the rainbow has many colours and the rainbow does not have a single colour"
regex = "\\b\w{3}\\b" # this matches 3 word characters specifically
result = re.findall(regex, word) # returns all of the matches as a list
print(result)


# In[ ]:


word = "The colors of the rainbow has many colours and the rainbow does not have a single colour"
regex = "\\b\w{5,9}\\b" # this matches 5 to 9 word characters specifically
result = re.findall(regex, word) # returns all of the matches as a list
print(result)


# In[ ]:


# shallow copy copies by referencing the original value, while deep copy copies with no reference
import copy 
x = [1,[2]] 
y = copy.copy(x) 
z = copy.deepcopy(x) 
y is z 


# ## Character classes

# - Character classes are stuffs that appear in between square brackets.
# - Each string inside the square brackets are alternatives to each other
# - Also, characters in the square brackets do not possess there meta characteristics, instead they are just literal characters.

# In[ ]:


word = "lynk is not the correct spelling of link"
regex = "l[yi]nk" # this matches either link or lynk
result = re.findall(regex, word) # returns all of the matches as a list
print(result)


# - The only two special characters inside the square brackets are the '-' and '^'
# - '-' inside a square brackets can be used when we want to get a range of strings, e.g. 'a-z1-9' matches any character from a to z and from 1 to 9

# In[ ]:


word = "I am in my 400L, I am currently XX years of age in the year 2018"
regex = "[0-3]{2}" # this matches characters from 0 to 3 and is max of two characters long
result = re.findall(regex, word) # returns all of the matches as a list
print(result)


# - '^' inside a square brackets can be used when we want to get anything that is not amongst the remaining characters after it.
# - Note that if '^' is not located at the beginning, after the first pair of square brackets, then it isn't a special/meta character again, but a literal one.

# In[ ]:


word = "I am in my 400L, I am currently XX years of age in the year 2018"
regex = "[^0-3]{2}" # this matches characters from 0 to 3 and is max of two characters long
result = re.findall(regex, word) # returns all of the matches as a list
print(result)


# ### Alternation

# - We know that in Character classes, each string inside the square brackets are alternatives to each other, which is a limitation.
# - With alternation, multiple strings can be alternatives to each other.
# - For example, in '(com|net)', we mean 'com' or 'net'.

# In[ ]:


word = "I am in my 400L, I am currently XX years of age in the year 2018. My email addresses are stanleydukor@gmail.com, stanleydukor@yahoo.com, stanleydukor@hotmail.edu"
regex = "\w+@\w+\.(?:com|net|org|live|edu)" # this matches email addresses
result = re.findall(regex, word) # returns all of the matches as a list
print(result)


# ## Capturing Groups

# - Suppose we have a stringm '212-555-1234', and we want to match it, we use:
#     "\d{3}-(\d{3})-(\d{4})"
# - Note that the whole string is automatically grouped by regex as "GROUP 0". 
# - Also, using a bracket in this context signifies that the content of the bracket is "Group 1" and "Group 2" respectively.
# - Accessing each group is with the use of a '$' or '\'. e.g. $1 or \1 signifies 'GROUP 1'

# In[ ]:


word = "These are some phone numbers 917-555-1234. Also, you can call me at 646.555.1234 and of course I'm always reachable at (212)867-5509"
regex = "\(?\d{3}[-.)]\d{3}[-.]\d{4}" # this matches phone numbers
result = re.findall(regex, word) # returns all of the matches as a list
print(result)


# In[ ]:


word = "[Google](http://google.com), [test] \n [itp](http:itp.nyu.edu)"
regex = "\[.*?\]\(http.*?\)" # ? matches the name of a link and the link itself
result = re.findall(regex, word) # returns all of the matches as a list
print(result)


# - To replace with the name of the link and the link itself in an html format, we first group them (i.e. "\[(.*?)\]\((http.*?)\)" 

# In[ ]:


word = "2017-07-05 16:04:18.000000"
regex = "\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}"
result = re.findall(regex, word) # returns all of the matches as a list
print(result)

