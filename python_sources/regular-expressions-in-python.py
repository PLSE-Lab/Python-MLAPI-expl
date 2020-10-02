#!/usr/bin/env python
# coding: utf-8

# # Regular Expressions

# ![](https://images.unsplash.com/photo-1517694712202-14dd9538aa97?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=750&q=80)

# * Regular expressions is a concept used to search for patterns in string text.
# 
# * This is a univerisal concept for any programming language. 
# 
# * The goal of regular expressions is to be able to search for a specific type of text inside of a string. If we have a form on our webpage where we ask for email addresses, can we check whether the inputted string actually follows the form of an email? some letters or numbers or special characters, then an @ sign then some more letters numbers or special characters then a . then a few more letters
# 
# Ref: [Introduction to Regex](https://scotch.io/tutorials/an-introduction-to-regex-in-python)

# # Need of Regex
# 
# ![Regex_Use.png](attachment:Regex_Use.png)

# In[ ]:


#The library "re" supports regular expression
import re

from IPython.display import Image
import os


# * below is the sample text which is used here.

# In[ ]:


sampletext_to_search = '''abcdefghijklmnopqurtuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
1234567890
123abc

Hello HelloHello

MetaCharacters (Need to be escaped):
. ^ $ * + ? { } [ ] \ | ( )

utexas.edu
 
821-545-4271 
823.559.1938

daniel-mitchell@utexas.edu

Mr. Johnson
Mr Smith
Ms Davis
Mrs. Robinson
Mr. T '''


# # Searching literals

# * first will create a pattern using complie function
#Signature: re.compile(pattern, flags=0)
#text need to search is 'abc'
#here "r" is to consider only raw character

#1.set the pattern
pattern_literal = re.compile(r'abc')
# # Split Function

# * Split function will split the sentence into list of words

# In[ ]:


re.split('\n',sampletext_to_search)


# # Search the matching character

# now trying to find the matching pattern/character using "finditer" method which is sits inside the pattern object

# In[ ]:


#2.use "finditer" function to search the matching character just like the  pattern created in previous step.
matching_results = pattern_literal.finditer(sampletext_to_search)

#3.print the results
for char in matching_results:
    print(char)


# 
# * Note: 
#     * Above result notice the "span" where its tells the index location of the matching result.
#     * Regilar Expressions are Case Sensitive and Order Sensitive  

# In[ ]:


# cross verify the result by searching the index and see the results 
print(sampletext_to_search[68:71])


# # Searching special characters
# *     Lets do the same for special characters

# In[ ]:


#create a pattern to find dot(.) character
pattern_specialchar = re.compile(r'.')

matching_results = pattern_specialchar.finditer(sampletext_to_search)

#print the results
for char in matching_results:
    print(char)


# * Note:
#     * Results having dot character and other character as well , but why ?
#     * Because dot(.) is a special character where dot(.) will do - any character after new line . so its returning all character after new line here.    

# To find just the dot(.) character we need to escape like below

# In[ ]:


pattern_specialchar = re.compile(r'\.')

matching_results = pattern_specialchar.finditer(sampletext_to_search)

#print the results
for char in matching_results:
    print(char)


# # List of Special Character
# * .(dot)       - Any Character Except New Line
# 
# * \d      - Digit (0-9)
# 
# * \D      - Not a Digit (0-9)
# 
# * \w      - Word Character (a-z, A-Z, 0-9, _)
# 
# * \W      - Not a Word Character
# 
# * \s      - Whitespace (space, tab, newline)
# 
# * \S      - Not Whitespace (space, tab, newline)
# 
# 
# * \b      - Word Boundary
# 
# * \B      - Not a Word Boundary
# 
# * ^       - Beginning of a String
# 
# * $       - End of a String
# 
# * []      - Matches Characters in brackets
# 
# * [^ ]    - Matches Characters NOT in brackets
# 
# * |       - Either Or
# 
# * ( )     - Group
# 
# 
# **Quantifiers:**
# 
# *  *- 0 or More
# 
# +  +- 1 or More
# 
# * ? - 0 or One
# 
# * {3} - Exact Number
# 
# * {3,4} - Range of Numbers (Minimum, Maximum)

# In[ ]:


#lets find any number character.

#set the pattern
pattern_anynum = re.compile(r'\d')


# In[ ]:


#pass the entire text to pattern to find matching result
matching_results = pattern_anynum.finditer(sampletext_to_search)

#print the results
for num in matching_results:
    print(num)


# # Any Number followed by Any Character

# In[ ]:


pattern_anyNumChar = re.compile(r'\d\w')
matching_results =  pattern_anyNumChar.finditer(sampletext_to_search)

#print the results
for char in matching_results:
    print(char)


# * Note:
#     *     Regular expression will search Only [Non Overlapping character](http://pages.cs.wisc.edu/~fischer/cs536.f12/lectures/Lecture13.pdf)

# # Word boundary
#     * A word boundary is a position that is either preceded by a word character and not followed by one, or followed by a word character and not preceded by one

# * apply word boundry on right side of the "Hello" word

# In[ ]:




# Search_text: Hello HelloHello
pattern_wordboundry = re.compile(r'Hello\b')
matching_results =  pattern_wordboundry.finditer(sampletext_to_search)

#print the results
for char in matching_results:
    print(char)


# * now apply word boundry on right side and Left side of the "Hello" word

# In[ ]:


# Search_text: Hello HelloHello
pattern_wordboundry = re.compile(r'\bHello\b')

matching_results = pattern_wordboundry.finditer(sampletext_to_search)

#print the results
for char in matching_results:
    print(char)





# # Character sets   

# * Character set will search any characher inside the pattern which is mentioned in square bracket.
# * Below pattern will look for any character 1 or 3.
# 

# In[ ]:


pattern_charset = re.compile(r'[13]')

matching_results = pattern_charset.finditer(sampletext_to_search)

for char in matching_results:
    print(char)


# * below pattern will search and return any 3 or 4 followed by word character

# In[ ]:


pattern_charset = re.compile(r'[34]\w')

matching_results = pattern_charset.finditer(sampletext_to_search)

for char in matching_results:
    print(char)


# * now will search any lower case character followed by lower case character using below pattern

# In[ ]:


pattern_charset = re.compile(r'[a-z][a-z]')

matching_results = pattern_charset.finditer(sampletext_to_search)

for char in matching_results:
    print(char)


# *  Below pattern will search any  NOT lower case character followed by NOT lower case character using

# In[ ]:


pattern_charset = re.compile(r'[^a-z][^a-z]')

matching_results = pattern_charset.finditer(sampletext_to_search)

for char in matching_results:
    print(char)


# # Character groups

# Character Groups used to find any sequence of character instead of individual character.

# In[ ]:


pattern_charsetgrp = re.compile(r'(bcd|efg|ijkl)')

matching_results = pattern_charsetgrp.finditer(sampletext_to_search)

for char in matching_results:
    print(char)


# Combination of Character Group and Character Set

# In[ ]:


pattern_charsergrp = re.compile(r'([A-Z]|io)[a-z]')

matching_results = pattern_charsetgrp.finditer(sampletext_to_search)

for char in matching_results:
    print(char)
                                

                                
                                


# # Quantifiers
# 
# Quantifiers:
# 1. *       - 0 or More
# 
# 1. +       - 1 or More
# 
# 1. ?       - 0 or One
# 
# 1. {3}     - Exact Number
#  
# 1. {3,4}   - Range of Numbers (Minimum, Maximum)
# 

# In[ ]:


pattern_quantify = re.compile(r'Mr\.?\s[A-Z]')

matching_results = pattern_quantify.finditer(sampletext_to_search)

for char in matching_results:
    print(char)


# In[ ]:


pattern_quantify = re.compile(r'M(s|rs)')

matching_results = pattern_quantify.finditer(sampletext_to_search)

for char in matching_results:
    print(char)

                              


# * Lets search for phone number below

# In[ ]:


pattern_quantify = re.compile(r'\d{3}[.-]\d{4}')

matching_results = pattern_quantify.finditer(sampletext_to_search)

for char in matching_results:
    print(char)


# * To find email address

# In[ ]:


pattern_email = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')

matching_results = pattern_email.finditer(sampletext_to_search)

for char in matching_results:
    print(char)

