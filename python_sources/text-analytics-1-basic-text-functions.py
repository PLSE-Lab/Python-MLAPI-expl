#!/usr/bin/env python
# coding: utf-8

# > <BIG><H1><Font color = "blue"> Introduction to Text Analytics </Font> </H1> </BIG>  
# 
# <P>Just to give you a quick motivation on why you should be interested to work with text is the immence scope of text analytics in today's world. You must have heard of various terms I am going to through below, they are very strong implementation of text analytics: 
# </P>
# <P>In today's world we have created <B>ROBOTs</B>, <B>Chatbots</B>, building <B>Artificially Intelligent Machines</B>, where you ask a question in your own languages (natural language) and it process the input using <B>NLP (Natural Language Processing)</B>. It converts the natural language instructions into machine language which can further be converted to machine instructions and then can be used to fetch the information to answer your query. It may use <B>Sentiment Analysis</B>  techniques by using <B>text analytics</B>/<B>stream analytics</B>/<B>text mining</B>, and tell you what people are buying, what are the latest trending topics, what issues are worrying people or whom they may prefer to vote in the next elections.</P> 
# <P>Sentiment mining, product recommendation by shopping portals, stock market analysis, market basket analysis, fraud detection are few use cases of text analytics. Getting involved in one of these would be quite exciting for you. Enjoy texting..</P>
# <BR><HR>   
#     
# 
# <H2><Font color = "navy">Working with Text in Python  </Font></H2>         
#   
# To experience the great fun of <B>working with text in python</B>, we should be aware of important basic operations on text data and their related functions. Next major learning on text processing would be exploring the <B>nltk</B> module. Basically you don't need to manually write long codes, nltk does most of it for you. But before jumping in to nltk & NLP, we should get some hands on basic text functions. The purpose of this kernal is to touch base these, then we will start the fun part of text processing.
# 
# Like any programming languages, we have several structures here as well to hold different types/shapes of data. <BR>
#     *Alphabets*        : A-Z, a-z  
#     *Digits*               : 0-9  
#     *Symbols*          : ~!@#$%^&*(){}\|:";'<>?,./ and so on.   
# 
# <BR>
#  > Constructs: Character, word & sentence
# <ul>
#   <li>A character is any of the above. </li>
#   <li>A word is meaningful combination of more than one character.  (Often alphabets only)</li>
#   <li>A Sentence is grammatically arranged combination of such words.  It is the basic unit of language which expresses a complete thought.</li>
#   <li>A file is collection of such sentences.</li>
# </ul>

# <P><SMALL><I>References: 
# i) Applied text mining in python @ Coursera. 
# ii) Natural Lanuage Processing in Python @ Udemy.  iii) nltk - The Book.</I></SMALL></P>
#     

# <BR>
# <H2><Font color = "blue"> BASIC FUNCTIONS </Font> </H2>
# <HR>

# In[ ]:


# Lets create a text object to work upon
txt = "A word is meaningful combination of more than one character."
print(txt)


# In[ ]:


# Find the length of a String (no. of charater).
len(txt)


# In[ ]:


## Split a sentence into words
# The split() function below breaks our sentence into words based on the split character provided as paraneter.
# The default splitting character is space. split() returns a list of tokens/words.
words = txt.split(' ')
print(words)


# In[ ]:


# How many words/splits we have?
len(words)
# len() works in same way for strings & lists both.


# > **Tokenization :** When we split text data into words or sentences then each split is called a token. The process of breaking the data into tokens is called tokenization. We have various built-in functions for these kind of tasks which we will be working-on in coming sessions. The **nltk** *(Natural Language Tool Kit)* module provides many more useful functions for text processing.

# In[ ]:


## Filterinag words in a list
# Suppose we want to subset only for those words that are 4 or more character long
[word for word in words if len(word) >= 4]


# In[ ]:


# Find words with first letter as CAPITAL
[word for word in words if word.istitle()]


# In[ ]:


# Find words ends with 'n'
[word for word in words if word.endswith('n')]


# In[ ]:


# Edit: MM_20180518
# A SHORTCUT to use key character to find both upper and lower case strings
# Use lower() or upper() in conjunction.

[word for word in 'To be or not to be'.split(' ') if word.upper().startswith('T')]


# In[ ]:


### Finding Unique words among the list of words
# When you are working with text, you wouldn't want to keep dulicate words in your bag.
# This will cause redundant works for the program, and will consume extra memory as well

# Let's create a string/sentence with duplicate words.
txt1 = 'To be or not to be'           # Here 'to' & 'be' are duplicates.
words1 = txt1.split(' ')                # Split the words

# Print number of elements in the list
len(words1)


# In[ ]:


# Print unique number of elements in the list
len(set(words1))


# In[ ]:


# Here, we had 4 unique words but the instances of 'to' have different cases.
# We can convert these in lower cases to remove complete redundancy.
len(set([word.lower() for word in words1]))


#  <BR>
# <H2><Font color = "blue"> Subsetting a String </Font> </H2>
# <HR>

# In[ ]:


# We can subset the string just like a list.
# index in python starts with 0, and is exclusive of upper range
txt2 = 'Hello World!'

txt2[0:4]


# Notice above that 0-4 is 5 numbers (0,1,2,3,4), but it returned only 4 characters because the upper bound in exclusive.

# In[ ]:


# We can omit lower bound in case we want to filter from start
txt2[:4]     # This will return same result as above


# In[ ]:


# We can omit upper bound in case we want to filter till the end
txt2[4:]


# In[ ]:


# We can also put subsiquent filters to put a series of conditions
txt2[6:][:1]      # To filter only the letter 'W' from the string


# In[ ]:


### Fetch characters from a String
# We can break our string into a character list in two ways
# Using the list() function:
print(list(txt2))

# By looping in:
print([c for c in txt2])

# Both would give same output.


# <BR>
# <H2><Font color = "blue">  Word comparison functions </Font> </H2>
# <HR>

# In[ ]:


txt3 = "Hello"
# Starts With
txt2.startswith('H')    # Whether the string starts with 'H'


# In[ ]:


# Ends With
txt3.endswith('H')     # Whether the string ends with 'H'


# In[ ]:


# Find character in string
'e' in txt3            # Whether we have character 'e' in string


# In[ ]:


# Whether it is in upper case
txt3.isupper()


# In[ ]:


# Whether it is in lower case
txt3.islower()


# In[ ]:


# Whether it is in camel case
txt3.istitle()


# In[ ]:


# Whether the string has all alphabets only
txt3.isalpha()


# In[ ]:


# Whether the string is all numerics/digits
txt3.isdigit()


# In[ ]:


# Whether the string is alpha-numeric
'Hello123'.isalnum() 


#  <BR>
#  <H2><Font color = "blue">  String Operations </Font> </H2>
#  <HR>

# In[ ]:


# Change cases of strings
print(txt3.lower())   # To lower case
print(txt3.upper())   # To upper case
print(txt3.title())   # to camel case or title case


# In[ ]:


# SPLIT AND JOIN strings

# We have seen how to split a string at the beginning, let's see how to join them back.
txt4 = "This is a sentence."
words4 = txt4.split()
print(words4)

# We can join the tokens back to a sentence using join() function
' '.join(words4)


# In[ ]:


# SPLIT String into sentences.

# The function splitlines() can be used to split a string into sentences.
# It splits by the new line character. Useful when working with text files (will see in later sections).
txt5 = "Hi, how are you?\nHope everything is fine at your end."
txt5.splitlines()


# In[ ]:


# Stripping/Trimming white spaces from string
txt6 = "  This is a string with white spaces on both sides..  "

# Length of the text before stripping
len(txt6)


# In[ ]:


# Stripping white spcae from both side
print(txt6.strip())
print(len(txt6.strip()))


# In[ ]:


# Stripping white spcae from right side only
print(txt6.rstrip())
print(len(txt6.rstrip()))


# In[ ]:


# Stripping white spcae from left side only
print(txt6.lstrip())
print(len(txt6.lstrip()))


# In[ ]:


## Find a character in a string from left
print(txt2)
txt2.find('o')   # Returns location/index of first occurance (from left) 'o' in String.


# In[ ]:


## Find a character in a string from right
txt2.rfind('o')   # Returns location/index of first occurance (from right) 'o' in String.


# That's it for this section. I hope you would have found some useful text functions. These functions along with some more *(coming up in further sections)*, are very helpful in preprocessing of the text data. As we all know that almost 70-80% of time is being spent by data scientists on preprocessing and feature engineering. This builds the base for any machine learning model. Our machine learning model is as good as the data it had been fed. See you in next section, till then happy texting.. 
# <BR>
# <!-- Update_20180522 : hyperlinks updated -->
# <CODE>Update :  I have updated the links for next kernal. Please click below to visit the same. Thanks</CODE>
# <HR>
# <SMALL><I>
# <A href="https://www.kaggle.com/manish341/text-analytics-2-text-functions-continued"  target="_blank">
# Coming up next: Handling Larger Text/Reading Data from Text Files <- click here to open
# </A>
# </I></SMALL>
#    
