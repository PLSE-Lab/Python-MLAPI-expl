#!/usr/bin/env python
# coding: utf-8

# # Extract Information Using Regular Expressions (RegEx)
# 
# The first thing that i want to start off is the notion of raw string
# 
# **r** expression is used to create a raw string. Python raw string treats backslash (\\) as a literal character.
# 
# 
# 
# Let us see some examples!

# In[ ]:


# normal string vs raw string
path = "C:\desktop\Ravikanth"  #string
print("string:",path)


# In[ ]:


path= r"C:\desktop\Ravikanth"  #raw string
print("raw string:",path)


# So, it is always recommended to use raw strings while dealing with regular expressions. 
# 
# Python has a built-in module to work with regular expressions called **re**. Some of the commonly used methods from the **re** module are listed below:
# 
# 1.re.match(): This function checks if 
# 
# 2.re.search()
# 
# 3.re.findall()
# 
# <br>
# 
# Let us look at each method with the help of example.
# 
# **1. re.match()**
# 
# The re.match function returns a match object on success and none on failure. 

# In[ ]:


import re

#match a word at the beginning of a string

result = re.match('Kaggle',r'Kaggle is the largest data science community of India') 
print(result)

result_2 = re.match('largest',r'Kaggle  is the largest data science community of India') 
print(result_2)


# Since output of the re.match is an object, we will use *group()* function of match object to get the matched expression.

# In[ ]:


print(result.group())  #returns the total matches


# <br>
# 
# **2. re.search()**
# 
# Matches the first occurence of a pattern in the entire string.

# In[ ]:


# search for the pattern "founded" in a given string
result = re.search('founded',r'Andrew NG founded Coursera. He also founded deeplearning.ai')
print(result.group())


# <br>
# 
# **3. re.findall()**
# 
# It will return all the occurrences of the pattern from the string. I would recommend you to use *re.findall()* always, it can work like both *re.search()* and *re.match()*.

# In[ ]:


result = re.findall('founded',r'Andrew NG founded Coursera. He also founded deeplearning.ai')  
print(result)


# ### Special sequences
# 
# 1. **\A**	returns a match if the specified pattern is at the beginning of the string.

# In[ ]:


str = r'Kaggle is the largest data science community of India'

x = re.findall("\AKaggle", str)

print(x)


# This is useful in cases where you have multiple strings of text, and you have to extract the first word only, given that first word is 'Analytics'.
# 
# If you would try to find some other word, then it will return an empty list as shown below.

# In[ ]:


str = r'Analytics Vidhya is the largest Analytics community of India'

x = re.findall("\AVidhya", str)

print(x)


# 2. **\b** returns a match where the specified pattern is at the beginning or at the end of a word.

# In[ ]:


#Check if there is any word that ends with "est"
x = re.findall(r"est\b", str)
print(x)


# It returns the last three characters of the word "largest".

# 3. **\B**	returns a match where the specified pattern is present, but NOT at the beginning (or at the end) of a word.

# In[ ]:


str = r'The good, The bad, and The Ugly '

x = re.findall(r"\Bhe", str)

print(x)


# 4. **\d** returns a match where the string contains digits (numbers from 0-9)

# In[ ]:


str = "2 million monthly visits in Jan'19."

#Check if the string contains any digits (numbers from 0-9):
x = re.findall("\d", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# In[ ]:


str = "2 million monthly visits in Jan'19."

# Check if the string contains any digits (numbers from 0-9):
# adding '+' after '\d' will continue to extract digits till encounters a space
x = re.findall("\d+", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# We can infer that **\d+** repeats one or more occurences of **\d** till the non maching character is found where as **\d** does character wise comparison.

# 5. **\D** returns a match where the string does not contain any digit.

# In[ ]:


str = "2 million monthly visits in Jan'19."

#Check if the word character does not contain any digits (numbers from 0-9):
x = re.findall("\D", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# In[ ]:


str = "2 million monthly visits'19"

#Check if the word does not contain any digits (numbers from 0-9):

x = re.findall("\D+", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# 6. **\w** helps in extraction of alphanumeric characters only (characters from a to Z, digits from 0-9, and the underscore _ character)
# 

# In[ ]:


str = "2 million monthly visits!"

#returns a match at every word character (characters from a to Z, digits from 0-9, and the underscore _ character)

x = re.findall("\w",str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# In[ ]:


str = "2 million monthly visits!"

#returns a match at every word (characters from a to Z, digits from 0-9, and the underscore _ character)

x = re.findall("\w+",str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# 7. **\W** returns match at every non alphanumeric character.

# In[ ]:


str = "2 million monthly visits9!"

#returns a match at every NON word character (characters NOT between a and Z. Like "!", "?" white-space etc.):

x = re.findall("\W", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# ## Metacharacters
# 
# Metacharacters are characters with a special meaning
# 
# 1. **(.)** matches any character (except newline character)

# In[ ]:


str = "rohan and rohit recently published a research paper!" 

#Search for a string that starts with "ro", followed by three (any) characters

x = re.findall("ro.", str)
x2 = re.findall("ro...", str)

print(x)
print(x2)


# 2. **(^)** starts with

# In[ ]:


str = "Data Science"

#Check if the string starts with 'Data':
x = re.findall("^Data", str)

if (x):
  print("Yes, the string starts with 'Data'")
else:
  print("No match")
  
#print(x)  


# In[ ]:


# try with a different string
str2 = "Big Data"

#Check if the string starts with 'Data':
x2 = re.findall("^Data", str2)

if (x2):
  print("Yes, the string starts with 'data'")
else:
  print("No match")
  
#print(x2)  


# 3. **($)** ends with

# In[ ]:


str = "Data Science"

#Check if the string ends with 'Science':

x = re.findall("Science$", str)

if (x):
  print("Yes, the string ends with 'Science'")

else:
  print("No match")
  
#print(x)


# 4. (*) matches for zero or more occurences of the pattern to the left of it

# In[ ]:


str = "easy easssy eay ey"

#Check if the string contains "ea" followed by 0 or more "s" characters and ending with y
x = re.findall("eas*y", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# 5. **(+)** matches one or more occurences of the pattern to the left of it

# In[ ]:


#Check if the string contains "ea" followed by 1 or more "s" characters and ends with y 
x = re.findall("eas+y", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# 6. **(?)** matches zero or one occurrence of the pattern left to it.

# In[ ]:


x = re.findall("eas?y",str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# 7. **(|)** either or

# In[ ]:


str = "Analytics Vidhya is the largest data science community of India"

#Check if the string contains either "data" or "India":

x = re.findall("data|India", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# In[ ]:


# try with a different string
str = "Analytics Vidhya is one of the largest data science communities"

#Check if the string contains either "data" or "India":

x = re.findall("data|India", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# ## Sets
# 
# 1. A set is a bunch of characters inside a pair of square brackets [ ] with a special meaning.

# In[ ]:


str = "Analytics Vidhya is the largest data science community of India"

#Check for the characters y, d, or h, in the above string
x = re.findall("[ydh]", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# In[ ]:


str = "Analytics Vidhya is the largest data science community of India"

#Check for the characters between a and g, in the above string
x = re.findall("[a-g]", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# <br>
# 
# Let's solve a problem.

# In[ ]:


str = "Mars' average distance from the Sun is roughly 230 million km and its orbital period is 687 (Earth) days."

# extract the numbers starting with 0 to 4 from in the above string
x = re.findall(r"\b[0-4]\d+", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# 2. **[^]** Check whether string has other characters mentioned after ^

# In[ ]:


str = "Analytics Vidhya is the largest data sciece community of India"

#Check if every word character has characters than y, d, or h

x = re.findall("[^ydh]", str)

print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")


# 3. **[a-zA-Z0-9]** : Check whether string has alphanumeric characters

# In[ ]:


str = "@AV Largest Data Science community #AV!!"

# extract words that start with a special character
x = re.findall("[^a-zA-Z0-9 ]\w+", str)

print(x)


# ---
# ## Solve Complex Queries
# 
# Let us try solving some complex queries using regex.
# 
# ### Extracting Email IDs
# 
# 

# In[ ]:


str = 'Send a mail to rohan.1997@gmail.com, smith_david34@yahoo.com and priya@yahoo.com about the meeting @2PM'
  
# \w matches any alpha numeric character 
# + for repeats a character one or more times 
#x = re.findall('\w+@\w+\.com', str)     
x = re.findall('[a-zA-Z0-9._-]+@\w+\.com', str)     
  
# Printing of List 
print(x) 


# ### Extracting Dates

# In[ ]:


text = "London Olympic 2012 was held from 2012-07-27 to 2012/08/12."

# '\d{4}' repeats '\d' 4 times
match = re.findall('\d{4}.\d{2}.\d{2}', text)
print(match)


# In[ ]:


text="London Olympic 2012 was held from 27 Jul 2012 to 12-Aug-2012."

match = re.findall('\d{2}.\w{3}.\d{4}', text)

print(match)


# In[ ]:


# extract dates with varying lengths
text="London Olympic 2012 was held from 27 July 2012 to 12 August 2012."

#'\w{3,10}' repeats '\w' 3 to 10 times
match = re.findall('\d{2}.\w{3,10}.\d{4}', text)

print(match)


# ## Extracting Title from Names - Titanic Dataset

# In[ ]:


import pandas as pd

# load dataset
data=pd.read_csv("../input/titanic.csv")


# In[ ]:


data.head()


# In[ ]:


# print a few passenger names
data['Name'].head(10)


# ### Method 1: One way is to split on the pandas dataframe and extract the title

# In[ ]:


name = "Allen, Mr. William Henry"
name2 = name.split(".")
name2[0].split(',')


# In[ ]:


title=data['Name'].apply(lambda x: x.split(".")[0].split(",")[1])
title.value_counts()


# This method might not work all the time. Therefore, another more robust way is to define pattern and search for it using regex

# ### Method 2: Use RegEx to extract titles

# In[ ]:


def split_it(name):
    return re.findall("\w+\.",name)[0]
title=data['Name'].apply(lambda x: split_it(x))
title.value_counts().sum()


# In the above result, we observe that the title is followed by '.' since we are searching for a pattern that includes '.'

# ### Thanks to Analytics Vidhya
# 
# - Please find the below link for Beginners Tutorial for Regular Expressions in Python
# 
# https://www.analyticsvidhya.com/blog/2015/06/regular-expression-python/
