#!/usr/bin/env python
# coding: utf-8

# # Python for Data 15: Working With Text Data
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)

# Last lesson we learned that there are a lot of questions to consider when you first look at a data set, including whether you should clean or transform the data. We touched briefly on a few basic operations to prepare data for analysis, but the Titanic data set was pretty clean to begin with. Data you encounter in the wild won't always be so friendly. Text data in particular can be extremely messy and difficult to work with because it can contain all sorts of characters and symbols that may have little meaning for your analysis. This lesson will cover some basic techniques and functions for working with text data in Python.
# 
# To start, we'll need some text data that is a little messier than the names in the Titanic data set. As it happens, Kaggle has a data exploration competition giving users access to a database of comments made on Reddit.com during the month of May 2015. Since the Minnesota Timberwolves are my favorite basketball team, let's extract the comments from the team's fan subreddit and use it as an example of messy text for this lesson.
# 
# Let's start by loading the data and checking its structure and a few of the comments:

# In[16]:


# This code is used for loading in data from the Reddit comment database
# Don't worry about the details of this code

import sqlite3
import pandas as pd


sql_conn = sqlite3.connect('../input/reddit-comments-may-2015/database.sqlite')

comments = pd.read_sql("SELECT body FROM May2015 WHERE subreddit = 'timberwolves'", sql_conn)

comments = comments["body"]     # Convert from df to series

print(comments.shape)
comments.head(8)


# The text in these comments is pretty messy. We see everything from long paragraphs to web links to text emoticons. We already learned about a variety of basic string processing functions in lesson 6; pandas extends built in string functions that operate on entire series of strings.

# ## Pandas String Functions

# String functions in pandas mirror built in string functions and many have the same name as their singular counterparts. For example, str.lower() converts a single string to lowercase, while series.str.lower() converts all the strings in a series to lowercase:

# In[18]:


comments[0].lower()      # Convert the first comment to lowercase


# In[19]:


comments.str.lower().head(8)  # Convert all comments to lowercase


# Pandas also supports str.upper() and str.len():

# In[20]:


comments.str.upper().head(8)  # Convert all comments to uppercase


# In[21]:


comments.str.len().head(8)  # Get the length of all comments


# The string splitting and stripping functions also have pandas equivalents:

# In[22]:


comments.str.split(" ").head(8)  # Split comments on spaces


# In[23]:


comments.str.strip("[]").head(8)  # Strip leading and trailing brackets


# Combine all the strings in a series together into a single string with series.str.cat():

# In[24]:


comments.str.cat()[0:500]   # Check the first 500 characters


# You can slice each string in a series and return the result in an elementwise fasion with series.str.slice():

# In[ ]:


comments.str.slice(0, 10).head(8)  # Slice the first 10 characters


# Alternatively, you can use indexing after series.str to take slices:

# In[ ]:


comments.str[0:10].head(8)  # Slice the first 10 characters


# Replace a slice with a new substring using str.slice_replace():

# In[ ]:


comments.str.slice_replace(5, 10, " Wolves Rule! " ).head(8)


# Replace the occurences of a given substring with a different substring using str.replace():

# In[ ]:


comments.str.replace("Wolves", "Pups").head(8)


# A common operation when working with text data is to test whether character strings contain a certain substring or pattern of characters. For instance, if we were only interested in posts about Andrew Wiggins, we'd need to match all posts that make mention of him and avoid matching posts that don't mention him. Use series.str.contains() to get a series of true/false values that indicate whether each string contains a given substring:

# In[ ]:


logical_index = comments.str.lower().str.contains("wigg|drew")

comments[logical_index].head(10)    # Get first 10 comments about Wiggins


# For interest's sake, let's also calculate the ratio of comments that mention Andrew Wiggins:

# In[ ]:


len(comments[logical_index])/len(comments)


# It looks like about 6.6% of comments make mention of Andrew Wiggins. Notice that this string pattern argument we supplied to str.contains() wasn't just a simple substring. Posts about Andrew Wiggins could use any number of different names to refer to him--Wiggins, Andrew, Wigg, Drew--so we needed something a little more flexible than a single substring to match the all posts we're interested in. The pattern we supplied is a simple example of a regular expression.

# ## Regular Expressions

# Pandas has a few more useful string functions, but before we go any further, we need to learn about regular expressions. A regular expression or regex is a sequence of characters and special meta characters used to match a set of character strings. Regular expressions allow you to be more expressive with string matching operations than just providing a simple substring. A regular expression lets you define a "pattern" that can match strings of different lengths, made up of different characters.
# 
# In the str.contains() example above, we supplied the regular expression: "wigg|drew". In this case, the vertical bar | is a metacharacter that acts as the "or" operator, so this regular expression matches any string that contains the substring "wigg" or "drew".
# 
# When you provide a regular expression that contains no metacharacters, it simply matches the exact substring. For instance, "Wiggins" would only match strings containing the exact substring "Wiggins." Metacharacters let you change how you make matches. Here is a list of basic metacharacters and what they do:
# 
# "." - The period is a metacharacter that matches any character other than a newline:

# In[25]:


my_series = pd.Series(["will","bill","Till","still","gull"])
 
my_series.str.contains(".ill")     # Match any substring ending in ill


# "[ ]" - Square brackets specify a set of characters to match:

# In[26]:


my_series.str.contains("[Tt]ill")   # Matches T or t followed by "ill"


# In[27]:


"""
Regular expressions include several special character sets that allow to quickly specify certain common character types. They include:
[a-z] - match any lowercase letter 
[A-Z] - match any uppercase letter 
[0-9] - match any digit 
[a-zA-Z0-9] - match any letter or digit
Adding the "^" symbol inside the square brackets matches any characters NOT in the set:
[^a-z] - match any character that is not a lowercase letter 
[^A-Z] - match any character that is not a uppercase letter 
[^0-9] - match any character that is not a digit 
[^a-zA-Z0-9] - match any character that is not a letter or digit
Python regular expressions also include a shorthand for specifying common sequences:
\d - match any digit 
\D - match any non digit 
\w - match a word character
\W - match a non-word character 
\s - match whitespace (spaces, tabs, newlines, etc.) 
\S - match non-whitespace
"^" - outside of square brackets, the caret symbol searches for matches at the beginning of a string:
"""

ex_str1 = pd.Series(["Where did he go", "He went to the mall", "he is good"])

ex_str1.str.contains("^(He|he)") # Matches He or he at the start of a string


# "$" - searches for matches at the end of a string:

# In[ ]:


ex_str1.str.contains("(go)$") # Matches go at the end of a string


# In[28]:


"""
"( )" - parentheses in regular expressions are used for grouping and to enforce the proper order of operations just like they are in math and logical expressions. In the examples above, the parentheses let us group the or expressions so that the "^" and "$" symbols operate on the entire or statement.
"*" - an asterisk matches zero or more copies of the preceding character
"?" - a question mark matches zero or 1 copy of the preceding character
"+" - a plus matches 1 more copies of the preceding character
"""


ex_str2 = pd.Series(["abdominal","b","aa","abbcc","aba"])

# Match 0 or more a's, a single b, then 1 or characters
ex_str2.str.contains("a*b.+") 


# In[29]:


# Match 1 or more a's, an optional b, then 1 or a's
ex_str2.str.contains("a+b?a+")


# In[ ]:


"""
"{ }" - curly braces match a preceding character for a specified number of repetitions:
"{m}" - the preceding element is matched m times
"{m,}" - the preceding element is matched m times or more
"{m,n}" - the preceding element is matched between m and n times
"""

ex_str3 = pd.Series(["aabcbcb","abbb","abbaab","aabb"])

ex_str3.str.contains("a{2}b{2,}")    # Match 2 a's then 2 or more b's


# "\" - backslash let you "escape" metacharacters. You must escape metacharacters when you actually want to match the metacharacter symbol itself. For instance, if you want to match periods you can't use "." because it is a metacharacter that matches anything. Instead, you'd use "." to escape the period's metacharacter behavior and match the period itself:

# In[30]:


ex_str4 = pd.Series(["Mr. Ed","Dr. Mario","Miss\Mrs Granger."])

ex_str4.str.contains("\. ") # Match a single period and then a space


# If you want to match the escape character backslash itself, you either have to use four backslashes "\\" or encode the string as a raw string of the form r"mystring" and then use double backslashes. Raw strings are an alternate string representation in Python that simplify some oddities in performing regular expressions on normal strings. Read more about them here.

# In[ ]:


ex_str4.str.contains(r"\\") # Match strings containing a backslash


# Raw strings are often used for regular expression patterns because they avoid issues that may that arise when dealing with special string characters.
# 
# There are more regular expression intricacies we won't cover here, but combinations of the few symbols we've covered give you a great amount of expressive power. Regular expressions are commonly used to perform tasks like matching phone numbers, email addresses and web addresses in blocks of text.
# 
# To use regular expressions outside of pandas, you can import the regular expression library with: import re.
# 
# Pandas has several string functions that accept regex patterns and perform an operation on each string in series. We already saw two such functions: series.str.contains() and series.str.replace(). Let's go back to our basketball comments and explore some of these functions.
# 
# Use series.str.count() to count the occurrences of a pattern in each string:

# In[ ]:


comments.str.count(r"[Ww]olves").head(8)


# Use series.str.findall() to get each matched substring and return the result as a list:

# In[ ]:


comments.str.findall(r"[Ww]olves").head(8)


# ## Getting Posts with Web Links

# Now it's time to use some of the new tools we have in our toolbox on the Reddit comment data. Let's say we are only interested in posts that contain web links. If we want to narrow down comments to only those with web links, we'll need to match comments that agree with some pattern that expresses the textual form of a web link. Let's try using a simple regular expression to find posts with web links.
# 
# Web links begin with "http:" or "https:" so let's make a regular expression that matches those substrings:

# In[ ]:


web_links = comments.str.contains(r"https?:")

posts_with_links = comments[web_links]

print( len(posts_with_links))

posts_with_links.head(5)


# It appears the comments we've returned all contain web links. It is possible that a post could contain the string "http:" without actually having a web link. If we wanted to reduce this possibility, we'd have to be more specific with our regular expression pattern, but in the case of a basketball-themed forum, it is pretty unlikely.
# 
# Now that we've identified posts that contain web links, let's extract the links themselves. Many of the posts contain both web links and a bunch of text the user wrote. We want to get rid of the text keep the web links. We can do with with series.str.findall():

# In[ ]:


only_links = posts_with_links.str.findall(r"https?:[^ \n\)]+")

only_links.head(10)


# The pattern we used to match web links may look confusing, so let's go over it step by step.
# 
# First the pattern matches the exact characters "http", an optional "s" and then ":".
# 
# Next, with [^ \n)], we create a set of characters to match. Since our set starts with "^", we are actually matching the negation of the set. In this case, the set is the space character, the newline character "\n" and the closing parenthesis character ")". We had to escape the closing parenthesis character by writing ")". Since we are matching the negation, this set matches any character that is NOT a space, newline or closing parenthesis. Finally, the "+" at the end matches this set 1 or more times.
# 
# To summarize, the regex matches http: or https: at the start and then any number of characters until it encounters a space, newline or closing parenthesis. This regex isn't perfect: a web address could contain parentheses and a space, newline or closing parenthesis might not be the only characters that mark the end of a web link in a comment. It is good enough for this small data set, but for a serious project we would probably want something a little more specific to handle such corner cases.
# 
# Complex regular expressions can be difficult to write and confusing to read. Sometimes it is easiest to simply search the web for a regular expression to perform a common task instead of writing one from scratch. You can test and troubleshoot Python regular expressions using [this](https://regex101.com/#python) online tool.
# 
# *Note: If you copy a regex written for another language it might not work in Python without some modifications.

# ## Wrap Up

# In this lesson, we learned several functions for dealing with text data in Python and introduced regular expressions, a powerful tool for matching substrings in text. Regular expressions are used in many programming languages and although the syntax for regex varies a bit for one language to another, the basic constructs are similar across languages.
# 
# Next time we'll turn our attention to cleaning and preparing numeric data.

# ## Next Lesson: [Python for Data 16: Preparing Numeric Data](https://www.kaggle.com/hamelg/python-for-data-16-preparing-numeric-data)
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)
