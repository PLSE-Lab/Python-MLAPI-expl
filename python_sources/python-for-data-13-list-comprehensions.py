#!/usr/bin/env python
# coding: utf-8

# # Python for Data 13: List Comprehensions
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)
# 

# Python prides itself on its clean, readable code and making it as simple as possible for you to do the things you want to do. Although basic control flow statements and functions have enough power to express virtually any program, Python includes many convenience functions and constructs to let you do things faster and with less code.
# 
# Populating lists and dictionaries is a common task that can be achieved with the loops we learned about in lesson 11. For instance, if we wanted to populate a list with the numbers 0 through 100, we could initialize an empty list as a container, run a for loop over the range of numbers from 0 to 100, and append each number to the list:

# In[ ]:


my_list = []

for number in range(0, 101):
    my_list.append(number)
    
print(my_list)


# *Note: range() creates a sequence of numbers from some specified starting number up to but not including an ending number. It also takes an optional argument for the step (counting increment) which defaults to 1.*
# 
# The code above works, but it is unnecessarily verbose. List comprehensions provide a way to do these sorts of constructions efficiently with less code.

# ## List Comprehensions

# List comprehensions let you populate lists in one line of code by taking the logic you would normally put a for loop and moving it inside the list brackets. We can construct the same list as the one above using the following list comprehension:

# In[ ]:


my_list2 = [number for number in range(0, 101)]

print(my_list2)


# In a list comprehension, the value that you want to append to the list come first, in this case "number", followed by a for statement that mirrors the one we used in the for loop version of the code. You can optionally include if clauses after the for clause to filter the results based on some logical check. For instance, we could add an if statement to filter out odd numbers:

# In[ ]:


my_list3 = [number for number in range(0, 101) if number % 2 == 0]

print(my_list3)


# In the code above we take all the numbers in the range for which the number modulus 2 (the remainder when divided by 2) is equal to zero, which returns all the even numbers in the range.
# 
# *Note: You could also get even numbers in a range more by including a step argument equal to 2 such as: range(0,101,2)*
# 
# It is possible to put more than one for loop in a list comprehension, such as to construct a list from two different iterables. For instance, if we wanted to make a list of each combination of two letters in two different strings we could do it with a list comprehension over the two strings with two for clauses:

# In[ ]:


combined = [a + b  for a in "life" for b in "study"]

print (combined)


# You also can nest one list comprehension inside of another:

# In[ ]:


nested = [letters[1] for letters in [a + b  for a in "life" for b in "study"]]

print(nested)


# Notice that while you can nest list comprehensions to achieve a lot in a single line of code, doing so can lead to long, verbose and potentially confusing code. It is often better to avoid the temptation to create convoluted "one-liners" when a series of a few shorter, more readable operations will yield the same result:

# In[ ]:


combined = [a + b  for a in "life" for b in "study"]
non_nested = [letters[1] for letters in combined]

print (non_nested)


# ## Dictionary Comprehensions

# You can create dictionaries quickly in one line using a syntax that mirrors list comprehensions. Consider the following dictionary that sets words as keys and their lengths as values:

# In[ ]:


words = ["life","is","study"]

word_length_dict = {}

for word in words:
    word_length_dict[word] = len(word)
    
print(word_length_dict)


# We could make the same dictionary using a dictionary comprehension where the key and value come first in the form key:value, followed a for clause that loops over some sequence:

# In[ ]:


words = ["life","is","study"]
word_length_dict2 = {word:len(word) for word in words}

print(word_length_dict2)


# It is common to create a dictionary from the items in two different ordered sequences, where one sequence contains the keys you want to use and the other sequence contains the corresponding values. You can pair the items in two sequences into tuples using the built in Python function zip():

# In[ ]:


words = ["life","is","study"]
word_lengths = [4, 2, 5]
pairs = zip(words, word_lengths)

for item in pairs:
    print (item)


# Using zip inside a dictionary comprehension lets you extract key:value pairs from two sequences:

# In[ ]:


words = ["life","is","study"]
word_lengths = [4, 2, 5]

word_length_dict3 = {key:value for (key, value) in zip(words, word_lengths)}

print( word_length_dict3 )


# ## Wrap Up

# List and dictionary comprehensions provide a convenient syntax for creating lists and dictionaries more efficiently and with less code than standard loops. Once you have data loaded into numpy arrays and pandas DataFrames, however, you can often avoid looping constructs all together by using functions available in those packages that operate on data in a vectorized manner.
# 
# Now that we know the basics of Python's data structures and programming constructs, the remainder of this guide will focus on data analysis. In the next lesson, we'll use Python to explore a real-world data set: records of passengers who rode aboard the RMS Titanic on its fateful maiden voyage.

# ## Next Lesson: [Python for Data 14: Data Exploration and Cleaning](https://www.kaggle.com/hamelg/python-for-data-14-data-exploration-and-cleaning)
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)
