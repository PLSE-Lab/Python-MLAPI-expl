#!/usr/bin/env python
# coding: utf-8

# # HELLO EVERYONE!! PYTHON IS EASY!!!! LET ME SHOW YOU!

# **LETS START WITH OUR FANTASTIC FOUR!!**
# * Tuples
# * Lists
# * Sets
# * Dict

# # TUPLES
# 
# In Python, tuples are similar to lists but they are immutable i.e. they cannot be changed. You would use the tuples to present data that shouldn't be changed, such as days of week or dates on a calendar.

# In[ ]:


# Can create a tuple with mixed types
t = (1,2,3)


# In[ ]:


# Check len just like a list
type(t)


# In[ ]:


# Can also mix object types
t = ('yeah',2)

# Show
l = ['aniket','gaikwad']
l.sort()
l


# In[ ]:


# Use indexing just like we did in lists
t[0]


# In[ ]:


# Slicing just like a list
t[-1]


# **What errors do we get while using tuples?**

# In[ ]:


# Use .index to enter a value and return the index
t.index(89)


# In[ ]:


# Use .count to count the number of times a value appears
t.count('yeah')


# **Immutability**
# 
# Remember, we cannot change the elements of tuple!
# 
# Just have a look at the errors below!

# In[ ]:


t[0]= 'change'


# In[ ]:


t.append('nope')


# # SETS
# Sets are an unordered collection of unique elements which can be constructed using the set() function.
# 
# Let's go ahead and create a set to see how it works.

# In[ ]:


x = set()


# In[ ]:


# We add to sets with the add() method
x.add(3)


# In[ ]:


# Add a different element
x.add(2)


# In[ ]:


x


# # DICTIONARIES
# 
# We have learned about "Sequences" in the previous session. Now, let's switch the gears and learn about "mappings" in Python. These dictionaries are nothing but hash tables in other programming languages.

# In[ ]:


# Make a dictionary with {} and : to signify a key and a value
my_dict = {True:'value1','key2':'value2','key1':'valuedfvdfg','key1':'abc'}
my_dict


# In[ ]:


my_dict = {'key1':123,'key2':[12,23,33],'key3':['item0','item1','item2']}


# In[ ]:


# Can call an index on that value
my_dict['key3'][0]


# In[ ]:


#Check
my_dict['key1']


# **NESTED DICT**

# In[ ]:


# Dictionary nested inside a dictionary nested in side a dictionary
d = {'key1':{'nestkey':{'subnestkey':'value'}}}


# In[ ]:


# Keep calling the keys
d['key1']['nestkey']


# **METHODS ON DICTIONARY**

# In[ ]:


# Create a typical dictionary
d = {'key1':1,'key2':2,'key3':3}


# In[ ]:


# Method to return a list of all keys 
f=d.keys()
list(f)[0]


# In[ ]:


# Method to grab all values
type(d.values())


# In[ ]:


# Method to return tuples of all items  (we'll learn about tuples soon)
d.items()


# # LISTS
# 
# A list is a collection which is ordered and changeable. In Python lists are written with square brackets.

# In[ ]:


thislist = ["apple", "banana", "cherry"]
print(thislist)


# In[ ]:


thislist = ["apple", "banana", "cherry"]
print(thislist[1])


# In[ ]:


thislist = ["apple", "banana", "cherry"]
print(thislist[-1])


# In[ ]:


thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]
print(thislist[2:5])


# In[ ]:


thislist = ["apple", "banana", "cherry"]
thislist[1] = "blackcurrant"
print(thislist)


# # THANK YOU FOR LEARNING!! STAY TUNED FOR MORE!

# **REMEMBER! YOU ARE DOING GREAT.JUST KEEP GOING....**
