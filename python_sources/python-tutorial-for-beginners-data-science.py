#!/usr/bin/env python
# coding: utf-8

# # Python Crash Course For Data Science Beginners
# 
# If there are any recommendations/changes you would like to see in this notebook, please leave a comment. Any feedback/constructive criticism would be genuinely appreciated. 
# 
# **This notebook is always a work in progress. So, please stay tuned for more to come.**
# 
# If you like this notebook or find this notebook helpful, Please feel free to **UPVOTE** and/or **leave a comment**.

# 
# 
# 
# 
# This notebook will just go through the basic topics in order:
# 
# * Data types
#     * Numbers
#     * Strings
#     * Printing
#     * Lists
#     * Dictionaries
#     * Booleans
#     * Tuples 
#     * Sets
# * Comparison Operators
# * if, elif, else Statements
# * for Loops
# * while Loops
# * range()
# * list comprehension
# * functions
# * lambda expressions
# * map and filter
# * methods

# # Data types
# 
# ### Numbers

# In[ ]:


1 + 5


# In[ ]:


4 * 4


# In[ ]:


16 / 4


# In[ ]:


2 ** 4


# In[ ]:


4 % 2


# In[ ]:


5 % 2


# In[ ]:


(2 + 3) * (5 + 5)


# # Variable Assignment

# In[ ]:


# Can not start with number or special characters
name_of_var = 2


# In[ ]:


x = 2
y = 3


# In[ ]:


z = x + y


# In[ ]:


z


# # Strings
# 

# In[ ]:


'single quotes'


# In[ ]:


"double quotes"


# In[ ]:


" wrap lot's of other quotes"


# # Printing
# 

# In[ ]:


x = 'hello'


# In[ ]:


print(x)


# In[ ]:


num = 12
name = 'Sam'


# In[ ]:


print('My number is: {one}, and my name is: {two}'.format(one=num,two=name))


# In[ ]:


print('My number is: {}, and my name is: {}'.format(num,name))


# # Lists

# In[ ]:


[1,2,3]


# In[ ]:


['hi',1,[1,2]]


# In[ ]:


my_list = ['a','b','c']


# In[ ]:


my_list.append('d')


# In[ ]:


my_list


# In[ ]:


my_list[0]


# In[ ]:


my_list[1]


# In[ ]:


my_list[1:]


# In[ ]:


my_list[:1]


# In[ ]:


my_list[0] = 'NEW'


# In[ ]:


my_list


# In[ ]:


nest = [1,2,3,[4,5,['target']]]


# In[ ]:


nest[3]


# In[ ]:


nest[3][2]


# In[ ]:


nest[3][2][0]


# # Dictionaries

# In[ ]:


d = {'key1':'item1','key2':'item2'}


# In[ ]:


d


# In[ ]:


d['key1']


# # Booleans

# In[ ]:


True and False


# In[ ]:


True or False


# # Tuples 

# In[ ]:


t = (1,2,3)


# In[ ]:


t[0]


# In[ ]:


t[0] = 'NEW'


# # Sets

# In[ ]:


{1,2,3}


# In[ ]:


{1,2,3,1,2,1,2,3,3,3,3,2,2,2,1,1,2}


# # Comparison Operators

# In[ ]:


1 > 2


# In[ ]:


1 < 2


# In[ ]:


1 >= 1


# In[ ]:


1 <= 4


# In[ ]:


1 == 1


# In[ ]:


'hi' == 'bye'


# # Logic Operators

# In[ ]:


(1 > 2) and (2 < 3)


# In[ ]:


(1 > 2) or (2 < 3)


# In[ ]:


(1 == 2) or (2 == 3) or (4 == 4)


# # if,elif, else Statements

# In[ ]:


if 1 < 2:
    print('Yep!')


# In[ ]:


if 1 < 2:
    print('first')
else:
    print('last')


# In[ ]:


if 1 > 2:
    print('first')
else:
    print('last')


# In[ ]:


if 1 == 2:
    print('first')
elif 3 == 3:
    print('middle')
else:
    print('Last')


# # for Loops

# In[ ]:


seq = [1,2,3,4,5]


# In[ ]:


for item in seq:
    print(item)


# In[ ]:


for item in seq:
    print('Yep')


# In[ ]:


for jelly in seq:
    print(jelly+jelly)


# # while Loops

# In[ ]:


i = 1
while i < 5:
    print('i is: {}'.format(i))
    i = i+1


# # range()

# In[ ]:


range(5)


# In[ ]:


for i in range(5):
    print(i)


# In[ ]:


list(range(5))


# # list comprehension

# In[ ]:


x = [1,2,3,4]


# In[ ]:


out = []
for item in x:
    out.append(item**2)
print(out)


# In[ ]:


[item**2 for item in x]


# # functions

# In[ ]:


def my_func(param1='default'):
    """
    Docstring goes here.
    """
    print(param1)


# In[ ]:


my_func


# In[ ]:


my_func()


# In[ ]:


my_func('new param')


# In[ ]:


my_func(param1='new param')


# In[ ]:


def square(x):
    return x**2


# In[ ]:


out = square(2)


# In[ ]:


print(out)


# # lambda expressions

# In[ ]:


def times2(var):
    return var*2


# In[ ]:


times2(2)


# In[ ]:


lambda var: var*2


# # map and filter

# In[ ]:


seq = [1,2,3,4,5]


# In[ ]:


map(times2,seq)


# In[ ]:


list(map(times2,seq))


# In[ ]:


list(map(lambda var: var*2,seq))


# In[ ]:


filter(lambda item: item%2 == 0,seq)


# In[ ]:


list(filter(lambda item: item%2 == 0,seq))


# # methods

# In[ ]:


st = 'hello my name is Sam'


# In[ ]:


st.lower()


# In[ ]:


st.upper()


# In[ ]:


st.split()


# In[ ]:


tweet = 'Go Sports! #Sports'


# In[ ]:


tweet.split('#')


# In[ ]:


tweet.split('#')[1]


# In[ ]:


d


# In[ ]:


d.keys()


# In[ ]:


d.items()


# In[ ]:


lst = [1,2,3]


# In[ ]:


lst.pop()


# In[ ]:


lst


# In[ ]:


'x' in [1,2,3]


# In[ ]:


'x' in ['x','y','z']


# # Thanks !! ;)
