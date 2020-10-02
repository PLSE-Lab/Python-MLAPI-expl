#!/usr/bin/env python
# coding: utf-8

# # Data Structures

# Data Structure is a way of collecting and organising data in such a way that we can perform operations on these data in an effective way. Data Structures is about rendering data elements in terms of some relationship, for better organization and storage. 
# 
# In simple language, Data Structures are structures programmed to store ordered data, so that various operations can be performed on it easily. It represents the knowledge of data to be organized in memory. It should be designed and implemented in such a way that it reduces the complexity and increases the efficiency.
# 
# **For Example:**
# - Arrays
# - Lists
# - Sets
# - Trees
# - Graphs

# ---
# ## Lists

# Lists are the most commonly used data structure. Think of it as a sequence of data that is enclosed in square brackets and data are separated by a comma. Each of these data can be accessed by calling it's index value.
# 
# Lists are declared by just equating a variable to '[ ]' or list.

# In[1]:


a = []


# In[2]:


print(type(a))


# One can directly assign the sequence of data to a list x as shown.

# In[3]:


x = ['apple', 'orange']


# ### Indexing

# In python, Indexing starts from 0. Thus now the list x, which has two elements will have apple at 0 index and orange at 1 index.

# In[4]:


x[0]


# Indexing can also be done in reverse order. That is the last element can be accessed first. Here, indexing starts from -1. Thus index value -1 will be orange and index -2 will be apple.

# In[5]:


x[-1]


# As you might have already guessed, x[0] = x[-2], x[1] = x[-1]. This concept can be extended towards lists with more many elements.

# In[6]:


y = ['carrot','potato']


# Here we have declared two lists x and y each containing its own data. Now, these two lists can again be put into another list say z which will have it's data as two lists. This list inside a list is called as nested lists and is how an array would be declared which we will see later.

# In[7]:


z  = [x,y]
print(z)


# Indexing in nested lists can be quite confusing if you do not understand how indexing works in python. So let us break it down and then arrive at a conclusion.
# 
# Let us access the data 'apple' in the above nested list.
# First, at index 0 there is a list ['apple','orange'] and at index 1 there is another list ['carrot','potato']. Hence z[0] should give us the first list which contains 'apple'.

# In[8]:


z1 = z[0]
print(z1)


# Now observe that z1 is not at all a nested list thus to access 'apple', z1 should be indexed at 0.

# In[9]:


z1[0]


# Instead of doing the above, In python, you can access 'apple' by just writing the index values each time side by side.

# In[10]:


z[0][0]


# If there was a list inside a list inside a list then you can access the innermost value by executing z[ ][ ][ ].

# ### Slicing

# Indexing was only limited to accessing a single element, Slicing on the other hand is accessing a sequence of data inside the list. In other words "slicing" the list.
# 
# Slicing is done by defining the index values of the first element and the last element from the parent list that is required in the sliced list. It is written as parentlist[ a : b ] where a,b are the index values from the parent list. If a or b is not defined then the index value is considered to be the first value for a if a is not defined and the last value for b when b is not defined.

# In[11]:


num = [0,1,2,3,4,5,6,7,8,9]


# In[13]:


print(num[0:4])
print(num[4:])


# You can also slice a parent list with a fixed length or step length.

# In[14]:


num[:9:3]


# ### Built in List Functions

# To find the length of the list or the number of elements in a list, **len( )** is used.

# In[15]:


len(num)


# If the list consists of all integer elements then **min( )** and **max( )** gives the minimum and maximum value in the list.

# In[16]:


min(num)


# In[17]:


max(num)


# Lists can be concatenated by adding, '+' them. The resultant list will contain all the elements of the lists that were added. The resultant list will not be a nested list.

# In[18]:


[1,2,3] + [5,4,7]


# There might arise a requirement where you might need to check if a particular element is there in a predefined list. Consider the below list.

# In[19]:


names = ['Earth','Air','Fire','Water']


# To check if 'Fire' and 'Renton' is present in the list names. A conventional approach would be to use a for loop and iterate over the list and use the if condition. But in python you can use 'a in b' concept which would return 'True' if a is present in b and 'False' if not.

# In[20]:


'Fire' in names


# In[104]:


'Renton' in names


# In a list with elements as string, **max( )** and **min( )** is applicable. **max( )** would return a string element whose ASCII value is the highest and the lowest when **min( )** is used. Note that only the first index of each element is considered each time and if they value is the same then second index considered so on and so forth.

# In[22]:


mlist = ['bzaa','ds','nc','az','z','klm']


# In[24]:


print(max(mlist))
print(min(mlist))


# Here the first index of each element is considered and thus z has the highest ASCII value thus it is returned and minimum ASCII is a. But what if numbers are declared as strings?

# In[25]:


nlist = ['1','94','93','1000']


# In[27]:


print(max(nlist))
print(min(nlist))


# Even if the numbers are declared in a string the first index of each element is considered and the maximum and minimum values are returned accordingly.

# But if you want to find the **max( )** string element based on the length of the string then another parameter 'key=len' is declared inside the **max( )** and **min( )** function.

# In[29]:


print(max(names, key=len))
print(min(names, key=len))


# But even 'Water' has length 5. **max()** or **min()** function returns the first element when there are two or more elements with the same length.
# 
# Any other built in function can be used or lambda function (will be discussed later) in place of len.
# 
# A string can be converted into a list by using the **list()** function.

# In[30]:


list('hello')


# **append( )** is used to add a element at the end of the list.

# In[31]:


lst = [1,1,4,8,7]


# In[33]:


lst.append(1)
print(lst)


# **count( )** is used to count the number of a particular element that is present in the list. 

# In[34]:


lst.count(1)


# **append( )** function can also be used to add a entire list at the end. Observe that the resultant list becomes a nested list.

# In[35]:


lst1 = [5,4,2,8]


# In[36]:


lst.append(lst1)
print(lst)


# But if nested list is not what is desired then **extend( )** function can be used.

# In[38]:


lst.extend(lst1)
print(lst)


# **index( )** is used to find the index value of a particular element. Note that if there are multiple elements of the same value then the first index value of that element is returned.

# In[39]:


lst.index(1)


# **insert(x,y)** is used to insert a element y at a specified index value x. **append( )** function made it only possible to insert at the end. 

# In[ ]:


lst.insert(5, 'name')
print lst


# **insert(x,y)** inserts but does not replace element. If you want to replace the element with another element you simply assign the value to that particular index.

# In[40]:


lst[5] = 'Python'
print(lst)


# **pop( )** function return the last element in the list. This is similar to the operation of a stack. Hence it wouldn't be wrong to tell that lists can be used as a stack.

# In[41]:


lst.pop()


# Index value can be specified to pop a ceratin element corresponding to that index value.

# In[42]:


lst.pop(0)


# **pop( )** is used to remove element based on it's index value which can be assigned to a variable. One can also remove element by specifying the element itself using the **remove( )** function.

# In[43]:


lst.remove('Python')
print(lst)


# Alternative to **remove** function but with using index value is **del**

# In[44]:


del lst[1]
print(lst)


# The entire elements present in the list can be reversed by using the **reverse()** function.

# In[45]:


lst.reverse()
print(lst)


# Note that the nested list [5,4,2,8] is treated as a single element of the parent list lst. Thus the elements inside the nested list is not reversed.
# 

# [](http://)For lists containing string elements, **sort( )** would sort the elements based on it's ASCII value in ascending and by specifying reverse=True in descending.

# In[59]:


names.sort()
print(names)
names.sort(reverse=True)
print(names)


# To sort based on length key=len should be specified as shown.

# In[61]:


names.sort(key=len)
print(names)
names.sort(key=len,reverse=True)
print(names)


# ### Copying a list

# Most of the new python programmers commit this mistake. Consider the following,

# In[62]:


lista= [2,1,4,3]


# In[64]:


listb = lista
print(listb)


# Here, We have declared a list, lista = [2,1,4,3]. This list is copied to listb by assigning it's value and it get's copied as seen. Now we perform some random operations on lista.

# In[65]:


lista.pop()
print(lista)
lista.append(9)
print(lista)


# In[66]:


print(listb)


# listb has also changed though no operation has been performed on it. This is because you have assigned the same memory space of lista to listb. So how do we fix this?
# 
# If you recall, in slicing we had seen that parentlist[a:b] returns a list from parent list with start index a and end index b and if a and b is not mentioned then by default it considers the first and last element. We use the same concept here. By doing so, we are assigning the data of lista to listb as a variable.

# In[68]:


lista = [2,1,4,3]


# In[69]:


listb = lista[:]
print(listb)


# In[70]:


lista.pop()
print(lista)
lista.append(9)
print(lista)


# In[71]:


print(listb)


# ---
# ## Tuples

# Tuples are similar to lists but only big difference is the elements inside a list can be changed but in tuple it cannot be changed. Think of tuples as something which has to be True for a particular something and cannot be True for no other values. For better understanding, Recall **divmod()** function.

# In[72]:


xyz = divmod(10,3)
print(xyz)
print(type(xyz))


# Here the quotient has to be 3 and the remainder has to be 1. These values cannot be changed whatsoever when 10 is divided by 3. Hence divmod returns these values in a tuple.

# To define a tuple, A variable is assigned to paranthesis ( ) or tuple( ).

# In[73]:


tup = ()
tup2 = tuple()


# If you want to directly declare a tuple it can be done by using a comma at the end of the data.

# In[74]:


27,


# 27 when multiplied by 2 yields 54, But when multiplied with a tuple the data is repeated twice.

# In[75]:


2*(27,)


# Values can be assigned while declaring a tuple. It takes a list as input and converts it into a tuple or it takes a string and converts it into a tuple.

# In[76]:


tup3 = tuple([1,2,3])
print(tup3)
tup4 = tuple('Hello')
print(tup4)


# It follows the same indexing and slicing as Lists.

# In[77]:


print(tup3[1])
tup5 = tup4[:3]
print(tup5)


# ### Mapping one tuple to another

# In[79]:


(a,b,c)= ('alpha','beta','gamma')


# In[80]:


print(a,b,c)


# In[81]:


d = tuple('Mark Renton')
print(d)


# ### Built In Tuple functions

# **count()** function counts the number of specified element that is present in the tuple.

# In[82]:


d.count('a')


# **index()** function returns the index of the specified element. If the elements are more than one then the index of the first element of that specified element is returned

# In[83]:


d.index('a')


# ---
# ## [](http://)Sets

# Sets are mainly used to eliminate repeated numbers in a sequence/list. It is also used to perform some standard set operations.
# 
# Sets are declared as set() which will initialize a empty set. Also set([sequence]) can be executed to declare a set with elements

# In[87]:


set1 = set()
print(type(set1))


# In[88]:


set0 = set([1,2,2,3,3,4])
print(set0)


# elements 2,3 which are repeated twice are seen only once. Thus in a set each element is distinct.

# ### Built-in Functions

# In[91]:


set1 = set([1,2,3])


# In[92]:


set2 = set([2,3,4,5])


# **union( )** function returns a set which contains all the elements of both the sets without repition.

# In[93]:


set1.union(set2)


# **add( )** will add a particular element into the set. Note that the index of the newly added element is arbitrary and can be placed anywhere not neccessarily in the end.

# In[94]:


set1.add(0)
set1


# **intersection( )** function outputs a set which contains all the elements that are in both sets.

# In[95]:


set1.intersection(set2)


# **difference( )** function ouptuts a set which contains elements that are in set1 and not in set2.

# In[96]:


set1.difference(set2)


# **symmetric_difference( )** function ouputs a function which contains elements that are in one of the sets.

# In[97]:


set2.symmetric_difference(set1)


# **issubset( ), isdisjoint( ), issuperset( )** is used to check if the set1/set2 is a subset, disjoint or superset of set2/set1 respectively.

# In[98]:


set1.issubset(set2)


# In[99]:


set2.isdisjoint(set1)


# In[100]:


set2.issuperset(set1)


# **pop( )** is used to remove an arbitrary element in the set

# In[101]:


set1.pop()
print(set1)


# **remove( )** function deletes the specified element from the set.

# In[102]:


set1.remove(2)
set1


# **clear( )** is used to clear all the elements and make that set an empty set.

# In[103]:


set1.clear()
set1

