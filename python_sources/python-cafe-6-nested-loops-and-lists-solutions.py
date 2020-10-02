#!/usr/bin/env python
# coding: utf-8

# # Nested Loops
# 
# A nested loop is the same as a regular loop but with an added level of loop execution. The inner loop always finishes looping through the entire range before the outer loop. When creating an outer loop you can create as many loops within that loop, depending on what you are trying to achieve. Let's look at some examples as this is usually hard to grasp. 

# In[ ]:


# Your Code Goes Here
for i in range(0, 4):
    print("outer loop ", i)
    for j in range(1, 5):
        print(j)
        


# In[ ]:


# Your Code Goes Here
for i in range(0, 4):
    print("first loop ", i)
    for j in range(1, 5):
        print("second loop",j)
        for k in range(1,10):
            print("third loop", k)
        


# # Lists 
# - List is a type of container in Data Structures, which is used to store multiple data at the same time
# 
# - The elements in a list are indexed according to a definite sequence and the indexing of a list is done with 0 being the first index
# 
# - Lists are ordered and have a definite count
# 
# - A single list may contain DataTypes like Integers, Strings, as well as Objects.
# 
# - Lists are mutable, and hence, they can be altered even after their creation.
# 
# - Warning: Avoid creating a list named "list". This is a function in Python that converts other data structures into lists
# 
# Let's try some examples now!
# 

# In[ ]:


# Your Code Goes Here
fruits = ["apple", "banana", 1]
print(fruits)


# In[ ]:


#Accessing the first element
fruits[0]


# In[ ]:


#Replacing an item
fruits[1] = "mango"
print(fruits)


# In[ ]:


####Multi-Dimensional List: 
l = [["list","of"], ["lists"]]


# In[ ]:


#Accessing elements
l[0][1]
l[1]


# In[ ]:


# Adding elements
a = [1,2,3,4,5]
b = [6,7,8,9,10]
c = [11,12,13,14,15]

#Single element
a.append(34)
print(a)

#Multiple elements
a.extend([6,7,8])
print(a)

#Concatenating lists
a+b+c


# In[ ]:


#Dropping elements

#Dropping by index
b = [10,11,12,13,14,15,15,15]

#Dropping the 4th element
b.pop(3)
print(b)
#obs: pop removes the last element if you don't specify the index value

#Dropping by value
b.remove(12)
print(b)


# ### Looping Through Lists
# 
# You can also loop through lists just as you can loop through a range of numbers and a string. There are actually many data structure you can loop through. Right now let's focus on looping through lists.
# 

# In[ ]:


# Your Code Goes Here
basketball = ['Toronto', 'Raptors', 'NBA', 'Champions', 1]

for x in basketball:
    print(x)


# ### Nested Loops and Lists
# 
# When we have Multi-Dimensional lists, a nested loop helps us access the elements of a list in each level. 
# 
# Let's try an example!

# In[ ]:


#Nested lists

a = [[1,2,3],['a','b','c'],['cat','dog','mouse']]

#Accessing the first dimension of the list
for i in a:
 print(i)

#Accessing each element within each list of the multi-dimension list
for i in a:
    print(i)
    for j in i:
        print(j)


# In[ ]:


#Creating a list with common elements
aaa = [1,2,3,4]
bbb = [3,4,5,6,7,9]

#Create empty list
common_num = []

for a in aaa:
    for b in bbb:
        if (a==b):
            common_num.append(a)

print(common_num)


# # Putting It All Together
# 
# Try the excercise below with the array numbers already set up for you. You will need to use if statements, booleans, a for loop, and the % symbol. 

# In[ ]:


###Exercise 4 - Write a Python program separate even numbers from odd numbers

numbers = [0,1,5,7,88,90,34,55,67,890]

#Answer
odd_numbers=[]
even_numbers=[]

for i in range(0, len(numbers)):
    if (numbers[i] == 0):
        continue
    elif(numbers[i] % 2 != 0):
        odd_numbers.append(numbers[i])
    else:
        even_numbers.append(numbers[i])

print(odd_numbers)
print(even_numbers)

