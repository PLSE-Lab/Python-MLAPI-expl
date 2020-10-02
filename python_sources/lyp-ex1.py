#!/usr/bin/env python
# coding: utf-8

# # Larn Yersel Python
# ## Exercise 1

# This exercise set contains a number of questions. Comments and answers where needed are provided at the end.
# 
# In code sections a hashtag is used for a a *comment*. This is a line of text that is used for explanation. It is not code and is ignored by Python

# 1) Read through the next code blocks. Try to predict what the output will be and then run the code.

# In[ ]:


# 1.1
print(1+3*7)


# In[ ]:


#1.2
print(5+12/4)


# In[ ]:


#1.3
print(5+12//4)


# In[ ]:


#1.4
a = 5
print(5*a)


# In[ ]:


#1.5
a = '5' 
print(5*a)


# In[ ]:


#1.6
print(2>1)


# In[ ]:


#1.7
a = 2>1
b = 3>1
print(a)
print(a+b)


# In[ ]:


#1.8
a = 'Hello '
b = 'World'
print(a+b)
print(b+a)


# In[ ]:


#1.9
list1 = [1,2,3,'a','b']
print(list1[1]+list1[2])
print(list1[3]+list1[4])


# In[ ]:


#1.10
list1[3] = 77
print(list1)


# 2. The next set of fragments use a method called *slicing* to create parts of lists or strings. The syntax `mylist[a:b]` pulls out the part of `mylist` starting at index `a` and ending at index `b-1` (this means you get `b-a` items)

# In[ ]:


#2.1
list2 = [0,1,2,3,4,5,6,7,8]
sublist1 = list2[2:5]
print(sublist1)


# In[ ]:


#2.2
newlist1 = list2[5:9]+list2[0:5]
print(newlist1)


# In[ ]:


#2.3
newlist2 = list2[5:]+list2[:5]
print(newlist2)


# In[ ]:


#2.4
word = 'hello'
letter1 = word[0]
letter2 = word[0].upper()
print(letter1)
print(letter2)


# In[ ]:


#2.5
badname = 'brian'
goodname = badname[0].upper()+badname[1:]
print(goodname)


# 3. The next set of code fragments all contain errors. Try to spot the error then run the code to see if you are correct.

# In[ ]:


#3.1
a = 3.1
print(A)


# In[ ]:


#3.2
a = 3.1
print a


# In[ ]:


#3.3
list3 = ['a','b','c','d','e','f']
print(list3[6])


# Comments on individual parts
# 
# 1. 
#    1. Integer operations give an integer answer
#    2. Division leads to a float anwer
#    3. // gives the integer part so leads to an integer answer
#    4. No comment
#    5. In this case multiplying by 5 concatenates 5 copies of the string.
#    6. Logical tests are evaluated to True or False
#    7. True=1 and False=0 in arithmetic
#    8. Did you spot that the second one would look like a single word because of the missing space?
#    9. Python uses + to mean different things depending on the type of variable.
#    10. You can change items in a list.
# 2. 
#    1. This is just a demonstration of the method
#    2. You can combine your new sublists. This one rearranges the original
#    3. This demonstrates that you can leave out a value to indicate that you want to go to the end or start at the beginning.
#    4. There are many different string functions. `.upper()` converts to upper case. You can find find more with an internet search for Python String Functions
#    5. There is a string function that converts the first letter to uppercase, but you don't need lots of functions. Ingenuity trumps obscure functions every time.
# 
# 
