#!/usr/bin/env python
# coding: utf-8

# > **Hello Pythonista's,**
# 
# The NoteBook includes the following,
# 
# * Data structures
# * Loops
# * Conditional statements
# * Functions and helper
# * List
# * Tuple
# * Dictionary
# * And some basics of Numpy and math Modules
# 

# In[ ]:


#Print hello world
print("Hello world!")


# In[ ]:


a = 2 #Assinging integer value 2 to variable a
print(a)
type(a) #Which will return the type of variable


# In[ ]:


a = 2.44 #Assinging float value 2.44 to variable a
print(a)
type(a) #Which will return the type of variable


# In[ ]:


a = 6 #assinging value 6 to variable a
b = 3 #assinging value 3 to variable b
c = a / b 
print(c)
type(c) #In case of division it always returns float value


# In[ ]:


a = 7 #assinging value 6 to variable a
b = 3 #assinging value 3 to variable b
c = a // b # The // operator gives us a result that's rounded down to the next integer.
print(c)


# In[ ]:


a = 700 #assinging value 6 to variable a
b = 3 #assinging value 3 to variable b
c = ( a * b ) / 100 # returns 21
print(c)
d = a + b / 100 # / and * has more weight than + , so 3/100 = 0.03 and then it is added to 700.
print(d)


# **Minimum and Maximum number**

# In[ ]:


print(max(10, 20, 30))
print(min(20, 10, 30))


# In[ ]:


print(abs(-21)) # print absolute value
print(abs(21))


# In[ ]:


print(int('201') + 1) #converting string to number
print(float(201)) #converting int to float


# In[ ]:


pi = 3.14159 # approximate
diameter = 3

radius = diameter / 2;
print(radius)
area = pi * ( radius ** 2)
print(area)


# **Lists**
# 
# List are array like datastructure in python.

# In[ ]:


a = [1, 2, 3]
b = [3, 2, 1]
print("value of a:", a)
print("value of b:", b)
print("Picking value from index [1]:",a[1]) # list index start from 0
a, b = b, a #easy way to swap values
print("value of a after swap:", a)
print("value of b after swap:", b)


# In[ ]:


print(7%2) #modular operation


# In[ ]:


help(round) #helper function


# **Define a function call**

# In[ ]:


def sum(a, b):
    return a+b
print(sum(5, 6))


# In[ ]:


def sum(a, b):
    """
    This function will sum of two numbers, eg: sum(5, 6)
    """
    return a+b


# In[ ]:


help(sum)


# In python you can return from a function without adding return value or return statement

# In[ ]:


def print_hello(name="world"):
    print("Hello,", name)
    
print_hello()
print_hello(name="ThatCoder")
print_hello("Pythonistas")


# **Passing function as Argument**

# In[ ]:


def print_hello(name):
    print("Hello,", name)
    
def call(fn, args): //passing function as argument
    fn(args)
    
call(print_hello, "world")


# **Boolean and Conditional Expression**

# In[ ]:


a = True
b = False
print(a)
print(b)
type(a)


# In[ ]:


def larger(a):
    return a > 30
print(larger(10))
print(larger(40))


# In[ ]:


3.0 == 3


# In[ ]:


3.0 == "3"


# In[ ]:


def larger(a):
    return a > 30 and a == 40
print(larger(32))
print(larger(40))


# In[ ]:


def larger(a):
    return a > 30 or a == 40
print(larger(32))
print(larger(40))


# **If - Else Statement**

# In[ ]:


def if_cond(a):
    if a > 5:
        print(a, "is greater than 5")
    elif a > 2:
        print(a, "is greater than 2")
    else:
        print(a, "is less than 2 ")

if_cond(6)
if_cond(3)
if_cond(1)


# In[ ]:


def cond(a):
    b = 'heyhey' if a < 50 else 'hihi'
    print(b)
    
cond(45)
cond(55)


# **Lists**

# In[ ]:


a = [1, 2, 3]
print("1: ", a)
b = ["apple", "orange"]
print("2: ", b)

print("3: ", b[0])
print("4: ", a[-1]) # last element of array

print("5: ", a[1:3]); # start from first index and scan upto third element
print("6: ", a[:3]); #assumed start index and 0
print("7: ",a[2:]) # from index 2 to all element

c = [1, "red"]
print("8: ",c)

d = [
    [1, 2, 3],
    [3, 4, 5],
    [6, 7, 8]
]  # multi dimensional list

print("9: ",d)
print("10: ",d[0]) 
print("11: ",d[0][0])
print("12: ",d[0][0:2])


# In[ ]:


a = [1, 2, 3, 4, 5, 6]
print("length of List a", len(a))
b = [4, 3, 6, 1, 2, 5]
c = ["orange", "apple"]
print("Sorted List:", sorted(b))
print("Max number in List b:", max(b))
print("Max number in List b:", min(b))
print("Sorted String List:", sorted(c))


# In[ ]:


a = [1, 2, 3, 4, 5, 6]
a.pop();
print(a)
a.append(7);
print(a)
c = 4 + 3j #imaginary number (complex number)
print(c.imag)


# Loops

# For loop

# In[ ]:


a = [1, 2, 3, 4, 5, 6]
for i in a:
    print(i,end= " ") #print with space

for i in a:
    print(i) #print in new line

for i in range(2):
    print(i)


# In[ ]:


a = (2, 2, 2, 3, 3, 5) # look at this new way of writing list
b = 1
for value in a:
    b = value * b

print(b)
b = (
    (1, 2, 3),
    (2, 3, 4)
)
print(b)


# **While Loop**

# In[ ]:


i = 0
while i < 10:
    print(i, end=" ")
    i = i+1
print("\n")
i = 0
while i < 10:
    print(i, end=" ")
    i+=1


# **Strings**

# In[ ]:


a = 'Hello World'
b = 'There\'s a villain'
print(a)
print(b)
c = 'This is a car'
print(c.upper())
print(c.lower())
print(c.index("is"))
print( a + " " + b )


# **Dictionaries**
# 
# > Key - value pair ( like JSON in JavaScript )

# In[ ]:


a = {'one':1, 'two':2, 'three':3}
print(a['two'])
print("\n")

a['four'] = 4
print(a)
print("\n")

print("loop dict elemets:")
for i in a: 
    print(i + ":", a[i])


# In[ ]:


import math

print(math.log(32, 2))
print(math.pi)


# In[ ]:


from math import *

print(log(32, 2))
print(pi)


# **numpy **
# 
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

# In[ ]:


import numpy

a = numpy.random.randint(low=1, high=6, size=10)
print(a)
type(a)


# In[ ]:


print(dir(a)) #dir() function attempts to return a list of valid attributes for the given object.


# In[ ]:


print(a.mean())


# In[ ]:


print(a.sum())


# In[ ]:


b= a.tolist() #converting numpy array to list
print(b)


# In[ ]:


a.ravel() # returing as flatened array


# In[ ]:


a < 3


# In[ ]:


x = [ [1, 2, 3], [4, 5, 6] ]
x_numpy = numpy.asarray(x) #list to numpy array
print(x_numpy)


# In[ ]:




