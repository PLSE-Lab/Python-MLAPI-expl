#!/usr/bin/env python
# coding: utf-8

# ## 1. More Numbers and Types!
# 
# Run through these calculation examples to get familiar with number operations in python.
# 
# ### Don't forget to hit the blue copy and edit button in the top right corner in order to open the kernel that will let you edit this notebook! 

# ### Exercise 1:

# In[ ]:


pi = 3.14159 # approximate
diameter = 3

# Create a variable called 'radius' equal to half the diameter

# Create a variable called 'area', using the formula for the area of a circle: pi times the radius squared


# ### Try these numerical operations yourself!
# 
# Refer to Python Coding Cafe 3 for more examples of different numerical operations to play around with.

# In[ ]:


x = 22
y = 2
w = x * y
z = x / y


# ### Last one...

# In[ ]:


#Assume variable a holds 21 and variable b holds 10, then..
a = 21
b = 10
c = 0

c = a + b
print("Line 1 - Value of c is ", c)

c = a - b
print("Line 2 - Value of c is ", c )

c = a * b
print("Line 3 - Value of c is ", c )
c = a / b
print("Line 4 - Value of c is ", c )

c = a % b
print("Line 5 - Value of c is ", c)

a = 2
b = 3
c = a**b 

print( "Line 6 - Value of c is ", c)

a = 10
b = 5
c = a//b 
print("Line 7 - Value of c is ", c)


# ### Now let's try some casting type examples.

# ### Exercise 2:
# Create a program that creates two integers, assigns them to a variable and prints the sum: 
# My sum is [number]

# In[ ]:


x = 2
y = 4
z = x + y
#print('My sum is: ', z)
print("My sum is: {}".format(z))
#print("My sum is: %i" %(z))


# ### Exercise 3:
# Write a program where you ask for a string, convert it into an integer and print out an integer

# In[ ]:


# Hint use the input() function showed in python cafe 1 - extra exercises 

myString = '23'
type(myString)
myInteger = int(myString)

print("My number is: {}".format(myInteger))
type(myInteger)

s = str(myInteger)
type(s)


# ### Exercise 4:
# 
# Converting float into an integer

# In[ ]:


myFloat = 4.1
print(myFloat)
type(myFloat)
int(myFloat)
round(4.1)


# ### Exercise 5:
# Converting different types to string
# 

# In[ ]:


x = str("s1") # x will be 's1'
print(x)

y = str(2)    # y will be '2'
print(y)

z = str(3.0)  # z will be '3.0
print(z)


# ### [Explore online](https://docs.python.org/3/library/functions.html) all the capabilities of these functions - you learn through simply playing around with the code and building programs. 
# 
# #### Add your own code blocks and play around with more print, input and format functions.
# 
# #### Add your own code blocks and play around with using variables with print, input and format functions. 
