#!/usr/bin/env python
# coding: utf-8

# ## Getting Familiar with Print, Variables and Types
# 
# #### Run the Examples below to see how python deals with different types and variables

# In[ ]:


# Run this example - the blue triangle at the side

print("Hello World! Again")


# 1. Try adding a code block and writing the above code yourself with a sentence (string) your choice!

# In[ ]:


#Run this example - print an Int
print(5)


# 2. Try adding a code block and writing the above code yourself with an Int of your choice!

# In[ ]:


# Run this example - print a float
print(3.5)


# 3. Try adding a code block and writing the above code yourself with a float of your choice!

# In[ ]:


# Run this example - print a variable that HOLDS a string

x = "Hi, how are you?"
print(x)


# 4. Try adding a code block and writing the above code yourself by creating a variable that holds ANY type (Int, String, Float) and then printing that vairable!

# In[ ]:


# Run this example - printing multiple variables

myString1 = "Hello world!"
myString2 = "Hello sun!"
myString3 = "Hello moon!"
print(myString1, myString2, myString3)


# 5. Try adding a code block and writing the above code yourself by creating 4 variables that holds ANY type (Int, String, Float) and then printing those vairables in one print statement!

# In[ ]:


# Run this example - printing multiple lines

print(
"""
Mary had a little lamb,
Its fleece was white as snow,
And every where that Mary went
The lamb was sure to go;
""")


# 6. Try adding a code block and writing the above code yourself with a multiline string of your choice! Pay attention to the number of quotations in this print statement. 

# In[ ]:


# Run this example - printing multiple lines

print("Mary had a little lamb,")
print("Its fleece was white as snow,")
print("And every where that Mary went")
print("The lamb was sure to go;")


# 7. Try adding a code block and writing the above code yourself with any multiline string of your choice!

# In[ ]:


# Run this example - formatting print statements using a variable 
# Notice how you can place the value of s = 23 anywhere in the print statment using the {} symbol and the .format() function
s = 23
print('My integer is: {}'.format(s))


# 8. Try adding a code block and writing the above code yourself - create a variable that holds any type (Int, Float, String) print that variable using the print and format statement AND make the variable be printed in the middle of you string by placing the {} in the right position. 

# In[ ]:


# Run this example - formatting multiple variablea of different types

x = 4500
f = 356.6
s = 'Hi!'

print('My integer is {}, my float is {} and my string is {}'.format(x,f,s))


# 9. Try adding a code block and writing the above code yourself - create 3 variables that hold any type (Int, Float, String) print that variable using the print and format statement AND make the variable be printed in different places in the string using the {}. 

# In[ ]:


#Run this Example - Adding Integers and printing the result
print(5 + 5)


# In[ ]:


# Run this Example - Adding STRINGs together called concatenating strings
# What is the difference between the above code that added the integers together and this code that is adding two strings together???

print("5" + "5")


# In[ ]:


#Run this Example - Subtracting Integers and printing the result
print(7-9)


# In[ ]:


#Run this Example - Multiplying Integers and printing the result
print(6*7)


# In[ ]:


#Run this Example - Dividing Integers and printing the result
print(12/3)


# In[ ]:


#Run this Example - Adding Integers, storing them in a variable and printing the result
# This can be done with adding, subtracting, multiplication and many more operations. 

x = 5 + 5

print(x)


# In[ ]:


#Run this Example - storing integers in 2 variables and then adding the variables together in another variable then printing the result
# This can be done with adding, subtracting, multiplication and many more operations. 

x = 5
y = 7
z = x + y

print(z)


# In[ ]:


# Run this Example - you can also add Strings together, called concatenation, that are stored in variables 

x = "Hello"
y = "World"

print(x + y)
print(x +" "+ y)


# ## Try the Exercise below on your own to get a sense of integers, variables, print and format

# In[ ]:


#Exercise 1 - Create a program that creates two integers, 
#assigns them to a variable and prints the sum:
# My sum is: [number]

