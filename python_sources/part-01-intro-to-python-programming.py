#!/usr/bin/env python
# coding: utf-8

# # Introduction to Python

# ![python](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQcABXOYVKzspiTdb9sW-aUzccv66s-KDUbsool5X4fTuIzvKhE&s)
# ## Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum and first released in 1991.
# 1. *Powerful easy to use language*
# 2. *Not a whole lot of syntax to learn*
# 3. *Each concepts has a bunch of example*

# ![Python used by](https://qph.fs.quoracdn.net/main-qimg-0613adfdf47a4691e947539c31be0247)
# <ul>
#     <a href='#0'><li>Introduction</li></a>
#     <a href='#1'><li>First Python Program</li></a>
#     <a href='#2'><li>Python program to draw a shape</li></a>
#     <a href='#3'><li>Creating variables</li></a>
#     <a href='#4'><li>String functions</li>
#     <a href='#5'><li>Working with Numbers</li></a>
#     <a href='#6'><li>Import external code into our python file</li></a>
#     <a href='#7'><li>Getting input from the user</li></a>
#     <a href='#8'><li>Building a basic calculator in python</li></a>
#     <a href='#9'><li>Build a mad libs game in python</li></a>
#     <a href='#10'><li>Working With lists in Python</li></a>
#     <a href='#11'><li>List function</li>
#     <a href='#12'><li>Tuples in python</li></a>
#     <a href='#13'><li>Using Functions</li></a>
#     <a href='#14'><li>Functions with variables</li></a>
#     <a href='#15'><li>if Statement in python</li></a>
#     <a href='#16'><li>Using camparisions with if statements</li></a>
#     <a href='#17'><li>Build an Advanced calculator</li></a>
#     <a href='#18'><li>While loops</li></a>
#  </ul>

# <p id='0'><h1> INSTALLATIONS</h1></p>
# * Install python python.org/download
# * download python 3.7.2
# * python 3 is maintained actively 

# # Editor
# * Choose a text editor
# * PyCharms community version

# # <p id='2'><h1>1. First Python Program</h1></p>
# 1. Setup and Hello World
# pycharm - create new project
# * seclect location
# * select intrepretor
# * click new python project
# 
# Write print statements

# In[ ]:


#this is a comment
#printing hello world
print("Hello World")


# <p id='2'><h1> 2. Python program to draw a shape<h1></p>

# In[ ]:


#Using forward, pipe and underscore slash to draw a triangle
print("    /|")
print("   / |")
print("  /  |")
print(" /   |")
print("/____|")


# <p id='3'><h1>  3. Creating variables</h1></p>

# In[ ]:


#Using variables in python
#printing a story using python
print("I saw a man named john")
print("he was 67 and white")
print("john loved to play with his grand son kattie")
print("but he didnt like being 67")


# Suddenly I decided to change the name of the person to mike and for that i need to manually change it John-to-mike

# In[ ]:


#Like this
print("I saw a man named Mike")
print("he was 67 and white")
print("mike loved to play with his grand son kattie")
print("but he didnt like being 67")

#but what if i had a long long story say about 20 pages or that 480


# Use the variable to store characters name and age

# In[ ]:


#creating variable
#variable_name = "vale_of_variable"
person_Name = "Mike"
person_Age = "57"
print("---------------------------------------------")
#usage : wrapping what to be printed in "" add a pluse sign + and call the variable + "statement"
print("I saw a man named " + person_Name + ",")


# In[ ]:


print("I saw a man named " + person_Name + ",")
print("he was " + person_Age + " and white")
print( person_Name + " loved to play with his grand son kattie")
print("but he didnt like being " + person_Age + ".")


# Change the value of a variable "update the character name" half the way in the story, lets say last 2 lines

# In[ ]:


print("I saw a man named " + person_Name + ",")
print("he was " + person_Age + " and white")
print("---------------------------------------------")
#changing name to beth and age to 60
person_Name = "Beth"
person_Age = "60"
print(""+ person_Name + " loved to play with his grand son kattie")
print("but he didnt like being " + person_Age + ".")


# What we used so far was a string, we can also store any type of number the variables store these different values
# 1. character: "A", "j"
# 2. string: "the name is", "lara loves"
# 3. Number: 34, 5, 5.576.
# 4. Boolean value: True, False

# In[ ]:


#storing string as variables
#use Qoutation mark: "string Text goes here"
print("learn python")
print("---------------------------------------------")
#create new line in the string "Learn\npython"
print("learn\npython")


# store this string into a variable and print it out
# concatenation using + to append
# 

# In[ ]:


phrase = "Learn Python"
print(phrase)
print(phrase + "is cool")


# <p id='4'><h1> 4. String functions</h1></p>
# 1. lower:makes all lower case
# 2. upper: makes phrase upper case
# 3. isupper: returns True/ False
# 4. use two function together
# 5. len: how many alphabets are inside the phrase
# 6. replace

# In[ ]:


print(phrase.lower())
print(phrase.upper())
print(phrase.upper().isupper())
print(phrase.isupper())
print(len(phrase))
print(phrase.replace("Learn","Awesome"))


# Print index values, string gets indexed starting with zero
# index says where that value is lacated

# In[ ]:


print(phrase[0])
print(phrase[1])
print(phrase[6])
print(phrase[10])
print(phrase[5])
print(phrase[3])
print("---------------------------------------------")
#printing the location
print(phrase.index("L"))
print(phrase.index("P"))
print(phrase.index("e"))
print(phrase.index("t"))
print(phrase.index("y"))


# <p id='5'><h1> 5. Working with Numbers</h1></p>

# In[ ]:


#ptinting a number
print(2)
#or decimals
print(2.333)
#also negative numbers
print(-234.8)


# Arithmetics: Addition, Sub, multi

# In[ ]:


print(3+2)
print(3-2)
print(3*2)
#complex eqations using paranthesis
print(3*2 + (56-89)-(6+2))
# A mod function
print(22%2)
#power functio
print(pow(4,5))
#min/max
print(max(4,5))
print(min(4,5))
#round
print(round(-5.6))
#this prints -6


# printing numbers with string gets an error

# In[ ]:


my_num = 5
print(5 + "printing digit")


# In[ ]:


my_num = 5
print("5" + " printing digit")


# <p id='6'><h1> 6. Import external code into our file</h1></p>
# * lets import math functions from a library called math

# In[ ]:


from math import *
#math module gives us access to lot of different math function
my_num = -5
print(sqrt(36))


# <p id='7'><h1>  7. Getting input from the user</h1></p>
# 1. take it and store it into a variable
# 2. process it
# 3. print the processed values

# In[ ]:


name = input("Enter your name:")
age = input("Enter your age:")
print("Hello " + name + "! you are " + age)


# <p id='8'><h1>  8. Building a basic calculator in python</h1></p>
# steps
# 1. cretate 2 variables
# 2. get values fromthe user
# 3. add it 
# 3. print the added value

# In[ ]:


num1 = input("Enter a number: ")
num2 = input("Enter another number: ")
result = num1 + num2
print(result)
print("---------------------------------------------")
# gets 34 which is wrong this is because python things its concatenation of two strings
#using int() function to covertthese into numbers if using decimal use float() rather than int()
num3 = input("Enter a number: ")
num4 = input("Enter another number: ")
result = int(num3) + int(num4)
print(result)


# <p id='9'><h1>  9. Build a mad libs game in python</h1></p>
# ![madlibs](https://cdn.shopify.com/s/files/1/2411/3585/products/SHOPIFY_Bridal_Mad_Libs.jpg?v=1545471747)

# 1. take 3 variables
# 2. Make a mad lib
# 3. print the madlib with user input

# In[ ]:


color = input("Enter a color:")
plural_noun = input("Enter a plural_noun:")
celebrity = input("Enter a celebrity")
print("Roses are" + color)
print(plural_noun +"are blue")
print("I love "+ celebrity)


# <p id='10'><h1>  10. Working with lists in python</h1></p>
# * large amount of data
# * organize and keep track of data
# * use later in the program
# 

# In[ ]:


#create a list
# lets give a discriptive name
deserts = ["Namib Desert","Atacama Desert","Sahara Desert","Gobi Desert","Mojave Desert" ]
print("---------------------------------------------")
#print the whole list
print(deserts)
print("---------------------------------------------")
#print specific elements 
#use index to print the specific elements in the list use [] with the list name
print(deserts[4])
#index from the back of the list
print(deserts[-1])
#select the portion of the list
print("---------------------------------------------")
print(deserts[1:3])
print(deserts[1:])
print(deserts[:3])
#Access 


# <p id='11'><h1> 11. List function</h1></p>
# * using function with lists 
# * modify the list

# In[ ]:


lucky_numbers = [4,5,6,6,7,8,2,2,3,3]
friends = ["mike","john", "lara", "anne"]
# 1 Printing all elements of the list
print(lucky_numbers)
print(friends)
print("---------------------------------------------")
#extend() to addon to the list
friends.extend(lucky_numbers)
print(friends)
print("---------------------------------------------")
#add individual elements to the list
friends = ["mike","john", "lara", "anne"]
friends.append("Karen")
print(friends)
print("---------------------------------------------")
#insert() 
friends.insert(1,"kelly")
print(friends)
print("---------------------------------------------")
friends.insert(4,"nike")
print(friends)
print("---------------------------------------------")
friends.clear()
print(friends)


# <p id='12'><h1>  12. Tuples in python</h1></p>
# * Different from list -immutable-
# * create tuples
# * for data thats never gonna change

# In[ ]:


coordinate = (6,7)
print(coordinate[1])


# <p id='13'><h1>  13. Using function in python</h1></p>
# * organize code
# * chunks that me taks easy
# * lines of code that does some thing and call it when ever needed
# steps
# 1. def<space>sayhi():
# 2. <indent> print("anythiong you want to print to user")
# 3. call the function --> name() ex- sayhi()

# In[ ]:


# creating a function to say hi to a variablie
def say_hi():
    print("Hello user")
say_hi()
print("---------------------------------------------")
#parameters that can be passed in to the function
def sayhi(name):
    print(" Hello " + name)    
sayhi("Mike")
print("---------------------------------------------")
print("MAKE FUCTIONS POWERFULL BY GIVING INFORMATION")
def sayhi(name, age):
    print("Hello " + name + " you are " + age)    
sayhi("Mike", "age")


# In[ ]:


# return keyword
def cube1(x):
    y = x*x*x
    return y
print(cube(4))

def cube2(x):
    return x*x*x
print(cube2(9))


# <p id='14'><h1> 14. Variables with function</h1></p>

# In[ ]:


def cube2(x):
    return x*x*x
result = cube2(9)
print(result)


# <p id='15'><h1> 15. If statement in python</h1></p>
# * when certian conditions are true
# * if and else
# Example:
# I wake up
# if I'm hungary
# <action>I eat breakfast
# 
# I leave my house
# if raining
# <action>take umbrella
# else
# <action>take sunglasses
# 
# 

# In[ ]:


# simple statement
is_male = True
if is_male:
    print("you are a male")
    
print("---------------------------------------------")
is_female = False
if is_female:
    print("you are a female")
else:
    print("you are a male")


# <p id='16'><h1> 16. Using camparisions with if statements</h1></p>
# ## Program for returning the max values

# In[ ]:


def max_num(num1,num2,num3):
    if num1 >= num2 and num1 >= num3:
        return num2
    elif num2 >= num1 and num2 >= num3:
        return num2
    else:
        return num3
    
print(max_num(3,4,5))


# <p id='17'><h1> 17. Build an Advanced calculator</h1></p>
# 1. get input from the user
# 2. convert the numbers into string
# 3. Take an operator from the user
# 4. repeat 1,2
# 5. print the result as <Userinputnum1>{userenteredoperator}<Userinputnum1>
# 6. use if else statements for all the operators

# In[ ]:


#input from the user and convert the numbers into string with float()
num1 = float(input("Enter first number:"))
op = input("Enter Operator:")
num2 = float(input("Enter second number:"))

if op == "+":
    print(num1 + num2)
elif op == "-":
    print(num1 - num2)
if op == "*":
    print(num1 * num2)
elif op == "/":
    print(num1 / num2)
elif op == "%":
    print(num1 % num2)
else:
    print("input invald")

try:
    number = int(input("Enter a number: "))
    print(number)
except:
    print("Invalid Input")


# <p id='18'><h1> 18. While loops</h1></p>
# 1. Create an integer
# 2. create a while loop
# 3. specify a condition
# 4. do some logic, Eg- Print something
# 5. loops as long as condition is true 

# In[ ]:


i = 1
while i <= 10:
    print(i)
    i += 1
print("Done with loop")

