#!/usr/bin/env python
# coding: utf-8

# # Python 
# According to [StackOverflow](https://insights.stackoverflow.com/survey/2018/) survey, Python is considered as one of the fastest growing programming language. Python is an [Open Source](https://en.wikipedia.org/wiki/Open-source_software) programming language and works across all platform. Get [Python](http://python.org/downloads/) if youve'nt installed it yet.
# 
# This notebook gives a refresh for experience programmer and an intro to **Python Programing Language** for newbies. 
# 
# Without much stories, lets dive straight in

# ## Python Basics
# 
# ##### Expression and Math Operators in Python
# Python Output is done using the **print()** function. Now lets write our first program.

# In[ ]:


#Print out Hello World in Python

print('Hello World')


# ###### Math Operators in Python
# Python follows the popular **BODMAS** rule. Operators include:
# * Exponential (**)
# * Remainder/Modulus (%)
# * Division (/)
# * Multiplication (*)
# * Addition (+)
# * Subtraction (-)
# 
# Math operation example in python
# 

# In[ ]:


#Exponential(**)
print(2 ** 2)
print(2 ** 3)
print(4 ** 2)
print(4 ** 4)


# In[ ]:


#Remainder/Modulus(%). Ouputs the remainder of a division
print(4 % 2) 
print(5 % 2)
print(8 % 3)
print(7 % 4)


# In[ ]:


#Division
print(4 / 2) 
print(5 / 2)
print(8 / 3)
print(7 / 4)


# In[ ]:


#Multiplication(*)
print(4 * 2) 
print(5 * 2)
print(8 * 3)
print(7 * 4)


# In[ ]:


#Addition (+)
print(4 + 2) 
print(5 + 2)
print(8 + 3)
print(7 + 4)


# In[ ]:


#Subtraction (-)
print(4 - 2) 
print(5 - 2)
print(8 - 3)
print(7 - 4)


# ##### Variables and Data Types in Python
# **Variables** in python are like containers in the kitchen for storing maggi, salt, palm oil e.t.c. They are used for storing data values in computer memory. Storing data in variables allows flexibility in our program. Going back to our kitchen example, we can easily replace the content of a Container storing maggi with salt or crayfish. Variable in programming gives this flexibilty of replacement too.
# 
# Rules in Naming Variables
# * It **can** be only one word e.g a, b, z e.t.c
# * It **can't** begin with a number e.g 123guy, 2morow
# * It **can** use only letters, numbers(inbetweeen or at the end) and underscore character e.g rice, be3n, _tomorrow e.t.c
# * No space is allowed inbetween variable names e.g rice and beans, ground nut
# **Generally, a good variable name describe the data type it contains.**
# 
# **Data Types** are like the maggi, salt, palm oil e.t.c stored inside the container.
# 
# Types of Data Types
# * String Data Type: are texts & statement
# * Integers Data Type: are whole numbers
# * Float Data Type: are point numbers
# * Boolean Data Type: represent True or False
# 
# With this brief intro, we can code :)

# In[ ]:


#string variables uses single or double quotation mark to store string
food = 'Rice & Beans'
category = 'Food'
ingredient = 'Maggi and Salt'

#Now we can output our variables
print(food)
print(category)
print(ingredient)


# In[ ]:


#Integers are numbers and they do not require quotqtion when storing them
a = 1
b = 2
c = 4

#Now we can output our variables
print(a)
print(b)
print(c)


# In[ ]:


#Floats are point numbers and they do not require quotqtion when storing them
ab_c = 1.7
b29 = 2.567
cGrade = 4.453

#Now we can output our variables
print(ab_c)
print(b29)
print(cGrade)


# In[ ]:


#Boolean essentially stores True or False. They are used to check validity of something
isMale = True
isBobrisky = False

#Now we can output our variables
print(isMale)
print(isBobrisky)


# ###### Working with strings
# We can concatenate(add) strings and also use built in python function on string variables.
# 
# Examples

# In[ ]:


print('Make Up Tutorial')


# We can create a new line inside a string using " \n "

# In[ ]:


print('Make Up \nTutorial') # "\n" inserted will push "Tutorial" into another line


# Assuming we want to put a quotation mark in a string, we need to escape it with a backslash **"\"**. This tells python to execute the string as a whole.

# In[ ]:


# without escape(\) gives error
print('Python is a 'Good' programming language')


# In[ ]:


# with escape
print('Python is a \'Good\'  programming language')


# In[ ]:


# We can also do it by combining single and double quote
print("Python is a 'Good'  programming language")


# **Concatinating String**

# In[ ]:


name = 'Demola'
sex = 'Male'
food = 'Amala'
print('My Name is ' + name )
print(name + ' ' + ' is a ' + sex)
print('And my best food is ' + food)


# **Working with Python in-built function for string**

# In[ ]:


plan = 'ai saturday meetUp'
# to get the length of a string. "len()" function counts whitespaces too
print(len(plan))


# In[ ]:


#Converting  string to lowercase
print(plan.lower())


# In[ ]:


#Convert to uppercase
print(plan.upper())


# In[ ]:


#we can also check if its all lower or upper case
print(plan.islower()) # will return false because we have a capital letter in our string


# In[ ]:


print(plan.isupper()) #also returns false because not letters in the string is upper case


# Chaining function in string
# 
# It involves combination of functions

# In[ ]:


# We are going to chain two functions here, first("upper()") is to convert it to uppercase and the second ("isupper") checks if the string is  Uppercase
print(plan.upper().isupper())


# Getting Index of a string
# 
# Python counts from a Zero(0) upward as suppose from a One(1) upward

# In[ ]:


ingredient = 'Curry'


# In[ ]:


#Let's get the first letter in the variable above
print(ingredient[0])


# In[ ]:


print(ingredient[3])


# We can also get the index of the letters directly using python in-built "index()" function 

# In[ ]:


#let's get the index "y" in Curry
print(ingredient.index("y"))


# In[ ]:


print(ingredient.index("C"))


# In[ ]:


#using a parameter like "K" that is not part of the "Curry" string throws error
print(ingredient.index("K"))


# Other functions

# In[ ]:


#replace the first letter the variable "ingredient"
print(ingredient.replace("C", "K")) #Replace function takes two parameters. First the original letter and second the new letter to replace it with


# Python provides alot of built in functions to work with string. Check them out on [Python Inbuilt Function](https://docs.python.org/3/library/functions.html)

# **Working with Python in-built function for Numbers(Integers & Float)**

# In[ ]:


num = 5


# Python has in-built function to change variable data types

# In[ ]:


#let's convert variable num to a string
print(str(num))


# Converting numbers to string help us concantenate strings with numbers

# In[ ]:


print('My favourite number is ' + str(num))


# In[ ]:


#Returns error if we do not convert to string first
print('My favourite number is ' + num)


# Common number related functions

# In[ ]:


#getting absolute value of a number
print(abs(-5.78))


# In[ ]:


#Power function
print(pow(4, 2)) #takes two parameter. it means 4 raise to the power of 2


# In[ ]:


#getting maximum number 
print(max(2, 5, 3, 4, 1))


# In[ ]:


#getting minimum number 
print(min(2, 5, 3, 4, 1))


# In[ ]:


#round function rounds a number with decimals to whole number

print(round(5.6745))

print(round(4.123))


# **Modules** are collection of precompiled functions and variables that we can use in our program. They are like a Cabinet for storing food stuffs and other ingredient and we can use them whenever we want to cook.
# 
# Inother to get access to this more inbuilt number based functions, we need to import the **Math** module

# In[ ]:


from math import * #This is how we import a module. The code is simply saying import all codes in math module


# In[ ]:


#floor function outputs a number without the decimal points
print(floor(3.456))

print(floor(4.567))


# In[ ]:


#Ceil function rounds the decimal numbers up
print(ceil(3.456))

print(ceil(4.567))


# In[ ]:


#sqrt function outputs the square root of a number
print(sqrt(25))

print(sqrt(144))


# ###### Getting Input from a User
# Python provides a function that allows us to get input from a user, store it as a variable and perform other functions with the input. Input function makes our program interactive 

# In[ ]:


#use the input() function and type in a prompt whenever its executed

input('What is your name? ') #"What is your name?" is the prompt describing what the user to do


# Now let's create a simple program to ask for user's name and and also say Hello

# In[ ]:


name = input('What is your name? ')
print('Hello ' + name + '!')


# How about we modify the above program alittle bit

# In[ ]:


name = input('What is your name?')
age = input('How old are you?')
print('Hello ' + name + '! You are ' + age)


# ###### Project: Basic Calculator
# 
# The calculator gets two number from user, multiply and outputs the result

# In[ ]:


#first we will create variables
num1 = input('Enter a number: ')
num2 = input('Enter another number: ')


#lastly, we will multiply and output our result. By default, inputs from users comes as a string. So we need to convert it into a number usting float() function
result = float(num1) * float(num2)
print(result)


# ###### Project: Mad Libs Game
# Mad Libs is a game where one player prompts another player input a list of words to fill in a blank space. Learn more about [Mad Libs Game](https://en.wikipedia.org/wiki/Mad_Libs)

# In[ ]:


#first we need to create variables to prompt inputs from users
color = input('Enter a color: ')
plural_noun = input('Enter a Plural Noun: ')
celebrity = input('Enter a celebrity name: ')

#Outputs of inputs from users to fill in blank spaces
print('Roses are ' + color)
print(plural_noun + ' are blue')
print('I love ' + celebrity)


# ### List
# A list  is a value that contains multiple value in an ordered an ordered sequence. List uses square brackets "[ ]". List accepts all data types

# In[ ]:


# let make a list of foods
food = ['Rice', 'Beans', 'Yam', 'Potatoes']

print(food)


# In[ ]:


#we can also refer to element with their index and output them
print(food[0])

print(food[3])

print(food[1])


# In[ ]:


# we can also access index from the back of the list using negative sign
print(food[-1])

print(food[-2])


# In[ ]:


#we can also group-select list

print(food[2:]) #this will print the last elements of the list. "2:" means print element with index 2 and above


# In[ ]:


#This will output element in index "0" & "1" and exclude "2"
print(food[0:2])


# We can also modify a list element

# In[ ]:


#Let's replace element with the name "Yam" with "Cocoyam"
food[2] = "Cocoyam"

print(food)


# In[ ]:


#more replacement. Using list element to replace  another list element
food[0] = food[-1]

print(food)


# In[ ]:


#Deleting an element in a list
del food[-1]

print(food)


# Just like strings and numbers, python also provides in-built functions to work with list

# In[ ]:


jackpot_num = [23, 45, 24, 55, 76, 88, 90]
surname = ['John', 'Jacob', 'Travota', 'James']


# Extend Function adds a list to another list

# In[ ]:


#let's add the surname list to jackpot_num list
surname.extend(jackpot_num)

print(surname)


# In[ ]:


#adding individual element to the end of a list
jackpot_num.append("Number")

print(jackpot_num)


# In[ ]:


#insert function adds elemnt to a specified index
jackpot_num.insert(1, 'B-29') 

print(jackpot_num)


# In[ ]:


#deleting element
jackpot_num.remove("Number")
print(jackpot_num)


# In[ ]:


#More on deleting element. Pop function removes element from the end of a list
jackpot_num.pop()
print(jackpot_num)


# In[ ]:


#checking index of element
print(jackpot_num.index(23)) #get the index of the first element in a list

print(jackpot_num.index('B-29'))


# In[ ]:


# we can also copy a list

jackpot = jackpot_num.copy()

print(jackpot)


# ### Tuples 
# Tuples are type of data structure similar to a list. Tuples are created using bracket "()". 
# 
# Tuples are immutable(cannot be changed or modify)

# In[ ]:


coordinates = (2, 5, 7)

print(coordinates)


# In[ ]:


#access tuple by index 
print(coordinates[-1])

print(coordinates[1])


# In[ ]:


#trying to change tuple element gives an error (they are immutable)
coordinates[1] = 2

print(coordinates)


# ## Functions in Python
# Function is a collection of codes that perform a specific task. Generally, function helps us to structure and breakdown to our code. We are already familiar with print(), input(), str() functions used above, we can also write our own function.
# 
# 

# In[ ]:


#now let's write a function that greets us
def greeting():
    print('Hello')
    print('Good Day')
    print('Enjoy your day')


# A code inside a function needs to be indented for python to consider it as codes inside the function.
# 
# for the function above to execute, we need to call the function 

# In[ ]:


greeting()


# ##### Functions with Parameters
# Parameters are piece of infos we give to our function to execute.
# 
# let's modify our function above

# In[ ]:


def greeting(name):
    print('Hello ' + name)
    print('Good Day ' + name)
    print('Enjoy your day ' + name)
    
    

#Now let's call the function with a parameter
greeting("demola")


# Another function example with more parameters

# In[ ]:


def hello(name, sex, age):
    print('My name is ' + name + ', A ' + sex + ' and i am ' + age +' years old!' )

    
#call the function with name, sex & age
hello('Adaobi', 'Female', '21')


# ##### Functions with Return Statement
# Return statement are used to get information from a function.
# 
# Now let's write a function that returns the square of a number

# In[ ]:


def square(num):
    return num ** num 

#call the function 
square(3)


# Any code inputed after the **return** statement in a function will not run.
# 
# let's see an example

# In[ ]:


def cube(digit):
    return digit * digit * digit
    print('Thanks for the calculation')  #This line of code wil not be execute
    
    

#call the function
cube(2)


# For the print statement to run above, it has to come before the return statement

# In[ ]:


def cube(digit):
    print('Thanks for the calculation')
    return digit * digit * digit

#call the function    
cube(2)


# ## Conditions in Python
# Conditions in python are like checks carried out before executing a code. For example, we need to check the kitchen and confirm if rice is available, before we can cook rice or opt for beans instead.

# #### IF statement 
# In IF statement we check if a condition is true, if it is, we do something and if it's not true we will skip it or do another thing.
# 
# Now let's code if statement

# In[ ]:


#Assuming we have rice in the kitchen

#let's create a boolean variable to check our condition
is_rice_at_home = True

#if statement below
if is_rice_at_home:
    print('Yeah!!  There is rice at home')


# The code above will execute the print statement because an if statement checks if the condition stated is true.
# 
# Assuming there is no Rice in the kitchen and and we change our boolean variable to be False

# In[ ]:


#No rice in the Kitchen
is_rice_at_home = False


#if statement
if is_rice_at_home:
    print('Yeah!!  There is rice at home')


# The code above will not give any output becuasethe if condition is not met.

# lets make our condition more dynamic by giving the program what to execute if the condition is not met. We can achieve  this using the **if else statement**
# 
# #### IF ELSE statement 

# In[ ]:


#No rice in the Kitchen but we can cook beans instead
is_rice_at_home = False


#if statement
if is_rice_at_home:
    print('Yeah!!  There is rice at home')
else:
    print('No rice in the Kitchen, but we can cook Beans!')


# **PS:** we need to indent our code properly

# Fortunately, python allows multiple condition in programs using "OR" & "AND"

# In[ ]:


#Another example with multiple condition. 
is_male = True
is_tall = True

#check multiple condition
if is_male or is_tall:
    print('You are a Male OR Tall OR Both')
else: 
    print('You are neither a Male nor Tall')


# The condition above checks if either one or both of the condition above is true before it execute.

# The code will still execute if we change one of the condition to false. This is becuase the stated condition is either one or both condition is true

# In[ ]:


is_male = False
is_tall = True

#check multiple condition
if is_male or is_tall:
    print('You are a Male OR Tall OR Both')
else: 
    print('You are neither a Male nor Tall')


# But if we change both condition to False, then it will not execute the IF statement but execute the ELSE statement instead 

# In[ ]:


is_male = False
is_tall = False

#check multiple condition
if is_male or is_tall:
    print('You are a Male OR Tall OR Both')
else: 
    print('You are neither a Male nor Tall')


# Assuming we want both of our condition to be TRUE before the code execute. We can do this using the "AND" operator

# In[ ]:


is_male = True
is_tall = True

#check multiple condition
if is_male and is_tall:
    print('You are a Male AND You are Tall')
else: 
    print('You are either not male or not tall or both')


# The else statement will execute if one or both the condition is False

# In[ ]:


is_male = False
is_tall = True

#check multiple condition
if is_male and is_tall:
    print('You are a Male AND You are Tall')
else: 
    print('You are either not male or not tall or both')


# We can make our program more dynamic with multiple if conditions using the **IF ELIF & ELSE statement** 
# 
# #### IF ELIF ELSE statement 

# In[ ]:


is_male = False
is_tall = True

#check multiple condition
if is_male and is_tall:
    print('You are a Male AND You are Tall')
    
elif is_male and not is_tall:
    print('You are a short Male')

elif not is_male and is_tall:
    print('You are not Male, but you are tall')
    
else: 
    print('You are  not male or tall')


# In[ ]:


is_male = True
is_tall = False

#check multiple condition
if is_male and is_tall:
    print('You are a Male AND You are Tall')
    
elif is_male and not is_tall:
    print('You are a short Male')

elif not is_male and is_tall:
    print('You are not Male, but you are tall')
    
else: 
    print('You are  not male or tall')


# ### Comparison using Operators in Conditions
# Operators in python are:
# 
# * '=='   **Equal to**
# * '!='   **Not equal to**
# * '<'    **Less than**
# * '>'    **Greater than**
# * '<='   **Less than or equal to**
# * '>='   **Greater than or equal to**
# 
# To illustrate this operators better, we will create a function that accepts user input and tells us the maximum number 

# In[ ]:


def max_number(num1, num2, num3):
    if num1 >= num2 and num1 >= num3:
        return 'The maximum number is: ', num1
    
    elif num2 >= num1 and num2 >= num3:
        return 'The maximum number is: ', num2
    else: 
        return 'The maximum number is: ', num3


# The code above accepts 3 parameters num1, num2 & num3.
# 
# The first if statement checks if **num1** is greater than or equal to either **num2** and **num3**. If the condition is true, then it will return **num1** as the maximum number
# 
# The second elif statement checks if **num2** is greater than or equal to either **num1** and **num3**. If the condition is true, then it will return **num2** as the maximum number
# 
# The third else statement returns **num3** if both conditions above are not met

# In[ ]:


#Now let's call the function
max_number(5, 6, 10)


# In[ ]:


max_number(26, 3, 15)


# In[ ]:


max_number(5, 44, 10)


# We can also use comparison operator on strings
# 

# In[ ]:


pet_name = input('Type in your pet namr: ')

if pet_name == 'Sisqo':
    print('My pet\'s name is Sisqo')
elif pet_name == 'Skippo':
    print('My pet\'s name is Skippo')
else:
    print('Type in your pet name')


# In[ ]:


pet_name = input('Type in your pet namr: ')

if pet_name == 'Sisqo':
    print('My pet\'s name is Sisqo')
elif pet_name == 'Skippo':
    print('My pet\'s name is Skippo')
else:
    print('Type in your pet name')


# In[ ]:


pet_name = input('Type in your pet namr: ')

if pet_name == 'Sisqo':
    print('My pet\'s name is Sisqo')
elif pet_name == 'Skippo':
    print('My pet\'s name is Skippo')
else:
    print('Type in your pet name')


# ###### Project: Building a Better Calculator

# In[ ]:


#first we will create a variable and convert it to a number since input by default converts it's input to a string
num1 = float(input('Enter first number: '))

#lets also get the operator
operator = input('Enter Operator: ')

num2 = float(input('Enter second number: '))

#Now we will write condition statement to figure out the kind of operations users want
if operator == '+':
    print(num1 + num2)

elif operator == '-':
    print(num1 - num2)
    
elif operator == '*':
    print(num1 * num2)

elif operator == '**':
    print(num1 ** num2)
    
elif operator == '/':
    print(num1 / num2)
    
else: 
    print('Invalid Operator!!')


# ## Dictionary
# Dictionary is a collection of many value like a **List** but with different kind of indexes. Dictionary allows a **key-value** pair. It works like a normal dictionary which has a word(key) and a definition(value).
# 
# We define dictionary using angle bracket "{}"
# 
# In a dictionary, the value on the left is KEY, the value on the right is the VALUE and colon(:) is used to assign a VALUE to a KEY. We seperate multiple values in dictionary using comma(,). The KEY in a dictionary must be unique

# In[ ]:


#lets create a dictionary that converts month
month_conversion = {
    'Jan': 'January',
    'Feb': 'February',
    'Mar': 'March',
    'Apr': 'April',
    'May': 'May',
    'Jun': 'June',
    'Jul': 'July',
    'Aug': 'August',
    'Sep': 'September',
    'Oct': 'October',
    'Nov': 'November',
    'Dec': 'December'
}

print(month_conversion)


# We can access dictionary values in different ways via it's  key

# In[ ]:


#Accessing dictionary using square bracket
print(month_conversion['Jan'])


# In[ ]:


print(month_conversion['Mar'])


# In[ ]:


#Accessing dictionary using get() function
print(month_conversion.get('Nov'))


# Using an invalid key a dictionay will return 'None'

# In[ ]:


print(month_conversion.get('Mow'))


# In[ ]:


# we can also print out a default value when the specified key is not valid 

print(month_conversion.get('Mow', 'Not a valid key, Please enter a valid key'))


# Dictionary also takes integers as key 

# In[ ]:


alpha = {1:'a', 2:'b', 3:'c', 4:'d'}
print(alpha)


# ## Iteration and Looping in Python
# Looping in python allows us to loop through a block of code repeatedly.
# 
# ###### While Loops
# 

# In[ ]:


a = 1

while a < 10:
    print(a)
    a = a + 1  #This line of code tells python to keep adding "1" to "a" value until the condition is met


# In[ ]:


# Better way of writing the  while loop
b = 1
while b < 6: #as long as this condition is true, it will keep looping over the code below
    print(b)
    b +=1   #this is the same thing as "a = a + 1"


# ###### Project: Building a Guesing Game
# This game involves us specifying a secret word and the user interacts with the program by guessing the secret word. The user has  a limit of 3 times, if the user guesses more than 3 times then they have lost the game

# In[ ]:


#Guessing game code

secret_word = 'Lion'
guess =''
guess_count = 0
guess_limit = 3
out_of_guess = False

while guess != secret_word and not out_of_guess:
    if guess_count <  guess_limit:
        guess = input('Guess a word : ')
        guess_count += 1
    
    else: 
        out_of_guess = True

        
if out_of_guess:
    print('Out of Guess: YOU LOSE!')

else:
    print('You win')
    


# In[ ]:


#Let's run the game again but with right word

secret_word = 'Lion'
guess =''
guess_count = 0
guess_limit = 3
out_of_guess = False

while guess != secret_word and not out_of_guess:
    if guess_count <  guess_limit:
        guess = input('Guess a word : ')
        guess_count += 1
    
    else: 
        out_of_guess = True

        
if out_of_guess:
    print('Out of Guess: YOU LOSE!')

else:
    print('You win')


# ###### For Loop
# This is probably the most common type of loop in any programming language. It can be use to iterate over any elements of an array, dictionary e.t.c
# 
# For loop takes in a variable and loop over a collection

# In[ ]:


#Example to print out all the letters in a string

for letters in 'Rice & Beans':
    print(letters)


# In[ ]:


#Example: Looping through an array

name = ['David', 'James', 'Jim', 'Janet', 'Skibo']

for index in name:
    print(index)


# In[ ]:


#Example: using a dictionary
month_in_a_year = {
    'jan': 'january',
    'feb': 'february',
    'mar': 'march',
    'apr': 'april',
    'may': 'may'
}

for month in month_in_a_year:
    print(month)


# In[ ]:


# to get the values in a dictionary. We will use a values() function
month_in_a_year = {
    'jan': 'january',
    'feb': 'february',
    'mar': 'march',
    'apr': 'april',
    'may': 'may'
}

for month in month_in_a_year.values():
    print(month)


# In[ ]:


#Example: Iterating over ranges of number

for index in range(5):
    print(index)


# In[ ]:


#More on range
for index in range(4, 9):
    print(index)


# In[ ]:


#Examples: More on loops

for index in range(5):
    if index == 0:
        print('First Index')
    elif index == 2:
        print('Third Index')
    else:
        print(index)


# **Exponent Function using Loop**

# In[ ]:


def expo(base_num, power_num):
    result = 1
    for index in range(power_num):
        result = result * base_num
    return result


#call the function
expo(4, 3)


# ## Nesting in Python
# Nesting is simply embedding a collection of element inside another collection

# In[ ]:


#Example: List
nest =[
    [1, 2, 3], 
    [4, 5, 6], 
    [7, 8, 9], 
    [0]
]

print(nest)


# Assuming we want to get individual number in a nested list. We can access it by [row][column]
# 
# [row] is the horzontal list index
# 
# [column] is the vertical lisyt index

# In[ ]:


#lets get 1
print(nest[0][0])


# In[ ]:


#lets get 4
print(nest[1][0])


# In[ ]:


#lets get 9
print(nest[2][2])


# ###### Nested For loop
# Python permits nesting of loop inside another loop.
# 
# Using the nest variable above

# In[ ]:


#get the nest variable row
for row in nest:
    print(row)


# In[ ]:


#Now let us print the column in each of the printed rows above
for row in nest:
    print(row)
    for col in row:
        print(col)


# ###### Project: Build A Translator
# The translator simply transform a vowel in any given string to a letter 'g'

# In[ ]:


def translator(phrase):
    translation = ''
    for letter in phrase:
        if letter.lower() in 'aeiou':
            if letter.isupper():
                translation = translation + 'G'
            else:
                translation = translation + 'g'
        else:
            translation = translation + letter
    return translation


# In[ ]:


#Now lets call the function
print(translator(input('Enter a phrase: ')))


# ## Comment in Python
# Comments are vital parts of a code. It help us remember our code and also help other developer reviewing our code understand what that part of code is meant to do
# 
# Comment in Python is either number sign (#this is a comment) for a line or tripple single or double quote (""" this is a comment""") for multi-line comment 

# In[ ]:


#this is a comment
#print('Hello')


# In[ ]:


"""This is also a multiline 
comment using a double quote thrice
"""


# In[ ]:


'''
this a multi-line
comment with 
single quote
'''


# ## Exception Handling in Python
# Python let us handle error that may inder our program from running

# In[ ]:


# Example: convert a user input into a number

number = int(input('Enter a number: '))
print(number)


# In[ ]:


#Assuming we break the rule and didn't enter a number
number = int(input('Enter a number: '))
print(number)


# The code throws an error becuase  the user entered letters as suppose to a number which will break our program. Inother to prevent breaking our program python provides a **TRY EXCEPT Block** for handling such cases 

# In[ ]:


#Now lets modify the program above
try:
    number = int(input('Enter a number: '))
    print(number)
except:
    print('Invalid Input!!')


# In[ ]:


#We can also make our code better by catch or excepting our error
try:
    value = 10/0
    number = int(input('Enter a number: '))
    print(number)
except ZeroDivisionError as err:
    print(err)
except ValueError:
    print('Invalid Input')


# ## Reading External File in Python
# Python provides functins for reading, closing and editing  files.
# 
# It is a good practice to close the file after opening or working on it.

# In[ ]:


#This open() fuction takes the path to the file we want to open

file = open('../input/ariba.txt', 'r+') 


# In[ ]:


# We can check if the file is readable
file.readable()


# In[ ]:


#output the content of the file
file.read()


# In[ ]:


#Read the content of the file one by one using readline() function
file.readline()


# In[ ]:


#Reading all the content in a loop at once using readlines()
file.readlines()


# In[ ]:


#We can close the file
file.close()


# Python also  allows us to write new file and also append to an existing file
# 
# Append

# In[ ]:


file = open('../input/ariba.txt', 'a') #change the mode to append 'a'


# In[ ]:


#adding to the file
file.write('Materials Management')


# In[ ]:


file.close()


# Writing a New File

# In[ ]:


new_file = open('../new.html', 'w')
new_file.write('<div> Hello Mr HTML </div> \n <p> It is a good day to Code</p>')
new_file.close()


# In[ ]:


print((open('new.html', 'r+')).readlines())


# ## Modules in Python
# Modules are collection of codes we can input into our program to make it more functional. We can also create a module ourself and use in other program. To read more and access other python modules check out [Modules](https://docs.python.org/3/py-modindex.html)
# 
# Now let's work with a python module called randint that generates random numbers

# In[ ]:


from random import randint

rand_num = randint(0, 10) #generates a random number within the range of 0 and 10
print(rand_num)


# In[ ]:


print(randint(25, 45))


# ## Installing Third-Party Modules and Library in Python
# Python has a huge community of developers constantly working on libraries and modules. we can simply google what we are trying to do and there's a  possibility that someone have developed a module or library for that already
# 
# ###### Installing Modules & Libraries in Python 
# Python uses a Package manager known as **PIP** which helps us install, manage, update and delete modules & libraries. To install a module/library:
# 
# 1. Click the START button in your computer 
# 2. Type in "CMD" in the search bar
# 3. Click the cmd program
# 4. Type in the library/module you want to install 
# Assuming we want to install  a module named "Python-docx".
# 
# Just type in **"pip install python-docx"**
# 
# To delete **pip uninstall python-docx**
# 
# Most third party modules installed on python are stored in a folder name **Site-Packages**.
# 
# 
# 

# ## Classes and Object  in Python
# In construction, an architect produces a **Plan or Blueprint** that civil engineers or builders follows in implemeenting a structural buildings.
# 
# In programming, classes are like blueprint/plan to produce Object(like buildings). Classes and Object makes our program powerful and dynamic.
# 
# Classes and Object are used to represent real-world datas.
# 
# Variables in Object are called Properties and Functions in python are called Methods.
# 
# Assuming we want to write a program representing a School in python, we can use Class and Object to implement all the sections in a school.
# 
# 

# In[ ]:


#First let's model  a student class
class Student:
    def __init__(self, name, sex, department, gpa, is_on_probation):
        self.name = name
        self.sex = sex
        self.department = department
        self.gpa = gpa
        self.is_on_probation = is_on_probation
        


# **def __init__(self)** statement is used in a class to map out properties and methods a class should have. Its essentially defines what a student is.
# 
# **self** is a class instance and it refers to the Object created at a particular instance.
# 
# Remember that a class is like a plan/blueprint for creating an object. So we can now create a Student Profile using our Class Blueprint above

# In[ ]:


#Create an Object for a student (John and Zainab)
student_1 = Student('John', 'Male', 'Civil Engineering', 3.95, False )
    
student_2 = Student('Zainab', 'Female', 'Medicine', 4.59, False)


# Now we can access each infos of a student

# In[ ]:


print(student_1.department)

print(student_1.name)

#accessing student 2
print(student_2.department)

print(student_2.sex)


# ###### Project: Multiple Choice Quiz
# This game allows user to answer quizzes, keep track of the scores and output the final score.

# In[ ]:


#Create a variable for the questions
questions = [
    'What colour are apples? \n (a) Red/Green \n (b) Purple \n (c) Orange \n\n',
    'What colour are bananas? \n (a) Teal \n (b) Magenta \n (c) Yellow \n\n',
    'What colour are strawberries? \n (a) Yellow \n (b) Red \n (c) Blue \n\n'
]

#To keep tract of the question, answer and scores, we need to use a class

class Quiz:
    def __init__(self, prompt, answer):
        self.prompt = prompt
        self.answer = answer


# In[ ]:


#Now we can create our quiz questions for users using the quiz class

quiz_question = [
    Quiz(questions[0], 'a'),
    Quiz(questions[1], 'c'),
    Quiz(questions[2], 'b')
]


# In[ ]:


#now we can create a function to ask the questions and also check if the answer is correct
def run_quiz(quiz_question):
    score = 0
    for quiz in quiz_question:
        answer = input(quiz.prompt)
        if answer == quiz.answer :
            score +=1
    print('You got ' + str(score) + '/' + str(len(quiz_question)) + ' correct')


# In[ ]:


#now let's run the quiz by calling the function
run_quiz(quiz_question)


# #### Object Function
# Object Function are also known as method, it's basically including a function in our class.
# 
# Let's modify Student class created earlier and have a function embed in it

# In[ ]:


# a function to check if student is on dean list if the gpa is greater than 4.0 
class Student:
    def __init__(self, name, sex, department, gpa, is_on_probation):
        self.name = name
        self.sex = sex
        self.department = department
        self.gpa = gpa
        self.is_on_probation = is_on_probation
        
    def on_dean_list(self):
        if self.gpa >= 4.0:
            print('Student is ON Dean\'s list')
        else:
            print('Student is NOT on Dean\'s list')
        


# Now we can two more student and test if they're on dean's list or not
# 

# In[ ]:


student_3 = Student('Racheal', 'Female', 'Business Administration', 3.45, False)

student_4 = Student('Micheal', 'Male', 'Architecture', 4.89, False)


# In[ ]:


#test if student is on dean's list
print(student_3.on_dean_list())


# In[ ]:


print(student_4.on_dean_list())


# ### Inheritance in Python Class
# Inheritance in python involves a Class  inheriting methods(function) and properties(variables) from another Class.
# 
# Using Student Class above, we can create another class for Lecturers that inherit some of the properties of the Student Class.
# 
# This is done by passing the class we want to inherit from as a parameter. So we will inherit the property of Name and Sex from the Student Class. 

# In[ ]:


class Lecturer(Student): 
    def __init__(self, faculty, position, sex, name):
        self.faculty = faculty
        self.poeistion = position
        Student.sex = sex
        Student.name = name
        


# In[ ]:


#Now let's define lecturers

lecturer_1 = Lecturer('Engineering', 'Professor',  'Male', 'John Traversy')


# In[ ]:


print(lecturer_1.name)


# In[ ]:


print(lecturer_1.sex)

