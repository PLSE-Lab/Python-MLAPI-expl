#!/usr/bin/env python
# coding: utf-8

# # Basics of python for data science 

# ## Data Types

# Python, like many programming languages has data types. If you don't know what a data type is, think of it as different entities having different behaviors. For example they can be integers, real numbers, text, etc. In python we have 4 types of data which are given as follows:
# - Integer
# - Float
# - String
# - Boolean

# In[ ]:


# Examples for integer 1, 2, 157
type(2)


# In[ ]:


# Examples for float 3.0, 5.71
type(4.5)


# In[ ]:


# Examples for string "Hi", 'this', "is a'a string","4",'53.0'
type("hello")


# In[ ]:


# Examples for boolean True, False
type(True)


# ## Type Casting 

# Type casting is a process wherein users can change the type of data. This might be very useful in certain scenarios.

# In[ ]:


# If we want to convert integer to float, the sybtax to use is as following
float(2)


# In[ ]:


# If we want to convert float to integer, the syntax to use is as following. Please note that while converting decinal values are lost in the process.
int(5.86)


# In[ ]:


# If we want to convert string to integer/float, the syntax to use is as following. Please note that this is only possible for numbers in string format.
print(int('8'))
print(float('5.86'))


# In[ ]:


# Please do note that trying to convert a float in string to integer will throw you an error as following
int('8.5')


# In[ ]:


# Please do note that trying to convert a non-numeric in string to integer will throw you an error as following
int('hi')


# In[ ]:


# Please do note that trying to convert a non-numeric in string to float will throw you an error as following
float('hi')


# In[ ]:


# If we want to convert float to string, the syntax to use is as following
str(5.86)


# In[ ]:


# If we want to convert integer to string, the syntax to use is as following
str(23)


# In[ ]:


# If we want to convert boolean to integer or float, the syntax to use is as following
print(int(True))
print(int(False))
print(float(True))
print(float(False))


# In[ ]:


# If we want to convert integer or float to boolean, the syntax to use is as following
print(bool(1))
print(bool(1.0))
print(bool(0))
print(bool(0.0))


# ## Basic Expressions 

# Now that you have an idea of what the data types are, the next step is to understand what are some of the basic operations you can do with them. I bet that most of us already know some of the basic computations we can do with integers and float as they are similar to school level mathematics. Some standard mathematical operations we can do in python are addition (+), subtraction (-), multiplication (*), division (/), and integer division (//).

# In[ ]:


#Simple addition
1+2


# In[ ]:


#Simple subtraction
5-6


# In[ ]:


#Simple multiplication
5*6


# In[ ]:


#Simple division
39/5


# In[ ]:


#Simple division which returns only integer value from division
39//5


# In[ ]:


#Python follows the standard mathematical rules as in in expression "4*2+5", python first computes 4*2 and then adds 5 to it
4*2+5


# In[ ]:


#If we can the addition to happen first we can specify that by giving parenthesis like
4*(2+5)


# ## Variables 

# To make it easier to understand, assume that variables are containers in you kitchen, ones that you can label as you wish. Now these containers can hold whatever you want them to. They can hold drinks, food, semi solids, etc. When you want to use things inside any container you would look at their labels and try to use them in cooking or munching. In a similar way variables are containers created by the computer to temporarily hold values or data structures and can be accessed by the name you assign to them. The variables themselves can be used to do some computations as needed.

# In[ ]:


#assigning integer 1 to variable called myfirstnumber
myfirstnumber=1
myfirstnumber


# In[ ]:


#Re-assigning variable called myfirstnumber to float 10.0
myfirstnumber=10.0 
myfirstnumber


# In[ ]:


#Inceases the value of myfirstnumber by 1 unit and assigns it back to itself
myfirstnumber+=1 
myfirstnumber


# In[ ]:


#In case expression is equated to variable, the output of expression is assigned to the variable
myfirstnumber=1+6*5.8 
myfirstnumber


# In[ ]:


#Using one variable to access its value and doing some operation on it before assigning it to another variable
mysecondnumber=myfirstnumber*20
mysecondnumber


# ## String Operations 

# String is a set of characters which are enclosed in quotes. It does not matter what are inside those quotes. Unlike other data types, we can access information in a single string using something python users call indexing. Please do note that in python index starts at 0, that means index 0 refers to first element, index 1 refers to second element and so on and so forth

# In[ ]:


v1='medium'
v1


# In[ ]:


#Retrieving 1st letter from the string variable v1
v1[0]


# In[ ]:


#Retrieving 1st 3 letters from the string variable v1
v1[0:3] #0 tells where the first letter is and 3 tells code that information should be fetched till 3 (excludes 3)


# In[ ]:


#Retrieve letters skipping 1 each in between
v1[::2]


# In[ ]:


#Retrieve letters starting from 2nd letter and skipping 2 letter in between
v2='thiswouldbeaquickcheck'
v2[1::3]


# In[ ]:


#We can fetch the length of string using below syntax
print(v1)
len(v1)


# In[ ]:


#We can add or concatenate strings using + or , within print statement
print('1'+'2') #Directly adds strings without any spaces
print('1','a') #Adds two strings with a space in between


# In[ ]:


#By multiplying the string with an integer, the string repeats itself with respect to the magnitude of the integer
print("hi"*2)
print("hi"*7)


# ### Escape characters 

# In python, few characters have fixed meaning. For example " and ' are used for string, but are also used on punctuations in english. So what would happen if we want to use punctuations in a string, wouldn't it be great if we could tell python that it should consider a particular character with it's original intent? That is exactly what an escape character does. Let's see it's usage below.

# In[ ]:


#This would throw an error as there is usage of ' as a punctuation which indicates string for python
'Hi, I'm new one here. Nice to meet you'


# In[ ]:


# The above could be solved by two methods

#Method 1 where the enclosing of string changes from ' to "
print("Hi, I'm new one here. Nice to meet you")

#Method 2 where we give escape character \ in front of punctation '
print('Hi, I\'m new one here. Nice to meet you')


# In[ ]:


#We can also use \ character to go to new line, add tab and lot of other stuffs, look below for some examples
print("Hi \n this will be new line \t this will be with extra space") #\n takes new line while \t gives extra spaces


# ### String Methods 

# Methods are inherent properties of data type or structures created using code. This are called by the method name followed by opening and closing paranthesis. Let us explore some useful methods of string data type

# In[ ]:


#Method capitalize capitalizes the first letter in the string
"random string".capitalize()


# In[ ]:


#Method title capitalizes the first letter of every word in the string
"random string".title()


# In[ ]:


#Method find fetches the position of the first occurance specified in the string
"random string".find('n')


# In[ ]:


#Method islower checks if the string has all of its elements in lower case and returns a boolean as output
"random string".islower()


# In[ ]:


#Method isupper checks if the string has all of its elements in upper case and returns a boolean as output
"random string".isupper()


# In[ ]:


#Method upper converts the string to upper case
"random string".upper()


# In[ ]:


#Method replace replaces the value in string as needed
"random string".replace('r','R')


# In[ ]:


#Method lstrip removes extra spaces in the left side of the string
"  random   string  ".lstrip()


# In[ ]:


#Method rstrip removes extra spaces in the right side of the string
"  random   string   ".rstrip()


# In[ ]:


#Method strip removes extra spaces in the string
"  random string".lstrip()


# In[ ]:


#Method split splits the existing string to multiple strings based on the value given to split, by default it splits where spaces exist. This returns list of substrings created due to split
print("random string".split())
print("random string".split('r'))
print("random string".split('n'))


# ## Data Structures 

# While drawing an analogy we used in data types section, we can say that data structures are similar to drawers/storage compartments in kitchen where we store our containers, and naturally they will have labels/names as well. However, each and every one of these data structures have their own characteristics. Lets discuss them in detail below:

# ### Tuples

# Tuples are a collection of data which can be either a data type or structure itself, so as to say that they can hold all data types along with data structures like tuples themselves.

# In[ ]:


#Declaring a tuple
var=(1,2,3)
print(type(var))

var=tuple((1,2,3)) #Notice we are enclosing the information in two brackets instead of one. This is to ensure we do not get error. Try by removing one bracket and see what you will get.
print(type(var))

var=(2) #This will not be a tuple as it will be considered as an integer. Add comma after entry to make it a tuple
print(type(var))

var=(2,)
print(type(var))


# In[ ]:


# Tuples are typically enclosed in ()
print((1,2,3))
type((1,2,3))


# Similar to strings, tuples can also be sliced using indexing

# In[ ]:


v2=(1,'hi',3.0)
print(v2)
print(type(v2))


# In[ ]:


#Fetching the elements of tuple can be done using the indexes
print(v2[0]) #fetches the 0 index i.e. first element of tuple v2
print(v2[::2]) #fetches all elements while skipping 1 element each


# In[ ]:


#Concatenating of tuples is also possible and this creates a new tuple
print((1,2)+(3,4))
print(type((1,2)+(3,4)))


# In[ ]:


#It is important to note that tuple are immutable, which means they cannot be changes
v21=(1,2,3,4)
v21


# In[ ]:


#When anyone tries to change an element of tuple they would get the following error
v21[1]=2.0


# #### Tuple Methods 

# Similar to many other classes, tuples also have methods which can be applied to them. Let us look at some frequently used methods in below examples

# In[ ]:


# Method count is used to see how many occurances of value given in paranthesis are part of the tuple
(1,2,'hello',1,2,5,12,'1').count(1) #Please note that this also takes data type into account while searching for elements to count


# In[ ]:


# Method index returns the index of the first occurance of value given in paranthesis
(1,2,'hello',1,2,5,12,'1').index(1)


# ### Lists 

# Lists are almost similar to tuples while the key difference being that they are mutable, which means that they can be changed or over-written.

# In[ ]:


#Declaring a list
var=[1,2,3]
print(type(var))

var=list((1,2,3)) #Notice we are enclosing the information in two brackets instead of one. This is to ensure we do not get error. Try by removing one bracket and see what you will get.
print(type(var))

var=[2] #This will still be a list unlike tuples
print(type(var))

var=[2,]
print(type(var))


# In[ ]:


# Lists are typically enclosed in []
print([1,2,3])
type([1,2,3])


# Similar to strings, lists can also be sliced using indexing

# In[ ]:


v3=[1,'hi',3.0]
print(v3)
print(type(v3))


# In[ ]:


#Fetching the elements of list can be done using the indexes
print(v2[0]) #fetches the 0 index i.e. first element of tuple v2
print(v2[::2]) #fetches all elements while skipping 1 element each


# In[ ]:


#Concatenating of lists is also possible and this creates a new list
print([1,2]+[3,4])
print(type([1,2]+[3,4]))


# In[ ]:


#It is important to note that unlike tuple are mutable, which means they can be changes
v21=[1,2,3,4]
v21


# In[ ]:


#When anyone tries to change an element of list they would do so as following
v21[1]=2.0
v21


# In[ ]:


#When we assign this list to another variable, it does not copy the elements, rather creates a new name for it and will get called upon by either name
#This means once we assign it to another variable v22, any change done in v21 will reflect in v22 and vice versa
print(v21)
v22=v21
print(v22)
v22[0]="First"
print(v21)
print(v22)


# In[ ]:


#To copy lists we have to do the following
v21=['First', 2.0, 3, 4]
v22=v21[:]
print(v21)
print(v22)
v21[0]=1
print(v21)
print(v22)


# #### List Methods 

# Similar to many other classes, lists also have methods which can be applied to them. Let us look at some frequently used methods in below examples

# In[ ]:


# The append method adds information to an existing list, however note that the append will add a single element to a list
l1=[1,2,3]
print(l1)
l1.append([4,5]) #Here list of [4,5] which is considered as single element is added to the list
print(l1)


# In[ ]:


# Individual types can be appened to list, however, multiple types or elements cannot be appeneded to list at once
l1.append(6)
print(l1)
l1.append(8,9)
print(l1)


# In[ ]:


# To add multiple elements into list, there is extend method. Point to note however is that the information to be added should be passed as a list
l1.extend([8,9])
print(l1)


# In[ ]:


# Method index returns the index of the first occurance of element with the specified value
l1.index(6)


# In[ ]:


# Method insert adds element at any specified location
l1.insert(1,0) #1 is index in which value 0 would be added
l1


# In[ ]:


# Method pop deletes element at mentioned location
l1.pop(1)
l1


# In[ ]:


# Method remove deletes the first occurance of item with mentioned value
l1.remove(6)
l1


# In[ ]:


# Method reverse reverses the order of the list
l1.reverse()
l1


# In[ ]:


# Method sort sorts the order of the list, but it is advised to be applied on lists with numeric data types only as multiple data types might throw an error
l1.sort()
l1


# In[ ]:


ll=[3,5,1,2,6]
ll.sort()
ll


# ### Dictionaries 

# The dictionary is a collection of key and value pairs. The keys are supposed to be immutable and unique which means they cannot to changed and the values can be either of data types or structures. As an address, the keys are used to find the value part of the collection. The indexing part is not applicable to dictionaries as dictionaries can only be called on by the keys instead of any numbers whatever the key might be.

# In[ ]:


# Declaring of dictionaries needs two parts, a key which has to be unique and immutable and a value which is to be associated with it
{'key':'value'}


# In[ ]:


# Values can be any data type and can be both immutable and mutable, but it is advisable that keys are either string or integer
{['a']:'value'} # We get an error here as list is used as key


# In[ ]:


# Even though integer and strings are advisable for creating keys, float type can also be used
print({1.0:'Value'})
type({1.0:'Value'})


# In[ ]:


# When trying to declare dictionary (dict), with same key values, the key takes the latest value without any error. So users have to be careful about it
{1:1,1:2}


# In[ ]:


#However, the values can be anything and can be duplicates too
print({1:1,"2":"two",3.0:[1.0,"2",3]})


# In[ ]:


# We can even have value as another dictionary itself. For people who are familier with JSONs, it works will be like similar structure
d={0:{"First":[1.0,2.0]},1:["hobby",{2:("Complex",'Values')}]}
d


# In[ ]:


# Similar to others, we can also extract information out of dictionary, but the indexing does not work here. Here, we have to extract them by the key names
d={'info_0':{"First":[1.0,2.0]},'info_1':["hobby",{2:("Complex",'Values')}]}
print(d)
d['info_0']


# In[ ]:


# If tried to extract using index style approach, you will get key error, which means python is not able to find the key 0 in your dictionary
d[0]


# In[ ]:


#If you want to fetch "complex" from dict, here is how you do it
print(d)
print(d["info_1"]) # Using key 'info_1' to extract value related to the key
print(d["info_1"][1]) # Using indexing to fetch the second element in the list
print(d["info_1"][1][2]) # Using key 2 to extract value related to the key
print(d["info_1"][1][2][0]) # Using indexing to fetch the first element in the tuple


# #### Dictionary methods 

# In[ ]:


# Keys method fetches all the keys in dict in dict_keys type
print(d.keys())
type(d.keys())


# In[ ]:


# values method fetches all the values in dict in dict_values type
print(d.values())
type(d.values())


# In[ ]:


# Pop methods removes the key value pair associated with the key specified and returns the value part
d1={0:'0',1:'1'}
print(d1.pop(0))
d1


# In[ ]:


# update method adds new key value pairs to an existing dict
d1={0:'0',1:'1'}
d1.update({2:'2'})
d1


# ### Sets 

# The sets are unique unordered collection of information. This means that for sets there is no particular order and all elements in it are unique.

# In[ ]:


# Sets also like dictionaries are declared with curly braces or with a keywork set
set((1,2,3,4,1,2,3,4))


# Note how only the unique values are identified and are made part of the set in the above example

# In[ ]:


# Sets does not take in all sorts or values as given below
{1,'2',['val1','val2'],{"k":'o'}}


# In[ ]:


# It is advisable to have either of int, float, or strings as below
{0,1.5,'two'}


# In[ ]:


# Sets can be see which elements exist in only one set when compared to another using subtraction operator as following
{1,2,3,4} - {3,4}


# #### Set methods 

# Here, we will see some of the useful set methods which are useful

# In[ ]:


# The first method in intersection. It works in same way that intersection in venn diagrams would work. It extracts common info in two sets
{1,2,3,4}.intersection({3,4,5,6})


# In[ ]:


# The second method in union. It works in same way that union in venn diagrams would work. It extracts all info in two sets
{1,2,3,4}.union({3,4,5,6})


# In[ ]:


# The method add adds another element into the set
s1={1,2,3,4}
s1.add(5)
s1


# In[ ]:


# The method update adds mutiple elements into the set
s1={1,2,3,4}
s1.update([5,6,7,8])
s1


# In[ ]:


# The method remove removes another element into the set
s1={1,2,3,4}
s1.remove(4)
s1


# In[ ]:


# The method pop removes the one item
s1={1,2,3,4}
s1.pop() # sets are unordered, so when using the pop method you will not know which item that gets removed
s1


# In[ ]:


# The issubset method tells if one set is a subset of the other
{3,4}.issubset({1,2,3,4})


# In[ ]:


# The issuperset method tells if one set is a superset of the other
{1,2,3,4}.issuperset({3,4})


# ## Conditional statements and loops 

# ### Conditional Statements 

# In any logical flow there exists, like a typical flow diagram or most of the programs there will always be a branching done based on some kind of logic. This most likely happens because the behavior of us humans is also like such. We always have some kind of logical reasoning behind anything we do. For example, people choose one job over another because it pays more salary or has better working environment. These are called conditions and such conditions are also very useful in coding languages like python. Let us look at what these conditional statements are and how to use them

# #### General comparision operators

# The general comparision operators are mostly used to check for a certain condition. These are often done by using inequality and equality check operators. We can look at few of them below

# In[ ]:


# We use the following syntax to check if the variable b is greater than 10 or not. The result is a boolean value
b=12
b>10


# In[ ]:


# for checking if variable b is less than 4 we can use the following
b < 4


# In[ ]:


# to check if b is greater/lesser than or equal to 12, we use the following
print(b>=12)
print(b<=12)


# In[ ]:


# To check if values are same we do not use = as that is for assigning values, hence we use == to check for same value
print(b==11)
print(b==12)


# A condition can also have subconditions. What this means is a certain thing needs to be done only if two or more criteria are met, or another scenario is if any of said multiple conditions are met some thing should happen. In the above two conditions we use "AND" and "OR" conditions respectively. Let us have a look at them below

# In[ ]:


# check for condition if variable v1 is greater than 10 and less than 20
v1=12  #This value will throw True as 12 is greater than 10 and less than 20
print(v1>10 and v1<20)


# In[ ]:


# check for condition if variable v1 is greater than 10 and less than 20
v1=9  #This value will throw False as 9 is not greater than 10
print(v1>10 and v1<20)


# In[ ]:


# check for condition if variable v1 is less than 10 or greater than 20
v1=12  #This value will throw False as 12 is neither greater than 20 nor less than 10
print(v1<10 or v1>20)


# In[ ]:


# check for condition if variable v1 is less than 10 or greater than 20
v1=22  #This value will throw True as 22 is greater than 20
print(v1<10 or v1>20)


# #### If statement

# If statement is a conditional block statement that is used in most if not all programming languages. In python the syntax for the if statement goes as following:<br><hr>
# if (condition):<br> &emsp;
#     Code to execute if condition is true <br><hr>
#     What happens here is that only when the condition satisfied will the code inside the if block will be executed. If condition fails, then the code will simply ignore whatever is present in if block

# In[ ]:


# Here is a simple working example of a simpe if statement
b=10
if b>0:
    print('Condition satisfied!') #The code comes into this block as condition satisfies
print('Code continues........') # The codes comes to this line of code as this is outside of if block


# In[ ]:


if b>10:
    print('Condition satisfied!') #The code does not come into this block as condition is not satisfied
print('Code continues........') # The codes comes to this line of code as this is outside of if block


# #### If else statement

# If else statement can be said as an improvement over the above if condition. What if condition says is that only when some condition is satisfied will some code be run. If else statement has two blocks namely if block and else block. When condition is satisfied, just like in that of if statement, code inside if block gets executed. But, in case the condition is not satisfied, the code in else block runs. This is the major difference between these two statements.<hr>The syntax for if else statement is as follows:<br>if (condition):<br> &emsp;
#     Code to execute if condition is true <br>else:<br> &emsp;
#     Code to execute if condition is false <br><hr>

# In[ ]:


# Here is a simple working example of a simpe if else statement
b=10
if b>0:
    print('Condition satisfied!') #The code comes into this block if condition satisfies
else:
    print('Condition is not satisfied') #The code comes into this block if condition is not satisfied
print('Code continues........') # The codes comes to this line of code as this is outside of if block


# In[ ]:


if b>10:
    print('Condition satisfied!') #The code comes into this block if condition satisfies
else:
    print('Condition is not satisfied') #The code comes into this block if condition is not satisfied
print('Code continues........') # The codes comes to this line of code as this is outside of if block


# #### elif statement 

# Elif else statement can be said as an improvement over the above if else condition. If else statement has two blocks namely if block and else block. When condition is satisfied, just like in that of if statement, code inside if block gets executed. But, in case the condition is not satisfied, the code in else block runs. But when someone wants to check for multiple exclusive conditions (i.e. not checking them at one time) this is where an elif statement is used. This creates additional block for everytime it is called upon and the code inside these blocks get executed only if the condition in elif statement is true
# 
# 
# <hr>The syntax for elif staement is as follows:<br>if (condition1):<br> &emsp;
#     Code to execute if condition1 is true <br>   
#     elif (condition2):<br> &emsp;
#     Code to execute if condition2 is true <br>
#         else:<br> &emsp;
#     Code to execute if condition is false <br><hr>

# In[ ]:


# Here is a simple working example of a simpe elif statement
b=100
if b>99:
    print('Condition1 satisfied!') #The code comes into this block if condition1 satisfies
elif b>50:
    print('Condition2 satisfied!') #The code comes into this block if condition2 satisfies
else:
    print('Condition is not satisfied') #The code comes into this block if conditions are not satisfied
print('Code continues........') # The codes comes to this line of code as this is outside of if block


# In[ ]:


# Here is a simple working example of a simpe elif statement
b=90
if b>99:
    print('Condition1 satisfied!') #The code comes into this block if condition1 satisfies
elif b>50:
    print('Condition2 satisfied!') #The code comes into this block if condition2 satisfies
else:
    print('Condition is not satisfied') #The code comes into this block if conditions are not satisfied
print('Code continues........') # The codes comes to this line of code as this is outside of if block


# In[ ]:


# Here is a simple working example of a simpe elif statement
b=30
if b>99:
    print('Condition1 satisfied!') #The code comes into this block if condition1 satisfies
elif b>50:
    print('Condition2 satisfied!') #The code comes into this block if condition2 satisfies
else:
    print('Condition is not satisfied') #The code comes into this block if conditions are not satisfied
print('Code continues........') # The codes comes to this line of code as this is outside of if block


# ### Loops in python 

# There might be cases in programing where we might want to continuously do the same operation over and over until certain criteria is met or certain operation is done after some number of times. These are achieved by using loops. We have multiple loops in python and let us have a look at what they are and how to use them

# #### For loop 

# One of the most utilized loop in python (and probably in history of coding!) is for loop. In this loop we tell the python to do a set of operations while setting the number of times or what it should loop through to accomplish said operations. 
# The syntax of for loop is as follows:<hr>
# for iterator in list/tuple:<br>
# &emsp; code to be executed #We can even utilize iterator which holds value in every iteration of the loop
# 
# <hr>Let us look at some examples to better understand the For loops

# In[ ]:


# Here let us loop through a list starting from 0 to 5 and we will be printing through iterator to better understand how we can use them
for i in [0,1,2,3,4,5]: # i here is iterator which will be looped through every iteration
    print(i) # Printing iterator i in every iteration of the loop


# In[ ]:


# Here let us loop through a list of string values and we will be printing through iterator to better understand how we can use them
for i in ['1','2','hi','you','data','science',1.0,5]: # i here is iterator which will be looped through every iteration
    print(i) # Printing iterator i in every iteration of the loop


# In[ ]:


# Let us say, we want to print numbers from above loop except that they should be squares of them instead of just the numbers themselves
for i in [0,1,2,3,4,5]: # i here is iterator which will be looped through every iteration
    print(i*i) # Using iterator i, printing i squared in every iteration of the loop


# In[ ]:


# Let us say, we want to print numbers from above loop and store them in a list. We can do so by using following code
l1=[] #Initializing an empty list
for i in [0,1,2,3,4,5]: # i here is iterator which will be looped through every iteration
    print(i*i) # Using iterator i, printing i squared in every iteration of the loop
    l1.append(i*i) # Using iterator i, appending i squared in every iteration of the loop to list l1
l1 #Printing list l1 after the loop


# In[ ]:


# Instead of using a static list let us now try to use range. Before using it in loops, we have to understand how range works first

print(range(5)) #Range creates a series of numbers from 0 to number mentioned (excludes mentioned number), sort of like a list
print(type(range(5))) #The type however is range


# In[ ]:


# Let us try to use range in a loop now
for i in range(5):
    print(i)
# This works in the same was as using [0,1,2,3,4] in the for loop statement


# #### While loop 

# While loop is another well known loop. Here, the loop continue as long as the specified condition is met. This is used when the number of times the code should run is not certain. The syntax for the while loop is as follows:<hr>
# initiating variable<br>
# while condition:<br>
# &emsp; code to run per iteration<br>
# &emsp; conditonal incrementor #increases initiated variable according to need <hr>Let us look at some examples to understand while loop better

# In[ ]:


# Let us initialize variable v1 as 6 and run loop till v1 is less than 10 and increase v1 by 1 for each iteration
v1=6 #Initiating variable
while v1<10: #starting loop with condition
    print(v1) #Printing variable value
    v1+=1 # Increasing value of variable v1 by unit for each iteration


# Similar to for loop the actual applications are endless depending on use case. An interesting point to note maybe is that since the loop runs on incremental check of condition, most of the time while loop is checked with some sort of number instead of string.

# __Note:__ We will not be covering nesting in this notebook as it is out of scope for now. However, you can get multiple online materials if you want to learn more about them 

# #### Useful keywords in loops 

# There are some special keywords which apply to loops and are quite helpfuli in few cases. These are
# - Pass
# - Continue
# - break
# <br>Let us have a look at them one by one

# In[ ]:


# We use keyword break when we want to stop the loop. Let's say irrespective of the main conditions or iterators, we want to come out of loop when certain additinoal condition is met. Let us look at an example
for i in range(90,120,2): #Curious about change in inputs given to range function? Go inside range function and click shift+tab to look at documentation to understand what are the inputs for (will work in jupyter notebooks)
    print(i)
    if i>100:
        break # The loop stops when the if condition is satisfied
#We can see that the looping has stopped after i=102 as at this point it went inside if block to execute break
#From this you have also realised that we can use conditional statements inside loops. Vice versa is also possible


# In[ ]:


# Keyword continue will skip remaining part of code for iteration and starts new iteration
# The below code works to print even numbers only by checking if they are divisable by 2
for i in range(1,10):
    if i%2!=0:
        continue # Will go to next iteration when code comes inside if block and ignores print statement below which is out of if block and originally should have been printed if continue wasn't there here
    print(i)


# In[ ]:


# Keyword pass does nothing and can be considered as place holder for code. It just let's code pass through its position without doing anything   
for i in range(1,10):
    if i%2!=0:
        pass # notice how this pass is doing nothing when compared to continue
    print(i)


# ## Functions in python 

# There are often certain processes that we usually follow in our day-to-day coding, to make the life of coders easier the concept of functions was conceived. We declare a series of operations that need to be done with some given input (not compulsory) to throw out some output (also not compulsory).
# 
# Even though the name functions might seem new, you have already come across them and used them as methods. Methods that we discussed for a few data structures are nothing but functions that the coders who created said data structures created.
# 
# Even though there are some standard functions existing in every package and some basic python objects, in many cases custom defined functions are helpful. Let us now look at syntax for functions:
# <hr>
# def function_name(input):<br>
# &emsp;code to finish

# In[ ]:


#Let us write a basic function called adding which adds two numbers
def adding(a,b):
    print(a+b) #prints addition of two inputs


# Notice how there is no output from the above piece of code. This is because we have just declared the definition adding and have not called it. Let us now try to call it.

# In[ ]:


adding(1,2) #The a and b in definition are inputs which we gice while calling function


# In[ ]:


#If we do not enter any input we would get the following
adding()


# In[ ]:


#To handle such cases, one can specify a default value while declaring the function as below
def adding(a=0,b=0): #Declaring same function with default values for a and b as 0
    print(a+b) #printing addition of two inputs


# In[ ]:


adding() #Now we do not get an error because we have default values for a and b as 0


# In[ ]:


adding(3,7) #It still works if we give some custom inputs


# In[ ]:


# However if we want to add multiple numbers we have to specify that in definitions, if not we get following
adding(1,2,3)


# In[ ]:


# The special syntax *args in function definitions in python is used to pass a variable number of arguments to a function. It is used to pass a non-keyworded, variable-length argument list
def adding(*args):
    print(type(args)) #print type of arguments

adding(1,2,3)


# We can see that this *args is taken in as tuple type. This allows the users to give any number of inputs as they want to

# In[ ]:


def adding(*args):
    print(sum(args)) #prints addition of all numbers specified

adding(1,2,3,4,5)


# In[ ]:


# Now let us try to return the value from function. The difference between printing and return is that printing 
# can just display information and we cannot store it in any of the variables. Storing output can be used to compute 
# anything required. Let us look at an example to see how this can be useful

def adding(a,b):
    return a+b

def subtracting(a,b):
    return a-b

v1=adding(5,4) #Using function to add 5 and 4 and storing it in variable v1

print(v1) #printing v1

v2=subtracting(v1,5) #Using function to subtract 5 from value stored in variable v1 and storing it in variable v2

print(v2) #printing v2


# There is another type of parameter type that can be used in declaring functions called **kwargs**, you can explore it on your own if interested as this is out of scope of this notebook

# In[ ]:




