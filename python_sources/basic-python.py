#!/usr/bin/env python
# coding: utf-8

# In[ ]:


your_local_path="C:/Users/tejks/Desktop/ML/practice/"


# 
# **IPython notebook**
# 
# IPython is a software package in Python that serves as a research notebook. IPython has a seamless interface using which you can write notes as well as perform data analysis in the same file. The data analysis part allows you to write the code and run it from within the notebook itself. The results also are displayed in the notebook. Among other things, IPython provides:
# 
# 1. An interactive shell
# 2. Notebook that is browser based. The notebook supports text, code, expressions, graphs and interactive data visualization, embed images and videos, and also links to web pages on the Internet. 
# 3. Tools for parallel computing.
# 4. Import external code and run it.
# 5. Export the notebook in a number of formats.
# 
# **Creating an IPython notebook**
# 
# In order to create an IPython notebook, choose File -> New -> IPython notebook. 
# 
# Hitting Alt+Enter creates a new cell. A cell is nothing but a placeholder where you can write some text or execute code. 
# To run press Shift+Enter

# In order to find help on some concept, make use of ? If you want to find a function or make use of wildcard entry, make use of followed by ? A quickref command gives you a quick reference to the most commonly used commands in Python.

# In[ ]:


import sys
print (sys.version)
print (sys.version_info)


# **Syntax Formalities**
# 
# + Python is <b>case sensitive</b>
# + Python makes use of <b>whitespace</b> for code structuring and marking logical breaks in the code (In contrast to other programming languages that use braces)
# + End of line marks end of a statement, so <b>does not require a semicolon at end of each statement</b>
# + Whitespace at the beginning of the line is important. This is called <b>indentation</b>.
# + Leading whitespace (spaces and tabs) at the beginning of the logical line is used to determine the <b>indentation level</b> of the logical line.
# + Statements which go together must have the same indentation. Each such set of statements is called a <b>block</b>.

# In[ ]:


i = 5
print ('Value is ', i) # Error! Notice a single space at the start of the line
print ('I repeat, the value is ', i)


# **Comments** 
# 
# One line comments are denoted by <font color='red'><b>#</b></font> at the start of line<br/>
# Multiple line/block comments start with <font color='red'><b>'''</b></font> and end with <font color='red'><b>'''</b></font>

# In[ ]:


# this is a single line comment
# print("I will not be printed, I am a single line comment")
print ("Aman")
'''This is a block comment 3 single quotes 
 print("We are in a comment")
print ("We are still in a comment") '''


# In[ ]:


#Example of indentation
for x in alist:
    if x < anumber:
            print(x)
    else:
        print(-x)
        
#is similar to                   <------------------ Let us compare with other language

for x in alist 
{
    if x < anumber
    {
        print(x)
    }
    else
    {
        print(-x)
    }
}


# **Variables and Data Structures**
# 
# + Built-in data types:<br/>
# <b>Integer, Floating point, String, Boolean Values, Date and Time</b>
# 
# + Additional data structures:<br/>
# <b>Tuples, Lists, Dictionary</b>
# 
# + A variable is a name that refers to a value.
# + No need to specify <b>type</b> to a variable; Python automatically assigns.
# 

# In[ ]:


counter = 100          # An integer assignment
miles   = 1000.0       # A floating point
name    = "Ajay"       # A string

print (counter);print (miles);print (name)



a = 1111111111110
b = 2.0
c = "0"
d = "Sai"
print(b + a);  # Sum of values
print(c  + d)  # Concatenating Strings


# **Strings**
# 
# + Built-in <b>string class</b> named <b>"str"</b> with many handy features
# + In addition to numerical data processing, Python has very strong string processing capabilities. 
# + <b>Subsets</b> of strings can be taken using the <b>slice operator ( [ ] and [ : ] )</b> with indexes starting at 0 in the beginning of the string and working their way from -1 at the end.
# + The plus ( + ) sign is the string concatenation operator and the asterisk ( * ) is the repetition operator.
# + Strings in Python are <b>immutable</b>. Unlike other datasets such as lists, you cannot manipulate individual string values. In order to do so, you have to take subsets of strings and form a new string. 
# + A string can be converted to a numerical type and vice versa (wherever applicable). Many a times, raw data, although numeric, is coded in string format. This feature provides a clean way to make sure all of the data is in numeric form.
# + Strings are sequence of characters and can be <b>tokenized</b>.
# + Strings and numbers can also be <b>formatted</b>.
# 
# 

# In[ ]:


strr = 'Hello World'
print (strr)            # prints complete string
print (strr[0])         # prints first character of string
print (strr[2:7])       # prints characters starting from 3rd to 6th. Exclude last element. 'W' is excluded.
print (strr[2:])        # prints string starting from 3rd character
print (strr*2)          # prints string two times
print (strr[:-3])
print (" Sai ".join([strr]*9))

print (strr + ' ' + strr + ' ' )
#print (strr + "TEST")   # prints concatenated string


# In[ ]:


strr ="hello"
print(strr[-4])

a = 'this is a string'

a = a.replace(' ','')
print (a) 

a = 'sai'
print (a + a)
#anum = int(a)         # Typecaste/Convert String '20' to Integer - Possible
#print (anum + anum)

a = 34.9
         # Typecaste/Convert String 'hi' to Integer - Not Possible
print (a + a)
anum = int(a)
print (anum + anum)

print ("sai")


# In[ ]:


a + len(a) # Concatenating String & Integer - Error


# In[ ]:


int(a) + len(a) # Convert to String so that concatenation works


# In[ ]:



# Lets us see how to create formatted outputs
strin = "this  is python"
strlist = list(strin)
print (strlist.sort())
sai = sorted(strlist)
print(strlist)
#print(sai)
#print (string)


form = '%d %s is $%d'            # Let us visualize the formatting '%.2f %s is $%d' is like 'Float String Integer'
print (form %(40.789, 'Argentine Pesos', 1))
print (form %(64,'Rupees',1))
form %(340,'Nigerian Nairas',1)


# In[ ]:


school = 'ISB'
print('s' in school)
print('S' in school)


# In[ ]:


#Consider two strings x and y
x = 'Confusing'
y = 'Strings'
#Swap first three characters of each string and join them by an _
strr1 = str(x[:3])
strr2 = str(y[:3])
print (strr2 + x[3:] + "_" + strr1 + y[3:])


# **Datetime**
# 
# Python has a <b>built-in datetime module</b> for working with dates and times.
# One can create strings from date objects and vice versa. 
# 
# 
# Python offers very powerful and flexible datetime capabilities that are capable of operating at a scale of time measured in nano-seconds. 

# In[ ]:


from datetime import datetime, date, time
#help(datetime.strptime)                                        # Help for a function
date1 = datetime(2014, 7, 16, 14, 45, 5)
print(date1.day)
print(datetime.strptime('20140516134328','%Y%m%d%H%M%S'))      # Parses a string representing a time according to a format
print (date1.strftime('We are in %d, %m %Y' ))                  #strftime() converts a time input into a string output. .
date1 = datetime(2014, 5, 16)
date2 = datetime(2015, 5, 19)
datediff = date2 - date1
print(datediff)

#date1


# **Lists**
# 
# + Lists, along with dictionary, are perhaps most important data types.
# + A list contains items separated by commas and enclosed within square brackets ([]).
# + All the items belonging to a list can be of <b>different data type</b>.
# + Lists are similar to arrays in C language.
# + The <b>plus ( + )</b> sign is the list concatenation operator, and the <b>asterisk ( * )</b> is the repetition operator.
# 

# In[ ]:


list_ = [ 'abcd', 786 , 2.23, 'ISB', 70.2 ]

tinylist = [123, 'ISB']

print (list_)            # Prints complete list
print (list_[0])         # Prints first element of the list
print (list_[1:3])       # Prints elements starting from 2nd till 3rd 
print (list_[2:])        # Prints elements starting from 3rd element
print (tinylist * 2)     # Prints list two times
print (list_ + tinylist) # Prints concatenated lists
print (len(list_))
str2 = list("This is Python")
str2


# In[ ]:


list_ = ['Ajay', 'Vijay', 'Ramesh']
list_.append("Sujay")         
list_.insert(0, 'NewGuy')       
#list_.extend('Sujay') 
print (list_)  
print (list_.index('Sujay')) 
list_.remove('Sujay')                 # Remove by Name
list_.pop(2)                         # Remove by location - Ajay is gone
print (list_)


# **Dictionary**
# 
# + One of the most important built-in data structures.
# + Python's dictionaries are kind of <b>hash tables</b>.
# + They work like associative arrays and consist of <font color='red'><b>key-value pairs</b></font>. 
# + A dictionary key can be almost any Python type, but are usually numbers or strings. 
# + Values, on the other hand, can be any arbitrary Python object.
# + Dictionaries are enclosed by <b>curly braces ( { } )</b> and values can be assigned and accessed using <b>square braces ( [] )</b>.
# 

# In[ ]:


dic = {}
dic['one'] = "This is one"
dic[2]     = "This is two"
dic[1]     = "This is the number one"
#dic[3] = "This is number 3"
dic['two'] = "This is the string two"
dic['three'] = "This is the three"
tinydict = {'name': 'isb','code':6734, 'dept': 'sales'}

print (dic)
#print (dic[0])       # Prints value for 'one' key

print (dic[1])           # Prints value for 2 key
print (tinydict)          # Prints complete dictionary
print (dic.keys())   # Prints all the keys
print (tinydict.values()) # Prints all the values
print (sorted(tinydict.keys()))
#print (dict)
    


# ** Lets see what is a Tuple? **<br/>
# A tuple is a sequence of immutable Python objects

# In[ ]:


a = (1,2,3,['sai',34])
print(a[0])

# Immutable, please read, try and tell me what do you understand
a[3][1]=10
print (a)
print(hex(id(a)))  # Address of the variable
a = a + (10,)
print(hex(id(a)))
id(a)


# **Conditional Statements**
# 
# **If-statement**
# 
# The <b>if</b> statement is used to check a condition: if the condition is true, we run a block of statements (called the if-block), else we process another block of statements (called the else-block). The <b>else clause is optional</b>.

# In[ ]:


a = 21.5
if a >= 22:
    
   print("if")
elif a <= 21:
    
   print("elif")
else:
   print("else")
print ("sai")


# In[ ]:


#Testing for an element in list
list_ = ['Ajay', 'Ramesh']
if  'Vijay' not in list_:
    print ('Found Vijay')
else:
    print ('Did not find Vijay')


# **While-statement**
# 
# The while statement allows you to <b>repeatedly execute</b> a block of statements as long as a condition is true. A while statement is an example of what is called a looping statement. A while statement can have an optional else clause.

# In[ ]:


count = 0
while (count < 9):
   print ('The count is:', count)
   count = count + 1
   print ('The count at the end of the loop', count)
print ("End of while loop!")


# **For-statement**
# 
# The for..in statement is another looping statement which iterates over a sequence of objects i.e. go through each item in a sequence. A sequence is just an ordered collection of items.

# In[ ]:


for (i=0;i<9;i++)


# In[ ]:


for i in range(0,9,1):
    print (i)


# The ``for`` loop specifies 
# - the variable name to use (in this case, `planet`)
# - the set of values to loop over (in this case, `planets`)
# 
# You use the word "``in``" to link them together.
# 
# The object to the right of the "``in``" can be any object that supports iteration. Basically, if it can be thought of as a group of things, you can probably loop over it. In addition to lists, we can iterate over the elements of a tuple:

# In[ ]:


planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
for planet in planets:
    print(planet, end=' ') # print all on same line


# In[ ]:



for i in range(0,9,1):
    print (i)
else:
   print ("The for loop is over and i value is %d" %(i))


# In[ ]:


# Traversing a list
# Guess What is the output
numbers = [2, 3, 5]
sum = 0

for i in numbers:
    i = i + 2
    print (i)
    sum = sum + i
print (sum)


# In[ ]:


numbers = [2, 3, 5]
getsum = [ i+2 for i in numbers ]
print (getsum)


# In[ ]:


numbers = (4, 5, 7)
getnum = [] i+2 for i in numbers if i<6]
print (getnum)


# ## ``while`` loops
# The other type of loop in Python is a ``while`` loop, which iterates until some condition is met:

# In[ ]:


i = 0
while i < 10:
    print(i, end=' ')
    i += 1


# The argument of the ``while`` loop is evaluated as a boolean statement, and the loop is executed until the statement evaluates to False.

# **Break statement**
# 
# The break statement is used to break out of a loop statement i.e. stop the execution of a looping statement, even if the loop condition has not become False or the sequence of items has not been completely iterated over.
# 
# An important note is that if you break out of a for or while loop, any corresponding loop else block is not executed.

# In[ ]:


for i in range(1,10):
       if i == 5:
           break
           print (i)
print('Done')


# **Functions**
# 
# + Functions are <b>reusable piece of software</b>.
# + Block of statements that <b>accepts some arguments, perform some functionality, and provides the output</b>.
# + Defined using <font color='red'><b>def</b></font> keyword
# + Similar to functions in R.
# + For example, implement code to perform two way clustering once and can be used again in the same program. 
# + We will look at functions for a number of features (Fama Mac Beth regression, two way clustering, industry code classification) in subsequent sessions. 
# + A function can take arguments.
# + Arguments are specified within parentheses in function definition separated by commas.
# + It is also possible to assign default values to parameters in order to make the program flexible and not behave in an unexpected manner.
# + One of the most <b>powerful feature of functions is that it allows you to pass any number of arguments (*argv) and you do not have to worry about specifying the number when writing the function</b>. This feature becomes extremely important when dealing with lists or input data where you do not know number of data observations before hand.
# + Scope of variables defined inside a function is <b>local</b> i.e. they cannot be used outside of a function.
# 

# In[ ]:


def sayHello():
    ''' This is a demonstration of 
    saying a big
    and 
    long hello'''
     # block belonging to the function
# End of function #

#sayHello()
help(sayHello)


# In[ ]:


def printMax(a, b):
   if a > b:
       print(a, 'is maximum')
   elif a == b:
       print(a, 'is equal to', b)
   else:
       print(b, 'is maximum')

printMax(3) 


# In[ ]:


x = 50
def func(x):
   print('x is', x)
   x = 2
   print('Changed local x to', x)
func(x)
x=3
print('x is still', x)


# **Modules**
# 
# + Functions can be used in the <b>same program</b>. 
# + If you want to use function (s) in <b>other programs, make use of modules</b>.
# + Modules can be imported in other programs and functions contained in those modules can be used.
# + Simplest way to create a module is to write a <b>.py file</b> with functions defined in that file.
# + Other way is to <b>import</b> using byte-compiled .pyc files. 

# In[ ]:


import math
x = -25
print (abs(x))
print(math.fabs(x))
print(math.factorial(abs(x)))


# In[ ]:


import sys
sys.path.append(your_local_path)
from mymodule import *
print(sayhi())


numb = input("Enter a non negative number: ")
num_factorial = factorial(int(numb))

