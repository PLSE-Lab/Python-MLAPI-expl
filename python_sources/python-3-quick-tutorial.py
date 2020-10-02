#!/usr/bin/env python
# coding: utf-8

# # Python 3 Quick Tutorial
# 
# This kernel explains important concepts in Python 3 for a beginner.  If you are able to understand this kernel in its entirely, then you are all set to proceed further with learning _NumPy and Pandas_ modules of Python.  If not, it is good idea to learn Python 3 quickly through this ebook:
# http://ebooks.mobibootcamp.com/python/index.html
# 
# Here are the key concepts covered in the book. Refer book for detailed explanation.
# 
# ### __for loops and range__
# 
# __for__ loops are commonly used in any programming language. Although using _Pandas_ module vastly reduces our need to use loops in Data Analytics/Science, it is still important to understand __for__ loops as it is one of the essential constucts of Python programming language.  __range__ function is specific to Python and is used quite commonly in Data Analytics.
# 
# Refer http://ebooks.mobibootcamp.com/python/controlstatements.html
# 
# http://ebooks.mobibootcamp.com/python/lists.html
# 
# for understanding more on _for_ loops and _range_ 

# In[ ]:


for x in range(0, 3):
    print ("Hip hip hurray!  %d" % (x))


# ### __try, catch, except blocks__
# 
# Any program should gracefully handle an exceptional condition. In Data Analytics, when you are processing large amounts of data, you are bound to come across scenarios when the data being processed does not meet the requirements of your program.  In other words, you never thought of data represented in a certain way when you wrote your program and when you run your program against a large set, you come across several different types of issues with the data format. In such conditions, you can end the program in two ways: 
# 
# * Program crashes by throwing an exception upon hitting the first exceptional condition.
# * Program proceeds further on every exceptional condition by noting down the unexpected erroneous data.
# 
# The first choice is relevant when there is a fundamental problem because of which the program cannot proceed, for e.g., the data file is missing. In such cases you have to crash the program by throwing an exception as there is no point in continuing without the data file.
# 
# Second choice is the best for purely data related issues as it saves tremendous time by saving you from multiple invocations of the program for handling multiple different data errors. After the proram finsihes its first run, you can fix your program or discard the erroneous data, as the case may be, for all the errors in one go and then run the program again the second and last time as you have taken care of all fixes for the second round.
# 
# 
# 
# Refer: http://ebooks.mobibootcamp.com/python/functions.html for understanding more on functions and exception handling
# 

# In[ ]:


try: # try statement encloses statements which could throw an exception
    deposit_amount = float(input("Enter deposit amount:\t\t")) # input function chained with float function
    interest_earned = float(input('Enter total interest earned:\t'))
    interest_rate = (interest_earned/deposit_amount) * 100
    print("Rate of interest: \t\t"+ str(round(interest_rate,2))) # round function rounds the result to 2 decimal places
except: # catch all exceptions if there is no name of the exception. Can catch just 'ValueError' in this case
    print("You entered an invalid integer. Please type a valid value")
finally:
    print("this will be executed no matter what") 


# ### __set__ structure
# 
# Set contains unique values and duplicates are not allowed
# 
# Refer: http://ebooks.mobibootcamp.com/python/dictionaries.html for more on unordered collections

# In[ ]:


americas = set(['Canada', 'United States', 'Mexico', 'Mexico'])
print(americas)


# ### __Functions__
# 
# Defining and using functions is another basic contruct which everyone should know before starting any meaningful analytics.
# 
# Refer: http://ebooks.mobibootcamp.com/python/functions.html for understanding more on functions
# 

# In[ ]:


# function definition with one argument, limit 
def generate_even_numbers(limit):  
    '''
    This method generates even numbers from 0 to the value 
    passed in the limit, excluding the limit
    '''
    even_number_list = []  # create an empty list 
    for i in range(limit):
        if(i % 2 == 0):  # remainder 0 indicates even
            even_number_list.append(i) # add to the list
    return even_number_list # return statement and function def ends

   
print(generate_even_numbers(10))   # call the defined function
help(generate_even_numbers) 


# ### __while loop__
# 
# This is similar to __for__ and another fundamental control statements in Python.
# 
# Refer: http://ebooks.mobibootcamp.com/python/controlstatements.html

# In[ ]:


# while loop
x = 1.456
while True:
    print ("Integer value of x {:1.0f}".format(x))
    print ("Value of x rounded to 2 decimal points {:.2f}".format(round(x,2)))
    x += 1
    if(x > 3):
        break;


# In[ ]:


#define function
name = 'My name'
def myfunc(name):
    print("hello there! " +  name) # name from outside the block is replaced with the local name. 

myfunc("Python")


# ### __conditional and logical expressions__
# 
# Any programming language needs to define how a conditional or logical expression can be constructed.  
# 
# Refer:  http://ebooks.mobibootcamp.com/python/firstConcept.html and http://ebooks.mobibootcamp.com/python/controlstatements.html
#             

# In[ ]:


# not keyword
game_over = False
i = 1;
while not game_over:
    print("playing " + str(i) + " time(s)")
    i+=1
    if(i>2):
        break


# ### __lists__
# 
# Although as Data Analysts/Scientists we use list type of constructs available in NumPy (Arrays) and Pandas (Series), it is still very important to understand the Python lists as both NumPy and Pandas are based out of this.
# 
# Refer: http://ebooks.mobibootcamp.com/python/lists.html

# In[ ]:


# sending list to function
topics = ["numpy",'pandas',"seaborn"]
def displayTopics(topics):
    for topic in topics:
        print(topic)
        
displayTopics(topics)
    


# In[ ]:


# list of lists - 2d 
topics = [['numpy', 1],
         ['pandas',2],
         ['seaborn',3]]

print(topics)
print(topics[1][0])


# ### __tuples__
# 
# Tuples are used heavily in data analytics as they are more efficient than lists for static structures
# 
# Refer: http://ebooks.mobibootcamp.com/python/lists.html
#         

# In[ ]:


#tuples are similar to lists except they are immutable so you can't add, remove or set items into a tuple
t_tuples = ('numpy',"pandas","seaborn")
print(t_tuples[0])
a,b,c = t_tuples  #tuples can be unpacked into multiple assignment statements
print(b)
# tuples are more efficnet than lists so for readonly structures use tuples


# ### __String Manipulation__
# 
# Understanding string manipulation is very important as it is heavily used in Data Wrangling/Display process.
# 
# Refer: http://ebooks.mobibootcamp.com/python/strings.html
# 

# In[ ]:


# string search
message = "Congratulations!  you have are all set to move on to learning numpy and pandas!"
print('you' in message)
print('congratulations' in message) # case sensitive
for char in "study":
    print(char)
print(message.startswith("Congratulations"))
    


# In[ ]:


# string manipulations
numbers = "12345"
print(numbers.isdigit()) # other functions islower(), isupper(), isalpha()

message = "this is really easy!"
print(message.title())

phoneNumber = "123 234 3489    "
print(phoneNumber.strip() + ".")
print(phoneNumber.replace(" ", "-")) # find returnes the index of the first occurance or -1 if not found
print("(" + phoneNumber[:3] + ")"+ phoneNumber[4:7] + "-"+ phoneNumber[8:13])
print(phoneNumber.split(" "))

print("book".ljust(14), "$9.99".rjust(10)) # justifies to the given length by adding spaces to fill the gap

print(message.upper())


# In[ ]:


# join() method adds the first string to every part of the second string and is more efficient than + or += with strings and lists
a = " "
b = "355"
print(a.join(b))


# ### __datetime__
# 
# Date, time, datetime are the fundamental building blocks of pandas _timeseries_ used in financial data analysis. 
# 
# Refer: http://ebooks.mobibootcamp.com/python/datetime.html
# 
# 

# In[ ]:


# date time
from datetime import date
from datetime import datetime
print(date.today())
print(datetime.now())
peace_day = datetime(1981,9,21, 17, 30)
print(peace_day)
print(peace_day.strftime("%Y/%m/%d"))


# ### __dictionaries__
# 
# Dictionary constructs are used to hold associations between a Key and its Values.  Lookup is based on using a Key to gets it respective value.
# 
# Refer: http://ebooks.mobibootcamp.com/python/dictionaries.html

# In[ ]:


#Dictionaries. lists are ordered but dictionaries are unordered collection. Keys are indexed. Key can be any type. 
# Value can be any type including complex types
countries = {'CA': "Canada",
            "US":"United States",
            "MX": "Mexico",
            3:10}
print (countries)
print(countries['CA'])
code = 'US'
if code in countries:
    print(countries[code])
    
print(countries.get("mx")) # case sensitive
print(countries.get("MX"))

countries['IN'] = "India"
print(countries['IN'])
countries['IN'] = 'Bharath'
print(countries['IN']) 
del countries['MX']
print(countries)
countries.pop("IN")  # you can use del, pop methods to remove an item from dictionary. clear() removes all items
print(countries)
print(countries.keys())
print(countries.values())
for name in countries.values():
    print (name)
    
for code,name in countries.items():  # unpack tuples
    print(code , name)


# ### __common dictionary manipulations__
# 

# In[ ]:


# convert dictionary to list
codes = list(countries.keys())
print(type(codes))
del codes[2] # remove the integer so sort can work

codes.sort() 
print(codes)


# ### YAY! You made it this far...
# 
# You are all set with understanding fundamental building blocks of Python. Now it is time to proceed with understanding NumPy and Pandas, the Python modules which makes working with data that much fun and easy!
# 
# Follow me on [Facebook](https://www.facebook.com/mobibootcampcorp/) and/or [Twitter](https://twitter.com/mobibootcampcor) to be the first to know when it is released!
# 
# 
# 
# 
