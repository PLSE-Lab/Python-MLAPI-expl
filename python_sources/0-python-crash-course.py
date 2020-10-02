#!/usr/bin/env python
# coding: utf-8

# # Python Crash Course

# `int` can be thought of as short for integers
# *example:* $1, 2, 3, 4, -5, -99998, 177098$ $etc.$
# 
# `float` can be thought of real numbers but always have a decimal
# *example:* $ 1.34565, 2.0, 4.9, -.5, -99998.$ $etc.$

# In[ ]:


a = 2
print(a, type(a))


# In[ ]:


a = 4.2
print(a, type(a))


# ### strings
# 
# `str`  (aka string) in python is a charachter or multiple charachters. Think of characters as the letter, number, symbol combo that you can type out with your keyboard.

# In[ ]:


a = "hey there"
print(type(a), a)

a = "hi"
print(type(a), a)

a = 'i'
print(type(a), a)

a = "d"
print(type(a), a)


# In[ ]:


b = 1

sum = a + b
print(sum, type(sum))


# Your first error! 
# 
# Now a good rule of thumb when you get errors is to **try deciphering the last line of the error message** and figure what went wrong or use the all powerful Google! Here's the [first link](https://stackoverflow.com/questions/51252580/new-to-coding-python-error-typeerror-can-only-concatenate-str-not-int-t/51252705). 
# 
# Else try the first line of the error message.

# Here's a style of writing code in jupyter notebook. When you write code in cells. Make sure you do everything in your power to keep your code sequential. An example to make that more concrete...

# In[ ]:


print(z)


# In[ ]:


z = "I'm down here. Help :("


# Python isn't  picky whether you use `" "` or `' '` both work just as fine. Not in other languages though!

# In[ ]:


a = "hey" + 'there'
print(a, type(a))


# `str.endswith()` is function that you can perform on a string.

# In[ ]:


str1 = "Papa Kevin"
str2 = "Mama Kevin"
str3 = "Kevin 11"
str4 = "7-11"

print(str1.endswith("Kevin"))
print(str2.endswith("in"))
print(str2.endswith("  11"))


# Hol' up! What is this output `True` and `False`? Try `type()` on the output.

# In[ ]:


type(str1.endswith("Kevin"))


# Try `shift+tab` on bool for more info!

# In[ ]:


bool


# You can think of bool of saying True or False. Kind of like 1 for True and 0 for False. So you only need 1 binary bit to store the answer and is more space efficient. Now `int` can be 32-bit or 64-bit binary number. 

# ### Lists
# 
# List can hold pretty much anything. They're far more versatile than arrays (if you know what they are...) But if you don't that's cool too! 
# 
# Think of it as a "list" that hold different things. And, indexing starts at zero. That is zeroth, first, second, third element and so on...

# In[ ]:


list1 = list()
list2 = []

print("list1:", list1)
print("list2:", list2)


# In[ ]:


list1.append("a")
list1.append("b")
list1.append("c")

# printing entire list
print(list1)

# printing the zeroth element of the list
print(list1[0])

# printing the second element of the list
print(list1[2])


# What happens if you try to print `list1[3]`? 

# In[ ]:


print(list1[3])


# There is no 3rd element in `list1`. Hence there `IndexError`.

# ### if elif else
# 

# In[ ]:


cholesterol = "hi"
# cholesterol = "med"
# cholesterol = "low"
# cholesterol = "over level 9000"

if (cholesterol == "hi"):
    print("Get help, now!")
elif (cholesterol == "med"):
    print("Take it easy!")
else:
    print("Enjoy :)")


# In[ ]:


movieRating = 3
# movieRating = 9.5
# movieRating = 7
# movieRating = 5.4
# movieRating = 9999
# movieRating = -2

if (movieRating > 8 and movieRating >=10):
    print("Loved")
elif (movieRating > 6):
    print("Liked it")
elif (movieRating > 4):
    print("Ok")
elif (movieRating > 2):
    print("Disliked it")
elif (movieRating > 0):
    print("Hated it")
else:
    print("Invalid movieRating")
    


# In[ ]:


sunOut = True
clouds = False
rain = False
snow = False

if (sunOut):
    print("Sunny")
elif (clouds):
    if (rain):
        print("Rainy")
        print("Take an Umbrella!")
    else:
        print("Cloudy")
elif (snow):
    print("Snowing")
    print("Wear a jacket")
else:
    print("It's cloudy with a chance of meatballs")


# In[ ]:


sunOut = False
clouds = True
rain = True
snow = False

if (sunOut):
    print("Sunny")
elif (clouds):
    if (not rain):
        print("Cloudy")
    else:
        print("Rainy")
        print("Take an Umbrella!")        
elif (snow):
    print("Snowing")
    print("Wear a jacket")
else:
    print("It's cloudy with a chance of meatballs")


# In[ ]:


sunOut = False
clouds = False
rain = False
snow = False

if (sunOut):
    print("Sunny")
elif (clouds):
    if (not rain):
        print("Cloudy")
    else:
        print("Rainy")
        print("Take an Umbrella!")        
elif (snow):
    print("Snowing")
    print("Wear a jacket")
else:
    print("It's cloudy with a chance of meatballs!")


# ### def function():
# 
# You can see how annoying it can get to copy-paste the same lines of code over and over again. To prevent all that copy-pasting and increasing the size of code in your file or in this case "cell". We use functions!

# In[ ]:


def predictWeather(sunOut, clouds, rain, snow):
    if (sunOut):
        print("Sunny")
    elif (clouds):
        if (not rain):
            print("Cloudy")
        else:
            print("Rainy")
            print("Take an Umbrella!")        
    elif (snow):
        print("Snowing")
        print("Wear a jacket")
    else:
        print("It's cloudy with a chance of meatballs!")
    


# Here are multiple ways to call the same fucntion. You can m some ways are more convenient than others. 

# In[ ]:


sunOut = True
clouds = False
rain = False
snow = False

predictWeather(sunOut, clouds, rain, snow)


# In[ ]:


predictWeather(True, False, False, False)


# In[ ]:


predictWeather(sunOut=True, clouds=False, rain=False, snow=False)


# In[ ]:


predictWeather(clouds=False, rain=False, snow=False, sunOut=True)


# In[ ]:


def predictWeather(sunOut, clouds, rain, snow=False):
    if (sunOut):
        print("Sunny")
    elif (clouds):
        if (not rain):
            print("Cloudy")
        else:
            print("Rainy")
            print("Take an Umbrella!")        
    elif (snow):
        print("Snowing")
        print("Wear a jacket")
    else:
        print("It's cloudy with a chance of meatballs!")


# In[ ]:


predictWeather(False, False, False)


# In[ ]:


def predictWeather(sunOut, clouds, rain, snow=True):
    if (sunOut):
        print("Sunny")
    elif (clouds):
        if (not rain):
            print("Cloudy")
        else:
            print("Rainy")
            print("Take an Umbrella!")        
    elif (snow):
        print("Snowing")
        print("Wear a jacket")
    else:
        print("It's cloudy with a chance of meatballs!")


# In[ ]:


predictWeather(False, False, False)


# In[ ]:


def predictWeather(sunOut, clouds=True, rain, snow=True):
    if (sunOut):
        print("Sunny")
    elif (clouds):
        if (not rain):
            print("Cloudy")
        else:
            print("Rainy")
            print("Take an Umbrella!")        
    elif (snow):
        print("Snowing")
        print("Wear a jacket")
    else:
        print("It's cloudy with a chance of meatballs!")


# Did you notice that `print()` is also a function? You can spot a function by `()`. 
# 
# Let's say you have to `print("ACM")` 5 times. You could copy-paste that. How about 10? Still do-able.
# 
# What about 100? You could count your number of pastes? **What about a 1000** `print("ACM")`**?** You'd definitely loose track of the count and loose your mind in the process!
# 
# And copy pasting code over and over and over and over and over again looks ugly. Get my flow? :P

# In[ ]:


print("ACM")
print("ACM")
print("ACM")
print("ACM")
print("ACM")
print("ACM")
print("ACM")


# ### for loops
# 
# Loops are used when you need to perform the same lines of code over and over again and when doing function calls just can't suffice. Below is a basic example of a `for` loop using `range()`

# In[ ]:


for i in range(0, 10, 1):
    print(i)


# Let's play around with the `range()` to figure out how it works. Whenever you learn a new coding topic. It's essential  you play around with it. 
# 
# Try weird values. Predict the output. And then check if it matches your assumption. Break it. Tweak it. Test it. Grok it.
# 
# **Don't just read code. Write more code than you read. And read a lot!**

# In[ ]:


for i in range(0, 10, 3):
    print(i)


# In[ ]:


for i in range(0, 10):
    print(i)


# I'm going to leave you to do more experimenting. Feel free to do that with all cells in this notebook. 
# 
# A good rule of thumb is never deleting your code. You can always comment your code out and then delete it at the very end. Often a time you'll delete code that was off by 1 variable and re-write something that just don't even work.
# 
# Comment your code. **Or** make a new cell in notebook. Get comfortable with navigating the notebook. 
# 
# *TODO show shortcuts*

# In[ ]:


print(list1)


# In[ ]:


for element in list1:
    print(element)


# ### modules

# Now what if writing functions, loops, if statements and all that jazz gets annoying? If you're working on a project that needs you to have multiple python files that use the same lines of code over and over again. You use modules. You could either make your own modules or install modules that someone else has made to make life easier.

# In[ ]:


import math

a = 2
result = math.pow(a, 5)
print(result)


# You could also do.

# In[ ]:


print(a ** 5)


# What things get far more complex? What if you have to find the $sine$ of a value?

# In[ ]:


a = 5
math.sin(a)


# In[ ]:


math.pi


# Let's try finding the sine of a list of values

# In[ ]:


list2 = [1, math.pi, 89, -3]

for element in list2:
    math.sin(element)
    
print(list2)


# Now let's try storing it...

# In[ ]:


list2 = [1, math.pi, 89, -3]

for elem in list2:
    elem = math.sin(elem)
    
print(list2)


# That didn't work. Reason being it just copies the element in `list2` to the variable `elem` and any changes you do to element doesn't do anything because `elem` only has a copy of the current element you're currently looping through in `list2`.

# In[ ]:


for i, element in enumerate(list2):
    list2[i] = math.sin(element)
    
print(list2)

