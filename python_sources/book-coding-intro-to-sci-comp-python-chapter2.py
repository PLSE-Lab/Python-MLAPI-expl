#!/usr/bin/env python
# coding: utf-8

# In[1]:


#* Book Title: Introduction to Scientific Computing in Python - Robert Johansson
#* Code Title: Chapter2 Introduction to Python programming Codes
#* Description: N/A
#* Version logs with details:
#    -Version |   Date   |      Nam       |   Details  
#    -  v0.1    04022019   GyeongBong Kim    Beginning   


# # Chapter 2.3 Modules

# ## &nbsp;&nbsp;&nbsp; Chapter 2.3.1 References (Import modules)

# In[2]:


import math
x = math.cos(2*math.pi)
print(x)


# In[3]:


from math import *
x = cos(2*pi)
print(x)


# ## &nbsp;&nbsp;&nbsp; Chapter 2.3.2  Looking at what a module contains, and its documentation 

# In[4]:


import math
print(dir(math))


# In[5]:


help(math.log) #how to use help() function
print(log(10))
print(log(10, 2))


# In[6]:


help(math) #how to use help() function 2


# # Chapter 2.4 Variables and types
# 
# ## &nbsp;&nbsp;&nbsp; Chapter 2.4.2 Assignment

# In[7]:


x = 1.0
my_variable = 12.2

print(type(x))
print(type(my_variable))


# In[8]:


x = 1
type(x)


# In[9]:


type(y) #error


# ## &nbsp;&nbsp;&nbsp; Chapter 2.4.3 Fundamental types

# In[ ]:


x = 1
type(x)


# In[ ]:


x=1.0
type(x)


# In[ ]:


b1 = True
b2 = False
print(type(b1))
print(type(b2))


# In[ ]:


x = 1.0 - 1.0j
print(type(x))
print(x)
print(x.real, x.imag)


# ## &nbsp;&nbsp;&nbsp; Chapter 2.4.4 Type utility functions

# In[10]:


import types
print(dir(types))


# In[11]:


x = 1.0
type(x) is float


# In[12]:


type(x) is int


# In[13]:


isinstance(x,  float)


# ## &nbsp;&nbsp;&nbsp; Chapter 2.4.5 Type casting

# In[14]:


x = 1.5
print(x , type(x))


# In[15]:


x = int(x)
print(x, type(x))


# In[16]:


z = complex(x)
print(z , type(z))


# In[17]:


x = float(z)


# In[18]:


y = bool(z.real)
print(z.real, "->", y, type(y))

y = bool(z.imag)
print(z.imag, "->", y, type(y))


# # Chapter 2.5 Operators and comparisons

# In[19]:


1+2, 1-2, 1*2, 1/2


# In[20]:


1.0+2.0, 1.0-2.0, 1.0*2.0, 1.0/2.0


# In[21]:


3.0//2.0


# In[22]:


2 ** 2


# In[23]:


True and False


# In[24]:


not False


# In[25]:


True or False


# In[26]:


2>1, 2<1


# In[27]:


2>2, 2<2


# In[28]:


2>=2, 2<=2


# In[29]:


print([1, 2] == [1, 2])
print([1, 2] == [3, 2])


# In[30]:


l1 = l2 = [1, 2]
print(l1, l2)
print(l1 is l2)


# # Chapter 2.6 Compound types: Strings, List and dictionaries
# ## &nbsp;&nbsp;&nbsp; Chapter 2.6.1 Strings

# In[31]:


s = 'Hello world'
type(s)


# In[32]:


len(s)


# In[33]:


s2 = s.replace('world', 'test')
print(s2)


# In[34]:


s[0]


# In[35]:


print(s[0:5])
print(s[4:5])


# In[36]:


print(s[:5])
print(s[6:])
print(s[:])


# In[37]:


print(s[::1])
print(s[::2])


# In[38]:


print('str1', 'str2', 'str3')


# In[39]:


print('str1', 1.0, False, -1j)


# In[40]:


print('str1' + 'str2' + 'str3')


# In[41]:


print('value = %f' % 1.0)


# In[42]:


s2 = 'value1 = %.2f, value2 = %d' % (3.1415, 1.5)
print(s2)


# In[43]:


s3 = 'value1 = {0}, value2 = {1}'.format(3.1415, 1.5)
print(s3)


# ## &nbsp;&nbsp;&nbsp; Chapter 2.6.2 List

# In[44]:


l1 = [1, 2, 3, 4]
print(type([]))
print(l1)


# In[45]:


print(l1)
print(l1[1:3])
print(l1[::2])
print(l1[0])


# In[46]:


l2 = [1, 'a', 1.0, 1-1j]
print(l2)

for i, elem in enumerate(l2):
    print(type(elem))
    


# In[47]:


nested_list = [1, [2, [3, [4, [5]]]]]
print(nested_list)


# In[48]:


start = 10
stop = 30
step = 2
a = range(10, 30, 2)
# for i in a:
#     print(i)
print(list(a))


# In[49]:


list(range(-10, 10))


# In[50]:


print(s)
s2 = list(s)
print(s2)
s2.sort()
print(s2)


# In[51]:


#Adding, inserting, modifying, and removing elements from lists
l = []
l.append('A')
l.append('B')
l.append('C')
print(l)


# In[52]:


l[1] = 'p'
l[2] = 'p'
print(l)


# In[53]:


l[1:3] = ['d', 'd']
print(l)


# In[54]:


l.insert(0, 'i')
l.insert(1, 'n')
l.insert(2, 's')
l.insert(3, 'e')
l.insert(4, 'r')
l.insert(5, 't')
print(l)


# In[55]:


l.remove('A')
print(l)


# In[56]:


del l[7]
del l[6]
print(l)


# In[57]:


test = ['Test sentence']
print(test)
test.append('hi') #Compare with 'extend'
print(test)
test.extend('hi')
print(test)


# ## &nbsp;&nbsp;&nbsp; Chapter 2.6.3 Tuples

# In[58]:


point = (10, 20)
print(point, type(point))


# In[59]:


point = 10, 20
print(point, type(point))


# In[60]:


x, y = point
print('x=', x)
print('y=', y)


# In[61]:


point[0] = 20


# ## &nbsp;&nbsp;&nbsp; Chapter 2.6.4 Dictionaries

# In[62]:


params = {'parameter1' : 1.0,
         'parameter2' : 2.0,
         'parameter3' : 3.0,}
print(type(params))
print(params)


# In[63]:


print('param1 = ' + str(params['parameter1']))
print('param2 = ' + str(params['parameter2']))
print('param3 = ' + str(params['parameter3']))


# In[64]:


params['parameter1'] = 'A'
params['parameter2'] = 'B'
params['parameter4'] = 'D'

print('param1 = ' + str(params['parameter1']))
print('param2 = ' + str(params['parameter2']))
print('param3 = ' + str(params['parameter3']))
print('param4 = ' + str(params['parameter4']))


# # Chapter 2.7 Control Flow
# ## &nbsp;&nbsp;&nbsp; Chapter 2.7.1 Conditional statements: if, elif, else

# In[65]:


statement1 = False
statement2 = False

if statement1:
    print('statement1 is True')
elif statement2:
    print('statement2 is True')
else:
    print('statement1 and statement2 are False')


# In[66]:


#Indentation examples
statement1 = statement2 = True

    #Good indentation
if statement1:
    if statement2:
        print('both statement1 and statement2 are True')


# In[67]:


#Bad indentation
if statement1:
if statement2:
print('both statement1 and statement2 are True')


# In[68]:


statement1 = False

if statement1:
    print('printed if statement1 is True')
    print('still inside the if block')


# In[69]:


if statement1:
    print('printed if statement1 is True')
print('now outside the if block')


# # Chapter 2.8 Loops
# ## &nbsp;&nbsp;&nbsp; Chapter 2.8.1 for loops:

# In[70]:


for x in [1, 2, 3]:
    print(x)


# In[71]:


for x in range(4):
    print(x)


# In[72]:


for x in range(-3, 3):
    print(x)


# In[73]:


for word in ['scientific', 'computing', 'with', 'python']:
    print(word)


# In[74]:


for key, values in params.items():
    print(key + '=' + str(values))


# In[75]:


for idx, x in enumerate(range(-3 , 3)):
    print(idx, x)


# ## &nbsp;&nbsp;&nbsp; Chapter 2.8.2 List comprehensions: Creating lists using for loops:

# In[76]:


l1 = [x**2 for x in range(0, 5)]
print(l1)


# ## &nbsp;&nbsp;&nbsp; Chapter 2.8.3 while loops:

# In[77]:


i = 0
while i < 5:
    print(i)
    i = i + 1
print('done')


# # Chapter 2.9 Functions:

# In[78]:


def func0():
    print('test')
    
func0()


# In[79]:


def func1(s):
    '''
    Print a string 's' and tell how many characters it has
    '''
    
    print(s + ' has ' + str(len(s)) + ' characters')
    
help(func1)

func1('test1')


# In[80]:


def square(x):
    '''
    return the square of x.
    '''
    
    return x ** 2

square(4)


# In[81]:


def powers(x):
    """
    Return a few powers of x.
    """
    
    return x ** 2, x ** 3, x ** 4

powers(3)


# In[82]:


x2 ,x3, x4 = powers(3)

print(x2, x3, x4)


# ## &nbsp;&nbsp;&nbsp; Chapter 2.9.1 Default argument and keyword arguments

# In[83]:


def myfunc(x, p=2, debug=False):
    if debug:
        print('evaluating myfunc for x = ' + str(x) + ' using exponent p = ' + str(p))
    return x ** p


print(myfunc(5))
print(myfunc(5, debug=True))


# In[84]:


myfunc(p=3, debug=True, x=7)


# ## &nbsp;&nbsp;&nbsp; Chapter 2.9.2 Unnamed functions (lambda function)

# In[85]:


f1 = lambda x: x**2

def f2(x):
    return x**2

f1(2), f2(2)


# In[86]:


var1 = map(lambda x: x**2, range(-3, 4))
print(list(var1))

help(map)


# # Chapter 2.10 Classes

# In[87]:


class Point:
    """
    Simple class for representing a point in a Cartesian coordinate system.
    """
    
    def __init__(self, x, y):
        """
        Create a new Point at x, y.
        """
        self.x = x
        self.y = y
        
    def translate(self, dx, dy):
        self.x += dx
        self.y += dy
        
    def __str__(self):
        return('Point at [%f, %f]' % (self.x, self.y))
    
    
p1 = Point(0, 0)
print(p1)

p2 = Point(1, 1)
p1.translate(0.25, 1.5)

print(p1)
print(p2)


# # Chapter 2.11 Modules

# In[88]:


get_ipython().run_cell_magic('file', 'mymodule.py # %%file is a ipython magic function to create a certain file. ', '"""\nExample of a python module. Contains a variable called my_variable, a function called my_function, and a class called MyClass.\n"""\n\nmy_variable = 0\n\ndef my_funoction():\n    \'\'\'\n    Example function\n    \'\'\'\n    return my_variable\n\nclass MyClass:\n    def __init__(self):\n        self.variable = my_variable\n    \n    def set_variable(self, new_value):\n        self.variable = new_value\n        \n    def get_variable(self):\n        return self.variable')


# In[89]:


import mymodule

help(mymodule)


# In[90]:


mymodule.my_variable


# In[91]:


mymodule.my_funoction()


# In[92]:


my_class = mymodule.MyClass()
my_class.set_variable(10)
my_class.get_variable()


# # Chapter 2.12 Exceptions:

# In[93]:


raise Exception('description of the error')


# In[94]:


try:
    print('new_test')
    print(new_test)
except:
    print('Caught an exception')


# In[95]:


try:
    print('new_test')
    print(new_test)
except Exception as e:
    print('Caught an exception: ' + str(e))
    

