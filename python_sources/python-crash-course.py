#!/usr/bin/env python
# coding: utf-8

# * Data types
# * Numbers
# * Strings
# * Printing
# * Lists
# * Dictionaries
# * Booleans
# * Tuples
# * Sets
# * Comparison Operators
# * if, elif, else Statements
# * for Loops
# * while Loops
# * range()
# * list comprehension
# * functions
# * lambda expressions
# * map and filter
# * methods

# **Data Types**

# **Numbers**

# In[ ]:


1


# In[ ]:


1.0


# In[ ]:


1+1


# In[ ]:


1+3


# In[ ]:


1/5


# In[ ]:


1/2


# **Exponents**

# In[ ]:


2**4  #2^4


# In[ ]:


2+3*5+5 #3*5


# In[ ]:


(2+3)*(5+5)


# **MOD FUNCtion**

# In[ ]:


4%2 #%=mod


# In[ ]:


5%2


# In[ ]:


8%2


# **Variable Assignment**

# In[ ]:


#can not start with number or spacial character
var = 2


# In[ ]:


var


# In[ ]:


x=2
y=3


# In[ ]:


x+y


# In[ ]:


x=x+x


# In[ ]:


x


# In[ ]:


x=x+x


# In[ ]:


x


# In[ ]:


#can't start with number
12var = 1


# In[ ]:


#can't start with special symbols
|var=1


# In[ ]:


#we can use underscore to separate them
var_sourav=1


# In[ ]:


var_sourav


# **Strings**

# In[ ]:


'Single quote'


# In[ ]:


"This is a doube quote string"


# In[ ]:


"I can't go"


# In[ ]:


x= 'hello'


# In[ ]:


x


# In[ ]:


print(x) #there will be no out indicator and there will be no single quotes


# **Print staff based on the variable**

# In[ ]:


num=12
name = 'sourav' #I wanna print stuffs based on that variable


# In[ ]:


print('my number is {} and my name is {}'.format(num,name))


# In[ ]:


print('My number is {one} and my name is {two}, more {one}'.format(one=num,two=name)) 
# In this case we dont need to think about the format order


# **How to Grab specific elements from the string**

# In[ ]:


s='hello'


# In[ ]:


s[0]


# In[ ]:


s[4]


# In[ ]:


s[5]


# In[ ]:


s[-1]


# In[ ]:


s[-2]


# **How to grab more than one element?**

# In[ ]:


s='abcdefghijk'


# In[ ]:


s[0]


# In[ ]:


s[0:]  #means starting at zero grab everything beyond it


# In[ ]:


s[:3] #grab everything upto a certain index without the mentioned number not including the number


# In[ ]:


s[0:3]


# In[ ]:


s[1:4]


# In[ ]:


# for 'def'
s[3:6]


# **LIST: The sequence of elemens in a set of squre brackets separeted by commas**
# 

# In[ ]:


# we can make a list of numbers 
[1,2,3]


# In[ ]:


#we can make a list of strings also
['a','b','c']


# In[ ]:


my_list=['a','b','c']


# **How to add element to the list???**
# **The answer is use the append method**

# In[ ]:


my_list.append('d')


# In[ ]:


my_list


# **How to grab the item of the list??**

# In[ ]:


my_list[0]


# In[ ]:


my_list[0:]


# In[ ]:


my_list[1:3]  #compared to strings index points are separeted by commas


# In[ ]:


#How to replace the values of the list?
my_list[0]='NEW'


# In[ ]:


my_list


# In[ ]:


#what about the strings???
s='hell'


# In[ ]:


s[0]='n'  #'str' object does not support item assignment


# WE can nest nest list inside of one another
# 

# In[ ]:


nest=[1,2,[3,4]]


# In[ ]:


nest


# In[ ]:


nest[2]


# In[ ]:


nest[2][1]


# In[ ]:


nest=[1,2,3,[4,5,['target']]]


# In[ ]:


nest[3]


# In[ ]:


nest[3][2]


# In[ ]:


print(nest[3][2][0])


# **Dictionaries : Uses the curly brackets  {'key1': 'value'}**

# In[ ]:


dic = {'key1': 'value','key2':'123'}


# In[ ]:


dic['key1']


# In[ ]:


dic['key2']


# In[ ]:


#dictionary can take in any items as their values
d={'k1':[1,2,3]}  #1,2,3 are items


# In[ ]:


my_list=d['k1']  #for any strings dont forget to put apostphes 


# In[ ]:


my_list


# In[ ]:


my_list[0]


# In[ ]:


d['k1'][1]


# **Nested Dictionary**

# In[ ]:


d={'k1':{'innerkey':[1,2,3]}}


# In[ ]:


d['k1']


# In[ ]:


my_dic=d['k1']['innerkey']


# In[ ]:


my_dic


# In[ ]:


my_dic[2]


# **Boolean**

# In[ ]:


True


# In[ ]:


False


# **Tuples**

# In[ ]:


my_list=[1,2,3]


# In[ ]:


my_list[1]  #tuples are similar to list
#but instead of squre brackets it uses the parenthesis


# In[ ]:


tuples=(1,2,3)


# In[ ]:


tuples[1]


# **Then whats the difference between tuples and list???**

# In[ ]:


my_list=[1,2,3]


# In[ ]:


my_list[0] ='new'


# In[ ]:


my_list


# In[ ]:


tuples


# In[ ]:


tuples[0]='new'  #'tuple' object does not support item assignment


# **I will  use tuple when I want a user cant change the value of the list.**
# **Tuple is immutable and list is mutable**

# **Set is a collection of unique elements. Uses the curly brackets like  dictionaries.**

# In[ ]:


{1,2,3}


# In[ ]:


{1,1,1,1,2,2,2,2,2,3,3,3,3,3,3}   #set is defined by only unique elements


# We can also call the set function.   set()
# whereas we can select the unique elements from the list

# In[ ]:


set([1,1,1,1,2,2,2,2,2,3,3,3,3])


# How to add elements to set???

# In[ ]:


s={1,2,3}


# In[ ]:


s.add(4)


# In[ ]:


s


# **Comparison Operators**

# In[ ]:


1>2


# In[ ]:


1<2


# In[ ]:


1>=2


# In[ ]:


1<=2


# In[ ]:


1==1


# In[ ]:


1!=3


# **We can do with the strings also**

# In[ ]:


'hi'=='bye'


# **Logic Operators**

# In[ ]:


(1<2) and (2<3)


# In[ ]:


(1<2) and (2>3)


# In[ ]:


(1<2) or (2>3) or (1==1)


# In[ ]:


True and True


# In[ ]:


True and False


# In[ ]:


True or False


# **if,elif, else Statements**

# In[ ]:


if (1<=2):
    print('yep')


# In[ ]:


if True:
    print('perform code')


# In[ ]:


if True:
    x=2+2


# In[ ]:


x


# In[ ]:


if (1==3):
    print('first')
else:
    print('Last')


# In[ ]:


if (1 != 2):
    print('First')
else:
    print('last')


# In[ ]:


# now for multiple condtions
if (1 == 2):
    print('First')
elif(4==4):
    print('Second')
    
#only executes the first true conditions
elif(3==3):
    print('Middle')
else:
    print('last')


# **For Loops**

# In[ ]:


item=[1,2,3,4,5,6]
#for item in item:
   # print(item)


# In[ ]:


#in place of items there can be anything like
#for jelly in (item):
     #print(jelly)
#int object dosent support iterations


# In[ ]:


#In the case of item example the appropriate word after for must be num
#item=[1,2,3,4,5,6]
#for num in item:
   # print (num)


# In[ ]:


#we can print anything other than the elements inside the list
item=[1,2,3,4,5,6]
for num in item:
    print('hello')


# In[ ]:


seq = [1,2,3,4,5]
for jelly in seq:
    print(jelly+jelly)


# **While loops**

# In[ ]:


i=1
while i<5:
    print('i is {}'.format(i))
    i=i+1


# **range()**

# In[ ]:


range(0,5) #ITS A GENERATOR


# In[ ]:


for x in range(0,5):
    print(x)


# In[ ]:


list(range(0,5))


# In[ ]:


list(range(10))


# **list comprehension**

# In[ ]:


x=[1,2,3,4]
out=[]
for num in x:
   out.append(num**2) 
out


# In[ ]:


[num**2 for num in x]


# In[ ]:


out=[num**2 for num in x]


# In[ ]:


out


# **functions**

# In[ ]:


# keyword for function is def
def my_func(param1):
    print(param1)


# In[ ]:


my_func('hello')


# In[ ]:


def my_fun(name='default name'):
    print('Hello '+name)


# In[ ]:


my_fun('Sourav')


# In[ ]:


my_fun()


# In[ ]:


my_fun(name='Sourav')


# In[ ]:


my_fun


# In[ ]:


def squre(num):
    return num**2


# In[ ]:


out=squre(2)
out


# **Documantation String**

# In[ ]:


def squre(num):
    """
    THIS IS A DOCUMENTATION STRING
    
    CAN GO MULTIPLE LINES
    
    THIS FUNCTION REQUIRES A NUMBER
    
    """
    return num**2


# In[ ]:


squre(4)


# In[ ]:


squre

