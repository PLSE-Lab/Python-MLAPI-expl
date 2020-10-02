#!/usr/bin/env python
# coding: utf-8

# ## Python Data Structures
# 
# - Boolean
# - Boolean and Logical Operators
# - Lists
# - Comparison operators
# - Dictionaries
# - Tuples
# - Sets
# 

# ### Boolean Variables
# 
# Boolean values are the two constant objects False and True. 
# 
# They are used to represent truth values (other values can also be considered
# false or true). 
# 
# In numeric contexts (for example, when used as the argument to an
# arithmetic operator), they behave like the integers 0 and 1, respectively.
# 
# The built-in function bool() can be used to cast any value to a Boolean,
# if the value can be interpreted as a truth value
# 
# They are written as False and True, respectively.
# 
# 

# ### If this Kernel help you in any way, some UPVOTES would be very much appreciated.

# In[ ]:


#print statment is used to print any value in python
print(True,False)


# In[ ]:


#type is used to print the data type of the arguments
type(True)


# In[ ]:


type(False)


# In[ ]:


#declaring a string variable my_str with value AMit Kumar
my_str='Amit Kumar'


# In[ ]:


#The istitle() returns True if the string is a titlecased string otherwise it returns False.
#titlecased:- String which has the first character in each word Uppercase and remaining all characters Lowercase alphabets.
my_str.istitle()


# In[ ]:


print(my_str.isalnum()) #check if all char are numbers
print(my_str.isalpha()) #check if all char in the string are alphabetic
print(my_str.isdigit()) #test if string contains digits
print(my_str.istitle()) #test if string contains title words
print(my_str.isupper()) #test if string contains upper case
print(my_str.islower()) #test if string contains lower case
print(my_str.isspace()) #test if string contains spaces
print(my_str.endswith('r')) #test if string endswith a d
print(my_str.startswith('A')) #test if string startswith H


# ### Boolean and Logical Operators

# In[ ]:


True and True


# In[ ]:


True and False


# In[ ]:


True or False


# In[ ]:


True or True


# In[ ]:


str_example='Hello World'
my_str='Amit'


# In[ ]:


my_str.isalpha() or str_example.isnum()


# ### Lists
# 
# A list is a data structure in Python that is a mutable, or changeable, ordered sequence of elements. Each element or value that is inside of a list is called an item. Just as strings are defined as characters between quotes, lists are defined by having values between square brackets [ ] 

# In[ ]:


type([])


# In[ ]:


#creating a empty list
lst_example=[]


# In[ ]:


type(lst_example)


# In[ ]:


lst=list()


# In[ ]:


type(lst)


# In[ ]:


lst=['Mathematics', 'chemistry', 100, 200, 300, 204]


# In[ ]:


len(lst)


# In[ ]:


type(lst)


# ### Append

# In[ ]:


#.append is used to add elements in the list
#adding one element
lst.append("Amit")


# In[ ]:


#adding multiple elements to a list
lst.append(["John","Bala"])


# In[ ]:


lst


# In[ ]:


##Indexing in List
lst[6]


# In[ ]:


lst[1:6]


# ### Insert

# In[ ]:


## insert in a specific order

lst.insert(2,"Kumar")


# In[ ]:


lst


# In[ ]:


lst.append(["Hello","World"])


# In[ ]:


lst


# In[ ]:


lst=[1,2,3]


# In[ ]:


lst.append([4,5])


# In[ ]:


lst


# ### Extend Method

# In[ ]:


lst=[1,2,3,4,5,6]


# In[ ]:


lst.extend([8,9])


# In[ ]:


lst


# ### Various Operations that we can perform in List

# In[ ]:


lst=[1,2,3,4,5]


# In[ ]:


sum(lst)


# In[ ]:


lst*5


# ### Pop() Method

# In[ ]:


#drop the last value from the list
lst.pop()


# In[ ]:


lst


# In[ ]:


lst.pop(0)


# In[ ]:


lst


# ### count():Calculates total occurrence of given element of List

# In[ ]:


lst=[1,1,2,3,4,5]
lst.count(1)


# In[ ]:


#length:Calculates total length of List
len(lst)


# In[ ]:


# index(): Returns the index of first occurrence. Start and End index are not necessary parameters
#syntex index(element.start,end)
lst.index(2,1,5) 


# In[ ]:


##Min and Max
min(lst)


# In[ ]:


max(lst)


# ## SETS
# 
# A Set is an unordered collection data type that is iterable, mutable, and has no duplicate elements. Python's set class represents the mathematical notion of a set.This is based on a data structure known as a hash table

# In[ ]:


## Defining an empy set

set_var= set()
print(set_var)
print(type(set_var))


# In[ ]:


#creating a set
set_var={1,2,3,4,3}


# In[ ]:


set_var


# In[ ]:


set_var={"Avengers","IronMan",'Hitman','Antmman'}
print(set_var)
type(set_var)


# In[ ]:


## Inbuilt function in sets

set_var.add("Hulk")


# In[ ]:


print(set_var)


# In[ ]:


set1={"Avengers","IronMan",'Hitman'}
set2={"Avengers","IronMan",'Hitman','Hulk2'}


# In[ ]:


set2.intersection_update(set1)


# In[ ]:


set2


# In[ ]:


##Difference 
set2.difference(set1)


# In[ ]:


set2


# In[ ]:


## Difference update

set2.difference_update(set1)


# In[ ]:


print(set2)


# ## Dictionaries
# 
# A dictionary is a collection which is unordered, changeable and indexed. In Python dictionaries are written with curly brackets, and they have keys and values.

# In[ ]:


dic={}


# In[ ]:


type(dic)


# In[ ]:


type(dict())


# In[ ]:


set_ex={1,2,3,4,5}


# In[ ]:


type(set_ex)


# In[ ]:


## Let create a dictionary

my_dict={"Car1": "Audi", "Car2":"BMW","Car3":"Mercidies Benz"}


# In[ ]:


type(my_dict)


# In[ ]:


##Access the item values based on keys

my_dict['Car1']


# In[ ]:


# We can even loop throught the dictionaries keys

for x in my_dict:
    print(x)


# In[ ]:


# We can even loop throught the dictionaries values

for x in my_dict.values():
    print(x)


# In[ ]:


# We can also check both keys and values
for x in my_dict.items():
    print(x)


# In[ ]:


## Adding items in Dictionaries

my_dict['car4']='Audi 2.0'


# In[ ]:


my_dict


# In[ ]:


my_dict['Car1']='MAruti'


# In[ ]:


my_dict


# ### Nested Dictionary

# In[ ]:


car1_model={'Mercedes':1960}
car2_model={'Audi':1970}
car3_model={'Ambassador':1980}

car_type={'car1':car1_model,'car2':car2_model,'car3':car3_model}


# In[ ]:


print(car_type)


# In[ ]:


## Accessing the items in the dictionary

print(car_type['car1'])


# In[ ]:


print(car_type['car1']['Mercedes'])


# ## Tuples

# In[ ]:


## create an empty Tuples

my_tuple=tuple()


# In[ ]:


type(my_tuple)


# In[ ]:


my_tuple=()


# In[ ]:


type(my_tuple)


# In[ ]:


my_tuple=("Amit","Krish","Ankur","John")


# In[ ]:


my_tuple=('Hello','World',"Amit")


# In[ ]:


print(type(my_tuple))
print(my_tuple)


# In[ ]:


type(my_tuple)


# In[ ]:


## Inbuilt function
my_tuple.count('Amit')


# In[ ]:


my_tuple.index('Amit')


# ### If this Kernel helped you in any way, some UPVOTES would be very much appreciated.
# # Thank You :)
# # Happy Learning 

# In[ ]:




