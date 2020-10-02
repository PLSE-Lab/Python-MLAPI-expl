#!/usr/bin/env python
# coding: utf-8

# Ref: https://www.youtube.com/watch?v=JeznW_7DlB0
# 
# https://realpython.com/lessons/how-and-when-use-__str__/
# 
# https://www.youtube.com/watch?v=aIdzBGzaxUo

# # Class in Python: Daily coding

# In[ ]:


x = 1
print(type(x))


# We get class int, int is a class actually

# In[ ]:


def hello():
    print("hello")
print(type(hello))
#hello is a function


# # Method in Python: Daily coding

# In[ ]:


string = "hello"
print(string.upper())
#upper() is a method usually name.method


# # Class template

# In[ ]:


class Dog:
    #2 underscore init two underscore
    def __init__(self, name,age):
        #pass #immidiately initiate the class
        #if we want to take a name immidiately
        self.name = name #name is attribute
        #print(name)
        self.age = age
        
    #creating a method, taking parameter self
    def bark(self):
        print("bark")
    #returning sth in method
    def meow(self):
        return "meow"
    #passing more than 1 arguement
    def add_one(self, x):
        return x + 1
    def get_name(self):
        return self.name
    def get_age(self):
        return self.age
    #modifying attributes method
    def set_age(self, age):
        self.age = age
        


# In[ ]:


d = Dog("Tim", 35)#creating Dog class
#now it has an attribute you should pass an attribute
d2 = Dog("mim", 2)
print(d2.get_name())#accessing by method
print(d2.get_age())
d2.set_age(40)
print(d2.get_age())
print(d2.name)#access the attribut by .name method
print(type(d))

d.bark()#applying method
print(d.add_one(5))


# In[ ]:


c = Dog(None)
#if i don't want to pass anything


# __main__ is telling us what module was the class  defined in
# By default it is main

# # Magic method
# Supposed to be core method in python, naming method
# 

# In[ ]:


class Car:
    def __init__(self, color, milage):
        self.color = color
        self.milage = milage
    
    def __str__(self):
        return 'a {self.color} car' .format(self = self)
    
my_car = Car("red", 37291)
print(my_car)
#__str__ method is called internally, controlling in my own way


# ## __str__

# In[ ]:


class Car:
    def __init__(self, color, milage):
        self.color = color
        self.milage = milage
    
    def __str__(self):
        return 'a {self.color} car' .format(self = self)
    
my_car = Car("red", 37291)
print(my_car)
#__str__ method is called internally, controlling in my own way


# # __repr__

# __str__ ==> easy to ready, for human consumtion
# 
# __repr__ ==> unambiguous

# In[ ]:




