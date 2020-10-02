#!/usr/bin/env python
# coding: utf-8

# # PYTHON CLASSES FUNDAMENTALS
# Python is an object oriented programming language.
# Almost everything in Python is an object, with its properties and methods.
# A Class is like an object constructor, or a "blueprint" for creating objects.
# 
# 

# class definitions cannot be empty, but if you for some reason have a class definition with no content, put in the
# pass statement to avoid getting an error

# In[ ]:


class Class_A:
    pass


# All classes have a function called __init__(), which is always executed when the class is being initiated.
# Use the __init__() function to assign values to object properties, or other operations that are necessary to do when the object is being created.
# The __init__() function is called automatically every time the class is being used to create a new object.
# 
# The self parameter is a reference to the current instance of the class, and is used to access variables that belongs to the class.
# It does not have to be named self , you can call it whatever you like, but it has to be the first parameter of any function in the class.

# In[ ]:


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # Objects can also contain methods. Methods in objects are functions that belong to the object.
    def my_method(self):
        print("Hello my name is " + self.name)


# In[ ]:


p1 = Person("John", 36)

print(p1.name)
print(p1.age)
p1.my_method()

# You can modify properties on objects
p1.age = 40

# You can delete properties of objects or objects themselves by using the del keyword
del p1.age
del p1


# To create a class that inherits the functionality from another class, send the parent class as a parameter when creating the child class

# In[ ]:


class Student(Person):
    pass


x = Student("Mike", "Olsen")
x.my_method()


# When you add the __init__() function, the child class will no longer inherit the parent's __init__() function.
# Note: The child's __init__() function overrides the inheritance of the parent's __init__() function.
# To keep the inheritance of the parent's __init__() function, add a call to the parent's __init__() function

# In[ ]:


class Student(Person):
    def __init__(self, fname, lname):
        Person.__init__(self, fname, lname)


# Python also has a super() function that will make the child class inherit all the methods and properties from its parent

# In[ ]:


class Student(Person):
    def __init__(self, fname, lname):
        super().__init__(fname, lname)
        self.student_property = 2019


# An object can only call public methods
# to define a private method prefix the member name with double underscore __

# In[ ]:


class Base:

    # Declaring public method
    def fun(self):
        print("Public method")

        # Declaring private method

    def __fun(self):
        print("Private method")

    # Creating a derived class


class Derived(Base):
    def __init__(self):
        # Calling constructor of
        # Base class
        Base.__init__(self)

    def call_public(self):
        # Calling public method of base class
        print("\nInside derived class")
        self.fun()

    def call_private(self):
        # Calling private method of base class
        self.__fun()


# In[ ]:


obj1 = Base()

# Calling public method
obj1.fun()

obj2 = Derived()
obj2.call_public()

# Uncommenting obj1.__fun() will
# raise an AttributeError

# Uncommenting obj2.call_private()
# will also raise an AttributeError


# private methods can be accessed by calling the private methods via public methods.

# In[ ]:


class A:

    # Declaring public method 
    def fun(self):
        print("Public method")

        # Declaring private method

    def __fun(self):
        print("Private method")

        # Calling private method via

    # another method
    def Help(self):
        self.fun()
        self.__fun()

    # Driver's code


obj = A()
obj.Help()

