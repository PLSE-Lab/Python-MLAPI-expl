#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# integer vs string 
str1 = 'this is a string'
int1 = 23

print('int1: ', int1)
print('my string: ', str1)


# **1. FUNCTIONS**

# In[ ]:


# simple funtion: expected salary 

def expected_salary(age, height):
    if age > 22:
        salary = 15 * height
    else: 
        salary = 10 * height
    return salary


james_salary = expected_salary(25, 200)
georges_salary = expected_salary(18, 150)

print('james_salary: {} euro'.format(james_salary))
print('georges_salary: {} euro'.format(georges_salary))


# **2. CLASSES**
# 
# - Lets say we want to create a database of our employees. To keep theis name, age and adress values I need that much of variables. But, what if I have thousands of employees in my company?
# 
# - objects in classes have attributes and behaviours (methods)

# In[ ]:


# just for 1 employee I need 3 different valiables!
employee1_name = 'messi'
employee1_age = 30
employee1_adress = 'barcelona'

# instead I will create a Class that contains similars objects


# In[ ]:


class Employee_Class:
    # attributes = age, name, adress
    # behaviours = pass 
    pass # right now class is empty, thats why we use pass command

employee1 = Employee_Class() # right now it is empty


# In[ ]:


employee1


# **2.1 Atributes**

# In[ ]:


class Football_player:
    football_club = 'fenerbahce'
    age = 30

alex = Football_player() 


# In[ ]:


alex # it is an object of Football_player class


# In[ ]:


alex.age


# In[ ]:


alex.football_club


# In[ ]:


# We will have an error here
# classes take objects not variables as an input
# (funstions take variables as an input)

class Football_player(age, football_club):
    football_club = 'fenerbahce'
    age = 30

semih = Football_player() 


# **Transfer season is opened**

# In[ ]:


alex.football_club = 'galatasaray'

alex.football_club


# **2.2 Methods**

# In[ ]:


class Square(object):
    
    edge = 5 # meters

s1 = Square()

print(s1)
print(s1.edge)


# In[ ]:


class Square(object):
    
    # attributes
    edge = 5 # meters
    
    # behaviours (methods)
    def calculate_area(self): # self = object (here we use self to say our method that it is our object   
        area = self.edge * self.edge
        return area 

s1 = Square() # create an object of class Square

a = s1.edge # this object has an attribute edge 

b = s1.calculate_area() # this object behaves as calculate_area

print('s1 has attribute: ', a)
print('s1 behaves as calculate_area: ', b)


# In[ ]:


s1.edge = 7 # update edge
print('new edge of s1: ', s1.edge)
print('new area: ', s1.calculate_area())


# **2.3. Methods vs Functions**

# In[ ]:


class Employee(object):
    age = 20
    salary = 1000 # in dollars
    
    # this is a method 
    def age_salary_ratio(self):
        print('age_salary_ratio of method: ', (self.age/self.salary))

# this is a function
def Age_Salary_Ratio(age, salary):
    print('age_salary_ratio of function: ', (age/salary))

empl1 = Employee()
empl1.age_salary_ratio()
Age_Salary_Ratio(30,3000)


# **2.4. Initializer Or Constructer**

# In[ ]:


class Animals(object):
    
    # we were writing attributes below manually
    # name = 'dog'
    # age = 2
    
    # __init__ is a method that will define my attributes. Previously, I was changing 
    # the attributes (age, name, salary etc) manually. Now I will create initializer for attributes
    
    def __init__(self, name, age, eat):
        self.name = name
        self.age = age
        self.eat = eat
        
animal1 = Animals('dog',2, 'meat')

print('name: ', animal1.name)
print('age: ', animal1.age)
print('eat: ', animal1.eat)


# In[ ]:


class Animals(object):
    
    # we were writing attributes below manually
    # name = 'dog'
    # age = 2
    
    # __init__ is a method that will define my attributes. Previously, I was changing 
    # the attributes (age, name, salary etc) manually. Now I will create initializer for attributes
    
    def __init__(self, a, b, c):
        self.name = a
        self.age = b
        self.eat = c
        
animal1 = Animals('dog',2, 'meat')

print('name: ', animal1.name)
print('age: ', animal1.age)
print('eat: ', animal1.eat)


# In[ ]:


animal2 = Animals('cat', 1, 'mouse')
print('{} eats {} '.format(animal2.name, animal2.eat))


# ** We see that, now I can create infinitely many objects in Animla class **

# In[ ]:


class Animals(object):
    
    # with __init__ method I have created below my attributes which are name, age and eat
    def __init__(self, name, age, eat):
        self.name = name
        self.age = age
        self.eat = eat
        
    # this is my method 
    def print_animal_attributes(self):
        print('name: {}, age: {}, eat: {}'.format(self.name, self.age, self.eat))

a1 = Animals('bird', 5, 'forage')


# In[ ]:


a1.print_animal_attributes()


# **3. Calculator Project**

# In[ ]:


# below the orange colored strings are called docstring, it is an explanation of class and methods 
# they are not compiled

class Calculator(object):
    'calculator'
    
    def __init__(self, a, b):
        'initialize values'
        self.num1 = a
        self.num2 = b
    
    def add(self):
        'addition'
        c = self.num1 + self.num2
        return c
    
    def multiply(self):
        'multiplication'
        return self.num1 * self.num2

v1 = input('type first value: ') # it returns a string
v2 = input('type second value: ') # it returns a string

val1 = int(v1)
val2 = int(v2)

# alternatively
# val1 = int(input('type first value: '))
# val2 = int(input('type second value: '))

c1 = Calculator(val1, val2)

selection = input('to add type 1, to multiply type 2: ')

if selection == '1':
    print ('addition: ', c1.add())
    
elif selection == '2':
    print('multiplication: ', c1.multiply())

else:
    print('print 1 or 2')
    
# print('addition = ', c1.add())
# print('multiplication = ', c1.multiply())


# **4. Encapsulation**

# In[ ]:


class BankAccount(object):
    def __init__(self, name, money, address):
        self.name = name
        self.money = money
        self.address = address

p1 = BankAccount('messi', 1000, 'barcelona')
p2 = BankAccount('ronaldo', 1200, 'turin')


# In[ ]:


# I can easily access to Messi`s amount of money
p1.money


# In[ ]:


# Besides, I can transfer messi`s money to ronaldo

p2.money += p1.money
p1.money = 0


# In[ ]:


print('ronaldo`s money right now: ', p2.money)
print('messi`s money right now', p1.money)


# I have easily changed money in bank accounts, which is not prefered. I have to restrict access to bank accounts. 

# In[ ]:


class BankAccount(object):
    def __init__(self, name, money, address):
        self.name = name # global variable: can be accessed from everyone
        self.__money = money # made private variable using self.__AnyAttribute 
        self.address = address

p1 = BankAccount('messi', 1000, 'barcelona')
p2 = BankAccount('ronaldo', 1200, 'turin')


# In[ ]:


print('name: ', p1.name) # global
print('messi`s money right now', p1.__money) # private, I can not see it


# **4.1 get and set**

# In[ ]:


class BankAccount(object):
    
    def __init__(self, name, money, address):
        self.name = name # global variable: can be accessed from everyone
        self.__money = money # has been made private variable using self.__AnyAttribute 
        self.address = address
     
    # define get
    def getMoney(self):
        return self.__money
    
    #define set
    def setMoney(self, amount):
        self.__money = amount

p3 = BankAccount('neymar', 1000, 'paris')
p4 = BankAccount('alex', 1200, 'istanbul')


# In[ ]:


p4.__money # no access


# In[ ]:


p4.getMoney() # now it has access using getMoney method


# In[ ]:


p4.setMoney(3000) # change money to 3000
p4.getMoney() # see money


# **4.2 but this get and set are still global, I can make a private method also using '__'**

# In[ ]:


class BankAccount(object):
    
    def __init__(self, name, money, address):
        self.name = name # global variable: can be accessed from everyone
        self.__money = money # has been made private variable using self.__AnyAttribute 
        self.address = address
     
    # define get (global method)
    def getMoney(self):
        return self.__money
    
    # define set (global method)
    def setMoney(self, amount):
        self.__money = amount
    
    # this is a private method, since we have used __
    # it can be used inside the class but not outside
    def __increaseMoney(self, a):
        self.__money += a
    
p5 = BankAccount('benzema', 2350, 'madrid')


# In[ ]:


p5.__increaseMoney(1000) # we will have an error, because the method is private now


# **5. Inheritance**
# 
# - Let say I have an 'animal' class which has a method (or an attribute) 'run'
# - Let also I have a 'monkey' class which has 'run' and 'climb' methods
# - For monkey class I dont want to create again 'run' method, so I want to use the 'run' method of 'animal' class.
# - (here we usually call animal class as 'parent' and monkey class as 'child')

# In[ ]:


# parent class
class Animal():
    
    def __init__(self):
        print('animal class is created')
        
    def toString(self):
        print('animal')
    
    def walk(self):
        print('animal walks')
        
# child class
class Monkey(Animal): # in paranthesyses I write the parent class that I want to use
    
    def __init__(self):
        super().__init__() # we use this line to use init of animal class
        print('monkey is created')
        
    def toString(self):
        print('monkey')
        
    def climb(self):
        print('monkey climbs')


# In[ ]:


m1 = Monkey() # it uses init of Animal class also


# In[ ]:


m1.toString() # method of Monkey


# In[ ]:


m1.climb() # method of Monkey


# In[ ]:


m1.walk() # method of Animal, but I can use also this method


# In[ ]:


# lets create another child class, Bird
class Bird(Animal):
    
    def __init__(self):
        super().__init__()
        print('bird is created')
    
    def fly(self):
        print('birds fly')


# In[ ]:


b1 = Bird()


# In[ ]:


b1.fly()


# In[ ]:


b1.walk()


# In[ ]:


b1.climb() # Bird class can not use methods of Monkey class


# **6. Inheritance Project**

# In[ ]:


class Website():
    'parent class'
    
    def __init__(self, name, surname):
        self.name = name
        self.surname = surname
        
    def LoginInfo(self):
        print(self.name + ' ' + self.surname)
        
class ChildWebsite(Website):
    'child class'
    
    def __init__(self, name, surname, mail_address):
        super().__init__(name, surname) # Alternatively we can use Website.__init__(self, name, surname)
        self.mail_address = mail_address
        
    def printUserInfo(self):
        print('name: ', self.name)
        print('surname: ', self.surname)
        print('mail address: ', self.mail_address)


# In[ ]:


user1 = Website('lionel', 'messi')
user1.LoginInfo()


# In[ ]:


user2 = ChildWebsite('alexander', 'pato', 'ap@hotmail.com')
user2.printUserInfo()


# In[ ]:


class GrandChildWebsite(ChildWebsite):
    'another child class, actually this is grandchild class, child of ChildWebsite which is child of Website'
    
    def __init__(self,name, surname, mail_address, ids):
        ChildWebsite.__init__(self, name, surname, mail_address) 
        self.ids = ids
        
    def printAllUserInfo(self):
        print('name: ', self.name)
        print('surname: ', self.surname)
        print('mail address: ', self.mail_address)
        print('ID: ', self.ids)


# In[ ]:


# user3 is an object of GrandChildWebsite class
user3 = GrandChildWebsite('david', 'beckam', 'db@hotmail.com', '1234')


# In[ ]:


user3.printAllUserInfo() # this method is from GrandChildWebsite class


# In[ ]:


user3.printUserInfo() # this method is from ChildWebsite class


# In[ ]:


user3.LoginInfo() # this method is from Website class


# In[ ]:


class Another_Child_Of_Website(Website):
        'this class is child of parent class = Website'
        
        def __init__(self, name, surname, tel):
            Website.__init__(self, name, surname)
            self.tel = tel
            
        def print_tel_number(self):
            print('tel_number: ', self.tel)


# In[ ]:


user4 = Another_Child_Of_Website('alex', 'morgan', 12345)


# In[ ]:


user4.print_tel_number() # this method is from Another_Child_Of_Website class


# In[ ]:


user4.name


# In[ ]:


user4.surname


# In[ ]:


user4.ids # Error: because, ids is GrandChildWebsite`s attribute

# user4 has attributes only from Another_Child_Of_Website and Website 


# **7. Abstract Classes**

# - Abstract Classes in Python: An abstract class can be considered as a blueprint for other classes, allows you to create a set of methods that must be created within any child classes built from your abstract class. A class which contains one or abstract methods is called an abstract class
# 
# -  super class can be a abstract class by using at least one abstract method
# - when we create sub class from this super class, the sub class must have the abstract method defined in super class

# In[ ]:


from abc import ABC, abstractmethod  # abc: abstract base class

# this class will build my template
class Animal(ABC):  # super class or parent class
    
    @abstractmethod
    def walk(self): pass  # here the walk method is an abstract method
    
    def run(self): 
        print('this animal can run')
    
    
class Bird(Animal): # sub class or child class
    
    def __init__(self, name):
        self.name = name
        print('{} is a bird'.format(self.name))
        
    def walk(self): # we have to use this method, if we want to create a sub class of super class
        print('this bird can walk')


# In[ ]:


a1 = Animal()


# In[ ]:


a2 = Bird('eagle')


# In[ ]:


a2.walk()


# In[ ]:


a2.run()


# **I will have an error here because?**

# In[ ]:


from abc import ABC, abstractmethod  # abc: abstract base class

# this class will build my template
class Animal(ABC):  # super class or parent class
    
    @abstractmethod
    def walk(self): pass  # here the walk method is an abstract method
    
    def run(self): 
        print('this animal can run')
    
    
class Bird(Animal): # sub class or child class
    
    def __init__(self, name):
        self.name = name
        print('{} is a bird'.format(self.name))


# In[ ]:


a3 = Bird('canary')


# I could not created canary animal becasue in Bird class I dont have abstract method walk

# **8. Overriding**

# - when I have the same methods both in super class and sub class which one will be called?

# In[ ]:


class Animal():
    
    def toString(self):
        print('animal')
        
class Monkey(Animal):
    
    def toString(self):
        print('monkey')


# In[ ]:


animal1 = Animal()
animal1.toString()


# In[ ]:


monkey1 = Monkey()
monkey1.toString()


# **9. Polymorphism**

# - In literal sense, Polymorphism means the ability to take various forms. In Python, Polymorphism allows us to define methods in the child class with the same name as defined in their parent class. As we know, a child class inherits all the methods from the parent class

# In[ ]:


class Employee():
    
    def raisee(self):
        raise_rate = 0.1
        salary = 100 + 100 * raise_rate
        print('emplyee salary: ', salary)
        
class CompEng(Employee):
    
    def raisee(self):
        raise_rate = 0.2
        salary = 100 + 100 * raise_rate
        print('CE salary: ', salary)
    

class EEE(Employee):
    
    def raisee(self):
        raise_rate = 0.3
        salary = 100 + 100 * raise_rate
        print('EEE salary: ', salary)


# In[ ]:


e = Employee()
ce = CompEng()
eee = EEE()

e.raisee()
ce.raisee()
eee.raisee()


# **10. Lets combine all topics in a project**

# In[ ]:


from abc import ABC, abstractmethod

class Shape(ABC):
    '''
    this is my abstarct class or template
    
    '''
    
    # these are examples for abstract methods
    @abstractmethod
    def area(self): pass
    
    @abstractmethod
    def perimeter(self): pass
    
    # this is an example for overriding and polymorphism   
    def toString(self): pass
    
class Square(Shape):
    ''' 
    child class of Shape
    
    '''
    
    def __init__(self,lenght):
        self.__lenght = lenght # encapsulation, private attribute
    
    def area(self):
        result = self.__lenght**2
        print('area of this square: ', result)
    
    def perimeter(self):
        result = 4 * self.__lenght
        print('perimeter of this square: ', result)
    
    def toString(self): # example for overriding and polymorphism
        print('lenght of the square: ', self.__lenght)
        
class Circle(Shape):
    ''' 
    child class of Shape
    
    '''
    pi = 3.14 # constant variable
    
    def __init__(self, radius):
        self.__radius = radius
        
    def area(self):
        result = self.pi * self.__radius**2
        print('area of the circle: ', result)
        
    def perimeter(self):
        result = 2*self.pi*self.__radius
        print('perimeter of the circle: ', result)
        
    def toString(self):
        print('radius of the circle: ', self.__radius)


# In[ ]:


c1 = Circle(5)
c1.area()
c1.perimeter()
c1.toString()


# In[ ]:


s1 = Square(6)
s1.area()
s1.perimeter()
s1.toString()


# In[ ]:





# In[ ]:





# In[ ]:




