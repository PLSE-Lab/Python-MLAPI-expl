#!/usr/bin/env python
# coding: utf-8

# # 03 Fundamentals

# - Conditions 
# - Branching
# - Loops
# - Functions
# - Objects 
# - Classes

# # Conditions 

# In[ ]:


a = 6
a


# In[ ]:


a == 7


# In[ ]:


a == 6


# In[ ]:


i = 101
i > 100


# In[ ]:


i = 99
i > 100


# In[ ]:


"AC/DC" == "Micheal Jackson"


# In[ ]:


"AC/DC" != "Micheal Jackson"


# ## Branching

# In[ ]:


age=17
if age>=18:
    print("Enter")
print("Move")


# In[ ]:


age=17
if age>=18:
    print("Enter")
else:
    print("Meat Loaf")
print("Move")


# In[ ]:


age=18
if age>18:
    print("Enter")
elif age==18:
    print("Pink Floyd")
else:
    print("Meat Loaf")
print("Move")


# ## Logic Operators

# In[ ]:


not (True)


# In[ ]:


not (False)


# In[ ]:


A = False
B = True
A or B


# In[ ]:


A = False
B = False
A or B


# In[ ]:


album_year = 1990
if (album_year<1980) or (album_year>1989):
    print("This 70's or 90's")
else:
    print("This 80's")


# In[ ]:


A = False
B = True
A and B


# In[ ]:


A = True
B = True
A and B


# In[ ]:


album_year = 1983
if (album_year>1979) and (album_year<1990):
    print("This 80's")


# # Loops

# In[ ]:


range(3)


# In[ ]:


list(range(3))


# In[ ]:


list(range(10,15))


# ## for loops

# In[ ]:


squares_indexes = range(5)
list(squares_indexes)


# In[ ]:


squares = ['red','yellow','green','purple','blue']
squares


# In[ ]:


print(f'Before squares {squares}')

for i in range(5):
    
    print(f'Before square {i} is {squares[i]}')
    
    squares[i]="white"
    
    print(f'After square {i} is {squares[i]}')

    print(f'After squares {squares}')    


# In[ ]:


squares = ['red','yellow','green']
squares


# In[ ]:


for square in squares:
    print(square)


# In[ ]:


for i,square in enumerate(squares):
    print(f'index {i},square {square}')


# ## while loop

# In[ ]:


squares = ['orange','orange','purple','blue']
squares


# In[ ]:


newsquares =[]
newsquares


# In[ ]:


i=0
i


# In[ ]:


while squares[i]=='orange':
    newsquares.append(squares[i])
    i+=1
newsquares


# # Functions

# In[ ]:


def function(a):
    """add 1 to a"""
    b = a + 1
    print(f'a + 1 = {b}')
    return b


# In[ ]:


function(3)


# In[ ]:


def f1(input):
    """add 1 to input"""
    output=input+1
    return output


# In[ ]:


def f2(input):
    """add 2 to input"""
    output=input+2
    return output


# In[ ]:


f1(1)
f2(f1(1))
f2(f2(f1(1)))
f1(f2(f2(f1(1))))


# ## Built-in Functions

# In[ ]:


album_ratings = [10.0,8.5,9.5]
album_ratings


# In[ ]:


Length=len(album_ratings)
Length


# In[ ]:


Sum=sum(album_ratings)
Sum


# ## Sorted vs Sort

# In[ ]:


print(f'Before album_ratings {album_ratings}')
sorted_album_ratings=sorted(album_ratings)
print(f'sorted_album_ratings {sorted_album_ratings}')
print(f'After album_ratings {album_ratings}')


# In[ ]:


print(f'Before album_ratings {album_ratings}')
album_ratings.sort()
print(f'After album_ratings {album_ratings}')


# ## Making Functions

# In[ ]:


def add1(a):
    """
    add 1 to a
    """
    b=a+1
    return b


# In[ ]:


help(add1)


# In[ ]:


add1(5)


# In[ ]:


c=add1(10)
c


# In[ ]:


def Mult(a,b):
    c=a*b
    return c


# In[ ]:


Mult(2,3)


# In[ ]:


Mult(2,'Micheal Jackson ')


# In[ ]:


def MJ():
    print('Micheal Jackson')


# In[ ]:


MJ()


# In[ ]:


def NoWork():
    pass


# In[ ]:


NoWork()


# In[ ]:


print(NoWork())


# In[ ]:


def NoWork():
    pass
    return None


# In[ ]:


NoWork()


# In[ ]:


print(NoWork())


# In[ ]:


def add1(a):
    b=a+1
    print(f'{a} plus 1 equals {b}')
    return b


# In[ ]:


add1(2)


# In[ ]:


def printStuff(Stuff):
    for i,s in enumerate(Stuff):
        print(f'Album {i} Rating is {s}')


# In[ ]:


album_ratings


# In[ ]:


printStuff(album_ratings)


# ## Collecting arguments

# In[ ]:


def ArtistNames(*names):
    for name in names:
        print(f'Name {name}')


# In[ ]:


ArtistNames("Micheal Jackson","AC/DC","Pink Floyd")


# In[ ]:


ArtistNames("Micheal Jackson","AC/DC")


# ## Scope

# ### Global Scope

# In[ ]:


def AddDC(y):
    x =y+"DC"
    print(f'Local x {x}')
    return x
x="AC"
print(f'Global x {x}')
z=AddDC(x)
print(f'Global z {x}')


# ### Local Variables

# In[ ]:


def Thriller():
    Date=1982
    return Date
Thriller()


# In[ ]:


# Date
# NameError: name 'Date' is not defined


# In[ ]:


Date = 2017


# In[ ]:


print(Thriller())


# In[ ]:


print(Date)


# In[ ]:


def ACDC(y):
    print(f'Rating {Rating}')
    return Rating+y


# In[ ]:


Rating=9
Rating


# In[ ]:


z=ACDC(1)
print(f'z {z}')


# In[ ]:


print(f'Rating {Rating}')


# In[ ]:


def PinkFloyd():
    global ClaimedSales
    ClaimedSales = '45 million'
    return ClaimedSales


# In[ ]:


PinkFloyd()


# In[ ]:


print(f'ClaimedSales {ClaimedSales}')


# In[ ]:


def type_of_album(artist, album, year_released):
    
    print(artist, album, year_released)
    if year_released > 1980:
        return "Modern"
    else:
        return "Oldie"
    
x = type_of_album("Michael Jackson", "Thriller", 1980)
print(x)


# # Objects  

# In[ ]:


type([1,34,3])


# In[ ]:


type(1)


# In[ ]:


type("yellow")


# In[ ]:


type({"dog":1,"cat":2})


# In[ ]:


Ratings = [10,9,6,5]
Ratings


# In[ ]:


Ratings.sort()
Ratings


# In[ ]:


Ratings.reverse()
Ratings


# # Classes

# ## Define Classes

# In[ ]:


# Class Circle
# Data Attributes radius,color

class Circle(object):
    pass


# In[ ]:


# Object 1: instance of type Circle

# Data Attributes 
# radius=4
# color='red'


# In[ ]:


# Object 2: instance of type Circle

# Data Attributes 
# radius=2
# color='green'


# In[ ]:


# Class Rectangle
# Data Attributes width,height,color

class Rectangle(object):
    pass


# In[ ]:


# Object 1: instance of type Rectangle

# Data Attributes 
# widrh=2
# height=2
# color='blue'


# In[ ]:


# Object 2: instance of type Rectangle

# Data Attributes 
# widrh=3
# height=1
# color='yellow'


# In[ ]:


class Circle(object):
    def __init__(self,radius,color):
        self.radius = radius
        self.color = color


# In[ ]:


class Rectangle(object):
    def __init__(self,height,width,color):
        self.height = height
        self.width = width
        self.color = color


# In[ ]:


RedCircle = Circle(10,"red")
print(f'RedCircle radius {RedCircle.radius}')
print(f'RedCircle color {RedCircle.color}')


# In[ ]:


C1 = Circle(10,"blue")
print(f'C1 radius {C1.radius}')
print(f'C1 color {C1.color}')


# In[ ]:


C1.color = 'yellow'
print(f'C1 radius {C1.radius}')
print(f'C1 color {C1.color}')


# In[ ]:


C1.radius = 25
C1.color = 'green'

print(f'C1 radius {C1.radius}')
print(f'C1 color {C1.color}')


# ## Methods

# In[ ]:


# Method add_radius to change Circle size


# In[ ]:


class Circle(object):
    def __init__(self,radius,color):
        self.radius = radius
        self.color = color
        
    def add_radius(self,r):
        self.radius = self.radius + r
        return self.radius
    
    def change_color(self,c):
        self.color = c
        return self.color
    
    def draw_circle():
        pass


# In[ ]:


C1=Circle(2,'red')
print(f'C1 radius {C1.radius}')
print(f'C1 color {C1.color}')


# In[ ]:


C1.add_radius(8)
print(f'C1 radius {C1.radius}')
print(f'C1 color {C1.color}')


# In[ ]:


C1.change_color('blue')
print(f'C1 radius {C1.radius}')
print(f'C1 color {C1.color}')


# In[ ]:


dir(Circle)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


class Circle(object):
    def __init__(self,radius,color):
        self.radius = radius
        self.color = color
        
    def add_radius(self,r):
        self.radius = self.radius + r
        return self.radius
    
    def change_color(self,c):
        self.color = c
        return self.color
    
    def draw_circle(self):
        plt.gca().add_patch(plt.Circle((0, 0), radius=self.radius, fc=self.color))
        plt.axis('scaled')
        plt.show() 


# In[ ]:


RedCircle = Circle(1,'red')

print(f'RedCircle radius {RedCircle.radius}')
print(f'RedCircle color {RedCircle.color}')


# In[ ]:


RedCircle.draw_circle()


# In[ ]:


# Create a new Rectangle class for creating a rectangle object

class Rectangle(object):
    
    # Constructor
    def __init__(self, width=2, height=3, color='r'):
        self.height = height 
        self.width = width
        self.color = color
    
    # Method
    def draw_rectangle(self):
        plt.gca().add_patch(plt.Rectangle((0, 0), self.width, self.height ,fc=self.color))
        plt.axis('scaled')
        plt.show()


# In[ ]:


SkinnyBlueRectangle = Rectangle(2, 10, 'blue')
print(f'SkinnyBlueRectangle height {SkinnyBlueRectangle.height}')
print(f'SkinnyBlueRectangle width {SkinnyBlueRectangle.width}')
print(f'SkinnyBlueRectangle color {SkinnyBlueRectangle.color}')


# In[ ]:


SkinnyBlueRectangle.draw_rectangle()

