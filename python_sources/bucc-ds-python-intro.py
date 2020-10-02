#!/usr/bin/env python
# coding: utf-8

# **Assigning value to a Variable in Python**
# * type not required to be specified
# 

# In[ ]:


website = "Apple.com"
print(website)
print(type(website))


# In[ ]:


a = 5
print(type(a), a)
a = 5.5
print(str( type(a) ) + " " + str(a))


# In[ ]:


a, b, c = 5, 3.2, "Hello"
print ("A:",a,"B:",b,"C:",c)


# **Collections of variables
# Four different collections List literals, Tuple literals, Dict literals, and Set literals.**
# 

# In[ ]:


fruits = ["apple", "mango", "orange"] #list
numbers = (1, 2, 3) #tuple
alphabets = {'a':'apple', 'b':'ball', 'c':'cat'} #dictionary
vowels = {'a', 'e', 'i' , 'o', 'u'} #set

print("fruits: ",fruits, "\n" "Numbers: ", numbers, "\n" 
      "Alphabets: ", alphabets, "\n" "Vowels: ", vowels)


# **IF ElSE statements**

# In[ ]:


#both works,
'''with or without brackets'''
num = 3.4

if num > 0 and num > 1:
    print("Positive number")
elif (num == 0):
    print("Zero")
elif (num==500 or num==600):
        print("just an example")
else:
    print("Negative number")


# **For loop without else statement**

# In[ ]:


numbers = [6, 5, 3, 8, 4, 2, 5, 4, 11]
sum = 0

# iterate over the list
for val in numbers:
    sum = sum+val
print("The sum is", sum)

# Program to iterate through a list using indexing
genre = ['pop', 'rock', 'jazz']

# iterate over the list using index
for i in range(len(genre)):
    print("I like", genre[i])


# In[ ]:


car = "toyota"
for i in range(0,5):
    print("car company "+ str(car))


# In[ ]:


""" 0
   012
  01234"""


# In[ ]:


row= 3
c=1
for rowCount in range (0,row):
    for space in range (0, row-rowCount):
        print(" ",end="")
    for col in range (0, c):
        print(col,end="")
    print()
    c=c+2


# In[ ]:


lst = [1,2 ]


# In[ ]:


type(lst)


# In[ ]:


lst.append(3) #adds 3 at the end 


# In[ ]:


lst.insert(2,7) #inserts 7 at index 2 


# In[ ]:


lst


# In[ ]:


lst[1]


# In[ ]:


lst[0:2] # slices from index 0 to 1 


# In[ ]:


lst[1:]# slices from index 1 to last  


# In[ ]:


lst[1:3]# slices from index 1 to 2  


# In[ ]:


lst[-1]# takes the last value 


# In[ ]:


lst[-3:-1]# slices from 3rd last to 2nd last   


# In[ ]:


lst


# In[ ]:


lst= lst[-3:-1]   #slices and updates 
lst


# In[ ]:


import numpy as np
#1 dimensional array
a = np.array([1,2,3])
print(a)
print( a[1] ) #second index


# In[ ]:


# 2 dimensional array
b = np.array([[1,2,3], [4,5,6]])
print(b)
print( b[0][1] )
print( b[0][2] )
print( b[1][2] )
print(b.shape)


# In[ ]:


#creates a numpy array with numbers from 'start' to 'end' given a step size
c = np.arange(0,12,2)
print( c )  
print(c.shape)


# In[ ]:


print( c.reshape(3,2))
print( c.reshape(3,-1)) 


# In[ ]:


x1 = np.random.randint(10, size=6)
print( x1 )


# In[ ]:


x2 = np.random.randint(10, size=(3,4))  
print( x2 )


# In[ ]:


x3 = np.random.randint(10, size=(3,4,5)) 
print( x3 )


# In[ ]:


x=np.arange(10)
print( x )
x[::3]# every other element
#mentioning the step size


# In[ ]:


print( x[2::2] )# every other element, starting at index 1
print( x[-1:-11:-1] )  # all elements, reversed
print( x[-1:-5:-2] )


# In[ ]:


np.zeros((4,3))


# In[ ]:


x=np.array([1,2,3])
y=np.array([3,2,1])
np.concatenate([x,y])


# In[ ]:


x=np.arange(10)
print( x )
p = np.where(x >= 4)
print(p)
print(np.shape(p))
p[0][0]


# In[ ]:


a = np.random.randint(10,size=20)
a[a <= 5] = 0
a[a > 5] = 1
a


# In[ ]:




