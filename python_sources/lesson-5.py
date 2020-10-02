#!/usr/bin/env python
# coding: utf-8

# In[ ]:


number = 22
for x in range(10):
    mynum = int(input('Choose your answer:'))
    if mynum == number:
        print('you are correct!')
    elif mynum < number:
        print('you guessed too little of a number.')
    else:
        print('you guessed too large of a number.')


# In[ ]:


import random

number = random.randint(1,100)
mylist = []
for x in range(10):
    mynum = int(input('Choose your answer:'))
    mylist.append(mynum)
    if mynum == number:
        print('you are correct!')
        break
    elif mynum < number:
        print('you guessed too little of a number.')
    else:
        print('you guessed too large of a number.')
print(mylist)


# In[ ]:


#Homework part 1
import random
i = number
number = random.randint(1,100)
mylist = []
while i != number:
    mynum = int(input('Choose your answer:'))
    mylist.append(mynum)
    if mynum == number:
        print('you are correct!')
        break
    elif mynum < number:
        print('you guessed too little of a number.')
    else:
        print('you guessed too large of a number.')
print(mylist)


# In[ ]:


number = 22
mylist = []
for x in range(10):
    mynum = int(input('Choose your answer:'))
    mylist.append(mylist)
    if mynum == number:
        print('you are correct!')
        break
    elif mynum < number:
        print('you guessed too little of a number.')
    else:
        print('you guessed too large of a number.')
print(mylist)


# In[ ]:


number = 22
    mynum = int(input('Choose your answer:'))
    if mynum == number:
        print('you are correct!')
    elif mynum < number:
        print('you guessed too little of a number.')
    else:
        print('you guessed too large of a number.')


# In[ ]:


number = 22
for x in range(10):
    mynum = int(input('Choose your answer: '))
    if mynum == number:
        print('you are correct!')
        break
    elif mynum < number:
        print('you guessed too little of a number.')
    else:
        print('you guessed too large of a number.')


# In[ ]:


#Homework part 2

*
**
***
****
*****


# In[ ]:


Write a Python program to construct the following pattern, using a nested for loop. you can input how many lines this will print.

* 
* * 
* * * 
* * * * 
* * * * * 


# In[ ]:


#Homework part 2
n=5;
for i in range(n):
    for j in range(i):
        print ('* ', end="")
    print('')


# In[ ]:


for i in range(n,0,-1):
    for j in range(i):
        print('* ', end="")
    print('')


# In[ ]:


print('*')
print('**')
print('***')
print('****')
print('*****')


# In[ ]:


Use for loop to print the following pattern;
Length can be modified
1*2 = 2
2*3 = 6
3*4 = 12
4*5 = 20


# In[ ]:


#Homework part 3
f=2;
for d in range(f):
    for f in range(d):
        print ('1*2 = 2')
        print ('2*3 = 6')
        print ('3*4 = 12')
        print ('4*5 = 20')

