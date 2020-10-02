#!/usr/bin/env python
# coding: utf-8

# # Basic Arithmetic

# In[ ]:


1+1


# In[ ]:


2-1


# In[ ]:


2*4


# In[ ]:


2**30


# In[ ]:


2*2*2*2


# In[ ]:


8/2


# In[ ]:


9/2


# In[ ]:


9//2


# In[ ]:


9%2


# 9 =  4 * 2 + 1

# In[ ]:


type(8)


# In[ ]:


type(9/2)


# # Variables assignment

# In[ ]:


a=55


# In[ ]:


a


# In[ ]:


a/2


# In[ ]:


A/2


# In[ ]:


b = a/2


# In[ ]:


b


# In[ ]:


a
b


# In[ ]:


type(a)


# In[ ]:


type(b)


# In[ ]:


a + b


# In[ ]:


type(a+b)


# In[ ]:


h = 1


# In[ ]:


h=2


# In[ ]:


h


# In[ ]:


del h


# In[ ]:


h


# # Strings and print

# In[ ]:


print(a)
print(b)


# In[ ]:


print(a,b)


# In[ ]:


print("a")


# In[ ]:


type("a")


# In[ ]:


print(1)


# In[ ]:


print("1")


# In[ ]:


type(1)


# In[ ]:


type("1")


# In[ ]:


print("1"+b)


# In[ ]:


print("1" + str(b))


# In[ ]:


print(int("1") + b)


# In[ ]:


name = "Charles"


# In[ ]:


type(name)


# In[ ]:


len(name)


# In[ ]:


name[0]


# In[ ]:


name[3]


# # List

# In[ ]:


myList = [90, 11, 23, 89897, 675]


# In[ ]:


type(myList)


# In[ ]:


myList[0]


# In[ ]:


len(myList)


# In[ ]:


myList[5]


# In[ ]:


myList2 = ["hello", 3, 6.5]


# In[ ]:


myList2


# In[ ]:


myList2[-1]


# In[ ]:


myList2[-2]


# In[ ]:


myList2[-3]


# In[ ]:


l1 = myList2.append(myList)
print(l1)


# In[ ]:


myList2


# In[ ]:


myList.append("test")
myList


# In[ ]:


myList2


# In[ ]:


l1 = ["a","b","a","c","2","d","r","g","dd","2"]


# In[ ]:


l2 = l1


# In[ ]:


l2.remove("a")


# In[ ]:


l2


# In[ ]:


del l2[3]


# In[ ]:


l2


# In[ ]:


l2[5]


# In[ ]:


l2[5] ="change"


# In[ ]:


l2


# In[ ]:


l1  #!!


# In[ ]:


l3 = l1.copy()


# In[ ]:


l3.append("will it work?")


# In[ ]:


l1


# In[ ]:


l3


# In[ ]:


l3.insert(2,33)


# In[ ]:


l3


# In[ ]:


l3.extend([5,6])
l3


# In[ ]:


l3.append([5,6])


# In[ ]:


print(l3)


# ## List Slicing

# In[ ]:


bigList = list(range(100))


# In[ ]:


print(bigList) #show without print too, one number per line


# In[ ]:


len(bigList)


# In[ ]:


bigList[-1]


# In[ ]:


bigList[90:] #90 inclusive


# In[ ]:


bigList[:10] #10 not inclusive


# In[ ]:


bigList[15:22]


#  #### Step

# In[ ]:


bigList[15:22:2] #2 is the step


# In[ ]:


bigList[::7]


# #### Negative step

# In[ ]:


print(bigList[::-1])


# In[ ]:


print(bigList[90:70:-2])#from, to! 90 first 


# In[ ]:


print(bigList[70:90:-2])


# #### copy of slice

# In[ ]:


test = bigList[5:10]


# In[ ]:


test[2]


# In[ ]:


test[2] = 88


# In[ ]:


test


# In[ ]:


print(bigList[5:10])


# #### exercise: 
# 
# give back all the multiple of 5 from 20 to 70 included, backward, in one line.

# In[ ]:





# In[ ]:





# In[ ]:


letterList = ["A","B","C","D","E","F","G","H","I","J","K"]


# In[ ]:





# # Tuples

# In[ ]:


myTuple = (9898, 7)


# In[ ]:


myTuple[0]


# In[ ]:


myTuple[0]=1


# In[ ]:


emptyTuple=()


# In[ ]:


type(emptyTuple)


# In[ ]:


print(a, b)


# In[ ]:


a, b = b, a


# In[ ]:


print(a,b)


# In[ ]:


tuple1 = 1,


# In[ ]:


type(tuple1)


# In[ ]:


tuple1


# In[ ]:


tuple2 = (1,)


# In[ ]:


tuple2


# # Python Dictionary

# In[ ]:


dico = {"key1" : "value1"}


# In[ ]:


dico


# In[ ]:


dico2 = {"name" : "Billy", "age": 43, "job" : "researcher"}


# In[ ]:


dico2


# In[ ]:


dico2["age"] 


# In[ ]:


dico2["age"]=5


# In[ ]:


dico2


# In[ ]:


dico2["adress"] = "55 water street"


# In[ ]:


dico2


# In[ ]:


dico2.pop("age")
dico2


# In[ ]:


del dico2["adress"]


# In[ ]:


dico2


# #### tuple as key

# In[ ]:


boardgame = {}


# In[ ]:


boardgame["a",1] = "white tower"


# In[ ]:


boardgame


# In[ ]:





# # Python Set

# In[ ]:


a = "abracadabra"
b = "alacazam"


# In[ ]:


a=set(a)
b=set(b)


# In[ ]:


print(a)
print(b)


# In[ ]:


print(a - b)


# In[ ]:


l4=[1,2,7,2,1,9,2,4,9]
l4


# In[ ]:


set(l4)


# In[ ]:


print(list(set(l4)))


# # Booleans

# In[ ]:


True


# In[ ]:


False


# In[ ]:


type(True)


# In[ ]:


a = True
a


# # Project / Exercise
# using only what we have seen. Create the list ["S","&","P"," ","G","l","o","b","a","l"] from the list given below
# 
# check the video for the answer!

# In[ ]:


spglobalToBe = ['o', '&', 'l', 'a', 'b', 'o', 'l', 'G', ' ', 'P', '&', 'S', 'u', 'K']


# In[ ]:





# In[ ]:





# # Readability

# In[ ]:


spglobalToBe = ['o', '&', 'l', 'a', 'b', 'o', 'l', 'G', ' ', 'P', '&', 'S', 'u', 'K']
start = len(spglobalToBe)-3
spglobalToBe = spglobalToBe[start:1:-1]


# In[ ]:


spglobalToBe

