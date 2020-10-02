#!/usr/bin/env python
# coding: utf-8

# # Dictionaries
# 
# A dictionary is a collection which is unordered, changeable and indexed. In Python dictionaries are written with curly brackets, and they have keys and values. While the values of a dict can be any Python object, 
# the keys generally have to be immutable objects like scalar types (int, float, string) or tuples (all the objects in the tuple need to be immutable, too).
# 
# Let's try and example!

# In[ ]:


#Your Code Goes Here

metal_dict = {'name': "Mustaine", 'band': "Megadeth", 'album': "Rust In Piece", 'year': 1990}

print(metal_dict)


# In[ ]:


#Print Full Dictionary

print(metal_dict)


# In[ ]:


#Printing Value of key

print(metal_dict['year'])


# In[ ]:


#Print Value of Key

print(metal_dict['band'])

print(metal_dict['band'], metal_dict['year'] )


# In[ ]:


#Adding an element

metal_dict['instrument'] = "guitar"
metal_dict['band'] = 'guitar'

print(metal_dict)


# In[ ]:


#Removing an element

#metal_dict.pop('album')

print(metal_dict)


# In[ ]:


#Storing Lists as the value of a key

metal_dict2 = {
    'name': ["Mustaine", "Friedman", "Burton"],
    'band': ["Megadeth", "Megadeth", "Metallica"],
    'instrument': ["Guitar", "Guitar", "Bass"],
    'year':1990
}

print(metal_dict2)


# In[ ]:


#Accesing elements Similar to nested lists 

print(metal_dict2['name'])

print(metal_dict2['name'][0])

print(metal_dict2['instrument'][1])


# In[ ]:


metal_dict2['genre']= {'pop': 100, 'rock': 1000}

print(metal_dict2)
print(metal_dict2['genre']['pop'])

#metal_dict2['metal_dict'] = metal_dict2

#print(metal_dict2)

#metal_dict2.pop('metal_dict')
#metal_dict2.pop('genre')

#metal_dict2.pop('genre')

#metal_dict2


# In[ ]:


#Copying a dictionary
'''
You cannot copy a dictionary simply by typing dict2 = dict1, 
because: dict2 will only be a reference to dict1, 
and changes made in dict1 will automatically also be made in dict2.
There are ways to make a copy, one way is to use the built-in Dictionary method 
copy()
'''

metal_dict3 = metal_dict2.copy()


# In[ ]:


#Looping through a dictionary

for key,value in metal_dict.items():
    print(key,value)


# In[ ]:


#Looping through a dictionary that's value is a list

for key,value in metal_dict2.items(): 
    print("{}: {}".format(key,value))


# In[ ]:


###Creating a disctionary with lists
music = {'name': ['Frankenstein', 'Waste', 'Make Light'],
         'band':['TPC', 'FTP', 'PP'],
         'album': ['Champ', 'Torches', 'Manners']
         }
###Looping though the dictionary with lists

for key,value in music.items():
    print(key,value)
#By key


#By element in list

for key,value in music.items():
    print(key)
    for element in value:
        print(element)


# In[ ]:


#Check if a key exists in a dictionary

key = 'name'

if key in music:
    print("Key Exists! Value:", music[key])
    
key2 = 'apple'

if key2 in music:
    print("In dictionary!")
else: 
    print("Not in dictionary!")
    


# # Tuples
# 
# A tuple is a collection which is ordered and unchangeable. In Python tuples are written with round brackets.
# 

# In[ ]:


#Your Code Goes Here

bands = ('Megadeth', 'Metallica', 'Slayer')
print(bands)


# In[ ]:


#Print one element

print(bands[0])
#Can't change an assignment


# In[ ]:


#Unpack them

tup = (4,5,6)

a,b,c = tup

print(a)


# # Putting It All Together
# 
# Write a Python program to sum all the values in the given dictionary! Hint you will need to use += , and a nested loop and make sure you set up the for loop correclty (look at previous examples to understand the difference between a dictionary loop and normal loops)
# 

# In[ ]:


countries = {'US': [100,40], 'Canada': [54,77], 'Mexico': [247,6,8]}

###Your Code Here

