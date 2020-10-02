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

metal_dict = {'name': 'Mustaine',
              'band': 'Megadeth',
              'album': 'Rust In Piece',
            'year': 1990}


# In[ ]:


print(metal_dict)


# In[ ]:


#Printing one key
print(metal_dict['year'])


# In[ ]:


#Print Key
print(metal_dict['band'])


# In[ ]:


#Adding an element

metal_dict['instrument'] = 'guitar'

print(metal_dict)


# In[ ]:


#Removing an element

metal_dict.pop('album')

print(metal_dict)


# In[ ]:


#Storing Lists as the value of a key

metal_dict2 = {
    'name': ['Mustaine', 'Friedman', 'Burton'],
    'band': ['Megadeth', 'Megadeth', 'Metallica'],
    'instrument': ['Guitar', 'Guitar', 'Bass']
}

print(metal_dict2)


# In[ ]:


#Accesing elements

metal_dict2['name'][0]


# In[ ]:


#Copying a dictionary
'''
You cannot copy a dictionary simply by typing dict2 = dict1, 
because: dict2 will only be a reference to dict1, 
and changes made in dict1 will automatically also be made in dict2.
There are ways to make a copy, one way is to use the built-in Dictionary method 
copy()
'''

metal_dict2_copy = metal_dict2.copy()


# In[ ]:


#Looping through a dictionary

for key, value in metal_dict.items():
    print(key, value)


# In[ ]:


#Looping through a dictionary that's value is a list

for key, value in metal_dict2.items():
    print(key, value)


# In[ ]:


###Creating a disctionary with lists
music = {'name': ['Frankenstein', 'Waste', 'Make Light'],
         'band':['TPC', 'FTP', 'PP'],
         'album': ['Champ', 'Torches', 'Manners']
         }
###Looping though the dictionary with lists

#By key
for key, value in music.items():
        print(key)
        
        
#By value       
for key, value in music.items():
    for element in value:
        print(value)
        
#By element in list
for key, value in music.items():
    for element in value:
        print(element)
        


# In[ ]:


#Check if a key exists in a dictionary
key = 'name'

if key in music:
    print(music[key])


# In[ ]:


###Exercise 1 - Write a Python program to check if a key already exists in a 
#given dictionary

d = {'1': 100, '2': 200, '3': 300, '4': 400, '5':500, '6':600}

key = input('Type the key: ')

if key in d:
    print('Key is present')
else:
    print('Key is not present')


# In[ ]:


###Exercise 2 - Write a Python program to sum all items in a dictionary

countries = {'US': 100, 'Canada': 54, 'Mexico': 247}
total = 0
for country in countries
    total += country
    
print

