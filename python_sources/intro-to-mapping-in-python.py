#!/usr/bin/env python
# coding: utf-8

# > # Welcome to Python
# 
# A more thorough introduction to Python programming: 
# (Some of these cells are borrowed from that introduction)
# 
# http://nbviewer.jupyter.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-1-Introduction-to-Python-Programming.ipynb
# 
# Python language reference:
# 
# https://docs.python.org/2/reference/index.html
# 
# A good long intro to Python :
# 
# http://introtopython.org/

# ### Python program files
# 
# - Python code is usually stored in text files with the file ending ".py":
# 
# 	`myprogram.py`
# 
# - Every line in a Python program file is assumed to be a Python statement, or part thereof.
#     The only exception is comment lines, which start with the character # (optionally preceded by an arbitrary number of white-space characters, i.e., tabs or spaces). Comment lines are usually ignored by the Python interpreter.
# 
# - To run our Python program from the command line we use:
# 
#   `$ python myprogram.py`
# 
# - On UNIX systems it is common to define the path to the interpreter on the first line of the program (note that this is a comment line as far as the Python interpreter is concerned):
# 
#   `#!/usr/bin/env python`
# 
# - If we do, and if we additionally set the file script to be executable, we can run the program like this:
# 
#   `$ myprogram.py`

# ### IPython notebooks
# 
# - This file - an IPython notebook - does not follow the standard pattern with Python code in a text file. Instead, an IPython notebook is stored as a file in the JSON format. The advantage is that we can mix formatted text, Python code and code output. It requires the IPython notebook server to run it though, and therefore isn't a stand-alone Python program as described above. Other than that, there is no difference between the Python code that goes into a program file or an IPython notebook.

# ## Basic programming

# ### Numbers and math

# In[ ]:


a = 10
b = 5
print(a+b)


# In[ ]:


# You can reference variables between cells
print(a/b)


# ### Strings and string stuff

# In[ ]:


h = "hello"
e = "world"
print(h + " " + e)


# In[ ]:


text = h + " " + e
print(text)
# Once defined, most variable types have methods that can be called.
# For example, string variables have .title()
print(text.title())


# ### Other variable types

# In[ ]:


# Lists
dogs = ['border collie', 'beagle', 'labrador retriever']
print(dogs)
dog = dogs[0]
print(dog.title())


# In[ ]:


# Lists and looping
for dog in dogs:
    print(dog)


# In[ ]:


# Modifying a list
dogs[0] = 'huskie'
dogs.append('basset hound')
print(dogs)
print('chihuahua' in dogs)


# In[ ]:


# Tuples: Lists that can't be changed
cats = ('russian blue', 'tortoise shell')
print(cats[1])


# In[ ]:


cats.append('tabby')


# ### Functions

# In[ ]:


# Can just perform an activity
def thank_you(name):
    print("You are doing very good work, %s!" % name)
    
thank_you('India')
thank_you('Terror')
thank_you('Disillusionment')


# In[ ]:


# Or can return a value
def add_numbers(x,y):
    return x + y
# You can use values
print(add_numbers(666, 333))
# Or variables
print(add_numbers(a,b))


# ### If statements
# Logical tests!

# In[ ]:


print(5 == 5)
print(3 == 5)
print(a == b)
print('a' == 'b')
print('Andy'.lower() == 'andy'.lower())
print(a != b)
print(a < b)


# ### Modules
# 
# Most of the functionality in Python is provided by modules. The Python Standard Library is a large collection of modules that provides cross-platform implementations of common facilities such as access to the operating system, file I/O, string management, network communication, and much more.

# In[ ]:


import this


# ### Showing a web map

# In[ ]:


# A simple map
import folium
m = folium.Map(location=[44.05, -121.3])
m


# In[ ]:


# Add a marker
m2 = folium.Map(
    location=[44.05, -121.3],
    zoom_start=12,
    tiles='Stamen Terrain'
)
folium.Marker([44.042, -121.333], popup='Here we are!').add_to(m2)
m2


# In[ ]:


# Get our geojson
import requests
import json
with open('../input/ne_110m_rivers_lake_centerlines.geojson') as geojson:
    rivers = json.load(geojson)

m3 = folium.Map(
    tiles='Mapbox Bright'
)

folium.GeoJson(
    rivers,
    name='Rivers'
).add_to(m3)

m3


# In[ ]:




