#!/usr/bin/env python
# coding: utf-8

# # Basics
# ## Variables
# * A variable is any "key" which references an object, such as a list or a string.
# 
# ### Assign

# In[ ]:


name = "Jamie"
print(name)


# ## Lists
# 
# * Lists are an ordered list of items where an item can be almost any object including, but not limited to:
#   * strings
#   * integers
#   * floating point values (aka. decimal)
#   * lists
#   * dictionaries
#   * generic objects  
# 
# ### Assign
# 
# * Lists are created using square brackets (`[]`).

# In[ ]:


people = ['Jamie', 'Kevin']
print(people)


# ### Access
# * Items are referenced by their place in the list, starting at `0`.
# * The placement value is encompassed by square brackets (`[]`).

# In[ ]:


print(people[0])


# ## Dictionaries
# 
# * Dictionaries are two-dimensional data structures which associate a "key" with a "value".
# * Dictionary values can hold almost any (any?) object which a list can hold.
# * Keys must be unique within an individual dictionary set, but a value may repeat.
# 
# ### Assign
# * Dictionaries are defined with curly brackets (`{}`) around the key/value pairs.
# * Key/value pairs are separated with a comma.

# In[ ]:


people = {'name': 'Jamie',
          'gender': 'male'}


# ### Access
# 
# * Dictionaries are accessed using the object name (aka. variable name) for the entire dictionary, followed by the key in square brackets (`[]`).

# In[ ]:


print(people)
print(people['name'])


# ## Nested Dictionaries
# 
# * Nested dictionaries are dictionaries within a dictionary.
# * The sub-dictionary is held as a value of a normal key/value pair.
# 
# ### Assign

# In[ ]:


people = {'Jamie': {'gender': 'male'},
          'Kevin': {'gender': 'male'}
         }


# ### Access
# * Nested dictionaries are referenced just like a non-nested dictionary, only nested dictionaries will refer to multiple keys.

# In[ ]:


print(people['Jamie']['gender'])


# ## List of Dictionaries
# * A list of dictionaries will contain multiple dictionaries in a specific order.
# ## Assign

# In[ ]:


people = [{'name': 'Jamie',
           'gender': 'male'},
          {'name': 'Kevin',
           'gender': 'male'}
]


# ### Access
# * A list of dictionaries is referenced just like a regular list, only a list of dictionary will refer to the list position number and  a dictionary key.

# In[ ]:





# In[ ]:


print(people[0])
print(people[0]['name'])


# ## If conditions
# * An if condition is a statement which takes actions whether the condition is true or false.
# * Most of the valid comparison operators are:
#   * `==` (equal to)
#   * `!=` (not equal to)
#   * `>` (greater than)
#   * `<` (less than)
#   * `>=` (great than or equal to)
#   * `<=` (less than or equal to)
#   * `is` (kind of equal to)
#   * `is not` (kind of not equal to)
# * If conditions support the following options:
#   * `if`
#   * `elif` (else if)
#   * `else` (if all the others aren't true)

# In[ ]:


name = 'Jamie'
if name == 'Jamie':
    print("His name is Jamie")
elif name == 'Kevin':
    print("His name is Kevin")
else:
    print("His name isn't Jamie or Kevin")


# ## For Loops
# * For loops step through an iterable object. Iterable objects include, but are not limited to:
#   * lists
#   * dictionaries
# * The first variable name in a `for` loop (ex. `person`) is the current "focus" of the loop. Actions performed within the loop normally reference this value.
#   * This variable only exists as long as the loop executes. Once it has completed, the variable is destroyed.
# * The second variable name in a `for` loop (ex. `people`) is the iterable object and is not normally referenced within the loop.

# In[ ]:


people = ['Jamie', 'Kevin']
for person in people:
    print(person)


# ## Functions
# ## Define
# * Functions are reusable "black boxes" which take optional input and should perform a single task.
# * Functions can take arguments as input and are listed, in order, in the parenthesis (`()`).
#   * The argument names (ex. `name`) are how arguments are referenced within the function.
#   * The user of the function doesn't need to match the name of the argument within their code.

# In[ ]:


def print_name(name):
    print(name)


# ### Use

# In[ ]:


print_name("Jamie")


# ## Import
# 
# * A program may need to use code which either isn't enabled by default in Python or is not included in Python.
# * An entire module can be imported using the `import` statement or a single object from a module can be imported using the `from/import` statement.
#   * If the entire module is imported, objects need to be referenced by the module name and object name.
#   * If a single object is imported, the object can be referenced by name alone.

# In[10]:


from random import randint

print(randint(0,10))

import random

print(random.randint(0,10))


# # Logical Concepts
# ## Using Data to Reference Data
# 
# Programming is all about using one piece of data to get to, or manipulate, another piece of data. It is very common for a Python programmer to do this. In this example, a program has two data structures - `admins` and `posts`. Information from both `admins` and `posts` will be used to properly display information on a web page.

# In[3]:


admins = [{'name': 'Jamie',
           'email': 'jamie@google.com',
           'admin_id': 1},
          {'name': 'Kevin',
           'email': 'kevin@google.com',
           'admin_id': 2}
         ]
posts = [{'subject': 'Welcome to my Blog',
          'post_date': '1/1/2019',
          'admin_id': 1}
        ]

for post in posts:
    print(post['subject'])
    print(post['post_date'])
    id = post['admin_id']  # Get the admin_id for the post and store the value in a variable
    name = ""
    for admin in admins:  # Loop through admins to find the admin based on id
        if admin['admin_id'] == id:
            name = admin['name']
    print("By: " + name)


# Notice the `posts` data structure has an `admin_id` reference. To get the name, this value is used to find the administrator's name in the `admins` data structure using a `for` loop and `if` statement.
