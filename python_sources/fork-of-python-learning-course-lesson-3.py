#!/usr/bin/env python
# coding: utf-8

# ## 1. Working with lists
# ### 1.1 Getting a List of characters from a String variable
# Assign the name "Clara" into a variable.

# In[23]:


name = "Clara"
print(name)


# ### 1.1.1 list()
# Convert the name into a **list** of characters.<br>
# *Hint: list()*

# In[24]:


my_list = list(name)
print(my_list)


# ### 1.1.2 Overwriting values and .append()
# Change the characters within the list from "Clara" to "Clarify", by:
# - assigning the letter "i" in the last position of the list
# - appending "f" and "y".

# In[25]:


my_list[-1] = "i"
my_list


# In[26]:


my_list.append("f")
my_list.append("y")
print(my_list)


# ### 1.1.3 .lower() and .upper()
# Change the first letter to a lower case.<br>
# *Hint: lower()*

# In[27]:


my_list[0] = my_list[0].lower()
print(my_list)


# ### 1.1.4 .insert()
# Make a copy of the list.<br>
# Add the pronoun "I" in the begining, in order to form the phrase "Iclarify"<br>
# *Hint: .insert(element_index, new_value)*

# In[28]:


new_list = my_list.copy()
new_list.insert(0, "I")
print(new_list)


# **.insert()** adds new elements in a list, at the specified position given, pushing all elements from the specified position onwards back. If two or more elements are added on the same position, then the last element added occupies the position, with all previous elements following in the order they were added.<br><br>
# Important Note: When using the **.insert()** function, the **length** of the array expands, maintaining all previous elements.<br>
# Example: Add a space between "I" and "clarify".

# In[32]:


new_list.insert(1, " ")
print(new_list)


# ### 1.1.5 .pop() versus .remove()
# - Convert "I" to lower case.
# - Use .pop(element_index) to delete the space " ".
# - Use .remove("i") to delete all the "i" values.
# Print the list and notice the output result.

# In[33]:


new_list.pop(1)
print(new_list)


# In[34]:


new_list[0] = new_list[0].lower()
print(new_list)


# In[35]:


new_list.remove("i")
print(new_list)


# Important Note: **.remove(value)** finds and deletes **only** the first occurance of the given value in an array.

# ### 1.1.6 Checking if a value exists within an array
# When you have an value and you want to see if it exists inside an array, you can use **in**.<br>
# The output of this, is a **boolean** value.<br>
# Check if "a" exists in *my_new_list*.

# In[36]:


"a" in new_list


# ## 2. IF Statements
# IF statements in programming are used whenever we want to do something, depending on the outcome of a check (True/False).<br>
# Their general structure follows the logic:<br>
# **if** check is *True*, **then** do something, **else** do something different.<br>
# We can have many IF checks, arranged in many ways (like linear arrangement, nested arrangement).<br>
# But let's stick to some simple things for now. We'll see the other arrangements in the future.<br><br>
# Copy and paste the following code to create a list of vowels:<br>
# vowels_list = ["e", "y", "u", "i", "o", "a"]

# In[37]:


vowels_list = ["e", "y", "u", "i", "o", "a"]
print(vowels_list)


# Create a **for loop** that will check if the elements in *my_new_list* are vowels or consonants.<br>
# For every element you check, print the following phrase:<br>
# "The letter {my_letter} is a {letter_type}."

# In[38]:


for k in range(0,len(new_list)):
    if new_list[k] in vowels_list:
        my_letter = new_list[k]
        letter_type = "vowel"
    else:
        my_letter = new_list[k]
        letter_type = "consonant"
    print("The letter {one} is a {two}.".format(one=my_letter, two=letter_type))


# In[ ]:




