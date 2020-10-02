#!/usr/bin/env python
# coding: utf-8

# # Extra Exercises: String Manipulation, Dictionaries, and Tuples

# ## Exercise 1

# a) Each of the below tuples represents a food item along with its price. Create a list named 'grocery_list' that contains all of the tuples.

# In[ ]:


apples = ('apples', 1.99)
oranges = ('oranges', 2.49)
bananas = ('bananas', 1.49)

# Your code goes here:


# Expected result:<br>
# [('apples', 1.99), ('oranges', 2.49), ('bananas', 1.49)]

# b) Create and add a new tuple to grocery_list. This new tuple should represent 'bread', and the price is 3.99.

# In[ ]:


# Your code goes here:


# Expected result:<br>
# [('apples', 1.99), ('oranges', 2.49), ('bananas', 1.49), ('bread', 3.99)]

# c) Create a dictionary called grocery_dict from grocery_list. The keys should be the food item and the values should be the price.

# In[ ]:


# Your code goes here:


# Expected result:<br>
# {'apples': 1.99, 'oranges': 2.49, 'bananas': 1.49, 'bread': 3.99}

# d) Calculate the total cost of all the items in grocery_dict.

# In[ ]:


# Your code goes here:


# Expected result:<br>
# 9.96

# e) Count the number of food items in grocery_dict.

# In[ ]:


# Your code goes here:


# Expected result: <br>
# 4

# f) Return the name of the highest priced item in grocery_dict.

# In[ ]:


# Your code goes here:


# Expected result: <br>
# 'bread'  

# g) The price of apples has changed to 2.99. Update grocery_dict to reflect the new price.

# In[ ]:


# Your code goes here:


# Expected result:<br>
# {'apples': 2.99, 'oranges': 2.49, 'bananas': 1.49, 'bread': 3.99}

# ## Exercise 2

# a) Given the following list of integers create a dictionary that stores the min, max, and average of the numbers in the list.
# <br><br>The dictionary should look like this:<br>
# {'min': 2, 'max': 99, 'average': 49.52}
# <br><br>
# *Hint: Use the functions min(), max(), sum(), and len()
# <br> e.g. my_list = [1, 2, 3]
# <br> min(my_list) returns 1
# <br> max(my_list) returns 3
# <br> sum(my_list) returns 6
# <br> len(my_list) returns 3 (the length of the list)*

# In[ ]:


my_list = [98, 69, 3, 79, 4, 53, 2, 95, 97, 58, 24, 57, 32, 90,
           15, 29, 92, 40, 22, 99, 23, 33, 55, 38, 12, 49, 9, 42,
           7, 76, 36, 6, 86, 82, 28, 71, 30, 11, 67, 68, 65, 45,
           94, 73, 44, 81, 54, 51, 20, 62]


# In[ ]:


# First run the code in the above cell. Your code goes here:


# b) Create a dictionary that records how frequently the numbers in 'my_list' fall into the following ranges:<br>
# * Greater than or equal to 0 and less than 10 -> use the key [0, 10)<br>
# * Greater than or equal to 10 and less than 20 -> use the key [10, 20)<br>
# * Greater than or equal to 20 and less than 30 -> use the key [20, 30)<br>
# * Greater than or equal to 30 and less than 40 -> use the key [20, 40)<br>
# * .. and so on up to 100
# 
# Expected output:<br>
# {'[0, 10)': 6, 
#  '[10,20)': 3, 
#  '[20,30)': 6, 
#  '[30,40)': 5, 
#  '[40,50)': 5, 
#  '[50,60)': 6, 
#  '[60,70)': 5, 
#  '[70,80)': 4, 
#  '[80,90)': 3, 
#  '[90,100'): 7}
# 

# In[ ]:


my_list = [98, 69, 3, 79, 4, 53, 2, 95, 97, 58, 24, 57, 32, 90,
           15, 29, 92, 40, 22, 99, 23, 33, 55, 38, 12, 49, 9, 42,
           7, 76, 36, 6, 86, 82, 28, 71, 30, 11, 67, 68, 65, 45,
           94, 73, 44, 81, 54, 51, 20, 62]


# In[ ]:


# First run the code in the above cell. Your code goes here:


# ## Exercise 3 (Optional)
# The following is a more challenging exercise taken from Project Euler.

# If the numbers 1 to 5 are written out in words: one, two, three, four, five, then there are 3 + 3 + 5 + 4 + 4 = 19 letters used in total.
# 
# If all the numbers from 1 to 1000 (one thousand) inclusive were written out in words, how many letters would be used?
# 
# 
# NOTE: Do not count spaces or hyphens. For example, 342 (three hundred and forty-two) contains 23 letters and 115 (one hundred and fifteen) contains 20 letters. The use of "and" when writing out numbers is in compliance with British usage.
