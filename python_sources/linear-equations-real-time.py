#!/usr/bin/env python
# coding: utf-8

# Here is a simple real time scenario where Linear Equation can be used to solve.
# 
# A small story:
# Assume your friend Chris is telling you that you owe him 49$. You are asking him for the reason.
# 
# He came to visit you on Saturday but you were not there. So, he went back to his place on Saturay and came to your place on Sunday.
# 
# "He spent 30 in Day 1 by eating 3 times out and paid 2times for the transportation"
# 
# "He spent 19 in Day 2 by eating 2 times out and reached your place by paying 1 time ticket"
#         
#  Also he tells you that you ate for the same amount and paid the same amount always for the ticket.
#  As you are a math guy, you aregoing to solve this problem by Linear Equations
#  
#  So, your equation will be:
#  
# >  3x + 2y = 30  ----- Day 1
#  
# >  2x +  y = 19  ----- Day 2
#  
#  Here x is food and y is the bus ticket. Assume x and y are always constant (I know not all the restaurants won't charge you the same price but for this scenario, we will keep them constant)
#  
#  You will solve like this:
#  
# >  3x + 2y = 30
# >  -4x- 2y  = -38 (Day2 is multiplied by -2)
# >  -------------
# >  -x       = -8
# >  ------------
#  
#  So, x = 8
#  
#  
#  Apply this to first
#  
#  3 (8) + 2y = 30
#  
#  24 + 2y = 30
#  
#  2y = 30 - 24
#  
#  2y = 6
#  
#  y = 3
#  
#  So, finally you got the answer: Your friend Chris spent 3 for the bus fare and 8$ for each time he had food outside. 
#  
#  

# If you are a datascientist, you will solve the same problem 
# 
# 

# In[ ]:


import numpy as np

items_spent = np.array([
        [3, 2],
        [2, 1]
    ])
    
spent_total = np.array([30, 19])
item_value = np.linalg.solve(items_spent, spent_total)

print(item_value)


# In[ ]:


# Check the solution is right
print(np.allclose(np.dot(items_spent, item_value), spent_total))


# So, 
# 
# He spent
# 
# 
# **day 1:**
# 
# 3 times eating 
# 
# 3 * 8 = 24
# 
# travel to and fro
# 2 * 3 = 6
# 
# day1 total: 
# 24 + 6 = 30
#  
#  
# 
# 
# 
# **day 2:**
# 
# 2 times eating:
# 2 * 8 = 16
#             
# just one way
# 1 * 3 = 3
# 
# day 2 total: 
# 16 + 3 = 19

# **Note:**
# 
# I am not sure how to fix size in the day 2 equation. If you can help me to fix the font issue, it would be great.
# 
# Also, feel free to give some feedback. Happy to learn from others!

# In[ ]:





# In[ ]:




