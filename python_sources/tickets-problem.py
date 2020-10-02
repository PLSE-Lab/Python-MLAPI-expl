#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from random import *
trials = 10000
counter_3yes = 0
counter_1yes_2no = 0
tickets = [1 for i in range(7)] + [0 for i in range(3)]
for n in range(trials):    
    draw = sample(tickets, 3)
    if draw[2] == 1:
        counter_3yes += 1
        if draw[0] == 1 and draw[1] == 0:
            counter_1yes_2no += 1
            
print(counter_1yes_2no / counter_3yes)

