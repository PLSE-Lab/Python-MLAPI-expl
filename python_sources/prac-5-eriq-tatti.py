#!/usr/bin/env python
# coding: utf-8

# Question 1
# 
# Algorithm SecondToLast(x)
# 
# if x.head equals None or x.head.next equals None: 
# 
# return None
# 
# current = L.head
# 
# next = L.head.next
# 
# while next.next is not equal to None: ****
# 
# current = next
# 
# next = next.next
# 
# return current

# Question 2

# In[ ]:


def nodeCounter(x):
    if x.current is None:
        return 0
    count = 1
    original = x.current
    current = original
    while current.next != original:
        count += 1
    return count


# Question 3
# 
# Algorithm listCheck(x, y):
# 
# current = x
# 
# while current.next is not y: 
# 
# if current.next is x: 
# 
# return False
# 
# current = current.next
# 
# return True

# Question 4

# In[ ]:


from positional_list import PositionalList
def list_to_positional_list(list_):
    pos_list = PositionalList()
    for element in list_:
        pos_list.add_last(element)
    return pos_list


# Question 5

# In[ ]:


def max(self):
    return max(element for element in self)

