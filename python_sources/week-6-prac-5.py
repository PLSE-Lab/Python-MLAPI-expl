#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Prac 5:
#Question 1.
#Algorithms find_second_last(L)

def find_second_to_last(L):
    if L.head is None or L.head.next is None:          #{check if the list has 0 or 1 element}
        return None                                      #{there is no second last node}
    current = L.head
    next = L.head.next
    while next.next is not None:                #{if next has a next, then current is not second last}
        current = next
        next = next.next
    return current


# In[2]:


# Question 2.

def count_nodes(L):
 if L.current is None:
     return 0
 count = 1
 length = L.current
 new_length = length
 while new_length.next != length:
     count += 1
 return count


# In[ ]:


# Question 3. 
#Algorithm same_list(x, y):

def same_list(x, y):
    current = x
    while current.next is not y:            # loop until y is found, then it's in the same list
        if current.next is x:                        # if x is next then y is not in the same list
            return False
        current = current.next
    return True


# In[ ]:


# Question 4.

from positional_list import PositionalList

def list_to_positional_list(list_):
 pos_list = PositionalList()
 for element in list_:
     pos_list.add_last(element)
 return pos_list


# In[ ]:


Question 5.

def max(self):
 return max(element for element in self)

