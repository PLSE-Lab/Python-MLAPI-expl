#!/usr/bin/env python
# coding: utf-8

# In[ ]:


O(n): 'n' is the length of input to your function


# Python is a "higher level" programming language, so you can accomplish a task with little code. However, there's a lot of code built into the infrastructure in this way that causes your code to actually run much more slowly than you'd think.

# In[1]:


class Node:
    def __init__(self,val):
        self.val = val
        self.next = None # the pointer initially points to nothing


# In[2]:


node1 = Node(12) 
node2 = Node(99) 
node3 = Node(37) 
node1.next = node2 # 12->99
node2.next = node3 # 99->37


# In[3]:


class Node:
    def traverse(self):
        node = self # start from the head node
        while node != None:
            print node.val # access the node value
            node = node.next # move on to the next node

