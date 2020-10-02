#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Question1:
    5


# question 2:
# 18

# In[21]:


# Question 3..    
def transfer(s, t):
    for i in range(len(s)):
        t.push(s.pop())


# In[ ]:


#question 4
4


# question 5
# 15 dequeue's minus 5 empty dequeue error = 10 dequeues
# 32 - 10 dequeues = 22
# 

# In[ ]:


question 6.
D = [1,2,3,4,5,6,7,8]
Q = []

Q.enqueue(D.delete_first())
Q = [1]
D = [2,3,4,5,6,7,8]
Q.enqueue(D.delete_first())
Q = [1]
D = [2,3,4,5,6,7,8]
Q.enqueue(D.delete_first())
Q = [1,2,3]
D = [4,5,6,7,8]
D.add_last(D.delete_first())
Q = [1,2,3]
D = [5,6,7,8,4]
Q.enqueue(D.delete_first()) 
Q = [1,2,3,5]
D = [6,7,8,4]
Q.enqueue(D.delete_last()) 
Q = [1,2,3,5,4]
D = [6,7,8]
Q.enqueue(D.delete_first()) 
Q = [1,2,3,5,4,6]
D = [7,8]
Q.enqueue(D.delete_first()) 
Q = [1,2,3,5,4,6,7]
D = [8]
Q.enqueue(D.delete_first()) 
Q = [1,2,3,5,4,6,7,8]
D = []


# In[ ]:


Quetion 7
D = [1,2,3,4,5,6,7,8]
S = []

S.push(D.delete_first())
S = [1]
D = [2,3,4,5,6,7,8]
S.push(D.delete_first())
S = [1]
D = [2,3,4,5,6,7,8]
S.push(D.delete_first())
S = [1,2,3]
D = [4,5,6,7,8]
D.add_last(D.delete_first())
S = [1,2,3]
D = [5,6,7,8,4]
S.push(D.delete_first()) 
S = [1,2,3,5]
D = [6,7,8,4]
S.push(D.delete_last()) 
S = [1,2,3,5,4]
D = [6,7,8]
S.push(D.delete_first()) 
S = [1,2,3,5,4,6]
D = [7,8]
S.push(D.delete_first()) 
S = [1,2,3,5,4,6,7]
D = [8]
S.push(D.delete_first()) 
S = [1,2,3,5,4,6,7,8]
D = []


# 
