#!/usr/bin/env python
# coding: utf-8

# 1- (R-6.1) What values are returned during the following series of stack operations, if executed upon
# an initially empty stack?
# push(5), push(3), pop(), push(2), push(8), pop(), pop(), push(9), push(1), pop(), push(7), push(6),
# pop(), pop(), push(4), pop(), pop().

# * 
# * 
# * 3
# * 
# * 
# * 8
# * 2
# * 
# * 
# * 1
# * 
# * 
# * 6
# * 7
# * 
# * 4
# * 9

# 2- (R-6.2) Suppose an initially empty stack S has executed a total of 25 push operations, 12 top
# operations, and 10 pop operations, 3 of which raised Empty errors that were caught and ignored.
# What is the current size of S?
# 

# 12

# 3- (R-6.3) Implement a function transfer(S, T) that transfers all elements from stack S onto stack T, so
# that the element that starts at the top of S is the first to be inserted onto T, and the element at the
# bottom of S ends up at the top of T. Use ArrayStack (ch06/array_stack.py) to test your function.
# 
# 

# In[ ]:


def reverse_function():
    S = ArrayStack()
    original = S
    for line in S:
        S.push(line.rstrip('\n'))

    T = ArrayStack()
    output = T
    while not S.is_empty():
        output.write(S.pop() + '\n')


# 4- (R-6.7) What values are returned during the following sequence of queue operations, if executed on
# an initially empty queue?
# enqueue(5), enqueue(3), dequeue(), enqueue(2), enqueue(8), dequeue(), dequeue(), enqueue(9),
# enqueue(1), dequeue(), enqueue(7), enqueue(6), dequeue(), dequeue(), enqueue(4), dequeue(),
# dequeue().
# 

# * 
# * 
# * 5
# * 
# * 
# * 3
# * 2
# * 
# * 8
# * 
# * 
# * 9
# * 1
# * 7
# * 6

# 5- (R-6.8) Suppose an initially empty queue Q has executed a total of 32 enqueue operations. 15
# dequeue operations were also executed, 5 of which raised Empty errors that were caught and
# ignored. What is the current size of Q?
# 

# 22

# 6- (R-6.13) Suppose you have a deque D containing the numbers (1,2,3,4,5,6,7,8), in this order.
# Suppose further that you have an initially empty queue Q. Give a code fragment that uses only D
# and Q (and no other variables) and results in D storing the elements in the order (1,2,3,5,4,6,7,8).

# In[ ]:


def queue_queue():
    D.dequeue = [1,2,3,4,5,6,7,8]
    for i in D:
        D.pop()

    Q = []
    while not D.is_empty():
        Q.enqueue(D)


# 7-  (R-6.14) Repeat the previous problem using the deque D and an initially empty stack S.

# In[ ]:


def queue_stack():
    D.dequeue = [1,2,3,4,5,6,7,8]
    for i in D:
        D.pop()

    S = ArrayStack()
    while not D.is_empty():
        S.push(D)

