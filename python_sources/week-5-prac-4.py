#!/usr/bin/env python
# coding: utf-8

# Question 1. 
# 
# push(5), push(3), pop(), push(2), push(8), pop(), pop(), push(9), push(1), pop(), push(7), push(6),
# pop(), pop(), push(4), pop(), pop():
# 
# Stack:
# 5
# 5, 3
# pop(): return 3 
# 5, 2
# 5, 2, 8
# pop(): return 8
# pop(): return 2
# 5, 9
# 5, 9, 1
# pop(): return 1
# 5, 9, 7
# 5, 9, 7, 6
# pop(): return 6
# pop(): return 7
# 5, 9, 4
# pop(): return 4
# pop(): return 9
# 

# Question 2.
# 
# 25 push()
# 12 top()
# 10 pop()
# 3 raise empty errors
# 
# 25 - 10 
# = 15
# 15 + 3  = 18         # 3 empty errors raised so only 7 elements removed
# 
# Current size of S = 18
# 

# Question 3.
# 
# 
# def transfer(s, t):
#     while not s.is_empty():
#         t.push(s.pop())
#        

# Question 4. 
# 
#          
# enqueue(5), enqueue(3), dequeue(), enqueue(2), enqueue(8), dequeue(), dequeue(), enqueue(9),
# enqueue(1), dequeue(), enqueue(7), enqueue(6), dequeue(), dequeue(), enqueue(4), dequeue(),
# dequeue().
# 
# 
# Output:
# 
# 
# Queue: 5
# Queue: 5, 3
# Dequeue(): Return 5
# Queue: 3, 2
# Queue: 3, 2, 8
# Dequeue(): Return 3
# Dequeue(): Return 2
# Queue: 8, 9
# Queue: 8, 9, 1
# Dequeue(): Return 8
# Queue: 9, 1, 7
# Queue: 9, 1, 7, 6
# Dequeue(): Return 9
# Dequeue(): Return 1
# Queue: 7, 6, 4
# Dequeue(): Return 7
# Dequeue(): return 6
# Queue: 4
# 
# 
# 
# 

# Question 5. 
# 
# 32 enqueue operations. 
# 15 dequeue operations were also executed, 
# 5 empty errors
# 
# 32 - 15 + 5 (empty errors) = 22

# Question 6. 
# 
# Dequeue
# [1,2,3,4,5,6,7,8]
# Desired output: [1,2,3,5,4,6,7,8]
# 
# Queue
# []
# 
# Methods:
# Q.enqueue(D.delete_first())
# Dequeue: [2,3,4,5,6,7,8]
# Queue: [1]
# 
# Q.enqueue(D.delete_first())
# Dequeue: [3,4,5,6,7,8]
# Queue: [1, 2]
# 
# Q.enqueue(D.delete_first())
# Dequeue: [4,5,6,7,8]
# Queue: [1, 2, 3]
# 
# D.add_last(D.delete_first())
# Dequeue: [5, 6, 7, 8, 4]
# Queue: [1, 2, 3]
# 
# Q.enqueue(D.delete_first())
# Dequeue: [6, 7, 8, 4]
# Queue: [1, 2, 3, 5]
# 
# Q.enqueue(D.delete_last())
# Dequeue: [6, 7, 8]
# Queue: [1, 2, 3, 5, 4]
# 
# Q.enqueue(D.delete_first()) * 3
# Queue: [1, 2, 3, 5, 4, 6, 7, 8]
# 
# D.add_first(D.dequeue()) * 8
# Dequeue: [1, 2, 3, 5, 4, 6, 7, 8]
# 
# 

# Question 7.
# 
# Dequeue: [1, 2, 3, 5, 4, 6, 7, 8]
# Stack: []
# 
# S.push(D.delete_last())                # delete_last instead of delete_first so that when S.pop() is called the elements are
# Dequeue: [1,2,3,4,5,6,7]               # stored in numerical order
# Stack: [8]
# 
# S.push(D.delete_last())
# Dequeue: [1,2,3,4,5,6]
# Stack: [8,7]
# 
# Q.enqueue(D.delete_last())
# Dequeue: [1,2,3,4,5]
# Stack: [8,7,6]
# 
# D.add_first(D.delete_last())
# Dequeue: [5,1,2,3,4]
# Stack: [8,7,6,]
# 
# Q.enqueue(D.delete_first())
# Dequeue: [1,2,3,4]
# Stack: [8,7,6,5]
# 
# Q.enqueue(D.delete_first() * 4)
# Dequeue: [1,2,3,4]
# Stack: [8,7,6,5]
# 
# Q.enqueue(D.delete_last())
# Dequeue: [6, 7, 8]
# Stack: [1, 2, 3, 5, 4]
# 
# Q.enqueue(D.delete_first()) * 3
# Stack: [1, 2, 3, 5, 4, 6, 7, 8]
# 
# D.add_first(D.dequeue()) * 8
# Stack: [1, 2, 3, 5, 4, 6, 7, 8]
# 
