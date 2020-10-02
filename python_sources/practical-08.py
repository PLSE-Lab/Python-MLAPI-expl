#!/usr/bin/env python
# coding: utf-8

# ## Q1
# 
# 0: [13], 1: [94 39], 2: [], 3: [], 4: [], 5: [44 88 11], 6: [], 7: [], 8: [12 23], 9: [16 5], 10: [20] 

# ## Q2
# 
# 0: [13], 1: [94], 2: [39], 3: [16], 4: [5], 5: [44], 6: [88], 7: [11], 8: [12], 9: [23], 10: [20]

# ## Q3
# 
# 0: [13], 1: [94], 2: [39], 3: [11], 4: [], 5: [44], 6: [88], 7: [16], 8: [12], 9: [23], 10: [20]
# 
# Entry '5' can not be put into this hash table as 4 does not appear for any answers from j=0 to j=10

# ## Q4
# 
# 0: [13], 1: [94], 2: [23], 3: [88], 4: [39], 5: [44], 6: [11], 7: [5], 8: [12], 9: [16], 10: [20]  
#   
# I cross-checked this with the answers, and I believe your answer is flawed at H(11)
# 
# The equation is  
# H(11) = ((3\*11+5) + j(7 - 11 mod 7)) mod 11  
# H(11) = (38+3) mod 11  
# H(11) = 41 mod 11 = 8   
# 
# In your answers, you have listed this as 9, which creates a vastly different hash table.  
# 

# ## Q5
# 
# 0: [], 1: [], 2: [12], 3: [18], 4: [41], 5: [], 6: [36], 7: [25], 8: [], 9: [54], 10: [], 11: [], 12: [38], 13: [10], 14: [], 15: [90], 16: [28], 17: [], 18: []

# ## Q6
# 
# Search each row of A for the first 0, then some all the positions to obtain the total # of 1s
# 
# Binary search will take O(lg n) time, and this being used on each row of a matrix makes it O(n lg n) time
