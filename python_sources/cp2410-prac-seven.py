#!/usr/bin/env python
# coding: utf-8

# Q1.<br>
# 
# The list during an insertion sort:
# 
# **5** <br>
# 5, **6**<br>
# **3**, 5, 6<br>
# **1**, 3, 5, 6<br>
# 1, **2**, 3, 5, 6<br>
# 1, 2, 3, 5, 6,** **7<br>
# 1, 2, 3, 5, 6, 7, **8**<br>
# 1, 2, 3, 5, 6, 7, 8, **9**<br>
# 
# 
# The list during a selection sort
# 
# **1**, 6, 3, **5**, 2, 7, 9, 8<br>
# 1, **2**, 3, 5, **6**, 7, 9, 8<br>
# 1, 2, **3**, 5, 6, 7, 9, 8<br>
# 1, 2, 3, **5**, 6, 7, 9, 8<br>
# 1, 2, 3, 5, **6**, 7, 9, 8<br>
# 1, 2, 3, 5, 6, **7**, 9, 8<br>
# 1, 2, 3, 5, 6, 7, **8**, **9**<br>
# 

# Q2.<br>
# 
# An insertion sort, in the worst case, will run for O(n^2) amount of times. This happens with insertions sorts that are in descending order of keys (e.g. 28:15, 13:10, 9:5). For n element in the sequence

# Q3.<br>
# 
# Add 5:
# 
#         5
#    
# Add 1:
# 
#         5               1
#        /        ->     /
#       1               5
#       
# Add 4:
# 
#         1
#        / \
#       5   4
#       
# Add 7:
# 
#         1
#        / \
#       5   4
#      /
#     7
#     
# Add 3:
# 
#                1                 1
#               / \               / \
#              5   4    ->       3   4
#             / \               / \
#            7   3             7   5
#            
# Add 9:
# 
#                 1
#                / \
#               /   \
#              3     4
#             / \   /
#            7   5 9
#            
# Add 0:
#     
#                 1                   0
#                / \                 / \
#               /   \               /   \
#              3     4    ->       3     1
#             / \   / \           / \   / \
#            7   5 9   0         7   5 9   4
#            
#            
# Add 2:
# 
#                 0                        
#                / \
#               /   \
#              3     1
#             / \   / \
#            7   5 9   4
#           /
#          2
#            
#        
# Swap:                
#           
#           
#                 0        
#                / \
#               /   \
#              2     1
#             / \   / \
#            3   5 9   4
#           /
#          7
#          
# Add 8:
# 
#                    
#                 0        
#                / \
#               /   \
#              2     1
#             / \   / \
#            3   5 9   4
#           / \
#          7   8
# 
#     
#           

# Q4.<br>
# 
#             2
#        3         4
#      8   5     7    6
#      
#      
# Move 6 to the root node:
# 
#             6
#        3         4
#      8   5     7    
#      
#  
# Move 6 down:
#  
#             4
#        3         6
#      8   5     7    

# Q5.<br>
# 
# The third smallest will be a child of the root node (aka position 2 or 3).

# Q6.<br>
# 
# The largest key will be down the bottom of the heap, this is called an external node.
