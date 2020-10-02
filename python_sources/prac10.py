#!/usr/bin/env python
# coding: utf-8

# Q1
# 
# Merge-sort algorithm takes O(n log n) time.
# 
# choosing 1 as the pivot point, the quick-sort algorithm takes a single iteration to partition data into two segments, the first containing all zeros and the second all ones. Time tacken to sort is O(n).
# 

# Q2
# 
# Apply the merge method on A and B to create C which takes O(n) time. Then conduct a linear scan
# through C while removing all duplicate elements. The total running time will be O(n).

# Q3
# 
# select 1 as the pivot point and perform quick-sort. After the first iteration, the sequence
# is sorted. This takes O(n) time.
# 

# Q4
# 
# Firstly sort the sequence S by the candidate's ID - which will take O(n lg n) time. After that walk through the sorted sequence, storing max count and the count of the current candidate ID as you go. When you move on to a new ID, check it against the current max and replace the max if necessary - this takes O(n) time. Therefore, the total running time is O(n lg n).
# 

# Q5
# 
# In this case, the input data has only a small constant number of different values. Let k be the
# number of candidates. So, we can perform the Bucket-Sort algorithm.
# Create an array A of size k, and initialise all values to 0. Create a table, by assigning every candidate a
# unique integer from 0 to k - 1. These two steps take O(1) time.
# Now, walk through the unsorted sequence S, and for every visited ID, add one to the content of A[i],
# where i is the corresponding number in the look-up table. This process takes O(n) time.
# Thus, the total running time is O(n).
# 

# Q6
# 
# ![Prac10Q6.jpg](attachment:Prac10Q6.jpg)
