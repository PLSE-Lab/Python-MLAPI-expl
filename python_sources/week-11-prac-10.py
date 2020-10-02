#!/usr/bin/env python
# coding: utf-8

# Question 1. 
# 
# Merge sort = O(n lg n) time, it does not know about the case of only two possible values.
# Choosing 1 as the pivot, the quick-sort algorithm takes only one iteration to partition the input into two segments, one containing all zeros and the other all ones. At this point the list is sorted in only O(n) time.
# 

# Question 2. 
# 
# Use merge method (on A and B) to create C - takes O(n) time. Next, perform a linear scan through C, removing duplicates - if next element = current elemnt then remove. Total run-time = O(n).
# 

# Question 3. 
# 
# Similarly to question 1, select 1 as the pivot and perform quick-sort. After the first iteration, the sequence
# is sorted. This takes O(n) time.

# Question 4. 
# 
# First sort the sequence S by the candidate's ID - this takes O(n lg n) time.
# 
# Then walk through the sorted sequence, storing the current max count and the count of the current
# candidate ID as you go. When you move on to a new ID, check it against the current max and replace the
# max if necessary - this takes O(n) time.
# 
# Therefore, the total running time is O(n lg n).

# Question 5. 
# 
# In this case, the input data has only a small constant number of different values. Let k be the
# number of candidates. So, we can perform the Bucket-Sort algorithm.
# 
# Create an array A of size k, and initialise all values to 0. Create a table, by assigning every candidate a
# unique integer from 0 to k - 1. These two steps take O(1) time.
# 
# Now, walk through the unsorted sequence S, and for every visited ID, add one to the content of A[i],
# where i is the corresponding number in the look-up table. This process takes O(n) time.
# Thus, the total running time is O(n).

# Question 6. 
# 
# * See attached file for answer
