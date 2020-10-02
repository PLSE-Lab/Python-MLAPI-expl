#!/usr/bin/env python
# coding: utf-8

# 1.
# Merge-sort takes O (n log n) time. By choosing 1 as the pivot, the quick-sort algorithm takes only one iteration to partition the input into two segments, one containing all zeros and the other all ones. At this point the list is sorted in only O(n) time.
# 
# 2.
# Apply the merge method on A and B to create C. This takes O(n) time. Now, perform a linear scan through C removing all duplicate elements (i.e. if the next element is equal to the current element, remove it). So, the total running time is O(n).
# 
# 3.
# Similarly to question 1, select 1 as the pivot and perform quick-sort. After the first iteration, the sequence is sorted. This takes O(n) time.
# 
# 4.
# First sort the sequence S by the candidate's ID - this takes O (n lg n) time. Then walk through the sorted sequence, storing the current max count and the count of the current candidate ID as you go. When you move on to a new ID, check it against the current max and replace the max if necessary - this takes O(n) time. Therefore, the total running time is O (n lg n).
# 
# 5.
# In this case, the input data has only a small constant number of different values. Let k be the number of candidates. So, we can perform the Bucket-Sort algorithm. Create an array A of size k and initialise all values to 0. Create a table, by assigning every candidate a unique integer from 0 to k - 1. These two steps take O (1) time. Now, walk through the unsorted sequence S, and for every visited ID, add one to the content of A[i], where i is the corresponding number in the look-up table. This process takes O(n) time. Thus, the total running time is O(n).
# 
# 6.
# * Using merge sort [1000, 80, 10, 50, 70, 60, 90, 20]
# * Splitting [1000, 80, 10, 50, 70, 60, 90, 20]
# * Splitting [1000, 80, 10, 50]
# * Splitting [1000, 80]
# * Splitting [1000]
# * Merging [1000]
# * Splitting [80]
# * Merging [80]
# * Merging [80, 1000]
# * Splitting [10, 50]
# * Merging [10, 50]
# * Merging [10, 50, 80, 1000]
# * Splitting [70, 60, 90, 20]
# * Splitting [70, 60]
# * Splitting [70]
# * Merging [70]
# * Splitting [60]
# * Merging [60]
# * Merging [60, 70]
# * Splitting [90, 20]
# * Splitting [90]
# * Merging [90]
# * Splitting [20]
# * Merging [20]
# * Merging [20, 90]
# * Merging [20, 60, 70, 90]
# * Merging [10, 20, 50, 60, 70, 80, 90, 1000]
