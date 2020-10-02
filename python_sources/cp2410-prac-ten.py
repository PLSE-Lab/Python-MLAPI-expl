#!/usr/bin/env python
# coding: utf-8

# Q1.<br>
# 
# Merge-sort algorithms typically take O(n log n) time because it does not take into account the number of possible values (even if it is only two). Quick-sort, however, will only have to loop through once to divide the list into two. The list will be sorted in only O(n) time.

# Q2.<br>
# 
# One way of removing duplicates is through a linear scan. But first we have to merge A and B to create a new sequence C. This will take O(n) time. Then the linear scan will search for and remove all duplicates which takes O(n) time.

# Q3.<br>
# 
# Here, a quick-sort can be implemented, using 1 as the pivot. After one iteration the sequence is sorted, taking only O(n) time.

# Q4.<br>
# 
# The sequence S is sorted on the ID given to each canidate, taking O(n log n) time.
# 
# Then, we will step through the now sorted sequence, keeping track of the max count and the count of the current canidate ID as we go. As we move along and check each ID, replace the max with the count if it exceeds it. This takes O(n) time.
# 
# The worst run time is O(n log n).

# Q5.<br>
# 
# To reduce the run-time on a constant number of canidates, we can use a bucket-sort.
# 
# Leting k be the number of canidates, we will create an array the size of k and a table of unique integers assigned to each canidate (k). These steps will take O(1) time.
# 
# Now, similarly to the previous question, we will walk through the unsorted sequence and for every ID we visit we will add one to the content of the array[i]. This will take O(n) time.
# 
# The final run time is O(n).

# Q6.<br>
# 
#                                                 1000, 80, 10, 50, 70, 60, 90, 20
#                                       1000, 80, 10, 50                    70, 60, 90, 20
#                                     1000, 80      10, 50                 70, 60      90, 20
#                                     1000   80     10   50                70  60      90  20
#                                      80, 1000     10, 50                 60, 70      20, 90
#                                        10, 50, 80, 1000                     20, 60, 70, 90
#                                                 10, 20, 50, 60, 70, 80, 90, 1000
