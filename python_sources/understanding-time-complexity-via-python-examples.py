#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import timeit
import numpy as np
import random
random.seed(42)

from math import factorial
# Any results you write to the current directory are saved as output.


# As someone whose introduction to computation was always via Physics, I have had to teach myself Order complexity of algorithms. I found some great resources on the internet, namely https://www.rookieslab.com/posts/how-to-compute-time-complexity-order-of-growth-of-any-program and https://towardsdatascience.com/a-data-scientists-guide-to-data-structures-algorithms-1176395015a0. So I figured I'd write a little kernel to test out the run-times of various programs (with varying time complexities) myself, just to internalize the information better. 
# 
# The big O notation for algorithms (https://en.wikipedia.org/wiki/Big_O_notation) ends up giving us the slowest possible estimate for the run-time of the algorithm, and how that run-time scales with the input size **n**.  Also, in all cases, we are only interested in n >> c, where c is the set of any other constants that may add to the time complexity (say an initialization step) so we don't care about these. 
# 
# Here we just mention the most important and commonly encountered time complexities via some simple example codes in python

# ### First we'll write a function to measure the time of execution

# In[ ]:


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def timer(func, *args):
    wrapped = wrapper(func, *args)
    time = timeit.timeit(wrapped,number=10000)
    return("Time of execution is {} ms".format(time))


# # 1. O(1) (Constant)
# ### Algorithms which always have the same running time

# In[ ]:


# Example from Ravi Ojha's post (see link above). 
# Adds 2 numbers
def O1_add(n1, n2):
    return (n1 + n2)


# In[ ]:


# No matter what the input, the function executes in one step, so roughly the same time complexity
for n in range(1,6):
    print(n,",",n + random.randint(1,int(1e10)))
    print(timer(O1_add, int(n), int(n) + random.randint(1,int(1e10))))
    print()


# ### So we see from the above, that for each run, across the sum of different pairs of numbers, the runtime remains approximately the same. 

# In[ ]:


# Checks whether a number is even or odd by checking last digit of binary representation
def O1_odd_check(num):
    is_odd = False
    if num & 1 == 1:
        is_odd = True
    return is_odd


# In[ ]:


check_lst = [1,5,8,82,101]
for num in check_lst:
    print(num,"::",O1_odd_check(num),"::",timer(O1_odd_check, num))


# ### Again, this function always requires just one operation (i.e. checking the last digit of the binary representation) so the time complexity is constant

# # 2. O(n) (Linear)

# ### This is linear time complexity, so the running time increases, at most, with the size of the input **n**. Once again, this is very neatly explained in https://www.rookieslab.com/posts/how-to-compute-time-complexity-order-of-growth-of-any-program

# In[ ]:


# Finds an item in an unsorted list
def On_simple_search(lst,number):
    is_found = False
    for num in lst:
        if num == number:
            is_found = True
    return is_found


# In[ ]:


lst1 = range(5)
lst2 = range(500)
lst3 = range(50000)

num1 = 2
num2 = -50
num3 = 4000
print(On_simple_search(lst1,num1),"::",timer(On_simple_search,lst1,num1))
print(On_simple_search(lst2,num2),"::",timer(On_simple_search,lst2,num2))
print(On_simple_search(lst3,num3),"::",timer(On_simple_search,lst3,num3))


# ### As you can see from the above, the runtimes scale linearly with the number of elements that the list has, i.e. the maximum number of elements that we have to look through to find the number we are searching for

# # 3. O(log n) (Logarithmic)

# ### In this case, as is self-evident, the time complexity will go as the logarithm of the input size. For a very succinct explanation of why the order complexity in a Binary Search Tree is O(log n) see https://stackoverflow.com/questions/14426790/why-lookup-in-a-binary-search-tree-is-ologn. Here  we will just illustrate a basic binary search.
# 

# In[ ]:


def Ologn_binary_search(list,number):
    first = 0
    last = len(list) - 1
    is_found = False
    while first <= last and not is_found:
        mid = (first + last)//2
        if list[mid] == number:
            is_found = True
        else:
            if number < mid:
                last = mid - 1
            else:
                first = mid + 1
    return is_found


# In[ ]:


lst1 = range(5)
lst2 = range(500)
lst3 = range(50000)

num1 = 2
num2 = -50
num3 = 4000
print(Ologn_binary_search(lst1,num1),"::",timer(Ologn_binary_search,lst1,num1),"::","log value = {}".format(np.log2(len(lst1))))
print(Ologn_binary_search(lst2,num2),"::",timer(Ologn_binary_search,lst2,num2),"::","log value = {}".format(np.log2(len(lst2))))
print(Ologn_binary_search(lst3,num3),"::",timer(Ologn_binary_search,lst3,num3),"::","log value = {}".format(np.log2(len(lst3))))


# ### A bit of quick mental math tells us that the runtimes do indeed scale as the logarithm (to the base 2) of the input size

# # 4. O(n log n) (Log-Linear)

# ### This will arise anytime we call an O(log n) algorithm inside a loop. A basic merge sort example is presented below. (https://stackoverflow.com/questions/18761766/mergesort-python)

# In[ ]:


def Onlogn_merge_sort(sequence):
    if len(sequence) < 2:
        return sequence
    
    m = len(sequence) // 2
    return Onlogn_merge(Onlogn_merge_sort(sequence[:m]), Onlogn_merge_sort(sequence[m:]))


def Onlogn_merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]

    return result


# In[ ]:


array = [4, 2, 3, 8, 8, 43, 6,1, 0]
ar = Onlogn_merge_sort(array)
print (" ".join(str(x) for x in ar))


# In[ ]:


lst1 = [4,2,3,8,8,43,6,1,0,83]
lst2 = []
for i in range(100):
    lst2.append(random.randint(0,i))


# In[ ]:


print("Sorted lst1:: ",Onlogn_merge_sort(lst1))
print(timer(Onlogn_merge_sort,lst1)," :: nlogn ~= {}".format(len(lst1)*np.log2(len(lst1))))

print("Sorted lst2:: ",Onlogn_merge_sort(lst2))
print(timer(Onlogn_merge_sort,lst2)," :: nlogn ~= {}".format(len(lst2)*np.log2(len(lst2))))


# ### So we see that the runtime for the merge sort algorithm scales as O (n log n)

# # 5. O ($n^2$) (Quadratic)

# ### Quadratic complexity is one general case of polynomial complexity (O($n^c$) where c is some positive integer). We test out the runtime on a basic bubble sort algorithm. 

# In[ ]:


def On2_bubble_sort(lst):
    for i in range(len(lst)-1):
        for j in range(len(lst)-1-i):
            if lst[j] > lst[j+1]:
                lst[j], lst[j+1] = lst[j+1], lst[j]
    return lst


# In[ ]:


lst1 = [4,2,3,8,8,43,6,1,0,83]
lst2 = []
for i in range(100):
    lst2.append(random.randint(0,i))


# In[ ]:


print("Sorted lst1:: ",On2_bubble_sort(lst1))
print(timer(On2_bubble_sort,lst1)," :: n^2 ~= {}".format(len(lst1)**2))

print("Sorted lst2:: ",On2_bubble_sort(lst2))
print(timer(On2_bubble_sort,lst2)," :: n^2 ~= {}".format(len(lst2)**2))


# ### It's easy to understand why understand why bubble sort is O($n^2$), since there are two nested for loops, each running **n** times. The runtimes are also roughly scale with the order of the square of the input size. 

# # 6. O($2^n$) (Exponential)

# ### The run-time or order complexity will double for every new addition to the input (https://stackoverflow.com/questions/1592649/examples-of-algorithms-which-has-o1-on-log-n-and-olog-n-complexities). A classic example is the Fibonacci series, calculated recursively. Each function call makes 2 more function calls till we get to zero.

# In[ ]:


# Sum of a Fibonacci series up to the nth term
def o2n_fibonacci(n):
    if n<2:
        return n
    return o2n_fibonacci(n-1) + o2n_fibonacci(n-2)


# In[ ]:


for n in range(2,12,2):
    print("Series sum for {} is {}".format(n,o2n_fibonacci(n))," :: ",timer(o2n_fibonacci,n)," :: 2^n = {}".format(2**n))


# ### And why is this exponential?  Well, the definition of *e* is literally $\lim_{n \to \infty} (1 +\frac{1}{n} )^n $. 

# # 7. O(n!) (Factorial)

# ### A program to get all the permutations of an array (list) would be a simple example of this, since the number of lists you have is n! (https://stackoverflow.com/questions/104420/how-to-generate-all-permutations-of-a-list-in-python)

# In[ ]:


def onfac_perm(a, k=0):
    if k==len(a):
#         print(a) # Commendted out for display purposes
        pass
    else:
        for i in range(k, len(a)):
            a[k],a[i] = a[i],a[k]
            onfac_perm(a, k+1)
            a[k],a[i] = a[i],a[k]


# In[ ]:


lst1 = [1,2,]
lst2 = [1,2,3,4]
lst3 = [1,2,3,4,5,6]

print("List of {} items :: ".format(len(lst1)), timer(onfac_perm,lst1), " :: factorial {} is {}".format(len(lst1),factorial(len(lst1))))
print("List of {} items :: ".format(len(lst2)), timer(onfac_perm,lst2), " :: factorial {} is {}".format(len(lst2),factorial(len(lst2))))
print("List of {} items :: ".format(len(lst3)), timer(onfac_perm,lst3), " :: factorial {} is {}".format(len(lst3),factorial(len(lst3))))


# ### One can see the run time trends clearly. 
# 
# ### This brings this kernel to a close. An effort to retain and understand some of the most common Time complexity types, along with a check of the actual run-times of the algorithms. I will build on this and write another kernel which talks about the Big O for the various kinds of sorting algorithms.
# 
# ### Upvote it if you like it and let me know if you find it helpful.  If I have mentioned mention any faulty concepts that are in need of addressing please feel free to correct me! 
# 
# ### Find me on LinkedIn at https://www.linkedin.com/in/panchajanya-banerjee/

# In[ ]:




