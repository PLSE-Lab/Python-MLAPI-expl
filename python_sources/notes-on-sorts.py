#!/usr/bin/env python
# coding: utf-8

# # Notes on sorts
# 
# I've written pieces of notebooks on the various sorting algorithms and their properties in different assorted places over time, it's time to run through them again and consolidate my notes into a single notebook with all of the bits in it.

# ## Things about sorting generally
# * Sorting is useful because having data in sorted order very often enables algorithms which are not otherwise possible. Logic that is based on a sorted array element has $O(1)$ access to its neighbors; logic based on an unsorted array element requires $O(n)$ time.
# * Because of their ubiquity, and because sorting is a relatively simple task to understand, sorts are a big chunk of introductory computer science algorithms curriculums.
# 
# 
# * Sorts fall into two general categories.
#   * The first category is comparison sorts: sorts which perform their work by comparing values in the array to one another (e.g. `a > b`). Comparison sorts have proveably at best $O(n\log{n})$ performance. There is a proof for why this is the lower bound in the Harvard intro to algorithms coursework that I watched long ago, but it's hopeless (and pointless) to try and remember it.
#   * The second category is noncomparison sorts. Noncomparison sorts do not compare values, and thus are not limited by their theoretical lower time complexity bound. Noncomparison sorts may perform the work in up to $O(n)$ time. The tradeoff is usually made in memory: comparison sorts generally require memory proportional to the size or arity of the array.
# 
# 
# * Sorts may have two other salient properties.
# * A sort is **stable** if elements with the same value appear in the same order in the final array in which they appeared in the originator array. This property is useful in certain application contexts.
# * A sort is **in-place** if it requires no additional memory allocation, e.g. it can be done only with swaps and pointers.
# 
# ## Simple sorts
# 
# * The simplest possible sort is **bogosort**, which simply randomizes the elements of the array and then compares them. Bogosort has unbounded time complexity and $O(n!)$ average time complexity.
# * The next simplest possible sort is **bubble sort**. Bubble sort selects each index in the list in iteration order and sorts it rightwards until the element is sorted with respect to its right neighbor. This process is repeated until a pass has to do no work, confirming the list is fully sorted.

# ```
# bubble_sort(arr):
#     i <- 0
#     swap_seen = True
#     while swap_seen:
#         swap_seen = False
#         while i < arr.length - 1:
#         if arr[i] > arr[i + 1]:
#             arr[i], arr[i + 1] = arr[i + 1], arr[i]
#             swap_seen = True
#     return arr
# ```

# In[ ]:


def bubble_sort(arr):
    swap_seen = True
    while swap_seen:
        i = 0
        swap_seen = False
        while i < len(arr) - 1:
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swap_seen = True
            i += 1
    return arr


# In[ ]:


bubble_sort([3,2,1,0])


#  Bubble sort also has $O(n^2)$ worst-case performance due to its use of two loops. The worst case occurs when the first value in the sorted list appears in the last value in the array.

# ## Selection sort

# * The **selection sort** picks the smallest value in the array and places it at the beginning of the array.

# In[ ]:


def selection_sort(arr):
    i = 0
    while i < len(arr):
        idx = i
        idxmin = None  # sentinel value
        while idx < len(arr):
            if idxmin is None or arr[idx] < arr[idxmin]:
                idxmin = idx
            idx += 1
        print(i, idxmin)
        arr[i], arr[idxmin] = arr[idxmin], arr[i]
        i += 1
    return arr


# Selection sort is similarly $O(n^2)$.

# ## Insertion sort

# * The **insertion sort** picks values one at a time, except instead of bubbling them up, it sorts them into place with respect to the values left of the current index. This is famously the sort that humans use for sorting things like hands of cards. The insertion sort is $O(n^2)$.

# In[ ]:


def insertion_sort(arr):
    for j in range(1, len(arr)):
        for k in range(j):
            if arr[k] > arr[j]:
                arr = arr[:k] + [arr[j]] + arr[k:j] + arr[j + 1:]
    return arr


# In[ ]:


insertion_sort([3,2,1])


# ## Quicksort
# 
# * Quicksort recursively partitions the list into sublists of elements bigger than a central pivot value and elements smaller than a central pivot value. The smaller elements are sorted onto the left side, and the bigger ones, onto the right side. These sublists are then, in turn, themselves sorted. The end result is a sorted array.
# * Quicksort is $O(n\log{n})$. Each sort (or subsort) requires $O(n)$ work. Assuming that the array is partitioned into lists of roughly equal length, the array will be partitioned for as many levels as it takes to partition the array into length 1 to 2 components: $\log_2{n}$. Each level requires $n$ work, as it again touches every element of the list (less pivot values). This gives us our final computational complexity: $O(n\log{n})$.

# In[ ]:


def quicksort(arr):
    if arr is None or len(arr) <= 1:
        return arr
    
    pivot = len(arr) // 2
    pv = arr[pivot]
    left, right = [], []
    for i in range(pivot):
        if arr[i] > pv:
            right.append(arr[i])
        else:
            left.append(arr[i])
    for i in range(pivot + 1, len(arr)):
        if arr[i] < pv:
            left.append(arr[i])
        else:
            right.append(arr[i])
    
    if len(left) > 1:
        left = quicksort(left)
    if len(right) > 1:
        right = quicksort(right)
        
    return left + [pv] + right


# In[ ]:


quicksort([1,2,3])


# ## Merge sort
# 
# * Merge sort, like quicksort, is $O(n\log{n})$. It's very much quicksort in reverse: partition the list into length-1 or length-2 sublists, then recursively sort them into one another. In other words, whereas quicksort is *top-down*, merge sort is *bottom-up*.

# In[ ]:


def mergesort(arr):
    if arr is None or len(arr) <= 1:
        return arr
    
    if len(arr) == 2:
        return arr if arr[0] < arr[1] else arr[::-1]
    
    pivot = len(arr) // 2
    l, r = mergesort(arr[:pivot]), mergesort(arr[pivot:])
    return join(l, r)
    
def join(l, r):
    ll, rl = len(l), len(r)
    li = ri = 0
    out = []
    while li < ll and ri < rl:
        if l[li] < r[ri]:
            out.append(l[li])
            li += 1
        else:
            out.append(r[ri])
            ri += 1
    while li < ll:
        out.append(l[li])
        li += 1
    while ri < rl:
        out.append(r[ri])
        ri += 1
    return out


# In[ ]:


mergesort([5,3,1,7,2,4,6])


# ## Counting sort
# * The **counting sort** is the simplest noncomparison sort. In a counting sort you allocate a hashmap of values to counts, iterate through the array and increment the corresponding value in the hashmap by one every time you see a particular value appear in the array.
# * This has $O(n)$ time complexity and $O(m)$ memory cost, where $m$ is the arity of the array.

# In[ ]:


def countingsort(arr):
    m = {}
    for v in arr:
        if v not in m:
            m[v] = 1
        else:
            m[v] += 1
    
    out = []
    for v in m:
        for _ in range(m[v]):
            out.append(v)
    
    return out


# ## Radix sort
# 
# * The Radix sort is the counting sort applied recursively on an integer (or character) sequence with some number of significant digits. Start with the least significant digit and bucket sort that. Then, sort the next least significant digit (using a counting sort), making sure to retain digits that tie *in iteration order*. If you repeat this process for each significant digit, you get a sorted array in $O(nm)$ time, where $n$ is the length of the list and $m$ is the number of digits in its most significant member.
# * This logic is actually kind of tricky to implement in practice, the following code doesn't quite get it right unfortunately. :(

# In[ ]:


def countingsortidx(arr):
    m = [[] for _ in range(max(arr))]
    for i, v in enumerate(arr):
        m[v - 1] += [i]
    return m

def radixsort(arr):
    maxlen = max([len(str(v)) for v in arr])
    for i in range(maxlen - 1, -1, -1):
        darr = []
        for v in arr:
            v_d = int(str(v).zfill(maxlen)[i])
            darr.append(v_d)
        darr_sorted_idxs = countingsortidx(darr)
        print(darr_sorted_idxs)
        new_arr = []
        for idxs in darr_sorted_idxs:
            for idx in idxs:
                new_arr.append(arr[idx])
        arr = new_arr
    return arr


# ## Bucket sort
# 
# * The bucket sort is a counting sort where values are binned into some number of buckets. Each bucket is then recursively subsorted. A counting sort is used at some level as a subroutine, once the values are sufficiently discretized by the bucketing process. This process is $O(nk)$, where $k$ is the depth of the bucket splits. Implementation omitted.
