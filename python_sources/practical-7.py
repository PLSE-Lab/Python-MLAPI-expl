#!/usr/bin/env python
# coding: utf-8

# # Question 1

# In[8]:


#https://interactivepython.org/courselib/static/pythonds/SortSearch/TheInsertionSort.html
def insertionSort(alist):
   for index in range(1,len(alist)):

     currentvalue = alist[index]
     position = index

     while position>0 and alist[position-1]>currentvalue:
         alist[position]=alist[position-1]
         position = position-1

     alist[position]=currentvalue

alist = [5,6,3,1,2,7,9,8]
insertionSort(alist)
print(alist)


# In[7]:


# https://interactivepython.org/lpomz/courselib/static/pythonds/SortSearch/TheSelectionSort.html
def selectionSort(alist):
   for fillslot in range(len(alist)-1,0,-1):
       positionOfMax=0
       for location in range(1,fillslot+1):
           if alist[location]>alist[positionOfMax]:
               positionOfMax = location

       temp = alist[fillslot]
       alist[fillslot] = alist[positionOfMax]
       alist[positionOfMax] = temp


alist = [5,6,3,1,2,7,9,8]
selectionSort(alist)
print(alist)


# ### Step by step
# 
# [5,6,3,1,2,7,9,8]
# #### Insertion sort
# - Collection C = [5,6,3,1,2,7,9,8] - Priority Queue P = []
# - Phase 1
# - a, C = [6,3,1,2,7,9,8] - P = [5]
# - b, C = [3,1,2,7,9,8] - P = [5,6]
# - c, C = [1,2,7,9,8] - P = [3,5,6]
# - d, C = [2,7,9,8] - P = [1,3,5,6]
# - d, C = [7,9,8] - P = [1,2,3,5,6]
# - e, C = [9,8] - P = [1,2,3,5,6,7]
# - f, C = [8] - P = [1,2,3,5,6,7,9]
# - g, C = [] - P = [1,2,3,5,6,7,8,9]
# - Phase 2
# - a, C = [1] - P = [2,3,5,6,7,8,9]
# - b, C = [1,2] - P = [3,5,6,7,8,9]
# - c, C = [1,2,3] - P = [5,6,7,8,9]
# - d, C = [1,2,3,5] - P = [6,7,8,9]
# - d, C = [1,2,3,5,6] - P = [7,8,9]
# - e, C = [1,2,3,5,6,7] - P = [8,9]
# - f, C = [1,2,3,5,6,7,8] - P = [9]
# - g, C = [1,2,3,5,6,7,8,9] - P = []
# 
# 
# ##### Selection sort
# 
# - Collection C = [5,6,3,1,2,7,9,8] - Priority Queue P = []
# - Phase 1
# - a, C = [6,3,1,2,7,9,8] - P = [5]
# - b, C = [3,1,2,7,9,8] - P = [5,6]
# - c, C = [1,2,7,9,8] - P = [5,6,3]
# - d, C = [2,7,9,8] - P = [5,6,3,1]
# - d, C = [7,9,8] - P = [5,6,3,1,2]
# - e, C = [9,8] - P = [5,6,3,1,2,7]
# - f, C = [8] - P = [5,6,3,1,2,7,9]
# - g, C = [] - P = [5,6,3,1,2,7,9,8]
# - Phase 2
# - a, C = [1] - P = [5,6,3,2,7,9,8]
# - b, C = [1,2] - P = [5,6,3,7,9,8]
# - c, C = [1,2,3] - P = [5,6,7,9,8]
# - d, C = [1,2,3,5] - P = [6,7,9,8]
# - d, C = [1,2,3,5,6] - P = [7,9,8]
# - e, C = [1,2,3,5,6,7] - P = [9,8]
# - f, C = [1,2,3,5,6,7,8] - P = [9]
# - g, C = [1,2,3,5,6,7,8,9] - P = []
# 

# ### Question 2
# 
# A worst case sequence for insertion sort would be one with decending keys. 
# for example 14,12,9,5,3,1. All elements would have to be compared to other elements, making it 
# 1 + 2 +  ... + (n - 1) + n = n(n + 1) = O(n2).

# # Question 3

# In[25]:


class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0


    def percUp(self,i):
        while i // 2 > 0:
          if self.heapList[i] < self.heapList[i // 2]:
             tmp = self.heapList[i // 2]
             self.heapList[i // 2] = self.heapList[i]
             self.heapList[i] = tmp
          i = i // 2

    def insert(self,k):
      self.heapList.append(k)
      self.currentSize = self.currentSize + 1
      self.percUp(self.currentSize)

    def percDown(self,i):
      while (i * 2) <= self.currentSize:
          mc = self.minChild(i)
          if self.heapList[i] > self.heapList[mc]:
              tmp = self.heapList[i]
              self.heapList[i] = self.heapList[mc]
              self.heapList[mc] = tmp
          i = mc

    def minChild(self,i):
      if i * 2 + 1 > self.currentSize:
          return i * 2
      else:
          if self.heapList[i*2] < self.heapList[i*2+1]:
              return i * 2
          else:
              return i * 2 + 1

    def delMin(self):
      retval = self.heapList[1]
      self.heapList[1] = self.heapList[self.currentSize]
      self.currentSize = self.currentSize - 1
      self.heapList.pop()
      self.percDown(1)
      return retval

    def buildHeap(self,alist):
      i = len(alist) // 2
      self.currentSize = len(alist)
      self.heapList = [0] + alist[:]
      while (i > 0):
          self.percDown(i)
          i = i - 1

bh = BinHeap()
bh.buildHeap([5,1,4,7,3,9,0,2,8])

print(bh.delMin())
print(bh.delMin())
print(bh.delMin())
print(bh.delMin())
print(bh.delMin())
print(bh.delMin())
print(bh.delMin())
print(bh.delMin())
print(bh.delMin())


# #### Will look something like this step by step
# 
# 
# ![prac7.PNG](attachment:prac7.PNG)

# ### Question 4 
# 
# To remove key 2 from the heap step 1 is to swap it with key 6, key 6 being 
# the last key inserted. 
# The secound step is to swap key 6 and 4. 
# 
# The heap will then look something like: 
# 
# ![prac7q4.PNG](attachment:prac7q4.PNG)
# 
# 

# ### Question 5 
# 
# The third smallest key can be stored at depth 1 or 2. 

# ### Questiom 6
# 
# The largest key will be stored in an external node
