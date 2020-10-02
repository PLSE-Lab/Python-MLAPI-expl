#!/usr/bin/env python
# coding: utf-8

# <h3>Name: Emily Van Gorp
# <br>Assignment 2 - CP2410
# <br>Worst Run Time: O(n^3)

# Load the libraries

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sympy as sp
import sys
from scipy.spatial import distance
from math import isinf


# Load 100% of the dataset

# In[ ]:


df_cities = pd.read_csv('../input/cities.csv')


# The data structure used in this algorithm is a list.
# 
# This function does not have a linear run time as it contains a while loop nested in a for loop
# At worst, it will run through n elements n times O(n^2)
# List comprehension runs n times with 2 operations (assignment, sequence access)
# 2 primitive operations are run (assignment)
# For loop runs n times 
# If statement runs 1 time with 1 primitive operations inside it
# A while loop runs n times with 2 operations

# In[ ]:


#Determine which cities are prime numbers
def sieve_of_eratosthenes(n):
    primes = [True for i in range(n+1)] # Start assuming all numbers are primes
    primes[0] = False # 0 is not a prime
    primes[1] = False # 1 is not a prime
    for i in range(2,int(np.sqrt(n)) + 1):
        if primes[i]:
            k = 2
            while i*k <= n:
                primes[i*k] = False
                k += 1
    return(primes)


# In[ ]:


df_cities['is_prime'] = sieve_of_eratosthenes(max(df_cities.CityId))


# In[ ]:


prime_cities = sieve_of_eratosthenes(max(df_cities.CityId))


# In[ ]:


#Visualising the map with the prime cities highlighted.
get_ipython().run_line_magic('matplotlib', 'notebook')
fig = plt.figure(figsize=(10,10))
plt.scatter(df_cities[df_cities['CityId']==0].X , df_cities[df_cities['CityId']==0].Y, s= 200, color = 'red')
plt.scatter(df_cities[df_cities['is_prime']==True].X , df_cities[df_cities['is_prime']==True].Y, s= 0.8, color = 'purple')
plt.scatter(df_cities[df_cities['is_prime']==False].X , df_cities[df_cities['is_prime']==False].Y, s= 0.1)
plt.grid(False)


# Determining the distances between two cities using a dictionary.

# In[ ]:


start_time = time.time()
def pair_distance(x,y):
    x1 = (df_cities.X[x] - df_cities.X[y]) ** 2
    x2 = (df_cities.Y[x] - df_cities.Y[y]) ** 2
    return np.sqrt(x1 + x2)
end_time = time.time()
dumbest_elapsed_time = end_time - start_time


# The built in python map type, dictionary, will only run for O(1) in an algorithm as simple as this. Sequence access and element assignment are both primitive operations with constant run time.

# In[ ]:


print("Total elapsed time of dumbest path algorithm: ", dumbest_elapsed_time)


# In[ ]:


def total_distance(path):
    distance = [pair_distance(path[x], path[x+1]) + 0.1 * pair_distance(path[x], path[x+1])
    if (x+1)%10 == 0 and df_cities.is_prime[path[x]] == False else pair_distance(path[x], path[x+1]) for x in range(len(path)-1)]
    return np.sum(distance)


# The worst run time of this algorithm will be O(n) because the most complex aspect is the list comprehension.
# 
# Beyond that it is comprised of primitive assignment and sequence access.

# In[ ]:


dumbest_path = df_cities['CityId'].values
#add North Pole add the end of trip
dumbest_path =  np.append(dumbest_path,0)


# In[ ]:


print('Total distance with the paired city path is '+ "{:,}".format(total_distance(dumbest_path)))


# In[ ]:


sys.setrecursionlimit(500000)
City_X=[]
for x in range(max(df_cities.CityId)+1):
    City_X.append(df_cities['X'][x])
City_Y=[]
for x in range(max(df_cities.CityId)+1):
    City_Y.append(df_cities['Y'][x])


# The worst run time for this algorithm is O(n) for the iteration being performed. It is not O(n^2) even though there are two for loops because they are not nested. This run time could also be specified to O(2n) for the two separate iterations through n.

# In[ ]:


path=[]
for x in range(1,max(df_cities.CityId)+1):
        path.append(x)


# This algorithms will only run for a maximum of O(n) due to the for loop.

# In[ ]:


def partition(arr,low,high): 
    i = ( low-1 )         # index of smaller element 
    pivot = arr[high]     # pivot 
  
    for j in range(low , high): 
  
        # If current element is smaller than or 
        # equal to pivot 
        if   City_X[arr[j]] <= City_X[pivot]: 
          
            # increment index of smaller element 
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i] 
  
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return ( i+1 ) 


# The worst run time of this algorithm is constant time, O(n). The biggest operation present is the iteration of a for loop.

# In[ ]:


start_time = time.time()
def quickSort(arr,low,high): 
    if low < high: 
  
        # pi is partitioning index, arr[p] is now 
        # at right place 
        pi = partition(arr,low,high) 
  
        # Separately sort elements before 
        # partition and after partition 
        quickSort(arr, low, pi-1) 
        quickSort(arr, pi+1, high) 
end_time = time.time()
quicksort_elapsed_time = end_time - start_time


# This divide and conquer algorithm depends on which element the partition algorithm starts on. In most cases, the algorithm will start on the element in the middle, making the run time equal to O(n log n). However, the quicksort algorithm can take up to O(n^2) for run time in the worst case.

# In[ ]:


print("Total elapsed time of quicksort algorithm: ", quicksort_elapsed_time)


# In[ ]:


quicksort_path=[]
for x in range(1,max(df_cities.CityId)+1):
        quicksort_path.append(x)


# In[ ]:


quickSort(quicksort_path,0,len(quicksort_path)-1)


# In[ ]:


quicksorted_path=[0]
for each in range(len(quicksort_path)):
    quicksorted_path.append(quicksort_path[each])
quicksorted_path.append(0)


# In[ ]:


print('Total distance with the quick sorted cities based on X path is '+ "{:,}".format(total_distance(quicksorted_path)))


# The quicksort algorithm provides a much better outcome in terms on the distance travelled at the cost of run time. The worst case run time of the dumbest path was only linear O(n) whereas the quicksort runs for O(n log n) at best and O(n^2) at worst.

# This algorithm uses the graphing data structure called the matrix. The run time of a BFS (Breadth-First Search) depends on the amount of edges (E) and vertices (V) in the matrix. The initialisation step runs once and thus takes O(V) time. Each vertex is enqueued and then dequeued at most once. So the total time for to perform queue operations is O(V). Each adjacency list is scanned at most once. Since the sum of the length of all the adjacency lists is O(E), the total time spent to scanning adjacency lists is O(E).
# Overall, the total running time of the BFS algorithm is O(V + E).

# In[ ]:


matrix=[]
def generate_graph():
    for i in range(20):
        #get array of list X,Y value from dataset
        coordinates = np.array([df_cities.X.values, df_cities.Y.values]).T[0:20]
        #calculate distance of all city from last city in path list
        dist = ((coordinates - np.array([City_X[i], City_Y[i]]))**2).sum(-1)
        matrix.append(dist.tolist())
generate_graph()


# In[ ]:


def size(int_type):
   length = 0
   count = 0
   while (int_type):
       count += (int_type & 1)
       length += 1
       int_type >>= 1
   return count

def length(int_type):
   length = 0
   count = 0
   while (int_type):
       count += (int_type & 1)
       length += 1
       int_type >>= 1
   return length


# In[ ]:


def generateSubsets(n):
    l = []
    for i in range(2**n):
        l.append(i)
    return sorted(l, key = lambda x : size(x) )


# In[ ]:


def inSubset(i, s):
    while i > 0 and s > 0:
        s = s >> 1
        i -= 1
    cond = s & 1
    return cond

def remove(i, s):
    x = 1
    x = x << i
    l = length(s)
    l = 2 ** l - 1
    x = x ^ l
    #print ( "i - %d x - %d  s - %d x&s -  %d " % (i, x, s, x & s) )
    return x & s
def findPath(p):
    n = len(p[0])
    number = 2 ** n - 2
    prev = p[number][0]
    path = []
    while prev != -1:
        path.append(prev)
        number = remove(prev, number)
        prev = p[number][prev]
    reversepath = [str(path[len(path)-i-1]+1) for i in range(len(path))]
    reversepath.append("1")
    reversepath.insert(0, "1")
    return reversepath


# All of the above algorithms will have a worst run time of the constant O(n) as they only have singular, non-nested for and while loops.

# In[ ]:


start_time = time.time()
def tsp():
    n=len(matrix) 
    l = generateSubsets(n)
    cost = [ [-1 for city in range(n)] for subset in l]
    p = [ [-1 for city in range(n)] for subset in l]
    count = 1
    total = len(l)
    
    for subset in l:
        for dest in range(n):
            if not size(subset):
                cost[subset][dest] = matrix[0][dest]
            elif (not inSubset(0, subset)) and (not inSubset(dest, subset)):
                mini = float("inf")
                for i in range(n):
                    if inSubset(i, subset):
                        modifiedSubset = remove(i, subset)
                        val = matrix[i][dest] + cost[modifiedSubset][i]
                        
                        if val < mini:
                            mini = val
                            p[subset][dest] = i
                            
                if not isinf(mini):
                    cost[subset][dest] = mini
        count += 1
    path = findPath(p)
    print(" => ".join(path))
    Cost = cost[2**n-2][0]
    print("Total distance with dynamic programing using graph:",Cost)
tsp()  
end_time = time.time()
matrix_elapsed_time = end_time - start_time


# The algorithm has one of the worst run times out of the entire solution. With three nested for loops, it runs for a time of O(n^3) at the very least. It also contains some non-nested list comprehensions that will run for O(n) each but the worst run time out of all of these will be O(n^3).

# In[ ]:


print("Total elapsed time of matrix algorithm: ", matrix_elapsed_time)


# In[ ]:


def final_output():
    dict = {'Path': tree_path}  
    df = pd.DataFrame(dict) 
    df.to_csv('Final_Submission.csv', index=False)


# This dictionary would have a run time of O(1).
