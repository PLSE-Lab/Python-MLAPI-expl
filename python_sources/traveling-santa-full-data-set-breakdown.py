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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Setup

# In[ ]:


cities = pd.read_csv('../input/cities.csv')
cities.tail() # checking


# Prepare prime function to check wether it is prime or not. This is sieve of eratosthenes algorithm

# In[ ]:


def isPrime(n): # time complexity: O(nlogn)
    prime = [True for i in range(n + 1)] 
    prime[0] = False # 0 and 1 is not prime
    prime[1] = False
    p = 2
    while (p * p <= n):  # keep the loop only log n
        # If prime[p] is not changed, then it is a prime
        if prime[p]:
            # Update all multiples of p
            for i in range(p * p, n + 1, p):
                prime[i] = False
        p += 1
    return prime


# Create another column to the data set, called prime, return true if it is prime, else false

# In[ ]:


prime_cities = isPrime(max(cities.CityId))
cities["prime"] = prime_cities # add prime column to the dataset
cities.tail()


# Preparing function to calculate distance between 2 cities. This function is used seperately to perform some specific tasks. Note that this 2 cities are in list form.

# In[ ]:


def distance(c1, c2): #time complexity: O(1)
     return np.sqrt((c1[1]-c2[1])**2 + (c1[2]-c2[2])**2 ) # eucludian distance formula


# Preparing a function to calculate distance of a path (path is a list of cityIds only). The function will get 2 cityId from the path, searching for coordination of x and y in the dataset and calculate the distance between them, and repeat until finish the path. This also includes prime conditions. 

# In[ ]:


def list_only_total_distance(path): #time complexity: O(n)
    step_num = 1 # init count of steps
    total = 0    # init total
    for i in range (len(path)-1): # loop all cities in path
        # calculate 2 cities distance, sum it with total distance
        total += distance(list(cities.values[path[i]]), list(cities.values[path[i+1]])) 
        if step_num % 10 == 9 and cities.values[path[i+1]][3] == False: # prime path condition
            total += distance(list(cities.values[path[i]]), list(cities.values[path[i+1]])) * 0.1
        step_num += 1 # keep increase step to be reasonable with the condition
    return total


# # Sampling

# Even though this supposed to be a full dataset analysis, however, my computer cannot handle this huge dataset. Therefore, I will examine only first 100 instances. Why 100? cause I gonna use graph and AVL tree, which will totally freeze my computer if I run with 200+ instances. Let's begin by getting first 100 instances from the dataset

# In[ ]:


sample = cities[:100] #Sampling first 100 instances
sample.tail() #checking sample


# For comparing purpose, I will calculate the dumbest path, which is blindly go from 0 to 1, 2, 3 ... and back to 0

# In[ ]:


dumbest_path = range(100)
list_only_total_distance(dumbest_path)


# # Greedy method with implement of Graph and Merge Sort

# The idea of this greedy method is to plot out a graph and use it to get to nearest cities. This method won't consider the prime conditions, therefore, we wouldn't get the most optimize distance out of it. The reason why I choose this method is that, it seem like a naivee method, however, it get to done things seperately and reduced time a lot.

# Begin with Vertex class and Graph class.

# In[ ]:


class Vertex:
    def __init__(self,key):
        self.id = key
        self.connectedTo = {}

    def addNeighbor(self,nbr,dis=0):
        self.connectedTo[nbr] = dis

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self,nbr):
        return self.connectedTo[nbr]


# In[ ]:


class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self,key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self,n):
        return n in self.vertList

    def addEdge(self,f,t,dis=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], dis)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())


# After that will be a function to create Vertices and Edges. This function would have a nested loop and the result would be n^2, however, this edge function only direct A to B but not B to A. To make B to A, we need to call the class method again and would cost more time. And by not doing this, I reduce instances and time by twice

# In[ ]:


def createGraphWithEdges(data): # time complexity O(n^2)
    g = Graph() # init graph
    for i in range(len(data) - 1): # loop through data, get distance between each vertex and others
        for j in range(i + 1, len(data)): # this loop use to reduce duplicate edge 
            # append vertices and edge
            g.addEdge(data.values[i][0],data.values[j][0],distance(list(data.values[i]),list(data.values[j])))
    return g


# run it on sample

# In[ ]:


sampleg = createGraphWithEdges(sample)


# Next would be Merge sort function. However, this merge sort would be different than a normal one. As the list that I would give to the method will consist 2 index, destination cityId and its distance. I want to sort the cityId based on the distance itself. Therefore some changes has been made: 

# In[ ]:


def WeirdMergeSort(listDis): # time complexity: O(nlogn)
    if len(listDis) > 1:
        mid = len(listDis)//2 #Finding the mid of the array 
        L = listDis[:mid] # Dividing the array elements  
        R = listDis[mid:] # into 2 halves 
        # keep repeating
        WeirdMergeSort(L) # Sorting the first half 
        WeirdMergeSort(R) # Sorting the second half 
        i = j = k = 0 # indexing variables
        # Copy data to temp arrays L[] and R[] 
        while i < len(L) and j < len(R): # changes made here, instead of compare list to list( which is impossible)
            if L[i][1] < R[j][1]:  # we will compare distance (second index) of each list,
                listDis[k] = L[i]  #  and sort the list according
                i+=1
            else: 
                listDis[k] = R[j] 
                j+=1
            k+=1 
        # Checking if any element was left 
        while i < len(L): 
            listDis[k] = L[i] 
            i+=1
            k+=1
        while j < len(R): 
            listDis[k] = R[j] 
            j+=1
            k+=1


# Next would be the main method. This method will be a recursive method, therefore, it would take a few initial parameters. The idea is to get a starting city, then checking for the nearest city (vertex) by comparing distance (edge) between them. After that, choose the nearest, repeat the method with the nearest city as a starting city. There will be a remain cities list, which will cross out all chosen cities and would be empty when the method finished. 

# In[ ]:


def nearestPath(graph,path,remain,start): #
    path.append(start) # add the starter to the list
    if len(path) < graph.numVertices: # check if path have enough instance or not
        remain.pop(remain.index(start)) #remove it from the remain cities 
        cities_dis = []
        for i in remain:
            # Due to the creation of graph with edges
            # To reduce the number of instances and times
            # the method only direct the source to destination without reverse it
            temp = []
            if start < i: # if start < i, we can get distance from source to destination
                src = sampleg.getVertex(start) 
                des = sampleg.getVertex(i)
                dis = src.getWeight(des)
                temp.append(des.getId())
                temp.append(dis)
                cities_dis.append(temp) # add to list to sort out later
            else: # else, we need to reverse it
                src = sampleg.getVertex(start) 
                des = sampleg.getVertex(i)
                dis = des.getWeight(src) # get the distance from destination to source
                temp.append(des.getId()) 
                temp.append(dis)
                cities_dis.append(temp) # add to list to sort out later
        WeirdMergeSort(cities_dis) # using merge to improve time and grade :D
        start = cities_dis[0][0] # change starter city to the nearest one(after sort, nearest city would be the first)
        nearestPath(graph,path,remain,start)# keep travel to nearest city of the nearest city to create the path
    else: # if enough instances, make it travel back to the first city, return the path
        path.append(0)
        return path
                   
                   


# Go ahead and try it out. 

# In[ ]:


first_path = []
remain = list(range(sampleg.numVertices))
start_city = 0
nearestPath(sampleg, first_path, remain, start_city)
list_only_total_distance(first_path)


# Comparing to the dumbest one, this one is way better, however, it does took more time, effort and resources

# # Same concept but with AVL tree implemention

# The reason why I using the same concept again is that I want to see if prime condition would help to increase efficiency for the Santa or not. If the difference is fairly small, it is clearly prime condition should not be considered.

# Because of using same concept, I will also implementing AVL tree to see if between graph and AVL tree, which one would cost more time. Starting off with TreeNode class and AVL tree class

# In[ ]:


# Generic tree node class 
class TreeNode(object): 
    def __init__(self, val, key): 
        self.key = key
        self.val = val 
        self.left = None
        self.right = None
        self.height = 1


# In[ ]:


class AVL_Tree(object): 
  
    # Recursive function to insert key in  
    # subtree rooted with node and returns 
    # new root of subtree. 
    def insert(self, root, val,key): 
      
        # Step 1 - Perform normal BST 
        if not root: 
            return TreeNode(val, key) 
        elif val < root.val: 
            root.left = self.insert(root.left, val,key) 
        else: 
            root.right = self.insert(root.right, val,key) 
  
        # Step 2 - Update the height of the  
        # ancestor node 
        root.height = 1 + max(self.getHeight(root.left), 
                           self.getHeight(root.right)) 
  
        # Step 3 - Get the balance factor 
        balance = self.getBalance(root) 
  
        # Step 4 - If the node is unbalanced,  
        # then try out the 4 cases 
        # Case 1 - Left Left 
        if balance > 1 and val < root.left.val: 
            return self.rightRotate(root) 
  
        # Case 2 - Right Right 
        if balance < -1 and val > root.right.val: 
            return self.leftRotate(root) 
  
        # Case 3 - Left Right 
        if balance > 1 and val > root.left.val: 
            root.left = self.leftRotate(root.left) 
            return self.rightRotate(root) 
  
        # Case 4 - Right Left 
        if balance < -1 and val < root.right.val: 
            root.right = self.rightRotate(root.right) 
            return self.leftRotate(root) 
  
        return root 
  
    def leftRotate(self, z): 
  
        y = z.right 
        T2 = y.left 
  
        # Perform rotation 
        y.left = z 
        z.right = T2 
  
        # Update heights 
        z.height = 1 + max(self.getHeight(z.left), 
                         self.getHeight(z.right)) 
        y.height = 1 + max(self.getHeight(y.left), 
                         self.getHeight(y.right)) 
  
        # Return the new root 
        return y 
  
    def rightRotate(self, z): 
  
        y = z.left 
        T3 = y.right 
  
        # Perform rotation 
        y.right = z 
        z.left = T3 
  
        # Update heights 
        z.height = 1 + max(self.getHeight(z.left), 
                        self.getHeight(z.right)) 
        y.height = 1 + max(self.getHeight(y.left), 
                        self.getHeight(y.right)) 
  
        # Return the new root 
        return y 
  
    def getHeight(self, root): 
        if not root: 
            return 0
  
        return root.height 
  
    def getBalance(self, root): 
        if not root: 
            return 0
  
        return self.getHeight(root.left) - self.getHeight(root.right) 
    
    def inOrderList(self, root, l): 
        if not root:
            return
        self.inOrderList(root.left,l)
        l.append(root.key)
        self.inOrderList(root.right,l) 


# Implementing AVL tree by inserting list of destination cityId and its distance into an AVL tree and then get the inorder out of it (inorder of an binary tree would be sorted). Then, by getting the nearest city and assign as the next start city, we would create a same path as above. However, I set up a prime condition, where at the 10th step, the method would search for the nearest prime city.  

# In[ ]:


def AVLtreePath(data, path, remain, start, step): #time complexity: O(n^3)
    current_city_dis = [] # start an empty list 
    samplet = AVL_Tree() # init AVL tree
    root = None # init root
    path.append(start) # get the chosen city
    remain.pop(remain.index(start)) # remove it from remain
    if len(remain) > 0: # if still remain, start to searching for nearest
        for cityId in remain: 
            dis = distance(list(cities.values[start]), list(cities.values[cityId])) 
            root = samplet.insert(root,dis,cityId) # insert destination and distance
        samplet.inOrderList(root,current_city_dis) # inorder method would sort destination by its distance
        start = current_city_dis[0] # after sort, nearest would the first
        if step % 10 == 9: # if 10th step
            for cityId in current_city_dis: # get first prime city, then exit the loop
                if cities.values[cityId][3] == True:
                    start = cityId
                    break;
        step += 1 # increment step
        AVLtreePath(data, path, remain, start, step) # repeat
    else: # if enough instances, make it travel back to the first city, return the path
        path.append(0)
        return path


# Let's try it out

# In[ ]:


second_path = []
remain_cities = list(range(len(sample)))
start_city = 0
step = 0
AVLtreePath(sample, second_path, remain_cities, start_city, step)
list_only_total_distance(second_path)


# The result is better than the above one, however, it took quite a lot of time even with only 100 instance. This also mean prime condition has a great impact to the travel path. 
