#!/usr/bin/env python
# coding: utf-8

# #  Assignment 1 (Traveling Santa 2018)
# Name: Quang Khai Nguyen (13410893)
# 

# Getting data:
# Import csv file and read only 10% data (19776 data) from the whole dataset.
# Time complexity for prime number: O(n)
# Time complexity for total dumbest path: O(n) 
# Time complexity for total selection sort path:O(n^2)
# Time complexity for total insertion sort path:O(n^2)
# TOtal time complexity for total: O(n^2)
# 

# In[ ]:


import numpy as np 
import pandas
import matplotlib.pyplot as plt
from scipy.spatial import distance
import time


# In[ ]:


filename = "../input/assignment/cities.csv"
df = pandas.read_csv(filename,nrows=19777)
df.head()


# In[ ]:


nb_cities = max(df.CityId)
print("Number of cities to visit : ", nb_cities)


# Getting Prime number:

# In[ ]:


start=time.time()
def is_prime(n):
    if n> 1:
        for i in np.arange(2, n-1) :
            if n % i == 0:
                return False
        
        return True
    
    return False
df['prime'] = df['CityId'].apply(is_prime)
end=time.time()


# In[ ]:


print(end-start)


# In[ ]:


sum(df.prime)


# In[ ]:


b=[]
for i in df['prime']:
    b.append(i)


# In[ ]:


fig, ax = plt.subplots(figsize = (10,10))
ax.scatter(x = df.X, y = df.Y, alpha = 0.5, s = 1)
ax.set_title('Cities chart. North Pole, prime and non prime cities', fontsize = 16)
ax.scatter(x = df.X[0], y = df.Y[0], c = 'r', s =12)
ax.scatter(x = df[df.prime].X, y = df[df.prime].Y, s = 1, c = 'purple', alpha = 0.3)
ax.annotate('North Pole', (df.X[0], df.Y[0]),fontsize =12)
ax.set_axis_off()


# Calcuclate the dumest path to compare to others 

# In[ ]:


start = time.time()
def total_distance(dfcity,path):
    prev_city = path[0]
    total_distance = 0
    step_num = 1
    for city_num in path[1:]:
        next_city = city_num
        total_distance = total_distance +             np.sqrt(pow((df.X[city_num] - df.X[prev_city]),2) + pow((df.Y[city_num] - df.Y[prev_city]),2)) *             (1+ 0.1*((step_num % 10 == 0)*int(not(b[prev_city]))))
        prev_city = next_city
        step_num = step_num + 1
    return total_distance


# In[ ]:


dumbest_path = list(df.CityId[:].append(pandas.Series([0])))
print('Total distance with the dumbest path is '+ "{:,}".format(total_distance(df,dumbest_path)))
end = time.time()


# In[ ]:


print(end-start)


# In[ ]:


import csv
with open('dumbest_path.csv','w',newline="") as fp:
    writter = csv.writer(fp)
    data=[["path:",total_distance(df,dumbest_path)]]
    writter.writerows(data)


# In[ ]:


s=pandas.read_csv('../input/assignment/Cities_Sorted.csv')
s['path']=np.array(dumbest_path)
s.to_csv("final_submission.csv",index=False)


# Algorithm 1:

# Selection sort:

# In[ ]:


start = time.time()
sort=[]
for i in range(1,max(df.CityId)+1):
    sort.append(i)
sort1=[]
for i in df["X"]:
    sort1.append(i)
def selectionsort(A):
    for i in range(0,len(A)):
        minIndex=i
        for j in range (i+1,len(A)-1):
            if sort1[A[j]]<sort1[A[minIndex]]:
                minIndex=j
       
        temp = A[i]
        A[i] = A[minIndex]
        A[minIndex] = temp
end = time.time()


# In[ ]:


print(end-start)


# In[ ]:


selectionsort(sort)


# In[ ]:


print('Total distance is '+ "{:,}".format(total_distance(df,sort)))


# In[ ]:


s=pandas.read_csv('../input/assignment/Cities_Sorted.csv')
s['path']=np.array(sort)
s.to_csv("Selection_sort.csv",index=False)


# Algorithm 2:

# Insertion sort:

# In[ ]:


start = time.time()
sort3=[]
for m in range(1,max(df.CityId)+1):
    sort3.append(m)
sort2=[]
for n in df["Y"]:
    sort2.append(n)
    
def insertionSort(alist):
    for index in range(1,len(alist)):
        currentvalue = alist[index]
        position = index
        while position>0 and sort2[alist[position-1]]>sort2[currentvalue]:
            alist[position]=alist[position-1]
            position = position-1
        alist[position]=currentvalue
end=time.time()


# In[ ]:


insertionSort(sort3)


# In[ ]:


print(end-start)


# In[ ]:


print('Total distance is '+ "{:,}".format(total_distance(df,sort3)))


# In[ ]:


s=pandas.read_csv('../input/assignment/Cities_Sorted.csv')
s['path']=np.array(sort3)
s.to_csv("neiborhoof.csv",index=False)

