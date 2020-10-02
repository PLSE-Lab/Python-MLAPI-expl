#!/usr/bin/env python
# coding: utf-8

# <h1>Assignment 1</h1>
# 
# <h2>Topic</h2>
# <p>
#     URL: <a href="https://www.kaggle.com/c/traveling-santa-2018-prime-paths/data">https://www.kaggle.com/c/traveling-santa-2018-prime-paths/data</a><br/>
#     All iterations used 10% of actual dataset
# </p>
# <ol>
#     <b>Contents</b>:
#     <li>First Iteration</li>
#     <li>Second Iteration v1</li>
#     <li>Second Iteration v2</li>
#     <li>Third Iteration</li>
#     <li>Linked List</li>
#     <li>Fourth Iteration</li>
#     <li>Fith Iteration</li>
#     <li>Queue</li>
#     <li>Solution</li>
# </ol>
#     

# In[ ]:


import csv
from time import time
import matplotlib.pyplot as plt
import queue
import math
from collections import deque


# In[ ]:


def isprime(n):
    """
    Check if integer n is a prime
    """
    # make sure n is a positive integer
    n = abs(int(n))
    # 0 and 1 are not primes
    if n < 2:
        return False
    # 2 is the only even prime number
    if n == 2: 
        return True    
    # all other even numbers are not primes
    if not n & 1: 
        return False
    # range starts with 3 and only needs to go up the squareroot of n
    # for all odd numbers
    for x in range(3, int(n**0.5)+1, 2):
        if n % x == 0:
            return False
    return True


# In[ ]:


def sort_list(list, axis1, axis2):
    """
    Sorts list in the order of axis1 followed by axis2
    """
    return sorted(list, key=lambda val: (val[axis1], val[axis2]))


# <h2>First Iteration</h2>
# 
# <div>
#     <ol>
#         <li>
#             Data Structure
#             <ul>
#                 <li>Lists of list</li>
#             </ul>
#         </li>
#         <li>
#             Data Algorithms
#             <ul>
#                 <li>While and For loops</li>
#             </ul>
#         </li>
#         <li>
#             Methods used:
#             <ul>
#                 <li>No sorting used</li>
#                 <li>Take csv file, load it into the list, using lists of list.</li>
#                 <li>Format: cities_list = [[0,316.836739061509,2202.34070733524] ... 
#                     [19776,1883.0509264308,1318.48663504281]]</li>
#                 <li>Compare every point with other points in the list, and calculate the distance between two 
#                     points.</li>
#                 <li>Remove the point with the shortest distance, add the distance to total distance, and add the 
#                     point to path_shortest_list.</li>
#             </ul>
#         </li>
#         <li>Time Complexity: O(n<sup>2</sup>)</li>
#     </ol>
# </div>    

# In[ ]:


# Calculate the execution time
start = time()

# List of lists
cities_list = []

# Open csv file
with open("../input/cities/cities_10p.csv", "r") as cities_file:
    cities_data = csv.reader(cities_file)
    for row in cities_data:
        cities_list.append([int(row[0]), float(row[1]), float(row[2])])

path_shortest_list = []
total_dist = 0
shortest_dist = math.inf
index = 0

# Counter for every step taken
step_counter = 1

# Finding nearest point to origin point
x_origin = cities_list[0][1]
y_origin = cities_list[0][2]
for i in range(1,len(cities_list)):
    x_index = cities_list[i][1]
    y_index = cities_list[i][2]
    eucl_dist = math.sqrt(((x_index-x_origin)*(x_index-x_origin))+((y_index-y_origin)*(y_index-y_origin)))
    if eucl_dist < shortest_dist:
        shortest_dist = eucl_dist
        index = i

step_counter += 1
total_dist += shortest_dist

# Saving origin point for bring back path to origin
origin_point = cities_list.pop(0)
path_shortest_list.append(origin_point)

x_2 = cities_list[index][1]
y_2 = cities_list[index][2]
path_shortest_list.append(cities_list.pop(index))

# As long as there is more than 1 item in cities_list, find the next nearest point
while len(cities_list)>0:
    shortest_dist2 = math.inf
    for i in range(0,len(cities_list)):
        x_index = cities_list[i][1]
        y_index = cities_list[i][2]
        eucl_dist = math.sqrt(((x_index-x_2)*(x_index-x_2))+((y_index-y_2)*(y_index-y_2)))

        # Check: Every 10th step (stepNumber % 10 == 0) is 10% more lengthy unless coming from a prime CityId.
        if (step_counter % 10 == 0) and not isprime(cities_list[i][0]):
            eucl_dist = eucl_dist*1.1

        if eucl_dist < shortest_dist2:
            shortest_dist2 = eucl_dist
            index = i

    step_counter += 1
    total_dist += shortest_dist2
    x_2 = cities_list[index][1]
    y_2 = cities_list[index][2]
    path_shortest_list.append(cities_list.pop(index))

# Adding point of origin to path_shortest_list
eucl_dist = math.sqrt(((x_2-x_origin)*(x_2-x_origin))+((y_2-y_origin)*(y_2-y_origin)))
total_dist += eucl_dist
path_shortest_list.append(origin_point)

print("Total Distance: {}".format(total_dist))
print("Shortest path list: {}".format(len(path_shortest_list)))
print("Cities list: {}".format(len(cities_list)))

end = time()

exec_time = end - start
print("Execution time: {}".format(exec_time))

# Plotting route of shortest path taken
fig, ax = plt.subplots(figsize=(15,10))
origin_point = path_shortest_list.pop(0)
plt.scatter(origin_point[1], origin_point[2], s=10, c="red")

prev_point = origin_point

for i in range(len(path_shortest_list)):
    point = path_shortest_list.pop(0)

    xs = point[1]
    ys = point[2]
    ax.plot([prev_point[1], xs], [prev_point[2], ys], c="blue", linewidth=0.5)

    prev_point = point

plt.show()


# <h3>Thoughts:</h3>
# <div>
#     <p>
#         Using python's standard lists, I used this first iteration as a benchmark to the other algorithms
#         I will be planning to make use of. <br/>
#     </p>
#     <p>
#         Alternative methods: <br/>
#         Try using binary search as it is a more efficient way to search for points.
#         > Sort by X and Y
#     </p>
# </div>

# <h2>Second Iteration v1</h2>
# 
# <div>
#     <ol>
#         <li>Data Structure
#             <ul>
#                 <li>Lists of list</li>
#             </ul>
#         </li>
#         <li>Data Algorithms
#             <ul>
#                 <li>While and For loops</li>
#                 <li>Binary Search</li>
#             </ul>
#         </li>
#         <li>Methods used:
#             <ul>
#                 <li>Take csv file, load it into the list, using lists of list.</li>
#                 <li>Format: cities_list = [[0,316.836739061509,2202.34070733524] ... 
#                     [19776,1883.0509264308,1318.48663504281]</li>
#                 <li>Before comparing, since I am using Binary Search, I will sorting list according to X 
#                     coordinates then Y coordinates.</li>
#                 <li>Once sorted, I will half the list at every loop, until i get the middle point
#                 and calculate the distance of the middle point with the previous point</li>
#                 <li>Remove the middle point, add the distance to total distance, and add the 
#                     point to path_shortest_list.</li>
#             </ul>
#         </li>
#         <li>Time Complexity: O(n log n)</li>
#     </ol>
# </div>    

# In[ ]:


def shortest_dist_i2_v1(list_x, x, y):
    """
    Get the closest distance between all the points inside list_x and (x,y)
    Return: list with shortest distance, r_list and shortest_distance
    """
    list_to_check = list_x

    while True:
        # Get length of x sorted list
        len_x = len(list_to_check)
        
        # If list is less than or equal to 3, manually calculate every point in the list
        # with the previous list so attain more accurate result
        if len_x <= 1:
            break
        else:
            # Get the middle point with no remainder
            midpoint = len_x // 2

            left_x = list_to_check[:midpoint]
            right_x = list_to_check[midpoint:]

            midpoint_x = list_to_check[midpoint][1]

            if x <= midpoint_x:
                list_to_check = left_x
            else:
                list_to_check = right_x
    
    dist = math.sqrt(((list_to_check[0][1] - x)*(list_to_check[0][1] - x))+((list_to_check[0][2] - y)*(list_to_check[0][2] - y)))

    return (list_to_check[0],dist)


# In[ ]:


# Calculate the execution time
start = time()

# List of lists
cities_list = []

# Open csv file
with open("../input/cities/cities_10p.csv", "r") as cities_file:
    cities_data = csv.reader(cities_file)
    for row in cities_data:
        cities_list.append([int(row[0]), float(row[1]), float(row[2])])

path_shortest_list = []
total_dist = 0

# Get origin point
x_origin = cities_list[0][1]
y_origin = cities_list[0][2]
city_origin = cities_list.pop(0)
path_shortest_list.append(city_origin)

# Sort list
# 1 = X, 2 = Y
x_sorted_list = sort_list(cities_list,1,2)

closest_list = list()

# Find nearest point to city_origin
(closest_list, d) = shortest_dist_i2_v1(x_sorted_list, x_origin, y_origin)
total_dist += d
x_sorted_list.remove(closest_list)
path_shortest_list.append(closest_list)

# Find the remaining nearest points
while len(x_sorted_list)>=1:
    (closest_list, d) = shortest_dist_i2_v1(x_sorted_list, closest_list[1], closest_list[2])
    total_dist += d
    x_sorted_list.remove(closest_list)
    path_shortest_list.append(closest_list)

print("Total Distance: {}".format(total_dist))
print("Shortest path list: {}".format(len(path_shortest_list)))
print("Cities list: {}".format(len(x_sorted_list)))

end = time()

exec_time = end - start
print("Execution time: {}".format(exec_time))

# Plotting route of shortest path taken
fig, ax = plt.subplots(figsize=(15,10))
origin_point = path_shortest_list.pop(0)
plt.scatter(origin_point[1], origin_point[2], s=10, c="red")

prev_point = origin_point

for i in range(len(path_shortest_list)):
    point = path_shortest_list.pop(0)

    xs = point[1]
    ys = point[2]
    ax.plot([prev_point[1], xs], [prev_point[2], ys], c="blue", linewidth=0.5)

    prev_point = point

    plt.grid(False)
    ax.autoscale()

plt.show()


# <h3>Thoughts:</h3>
# <div>
#     <p>
#         This method is very time efficient, but distance is inaccurate, at about 41.77x worse without factoring 
#         in the additional every 10 steps and prime cities. <br/>
#         This is due to the binary search algorithm searching for the middle x-coordinate point, resulting in 
#         extra distance moved along the y-axis.
#     </p>
#     <p>
#         Alternative methods: <br/>
#         Even with huge improvements to time complexity but distance being off by a huge amount, sorting by
#         X then Y is not enough. <br/>
#         Try incorporating iteration 1 and iteration 2 idea together.
#     </p>
# </div>
# 
# 
# 
# 

# <h2>Second Iteration v2</h2>
# 
# <div>
#     <ol>
#         <li>Data Structure
#             <ul>
#                 <li>Lists of list</li>
#             </ul>
#         </li>
#         <li>Data Algorithms
#             <ul>
#                 <li>While</li>
#                 <li>Binary Search</li>
#             </ul>
#         </li>
#         <li>Methods used:
#             <ul>
#                 <li>Take csv file, load it into the list, using lists of list.</li>
#                 <li>Format: cities_list = [[0,316.836739061509,2202.34070733524] ... 
#                     [19776,1883.0509264308,1318.48663504281]</li>
#                 <li>Before comparing, since I am using Binary Search, I will sorting list according to X 
#                     coordinates then Y coordinates.</li>
#                 <li>Once sorted, I will compare half the list at every loop, until the list has a size of less
#                     than 4. I will then calculate the distance between all the points that are left in the 
#                     list with the previous point and get the minimum distance</li>
#                 <li>The condition to check the size of list can be altered</li>
#                 <li>Remove the point with the shorter distance, add the distance to total distance, and add the 
#                     point to path_shortest_list.</li>
#                 <li>This iteration does not factor in the additional every 10 steps and prime cities
#                     for the calculation.</li>
#             </ul>
#         </li>
#         <li>Time Complexity: O(n log n)</li>
#     </ol>
# </div> 

# In[ ]:


def two_points_dist_i2_v2(p1, x, y):
    """
    Calculate the shortest distance between the p1 and (x,y)
    Return: euclid distance
    """
    return math.sqrt(((p1[1]-x)*(p1[1]-x))+((p1[2]-y)*(p1[2]-y)))

def get_closest_point_i2_v2(lst, x, y):
    """
    Get the closest distance between all the points lst and (x,y)
    Return: list with shortest distance, r_list and shortest_distance
    """
    shortest_dist = math.inf
    r_list = list()

    for point in lst:
        dist = two_points_dist_i2_v2(point, x, y)
        if dist < shortest_dist:
            shortest_dist = dist
            r_list = point
    return r_list,shortest_dist


# In[ ]:


def closest_point_i2_v2(list_x, x, y):
    
    list_to_check = list_x
    

    while True:
        # Get length of x sorted list
        len_x = len(list_to_check)
        
        # If list is less than or equal to 3, manually calculate every point in the list
        # with the previous list so attain more accurate result
        # Do checks for 3....10000
        if len_x <= 10000:
            return get_closest_point_i2_v2(list_to_check, x, y)
        else:
            # Get the middle point with no remainder
            midpoint = len_x // 2

            left_x = list_to_check[:midpoint]
            right_x = list_to_check[midpoint:]

            midpoint_x = list_to_check[midpoint][1]

            if x <= midpoint_x:
                list_to_check = left_x
            else:
                list_to_check = right_x


# In[ ]:


# Calculate the execution time
start = time()

# List of lists
cities_list = []

# Open csv file
with open("../input/cities/cities_10p.csv", "r") as cities_file:
    cities_data = csv.reader(cities_file)
    for row in cities_data:
        cities_list.append([int(row[0]), float(row[1]), float(row[2])])

path_shortest_list = []
total_dist = 0

# Get origin point
x_origin = cities_list[0][1]
y_origin = cities_list[0][2]
city_origin = cities_list.pop(0)
path_shortest_list.append(city_origin)

# Sort list
# 1 = X, 2 = Y
x_sorted_list = sort_list(cities_list,1,2)

closest_list = list()

# Find nearest point to city_origin
(closest_list, d) = closest_point_i2_v2(x_sorted_list, x_origin, y_origin)
total_dist += d
x_sorted_list.remove(closest_list)
path_shortest_list.append(closest_list)

# Find the remaining nearest points
while len(x_sorted_list)>=1:
    (closest_list, d) = closest_point_i2_v2(x_sorted_list, closest_list[1], closest_list[2])
    total_dist += d
    x_sorted_list.remove(closest_list)
    path_shortest_list.append(closest_list)


print("Total Distance: {}".format(total_dist))
print("Shortest path list: {}".format(len(path_shortest_list)))
print("Cities list: {}".format(len(x_sorted_list)))

end = time()

exec_time = end - start
print("Execution time: {}".format(exec_time))

# Plotting route of shortest path taken
fig, ax = plt.subplots(figsize=(15,10))
origin_point = path_shortest_list.pop(0)
plt.scatter(origin_point[1], origin_point[2], s=10, c="red")

prev_point = origin_point

for i in range(len(path_shortest_list)):
    point = path_shortest_list.pop(0)

    xs = point[1]
    ys = point[2]
    ax.plot([prev_point[1], xs], [prev_point[2], ys], c="blue", linewidth=0.5)

    prev_point = point

plt.show()


# 
# <p>
#     The above graph is when condition of len_x &lt;= 10000<br/>
#     The results for varrying different checking length of len_x
# </p>

# In[ ]:


check_time = []
check_distance = []
# when len = 3
check_time.append([3,1.3901])
check_distance.append([3,11567988.2091])
# when len = 10
check_time.append([10,1.4381])
check_distance.append([10,5266561.8749])
# when len = 50
check_time.append([50,1.6832])
check_distance.append([50,1391942.2088])
# when len = 200
check_time.append([200,2.4695])
check_distance.append([200,675473.6778])
# when len = 1000
check_time.append([1000,6.9691])
check_distance.append([1000,512834.0704])
# when len = 2000
check_time.append([2000,11.8996])
check_distance.append([2000,489427.8288])
# when len = 3000
check_time.append([3000,16.9225])
check_distance.append([3000,474417.3427])
# when len = 5000
check_time.append([5000,27.3476])
check_distance.append([5000,476250.5593])
# when len = 8000
check_time.append([8000,39.8815])
check_distance.append([8000,458262.9243])
# when len = 10000
check_time.append([10000,46.4981])
check_distance.append([10000,455349.1702])


# In[ ]:


# Mapping length checked against Time and Distance
fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(15,10))
fig.suptitle("Mapping length check against Time and Distance")

# Plotting for Time
prev_t_point = check_time.pop(0)

for i in range(len(check_time)):
    time_point = check_time.pop(0)

    x_t = time_point[0]
    y_t = time_point[1]
    ax1.plot([prev_t_point[0], x_t], [prev_t_point[1], y_t], c="blue", linewidth=1, label = "Time")

    prev_t_point = time_point

    plt.grid(False)
    ax.autoscale()

    
# Plotting for Distance
prev_d_point = check_distance.pop(0)

for i in range(len(check_distance)):
    dist_point = check_distance.pop(0)

    x_d = dist_point[0]
    y_d = dist_point[1]
    ax2.plot([prev_d_point[0], x_d], [prev_d_point[1], y_d], c="green", linewidth=1, label = "Distance")

    prev_d_point = dist_point

    plt.grid(False)
    ax.autoscale()
ax1.set_ylabel("Time")
ax2.set_ylabel("Distance")
ax2.set_xlabel("Value for Condition of len_x <= value")
plt.show()


# <h3>Thoughts:</h3>
# <div>
#     <p>
#         This time with a slight modification to iteration two version one's code, there has been a big 
#         improvement to the distance calculated. <br/>
#         As I increased the value to check for len_x, the execution time increased because of 
#         the increased number of computation required to compute the shortest distance between 2 points.
#         At the same time, the total distance gap between the first iteration narrowed. <br/>
#         For small values, the time complexity remains at O(n log n) <br/>
#         However for bigger values, the time complexity changes to O(n<sup>2</sup>)
#     </p>
#     <p>
#         Alternative methods: <br/>
#         Refactor code, use list comprehensions<br/>
#         Try using recursive.<br/>
#         Try using linked lists.
#     </p>
# </div>
# 
# 
# 
# 

# <h2>Third Iteration</h2>
# 
# <div>
#     <ol>
#         <li>Data Structure
#             <ul>
#                 <li>Lists of list</li>
#             </ul>
#         </li>
#         <li>Data Algorithms
#             <ul>
#                 <li>For loop</li>
#                 <li>Recursive function</li>
#                 <li>Binary Search</li>
#             </ul>
#         </li>
#         <li>Methods used:
#             <ul>
#                 <li>This iteration is exactly the same as Second Iteration v2 except that
#                     at this iteration, it will be using recursive.</li>
#                 <li>--------------------------------------------------------------------------------------------------------------</li>
#                 <li>Take csv file, load it into the list, using lists of list.</li>
#                 <li>Format: cities_list = [[0,316.836739061509,2202.34070733524] ... 
#                     [19776,1883.0509264308,1318.48663504281]</li>
#                 <li>Before comparing, since I am using Binary Search, I will sorting list according to X 
#                     coordinates then Y coordinates.</li>
#                 <li>Once sorted, I will compare half the list at every loop, until the list has a size of less
#                     than 4. I will then calculate the distance between all the points that are left in the 
#                     list with the previous point and get the minimum distance</li>
#                 <li>The condition to check the size of list can be altered</li>
#                 <li>Remove the point with the shorter distance, add the distance to total distance, and add the 
#                     point to path_shortest_list.</li>
#                 <li>This iteration does not factor in the additional every 10 steps and prime cities
#                     for the calculation.</li>
#             </ul>
#         </li>
#         <li>Time Complexity: O(n log n)</li>
#     </ol>
# </div>
# 

# In[ ]:


def two_points_dist_i3(p1, p2):
    """
    Calculate the shortest distance between the p1 and p2
    Return: euclid distance
    """
    return math.sqrt(((p1[1]-p2[1])*(p1[1]-p2[1]))+((p1[2]-p2[2])*(p1[2]-p2[2])))

def get_closest_point_i3(list_1, list2):
    """
    Get the closest distance between all the points inside list 1 and list 2
    Return: list with shortest distance, r_list and shortest_distance
    """
    shortest_dist = math.inf
    r_list = list()

    for point1 in list_1:
        dist = two_points_dist_i3(point1, list2)
        if dist < shortest_dist:
            shortest_dist = dist
            r_list = point1

    return r_list,shortest_dist


# In[ ]:


def closest_point_i3(list_1, list_prev):
    """
    Run recursively to get the closest point with the previous point
    """
    # Get length of list
    len_list = len(list_1)

    # If list is less than or equal to 3, manually calculate every point in the list
    # with the previous list so attain more accurate result
    if len_list <= 3:
        return get_closest_point_i3(list_1, list_prev)
    
    # Get the middle point with no remainder
    midpoint = len_list // 2

    # Split list into left and right and assign a midpoint
    left_x = list_1[:midpoint]
    right_x = list_1[midpoint:]
    midpoint_x = list_1[midpoint][1]


    if list_prev[1] <= midpoint_x:
        (result_list, dist) = closest_point_i3(left_x, list_prev)
    else:
        (result_list, dist) = closest_point_i3(right_x, list_prev)
    return (result_list, dist)


# In[ ]:


# Calculate the execution time
start = time()

# List of lists
cities_list = []

# Open csv file
with open("../input/cities/cities_10p.csv", "r") as cities_file:
    cities_data = csv.reader(cities_file)
    for row in cities_data:
        cities_list.append([int(row[0]), float(row[1]), float(row[2])])

path_shortest_list = []
total_dist = 0

# Get origin point
x_origin = cities_list[0][1]
y_origin = cities_list[0][2]
city_origin = cities_list.pop(0)
path_shortest_list.append(city_origin)

# Sort list
# 1 = X, 2 = Y
x_sorted_list = sort_list(cities_list,1,2)

# Get the next nearest point from origin city
(result_list, d) = closest_point_i3(x_sorted_list, city_origin)
total_dist += d
x_sorted_list.remove(result_list)
path_shortest_list.append(result_list)

# Get the remaining points from cities_list
for i in range(len(x_sorted_list)):
    (result_list, d) = closest_point_i3(x_sorted_list, result_list)
    total_dist += d
    x_sorted_list.remove(result_list)
    path_shortest_list.append(result_list)

# Loop back to the origin city

print("Total Distance: {}".format(total_dist))
print("Shortest path list: {}".format(len(path_shortest_list)))
print("Cities list: {}".format(len(x_sorted_list)))

end = time()

exec_time = end - start
print("Execution time: {}".format(exec_time))

# Plotting route of shortest path taken
fig, ax = plt.subplots(figsize=(15,10))
origin_point = path_shortest_list.pop(0)
plt.scatter(origin_point[1], origin_point[2], s=10, c="red")

prev_point = origin_point

for i in range(len(path_shortest_list)):
    point = path_shortest_list.pop(0)

    xs = point[1]
    ys = point[2]
    ax.plot([prev_point[1], xs], [prev_point[2], ys], c="blue", linewidth=0.5)

    prev_point = point

plt.show()


# <h3>Thoughts:</h3>
# <div>
#     <p>
#         Above graph is when condition of len_x &lt;= 3 <br/>
#         Used recursive to loop through all the points. The results of execution time was slightly 
#         worse than the forth iteration by 1.16x which was surprising to me as I thought I would have a much 
#         worse result than that.
#     </p>
#     <p>
#         Alternative methods: <br/>
#         Try using linked lists.
#     </p>
# </div>

# <h4>Linked List:</h4>
# <div>
#     <span>class Node</span>
#     <ul>
#         <b>Methods</b>:<br/>
#         <li>__iter__</li>
#         <li>Insert Head</li>
#         <li>Insert Tail</li>
#         <li>Delete Head</li>
#         <li>Delete Tail</li>
#         <li>Insert_between</li>
#         <li>Pop</li>
#         <li>Remove</li>
#         <li>Display</li>
#     </ul>
# 
# </div>

# In[ ]:


# class for linked list
class LinkedListIterator:
    def __init__(self, head):
        self.current = head

    def __iter__(self):
        return self

    def __next__(self):
        if not self.current:
            raise StopIteration
        else:
            item = self.current._data
            self.current = self.current._next
            return item

class LinkedList:
    class Node:
        def __init__(self, val, nxt):
            self._data = val
            self._next = nxt

    def __init__(self):
        #create empty head and tail here
        self._head = self.Node(None, None)
        self._tail = self.Node(None, None)
        self._size = 0

    def __iter__(self):
        """
        Generate a forward iteration of the elements of the list
        """
        return LinkedListIterator(self._head)

    def insert_head(self, val):
        new_node = self.Node(val, None)
        if self._size == 0: #i.e first Node
            self._head = new_node
            self._tail = new_node
        else:
            new_node._next = self._head #new node now is poinint to old head
            self._head = new_node #change the head to the beginning of the list

        self._size += 1

    def insert_tail(self,val): #inserts node at the end of the list
        new_node = self.Node(val, None)
        if self._size == 0: #i.e first Node
            self._head = new_node
            self._tail = new_node
        else: #at least one element in the list
            self._tail._next = new_node
            self._tail = new_node
        self._size += 1

    def delete_head(self):
        if self._size == 0:
            print("Deletion error: List empty")
            return
        #at least one node in the list
        cur = self._head #take backup of head
        self._head = self._head._next #change head to point to its next neighbour
        cur._next = None #break the connection of the old head from rest of the list
        self._size -= 1

    def delete_tail(self):
        if self._size == 0:
            print("Deletion error: List empty")
            return
        #at least one node in the list
        cur = self._head
        while cur._next != self._tail:
            cur = cur._next
        # tail = cur._next
        self._tail = cur
        self._tail._next = None
        self._size -= 1

    def insert_between(self, pos, val):
        """
        Inserts a value at position pos
        """
        if self._size == 0:
            print("Insertion error: List empty")
            return
        elif self._size < 3:
            print("List has only 2 elements")
            return
        elif pos >= self._size:
            print("Index error: Pos > List")
            return
        cur = self._head
        for i in range(pos-1):
            # prev = cur
            cur = cur._next
        nxt = cur._next
        new_node = self.Node(val, nxt)
        cur._next = new_node
        self._size += 1

    def pop(self, pos):
        """
        Deletes and returns current data at pos
        """
        if self._size == 0:
            print("Insertion error: List empty")
            return
        elif pos >= self._size:
            print("Index error: Pos > List")
            return
        cur = self._head
        # Check for if pos = 0
        if pos == 0:
            self._head = cur._next
            cur._next = None
            self._size -= 1
            return cur._data
        for i in range(pos-2):
            cur = cur._next
        prev = cur
        target = cur._next
        nxt = target._next
        target._next = None
        prev._next = nxt
        self._size -= 1
        return target._data
    
    def remove(self, lst):
        """
        Deletes and returns current data at pos
        """
        if self._size == 0:
            print("Insertion error: List empty")
            return
        cur = self._head
        if self._head._data == lst:
            prev = cur
            self._head = cur._next
            prev._next = None
            self._size -= 1
            return
        prev = None
        # Check if the next point is equal to list
        try:
            while cur._data != lst:
                prev = cur
                cur = cur._next
            # if next point is equal, remove it
            if cur._data == lst:
                to_remove = cur
                prev._next = cur._next
                to_remove._next = None
                self._size -= 1
                return
        except Exception:
            print("List not found")
            

    #display contents of the linked list
    def display(self):
        #Traverse the list here
        print("\nLinked list num of nodes:", self._size)
        if self._size == 0:
            print("\nList empty")
            return
        cur = self._head
        while cur != None:
            print("Node data: ", cur._data)
            cur = cur._next


# <h2>Fourth Iteration</h2>
# 
# <div>
#     <ol>
#         <li>Data Structure
#             <ul>
#                 <li>Linked List</li>
#                 <li>Deque</li>
#             </ul>
#         </li>
#         <li>Data Algorithms
#             <ul>
#                 <li>While and For loop</li>
#                 <li>Linked list - insert/delete/search</li>
#             </ul>
#         </li>
#         <li>
#             Methods used:
#             <ul>
#                 <li>This iteration is similiar to the First Iteration except that
#                     at this iteration, it is using Linked List.
#                 <li>Node._data = [0,316.836739061509,2202.34070733524]</li>
#                 <li>Every Node is linked via a _next pointer</li>
#                 <li>Each Node's data can be accessed through _data</li>
#                 <li>No sorting used</li>
#                 <li>Take csv file, load it into the Linked List</li>
#                 <li>Compare every point with other points in the linked list, and calculate the distance between two 
#                     points.</li>
#                 <li>Remove the point with the shortest distance, add the distance to total distance, and add the 
#                     point to shortest_path_list.</li>
#             </ul>
#         </li>
#         <li>Time Complexity: O(n<sup>2</sup>)</li>
#     </ol>
# </div> 

# In[ ]:


# Calculate the execution time
start = time()

# Create LinkedList object
cities_list = LinkedList()

# Load csv file into linked list, storing a list for each Node
with open("../input/cities/cities_10p.csv", "r") as cities_file:
    cities_data = csv.reader(cities_file)
    for row in cities_data:
        cities_list.insert_tail([int(row[0]), float(row[1]), float(row[2])])

# Create deque object
shortest_path_list = deque()
total_dist = 0

# Size of Linked List
total_cities = cities_list._size
print("Total Cities:", total_cities)

# Get origin point
city_origin = cities_list.pop(0)
shortest_path_list.append(city_origin)

# Initiating variables
shortest_dist = math.inf
result_list = []
step_counter = 0

# Get point nearest to city_origin
for point in cities_list:
    x_index = point[1]
    y_index = point[2]
    eucl_dist = math.sqrt(((x_index-city_origin[1])*(x_index-city_origin[1]))+((y_index-city_origin[2])*(y_index-city_origin[2])))

    if eucl_dist < shortest_dist:
        shortest_dist = eucl_dist
        result_list = point
total_dist += shortest_dist
cities_list.remove(result_list)
shortest_path_list.append(result_list)
step_counter += 1

# Get the remaining points from cities_list
while cities_list._size >= 1:
    if cities_list._size%1000 == 0:
            print("Cities left: {}".format(cities_list._size))
    dist = math.inf
    r_list = []
    for point in cities_list:
        x_index = point[1]
        y_index = point[2]
        eucl_dist = math.sqrt(((x_index-result_list[1])*(x_index-result_list[1]))+((y_index-result_list[2])*(y_index-result_list[2])))

        # Check: Every 10th step (stepNumber % 10 == 0) is 10% more lengthy unless coming from a prime CityId.
        if (step_counter % 10 == 0) and not isprime(point[0]):
            eucl_dist = eucl_dist*1.1

        if eucl_dist < dist:
            dist = eucl_dist
            r_list = point

    step_counter += 1
    total_dist += dist
    result_list = r_list
    cities_list.remove(r_list)
    shortest_path_list.append(r_list)

# Get distance back to origin_city
eucl_dist = math.sqrt(((city_origin[1]-result_list[1])*(city_origin[1]-result_list[1]))+((city_origin[2]-result_list[2])*(city_origin[2]-result_list[2])))
total_dist += eucl_dist
shortest_path_list.append(city_origin)

print("Total Distance: {}".format(total_dist))
print("Shortest path list: {}".format(len(list(shortest_path_list))))
print("Cities list: {}".format(cities_list._size))

end = time()

exec_time = end - start
print("Execution time: {}".format(exec_time))

# Plotting route of shortest path taken
fig, ax = plt.subplots(figsize=(15,10))
origin_point = shortest_path_list.popleft()
plt.scatter(origin_point[1], origin_point[2], s=10, c="red")

prev_point = origin_point

for i in range(len(list(shortest_path_list))):
    point = shortest_path_list.popleft()

    xs = point[1]
    ys = point[2]
    ax.plot([prev_point[1], xs], [prev_point[2], ys], c="blue", linewidth=0.5)

    prev_point = point

plt.show()


# <h3>Thoughts:</h3>
# <div>
#     <p>
#         Expected poorer results due to linked list not being as efficient as python's list. However, based off the
#         same logic as the first iteration, it has a better result.
#     </p>
#     <p>
#         Alternative methods: <br/>
#         Try using linked lists with Recursive and Binary search
#     </p>
# </div>

# <h2>Fifth Iteration</h2>
# 
# <div>
#     <ol>
#         <li>Data Structure
#             <ul>
#                 <li>Linked List</li>
#                 <li>Deque</li>
#             </ul>
#         </li>
#         <li>Data Algorithms
#             <ul>
#                 <li>For loop</li>
#                 <li>Linked list - insert/delete/search</li>
#                 <li>Recursive function</li>
#                 <li>Binary Search</li>
#             </ul>
#         </li>
#         <li>
#             Methods used:
#             <ul>
#                 <li>This iteration is similiar to the Second Iteration v2 except that
#                     in this iteration, it is using Linked List.
#                 <li>Node._data = [0,316.836739061509,2202.34070733524]</li>
#                 <li>Every Node is linked via a _next pointer</li>
#                 <li>Each Node's data can be accessed through _data</li>
#                 <li>--------------------------------------------------------------------------------------------------------------</li>
#                 <li>Take csv file, load it into the list, using lists of list.</li>
#                 <li>Format: cities_list = [[0,316.836739061509,2202.34070733524] ... 
#                     [19776,1883.0509264308,1318.48663504281]</li>
#                 <li>Before comparing, since I am using Binary Search, I will sorting list according to X 
#                     coordinates then Y coordinates.</li>
#                 <li>Once sorted, I will append the sorted cities_list into c_linked_list by using the 
#                     insert_tail method.</li>
#                 <li>I will compare half the list at every loop, until the list has a size of less
#                     than the condition fulfilled. I will then calculate the distance between 
#                     all the points that are left in the list with the previous point and get 
#                     the minimum distance</li>
#                 <li>The condition to check the size of list can be altered</li>
#                 <li>Remove the point with the shorter distance, add the distance to total distance, and add the 
#                     point to path_shortest_list.</li>
#             </ul>
#         </li>
#         <li>Time Complexity: O(n<sup>2</sup> log n)</li>
#     </ol>
# </div>

# In[ ]:


def two_points_dist_i5(p1, p2):
    """
    Calculate the shortest distance between the p1 and p2
    Return: euclid distance
    """
    return math.sqrt(((p1[1]-p2[1])*(p1[1]-p2[1]))+((p1[2]-p2[2])*(p1[2]-p2[2])))


# In[ ]:


def get_closest_point_i5(list_1, list2):
    """
    Get the closest distance between all the points inside list 1 and list 2
    Return: list with shortest distance, r_list and shortest_distance
    """
    shortest_dist = math.inf
    r_list = list()

    for point1 in list_1:
        dist = two_points_dist_i5(point1, list2)
        # Check: Every 10th step (stepNumber % 10 == 0) is 10% more lengthy unless coming from a prime CityId.
        if (step_counter % 10 == 0) and not isprime(point1[0]):
            dist = dist*1.1
        if dist < shortest_dist:
            shortest_dist = dist
            r_list = point1

    return r_list,shortest_dist


# In[ ]:


def closest_point_i5(c_linked_list, list_prev):
    """
    Run recursively to get the closest point with the previous point
    """
    # Get length of list
    len_list = c_linked_list._size
    counter = 0
    # If list is less than 10, manually calculate every point in the list
    # with the previous list so attain more accurate result
    if len_list <= 10000:
        return get_closest_point_i5(c_linked_list, list_prev)
    
    # Get the middle point with no remainder
    midpoint = len_list // 2

    # Split list into left and right and assign a midpoint
    left_x = LinkedList()
    right_x = LinkedList()
    midpoint_list = []
    
    for point in c_linked_list:
        if counter < midpoint:
            left_x.insert_tail(point)
            counter += 1
        elif counter == midpoint:
            midpoint_list = point
            right_x.insert_tail(point)
            counter += 1
        else:
            right_x.insert_tail(point)

    if list_prev[1] <= midpoint_list[1]:
        (result_list, dist) = closest_point_i5(left_x, list_prev)
    else:
        (result_list, dist) = closest_point_i5(right_x, list_prev)
    return (result_list, dist)


# In[ ]:


# Calculate the execution time
start = time()

step_counter = 1

cities_list = []

# Create LinkedList object
c_linked_list = LinkedList()

# Load csv file into linked list, storing a list for each row of record
with open("../input/cities/cities_10p.csv", "r") as cities_file:
    cities_data = csv.reader(cities_file)
    for row in cities_data:
        cities_list.append([int(row[0]), float(row[1]), float(row[2])])

# Create deque object
shortest_path_list = deque()
total_dist = 0

# Get origin city
city_origin = cities_list.pop(0)
shortest_path_list.append(city_origin)

# Sort list
# 1 = X, 2 = Y
x_sorted_list = sort_list(cities_list,1,2)

# Add x_sorted_list into c_linked_list
[c_linked_list.insert_tail(row) for row in x_sorted_list]

# Size of Linked List, adding city_origin
total_cities = c_linked_list._size + 1 
print("Total Cities:", total_cities)

# Get the next nearest point from origin city
(result_list, d) = closest_point_i5(c_linked_list, city_origin)
total_dist += d
c_linked_list.remove(result_list)
shortest_path_list.append(result_list)

# Get the remaining points from cities_list
for i in range(c_linked_list._size):
    if c_linked_list._size%1000 == 0:
        print("Cities left: {}".format(c_linked_list._size))
    (result_list, d) = closest_point_i5(c_linked_list, result_list)
    total_dist += d
    c_linked_list.remove(result_list)
    shortest_path_list.append(result_list)
    step_counter += 1

# Adding the last stop, routing back to city_origin point
final_dist = two_points_dist_i5(result_list, city_origin)
total_dist += final_dist
shortest_path_list.append(city_origin)


print("Total Distance: {}".format(total_dist))
print("Shortest path list: {}".format(len(list(shortest_path_list))))
print("Cities list: {}".format(c_linked_list._size))

end = time()
exec_time = end - start
print("Execution time: {}".format(exec_time))

# Plotting route of shortest path taken
fig, ax = plt.subplots(figsize=(15,10))
origin_point = shortest_path_list.popleft()
plt.scatter(origin_point[1], origin_point[2], s=10, c="red")

prev_point = origin_point

for i in range(len(list(shortest_path_list))):
    point = shortest_path_list.popleft()

    xs = point[1]
    ys = point[2]
    ax.plot([prev_point[1], xs], [prev_point[2], ys], c="blue", linewidth=0.5)

    prev_point = point

plt.show()


# <h3>Thoughts:</h3>
# <div>
#     <p>
#         This method out of all the iterations thus far is performing the worst in terms of time complexity. 
#         Also, from the Second Iteration v2 when the condition of len x &lt;= 10000 seemed to perform the best 
#         out of all the attempts, still performed poorer than the Fourth Iteration when using linked list 
#         with searching next nearest point.<br/>
#         There are also limitations to Linked List when using Binary Search. Without touching the original
#         linked list, upon split at every recursive loop, I have to create 2 new Linked Lists with a for loop
#         -- left_list and right_list. This causes the massive spike in time complexity, from O(n log n) to 
#         O(n<sup>2</sup> log n)
#     </p>
#     <p>
#         Alternative methods: <br/>
#         It might just be better to go back to the Fourth Iteration, where there is better distance accuracy and 
#         a lower time complexity of O(n<sup>2</sup>).
#     </p>
# </div>
# 
# 
# 
# 

# <h4>Queue</h4>
# <div>
#     <ul>
#         <b>Methods</b>:<br/>
#         <li>__init__</li>
#         <li>__len__</li>
#         <li>Enqueue</li>
#         <li>Dequeue</li>
#     </ul>
# </div>

# In[ ]:


# Queue
class Queue:
    def __init__(self):
        """
        Create an empty Queue
        """
        self._data = []
        
    def __len__(self):
        """
        Returns the number of elements in the queue.
        """
        return len(self._data)
    
    def enqueue(self, val):
        """
        Adds the val to the end of the list
        """
        self._data.append(val)
        
    def dequeue(self):
        """
        Removes and the first element of the list
        """
        if len(self._data) == 0:
            print("List is empty")
        return self._data.pop(0)


# <h2>Solution</h2>
# 
# <div>
#     <ol>
#         <li>Data Structure
#             <ul>
#                 <li>Linked List</li>
#                 <li>Queue</li>
#             </ul>
#         </li>
#         <li>Data Algorithms
#             <ul>
#                 <li>Queue - enqueue and dequeue</li>
#                 <li>Linked list - insert/delete/search/pop/remove</li>
#                 <li>For loops</li>
#             </ul>
#         </li>
#         <li>
#             Methods used:
#             <ul>
#                 <li>This Solution uses Linked List and Queue</li>
#                 <li>Inside every Node in the LinkedList stores _data of a point - 
#                     for example: [0,316.836739061509,2202.34070733524]</li>
#                 <li>Every Node is linked to the next Node via a _next pointer</li>
#                 <li>Each Node's data can be accessed through _data</li>
#                 <li>There is no sorting used</li>
#                 <li>The steps are as follow:</li>
#                 <li>Take csv file, load it into the Linked List</li>
#                 <li>At every iteration of the loop, compare the current point with other points in the linked 
#                     list. Calculate the distance between two points and return the smallest distance with the 
#                     point.</li>
#                 <li>Add the returned distance to total distance and remove the point from the Linked List using
#                     the remove method.</li>
#                 <li>Add the same point to shortest_path_list Queue object using the enqueue method.</li>
#                 <li>After every Node is visited exactly once, return back to the origin point</li>
#                 <li>Run through shortest_path_list Queue object, using the dequeue method to plot the graph
#                     of the route of the shortest path taken.</li>
#             </ul>
#         </li>
#         <li>Time Complexity: O(n<sup>2</sup>)</li>
#     </ol>
# </div> 

# In[ ]:


# Calculate the execution time
start = time()

# Create LinkedList object
cities_list = LinkedList()

# Load csv file into Linked List, storing a list for each Node in the Linked List
with open("../input/cities/cities_10p.csv", "r") as cities_file:
    cities_data = csv.reader(cities_file)
    for row in cities_data:
        cities_list.insert_tail([int(row[0]), float(row[1]), float(row[2])])

# Create Queue object
shortest_path_list = Queue()
# Initiating total_dist variable
total_dist = 0

# Display size of Linked List
total_cities = cities_list._size
print("Total Cities:", total_cities)

# Get origin point
city_origin = cities_list.pop(0)
shortest_path_list.enqueue(city_origin)

# Initiating variables
shortest_dist = math.inf
result_list = []
step_counter = 0

# Get point nearest to city_origin
for point in cities_list:
    x_index = point[1]
    y_index = point[2]
    # Finding the euclidean distance between two points 
    eucl_dist = math.sqrt(((x_index-city_origin[1])*(x_index-city_origin[1]))+((y_index-city_origin[2])*(y_index-city_origin[2])))
    
    # If euclideon distance is smaller than shortest_dist, replace it and set the return_list to point
    if eucl_dist < shortest_dist:
        shortest_dist = eucl_dist
        result_list = point
total_dist += shortest_dist
cities_list.remove(result_list)
shortest_path_list.enqueue(result_list)
step_counter += 1

# Get the remaining points from cities_list
while cities_list._size >= 1:
    if cities_list._size%1000 == 0:
            print("Cities left: {}".format(cities_list._size))
    dist = math.inf
    r_list = []
    for point in cities_list:
        x_index = point[1]
        y_index = point[2]
        # Finding the euclidean distance between two points
        eucl_dist = math.sqrt(((x_index-result_list[1])*(x_index-result_list[1]))+((y_index-result_list[2])*(y_index-result_list[2])))

        # Check: Every 10th step (stepNumber % 10 == 0) is 10% more lengthy unless coming from a prime CityId.
        if (step_counter % 10 == 0) and not isprime(point[0]):
            eucl_dist = eucl_dist*1.1
        
        # If euclideon distance is smaller than shortest_dist, replace it and set the return_list to point
        if eucl_dist < dist:
            dist = eucl_dist
            r_list = point

    step_counter += 1
    total_dist += dist
    result_list = r_list
    cities_list.remove(r_list)
    shortest_path_list.enqueue(r_list)

# Get the distance back to origin_city
eucl_dist = math.sqrt(((city_origin[1]-result_list[1])*(city_origin[1]-result_list[1]))+((city_origin[2]-result_list[2])*(city_origin[2]-result_list[2])))
total_dist += eucl_dist
shortest_path_list.enqueue(city_origin)

print("Total Distance: {}".format(total_dist))
print("Shortest path list: {}".format(len(shortest_path_list)))
print("Cities list: {}".format(cities_list._size))

end = time()

exec_time = end - start
print("Execution time: {}".format(exec_time))

# Plotting route of shortest path taken
fig, ax = plt.subplots(figsize=(15,10))
origin_point = shortest_path_list.dequeue()
plt.scatter(origin_point[1], origin_point[2], s=10, c="red")

prev_point = origin_point

for i in range(len(shortest_path_list)):
    point = shortest_path_list.dequeue()

    xs = point[1]
    ys = point[2]
    ax.plot([prev_point[1], xs], [prev_point[2], ys], c="blue", linewidth=0.5)

    prev_point = point

plt.show()


# <div>
#     <p><b>Conclusion</b></p>
#     <p>
#         Overall, a Linked list is not as efficient and versatile as Python's list structure. 
#         There are also other limitations to using Linked Lists for algorithms such as Binary Search where
#         you have to split the Linked List into 2 but unable to do so, resulting in having to create 2 additional
#         Linked List at every iteration of the loop. This addition causes a spike in the time complexity.
#     </p>
#     <p>
#         That being said about Linked Lists with Binary Search, from results, Binary Search is the most time efficient
#         algorithm used in this assignment if you just need to find an exact point. However due to the point have 
#         2 coordinates - x and y, the distance calculated from Binary Search is terrible.
#     </p>
#     <p>
#         Recursive functions compared to using loops are also inefficient when writing algorithms for big data 
#         structure as there will be many more layers of function call and return mechanism which will slow down 
#         the program.
#     </p>
# </div>
