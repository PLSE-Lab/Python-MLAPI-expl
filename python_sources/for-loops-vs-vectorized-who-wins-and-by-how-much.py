#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# So we all have heard of matrices and vectors, and we use numpy all the time. But what is the importance of it. Why can't we just use a for-loop instead of using the numpy syntax, np.(function_name here). The difference is in performance.
# 
# In this notebook, I will be running only dot products and element wise multiplication. Feel free to fork the notebook and try different input sizes, or different commonly used operations.
# 
# Vectorized implementations (numpy) are much faster and more efficient as compared to for-loops. To really see HOW large the difference is, let's try some simple operations used in most machine learnign algorithms (especially deep learning).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        pass
        
print("Finished Importing Libraries")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# So, first let's create 2 new vectors each with 1 million rows, all with random numbers.

# In[ ]:


v1 = np.random.rand(1000000, 1)
v2 = np.random.rand(1000000, 1)


# # Multiplication by a Scalar

# Scaling a vector by a constant is important in processes like normalization. Quite often, there are for-loop implementations used, which take unnecessary amounts of time.
# 
# Let's look at the difference.

# In[ ]:


# Scaling Vector - For loop
start = time.process_time()
v1_scaled = np.zeros((1000000, 1))

for i in range(len(v1)):
    v1_scaled[i] = 2 * v1[i]

end = time.process_time()
    
print("Scaling vector Answer = " + str(v1_scaled))
print("Time taken = " + str(1000*(end - start)) + " ms")  


# In[ ]:


#Scaling Vector - Vectorized
start = time.process_time()
v1_scaled = np.zeros((1000000, 1))

v1_scaled = 2 * v1

end = time.process_time()
    
print("Scaling vector Answer = " + str(v1_scaled))
print("Time taken = " + str(1000*(end - start)) + " ms")  


# Wow, vectorization is almost 300 times faster than for-loops. That is impressive.

# # Dot Products

# Now we are going to perform the dot product of the two vectors. This is the formula used to multiply the different values of the features of a training example and the weights that the model has learned. In deep learning, this process is repeated lots and lots of times. Even simple algorithms like Linear Regression make use of this a lot.
# 
# If you are not familiar with the formula, it is basically multiplying the two vectors element wise and then summing all of the elements. Here is the for loop implementation.

# In[ ]:


# Dot product For loop
start = time.process_time()
product = 0

for i in range(len(v1)):
    product += v1[i] * v2[i]

end = time.process_time()

print("Dot product Answer = " + str(product))
print("Time taken = " + str(1000*(end - start)) + " ms")


# Now the vectorized implementation.

# In[ ]:


#Dot product Vectorized
start = time.process_time()
product = 0

product = np.dot(v1.T, v2)

end = time.process_time()

print("Dot product Answer = " + str(product))
print("Time taken = " + str(1000*(end - start)) + " ms")


# Wow, vectorized implementation is roughtly 600 times faster! (It may change slightly, but overall it should roughly be the same).

# # Element Wise multiplication

# Another important operation is just multiplying the two vectors element wise. Let's see the difference between the for-loop and vectorized version for this.

# In[ ]:


#Element wise mutliplication For loop
start = time.process_time()

answer = np.zeros((1000000, 1))

for i in range(len(v1)):
    answer[i] = v1[i] * v2[i]
    
end = time.process_time()

print("Element Wise answer = " + str(answer))
print("Time Taken = " + str(1000*(end - start)) + " ms")


# In[ ]:


#Element wise multiplication Vectorized
start = time.process_time()

answer = np.zeros((1000000, 1))

answer = v1 * v2

end = time.process_time()

print("Element Wise answer = " + str(answer))
print("Time Taken = " + str(1000*(end - start)) + " ms")


# Wow, vectorized implementation is almost 500 times faster! 
# Huge boost in performance.
# 

# # Element Wise Matrix Multiplication
# 
# Now let's investigate element wise matrix multiplication. Since this will have a complexity of O(n2), I will use smaller matrix sizes (just enough to see a good difference, but not too large).

# In[ ]:


#Element wise matrix multiplication For loop

m1 = np.random.rand(1000, 1000)
m2 = np.random.rand(1000, 1000)
answer = np.zeros((1000, 1000))

start = time.process_time()

for i in range(m1.shape[0]):
    for j in range(m1.shape[1]):
        answer[i, j] = m1[i, j] * m2[i, j]
    
end = time.process_time()

print("Element Wise Matrix answer = " + str(answer))
print("Time Taken = " + str(1000*(end - start)) + " ms")


# In[ ]:


#Element wise matrix multiplication Vectorized
answer = np.zeros((1000, 1000))

start = time.process_time()

answer = np.multiply(m1, m2)

end = time.process_time()

print("Element Wise Matrix answer = " + str(answer))
print("Time Taken = " + str(1000*(end - start)) + " ms")


# Wow, Numpy is almost 370 times faster than for-loops. This is going to save lots of time. Imagine having hundreds of features and millions of rows. Using a for-loop would unnecessariy kill your computer.

# # Time-complexity Plot
# 
# Now let's try and plot the performance over different sizes of input. This way, we can actually compare how each method's time complexity grows.

# In[ ]:


sizes = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
complexity = pd.DataFrame(columns=['sizes', 'for_loop', 'numpy'])
complexity['sizes'] = sizes


# I will be using a for-loop to iterate through all of the input sizes (excuse me).

# In[ ]:


for_loops = []
numpy = []

for size in sizes:
    v1 = np.random.rand(size, 1)
    v2 = np.random.rand(size, 1)
    
    #For loop implementation
    start = time.process_time()
    product = 0

    for i in range(len(v1)):
        product += v1[i] * v2[i]

    end = time.process_time()
    
    for_loops.append(1000*(end-start))
    
    #Vectorized implementation
    
    start = time.process_time()
    product = 0

    product = np.dot(v1.T, v2)

    end = time.process_time()
    numpy.append(1000*(end - start))
    


# In[ ]:


complexity['for_loops'] = for_loops
complexity['numpy'] = numpy


# In[ ]:


plt.plot(complexity['sizes'], complexity['for_loops'])
plt.plot(complexity['sizes'], complexity['numpy'])

plt.xscale(value='log')
plt.xlabel("Size of input")
plt.ylabel("Time taken in ms")
plt.legend(['for loop', 'numpy'])
plt.show()


# My God!! Look at the way the for-loop implementation grows and compare that with numpy. Numpy is almost flat, but for loop grows drastically (by the way, the graph actually grows linearly, I have just made the x-axis logarithmic as it makes reading the plot a lot easier.
# 
# 
# # Summary
# 
# If you are curious, for-loop looks like it is following the time-complexity of O(N). Numpy however, is basically following a complexity of O(1). 
# 
# Most real-world datasets used for ML/AI use millions or billions of rows. As you can see from the plot, using vectorized implementations can save a lot of time.
# 
# 
# 
# So I just have 1 final request: Please use vectorized implementations using numpy or pandas whenever you can. Try to avoid using for-loops, it will save you a lot of time (and money if you use servers).
# 
# 
# I hoped you like this simple analysis of why you should try using numpy (make friends with the library. It will reward you).
