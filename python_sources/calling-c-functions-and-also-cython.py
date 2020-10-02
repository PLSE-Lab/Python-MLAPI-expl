#!/usr/bin/env python
# coding: utf-8

# Using C++ functions in your Code
# ==
# 
# This notebook demonstrates how you can use C++ functions in your code if you need a bit of a speed up (It could speed up your code by 30 times in some cases). I don't know if this will work for competitions or not. You can change the C++ functions and click **Run All** and it should work.
# 
# A comparrison with Cython at the end. It turns out it may be better to use cython  after all instead of messing about with C++! Whatever is best for you. One downside of Cython is it seems like it takes a very long time to compile large files compared to C++. Probably use Cython if you only have a small function to speed up and C++ if you have many functions.

# Tutorial
# ==

# It is important to keep a track of the handles of the libraries you have loaded and then unload them before trying to recompile the libaries. So we have an array to keep track of the libraries you have loaded:

# In[ ]:


#We set the handles to empty only once
if not 'handles' in globals():
    global handles
    handles=[]


# Define a function to unload the libraries so we can recompile them later:

# In[ ]:


import _ctypes
def unloadAllLibs():
    global handles
    for handle in handles:
        _ctypes.dlclose(handle)
    handles=[]


# Create your C++ library functions here:

# In[ ]:


get_ipython().run_cell_magic('writefile', 'myfunc.cpp', '\nextern "C" {\n    \n    struct GRID{\n        int* data;\n        int W;\n        int H;\n    };\n    \n    long sumOfSquares(long x) {\n        long total=0;\n        for(long n=0;n<x;n++){\n           total+=n*n;\n        }\n        return total;\n    }\n    int* doubleElements(int* A, int W, int H){\n        int* B=new int[W*H];\n        for(int x=0;x<W;x++){\n            for(int y=0;y<H;y++){\n                B[W*y+x] = A[W*y+x]*2;\n            }\n        }\n        return B;\n    }\n    \n    GRID addOne(GRID a){\n        for(int n=0; n<a.W * a.H; n++){\n            a.data[n]++;\n        }\n        return a;\n    }\n}')


# Compile the library:

# In[ ]:


get_ipython().system('g++  -shared -o myfunctions.dll myfunc.cpp')


# Now you can call the functions

# In[ ]:


import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer

unloadAllLibs()
lib = ctypes.cdll.LoadLibrary('/kaggle/working/myfunctions.dll')
#Always keep a track of the handles:
handles.append(lib._handle)

mylist = [1,2,3,4,5,6]

#set the output type:
lib.doubleElements.restype = ndpointer(dtype=ctypes.c_int32, shape=[3,2])
lib.sumOfSquares.restype = ctypes.c_int64
#lib.doubleElements.argtypes = [ctypes.POINTER(ctypes.c_int32)] #<--(Not neccessary - the input type is inferred)

#create a C array from the list
clist =  (ctypes.c_int32 * len(mylist)) (*mylist)

#Call the functions
print(lib.sumOfSquares(100))
print(lib.doubleElements( clist , 3, 2 ))

#Alternatively just return a C array:
lib.doubleElements.restype = ctypes.POINTER(ctypes.c_int32)
x = lib.doubleElements( clist , 3, 2 )

#We change the C array back into a list like this:
print([x[i] for i in range(6)])


# Multidimensional Arrays
# ==
# For a 2x2 multidimensional array 
# 
# int\*\*foo(int\*\* x) {..}
# 
# you have for example:

# In[ ]:


#lib.foo.restype = POINTER(POINTER(c_int))
#clist2d = (POINTER(c_int) * 2) ( (c_int * 2) (10,20) , (c_int * 2) (30,40))


# Casting is easier with numpy arrays. To turn an numpy array into a 1D C array use:

# In[ ]:


a = np.array([[2,4],[6,8],[10,12]],dtype=np.int32)
carray = a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
#lib.doubleElements.restype = ndpointer(dtype=ctypes.c_int32, shape=[3,2]) #<--could use this instead of casting
b = lib.doubleElements(carray,3,2)
#get back the array:
np.ctypeslib.as_array(b,shape=[3,2])


# Clean up:
# 

# Structs
# ==
# It is sometimes useful to put things in structs. We just have to make the same Structure in Python:

# In[ ]:


class GRID(ctypes.Structure):
    _fields_=[("data",ctypes.POINTER(ctypes.c_int32)),("W",ctypes.c_int32),("H",ctypes.c_int32 )]
 
lib.addOne.restype = GRID;
    
A = GRID(carray,3,2)
B = lib.addOne(A)
np.ctypeslib.as_array(B.data,shape=[B.W,B.H])


# Tips
# ==
# For fastest performance, keep your arrays as C arrays rather than convert back and forth between numpy arrays and C arrays. Put loops in C++ rather than calling C++ functions in a loop. Use Python for things like visualisation and printing results. If you have a lot of C++ code, it may be useful to put this in another file while working, and only add it to the notebook when submitting to competitions. (I have found ndpointer doesn't work on all systems but seems to work on Kaggle.)

# Thankyou
# ==
# I hope you enjoyed this notebook

# Cython
# ==
# This is way of compiling python down to C. But it only works if you have the Cython extension. It works on Kaggle but maybe not offline unless you install the extension. 

# In[ ]:


get_ipython().run_line_magic('load_ext', 'Cython')


# Write your cython code. It is like python but with types. 

# In[ ]:


get_ipython().run_cell_magic('cython', '', 'cpdef long sumOfSquares(long x):\n    cdef long total=0\n    cdef long n=0\n    for n in range(x):\n        total+=n*n\n    return total')


# We'll compare this with a function written in pure python:

# In[ ]:


def sumOfSquaresPython(x):
    total=0
    for n in range(x):
        total+=n*n
    return total


# Time comparisons (obviously the results are wrong since the numbers are bigger than machine precission):

# In[ ]:


import time
L=1000000
#Commented out pure python because it's too slow
'''
start=time.time()
z=sumOfSquaresPython(L)
end=time.time()
print(f'Pure python {end-start}    {z}')
'''

start=time.time()
z=lib.sumOfSquares(L)
end=time.time()
print(f'Pure C++ {end-start}    {z}')

start=time.time()
z=sumOfSquares(L)
end=time.time()
print(f'Cython {end-start}    {z}')


# In[ ]:


#Lets unload the libraries just for good measure
unloadAllLibs()

