# This Python 3 environment comes with many helpful analytics libraries installed
# Code Written By Darien Schettler
# May 31, 2018

# This is code I wrote when I was going through literature related to numpy
# It is the first in a series I will be doing covering numpy, pandas, scipy, and matplotlib
# I hope this is helpful to people

# You can run the code as it stands now and it will explain what is happening and the code to do it
# As such you can get an understanding of the commands just from running it without having to scroll through

# It has multiple "please press enter to continue..." input commands to help break up the learning
# This method of stopping doesn't seem to work well with the kernel system on kaggle as such they are commented out...
# Enjoy

import numpy as np

print("\nNumPy is a fundamental Python package for scientific computing.\n"
      "It contains among other things:\n"
      "• A powerful N-dimensional array object\n"
      "• Tools for integrating C/C++ and Fortran code\n"
      "• Useful linear algebra, Fourier transform, and random number capabilities\n\n")

#input("\n\nPress Enter to continue...")

print("\nUsing the np.array command we can specify the values to include and the datatype(dtype)\n")
# 1D array
a = [1, 2, 3, 4]
a1 = np.array(a, dtype=np.int16)
print("This is a 1D array ")
print(a1)
print("Data type is %s" % a1.dtype)

# 2D array
a = [[1, 2], [3, 4]]
a1 = np.array(a, dtype=np.float32)
print("\nThis is a 2d array ")
print(a1)
print("Data type is %s" % a1.dtype)

#input("\n\nPress Enter to continue...")

# The type of acceptable datatypes
dataTypes = np.array(['bool', 'int8', 'uint8', 'int16', 'uint16', 'int64', 'uint64',
                      'float16', 'float32', 'float64', 'complex64', 'complex128', 'complex256', 'string'],
                     dtype=np.str)
print("\n\nThe following data types are accepted:\n")
print(dataTypes)

#input("\n\nPress Enter to continue...")

# Simple array information functions
print("\n\nLet's look at the first array a = [1, 2, 3, 4] and print some useful information\n")

a = [1, 2, 3, 4]
a1 = np.array(a, dtype=np.int16)

print("Array A")
print(a1)

print("\nThe dimension of the array is %d acquired via a1.ndim" % a1.ndim)
print("The size of array dimension %d acquired via a1.shape" % a1.shape)
print("The length of the array %d acquired via len(a1)" % len(a1))
print("The data type of the array %s acquired via a1.dtype" % a1.dtype)

#input("\n\nPress Enter to continue...")

# Arithmetic using numpy
print("Let's introduce another array b which will also be a numpy array from 4 to 1\n")

b = [4, 3, 2, 1]
b1 = np.array(b, dtype=np.int16)

print("\nArray B")
print(b1)

print("\nNumpy lets us add, subtract, multiply and divide (not normally allowed between lists in python)\n")

print("\n---Addition---")
print(a1 + b1)
#input("\n\nPress Enter to continue...")

print("\n---Subtraction---")
print(a1 - b1)
#input("\n\nPress Enter to continue...")

print("\n---Multiplication---")
print(a1 * b1)
#input("\n\nPress Enter to continue...")

print("\n---Division---")
print(a1 / b1)
#input("\n\nPress Enter to continue...")

# Special Functions Using Numpy
print("\n\nUsing some special functions we can...\n")
print("\nMake a Numpy array from 2 to 20 counting by 3s... np.arrage\n")
print(np.arange(2, 20, 3))
print("\nMake a Numpy array from 2 to 20 of evenly spaced numbers (length 3)... np.linspace\n")
print(np.linspace(2, 20, 3))
print("\nMake a Numpy array from 2 to 20 counting by 3s... np.random(.seed and .randn)\n")

# the number entered in the brackets is the seed if left empty it will always return different numbers
np.random.seed()

# number in the bracket is size of array, will auto generate between 0 and 1
print(np.random.randn(10) * 4 + 3)

#input("\n\nPress Enter to continue...")

# More complicated arithmetic
print("\nWe can also use more complicated arithmetic\n")
print("We will perform to the e^, sqrt, sin, and cos to the original a array\n")
print("Original A Array")
print(a1)
#input("\n\nPress Enter to continue...")

print("\n---e^ Power Using np.exp---")
print(np.exp([a]))
#input("\n\nPress Enter to continue...")

print("\n---Square Root using np.sqrt---")
print(np.sqrt([a]))
#input("\n\nPress Enter to continue...")

print("\n---Sin using np.sin (note np also has a pi function at np.pi)---")
print(np.sin([a]))
#input("\n\nPress Enter to continue...")

print("\n---Cos using np.cos (note np also has a pi function at np.pi)---")
print(np.cos([a]))
#input("\n\nPress Enter to continue...")

# How to select specific array elements
print("\n\nNow we learn how to select specific array elements from a new array"
      " of a = np.arange(2,20,1.5)\n")

a = np.arange(2, 20, 1.5)

print("now we will grab various portions of the array with the commands dictated prior\n\n")

print("New A Array")
print(a)

#input("\n\nPress Enter to continue...")
print("\n\nNOTE --> If the array was not 1D we would specifiy a[row:row,column:column]\n\n")

print("---Print the first 3 values using a[:3]---")
print(a[:3])
#input("\n\nPress Enter to continue...")

print("\n---Print the 3rd value to the last value using a[3:]---")
print(a[3:])
#input("\n\nPress Enter to continue...")

print("\n---Print the all values using a[:]---")
print(a[:])
#input("\n\nPress Enter to continue...")

print("\n---Print the 2nd to the 7th values using a[2:7]---")
print(a[2:7])
#input("\n\nPress Enter to continue...")

print("\n---Print the 3rd value from the end working backwords a[-3]---")
print(a[-3])
#input("\n\nPress Enter to continue...")

# How to select specific array elements
print("\n\nNow we learn how to filter based on certain array elements using new array a = np.arrange(-10,10,2)")
print("\nThis is our new array A\n")
a = np.arange(-10, 10, 2)
print(a)

#input("\n\nPress Enter to continue...")

print("\n---Print all values greater than 0 using a[a>0]---")
print(a[a > 0])
#input("\n\nPress Enter to continue...")

print("\n---{CHALLENGE} Print all values  between 0 and 100 NOT divisible by 3 {CHALLENGE}---")
print("\n\nNew A Array")
a = np.arange(1, 100, 1)
print(a)

#input("\n\nPress Enter to continue to answer...")

print("\nWe use the command: a[a%3 != 0] i.e. req. remainder when number divided by 3")
print(a[a % 3 != 0])

#input("\n\nPress Enter to continue...")

print("\nNow we look at transforming and flattening of arrays using a = np.arange(24)")
print("\nNew A Array")
# simple way to list the number from 0 < number in brackets (NUMBER IS NOT INCLUDED)
a = np.arange(24)
print(a)
#input("\n\nPress Enter to continue...")

print("\n---Reshape the list into a 6 by 4 matrix using a.reshape(6,4)---")
print(a.reshape(6, 4))
#input("\n\nPress Enter to continue...")

print("\n---Reshape the list into a 6 by ? matrix using a.reshape(6,-1)---")
print(a.reshape(6, -1))
#input("\n\nPress Enter to continue...")

print("\n---Reshape the list into a ? by 6 matrix using a.reshape(-1,6)---")
print(a.reshape(-1, 6))
#input("\n\nPress Enter to continue...")

print("\n---Now we take a 4x6 array as previously shown and flatten it using a.ravel()")
print(a.reshape(-1, 6).ravel())

print("\nNow we look at statistical operations using a = np.random.randn(50)*2+5, "
      "and the reshape command a.reshape(10,5)")

print("\nNew A Array")
# simple way to list the number from 0 < number in brackets (NUMBER IS NOT INCLUDED)
a = np.random.randn(50) * 2 + 5
a = a.reshape(10, 5)
print(a)
#input("\n\nPress Enter to continue...")

print("\n---Find the overall mean of the matrix simply using np.mean(a)---")
print(np.mean(a))
#input("\n\nPress Enter to continue...")

print("\n---Find the mean of the rows of the matrix using np.mean(a, axis=1)---")
print(np.mean(a, axis=1))
#input("\n\nPress Enter to continue...")

print("\n---Find the mean of the columns of the matrix using np.mean(a, axis=0)---")
print(np.mean(a, axis=0))
#input("\n\nPress Enter to continue...")
# could also use a.mean()

print("\n---Find the standard deviation of the matrix using np.std(a)---")
print(np.std(a))
#input("\n\nPress Enter to continue...")
# could also use a.std()

print("\n---Find the variance of the matrix using np.var(a)---")
print(np.var(a))
#input("\n\nPress Enter to continue...")
# could also use a.var()

print("\n---{CHALLENGE} 1. List out all the non-negative elements in the list "
      "[2,3,-2,-3,2,6,-3,4,-3,5,-6,5,5,-2]\n2, Determine the number of "
      "non-negative elements {CHALLENGE}---")
print("\n\nNew A Array")
a = np.array([2, 3, -2, -3, 2, 6, -3, 4, -3, 5, -6, 5, 5, -2])
print(a)
#input("\n\nPress Enter to continue to answer...")

print("\nWe use the command: a[a>=0] and the len command to determine the new array size")
print("\nArray A without the negative elements...\n")
print(a[a >= 0])
print("\nThe length of the above positive array...\n")
print(len(a[a >= 0]))