# Dealing with NaN and None in NumPy and Pandas.

import numpy as np
import pandas as pd

# NAN and None in NumPy

# NONE : Python's Singleton Object used for Missing Data.

# As a Python Object None cannot be used in any Arbitrary NumPy / Pandas Array.

# But None can be used in Arrays with Data Type Object.

Array = np.array([1, 2, 3, None, 4])
print(f'Array : {Array}')
print(f'Data Type : {Array.dtype}')
print(f'Type of Array : {type(Array)}') 
print(f'Type of Elements inside Array : {type(Array[0])}') 

# Unlike other Objects we cannot perform Aggregations like sum() or min() across an Array with a None Value, we will generally get an Error.

# Addition between an Integer and None is Undefined.



# NAN : Missing Numerical Data

# NAN is a Special value which is part of the IEEE Floating Point Specification.

# NAN is a Floating Point Value, there is no Equivalent NAN values for Integers, Strings and other Data Types.

# NAN is bit like a Virus, it Infects any other Object it Touches, Means the Arithmetic with NAN will give another NAN.

print(1 + np.NAN)
print(0 * np.NAN)

# This Means that Aggregates over the Values are well defined (They don't Result in an Error) but not always Useful.

Array = np.array([1, 2, 3, np.NAN])
print(f'Sum : {Array.sum()}')
print(f'Min : {Array.max()}')
print(f'Max : {Array.min()}')
print(f'Data Type : {Array.dtype}')

# NumPy does provides some Special Aggregations ( Functions ) that will ignore the Missing Values

print(f'Sum : {np.nansum(Array)}')
print(f'Min : {np.nanmin(Array)}')
print(f'Max : {np.nanmax(Array)}')

# In NumPy 'NAN', 'NaN' and 'nan' are Same things

print(np.NAN is np.NaN is np.nan)



# NAN and None in Pandas

# Pandas handles both Interchangeably, Converting them where Appropriate.

# Pandas Automatically Typecasts when Missing Values (NaN) and Null Values are present

# The Moment we Add a None or NaN in Integer Series it will type cast to Float Point type

# 1.Integer

A = pd.Series([1, 2])
print(f'Before Adding NaN : \n{A}')

print()

A[0] = np.nan
print(f'After Adding NaN : \n{A}')

A = pd.Series([1, 2])
print(f'Before Adding None : \n{A}')

print()

A[0] = None
print(f'After Adding None : \n{A}')

# Pandas cast the Integer Array to Floating Point, Pandas also converts the None to a NaN Value.


# 2.Float

A = pd.Series([1.1, 2.1])
print(f'Before Adding NaN : \n{A}')

print()

A[0] = np.nan
print(f'After Adding NaN : \n{A}')

A = pd.Series([1.1, 2.1])
print(f'Before Adding None : \n{A}')

print()

A[0] = None
print(f'After Adding None : \n{A}')

# No Change on Float, The Data Type remains Float itself.



# 3.Boolean

A = pd.Series([True, False])
print(f'Before Adding NaN : \n{A}')

print()

A[0] = np.nan
print(f'After Adding NaN : \n{A}')

# Pandas cast the Boolean Array to Floating Point if NaN is Added.

A = pd.Series([True, False])
print(f'Before Adding None : \n{A}')

print()

A[0] = None
print(f'After Adding None : \n{A}')

# Pandas consider None as False and Data Type remains Boolean.



# 4.Object

A = pd.Series(['Apple', 'Samsung'])
print(f'Before Adding NaN : \n{A}')

print()

A[0] = np.NaN
print(f'After Adding NaN : \n{A}')

A = pd.Series(['Apple', 'Samsung'])
print(f'Before Adding None : \n{A}')

print()

A[0] = None
print(f'After Adding None : \n{A}')

# No Change on Objects.



# Operating on Null Values.

# Pandas uses None and NaN as Interchangeable for indicating Missing Values and Null Values.

# There are several useful Methods in Pandas for Detecting, Replacing and Removing Missing and Null Values.

A = pd.Series([1, 2, 3, np.nan])
B = pd.Series([1, 2, 3, None])

# Detecting Missing and Null Values :
# 1.`isnull()` : Generate a Boolean mask Indicating Missing Values and Null Values

print(f'Series with Missing Value : \n{A.isnull()}')
print()
print(f'Series with Null Value : \n{B.isnull()}')

# 2.`isnull()` : Similar to isnull() also Generate a Boolean mask Indicating Missing Values and Null Values

print(f'Series with Missing Value : \n{A.isna()}')
print()
print(f'Series with Null Value : \n{B.isna()}')

# 3.`notnull()` : Opposite of isnull()

print(f'Series with Missing Value : \n{A.notnull()}')
print()
print(f'Series with Null Value : \n{B.notnull()}')

# Dropping Missing and Null Values :
# `dropna()` : Removes the Missing Values and Null Values

print(f'Series with Missing Value : \n{A.dropna()}')
print()
print(f'Series with Null Value : \n{B.dropna()}')

# Filling Missing and Null Values :
# `fillna()` : Fill or Impute Missing Values and Null Values

print(f'Series with Missing Value : \n{A.fillna(0)}')
print()
print(f'Series with Null Value : \n{B.fillna(0)}')