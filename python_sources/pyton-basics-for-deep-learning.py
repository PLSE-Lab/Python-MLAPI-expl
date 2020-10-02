#!/usr/bin/env python
# coding: utf-8

# References:
# - Datai team
# - w3school
# 
# Contents
# * Pyton
#     * Variables
#         - String
#         - Number
#         - List
#         - Set
#         - Boolean
#         - Bytes
#     * Operators
#     * Loops
#     * Functions
#     * Modules
#     * Error cathing
# * Numpy
# * Pandas

# # Variables
# 
# - Variables cann be declared globally (outside function) with global method.
#     - global x
#     - type(var) => Type of variable
# 
# ## String
# 
# - string => "Adem"
#     - str("Adem")
#         - Pyton string variables can be declared with ' and ". For multiline use ''' or """.
#         - All strings are array of unicode. 
#         
# #### Escape characters 
#     \' or \" => quatos
#     \\       => Backslash
#     \n       => new Line
#     \r       => Carriage return
#     \t       => Tab
#     \b       => Backspace
#     \f       => Form Feed
#     \xhh     => Hex value
# 
# #### String methods
#     len(var) => Length of a string variable
#     var[0:3] => From - to index of string variable
#     var.replace("A","B") => Replace x to y in string variable
#     var.split(",") => Splits string to list.
#     "the" in var => Test the string (if found creates boolean True)
#     "the" not in var => Test the string (if found creates boolean False)
#     "For combining {:.2f} and {} numbers use this sttructure".format(x,y) => To use index numbers {0:.2f},{1}
#     "The output is {car} for {value}".format(car="model", value= 666)
#     
# #### Other methods
# *     capitalize()	Converts the first character to upper case
# *     casefold()	Converts string into lower case
# *     center()	Returns a centered string
# *     count()	Returns the number of times a specified value occurs in a string
# *     encode()	Returns an encoded version of the string
# *     endswith()	Returns true if the string ends with the specified value
# *     expandtabs()	Sets the tab size of the string
# *     find()	Searches the string for a specified value and returns the position of where it was found
# *     format()	Formats specified values in a string
# *     format_map()	Formats specified values in a string
# *     index()	Searches the string for a specified value and returns the position of where it was found
# *     isalnum()	Returns True if all characters in the string are alphanumeric
# *     isalpha()	Returns True if all characters in the string are in the alphabet
# *     isdecimal()	Returns True if all characters in the string are decimals
# *     isdigit()	Returns True if all characters in the string are digits
# *     isidentifier()	Returns True if the string is an identifier
# *     islower()	Returns True if all characters in the string are lower case
# *     isnumeric()	Returns True if all characters in the string are numeric
# *     isprintable()	Returns True if all characters in the string are printable
# *     isspace()	Returns True if all characters in the string are whitespaces
# *     istitle()	Returns True if the string follows the rules of a title
# *     isupper()	Returns True if all characters in the string are upper case
# *     join()	Joins the elements of an iterable to the end of the string
# *     ljust()	Returns a left justified version of the string
# *     lower()	Converts a string into lower case
# *     lstrip()	Returns a left trim version of the string
# *     maketrans()	Returns a translation table to be used in translations
# *     partition()	Returns a tuple where the string is parted into three parts
# *     replace()	Returns a string where a specified value is replaced with a specified value
# *     rfind()	Searches the string for a specified value and returns the last position of where it was found
# *     rindex()	Searches the string for a specified value and returns the last position of where it was found
# *     rjust()	Returns a right justified version of the string
# *     rpartition()	Returns a tuple where the string is parted into three parts
# *     rsplit()	Splits the string at the specified separator, and returns a list
# *     rstrip()	Returns a right trim version of the string
# *     split()	Splits the string at the specified separator, and returns a list
# *     splitlines()	Splits the string at line breaks and returns a list
# *     startswith()	Returns true if the string starts with the specified value
# *     strip()	Returns a trimmed version of the string
# *     swapcase()	Swaps cases, lower case becomes upper case and vice versa
# *     title()	Converts the first character of each word to upper case
# *     translate()	Returns a translated string
# *     upper()	Converts a string into upper case
# *     zfill()	Fills the string with a specified number of 0 values at the beginning
#     
# ## Number
# 
# - integer => 56
#     - int(56)
# - Float   => 4.42
#     - float(4.43)
#         - Scientific numbers can be float 23e2 for exponential 10 (2e2 = 2*10^2 = 200.0)
# - Complex numbers => 1j
#     - complex(1j)
# 
# #### Random numbers
#     Import random
#     random.randrange(1,10)
# 
# ## List
# 
# - List  => ["a", "b", "c"]
#     - list(("apple", "banana", "cherry"))
# 
# #### List options
#     samplelist[8]     => index of list. [-1] for last item
#     samplelist[2:5]   => 2 (included), 5 (not included)
#     samplelist[:2]    => Starts first value (index 0)
#     samplelist[-4:-2] => From -4 (included) to -2 (not included)
#     samplelist[3] = x => Can change item value.
#     for each in samplelist:  => Can loop in list
#     if x in samplelist:      => Can check if x is in the list
#     len(samplelist)          => Length of list
#     samplelist.append(x)     => Append x to list
#     samplelist.insert(i,x)   => Insert x to spesified index (i)
#     samplelist.remove(x)     => Remove an item from list
#     samplelist.pop(i)        => Remove indexed item or last item if empty
#     del samplelist[i].       => Removes indexed item (if empty delete list completely)
#     sample1 + sample2        => Combines list
#     sample1.extend(sample2)  => Combines list
# 
# #### Other methods
# * append()	Adds an element at the end of the list
# * clear()	Removes all the elements from the list
# * copy()	Returns a copy of the list
# * count()	Returns the number of elements with the specified value
# * extend()	Add the elements of a list (or any iterable), to the end of the current list
# * index()	Returns the index of the first element with the specified value
# * insert()	Adds an element at the specified position
# * pop()	    Removes the element at the specified position
# * remove()	Removes the item with the specified value
# * reverse()	Reverses the order of the list
# * sort()	Sorts the list
#     
# - Tuple => ("a", "b", "c")
#     - tuple(("apple", "banana", "cherry"))
#         - Tuple is ordered and unchangable collection. 
#         - You can not append or remove a tuple. But you can delete a tuple entirely.
#         - You can join two tuples with (sample1 + sample2)
#         
# - range => range(8)
#     - range(6)
#     
# - dict => {"name" : "John", "age" : 42}
#     - dict(name="John", age=36)
#         - Can be user nested.
# 
# #### Methods
#     dic.values  => Get values
#     dic.keys    => Get keys
#     sample["key"] or sample.get["key"]  ==> For access
#     for x in sample   ==> x: for keys, sample[x] for values
#     for x,y in sample ==> return x: keys, y: values
#     "key" in sample   ==> Check if key exist
#     sample.remove["key"] ==> Removes item
#     del sample ==> Deletes dictionary
# 
# #### Other methods
# * clear()	Removes all the elements from the dictionary
# * copy()	Returns a copy of the dictionary
# * fromkeys()	Returns a dictionary with the specified keys and values
# * get()	Returns the value of the specified key
# * items()	Returns a list containing a tuple for each key value pair
# * keys()	Returns a list containing the dictionary's keys
# * pop()	Removes the element with the specified key
# * popitem()	Removes the last inserted key-value pair
# * setdefault()	Returns the value of the specified key. If the key does not exist: insert the key, with the specified value
# * update()	Updates the dictionary with the specified key-value pairs
# * values()	Returns a list of all the values in the dictionary
# 
# ## Set
# 
# - set       => {"apple", "banana", "cherry"}
#     - Unordered and unindexed collection. 
#     - There is no index. So access with for loop.
#     - add(), update(), remove(), discard(), pop(), clear(), del() can be used. 
# - frozenset => frozenset({"apple", "banana", "cherry"})
# 
# ## Boolean
# 
# - boolean => true
#     - bool(5)
#         - bool(x) => True for any value except empty string, list, dictionary, tuple and number: 0
#         - isinstance(x, int) => Check type of a variable and returns boolean
# 
# ## Bytes
# 
# - bytes      => b"Hello"
# - bytearray  => bytearray(5)
# - memoryview => memoryview(bytes(5))

# # Operators
# 
# #### Arithmetic operators
#     + Addition
#     - Substruction
#     * Multiplication
#     / Division
#     % Modulus
#     // Floor division
#     ** Exponential
# 
# #### Assignment Operators
#     x = 5.      x = 5         (5)
#     x += 3      x = x + 3     (8)
#     x -= 3      x = x - 3     (2)
#     x *= 3      x = x * 3     (15)
#     x /= 3      x = x / 3     (1.6666666666666667)
#     x %= 3      x = x % 3     (2)
#     x //= 3     x = x // 3    (1)
#     x **= 3     x = x ** 3    (125)
#     x &= 3      x = x & 3     (1)
#     x |= 3      x = x | 3     (7)
#     x ^= 3      x = x ^ 3     (6)
#     x >>= 3     x = x >> 3    (0)
#     x <<= 3     x = x << 3    (40)
#   
# #### Comparison operators
#     and, or, not
#     ==, !=, <, <=, >, >=
#     is, is not => Object detection
#     in, inn not => Sequence detection
# 
# #### Binary comparison
#     & and
#     | or
#     ^ xor
#     ~ not
#     << zero fill left
#     >> signed right

# # Loops 
# 
# - Indent loops smilar to curly brackets
# - Use "elif" for "else if" 
# - You can use one line statement. print("A") if a > b else print("=") if a == b else print("B")
# - Nested if can be use.
# - Use "pass" for empty if statement.
# 
# ## While loops
# 
# - Continue as long as condition is true
# - "break" can stop loop.
# - "continue" stops current iteration and continue to next.
# - "else" run a code block to run when condition is no longer true.
# 
# ## For loops
# 
# - iterate a list or string,
# - for x in range(6): || for x in range(2, 6, 2): ==> 2 is increment
# - "else" run a code block to run when loop is finished.
# 
# 
# 

# # Functions
# 
# - Pass arguments to function
#     - def func(x)      => func(a)
#     - def func(x,y)    => func(a,b)
#     - def func(*arg)   => func(a,b,c). use a[1] in function
#     - def func(x,y,z)  => func(x=a, y=b, z=c)
#     - def func(** var) => func(key1=a, key2=b). use var["key1"] in function
#     - def func(x = 2)  => Default parameter for function
#     - pass             => For empty functions.
#     - function can call itself recursively
#     
# ## Lambda function
# 
# - Anonymous function
#     - x = lambda a, b : a * b
#     
# ## Zip
# 
# - zipList = zip(listKey, listVal)        => zip lists
# - unList1, unList2 = list(zip(*myList))  => unzip lists to tuple. use list to convert list
#     

# # Modules
# 
# - Module is a code library (.py) and can be import pyton
#     - import thismodule as tm
# - dir(modulename) => List all functions in a module
# - importinng a function or variable from a module is possible
#     - from xmodule import yfunction  
#         - Dont necessary to use module name when accessing yfunction(x,y) instead of xmodule.yfunction(x,y)
# 
# 

# # Error cathing
# 
# - try, except, except NameError, else
# - finnally 
# - raise => use for raise an error
# 

# 
#     
#  # NumPy
#  
#  Numpy is the core library for scientific computing in Python.
#  
#  - import numpy as np
#  
# #### Arrays
# 
#     np.array([1,2,3]) => An array matrix (1,)
#     x.shape           => Shape of an array
#     x.reshape(3,4)    => Reshape an array (3,5)
#     x.resize(3,4)     => Reshape an array (3,5)   #Assigns to array.
#     x.ravel()         => Flatten an array
#     np.vstack(x,y)    => Combine two arrays
#     x.ndim            => Dimension of an array (2)
#     x.dtype.name      => datatypes in array
#     x.size            => size of an array
#     x.T               => Transpoze of an array
#     np.array(listvar) => Convert list to array
#     list(arrayvar)    => Convert array to list
#     a = b.copy.       => Copy b array to a.
#     
#     np.zeros((3,4))   => Allocate a (3,4) matrix with 0
#     np.ones((3,4))    => Allocate a (3,4) matrix with 1
#     np.empty((3,4))   => Allocate a (3,4) matrix with empty
#     np.full((3,4), 7) => Allocate a (3,4) matrix with 7
#     np.random.random((3,4)) => Allocate a (3,4) matrix with random
#     np.arrange(10,50,5)  => Start 10 to 50, increase 5
#     np.linspace(10,50,5) => Take 5 number from 10 to 50
#     
#     a[4]    => Index of an array (x) matrix
#     a[::-1] => Reverse an array (x) matrix
#     a[2,4]    => Index of an array (x,y) matrix
#     a[:,1]    => Take all rows and 0,1 columns
#     a[:,1:4]  => Take all rows and 1-4 columns
#     a[-1,:]   => Take last row
#     a[:,-1]   => Take last column
#     
#     np.add(x,y), np.substract(x,y), np.multiply(x,y), np.divide(x,y), np.sqrt(x,y)
#     np.sum(x)          => Compute sum of all elements
#     np.sum(x, axis=0)  => Compute sum of each column
#     np.sum(x, axis=1)  => Compute sum of each row
#     np.max, np.min, np.mean
# 
#  
#  ## Matrix operations
#  
#     For addition or substraction (rows and columns must be the same) (a,b) +- (a,b)
#      [a, b]      [x, y]       [a+x, b+y]
#      [c, d]      [z, t]       [c+z, d+t]
#      
#     For multiplication or division
#      [a, b]      [x, y]       [ax+bz, by+bt]    (2,2)*(2,2) Produces (2,2)
#      [c, d]      [z, t]       [cx+cz, dy+dt]
#      
#     For dot (matrix multiplation) 
#      [a11, a12]      [b11, b12, b13]       [a11.b11 + a12.b21, a11.b12 + a12.b22, a11.b13 + a12.b23]   (3,2)* (2,3) Produces (3,3)
#      [a21, a22]      [b21, b22, b23]       [a21.b11 + a22.b21, a21.b12 + a22.b22, a21.b13 + a22.b23]
#      [a31, a32]                            [a31.b11 + a32.b21, a31.b12 + a32.b22, a31.b13 + a32.b23]

# 
# 
# # Pandas library
# 
# - import pandas as pd
# - pd.read_scv("filename")
# 
# #### dataframe
#     dic = {"NAME": ["a","b","c"]
#            "AGE": [1,2,3]}
#     df = pd.DataFrame(dic)
#     df["New"] = [] => Create a new column
# 
# #### methods
#     df.head(x)             => First x rows (default 5)
#     df.tail(x)             => Last x rows (default 5)
#     df.info()              => Information
#     df.dtypes              => Get column types of dataframe
#     df.col.astype(float)   => Change col type to fload
#     df.columns             => Columns of dataframe
#     df.describe            => Descriptive statistics
#     df.corr()              => Show corelations
#     df.loc[rows, column]       => Locate Index
#     df.loc[0:3,["AGE","NAME"]] =>from x to y (y included)
#     df.loc[::-1, :]            => Reverse rows and take all columns
#     df.iloc[:,3]               => Index for column
#     
#     df.set_index("age")        => Change index to age column
#     df.index_name("myI")       => Change index name
#     df.index = range(0,100,3)  => Renumerate index
#     df.reindex(x).             => Reindex with x (x= df.age.sort_values().index.values)
#     df.set_index(["age", "m"]) => set an outher index
#     df.unstack(level=0)        => Remove outher index
# 
#     df.AGE.mean()
#     np.mean(df.AGE)  => With numpy library
#     df["new-feature"] = ["t_val" if each > x else "f_val" for each in df.AGE]
#     df.columns = [each.lower() for each in df.columns]
#     df.columns = [each.split()[0] +"_"+ each.split()[1] if (len(each.split()) > 1) else each for each in df.columns]
#     
#     df.groupby("sex")[["age", "chol"]].min()  => Group data
#     df.pivot( columns="cp", values="chol")    => Pivot data
#     
#     df.drop(["NEW"], axis=1, inplace=True)
# 
#     df.sex.value_counts(dropna=False)   # Print value counts (drop null data = false)
#     df.sex.dropna(inplace= True)        # Check data. If value is null drop that raw
# 
#     pd.concat([df1, df2], axis=1)   =>  0: vertical, 1: horizontal  #use pd!!!
#     
#     def multi(x):
#         return x*2+4
#     df["NEW"] = df.age.apply(multi)
# 
#     for index, value in enumerate(df["age"][0:10]): => Enumerate index and value
#     
#     list(df["Area"].unique())                 => Create a list of unique values
#     df.age.sort_values(ascending=False))      => Sort values
#     df.age.value_counts()                     => Counts value
#     df.name.replace(["-"], 0.0, inplace=True) => Replace. Only for string values
#     
# #### Filters
#     xf = df.age > 30 
#     yf = df.salary > 300
#     df[xf & yf]
#     df[df.age > 60 & df.salary > 300]
#     df[(df["age"] > 60) & (df["chol"] > 300)]
# 

# # MatplotLab
# 
# - import matplotlib.pyplot as plt
# 
# https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
# 
# #### Methods
# 
#     plt.plot(x,y)  
#     plt.plot(a,b)  => Repeat to draw a second graph
#     
#     df.plot(kind="scatter", x="age", y="chol", alpha=.7, color="green") => kind=line, bar, scatter, hist
#     df.plot(subplots=True)
#     
#     df.age.plot(kind="line" color="red" label="label" linewidth=1, alpha=.7, grid=True, linestyle="-")
#     df.boxplot(column = "chol", by = "sex")
#     
#     plt.pie([5,10,15], explode=[0,0,0], labels=["x","y","z"], colors=["red","green","blue"], autopct='%1.1f%%')
#     
#     plt.title("title")
#     plt.xlabel("label")
#     plt.ylabel("label")
#     plt.legend(["xl","yl"])
#     plt.xticks(rotation= 45)
#     plt.grid()
#     plt.text(10,0.7,"Text on graph", fontsize=19, style="italic")
#     plt.savefig('graph.png')  => For save 
#     plt.show() 
#     
#     plt.subplot(2,1,1)   => Set a grid (h,w,i) and place 1.
#     plt.plot(x,y)
#     plt.subplot(2,1,2)   => Set a grid (h,w,i) and place 2.
#     
#     
#     plt.imshow(img)      => Draw ann image

# # Seaborn
# 
# #### plot types
#     sns.barplot(x=x, y=y) => palette = sns.cubehelix_palette(len(x))
#     sns.pointplot(x="area", y="rate", data=df, color="red", alpha=0.5)
#     sns.jointplot(data=df, x="col1", y="col2")
#     sns.jointplot(data=df, x="col1", y="col2", kind="kde")
#     sns.kdeplot(df["col1"], df["col2"], shade=True, cut=3)
#     sns.lmplot(data=df, x="col1", y="col2")
#     sns.violinplot(data=df, inner="points")
#     sns.heatmap(df.corr(), annot=True, linecolor="gray", linewidth=.1, fmt=".1f")
#     sns.pairplot(df)
#     sns.boxplot(x="gender", y="age", hue="malign", data=df,  palette="PRGn")
#     sns.swarmplot(x="gender", y="age", hue="malign", data=df)
#     sns.countplot(df.gender)
