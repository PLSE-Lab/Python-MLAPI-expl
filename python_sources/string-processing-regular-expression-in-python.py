#!/usr/bin/env python
# coding: utf-8

# **String Processing & Regular Expression in Python**
# 
# Priyaranjan Mohanty

# **What is Regular Expression**
# 
# A regular expression is a special text string for describing a search pattern. You can think of regular expressions as wildcards on steroids. You are probably familiar with wildcard notations such as "\*.txt" to find all text files in a file manager. The regex equivalent is ".*\.txt".
# 
# Regular Expression is also called RegEx

# **Why we need Regular Expression**
# 
# Regular expressions are used in many situation in computer programming.
# Majorly in search, pattern matching, parsing, filtering of results and so on.
# In lay man words, its a kind of rule which programmer tells to the computer to understand.

# In[ ]:





# Before we start on Regular Expression , lets explore STRING objects in Python -
# 
#      a) What is String 
#      b) How string is being represented in Python
#      c) What String operations / functions are available in Python

# **1) Strings in Python -**
# 
#    Strings in Python are any sequence of characters enclosed within Quotes.
#      
#    The string can be enclosed within a single quote or within a double quote.
#    
#    Note - The double quote is very useful as it can enclose a string which is containing a single quote within itself . Ex : "Mr. Parker's Pen"

# In[ ]:


# Defining a string variable - Using Double Quotes
#-------------------------------------------------

# String enclosed with double quotes
String_Var_1 = "String enclosed with double quotes"

# Print the content of the String Variable
print(String_Var_1)

# Print the Data type of the String Variable & Check whether it is a string datatype or not
print("Data Type of String_Var_1 is :",type(String_Var_1))


# In[ ]:


# Defining a string variable - Using Single Quotes
#-------------------------------------------------

# String enclosed with single quotes
String_Var_2 = 'String enclosed with single quotes'

# Print the content of the String Variable
print(String_Var_2)

# Print the Data type of the String Variable & Check whether it is a string datatype or not
print("Data Type of String_Var_2 is :" , type(String_Var_2))


# 
# 
# Some of us might be wondering whether 2 strings in Pyython containing the same set of characters are same or different when one of them is enclosed in single quotes and another one is enclosed in double quotes.
# 
# Let's check --

# In[ ]:


# Check whether 2 strings having a same content are same 
# irrespective of whether the strings are enclosed in
# single quote or by double quote

# Create a string enclosed with double quote
String_Dbl_Quote = "String for comparison"

# Create a second string enclosed with single quote
String_Sngl_Quote = 'String for comparison'

# Print both the strings 
print("String 1 with Double Quotes :" ,String_Dbl_Quote)
print("String 2 with Single Quotes :" ,String_Sngl_Quote)

# Print New Line for readability 
print("\n")


# Now lets check whether the above 2 strings are same or different 
print("Strings comparison result :-")
if String_Dbl_Quote == String_Sngl_Quote :
    print("    Both the Strings are Same ")
else :
    print("    Both the Strings are Different")


# 
# 
# 
# **What are the Functions / Methods / Operators available for String Objects**
# 
# 

# Now , lets explore few widely used functions available or applicable for Strings.

# * **len()** function - it returns the number of charachters in a string

# In[ ]:


# len() function - it returns the number of charachters in a string

# Define a String Variable
String_Var_3 = "ABCDEFGH"

# Print the Length ( no. of characters in a String ) of the string
print( "Nbr of Characters in String_Var_3 is :" , 
      len(String_Var_3) )


# 
# 
# * **str()** function - to convert a non string object to string 

# In[ ]:


# str() function - to convert a non string object to string 

# Define an Integer Variable
Integer_Var = 1234 

# Check the datatype of the integer variable 
print("Value of Integer Variable : " ,Integer_Var)

# Convert the interger content into string using str() function
Integer_Converted_to_String = str(Integer_Var)

# Check the datatype of the converted string variable 
print("Data Type of String containing the integer : " ,
      type(Integer_Converted_to_String))

# Print the content of the string containing the number
print("Value of String Variable containing Integer: " ,
      Integer_Converted_to_String)


# **String Concatenation** : Cobining 2 strings into a single string 
# 
#        Using the "+" operator

# In[ ]:


# String concatenation using + operator 

# Define two Strings
String_1 = "Hello "
String_2 = "World !"

# Print the concatenated string 
# Concatenation using the "+" Operator
print( "Concatenated String :" ,String_1 + String_2 )

# We can also assign the concatenated string to a new string variable 
String_Concat = String_1 + String_2
print("Variable containing Concatenated String :" ,
      String_Concat)


# * **Indexing** : accessing the individual characters of a string using index
# 
# Note - Python uses 'Zero Based Indexing' 

# In[ ]:


# Accessing the individual character of a string using Indexing

# Lets first print the string 
print("Here is the Complete String :",
      String_Concat)

# Print an Empty Line
print("\n")

# Now , lets access the 7th character of the string 
# As , Python uses zero indexing , 
# we have to use index of 6 to access 7th pos
print("Character in 7th Position is : " , String_Concat[6])

# Print an Empty Line
print("\n")


# We can also access the characters from end of the string using negative index
# lets access the last element of the string using the index of -1
print("Last Character in the String is : " , String_Concat[-1])


# **Extractng a set of characters from a String **
# 
# We can also extract a set of contigous characters from a string 
# using range .
# 
# lets fetch characters which are in - 
#      4th position till 7th position
# 
# In this case we have to use range of index in square bracket to fetch characters
# 
#   the index range to be used here is [3:8]
# 
# Note : the value before ':' is starting index position 
# 
#   the value after ':' is end index position (not included in fetch)

# In[ ]:


# Show the content of the Original String 
print("Original String is : " , String_Concat)

# Extract the characters between 4th and 8th postion
print("Characters between 4th and 8th positions are : " , 
      String_Concat[3:8])


# **Some additional character extraction tips from a String **

# In[ ]:


# Tip 1 : Extracting characters from start till certain index  
# leaving the index on the left of ':' empty leads to 
# 0 being considered as Start Index

print("Original String is : " , String_Concat)
print("First 4 characters in the String are : " ,
      String_Concat[:4])


# In[ ]:


# Tip 2 : Extracting characters from certain start index till end 
# leaving the index on the right of ':' , leads to 
# end of the string being considered as stop point 

print("Original String is : " , String_Concat)
print("Characters from 7th index till end of the String are : " , String_Concat[7:])


# In[ ]:


# Tip 3 :Extracting characters between certain indexes with stride 
# we will use 3rd argument in indexing which will denote the stride
# Stride denotes the steps to jump after each character is fetched
# Example : [2:8:2] means 
# Start from second index till 7th index while taking steps of 2

print("Original String is : " , String_Concat)
print("2nd index till 7th index with strides of 2 is : " , 
      String_Concat[2:8:2])


# In[ ]:


# Tip 4 : Reversing the string using indexing 
# leaving the start and end index as empty and 
# providing -1 as stride
# will return the string in reverse

print("Original String is : " , String_Concat)
print("String in Reversed order is : " , 
      String_Concat[::-1])


# 

# **Time to explore the methods available for string objects**

# Method to convert a string to lower case 

# In[ ]:


# Method to convert a string to lower case 

String_Var = "This is a String with Mixed Case"

# Print the content of the string variable 
print("The Original String is : ",String_Var)

# Print an Empty Line
print("\n")

# Convert all the characters of the String to lower and print it 
print("Convertng the string characters to lower case : " ,
      String_Var.lower())


# 
# Method to convert a string to Upper case 

# In[ ]:


# Method to convert a string to Upper case 

String_Var = "This is a String with Mixed Case"

# Print the content of the string variable 
print("The Original String is : ",String_Var)

# Print an Empty Line
print("\n")

# Convert all the characters of the String to lower and print it 
print("Convertng the string characters to upper case : " ,
      String_Var.upper())


# Method to convert only the first character of the string to Upper case
# 
# **capitalize()** method converts the first character of the string to Upper case while converting all other characters to lower case

# In[ ]:


# Method to convert only the first character of the string 
# to Upper case 

String_Var = "mt Everest"

# Print the content of the string variable 
print("The Original String is : ",String_Var)

# Print an Empty Line
print("\n")

# Convert all the characters of the String to lower and print it 
print("Convertng the first character of String to upper case : " ,String_Var.capitalize())


# 

# **Methods for splitting a string into substrings **

# In[ ]:


# splitting a string into individual characters 
# the output of this operation will be a list object 

String_Var = 'Python String Processing'

# Print the content of the string 
print("The Original String is :" ,String_Var )

# Print an Empty Line
print("\n")


# Applying the list() function on a string will 
# return a list object containing 
# each character of the string as individual list element 
print("The individual characters of String : ",
      list(String_Var))


# In[ ]:


# Splitting a String into sub-strings and 
# the splitting to be done based on a separator 

String_Var = 'Python String Processing'

# Print the content of the string 
print("The Original String is :" ,String_Var )

# Print an Empty Line
print("\n")

# using the split() method , 
# split the string into substrings where 
# <empty space> is the separator 
# Note - split() method returns a list object 
# containing the substrings
print("Substrings from the String :" ,
      String_Var.split(" "))

# Print an Empty Line
print("\n")

# Note : By default the split happens on <empty space> 
# and hence the argument to the split method can be 
# left empty if the string has to be splitted by space
print("Substrings from the String :" ,
      String_Var.split())


# 

# **Method for Joining a set of substrings into a single string **
# 
# Note - We can also provide a separator which is added between the substrings  in the combined string 

# In[ ]:


# Joining a set of substrings into a single string 
# join() method takes a set of substrings as input and 
# returns a single combined string

# Define a List of strings 
List_of_SubStrings =[ "Euclid" , "Newton" , "Einstien" , "Pascal"]

# Print the content of the list containing sub-strings
print("Content of the List containing strings :" ,
      List_of_SubStrings)

# Print an Empty Line
print("\n")

# Join the substrings into a single string separated by "/"
Sep_Str = "/"
print( "Combined / Joined String is :",
      Sep_Str.join(List_of_SubStrings))


# **Methods to trim leading and trailing characters from a string 
# 
# The method for the same are -
# 
#           strip()
#           rstrip()
#           lstrip()

# In[ ]:


# using strip() method to trim the 
# leading and trailing spaces form a string

# Define a string having leading and trailing spaces
String_Var = "         A String  having  leading and trailing   spaces            "

# Print the content of the string 
print("The Original String is :" ,String_Var ,"*")

# Print an Empty Line
print("\n")


# Remove the leading and trailing spaces from the string 
# using strip() method
# Added a '*' at the end to show trimming of trailing spaces
print("Trimmed string is : " ,
      String_Var.strip(),
     "*")

# Note : Only leading and trailing spaces are removed , 
# No chnage to embedded spaces in the string


# In[ ]:


# using lstrip() method to trim the leading spaces form a string

# Define a string having leading spaces
String_Var = "         A String  having  leading and trailing   spaces     "

# Print the content of the string 
print("The Original String is :" ,String_Var )

# Print an Empty Line
print("\n")

# Remove the leading spaces from the string using 
# lstrip() method
print("Leading Spaces Trimmed string is : " ,
      String_Var.lstrip())


# In[ ]:


# using rstrip() method to trim the 
# trailing spaces form a string

# Define a string having leading spaces
String_Var = " A String  having trailing   spaces                 "

# Print the content of the string 
print("The Original String is :" ,String_Var , "*")

# Print an Empty Line
print("\n")

# Remove the trailing spaces from the string using 
# rstrip() method
print("Trailing Spaces Trimmed string is : " ,
      String_Var.rstrip(),
     "*")


# strip() method can be used to strip any charachter at the start or end of a string

# In[ ]:


# Strip leading & trailing '$' symbols from a string 

# Define a string 
String_Var = "$$$$$  String with leading & trailing Dollars $$$$$$" 

# Print the content of the string 
print("The Original String is :" ,String_Var )

# Print an Empty Line
print("\n")

# Remove the leading & trailing $ from the string using 
# strip() method
print("String after removing leading & trailing $ : " ,
      String_Var.strip('$'))


# 

# **Searching / Finding Substrings from a String**

# **find()**
# 
# The find() methods works in the following way -
# 
# Synatx :  String.find(Sub_String,start,end)
# 
#    String - It is the Target String from which a substring has to be searched
#    
#    Sub_String - this is the string to be searched [its a madatory argument ]
#           
#    
#    Following are Optional Arguments to find():
#    
#    start - this is the starting position from where the search has to be started
#    
#    end - this is the end position at which the search has to be stopped 
#           
#    
#    Values Returned :
#    
#    The function returns the index position at which the substring is found first
#    
#    in case the substring is not found , the function returns -1

# In[ ]:


# Example code for find() 

# Define a Target string 
String_Var = "Python is powerful language with power packed features"

# Print the content of the Target String 
print("The Target String is : ",String_Var)

# Define a Sub string 1
Sub_String_Var1 = "power"

# Print the content of the Target String 
print("The Sub String 1 is : ",Sub_String_Var1)

# Define a Sub string 2
Sub_String_Var2 = "beauty"

# Print the content of the Target String 
print("The Sub String 2 is : ",Sub_String_Var2)

# Print Empty Line
print("\n")

# Find the position of Sub String in the Target String
# When the substring does exist in Target String 
print("The position at which the Sub String is found : " ,
      String_Var.find(Sub_String_Var1))

# Find the position of Sub String in the Target String
# When the substring does not exist in Target String 
print("The position at which the Sub String is found : " ,
      String_Var.find(Sub_String_Var2))


# In[ ]:


# Example code for find() function 
# to find a substring from a target string 
# within range of positions

# Define a Target string 
String_Var = "Python is powerful language with power packed features"

# Print the content of the Target String 
print("The Target String is : ",String_Var)

# Define a Sub string 1
Sub_String_Var1 = "power"

# Print the content of the Target String 
print("The Sub String 1 is : ",Sub_String_Var1)

# Print Empty Line
print("\n")

# Find the position of Sub String in the Target String
# between range of positions - Example 1 
print("The position at which the Sub String is found (Ex 1 ): " ,
      String_Var.find(Sub_String_Var1,4,20))

# Print Empty Line
print("\n")

# Find the position of Sub String in the Target String
# between range of positions - Example 2
print("The position at which the Sub String is found (Ex 2 ): " ,
      String_Var.find(Sub_String_Var1,15,40))


# **Counting the number of occurences of a Sub String with in a String**
# 
# using count() method
# 
# Syntax of count() method is : Target_String.count(substring,start,end)
# 
# 
# 
# Where :
# 
# Target_String : is the String from which the substring occurences to be counted 
# 
# substring : the substring to be searched and occurences counted
# 
# start : Start position in Target String from where the substring to be searched 
# 
# start : end position in Target String at which the substring searched to be stopped
# 
# 
# 
# The count() method returns :
# 
# The number of instances the substring is found within the Target string 

# In[ ]:


# Example code to count the occurences of a Substring in a String

# Define a Target string 
String_Var = "Python is a powerful programming language when compared with other programming languages"

# Print the content of the Target String 
print("The Target String is : ",String_Var)

# Define a Sub string 1
Sub_String_Var = "language"

# Print the content of the Target String 
print("The Sub String is : ",Sub_String_Var)

# Print Empty Line
print("\n")

# Count the occurences of a Substring using count() method
print("# of times the substring occurs in the Target String is : ",
     String_Var.count(Sub_String_Var))

# Count the occurences of a Substring using count() method
# between a range of positions 
print("# of times the substring occurs in the Target String is : ",
     String_Var.count(Sub_String_Var,40,95))


# **Replacing substring within a String with another substring **
# 
# The method to be used for replacing is replace()
# 
# The Syntax for replace is : string.replace(Old_sub , New_Sub , Count )
# 
# Where -
# 
# string : is the Target string within which a substring has to be replaced with new substring
# 
# Old_sub : The substring which is to be replaced 
# 
# New_Sub : The new substring to be replaced with 
# 
# Count : Specifies the number of instances to be replaced 
# 

# In[ ]:


# Example code to replace the occurences of a Substrings in a String
# with new Substrings

# Define a Target string 
String_Var = "Python is a powerful programming language when compared with other programming languages"

# Print the content of the Target String 
print("The Target String is : ",String_Var)

# Define a Sub string 1
Sub_String_Var = "language"

# Print the content of the Target String 
print("The Sub String is : ",Sub_String_Var)

# Print Empty Line
print("\n")

# Replace all occurences of a Substring with new substrings
print("Updated String with replaced sub string is :",
     String_Var.replace(Sub_String_Var,'tool'))

# Print Empty Line
print("\n")

# Replace first occurence of a Substring with new substring
print("Updated String with replaced sub string is :",
     String_Var.replace(Sub_String_Var,'tool',1))

