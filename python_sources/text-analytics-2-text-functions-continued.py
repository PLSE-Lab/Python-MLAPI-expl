#!/usr/bin/env python
# coding: utf-8

# > <BIG><H1><Font color = "blue"> Introduction to Text Analytics </Font> </H1> </BIG> <I>continued..</I>    
#   
# Hello again, this is second kernal of the series on **Text Analytics**.  You can access the pevious kernal in the series by clicking on below link.  
# 
# <A href="https://www.kaggle.com/manish341/text-analytics-1-basic-text-functions"  target="_blank">
# Text Analytics 1 : Basic Text Functions
# </A>   
# <BR>
# First of all, I owe a BIG "Thank You" to all You for liking my first kernal. It was really motivating to see your comments and thoughts about it. I hope we will quickly clear the basic stages and move  to the magic of text analytics.    
#   
# As promised in previous kernal, we would have a look at some file handling functions first and then we will proceed further for more complex text handling techniques.    
# <BR>
# 
# <H2><Font color = "navy"> Handling Larger texts </Font> </H2> (File Handling: Reading Data from Text Files.)
# <HR>  
#   
# Usually in real world, we are supposed to work on larger text files and the data are not always in a structured format. Let's pull some sample text data I have created for this series and see how we can read the data from text files with various functions. I have kept it small for learning purpose. We can, and will dig the larger data sources/corpora at later stage once we are comfortable with text processing techniques.

# In[ ]:


## File Operations
# Open a file
# f = open(filename, mode) <-- format for open function.

f = open('../input/test.txt', 'r')  # The default mode is 'rt' (open for reading text).
                           # 'r' = read mode,     'x' : create a new file,
                           # 'w' = write mode,    'a' = open for writing
                           # 'b' = binary mode,   't' = text mode
                           # '+' = update mode,   'U' = Universal new line mode (deprecated)

# Reading file line by line
f.readline()    # Read the first line


# In[ ]:


# Reading next line? Run the same code again.
f.readline()


# > You see calling the same function twice gave different results. The reason behind this behavior is that on every read call, the function `readline()` **moves the cursor** to the next line once it has read the current one. So when we called the `readline()` function again, the cursor was already at the second line and so it returned texts from second line and moved the cursor on third line.
# 
# > So how to read read the first line again: For this we have `seek(position)` fuction, which sets the cursor at given byte position.

# In[ ]:


#Set the cursor to start of file
f.seek(0) 
print(f.readline())


# > The readline function `fileObject.readline(size)` also takes size as an optional parameter. 
# Size refers to number of bytes to read from the file. 
# It reads data in chunks, so next time you call the readline function it will fetch next chunk of given size.

# In[ ]:


#  Read data in chunks of 10 characters
f.seek(0) 
print(f.readline(10))     # Read first 10 bytes
print(f.readline(50))     # Read next few bytes


# > Notice that I had given 50 characters in second read command but it only returned next available characters from the first line, and didn't jump to second line to fetch more characters to complete the chunk to 50. To read the next line, you will need to call the `readline()` function again.  
# 
# > When one read command is complete and it reached the next line character at the end of the line, then it stops the read and move the cursor to next line. So when you read again, it returns characters from next line only, till it again finishes reading it.  

# In[ ]:


print(f.readline(15))     # Read bytes from another line


# > We also have `readlines()`  function (notice character 's' at end), which returns a list with lines as elements. 
# Try playing around and pass size of bytes you want to read. It would fetch the complete lines for any character overlap from one line to another. E.g. calling readlines(40) returns two complete line, instead of returning first line and only 7 characters from second line.

# In[ ]:


# seek to start and read all lines as a list element
f.seek(0)
print(f.readlines())

# Try reading only 40 characters from start
f.seek(0)
print(f.readlines(40))


# `readlines()` also brings the next line char `\n` at the end of every element. To remove these we can use the function `strip()` or `rstrip()`. Let's try this.

# In[ ]:


# Strip out the next line character (or any white space character) from output
f.seek(0)
lines = f.readlines()
[line.rstrip() for line in lines]


# The `strip()` function is very helpful in stripping/trimming white spaces from texts. It removes all the white spaces from both side of the string. `rstrip()` does the same but only from right side and `lstrip()` from left side.

# In[ ]:


# Removing white spaces from texts
txt1 = " This sentence has white spaces at both sides. "
txt2 = "This sentence has white spaces at right side only. "
txt3 = " This sentence has white spaces on the left side."

# Removing white spaces from both sides
print(txt1.strip())
print("Length before: " + str(len(txt1)) + ", and after: " + str(len(txt1.strip())) + " \n")

# Removing white spaces from right side
print(txt2.rstrip())
print("Length before: " + str(len(txt2)) + ", and after: " + str(len(txt2.rstrip())) + " \n")

# Removing white spaces from left side
print(txt3.lstrip())
print("Length before: " + str(len(txt3)) + ", and after: " + str(len(txt3.lstrip())) + " \n")


# > To read the full file at once use the `read()` function.

# In[ ]:


# Reading the full file
f.seek(0)
txt1 = f.read()
txt1


# In[ ]:


# split the file into list by lines (by '\n')
txt1.splitlines()      # It will return a list of all lines


# In[ ]:


# We can also write the data back to a back
f.write("We are writing new data to the text file.")


# >*Oops, that gave an error! *
# If you remember we pulled the data from text file with a read mode `'r'` connection. So to be able to write new data to text file, we should open this in write mode. But before opening the file again, we should close the file connection as well. Closing the unsed connections is a good programming practice.
# 
# > I just realized that opening the text file in write mode doesn't not work at here at kaggle dataset. You can try this on your local machine, I am writing some code below for reference (*commented*). `write()` would also create a new file for you if it doesn't exist already at the given path.

# In[ ]:


f.close()      # Close the current file connection

# Open it back in write mode
# f = open('test_new.txt', 'w')
# f.write("This is a new line we are inserting to the data.")
# f.close()


# At the end we also have a flag to check whether a connection is open or closed. The `file.closed` property returns logical values (True/False) for file handle's status. <BR>
# (*Try running the code before and after closing the connection.*)

# In[ ]:


f.closed


#  <BR>That's it for now for file handling functions. I am keeping this kernal a bit shorter, as we are about to begin the journey to NLTK module and I have a lot to write there. Thanks for all your support, hope this kernal brought you some useful functions along. Please keep sharing your thoughts in comments and upvote if you liked the kernal. See you again in the next section, till then happy texting..
#  
#  <BR>
# <U><B> Previous Kernals in the series </B></U>:  
# <A href="https://www.kaggle.com/manish341/text-analytics-1-basic-text-functions"  target="_blank">
# Text Analytics 1 : Basic Text Functions
# </A>
# 
# <BR> 
# <!-- <CODE>Update_20180522: I have updated the links below. You can click to open the next kernal of series.</CODE> -->
# 
# 
# <SMALL><I>The link to next kernal will be updated here in a week's time.
# 
# 
# <HR>
# <small><I>Coming up next: The NLTK Module<I></small>
#   
#     
