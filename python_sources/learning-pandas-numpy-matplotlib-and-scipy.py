#!/usr/bin/env python
# coding: utf-8

# #Intro
# If you are reading this, it means that tutorial is not full and in the process of updating. 
# 
# This is my Tutorial about ML. Its goal is to help me remember some details about these libs and give you an opportunity to learn something new if you are a pure newbie.

# ###Tabble of imports
# First of all we need to import all of our libs before we can use them. Let's do it!

# In[ ]:


import pandas as pd


# #Part I
# ##Pandas. Data Frame.
# 
# The concept of a data frame comes from the world of statistical software used in empirical research; it generally refers to "tabular" data: a data structure representing cases (rows), each of which consists of a number of observations or measurements (columns). Alternatively, each row may be treated as a single observation of multiple "variables". In any case, each row and each column has the same data type, but the row ("record") datatype may be heterogenous (a tuple of different types), while the column datatype must be homogenous. Data frames usually contain some metadata in addition to data; for example, column and row names. ([source][1])
# 
# So, let's create our first Data Frame.In order to do this we gonna use Pandas, which we imported in first step, and Python's dict. Our Data Frame will consist of numbers ranging 0-9 and 10 chars 'a'.
# 
# 
#   [1]: https://github.com/mobileink/data.frame/wiki/What-is-a-Data-Frame%3F

# In[ ]:


frame = pd.DataFrame({'numbers':range(10), 'chars':['a']*10})


# Let's have a  look at what we've got. 

# In[ ]:


frame


# Great! As you can see, we have two columns "chars" and "numbers" with rows. 
# 
# In fact such kind of notation is rarely used. Most of the time we gona work with an external data from different files. So it would be more comfortable to read these data to Data Frame and then work with it as a table. Let's try to do it. For this porpose we have the special function called `pd.read_csv()`
# 
# First argument of this func is a name of our file. In this tutorial I decided to use Happines Rank of different countries. Second argument is a number of a row where a header of table is placed. And the third one is a separator used in the file. In our case it's commas.

# In[ ]:


frame = pd.read_csv('../input/2015.csv', header=0, sep=',')
frame[:5] #list 5 rows of our table


# Now let's get some information about our Data set. In this case we're going to use a special attribute called "columns". As you can see, it shows us all columns in this particular data set.

# In[ ]:


frame.columns


# Also another useful attribute is called "shape". With its help we can have a look at the size of our table. It shows us that there are 158 rows and 12 columns.

# In[ ]:


frame.shape


# Ok. It's time to learn something about modifying our Data Frame. We can delete and add rows or columns. I will start from something easy like adding a row at the end of my table using Python's dictionary. First of all, we need to create a dict. 

# In[ ]:


mydict = {'Country':'AAAland', 'Region':'Some Region', 'Happiness Rank':0}


# I've created a simple dict with only two items. Now let's add it to the Data Frame.

# In[ ]:


frame.append(mydict, ignore_index=True)


# As we can see, there is our AAAland at the bottom of table. Now let's print out part of our Data Frame again.

# In[ ]:


frame[155:]


# Wait a minute! But where is our 158th row with AAAland?! In fact it's because `.append` command doesn't work inplace. It makes changes in our Data Frame and returns a copy of it. If we want to commit changes, then we need to modify previous command. 

# In[ ]:


frame = frame.append(mydict, ignore_index=True)
frame[155:]


# Great! Now we have all of our changes. Now I'm going to add a new column. 

# In[ ]:


frame['MyColumn'] = ['YEAH!']*159
frame['MyColumn'][4]


# As you can see, I've created a new column as a part of Python's dict and showed you the 4th row of MyColumn. 
# Ok. Now we know how to add elements. It's time to look at removing objects. :)
# 
# In order to remove a row we can use `.drop()` method with indexes as the first element and axis as the second. Notice that this function doesn't work inplace too as `.append`.

# In[ ]:


for i in range(150):
    frame = frame.drop([i], axis=0)
frame


# Yeah! We removed 150 rows from our data set. The columns can be removed the same way.  This time we need to specify the name of the column as the first element ,axis as the second. And to avoid copying frame, I specified the third element which forces our function to work inplace. 

# In[ ]:


frame.drop('MyColumn', axis=1, inplace=True)
frame


# Well, I think that it's time to save our modified Data Frame to a file. Pandas allows us to do it with using a simple method `.to_csv()`. First of all, we need to fill it with the name of our file in which we want to save the table. I've chosen "new_DataFrame.csv". Further we have a lot of freedom in how to save our file. For example we can change the separator from commas to tabs. Do we want to save the headers in a new_DataFrame.csv? Let's leave them. The same is allowed for indexes. In this example I've removed them.

# In[ ]:


frame.to_csv('new_DataFrame.csv', sep='\t', header=True, index=None)


# Let's make sure that new_DataFrame.csv is saved as we wanted. 

# In[ ]:


get_ipython().system('cat new_DataFrame.csv')


# Well, that's all for today. We've learned some basics of Pandas library. In the next episode we're going to continue our journey to Pandas and have a  look at some more complicated examples.

# #Part II
# ##Pandas. Indexing and selection.
# In this part we'll continue working with Pandas library and will study about indexing and selection.

# In[ ]:


#get our data frame again.
frame = pd.read_csv('../input/2015.csv', header=0, sep=',')


# First of all we want to know type of columns in data set. It can be done with `.dtypes`.

# In[ ]:


frame.dtypes


# In[ ]:





# In[ ]:




