#!/usr/bin/env python
# coding: utf-8

# # Python for Data 10: Reading and Writing Data
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)

# Reading data into pandas DataFrames is often the first step when conducting data analysis in Python. The pandas package comes equipped with several data reading and writing functions that let you read data directly from common file formats like comma separated values files (CSV) and Microsoft Excel files. This lesson will focus on reading and writing data from these common file formats, but Python has packages available to work with just about every data format you encounter.

# ## Python Working Directory and File Paths

# Before we can jump into reading and writing data, we need to learn a little bit about Python's working directory and file paths. When you launch Python, it starts in a default location in your computer's file system (or the remote computer you are using) known as the working directory. You can check your current working directory by importing the os module and then using os.getcwd():

# In[ ]:


import os          

os.getcwd()


# The working directory acts as your starting point for accessing files on your computer from within Python. To load a data set, you either need to put the file in your working directory, change your working directory to the folder containing the data or supply the data file's file path to the data reading function.
# 
# You can change your working directory by supplying a new file path in quotes to the os.chdir() function:

# In[ ]:


os.chdir('/kaggle/')
        
os.getcwd()                     # Check the working directory again


# You can list all of the objects in a directory by passing the file path to the os.listdir( ) function:

# In[ ]:


os.listdir('/kaggle/input')


# Notice that the Kaggle input folder contains "titanic" and "draft2015". Titanic contains the data files realted to the Titanic Distaster competition, while draft2015 contains an Excel file I've uploaded to Kaggle for the purposes of this lesson.

# ## Reading CSV and TSV Files

# Data is commonly stored in simple flat text files consisting of values delimited(separated) by a special character like a comma (CSV) or tab (TSV).
# 
# You can read CSV files into a pandas DataFrame using the pandas function pd.read_csv():

# In[ ]:


import pandas as pd

titanic_train = pd.read_csv('input/titanic/train.csv')    # Supply the file name (path)

titanic_train.head(6)                           # Check the first 6 rows


# To load a TSV file, you can use pd.read_table(). The read_table() function is a general file reading algorithim that reads TSV files by default, but you can use to to read flat text files separated by any delimiting character by setting the "sep" argument to a different character. Read more about the options it offers [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_table.html).

# ## Reading Excel Files

# Microsoft Excel is a ubiquitous enterprise spreadsheet program that stores data in its own format with the extension .xls or .xlsx. Although you can save Excel files as CSV from within Excel and then load it into Python with the functions we covered above, pandas is capable of loading data directly from Excel file formats.
# 
# To load data from Excel, you can use the "xlrd" module. This module comes with the Python Anaconda distribution. If you don't have it installed, you can get it by opening a command console and running "pip install xlrd" (without quotes).
# 
# Load data from an Excel file to a DataFrame with pd.read_excel(), supplying the file path and the name of the worksheet you want to load:

# In[ ]:


draft = pd.read_excel('input/draft2015/draft2015.xlsx', # Path to Excel file
                     sheet_name = 'draft2015')         # Name of sheet to read from

draft.head(6)                            # Check the first 6 rows


# ## Reading Web Data

# The Internet gives you access to more data than you could ever hope to analyze. Data analysis often begins with getting data from the web and loading it into Python. Websites that offer data for download usually let you download it as CSV, TSV or excel files. Perhaps the easiest way load web data, is to simply download data to your hard drive and then use the functions we discussed earlier to load it into a DataFrame.
# 
# If you are running Python locally, reading from the clipboard is another quick and dirty option for reading web data and other tabular data. To read data from the clipboard, highlight the data you want to copy and use the appropriate copy function on your keyboard (typically control+C) as if you were going to copy and paste the data. Next, use the pd.read_clipboard() function with the appropriate separator to load the data into a pandas DataFrame. Since we are using Kaggle's kernel environment for this guide we will not be reading from the clipboard.
# 
# Pandas also comes with a read_html() function to read data directly from web pages. To use read_html() you need the HTML5lib package. Install it by opening a command console and running "pip install HTLM5lib" (without quotes). Note that HTML can have all sorts of nested structures and formatting quirks, which makes parsing it to extract data troublesome. The read_html() function does its best to draw out tabular data in web pages, but the results aren't always perfect. Again, since we are using the Kaggle kernel environment for this guide, we won't be using read_html() as it does not seem to play well with Kaggle's notebook environment. When it comes to using outside data on Kaggle, your best bet is to download the data to your local machine and then upload it to Kaggle as a dataset that you can add to your project.
# 
# Data comes in all sorts of formats other than the ones we've discussed here. The pandas library has several other data reading functions to work with data in other common formats, like json, SAS and stata files and SQL databases.
# 

# ## Writing Data

# Each of the data reading functions in pandas has a corresponding writer function that lets you write data back to into the format it came from. Most of the time, however, you'll probably want to save your data in an easy-to-use format like CSV. Write a DataFrame to CSV in the working directory by passing the desired file name to the df.to_csv() function:

# In[ ]:


draft.to_csv("draft_saved.csv") 

os.listdir('/kaggle/')


# Notice 'draft_saved.csv' now exists in the Kaggle folder. When you use Kaggle kernels to create submissions for competitions, you can write them to csv and then after committing your kernel, you can select the desired output file from the kernel's page and submit it to the competition for scoring.

# ## Wrap Up

# The pandas library makes it easy to read data into DataFrames and export it back into common data formats like CSV files. While there are many data fromats that we did not cover in this lesson, as the most popular programming language for data science, Python has functions available in packages to read just about any data format you might encounter.
# 
# Now that we know how to load data into Python we're almost ready to start doing data analysis, but before we do, we need to learn some basic Python programming constructs and how to write our own functions.

# ## Next Lesson: [Python for Data 11: Control Flow](https://www.kaggle.com/hamelg/python-for-data-11-control-flow)
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)
