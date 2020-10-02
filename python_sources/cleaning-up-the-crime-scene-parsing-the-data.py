#!/usr/bin/env python
# coding: utf-8

# # Cleaning Up The Crime Scene: Parsing The Data
# 
# ### Table of Contents
# 
#  * Introduction
#  * Loading the Data
#  * Properly Parsing School Names
#  * Getting a DataFrame
#  * Divvying Data Using Quantiles
#  * Plotting a Distribution
#  * Functional Data Processing
#  * Parsing Law Enforcement Data
#  * Parsing Offenses Data
#  * Summary: Where To From Here
# 
# <br />
# <br />
# <br />

# ## Introduction
# 
# This data set contains information about crimes and law enforcement agencies in the State of California, and is provided by the FBI. The data set consists of 8 CSV files, 4 with data about law enforcement officers and 4 with data about crimes. The data is poorly formatted, and only 2 of the CSV files can be loaded as Pandas DataFrames without much effort.
# 
# In this notebook, we'll be walking through the use of regular expressions and other low-level drudgery to parse this odd assortment of typos, Linux and Windows newline characters, and an all-around tasteless use of commas, and stuff this processing into some functions that will magically return nice tidy DataFrames.
# 
# (An alternative to using Python is to use command line tools like `sed` or `awk` to do the same kind of processing. However, `sed` is a line-based tool, and as we'll see, there are some issues with the data that require dealing with multiple lines. It's possible to use `awk`, but mastery of `awk` requires time and practice. Python is a good alternative.)

# In[ ]:


# must for data analysis
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from matplotlib.pyplot import *

# useful for data wrangling
import io, os, re, subprocess

# for sanity
from pprint import pprint


# In[ ]:


data_files = os.listdir('../input')
pprint(data_files)


# This notebook will demonstrate how to load just _one_ of these files - but each one presents its own challenges. Ultimately, we can abstract away the details of each file into (at least) 8 classes or 8 static methods. This notebook will cover the procedural programming that would go into those methods and classes, and illustrate how the components work. Then they can all be packaged up into a function.
# 
# ## Loading the Data
# 
# Start by loading the lines of the file into a list of strings:

# In[ ]:


filename = 'ca_law_enforcement_by_campus.csv'
filewpath = "../input/"+filename

with open(filewpath) as f:
    lines = f.readlines()

# First 6 lines are part of the header
header = ' '.join(lines[:6])
header = re.sub('\n','',header)
data = lines[6:]

pprint([p.strip() for p in data[:10]])


# This data should not be too tricky. Checking the number of commas:

# In[ ]:


number_of_commas = [len(re.findall(',',p)) for p in data]
print(number_of_commas)


# All lines have 6 fields, one of which (attendance) always has one comma, which means all lines have at least 6 commas. Lines without 6 commas are empty and can be thrown out.
# 
# Once we are finished manipulating the data we can pass the resulting list of parsed strings on to a DataFrame.

# ## Properly Parsing School Names
# 
# Properly parsing school names is straightforward: there are two fields, University/College (which gives the university system of campuses) and Campus (which gives the particular campus location). 
# 
# While parsing this data, we can also add a check to make sure we're ignoring empty lines, which in this file means four or more commas in a row: `,,,,`

# In[ ]:


# Parse each line with a regular expression
newlines = []
for p in data:
    if( len(re.findall(',,,,',p))==0):
        newlines.append(p)

pprint(newlines[:10])


# ## Getting a DataFrame
# 
# We now have the raw data as a list of strings. We can process the string using some regular expression magic. Here's what the procedure will look like:
# * Join the list of strings together into one long string
# * Create a StringIO object to turn that string into a stream that can be read.
# * Extract column names from the (badly-mangled and poorly-formatted) header file
# * Pass the StringIO object and properly formatted column names to the Pandas `read_csv()` method to get a DataFrame.
# 
# The end result is a DataFrame that we can use to do a statistical analysis.

# In[ ]:


one_string = '\n'.join(newlines)
sio = io.StringIO(one_string)

columnstr = header

# Get rid of \r stuff
columnstr = re.sub('\r',' ',columnstr)
columnstr = re.sub('\s+',' ',columnstr)

# Fix what can ONLY have been a typo, making this file un-parsable without superhuman regex abilities
columnstr = re.sub(',Campus','Campus',columnstr)

columns = columnstr.split(",")

df = pd.read_csv(sio,quotechar='"',header=None,  names=columns, thousands=',')
df.head()


# ## Plotting a Distribution
# 
# Now that we've successfully wrangled some of this data into a DataFrame, let's take it for a spin and make sure we're able to visualize the data without any issues.

# In[ ]:


import seaborn as sns


# In[ ]:


sns.pairplot(df)


# In[ ]:


sns.distplot(df['Student enrollment'],bins=15,kde=False)


# ## Divvying Data using Quantiles
# 
# On a somewhat unrelated note - we can use the student enrollment data to divvy up schools into small, medium, and large schools by getting the quantiles of student enrollment. If we split the student enrollment distribution at its 33rd and 66th quantiles, we'll bin the schools into three groups of three sizes:

# In[ ]:


# Divide the schools into three size bins using quantiles
slice1 = np.percentile(df['Student enrollment'],q=33)
slice2 = np.percentile(df['Student enrollment'],q=66)

def school_size(enrollment):
    if enrollment < slice1:
        return 'Small'
    elif enrollment < slice2:
        return 'Medium'
    else:
        return 'Large'

df['Size'] = df['Student enrollment'].map(lambda x : school_size(x))


# Now we can use that to get a conditional pair plot, with colors corresponding to the (terribly bland) labels of "Small", "Medium", and "Large". 

# In[ ]:


sns.pairplot(df, hue="Size")


# ## Functional Data Processing
# 
# Of the functionality we implemented above, the most useful to abstract away into an object or a function is the process of turning a file into a DataFrame. Furthermore, each file will likely have its own challenges with parsing and processing that will be unique to it.
# 
# This is a task best suited for 8 static methods, for the following reasons:
# * We have a large number of files, and each one has different formatting issues.
# * Each file has different special patterns, possible typos, etc.
# * The only shared information among each file parser is where the data file is located.
# 
# Writing the parsing scripts as functions will help to separate the task of processing data and the task of analyzing data (hence, the reason we don't delve too deeply into the data above, just test out a few plots to make sure the DataFrame we imported is robust).

# ## Parsing Enforcement Data
# 
# We'll start with the first four files, which contain information about law enforcement agencies broken down by agency, campus, city, and county. Each one has a roughly similar structure, so we can share some code, but there are too many particulars to make more code sharing among these methods useful.

# In[ ]:


def ca_law_enforcement_by_agency(data_directory):
    filename = 'ca_law_enforcement_by_agency.csv'

    # Load file into list of strings
    with open(data_directory + '/' + filename) as f:
        content = f.read()

    content = re.sub('\r',' ',content)
    [header,data] = content.split("civilians\"")
    header += "civilians\""
    
    data = data.strip()
    agencies = re.findall('\w+ Agencies', data)
    all_but_agencies = re.split('\w+ Agencies',data)
    del all_but_agencies[0]
    
    newlines = []
    for (a,aba) in zip(agencies,all_but_agencies):
        newlines.append(''.join([a,aba]))
    
    # Combine into one long string, and do more processing
    one_string = '\n'.join(newlines)
    sio = io.StringIO(one_string)
    
    # Process column names
    columnstr = header.strip()
    columnstr = re.sub('\s+',' ',columnstr)
    columnstr = re.sub('"','',columnstr)
    columns = columnstr.split(",")
    columns = [s.strip() for s in columns]

    # Load the whole thing into Pandas
    df = pd.read_csv(sio,quotechar='"',header=None,names=columns)

    return df


df1 = ca_law_enforcement_by_agency('../input/')
df1.head()


# In[ ]:


def ca_law_enforcement_by_campus(data_directory):
    filename = 'ca_law_enforcement_by_campus.csv'

    # Load file into list of strings
    with open(data_directory + '/' + filename) as f:
        lines = f.readlines()
    
    header = ' '.join(lines[:6])
    header = re.sub('\n','',header)
    data = lines[6:]
    
    # Process each string in the list
    newlines = []
    for p in data:
        if( len(re.findall(',,,,',p))==0):
            newlines.append(re.sub(r'^([^"]{1,})(,"[0-9])' ,  r'"\1"\2', p))

    # Combine into one long string, and do more processing
    one_string = '\n'.join(newlines)
    sio = io.StringIO(one_string)

    columnstr = header

    # Get rid of \r stuff
    columnstr = re.sub('\r',' ',columnstr)
    columnstr = re.sub('\s+',' ',columnstr)

    # Fix what can ONLY have been a typo, making this file un-parsable without superhuman regex abilities
    columnstr = re.sub(',Campus','Campus',columnstr)

    columns = columnstr.split(",")

    df = pd.read_csv(sio,quotechar='"',header=None,  names=columns, thousands=',')

    return df


df2 = ca_law_enforcement_by_campus('../input/')
df2.head()


# In[ ]:


def ca_law_enforcement_by_city(data_directory):
    filename = 'ca_law_enforcement_by_city.csv'

    # Load file into list of strings
    with open(data_directory + '/' + filename) as f:
        content = f.read()

    content = re.sub('\r',' ',content)
    [header,data] = content.split("civilians\"")
    header += "civilians\""
    
    data = data.strip()
        
    # Combine into one long string, and do more processing
    one_string = re.sub(r'([0-9]) ([A-Za-z])',r'\1\n\2',data)
    sio = io.StringIO(one_string)
    
    # Process column names
    columnstr = header.strip()
    columnstr = re.sub('\s+',' ',columnstr)
    columnstr = re.sub('"','',columnstr)
    columns = columnstr.split(",")

    # Load the whole thing into Pandas
    df = pd.read_csv(sio,quotechar='"', header=None, names=columns, thousands=',')

    return df


df3 = ca_law_enforcement_by_city('../input/')
df3.head()


# In[ ]:


def ca_law_enforcement_by_county(data_directory):
    filename = 'ca_law_enforcement_by_county.csv'

    # Load file into list of strings
    with open(data_directory + '/' + filename) as f:
        content = f.read()

    content = re.sub('\r',' ',content)
    [header,data] = content.split("civilians\"")
    header += "civilians\""
    
    data = data.strip()
        
    # Combine into one long string, and do more processing
    one_string = re.sub(r'([0-9]) ([A-Za-z])',r'\1\n\2',data)
    sio = io.StringIO(one_string)
    
    # Process column names
    columnstr = header.strip()
    columnstr = re.sub('\s+',' ',columnstr)
    columnstr = re.sub('"','',columnstr)
    columns = columnstr.split(",")

    # Load the whole thing into Pandas
    df = pd.read_csv(sio,quotechar='"',header=None,names=columns,thousands=',')

    return df


df4 = ca_law_enforcement_by_county('../input/')
df4.head()


# ## Parsing Offenses Data
# 
# Now that we've parsed information about law enforcement agencies and how many officers and civilians they employ, we can get to the business of parsing data that tells us how good a job the bobbies are doing - we'll be loading data about criminal offenses.

# In[ ]:


def ca_offenses_by_agency(data_directory):
    filename = 'ca_offenses_by_agency.csv'

    # Load file into list of strings
    with open(data_directory + '/' + filename) as f:
        lines = f.readlines()
    
    one_line = '\n'.join(lines[1:])
    sio = io.StringIO(one_line)
    
    # Process column names
    columnstr = lines[0].strip()
    columnstr = re.sub('\s+',' ',columnstr)
    columnstr = re.sub('"','',columnstr)
    columns = columnstr.split(",")
    
    # Load the whole thing into Pandas
    df = pd.read_csv(sio,quotechar='"',names=columns, thousands=',')

    return df

df5 = ca_offenses_by_agency('../input/')
df5.head()


# In[ ]:


def ca_offenses_by_campus(data_directory):
    filename = 'ca_offenses_by_campus.csv'

    # Load file into list of strings
    with open(data_directory + '/' + filename) as f:
        lines = f.readlines()
    
    # Process each string in the list
    newlines = []
    for p in lines[1:]:
        if( len(re.findall(',,,,',p))==0):
            # This is a weird/senseless/badly formatted line
            if( len(re.findall('Medical Center, Sacramento5',p))==0):
                newlines.append(re.sub(r'^([^"]{1,})(,"[0-9])' ,  r'"\1"\2', p))

    one_line = '\n'.join(newlines)
    sio = io.StringIO(one_line)
    
    # Process column names
    columnstr = lines[0].strip()
    columnstr = re.sub('\s+',' ',columnstr)
    columnstr = re.sub('"','',columnstr)
    columnstr = re.sub(',Campus','Campus',columnstr)
    columns = columnstr.split(",")
    
    # Load the whole thing into Pandas
    df = pd.read_csv(sio, quotechar='"', thousands=',', names=columns)
    
    return df

df6 = ca_offenses_by_campus('../input/')
df6.head()


# In[ ]:


def ca_offenses_by_city(data_directory):
    filename = 'ca_offenses_by_city.csv'

    # Load file into list of strings
    with open(data_directory + '/' + filename) as f:
        content = f.read()
    
    lines = content.split('\n')
    one_line = '\n'.join(lines[1:])
    sio = io.StringIO(one_line)
    
    # Process column names
    columnstr = lines[0].strip()
    columnstr = re.sub('\s+',' ',columnstr)
    columnstr = re.sub('"','',columnstr)
    columns = columnstr.split(",")
    
    # Load the whole thing into Pandas
    df = pd.read_csv(sio,quotechar='"',names=columns,thousands=',')

    return df

df7 = ca_offenses_by_city('../input/')
df7.head()


# In[ ]:


def ca_offenses_by_county(data_directory):
    filename = 'ca_offenses_by_county.csv'

    # Load file into list of strings
    with open(data_directory + '/' + filename) as f:
        lines = f.readlines()
    
    one_line = '\n'.join(lines[1:])
    sio = io.StringIO(one_line)
    
    # Process column names
    columnstr = lines[0].strip()
    columnstr = re.sub('\s+',' ',columnstr)
    columnstr = re.sub('"','',columnstr)
    columns = columnstr.split(",")
    
    # Load the whole thing into Pandas
    df = pd.read_csv(sio,quotechar='"',names=columns,thousands=',')

    return df

df8 = ca_offenses_by_county('../input/')
df8.head()


# ## Summary: Where To From Here
# 
# Now that we've got functions to handle the ugly bits of regular expressions, cleanup, and parsing, we can set to work with analysis.
# 
# We have *many* choices of where to go from here with the analysis. I won't even begin to plot distributions, since, with 8 tables of data, we'll be here all day.
# 
# This notebook provides functions for obtaining each data file provided in this data set as a cleaned up DataFrame with a single function call, making it easier to combine these 8 DataFrames to explore this rich, multivariate dataset. Now get a move on, and get to the good stuff.
