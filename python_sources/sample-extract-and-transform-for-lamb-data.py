#!/usr/bin/env python
# coding: utf-8

# # Boilerplate code 
# 
# First, we need to get our environment set up and read in every thing we'll need. 

# In[ ]:


# install magic module
get_ipython().system(' pip install python-magic')


# In[ ]:


# import module we'll need to import our custom module
from shutil import copyfile

# copy our file into the working directory (make sure it has .py suffix)
copyfile(src = "../input/sheep-data-validation-script/virginia-lamb-data-validation-script.py", dst = "../working/script.py")

# import all our functions
from script import *


# # Data validation
# 
# This step (which we talked about yesterday) lets you check to make sure your data meets your expectations. 
# 
# Here, I've written a little Python script for data validation ([you can see a copy here](https://www.kaggle.com/rtatman/virginia-lamb-data-validation-script?scriptVersionId=10091127)), imported it in the code cell above and am using it to quickly validate my data. 

# In[ ]:


# use the script we imported for data validation 
# you can see a copy of the script here
check_lamb_data("../input/lamb-auction-data/")


# # Data cleaning
# 
# Now that I've validated my data and see that everything looks ok, I can do my data cleaning! These files have a lot of information on Cattle in them, but I'm only interested in the number of sheep sold. I can get that information from these flat text files with a little bit of general data munging, like so:

# In[ ]:


import pandas as pd

# create an empty dataframe for our clean data
cleaned_data = pd.DataFrame()

# we'll use these labels in each loop, so it's more efficient
# to declare them out of the loop
column_labels = ["category","head_sold"]

# loop through every .txt file in our target directory
for file in glob.glob("../input/lamb-auction-data/*.txt"):
   
    # read in all the data for a specific file (they're small
    # so this shouldn't be a big problem)
    data = open(file).read()
   
    # find # of head sold
    sales = re.findall('(lamb|ram|ewe|wether|sheep).* (.*) head', data.lower())
    
    # name of market (always on fourth line)
    market = data.split(sep="\n")[3]

    # date reported
    reported = re.findall("Richmond, VA(.*)", data)
    date = dparser.parse(reported[0],fuzzy=True).date()

    # only proceed from here if there was one or more sales
    if len(sales) > 0:
        
        df_temp = pd.DataFrame.from_records(sales, columns=column_labels)
        df_temp["market"] = market
        df_temp["date_reported"] = date

    # add data about markets with no sheep sales
    else:
        df_temp = pd.DataFrame(data={"category": ["none"], "head_sold": [0]})
        df_temp["market"] = market
        df_temp["date_reported"] = date
    
    # append data to our cleaned data
    cleaned_data = cleaned_data.append(df_temp, ignore_index=True)


# In[ ]:


# now let's take a peek at the cleaned data and make sure it looks good
cleaned_data


# # Save out our data
# 
# Now that our data is all cleaned and good to go, we can save it out as its own file.
# 
# > **Note**: You should save your data to the directory "/kaggle/working/". This ensures that when you commit your kernel, your data will be saved as output. 
# 
# With our file saved, we can commit our kernel using the "Commit" button. This will run our code top to bottom, save the current version of our kernel, and create an output file we can use to create a new dataset from.

# In[ ]:


# save our data as a .csv file
cleaned_data.to_csv("/kaggle/working/cleaned_lamb_data.csv", index=False)


# 
