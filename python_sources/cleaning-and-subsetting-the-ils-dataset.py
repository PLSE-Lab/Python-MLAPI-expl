#!/usr/bin/env python
# coding: utf-8

# ## Introduction:
# 
# Our Dataset is about 3.6GB on disk, and "wc -l datafile.csv" tells us that we have 37 768 482 rows. Each line in the file takes up about 3 lines, so we have about 12589494 potential rows to access, in the dataset. 
# 
# I had problems trying to load this dataset with the Dask framework. Trying to load the entire dataset using pandas on a small laptop was also painful. I make use of the 16Gb of RAM a Kaggle Kernel provides.
# 
# **Note:** This script is not currently optimized for efficiency. It takes about ~300s to run. 
# 
# ## Goal: 
# 
# Tidy the dataset, and create a smaller dataframe with redundant information stored in separate files. Users can join this information if they need to. 

# In[ ]:


#Imports:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Settings:
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
alt.data_transformers.enable('default', max_rows=None)
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 1000)


# In[ ]:


#Support functions:

#for each string, and a compiled pattern object, use the findall method.
#return true if we find ONE SINGLE 5 digit zipcode. False otherwise. 
def regapp(x,pattObj):
    hold = pattObj.findall(x)
    result = False
    if (len(hold) == 1):
        result = True
    return result

def cleanup(cell):
    return float(cell.replace("$",""))

def cutgps(cell,myregex):
    store = re.findall(myregex,cell)
    if (store): #empty
         retVal = store[0]
    else:
         retVal = "NA" #Some of the entries are missing their lat/long data
    return retVal

#first, lets extract the GPS coordinates:
#What can we assume? each column is a non-empty string, at least.
def getlatlong(cell, pos):
    if (cell == "NA"):
        return cell
    sectionList = cell.replace("(","").replace(")","").split(",")
    return float(sectionList[pos]) #0 or 1. Float will clip out spaces for us!


# ## Data Loading:

# In[ ]:


dfILS = pd.read_csv("../input/Iowa_Liquor_Sales.csv")
dfILS.columns.values


# ## Data Tidying:

# In[ ]:


#we will use this to calculate the percentage of rows lost after cleaning.
lossDict = {}
lossDict['fullsize'] = dfILS.shape[0] 


# Lets get some basic information about our data frame. We can already see some columns are upcast (float instead of ints).

# In[ ]:


dfILS.info()


# Checking out nulls:

# In[ ]:


dfILS.isnull().sum() #our number of nas.


# In[ ]:


#We should be able to drop NAs, and still have over 12M rows to choose from.
#This loss is acceptable.
dfILS.dropna(inplace=True)
lossDict['nullloss'] = (lossDict['fullsize']- dfILS.shape[0])


# Our Column Names need to be tidied up.

# In[ ]:


nameDict = {"Invoice/Item Number":"invoicenumber"
,"Date":"date"
,"Store Number":"storenumber"
,"Store Name":"storename"
,"Address":"address"            
,"City":"city"
,"Zip Code":"zipcode"
,"Store Location":"storelocation"
,"County Number":"countynumber"
,"County":"countyname"
,"Category":"categorynumber"
,"Category Name":"categoryname"
,"Vendor Name":"vendorname"
,"Vendor Number":"vendornumber"
,"Item Number":"itemnumber"
,"Item Description":"itemdescription"
,"Pack":"pack"
,"Bottle Volume (ml)":"bottlevolumeml"
,"State Bottle Cost":"statebottlecost"
,"State Bottle Retail":"statebottleretail"
,"Bottles Sold":"bottlessold"
,"Sale (Dollars)":"saleprice"
,"Volume Sold (Liters)":"volumesoldlitre"
,"Volume Sold (Gallons)":"volumesoldgallon"}

dfILS.rename(columns = nameDict,inplace=True)


# Pandas has a tendency to upcast a lot of the columns. We need to make the datatypes more specific (example: float64 -> int32). 
# countynumber, category, vendornumber, and zipcode are cast as floats or string objects. Observe below:

# In[ ]:


dfILS.tail(3)


# In[ ]:


#The following are easy to correct. #the ints are not that big, so we can use int32 instead of 64 to save space.
convertList = ["countynumber","vendornumber","storenumber","categorynumber","pack","bottlevolumeml","bottlessold",
               "volumesoldlitre","volumesoldgallon","itemnumber"]

for item in convertList:
    dfILS = dfILS.astype({item: "int32"},inplace=True)    


# ### Dealing with Zipcodes:
# 
# Zipcode is classed as a string object type, because of zipcode anomolies in the data. There are zipcodes of the form "752-6". Casting occured because of non-numeric characters. We need to cut out rows that don't conform to a 5 digit code, before casting to int.

# In[ ]:


dfILS['zipcode'].unique() #if you want to look for yourself.


# There appear to be *non numeric characthers ("712-2"), floats, strings and ints all mixed together*. Yikes. Upcast to strings,
# filter anomolies, and then convert whats left to ints.

# In[ ]:


dfILS['zipcode'] = dfILS['zipcode'].apply(str)


# In[ ]:


pattObj = re.compile(r"[0-9]{5}")
boolSelect = dfILS['zipcode'].apply(regapp,args=(pattObj,))
lossDict['ziploss'] = boolSelect.value_counts().loc[False]
dfILS = dfILS[boolSelect]


# We have to step down type incrementally. string '7723632.0' throws a ValueError.

# In[ ]:


dfILS['zipcode'] = dfILS['zipcode'].apply(float)
dfILS['zipcode'] = dfILS['zipcode'].apply(int) 
dfILS = dfILS.astype({"zipcode": "int32"}, inplace=True) 
#done!


# ### Dealing with Sales Columns:

# Next, we need to clean up the sales columns. They are strings because the dollar sign symbol was included in the spreadsheet. Again,
# the numbers in sales aren't that large, so lets use a float32 instead of a float64 to save some space. 

# In[ ]:


for columnname in ["statebottlecost","statebottleretail","saleprice"]:
    dfILS[columnname] = dfILS[columnname].apply(cleanup)
    dfILS.astype({columnname:"float32"},inplace=True)
    
#dfILS.info() #check to see that the three cols are now floats.


# ### Separating GPS coordinates from the Store Location Column:
# 
# #### Note: This code was unused for the final dataframe (I didn't need it). Uncomment code below to use GPS coordinates.
# 
# Next we will split the Store Location Column. There appears to be zipcode and GPS information encoded in these columns. Individual addresses for a store don't matter, as this dataset needs to be aggregated at a larger geographical area to do reasonable modelling. The storelocaiton column will be split (and dropped). We will add a latitude and longitude column, instead.

# In[ ]:


#First, mutate the store location column; replace it with just the GPS substring.
#myregex = r"(\(.+,.+\))"
#dfILS['storelocation'] = dfILS['storelocation'].apply(cutgps,args=(myregex,))


# In[ ]:


#suppressed as I don't need this for tableau!
#dfILS['latitude'] = dfILS['storelocation'].apply(getlatlong,args=(0,))
#dfILS['longitude'] = dfILS['storelocation'].apply(getlatlong,args=(1,))


# In[ ]:


dfILS.drop(columns=["storelocation"], axis=1,inplace=True)


# Finally, our GPS conversion has introduced some NAs - as not every storelocation had GPS coordinates. Lets check string NAs.

# In[ ]:


#boolSelect2 = dfILS['latitude'] == "NA"
#boolSelect2.value_counts() #our number of nas.


# 931655 NAs is roughly 10 percent of our data. Should they be dumped? I choose not to. I'll deal with "NA"s further down the pipeline. 

# ### Summary and Check of Data Tidying:
# 
# We have saved some memory by casting the columns to smaller types.

# In[ ]:


dfILS.info()


# In[ ]:


rowsum = lossDict['nullloss'] + lossDict['ziploss']
print("Rows Lost: " + str(rowsum) + "\n Percentage Loss: " + str(rowsum/lossDict['fullsize']))


# ## Subsetting and Saving our Data:

# Now that our data is tidy, we can subset and save it to .csv files. There are some examples below:

# In[ ]:


dfILS.head(5)


# ## Putting Redundant Information into Separate Tables:
# 
# We can shrink this data frame significantly, by putting coupled elements in a separate table, that can be joined by the user
# at a later date. In particular:
# 
# - store number and store name
# - county number and county name
# - category and category name
# - vendor number and vendor name
# - item number and item description
# 
# We can just store the integer number column, and store a reduced table of unique values in a separate file. Via a left join, we can 
# reconstruct the data if needed.

# In[ ]:


def uniquefilewrite(colA,colB,filename):
    storage = dfILS.drop_duplicates(subset=[colA,colB],keep="first",inplace=False)
    storage.loc[:,[colA,colB]].to_csv(filename + ".csv",index=False)
    return


# In[ ]:


colA = ["storenumber","countynumber","categorynumber","vendornumber","itemnumber"]
colB = ["storename","countyname","categoryname","vendorname","itemdescription"]

for tup in list(zip(colA,colB)):
    uniquefilewrite(tup[0],tup[1],tup[1])
    


# ### Writing out our dataframe
# 
# The data is now tidy and free of redundant data.

# ![](http://)Kaggle Session information indicates our data takes up < 1.4GB of space. Much better!

# In[ ]:


colList = ["invoicenumber","date","city", "zipcode", "storenumber","countynumber",
           "categorynumber", "vendornumber","itemnumber","bottlevolumeml","statebottlecost",
            "statebottleretail", "bottlessold", "saleprice" ,"volumesoldlitre"]

dfILS.loc[:,colList].to_csv("ILS_clean.csv",index=False)

