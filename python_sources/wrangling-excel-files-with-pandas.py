#!/usr/bin/env python
# coding: utf-8

# As a consultant, it's very common for me to get client data in the form of many Excel files. We've probably all seen the thing where each department is a different folder, each year a different file and each month a different tab. Piled on top of this file jungle is often "data" that's not really data. Descriptions, footnotes, sums and blank columns are part of the fun obstacles in our way.
# 
# I'm  learning that Python is our friend here. There are some great modules out there with advanced capabilites like [openpyxl](https://openpyxl.readthedocs.io/en/default/). We can also use Pandas with it's very nice pd.read_excel function, which I'll do here. It's less flexible than a dedicated Excel package, but surprisingly capable for most of the processing you'll need.
# 
# I'll start with two use cases. The first one is to select usable data from two different workbooks. I use this data in another kernel on [Benefits for Kids](https://www.kaggle.com/jpmiller/looking-at-benefits-for-kids). The second use case is to get multiple csv files from a workbook with several tabs.
# 
# #### Part 1. Selecting Usable Data
# 
# My desired end state here is to get a nice tidy dataframe in long format for further analysis. 

# In[ ]:


# %autosave 600
import os
import numpy as np
import pandas as pd
from glob import glob


# We can use Python to scan files in a directory and pick out the ones we want. Here I'll get a list of the WIC filenames. 

# In[ ]:


wkbks = glob(os.path.join(os.pardir, 'input', 'xls_files_all', 'WIC*.xls'))
sorted(wkbks)


# Taking a look at the WIC files, I'll identify which parts to pull in. I'm also noting which rows to skip and all that. Here's the file for 2012. Notice the Regional Subtotals typical of most reports.
# ![excel](https://s3.amazonaws.com/nonwebstorage/excel.png)

# I've decided to grab three specifc sheets from each workbook and combine them into one data frame. 
# 
# NOTE: The current version of Pandas uses 'sheet_name', not 'sheetname' as you see below. 

# In[ ]:


shts = ['Total Infants', 'Children Participating', 'Food Costs']

wic = pd.DataFrame()
for w in wkbks:
    for s in shts:
        frame = pd.read_excel(w, sheetname=s, skiprows=4
                  , skip_footer=4)
        frame['Type'] = s
        frame = frame.melt(id_vars=['Type', 'State Agency or Indian Tribal Organization'])
        wic = wic.append(frame, ignore_index=True)

print(wic.shape)
print(wic.columns)


# Now it's a matter of cleaning up the dataframe. I want to have dates and numbers of the right type and get rid of the region totals, yearly averages, and so on.

# In[ ]:


# clean up
wic.columns=['Type', 'Area', 'Month', 'Value']
wic = wic[(wic.Area.str.contains('Region') == False)].copy()
wic['Month'] = pd.to_datetime(wic['Month'], errors='coerce')
wic = wic[wic.Month.notnull()]
wic['Value'] = pd.to_numeric(wic['Value'], downcast='integer')
wic = wic[wic.Value.notnull()]
wic.head()


# Much better! Now I want to grab some census data. I'm only interested in the number of children under the poverty line in each state for each year.

# In[ ]:


wkbks = glob(os.path.join(os.pardir, 'input', 'xls_files_all', 'est*.xls'))
sorted(wkbks)


# Looking at the 2013 Workbook, we see there is only one worksheet and it's not too bad. Here I used Libre Calc to open the Excel file.
# 
# ![Census](https://s3.amazonaws.com/nonwebstorage/libre.png)
# 
# Again we'll use the append method to create a single frame.

# In[ ]:


pov = pd.DataFrame()
y=2013
for w in wkbks:
        frame = pd.read_excel(w, skiprows=3, usecols=[2,26])
        frame['Year'] = y
        pov = pov.append(frame, ignore_index=True)
        y=y+1
        


# The last step is to save the usable dataframes to files and move on to some real analysis!

# In[ ]:


wic.to_csv('wicdata.csv', index=False)
pov.to_csv('povertydata.csv', index=False)


# #### Part 2. Creating Multiple Data Files
# 
# Here we will rip all tabs of interest from a set of Excel files into separate CSVs for easier consumption. I'll use the WIC files for this example.

# In[ ]:


# get a list of all excel files in the folder
wkbks = glob(os.path.join(os.pardir, 'input', 'xls_files_all', 'WIC*.xls'))
sorted(wkbks)


# Here are the sheet names I wanted to use. There are shorter ways to do this, but I wanted to explicitly name the worksheets for clarity.

# In[ ]:


shts =['Pregnant Women Participating',
     'Women Fully Breastfeeding',
     'Women Partially Breastfeeding',
     'Total Breastfeeding Women', 
     'Postpartum Women Participating',
     'Total Women', 
     'Infants Fully Breastfed',
     'Infants Partially Breastfed',
     'Infants Fully Formula-fed',
     'Total Infants', 
     'Children Participating',
     'Total Number of Participants', 
     'Average Food Cost Per Person',
     'Food Costs', 
     'Rebates Received',
     'Nut. Services & Admin. Costs']


# Now it's just a matter of iterating through the files. You can save tabs from each file to a separate directory like I do here, or zip them together or combine them into a single frame like we do above. 

# In[ ]:


year = 2013

for w in wkbks:
    savedir = os.path.join(os.pardir, str(year))
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    year = year+1
    for s in shts:
        frame = pd.read_excel(w, sheet_name=s, skiprows=4
                  , skip_footer=4)
        frame.to_csv(os.path.join(savedir, '{0}.csv'.format(s)), index=False)
        
        


# This code throws an error on Kaggle because we can't create subdirectories from code. It should work on your local machine, however, and make a whole bunch of  csv's organized by folder.
# 
# And that's it. Now you can save countless hours when faced with a folder full of Excel files!

# In[ ]:




