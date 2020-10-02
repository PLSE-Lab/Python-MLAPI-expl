#!/usr/bin/env python
# coding: utf-8

# # 01 Exploring The Folder-Structure
# 
# This notebook summarizes the structure in which the data is presented.  
# You can use it to get an overview of where to find the main files and how the articles have been grouped.
# 
# 
# # 1. Import Libraries

# In[ ]:


# To search directories
import os

# To store the data
import pandas as pd


# # 2. Inspect The Directories

# In[ ]:


# Variable to store all files
files = []

# Iterate over all files and subfolders
for dirname, _, filenames in os.walk('/kaggle/input'):
    
    # Iterate all filenames
    for filename in filenames:
        
        # Store the filename for later inspection
        files.append(os.path.join(dirname, filename))

print('{} files are in the directories.'.format(len(files)))


# Split the filenames into subfolders and filename
# Remove the first three folders (home, kaggle and input) since they do not add new information
files_split = [file.split('/')[3:] for file in files]


# Store the split files as DataFrame to get aggregated summaries 
df = pd.DataFrame(files_split, columns=['Folder_Depth_0', 'Folder_Depth_1', 'Folder_Depth_2', 'Folder_Depth_3', 'Folder_Depth_4'])

print('\nThese are some sampled entries:')
df.sample(3)


# In[ ]:


# Group the folders and count the number of entries
print('Here you can see the main files and the count of the articles in the subfolders:')
df.groupby(['Folder_Depth_0', 'Folder_Depth_1', 'Folder_Depth_2']).size()


# The **zeroth folder-level** ("CORD-19-research-challenge") contains all subsequent files and is **named after the challenge**.
# 
# The **first folder-level** ("2020-03-13") contains all subsequent files and seems to be **named after the date of the data**.
# 
# The **second folder-level** contains **one file for the metadata of all articles** ("all_sources_metadata_2020-03-13.csv")  
# and **three files to explain the data** ("COVID.DATA.LIC.AGMT.pdf", "all_sources_metadata_2020-03-13.readme" and "json_schema.txt").  
# Furthermore it contains **four folders with the json-articles** separated by their license and origin.

# In[ ]:




