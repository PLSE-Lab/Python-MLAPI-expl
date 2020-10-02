#!/usr/bin/env python
# coding: utf-8

# This file is in reference to discussion topic : 
# https://www.kaggle.com/rohanrao/chai-time-data-science/discussion/163343
# ( Since  the 2nd and 3rd line for every file ( in "Cleaned Subtitles" ) ( E(1-45).csv has been merged with the 1st line, ending up with 2 lines missing from every E*.csv file. 
# 
# Below is a quick and dirty way to create csv files with all lines present. 
# And removing the first line . ( which is common across all subtitles files ) 
# 

# In[ ]:


import re 
import os 
import pandas as pd 


# Create a list of all the raw files

# In[ ]:


DIR = '../input/chai-time-data-science/Raw Subtitles/'
L_fname_txt = os.listdir(DIR)
L_fname_csv = [k.replace('.txt','.csv') for k in L_fname_txt]   # create a list of same files with .csv ext


# In[ ]:


#Just a function to write one line to a file - not the efficient way, but yeah works 
def Fn_write_line(sLine,fname):
    fname = '../working/'+fname
    target = open(fname, 'a')
    target.write(sLine)
    target.close()


# Read all the txt files (loop) in 'Raw Subtitles' and write them to '/working' as csv files.

# In[ ]:


for fname_txt in L_fname_txt:
    fname = open(DIR+fname_txt, 'r') 
    Lines = fname.readlines() 
    pattern1 = "  [0-9]+:[0-9]+  "   # time pattern eg: 10:43
    pattern2 = "  [0-9]+:[0-9]+:[0-9]+  " # time pattern eg: 1:00:14  

    for line in Lines: 
        if (re.search(pattern1,line)) or (re.search(pattern2,line) ):  # line [8]
            line = line.replace("  ","_-_").replace("\n","")  # line [9] - See comments below
        if len(line) >1 :    # Line [10]
            Fn_write_line(line,fname_txt)
            df = pd.read_csv( '../working/'+fname_txt,sep = "_-_" , header = None)
            df.columns = ['speaker','time','text']
            df = df[1:-1:]   # line [14]
            df.to_csv('../working/'+fname_txt.replace('.txt','.csv'),index=False)    


# #### Comments for above code block
# Line [8] if any of the pattern occurs, do some replacements
# Line [9] if the line has any 'time' pattern, remove the \n and replace "  " with _-_ ( or something unique) <br>
# Line [10] Skip that weird empty line ( which has a carriage return / newline) <br>
# line [14] Skip the first line in every txt file ( which seems to be common across all files )

# Check a random file , here 'E1.csv' , to see the output

# In[ ]:


L_fname_csv


# In[ ]:


L_fname_csv[18]


# In[ ]:


df = pd.read_csv( '../working/'+ L_fname_csv[18] )
df.head()


# eg : In E1.csv , we got back that missing lines at 0:46 and 1:38 into the dataframe

# In[ ]:




