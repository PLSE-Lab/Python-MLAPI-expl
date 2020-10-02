#!/usr/bin/env python
# coding: utf-8

# In[46]:


import csv
import sys
import os
import pandas as pd
import numpy as np


# In[47]:


job_dir='../input/cityofla/CityofLA/Job Bulletins/'
listOfFile = os.listdir(job_dir)
listOfFile


# In[51]:


txt_input = '../input/cityofla/CityofLA/Job Bulletins/SENIOR HOUSING INSPECTOR 4244 042718.txt'
csv_output ='../mycsv.csv'
with open(txt_input, 'r') as infile, open(csv_output, 'w') as outfile:
     stripped = (line.strip() for line in infile)
     lines = (line.split(",") for line in stripped if line)
     writer = csv.writer(outfile)
     writer.writerows(lines)
print('OK')


# In[ ]:





# In[ ]:




