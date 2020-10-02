#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time
from datetime import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/b3-stock-quotes"]).decode("utf8"))
print(check_output(["ls", "."]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# The original files from the Brazillian Stock Exchange are not comma separated files. This Kernel will decode the files and generate a csv file to facilitate the use in others Kernels. The file format specification is available in this [link](http://www.bmfbovespa.com.br/lumis/portal/file/fileDownload.jsp?fileId=8A828D294E9C618F014EB7924B803F8B) (Portuguese only...).

# In[ ]:


# Here we'll specify the columns width
widths = [2, 8, 2, 12, 3, 12, 
          10, 3, 4, 13, 13, 13, 13, 13, 13, 13, 5, 18, 
          18, 13, 1, 8, 7, 13, 12, 3]

# and the columns names, based in the original specification
column_names = ['TIPREG', 'DATPRE', 'CODBDI', 'CODNEG', 'TPMERC', 'NOMRES', 'ESPECI', 'PRAZOT', 
                'MODREF', 'PREABE', 'PREMAX', 'PREMIN', 'PREMED', 'PREULT', 'PREOFC', 'PREOFV', 
                'TOTNEG', 'QUATOT', 'VOLTOT', 'PREEXE', 'INDOPC', 'DATVEN', 'FATCOT', 'PTOEXE', 
                'CODISI', 'DISMES']


# In[ ]:


# Most of the prices are defined with two decimals. 
# This function is used to adjust this while loading...
def convert_price(s):
    return (float(s) / 100.0)

# The date fields are in the format YYYYMMDD
def convert_date(d):
    struct = time.strptime(d, '%Y%m%d')
    dt = datetime.fromtimestamp(time.mktime(struct))
    return(dt)


# In[ ]:


# Specify dtype while loading
dtype_dict = {
    'TOTNEG':np.int32
}

# Use the functions defined above to convert data while loading
convert_dict = {
    'DATPRE':convert_date, 
    'PREABE':convert_price, 'PREMAX':convert_price, 
    'PREMIN':convert_price, 
    'PREMED':convert_price, 'PREULT':convert_price, 'PREOFC':convert_price, 
    'PREOFV':convert_price,
    'DATVEN':convert_date, 
}


# In[ ]:


# Load the raw file
def load_and_preprocess(file_path):
    df = pd.read_fwf(
        file_path, 
        widths=widths, 
        names=column_names, 
        dtype=dtype_dict, 
        converters=convert_dict,
        #compression='zip',
        skiprows=1,              # Skip the header row
        skipfooter=1             # Skip the footer row
    )
    return(df)


# In[ ]:


import zipfile

# Will unzip the files so that you can see them..
with zipfile.ZipFile("../input/b3-stock-quotes/COTAHIST_A2009.ZIP", "r") as z:
    z.extractall(".")
with zipfile.ZipFile("../input/b3-stock-quotes/COTAHIST_A2010.ZIP", "r") as z:
    z.extractall(".")
with zipfile.ZipFile("../input/b3-stock-quotes/COTAHIST_A2011.ZIP", "r") as z:
    z.extractall(".")
with zipfile.ZipFile("../input/b3-stock-quotes/COTAHIST_A2012.ZIP", "r") as z:
    z.extractall(".")
with zipfile.ZipFile("../input/b3-stock-quotes/COTAHIST_A2013.ZIP", "r") as z:
    z.extractall(".")    
with zipfile.ZipFile("../input/b3-stock-quotes/COTAHIST_A2014.ZIP", "r") as z:
    z.extractall(".")    
with zipfile.ZipFile("../input/b3-stock-quotes/COTAHIST_A2015.ZIP", "r") as z:
    z.extractall(".")    
with zipfile.ZipFile("../input/b3-stock-quotes/COTAHIST_A2016.ZIP", "r") as z:
    z.extractall(".")    
with zipfile.ZipFile("../input/b3-stock-quotes/COTAHIST_A2017.ZIP", "r") as z:
    z.extractall(".")    
with zipfile.ZipFile("../input/b3-stock-quotes/COTAHIST_A2018.ZIP", "r") as z:
    z.extractall(".")    
with zipfile.ZipFile("../input/b3-stock-quotes/COTAHIST_A2019.ZIP", "r") as z:
    z.extractall(".")    


# Read all files and concatenate in one Dataframe
df1 = load_and_preprocess("./COTAHIST_A2009.TXT")
df2 = df1.append(load_and_preprocess("./COTAHIST_A2010.TXT"), ignore_index=True)
df3 = df2.append(load_and_preprocess("./COTAHIST_A2011.TXT"), ignore_index=True)
df4 = df3.append(load_and_preprocess("./COTAHIST_A2012.TXT"), ignore_index=True)
df5 = df4.append(load_and_preprocess("./COTAHIST_A2013.TXT"), ignore_index=True)
df6 = df5.append(load_and_preprocess("./COTAHIST_A2014.TXT"), ignore_index=True)
df7 = df6.append(load_and_preprocess("./COTAHIST_A2015.TXT"), ignore_index=True)
df8 = df7.append(load_and_preprocess("./COTAHIST_A2016.TXT"), ignore_index=True)
df9 = df8.append(load_and_preprocess("./COTAHIST_A2017.TXT"), ignore_index=True)
df10 = df9.append(load_and_preprocess("./COTAHIST_A2018.TXT"), ignore_index=True) # New file with full 2018 data (previous was partial)
df  = df10.append(load_and_preprocess("./COTAHIST_A2019.TXT"), ignore_index=True) # New file with full 2019 data


# In[ ]:


pd.set_option('display.max_columns', 26)
df.head()


# In[ ]:


df.to_csv('COTAHIST_A2009_to_A2019.csv')

