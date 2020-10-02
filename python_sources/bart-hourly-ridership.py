#!/usr/bin/env python
# coding: utf-8

# # BART Hourly Ridership Data Downloader

# Bay Area Rapid Transit (BART) hourly ridership by origin-destination can be pulled from BART's provided source: https://www.bart.gov/about/reports/ridership

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# Typing ftw
from typing import List

import requests # For scraping from original source
import re # To pull out specific URLs in scraping


# In[ ]:


# Flag to run with debugging information (printouts)
DEBUG = False


# In[ ]:


def get_origin_destination_urls(url_origin_destination: str = 'http://64.111.127.166/origin-destination', 
               ext: str ='csv.gz', use_full_path: bool = False, 
               printout: bool = False) -> List:
    '''Return list of URLs of files (from given extension).
        
        Args: 
            url_origin_destination: Source to scrape file URLs from.
            ext: Extension of files to search for (defaults "csv.gz").
            use_full_path: Flag to return full path of files or relative path.
            printout: Prints out information for basic debugging.
    '''
    
    resp = requests.get(url_origin_destination)
    # DEBUG
    if printout:
        print(resp.text)
        
    # Pattern to pull from HTML links for files
    href_pattern = r'<a\s+(?:[^>]*?\s+)?href=(["\'])(\S*\.{})\1+>'.format(ext)

    # More efficient to compile (https://docs.python.org/3/library/re.html#re.compile)
    prog = re.compile(href_pattern)
    file_group_list = prog.findall(resp.text)

    # Decide to return full path in list of files
    file_list = [group[-1] for group in file_group_list]
    if use_full_path:
        return [f'{url_origin_destination}/{fname}' for fname in file_list]
    else:
        return file_list
    


# In[ ]:


# Example usage
if DEBUG:
    csv_list = get_origin_destination_urls(use_full_path=True)
    df = pd.read_csv(csv_list[0], 
                     names=['Date','Hour','Origin Station','Destination Station','Trip Count']
                    )


# In[ ]:


# Retrieve the README (also will grab all other `txt`) files
txt_list = get_origin_destination_urls(use_full_path=True, ext='txt')
for txt_url in txt_list:
    resp_txt_urls = requests.get(txt_url)
    # Remove HTML escaped spaces
    export_fname = txt_url.split('/')[-1].replace('%20','')
    with open(export_fname,'wb') as f:
        f.write(resp_txt_urls.content)


# In[ ]:


# Directory for all data
try:
    os.mkdir('ridership')
except:
    print('New directory not made')


# In[ ]:


# Using the definition of columns from "READ ME.txt"
col_names = ['Date','Hour','Origin Station','Destination Station','Trip Count']

# Retrieve all CSVs (`csv.gz`)
csv_list = get_origin_destination_urls(use_full_path=True)

# Iterated over each URL to create a separate file (for each year)
for csv_fname in csv_list:
    df = pd.read_csv(csv_fname, names=col_names)
    export_fname = csv_fname.split('/')[-1]#.replace('.gz','')
    # Ignore the default index (useless for this dataset)
    df.to_csv(f'ridership/{export_fname}', index=False, compression='gzip')

