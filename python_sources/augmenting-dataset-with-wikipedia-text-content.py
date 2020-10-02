#!/usr/bin/env python
# coding: utf-8

# ### What is this kernel for?
# To demo how to easily add text content from wikipedia to our dataset.
# 
# This might allow us to extract some interesting features to use in the competition. I might explore this idea in another Kernel.
# 
# 
# ---------
# __Note__ Unfortunately Kaggle blocks Internet access so this kernel won't retrieve the data but it should run just fine outside Kaggle.
# 
# 

# In[ ]:


import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm, tqdm_notebook
import time
import requests


SLEEP_TIME_S = 0.1


# In[ ]:


def extract_URL_and_Name(page):
    """ From the page name in the input file extract the Name and the URL """
    return (['_'.join(page.split('_')[:-3])]
            + ['http://' + page.split("_")[-3:-2][0] +
               '/wiki/' + '_'.join(page.split('_')[:-3])])


# In[ ]:


# Load the dataset
train = pd.read_csv('../input/train_1.csv')

# We will just take a sample of the data, 
# remove this line to run on all the data
train = train.sample(2)

# Extract the Page name and URL:
page_data = pd.DataFrame(
    list(train['Page'].apply(extract_URL_and_Name)),
    columns=['Name', 'URL'])


# In[ ]:


page_data.head()


# In[ ]:


# Since Kaggle kernels don't have internet access this method will always return 
# an empty string
def fetch_wikipedia_text_content(row):
    """Fetch the all text data of a given page"""
    try:
        r = requests.get(row['URL'])
        # Sleep for 100 ms so that we don't use too many Wikipedia resources 
        time.sleep(SLEEP_TIME_S)
        to_return = [x.get_text() for x in 
                     BeautifulSoup(
                         r.content, "html.parser"
                     ).find(id="mw-content-text").find_all('p')]
    except:
        to_return = [""]
    return to_return


# In[ ]:


# This will fail due to lack of Internet
tqdm.pandas(tqdm_notebook)
page_data['TextData'] = page_data.progress_apply(fetch_wikipedia_text_content, axis=1)

page_data.head()

