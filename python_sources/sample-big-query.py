# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data",dataset_name = "hacker_news")
hacker_news.list_tables();

hacker_news.head("full")


#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.