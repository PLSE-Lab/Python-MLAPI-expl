# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Important prerequisites : when using this script, you need to
# 1. Install the python package mstrio-py
# 2. Enable internet (Kaggle beta feature)

# mstr initialization
mstr_username = 'administrator' #changeme
mstr_password = '' #changeme
mstr_library_api_url = 'http://hp:8080/MicroStrategyLibrary/api' #changeme
mstr_project_name = 'MicroStrategy Tutorial'

import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mstrio import microstrategy # MicroStrategy library for sending data to in-memory cubes

mstr_conn = microstrategy.Connection(base_url=mstr_library_api_url, username=mstr_username, password=mstr_password, project_name=mstr_project_name)
print('1. Connecting to MicroStrategy')
mstr_conn.connect()

# Loop through all datasets added to the Kaggle Kernel
print('2. Starting pushing datasets')
for dirname, dirnames, filenames in os.walk('../input'):
    # loop through all files included in the dataset
    print('3. Dataset: '+dirname)
    for filename in filenames:
        print('  Filename: '+os.path.join(dirname, filename))
        dataset_name = filename.replace('.csv', '') #your login goes here
        print('  Target Cube Name: '+dataset_name)
        table_name = dataset_name
        newDatasetId, newTableId = mstr_conn.create_dataset(data_frame=pd.read_csv(os.path.join(dirname, filename)), dataset_name=dataset_name, table_name=table_name)
   # print('3. End of Dataset')
#print('4. All datasets sent')
#mstr_conn.close()
#print('5. Disconnected from server. End of script')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.