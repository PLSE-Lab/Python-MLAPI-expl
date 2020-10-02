# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
# name the dataset

dummydata = '/kaggle/input/dummy-data/Dummy data.csv'

# %% [code]
# read the dataset, define dataset to be df

df=pd.read_csv(dummydata)

# %% [code]
# show the first 5 rows of dataset

df.head()

# %% [code]
# plus one to the second column

df['number']=df['number']+1


# %% [code]
# display the file again

df.head()


# save the submission file
df.to_csv('/kaggle/working/df.csv',index=False)