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

shoplist=["apples","bananas","oranges","cherries"]
shoplist[0]

shoplist[0:2]

shoplist.append("blueberries")
shoplist
#append is used to add new data

shoplist[0]="cherries"
shoplist
#to update the data

del shoplist[1]
shoplist
#to delete the data 

#index starts from 0 (left to right)
#index starts from -1 (right to left)



shoplist2=["mango","papaya"]
shoplist+shoplist2


listNum=[1,4,28,9,56]
min(listNum)
max(listNum)
