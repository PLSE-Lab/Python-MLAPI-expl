# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# TODO: https://stackoverflow.com/questions/24251219/pandas-read-csv-low-memory-and-dtype-options to be used for an error in data types...
dataset2017 = pd.read_csv("../input/2017-fCC-New-Coders-Survey-Data.csv") 

print(dataset2017.head())
print()
print()
print("columns about resources: ")
for c in dataset2017.columns:
    if "Resource" in c:
        print(c)
print()
print()
print("number of examples ", len(dataset2017))
print("evaluating just one column: ResourceCodeacademy \n",dataset2017['ResourceCodecademy'].describe(),"\n data invoked as 'missing' : ", dataset2017['ResourceCodecademy'].isnull().sum())
print()
print()
print("The designer assumed that if the user didn't fill the answer, he/she didn't use the resource.\nThis assumption is incorrect but we will use it anyway.")
print()
print()
print("porcentage of users that might have been using the following resources to learn coding in 2017:")
for c in dataset2017.columns:
    if "Resource" in c:
        print("{0}{1} : {2:5.1%}".format(c.replace('Resource',''), ' '*(20-len(c)), 1-dataset2017[c].isnull().sum()/len(dataset2017)))
