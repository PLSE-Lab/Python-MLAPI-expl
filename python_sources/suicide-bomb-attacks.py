# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import datetime as dt
import matplotlib as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

if os.path.exists("../input/PakistanSuicideAttacks Ver 4.csv"):
    print("File exists")

def parser(x):
    t = re.findall(r"[\w]+",x)
    t[1] = t[1][:3]
    t.pop(0)
    t1 = " ".join(t)
    return dt.datetime.strptime(t1, "%b %d %Y")
    
df = pd.read_csv("../input/PakistanSuicideAttacks Ver 4.csv", encoding="latin1", parse_dates=['Date'], date_parser = parser, dayfirst = True, index_col=[0], converters={'Longitute':np.float64})
df.Longitude = pd.Series(map(float, df.Longitude))

print("First look at the features and their types")
print(df.dtypes)

print("Looking at the data itself")
print(df.head(n=10))

print("Stats of Data")
print(df.describe())

df.boxplot(figsize=(16,9))
df.plot.hist()
