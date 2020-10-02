# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pip

for package in sorted(pip.get_installed_distributions(), key=lambda package: package.project_name):
    print("{} ({})".format(package.project_name, package.version))
# from iso3166 import countries

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df15 = pd.read_csv("../input/2015.csv")
#print(input_df15.head(5))
df16 = pd.read_csv("../input/2016.csv")
# input_df = input_df.append(pd.read_csv("../input/2016.csv"))
#print(input_df16.head(5))

print(df16.keys())

test_df = df16[['Country', 'Happiness Rank']]
print(test_df.head(5))
df1 = pd.merge(df15[['Country', 'Happiness Rank']], df16[['Country', 'Happiness Rank']], on='Country', suffixes=(' 2015', ' 2016'))

df1['Rank Movement'] = df1['Happiness Rank 2015'] - df1['Happiness Rank 2016']
print(df1.head(5))

# Any results you write to the current directory are saved as output.