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
import csv
import pandas as pd

file = '../input/monthly_salary_brazil.csv'
f = open(file,'rt')
reader = csv.reader(f)

#once contents are available, I then put them in a list
csv_list = []
for l in reader:
    csv_list.append(l)
f.close()
#now pandas has no problem getting into a df
df = pd.DataFrame(csv_list)
#Solving the problem on line 845 and 847
i=1
while (i<10):
    df.loc[845,i] = df.loc[845,i+1]
    df.loc[847,i] = df.loc[847,i+1]
    i += 1
df.drop(labels=10,axis=1,inplace=True)

df.columns = df.iloc[0]
df = df.reindex(df.index.drop(0),)