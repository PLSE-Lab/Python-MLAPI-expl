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

deliveries = pd.read_csv("../input/deliveries.csv")
col_Names = deliveries.columns.tolist()
print(col_Names)

deliveries['fielder']
deliveries['fielder'].unique()

count = {}
for b in deliveries['fielder'].unique():
    catches = deliveries[(deliveries['fielder'] == b) & (deliveries['dismissal_kind'] == 'caught')]
    count[b] = catches['fielder'].count()
#print(count)    

Most_catches=sorted(count, key=count.get, reverse=True)[:5]
print(Most_catches)


    