# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
all_salaries = pd.read_csv('../input/salaries-by-college-type.csv')
all_salaries.columns = ['School','Type','Starting','Mid', 'Mid_10', 'Mid_25', 'Mid_75', 'Mid_90']
#Remove all $'s
for x in all_salaries.columns:
    if x != 'School' and x!='Type':
        salary = all_salaries[x].str.replace("$", "")
        salary = salary.str.replace(",", "")
        all_salaries[x] = pd.to_numeric(salary)

#Change the sort_by and get graphs for Starting, Mid, Mid_10, etc.
sort_by = 'Mid_90'
top_degrees = all_salaries.nlargest(10, sort_by).reset_index()
#Bar chart view of the data
sb.barplot(sort_by,'School', data=top_degrees)