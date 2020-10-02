# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#I used the patents field in the CWUR dataset as a metric for innovation i.e. the greater the number of patents filed
#the greater the culture of innovation 

df = pd.read_csv('../input/cwurData.csv')
df2015 = df[df.year == 2015]
df2015_top50 = df2015.sort_values(by='world_rank', ascending=True)[:50]
df2015_top50_sum = df2015_top50['country'].value_counts()
print(df2015_top50_sum)
df2015_top50_patents = df2015.sort_values(by='patents', ascending=True)[:50]
df2015_top50_patents_sum = df2015_top50_patents['country'].value_counts()
print(df2015_top50_patents_sum)