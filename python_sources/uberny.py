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


df = pd.read_csv('../input/uber-raw-data-sep14.csv')

import matplotlib.pyplot as plt

#Adding new data to the table.
df['Date/Time'] = pd.to_datetime(df['Date/Time'],format ="%m/%d/%Y %H:%M:%S")
df['DayofweekNum'] = df['Date/Time'].dt.dayofweek
df['Dayofweek'] = df['Date/Time'].dt.weekday_name
df['MonthdayNum'] = df['Date/Time'].dt.day
df['Hourofday'] = df['Date/Time'].dt.hour


df_weekdays = df.pivot_table(index=['DayofweekNum','Dayofweek'], values='Base',aggfunc='count')
df_weekdays.plot(kind='bar',figsize=(8,6))
plt.ylabel('Total Journeys')
plt.title('Journeys By WeekDay')
print(df_weekdays)


