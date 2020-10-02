# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

dat = pd.read_csv('../input/data.csv')
dat.info()
print(dat.head())

# number_people           62184 non-null int64
# date                    62184 non-null object
# timestamp               62184 non-null int64
# day_of_week             62184 non-null int64
# is_weekend              62184 non-null int64
# is_holiday              62184 non-null int64
# temperature             62184 non-null float64
# is_start_of_semester    62184 non-null int64
# is_during_semester      62184 non-null int64
# month                   62184 non-null int64
# hour                    62184 non-null int64

# # Day of the week
# day = dat.pivot(columns='day_of_week', values='number_people')

# plt.figure()
# day.plot.box()
# plt.xlabel('Day of the Week')
# plt.ylabel('Number of People')
# plt.savefig('day_hist.png')
# plt.close()

# # Time of day
# hr = dat.pivot(columns='hour', values='number_people')

# plt.figure()
# hr.plot.box()
# plt.xlabel('Hour of the day')
# plt.ylabel('Number of People')
# plt.savefig('all_hours.png')
# plt.close()

# Weekend Hours
hr = dat.loc[dat['is_weekend'] == 1].loc[dat['is_during_semester']==1].pivot(columns='hour', values='number_people')

plt.figure()
hr.plot.box()
plt.title('Semester Weekend Hours')
plt.xlabel('Hour of the day')
plt.ylabel('Number of People')
plt.savefig('weekend_hours.png')
plt.close()

# Weekday Hours
hr = dat.loc[dat['is_weekend'] == 0].loc[dat['is_during_semester']==1].pivot(columns='hour', values='number_people')

plt.figure()
hr.plot.box()
plt.title('Semester Weekday Hours')
plt.xlabel('Hour of the day')
plt.ylabel('Number of People')
plt.savefig('weekday_hours.png')
plt.close()

# Weekend Hours
hr = dat.loc[dat['is_weekend'] == 1].loc[dat['is_during_semester'] == 0].pivot(columns='hour', values='number_people')

plt.figure()
hr.plot.box()
plt.title('Summer Weekend Hours')
plt.xlabel('Hour of the day')
plt.ylabel('Number of People')
plt.savefig('semester_hours.png')
plt.close()

# Weekday Hours
hr = dat.loc[dat['is_weekend'] == 0].loc[dat['is_during_semester'] == 0].pivot(columns='hour', values='number_people')

plt.figure()
hr.plot.box()
plt.title('Summer Weekday Hours')
plt.xlabel('Hour of the day')
plt.ylabel('Number of People')
plt.savefig('summer_hours.png')
plt.close()

# Monthly
hr = dat.pivot(columns='month', values='number_people')

plt.figure()
hr.plot.box()
plt.xlabel('Month')
plt.ylabel('Number of People')
plt.savefig('month_box.png')
plt.close()













