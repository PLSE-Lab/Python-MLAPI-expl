#!/usr/bin/env python
# coding: utf-8

# I wish to cleanup the datetime columns of this data, turning it into proper timeseries format. In doing this I also wanted to add new columns, labelling the day of the week that the incidents occurred on and whether or not the incidents occurred on holidays. The reason for this is that I wanted to test the hypotheses that the distribution of laser related incidents will not be equal across the days of the week, and that the frequency of incidents will differ on holidays. Before conducting these statistical analyses I needed to get the data into a more usable format.
# 
# This script doesn't run on kaggle due to the use of the 'holidays' library which kaggle doesn't support. You can download and run locally. Or see https://github.com/CNuge/kaggle_code for a working notebook
# 
# I merged all the .csv files into one, removing the 'Total' row for the bottom of each. This was done using unix commands:
# echo 'DATE,TIME (UTC),ACID,No. A/C,TYPE A/C,ALT,MAJOR CITY,COLOR,Injury Reported,CITY,STATE' > all_laser_dat.csv
# grep -v 'Total' *.csv >> all_laser_dat.csv
# 
# you can then run the following 

# In[ ]:


from datetime import date, datetime
import calendar
import pandas as pd
from pandas import Series, DataFrame
import holidays

""" Looking at the FAA laser incident reports data, I have the
	following two alternative hypotheses I wish to explore.
	1. There will be a higher number of laser incidents on the weekend 
		(friday night - early sunday morning)
	2. There will be a higher number of laser incidents on holidays. """

# First read in the data

laser_dat = pd.read_csv('all_laser_dat.csv')
laser_dat.head()

#drop the 4 rows that do not have a time associated with them
laser_dat = laser_dat[laser_dat['TIME (UTC)'] != 'UNKN']

#turn the dates to datetimes.
laser_dat['TIME (UTC)']

laser_dat['hour'] = laser_dat.apply(lambda x: x['TIME (UTC)'][:-2], axis=1)
laser_dat['min'] = laser_dat.apply(lambda x: x['TIME (UTC)'][-2:], axis=1)

laser_dat['min'].fillna(0, inplace=True)
laser_dat['hour'].fillna(0, inplace=True)

#account for lack of zeros
min_changed = []
for i in laser_dat['min']:
	if len(i) == 0:
		min_changed.append('00')
	elif len(i) == 1:
		min_changed.append('0'+i)
	else:
		min_changed.append(i)

hr_changed = []
for i in laser_dat['hour']:
	if len(i) == 0:
		hr_changed.append('00')
	elif len(i) == 1:
		hr_changed.append('0'+i)
	else:
		hr_changed.append(i)


laser_dat['min_adj'] = min_changed
laser_dat['hr_adj'] = hr_changed

laser_dat['time'] = laser_dat.apply(lambda x: '%s:%s:%s' % (x['DATE'] , x['hr_adj'],  x['min_adj'] ), axis=1)

laser_dat['date_time'] = laser_dat.apply(lambda x: datetime.strptime(x['time'], '%d-%b-%y:%H:%M'), axis=1)

#drop the making of datetime columns, except for the 'hour' column
laser_dat = laser_dat.drop(['time','hour','min', 'min_adj','TIME (UTC)','DATE'], axis=1)

# add a column with the day of the week

laser_dat['day_of_week'] = laser_dat.apply(lambda x:  calendar.day_name[x['date_time'].weekday()] , axis = 1)


# add a column with holiday/no holidays

us_holidays = holidays.UnitedStates()  # or holidays.US()

holiday_tf = []
for date in laser_dat['date_time']:
	if date in us_holidays:
		holiday_tf.append(True)
	elif date not in us_holidays:
		holiday_tf.append(False)


laser_dat['holidays'] = holiday_tf

laser_dat['holidays'].value_counts()


laser_dat.to_csv('adjusted_laser_data.csv')

