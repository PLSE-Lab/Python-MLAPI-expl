# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd
from haversine import haversine, Unit


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Import data
data = pd.read_csv('../input/crimes-in-boston/crime.csv', encoding='latin-1')

'''
########
# Comment out since we modify the objective.
########
# manually encoded lat/lng of Boston Police department 
police_department_list = [['Boston Police Headquarters', 42.333790,-71.091950], ['District A-1 & A-15 Downtown & Charlestown', 42.361820, -71.060310], ['District A-7 East Boston', 42.369760, -71.039990], ['District B-2 Roxbury',42.328239,-71.085632], ['District B-3 Mattapan',42.284710,-71.091750],
['District C-6 South Boston',42.341202,-71.054962], ['District C-11 Dorchester',42.297989,-71.059258], ['District D-4 South End',42.339540,-71.069150], ['District D-14 Brighton',42.349400, -71.150590],
['District E-5 West Roxbury',42.286840,-71.148390], ['District E-13 Jamaica Plain',42.309720,-71.104590], ['District E-18 Hyde Park',42.256480, -71.124270]]
police_department_df = pd.DataFrame(police_department_list, columns = ['police_department', 'police_latitude', 'police_longitude'])


import time

# Minimal code needed for this one. Create a common 'key' to cartesian merge the two:
now1 = time.time()
data['key'] = 0
police_department_df['key'] = 0
crime = ['Larceny / Theft'] * 5 + ['Burglary'] * 3 + ['Aggravated Assault'] *4 + ['Robbery']*3
crime_prob = [1] * 5 + [2] * 3 + [5] *4 + [4]*3
offense= ['Auto Theft','Auto Theft Recovery','Embezzlement','Evading Fare','Larceny From Motor Vehicle'] + ['Burglary - No Property Taken','Commercial Burglary','Residential Burglary'] + ['Aircraft','Arson','Biological Threat','Bomb Hoax'] + ['Simple Assault', 'Towed', 'Vandalism']
crime_code = {'crime_group': crime, 'police_demand': crime_prob, 'OFFENSE_CODE_GROUP': offense}
crime_offense_group = pd.DataFrame.from_dict(crime_code)
data = data.merge(crime_offense_group, how='inner', on ='OFFENSE_CODE_GROUP')

df_cartesian = data.merge(police_department_df, how='outer', on ='key')
df_cartesian = df_cartesian.drop(columns=['key'])

now2 = time.time()
print(now2 - now1)

df_cartesian['distance'] = df_cartesian.apply(lambda x: haversine((x['Lat'],x['Long']), (x['police_latitude'], x['police_longitude']), unit=Unit.MILES), axis = 1)
print(time.time() - now2)
df_cartesian.to_csv('crimes_police_distance.csv', index=False)
'''

# generate data for (region, crime_type, time_window, prob)
crime = ['Larceny / Theft'] * 5 + ['Burglary'] * 3 + ['Aggravated Assault'] *4 + ['Robbery']*3
crime_prob = [1] * 5 + [2] * 3 + [5] *4 + [4]*3
offense= ['Auto Theft','Auto Theft Recovery','Embezzlement','Evading Fare','Larceny From Motor Vehicle'] + ['Burglary - No Property Taken','Commercial Burglary','Residential Burglary'] + ['Aircraft','Arson','Biological Threat','Bomb Hoax'] + ['Simple Assault', 'Towed', 'Vandalism']
crime_code = {'crime_group': crime, 'police_demand': crime_prob, 'OFFENSE_CODE_GROUP': offense}
crime_offense_group = pd.DataFrame.from_dict(crime_code)
data = data.merge(crime_offense_group, how='inner', on ='OFFENSE_CODE_GROUP')

# group time [0, 8), [8,16), [16, 24)
# def bucketize(x):
#     if x['HOUR'] >= 0 and x['HOUR'] <8:
#         return(0)
#     elif x['HOUR'] >= 8 and x['HOUR'] <16:
#         return(1)
#     else:
#         return(2)
    
# data['time_windows'] = data.apply(lambda x: bucketize(x), axis= 1)

# generate crime distribution
probabilities = data[['DISTRICT', 'crime_group', 'HOUR', 'INCIDENT_NUMBER']].groupby(['DISTRICT', 'crime_group', 'HOUR']).count()['INCIDENT_NUMBER'] 

crime_distribution = pd.DataFrame({'district': probabilities.index.get_level_values(0).tolist(),
                         'crime_group': probabilities.index.get_level_values(1).tolist(),
                         'time_windows': probabilities.index.get_level_values(2).tolist(),
                         'prob': probabilities.values.tolist()})
crime_distribution.to_csv('crimes_distribution.csv', index=False)