# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


hotrev = pd.read_csv('../input/7282_1.csv')

#print(hotrev['name'][hotrev['city']=='Albert Lea'])

####

#Listing the Unique Cities and Hotels overall
ndarcity = (np.unique(hotrev['city']))
ndarhotel = (np.unique(hotrev['name']))

#Finding total number of Unique Cities and Hotels
lencity = len(ndarcity)
lenhotel = len(ndarhotel)

print('Number of distint city = ',lencity)
print('Number of distint hotel = ',lenhotel,'\n')

####

#City wise number of hotels
city_numof_hotels = pd.DataFrame(columns=['City','NumofHotels'])

for i in range(lencity):
    city_numof_hotels.loc[i] = (ndarcity[i],len(np.unique(hotrev['name'][hotrev['city']==ndarcity[i]])))

city_numof_hotels.to_csv('Derived_City_Numof_Hotels.csv')
#print(city_numof_hotels)
print('City with Maximum hotels:')
print((city_numof_hotels['City'][city_numof_hotels['NumofHotels'] == max(city_numof_hotels['NumofHotels'])]),max(city_numof_hotels['NumofHotels']))

####

#Hotel wise Avg Review
hotels_list = pd.DataFrame(columns=['Hotel_Name','Avg_Review','Percentage'])

for i in range(50):
    hotels_list.loc[i] = (ndarhotel[i],(np.mean(hotrev['reviews.rating'][hotrev['name'] == ndarhotel[i]])),((np.mean(hotrev['reviews.rating'][hotrev['name'] == ndarhotel[i]]))/5))

hotels_list.to_csv('Derived_Hotel_List_Review.csv')
#print(hotels_list)
print('Hotel with Maximum Average Review')
print((hotels_list['Hotel_Name'][hotels_list['Avg_Review'] == max(hotels_list['Avg_Review'])]),max(hotels_list['Avg_Review']))
#print (np.mean(hotrev['reviews.rating'][hotrev['name'] == 'Agate Beach Motel']))

l = list(range(len(hotels_list['Hotel_Name'])))
plt.xticks(l,hotels_list['Hotel_Name'],rotation = 'vertical')
plt.bar(l,hotels_list['Avg_Review'],align='center')

plt.show()

# Any results you write to the current directory are saved as output.