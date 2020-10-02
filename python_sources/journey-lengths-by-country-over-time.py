import pandas as pd
import matplotlib.pyplot as plt

shipdata = pd.read_csv('../input/CLIWOC15.csv')
ydc = shipdata[['Nationality','Year','Distance','DistTravelledUnits']]
pt = ydc.pivot_table(values = 'Distance', columns = 'Nationality', index = 'Year',aggfunc = sum, fill_value = 0)
Yeardata = pd.DataFrame(index = list(range(1750,1851)))
set_years = pd.merge(Yeardata, pt, how = 'inner', left_index = True,right_index = True)
set_years['Year'] = set_years.index

plt.plot(set_years['Year'], set_years['British'])
plt.plot(set_years['Year'], set_years['Dutch'])
plt.plot(set_years['Year'], set_years['Spanish'])
plt.legend(['British', 'Dutch', 'Spanish'], loc='upper right')
plt.xlim([1750,1851])
plt.ylabel('Distance (Nautical Miles)')
plt.xlabel('Year')
plt.savefig("output.png")
