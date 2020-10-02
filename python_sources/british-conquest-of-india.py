# Inspired by https://www.kaggle.com/katacs/climate-data-from-ocean-ships/captain-cook-s-travels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.basemap import Basemap

def isInIndia( lon, lat ):
	# Check if a given lat/lon falls in India
	return lon < 97.16712 and lon > 68.03215 and lat > 6.7476 and lat < 37.4

def find_indian_destinations( destinations ):
	#Find all the Indian Destinations from the list of Indian destiantions
	geodata = pd.read_csv('../input/Geodata.csv')
	dest_lat = geodata.DecLatitude
	dest_lon = geodata.DecLongitude
	data_dest = geodata.Place

	reqd_data = np.column_stack((data_dest,dest_lon,dest_lat))

	indian_destinations = []

	for destination in destinations:
		if not pd.isnull(destination):

			row = reqd_data[ reqd_data[:,0] == destination ]
			if len(row) != 0:
				if isInIndia( float(row[0][1]), float(row[0][2]) ) :
					indian_destinations += [destination]

	return indian_destinations

def isArrayInIndia( indian_destinations, destinations):
	# Return np bool array for indexing
	result = []
	for dest in destinations:
		if dest in indian_destinations:
			result += [True]
		else:
			result += [False]
	return np.array(result)

# Read Main Climate Data
shipdata = pd.read_csv('../input/CLIWOC15.csv')
lat = shipdata.Lat3
lon = shipdata.Lon3
coord = np.column_stack((list(lon),list(lat)))
Destination = shipdata.VoyageTo
Nationality = shipdata.Nationality
Name = shipdata.ShipName
year = shipdata.Year
month = shipdata.Month
day = shipdata.Day
utc = shipdata.UTC

# Remove lat/lon nans
Destination = Destination[~np.isnan(coord).any(axis=1)]
Nationality = Nationality[~np.isnan(coord).any(axis=1)]
Name = Name[~np.isnan(coord).any(axis=1)]
coord = coord[~np.isnan(coord).any(axis=1)]
year = year[~np.isnan(coord).any(axis=1)]
month = month[~np.isnan(coord).any(axis=1)]
day = day[~np.isnan(coord).any(axis=1)]
utc = utc[~np.isnan(coord).any(axis=1)]
data = np.column_stack((coord,Destination,Nationality,Name,year,month,day,utc))

# Find Indian Destinations
#indian_destinations = find_indian_destinations( np.unique(Destination) )

# Find British Ships travelling to India
British = data[np.logical_or( data[:,3]=='British', data[:,3]=='British ')]
#idx = isArrayInIndia( indian_destinations, British[:,2] )
idx = British[:,2] == 'Calcutta'
British = British[idx]

# Setup Base Map
m = Basemap(projection='robin',lon_0=180,resolution='c', llcrnrlon=120, urcrnrlon=-30)
m.drawcoastlines()
m.drawcountries()
m.drawmeridians(np.arange(0,360,30))
m.drawparallels(np.arange(-90,90,30))
m.fillcontinents(color='grey')

# animation

def init():
	point.set_data([],[])
	return point,

def animate(i):
	x,y=m(data_to_plot[i][0],data_to_plot[i][1])
	point.set_data(x,y)
	plt.title('%s %d %d %d' % (data_to_plot[i][4],data_to_plot[i][5],data_to_plot[i][6],data_to_plot[i][7]))
	return point,


data_to_plot = []
for ship in np.unique(British[:,4]):
	shipTravels = British[ British[:,4] == ship ]
	shipTravels = shipTravels[ shipTravels[:,8].argsort() ]
	for row in shipTravels:
		data_to_plot += [row]

data_to_plot = np.array(data_to_plot)
x,y = m(0,0)
point = m.plot(x,y,'o',markersize=7,color='blue')[0]
ouput =animation.FuncAnimation(plt.gcf(),animate,init_func = init, frames=len(data_to_plot), interval=100,blit=False,repeat=False)
plt.show()
ouput.save('conquest_ofindia.gif',writer='imagemagick')

