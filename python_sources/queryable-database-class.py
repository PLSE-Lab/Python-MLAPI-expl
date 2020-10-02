import math
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
import datetime

# read the raw data from file
data = pd.read_csv('../input/CLIWOC15.csv');

# define the database class
class Database:
    def __init__ (self, initial_list=False):
        """
        Queryable database class of entries from CLIWOC15.csv.
        @param initial_list: Optional initial list of entries
        """
        self.database = []
        if not initial_list == False:
            self.append(initial_list)
        
    def add_database(self, new_database):
        """
        Adds a database to the class.  For use in creating unions of databases.
        Does not check for repeated entries
        @param new_database: database to add
        """        
        self.database.append(new_database)
        
    def add_entry(self, entry):
        """
        Adds an instance of the entry class into the database.
        @param entry: entry of type Entry
        """
        self.database.append(entry)
        
    def all_entries(self, queries=True, **kwargs):
        """
        Returns all entries which obey the queries or have the requested attribute value.
        @param queries: an optional list of (attribute, lambda expression) tuples 
                        which is true for the returned entries
        """
        return list(self.__iter_entry(queries, **kwargs))
    
    def __iter_entry(self, queries, **kwargs):
        return (entry for entry in self.database if entry.match(queries, **kwargs))
        
# define the entry class from which the database is composed
class Entry:
    def __init__(self, year, month, day, lon, lat, temperature, windforce, winddirection, 
                 rain, fog, snow, thunder, hail, seaice,
                 nationality, shipname):
        
        # set the date
        try:
            self.has_date = True
            self.date = datetime.datetime(year, month, day)
            self.dayofyear = int(self.date.timetuple().tm_yday)
        except:
            self.has_date = False
        
        # set the longitude and latitude
        self.lon = lon
        self.lat = lat
        self.coord = [lon, lat]
        self.has_coord = self.check_valid_data(self.lon) and self.check_valid_data(self.lat)
        
        # set temperature
        self.temperature = temperature
        self.has_temperature = self.check_valid_data(self.temperature)
        
        # set wind force
        self.windforce = get_windforce(windforce)
        self.has_windforce = self.check_valid_data(self.windforce)
        
        # set wind direction
        self.winddirection = get_winddirection(winddirection)
        self.has_winddirection = self.check_valid_data(self.winddirection)
        
        # set weather
        self.rain = rain
        self.fog = fog
        self.snow = snow
        self.thunder = thunder
        self.hail = hail
        self.seaice = seaice
        
        # set ship name and nationality
        self.shipname = shipname
        self.nationality = nationality
    
    def check_valid_data(self, value):
        if value == None:
            return False
        elif math.isnan(value):
            return False
        else:
            return True
    
    def match(self, queries, **kwargs):
        if not queries == True:
            for query in queries:
                if not query[1](getattr(self, query[0], True)):
                    return
        return all(getattr(self, key) == value for (key, value) in kwargs.items())

# load the wind speed translations
lookup_es = pd.read_csv('../input/Lookup_ES_WindForce.csv')
lookup_nl = pd.read_csv('../input/Lookup_NL_WindForce.csv')
lookup_fr = pd.read_csv('../input/Lookup_FR_WindForce.csv')
lookup_uk = pd.read_csv('../input/Lookup_UK_WindForce.csv')

# turns these into dictionaries
windforcedict = dict()
for index in range(1, len(lookup_es.index.values)):
    windforcedict[lookup_es.ix[index].WindForce] = lookup_es.ix[index].mps
for index in range(1, len(lookup_nl.index.values)):
    windforcedict[lookup_nl.ix[index].WindForce] = lookup_nl.ix[index].mps
for index in range(1, len(lookup_fr.index.values)):
    windforcedict[lookup_fr.ix[index].WindForce] = lookup_fr.ix[index].mps
for index in range(1, len(lookup_uk.index.values)):
    windforcedict[lookup_uk.ix[index].WindForce] = lookup_uk.ix[index].mps

def get_windforce(windforceraw):
    """
    Gets the wind force in metres per second
    from CLIWOC15.csv data based on the lookups given for all languages.
    Returns NaN if the argument is not found in the dictionary.
    
    @param windforceraw: string with wind force from CLIWOC15.csv
    """
    try:
        return windforcedict[windforceraw]
    except:
        return float('nan')
    
# load the wind speed translations
lookup_es = pd.read_csv('../input/Lookup_ES_WindDirection.csv')
lookup_nl = pd.read_csv('../input/Lookup_NL_WindDirection.csv')
lookup_fr = pd.read_csv('../input/Lookup_FR_WindDirection.csv')
lookup_uk = pd.read_csv('../input/Lookup_UK_WindDirection.csv')

# turns these into dictionaries
winddirectiondict = dict()
for index in range(1, len(lookup_es.index.values)):
    winddirectiondict[lookup_es.ix[index].WindDirection] = lookup_es.ix[index].ProbWindDD
for index in range(1, len(lookup_nl.index.values)):
    winddirectiondict[lookup_nl.ix[index].WindDirection] = lookup_nl.ix[index].ProbWindDD
for index in range(1, len(lookup_fr.index.values)):
    winddirectiondict[lookup_fr.ix[index].WindDirection] = lookup_fr.ix[index].ProbWindDD
for index in range(1, len(lookup_uk.index.values)):
    winddirectiondict[lookup_uk.ix[index].WindDirection] = lookup_uk.ix[index].ProbWindDD


def get_winddirection(winddirectionraw):
    """
    Gets the wind direction in degrees (0 is north) 
    from CLIWOC15.csv data based on the lookups given for all languages.
    Returns NaN if the argument is not found in the dictionary.
    
    @param winddirectionraw: string with wind direction from CLIWOC15.csv
    """
    try:
        return winddirectiondict[winddirectionraw]
    except:
        return float('nan')

# Load all of the data into the database
nodata = len(data.index.values)
database = Database()
for index in range(1, nodata):
    database.add_entry(Entry(data.Year[index], data.Month[index], data.Day[index], data.Lon3[index], data.Lat3[index], 
                          data.ProbTair[index], data.WindForce[index], data.WindDirection[index], 
                          data.Rain[index], data.Fog[index], data.Snow[index], data.Thunder[index], data.Hail[index], data.SeaIce[index],
                          data.Nationality[index], data.ShipName[index]))

# An example query
# Here, we return all entries for Spanish ships in the northern hemisphere
# where the wind force is less than 10 metres per second and it is raining
my_queries = []
my_queries.append(('lat', lambda val: val > 0))
my_queries.append(('windforce', lambda val: val < 10.))
my_query_result = database.all_entries(has_windforce=True, rain=True, nationality='Spanish', queries=my_queries)

# Get the longitude, latitude and temperature for all of the entries found by this query
lon = [value.lon for value in my_query_result]
lat = [value.lat for value in my_query_result]
wind = [value.windforce for value in my_query_result]

# Initialize a basemap
map = Basemap(projection='kav7', lon_0=0);
map.drawmapboundary(color='aqua',fill_color='aqua');
map.fillcontinents(color='coral',lake_color='aqua');

# get the coordinates to x,y and plot
x, y = map(lon, lat)
fig = map.scatter(x, y, 5, c = wind, lw=0);
fig.set_clim(vmin=0, vmax=10)
plt.title('Spanish ships experiencing rain and wind speeds\ngreater than 10mps in the Northern Hemisphere')
cbar = plt.colorbar();
cbar.set_label('Windforce / mps')
plt.savefig('result.png')