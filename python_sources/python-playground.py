import sqlite3
import matplotlib.pyplot as plt
import pandas as pd

con = sqlite3.connect('../input/database.sqlite')
cursor = con.cursor()

results = cursor.execute('SELECT Year, COUNT(*) FROM NationalNames GROUP BY Year ORDER BY Year')
numrows = int(cursor.rowcount)

data = []

for row in results.fetchall():
    data.append((row[0], row[1]))

dic = dict(data)

print(dic)

df = pd.Series(dic)

plt.plot(df.values)
plt.xticks(range(len(df)), df.index.values, rotation='vertical')
plt.savefig("CountsByYear.png")

from mpl_toolkits.basemap import Basemap

m = Basemap(llcrnrlat=22, urcrnrlat=47, llcrnrlon=-120, urcrnrlon=-60, 
            lat_0=38.5, lon_0=-97.5, lat_1=38.5, lat_2=38.5, 
            resolution='c', projection='lcc')

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.fillcontinents()
