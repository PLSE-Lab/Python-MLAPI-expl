import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../input/CLIWOC15.csv', header=0, parse_dates={'Date':['Year', 'Month', 'Day']})

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# how many records indicate sea ice?  not many
df.groupby('SeaIce').size()
# filter out records with sea ice and a latitude
df = df[(df.SeaIce == 1) & (df.LatDeg.notnull())]
df['DOY'] = pd.to_datetime(df.Date).dt.dayofyear
dfN = df[df.LatHem == 'N']
dfS = df[df.LatHem == 'S']
# plot results
plt.scatter(dfN.DOY, dfN.Lat3, color='r', label='North hemisphere')
plt.scatter(dfS.DOY, -dfS.Lat3, color='b', label='South hemisphere')
plt.xlim(0, 365)
plt.xlabel('day of year')
plt.ylabel('latitude')
plt.legend(loc='lower left')
fig = plt.gcf()
fig.savefig('figure1.png')
# I was expecting a more sinesoidal shaped distribution of points.
# Interesting that sea ice records occur mostly in summer.