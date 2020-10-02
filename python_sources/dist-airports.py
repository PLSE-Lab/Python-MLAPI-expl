import numpy as np
import pandas as pd
import itertools

airports = pd.read_csv('../input/airports.csv')
IATA = airports.IATA_CODE
latitude = np.array(airports.LATITUDE)
longitude = np.array(airports.LONGITUDE)

pairs = lambda: itertools.product(range(len(IATA)), range(len(IATA)))
lat1 = np.array([latitude[i] for i,j in pairs()])
lat2 = np.array([latitude[j] for i,j in pairs()])
dlong = np.abs([longitude[i]-longitude[j] for i,j in pairs()])
dist = np.arccos(np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(dlong))

with open('dist.csv', 'w') as f:
    for ((i,j),d) in zip(pairs(), dist):
        if i<j:
            f.write(','.join([IATA[i], IATA[j], str(round(d*3959,1))]))
            f.write('\n')