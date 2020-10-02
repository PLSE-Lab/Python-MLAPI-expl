import numpy as np
import itertools

IATA = []
latitude = []
longitude = []
    
with open('../input/airports.csv') as f:
    for line in f:
        col = line.split(',')
        if col[3] == 'MO':
            IATA.append(col[0])
            latitude.append(float(col[5].strip()))
            longitude.append(float(col[6].strip()))

pairs = lambda: itertools.product(range(len(IATA)), range(len(IATA)))
lat1 = np.array([latitude[i] for i,j in pairs()])
lat2 = np.array([latitude[j] for i,j in pairs()])
dlong = np.abs([longitude[i]-longitude[j] for i,j in pairs()])
dist = np.arccos(np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(dlong))
for ((i,j),d) in zip(pairs(), dist):
    if i!=j:
        print(IATA[i], IATA[j], d)

print(dist.reshape((len(IATA), len(IATA))))
