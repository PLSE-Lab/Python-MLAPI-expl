import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile

nbin = 1000
lat_mid = 41.1496100
lon_mid = -8.6109900
w = 0.1 # window size

with ZipFile('../input/train.csv.zip') as zf:
    
    data = pd.read_csv(zf.open('train.csv'),
                       usecols=['POLYLINE'],
                       chunksize=10000,
                       converters={'POLYLINE': lambda x: json.loads(x)})
    
    # process data in chunks to avoid using too much memory
    z = np.zeros((nbin, nbin))
    
    for chunk in data:

        latlon = np.array([(lat, lon) 
                           for path in chunk.POLYLINE
                           for lon, lat in path if len(path) > 0])

        z += np.histogram2d(*latlon.T, bins=nbin, 
                            range=[[lat_mid - w, lat_mid + w],
                                   [lon_mid - w, lon_mid + w]])[0]


log_density = np.log(1+z)
plt.imshow(log_density[::-1,:]) # flip vertically and plot
plt.axis('off')
plt.savefig('heatmap.png')