import pandas as pd
clicow = pd.read_csv('../input/CLIWOC15.csv')
lon = clicow.Lat3
lat = clicow.Lon3
import matplotlib.pyplot as plt
coords = list(zip(lat,lon))
good_coords = []
for i in coords:
    try: 
        x = int(i[0])
        y = int(i[1])
        good_coords.append(i)
    except: pass
for i in good_coords[:50000]:
    plt.scatter(i[0], i[1],s=4, alpha=.02)
plt.savefig('coords.png')