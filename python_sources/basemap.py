
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
import pylab

def plot_3d(df):
    #initialise a figure object
    fig = plt.figure(figsize=(14,10))
    
    #add a 3D object
    ax = fig.add_subplot(111, projection='3d')
    
    #scatter plot
    ax.scatter(df['event-id'], df['location-long'], df['location-lat'], c="#1292db", lw=0, alpha=1, zorder=5, s = 3)
    
        
    ax.set_xlabel('Time of Year')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Latitude')

    plt.show()
    
d = pd.read_csv('../input/migration_original.csv')
#91732 is the id of the bird whose trajectory we are interested in.
#feel free to change the id of the bird
k = d.loc[d['tag-local-identifier'] == 91732]

#normalise the time axis
#use the event-id to determine the time stamp
#Note: event-id is in increasing order between a range of numbers,
#so we can easily normalise for a given bird
#In the next post, I will plot using time-stamps
#and that will make things more clear

k['event-id'] = (k['event-id'] - k['event-id'].mean())/k['event-id'].std(ddof=0)

plot_3d(k)