
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pylab as pl
import numpy as np
import seaborn as sns


data = pd.read_csv('../input/CLIWOC15.csv')



data.Nationality.replace('British ', 'British', inplace=True)
data.LogbookLanguage.replace('British', 'English', inplace=True)


#get rain information
rainDat = data[['Year', 'Rain', 'Fog','Gusts', 'Snow', 'Thunder', 'Hail', 'SeaIce']]

rainPV = rainDat.pivot_table(index = 'Year', aggfunc = np.sum)

#fog
fogDat = data[['Year', 'Fog']]

fogPV = fogDat.pivot_table(index = 'Year', aggfunc = np.sum)





fig = plt.figure()
ax = fig.add_subplot(111)
rainPV.plot(kind = 'line', ax = ax)
plt.title('Count of Weather Events')
plt.locator_params(nbins = 10)
fig.savefig('fig_1.png')


    
