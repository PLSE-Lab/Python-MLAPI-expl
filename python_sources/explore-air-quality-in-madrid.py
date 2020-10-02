# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 25

plt.rc('font', size = SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize = SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize = MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize = SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title
print("")

################################################
################################################
############ Open an read configuration file
iprefix = '../input/'
iplots  = ''

############ End of Open an read configuration file
################################################
################################################

Vars  = ['SO_2', 'CO', 'NO', 'NO_2', 'PM25', 'PM10', 'NOx', 'O_3', 'TOL', 'BEN', 'EBE', 'MXY', 'PXY', 'OXY', 'TCH', 'CH4', 'NMHC']
nVars = len(Vars)

#Data = xr.open_dataset(iprefix + 'madrid.h5', engine = 'h5netcdf')
store    = pd.HDFStore(iprefix + 'madrid.h5')
stations = store['master']

################################################
################################################

for var in Vars:
   print(var+' : ')
   print('\t--> Reading data')
   ############ Read data
   for stat in stations.id: #Names of the groups in HDF5 file.
      try:
         data = store[str(stat)][var]
         data.rename(str(stat), inplace = True)
      except:
         continue
      
      try:
         Data[str(stat)] = data
      except NameError:
         Data = data.to_frame()

   # Some resampling
   Cycle    = Data.groupby(Data.index.dayofyear).mean()
   Yearly   = Data.resample('Y').mean()
   Seasonal = Data.resample('Q-NOV').mean()
   ############# End of Read data
   

   ################################################
   ############ Plot : Seasonal variabiliy, Interannual variabiliy and seasonal cycle
   print('\t--> Summary plot')
   myseason = ['DJF', 'MAM', 'JJA', 'SON', 'YRS', 'CYC']
   mymth = [2,5,8,11,None,None]
   nseas = len(myseason)

   fig = plt.figure()
   fig.set_size_inches(12*2, 12*3, forward = True)
   gs  = gridspec.GridSpec(3, 2)

   for i,s,m in zip(range(nseas), myseason, mymth):
      ax = fig.add_subplot(gs[i])
      
      if s == 'YRS':
         data = Yearly
      elif s == 'CYC':
         data = Cycle
      else:
         data = Seasonal[Seasonal.index.month == m]

      ### Plot the data for all stations
      #                     and average across all stations (black thick)
      if s == 'CYC':
         plt.plot(data.index, data)
         plt.plot(data.index, data.mean(axis = 1), linewidth = 5., color = 'black')
      else:
         plt.plot(data.index.year, data)
         plt.plot(data.index.year, data.mean(axis = 1), linewidth = 5., color = 'black')
            
      ### Make the plot prettier
      if s == 'CYC':
         ax.set_xlim(Cycle.index.min(), Cycle.index.max())
      else:
         ax.set_xlim(Data.index.year.min(), Data.index.year.max())
         
      #ax.set_xlabel('Years')
      #ax.set_ylabel('Units')
      ax.set_title(s)
      

   plt.suptitle(var)
   fig = plt.gcf()
   #plt.show()
   fig.savefig(iplots+var+'_summary_allstats.png')
   plt.close(fig)
   ############ End of Plot
   ################################################