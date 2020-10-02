#This script will clean up the Weather Conditions in World War Two data set.
#I'm not super happy with it yet.  It does work, but I feel the code is a
#tad sloppy, and most of the bad/missing data values are replaced with basic
#values.  It would be better to figure out more creative replacements.
#If anyone wants to point out any lackluster pieces I'd be more that happy.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#This function replaces the Snowfall values.  There are null values in the original
#and the value '#VALUE!' in some.
def impute_snowfall(col):
  snow=str(col[0]).replace(" ", "")
  if pd.isnull(snow):
      return float(0)
  elif (snow == '#VALUE!') or (snow == 'nan'):
      return float(0)
  else:
      return float(snow)

#This replaces the null values in the Poor Weather column.  Since they are only 1 or 0
#values I had to split the function out from the generic one below.
def impute_poorweather(col):
  snow=str(col[0]).replace(" ", "")
  if pd.isnull(snow):
      return float(0)
  elif (float(snow) != 0) and (float(snow) != 1):
      return float(0)
  else:
      return float(snow)

#This function replaces any generic columns that only need nulls replaced.      
def impute_misc(col):
  snow=str(col[0]).replace(" ", "")
  if pd.isnull(snow):
      return float(snow).mean()
  elif snow == 'T' or (snow == 'nan'):
      return float(0)
  else:
      return float(snow)
      
              
weather=pd.read_csv("../input/Summary of Weather.csv", dtype={"WindGustSpd": np.str, "Snowfall": np.str, "PoorWeather": np.str, "TSHDSBRSGF": np.str, "SNF": np.str,})

#Applying any of the above functions to required columns.
weather['Precip'] = weather[['Precip']].apply(impute_misc,axis=1)
weather['WindGustSpd'] = weather[['WindGustSpd']].apply(impute_misc,axis=1)
weather['Snowfall'] = weather[['Snowfall']].apply(impute_snowfall,axis=1)
weather['PoorWeather'] = weather[['PoorWeather']].apply(impute_poorweather,axis=1)
weather['PRCP'] = weather[['PRCP']].apply(impute_misc,axis=1)
weather['DR'] = weather[['DR']].apply(impute_misc,axis=1)
weather['SPD'] = weather[['SPD']].apply(impute_misc,axis=1)
weather['MAX'] = weather[['MAX']].apply(impute_misc,axis=1)
weather['MIN'] = weather[['MIN']].apply(impute_misc,axis=1)
weather['MEA'] = weather[['MEA']].apply(impute_misc,axis=1)
weather['SNF'] = weather[['SNF']].apply(impute_misc,axis=1)
weather['SND'] = weather[['SND']].apply(impute_misc,axis=1)
weather['PGT'] = weather[['PGT']].apply(impute_misc,axis=1)
weather['TSHDSBRSGF'] = weather[['TSHDSBRSGF']].apply(impute_poorweather,axis=1)

#These columns are ALL filled with only null values, so I've removed them.
weather.drop(['FT'],axis=1,inplace=True)
weather.drop(['FB'],axis=1,inplace=True)
weather.drop(['FTI'],axis=1,inplace=True)
weather.drop(['ITH'],axis=1,inplace=True)
weather.drop(['SD3'],axis=1,inplace=True)
weather.drop(['RHX'],axis=1,inplace=True)
weather.drop(['RHN'],axis=1,inplace=True)
weather.drop(['RVG'],axis=1,inplace=True)
weather.drop(['WTE'],axis=1,inplace=True)

#This will just output a heatmap showing that all the null values are gone.
sns.heatmap(weather.isna(),yticklabels=False,cbar=False,cmap='cubehelix')
plt.show()

