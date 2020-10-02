# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import pandas as pd
import datetime
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

##################################################
# 1. Preparaing covidDataset for preprocessing
##################################################
covidData = pd.read_csv("../input/20200222ncovidcsv/combined_csv.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 28)

##################################################
# 1.1 Rename the columns to easy programming
##################################################
covidData = covidData.rename(columns=({'Country/Region':'Country'}))
covidData = covidData.rename(columns=({'Province/State':'State'}))
covidData = covidData.rename(columns=({'Last Update': 'Date'}))

##################################################
# 2. Drop records that are NOT  from China/Mainland China/Hong Kong/Macau
#       i.e. report on records that are from China only
#     Drop records that have ZERO confirmations
##################################################
countryFilter = ['China', 'Mainland China', 'Macau', 'Hong Kong']
zeroConfirmations = ['0']
covidData = covidData[covidData.Country.isin(countryFilter) &
                ~covidData.Confirmed.isin(zeroConfirmations)]
print(covidData.shape)

##################################################
# 3. Replace "Mainland China" with "China" for consisteny
#       as per Wikipedia, Mainland China does not include Hong Kong and Macau
#       The rest of the provinces belong to China and synonymous with Mainland
#       China
##################################################
app_type = {'Mainland China' : 'China'}
covidData.Country.replace(app_type, inplace=True)
covidData['Confirmed'].fillna(0, inplace=True)
covidData['Deaths'].fillna(0, inplace=True)
covidData['Recovered'].fillna(0, inplace=True)

##################################################
# 4. Review variables for drops and edits
##################################################
print(pd.value_counts(covidData.Country).to_frame())

##################################################
# 5. Now, lets cleanup the date formats in the Date Column
##################################################
def convertToDateTime(dateColumn):

    list = []
    for d in dateColumn:
        if d.find('T') > 0:
            try:
                date_format_obj = datetime.datetime.strptime(d, "%Y-%m-%dT%H:%M")
            except ValueError:
                date_format_obj = datetime.datetime.strptime(d, "%Y-%m-%dT%H:%M:%S")
#            print("Converted from : ", d, " To : ", date_format_obj)
        elif d.find("/") > 0:
            try:
                date_format_obj = datetime.datetime.strptime(d, "%m/%d/%Y %H:%M")
            except ValueError:
                date_format_obj = datetime.datetime.strptime(d, "%m/%d/%Y %H:%M:%S")
#            print("Converted from : ", d, " To : ", date_format_obj)
        elif d.find("-") > 0:
            try:
                date_format_obj = datetime.datetime.strptime(d, "%m-%d-%Y %H:%M")
            except ValueError:
                date_format_obj = datetime.datetime.strptime(d, "%m-%d-%Y %H:%M:%S")
#            print("Converted from : ", d, " To : ", date_format_obj)
        else:
            print("Error -- no match :", d)
        list.append(date_format_obj)
    formattedDates = pd.Series(list)
    return formattedDates

formattedDates = convertToDateTime(covidData['Date'])
covidData['Date'] = formattedDates.values
covidData = covidData.drop_duplicates()  # Sometimes the raw data files are duplicates

allStates = pd.value_counts(covidData.State).keys()
# Add 3 empty columns here
covidData = covidData.reindex(columns = covidData.columns.tolist() + ['MaxConfirmed','MaxDeaths', 'MaxRecovered'])

for state in allStates:
    stateData = covidData[covidData['State'] == state]
    maxConfirmed = stateData["Confirmed"].max()
    maxDeaths = stateData['Deaths'].max()
    maxRecovered = stateData['Recovered'].max()
    #print(state, '--', maxConfirmed, '--', maxDeaths, '--', maxRecovered)
    covidData.loc[covidData['State'] == state, 'MaxConfirmed'] = maxConfirmed
    covidData.loc[covidData['State'] == state, 'MaxDeaths']= maxDeaths
    covidData.loc[covidData['State'] == state, 'MaxRecovered'] = maxRecovered

#covidData.to_csv("../output/covidData.csv")

##################################################
# Lets plot the graphs for about 4 cities over time
##################################################
import geopandas

data = geopandas.read_file('../input/chinashapefile/CHN/CHN_adm3.shp')
data = data.reindex(columns = data.columns.tolist() + ['MaxConfirmed','MaxDeaths', 'MaxRecovered'])

allStates = pd.value_counts(data.NAME_1).keys()
for state in allStates:
    stateData = covidData[covidData['State'] == state]
    data.loc[data['NAME_1'] == state, 'MaxConfirmed'] = stateData["MaxConfirmed"].max()
    data.loc[data['NAME_1'] == state, 'MaxDeaths']= stateData['Deaths'].max()
    data.loc[data['NAME_1'] == state, 'MaxRecovered'] = stateData['Recovered'].max()

mergedData = data[['NAME_1', 'MaxConfirmed', 'MaxDeaths', 'MaxRecovered']]
#print(mergedData)

# add the columns which can serve as content in annotations as well anything
# that is needed for the next level
province_boundary = data[['ID_1', 'NAME_1', 'MaxConfirmed', 'MaxDeaths', 'MaxRecovered', 'geometry']]
# Disssolve all boundaries beyond ID_1 so that the plot is at ID_1 level
provincial_regions = province_boundary.dissolve(by='ID_1')
confirmedAx = provincial_regions.plot(column='MaxConfirmed', cmap='Blues', scheme='quantiles',legend=True, edgecolors='lightgray')
confirmedAx.set_title('Confirmed Cases')
deathsAx = provincial_regions.plot(column='MaxDeaths', cmap='Reds', scheme='quantiles',legend=True, edgecolors='lightgray')
deathsAx.set_title('Deaths')
recoveredAx = provincial_regions.plot(column='MaxRecovered', cmap='GnBu', scheme='quantiles',legend=True, edgecolors='lightgray')
recoveredAx.set_title('Recovered')

plt.show()
