import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Load the raw data
crime = pd.read_csv("../input/ucr_by_state.csv")
raw_prisoners = pd.read_csv("../input/prison_custody_by_state.csv")

#prisoners data is in 'wide' format, with column per each year
raw_prisoners.head()

#cast boolean indicators
raw_prisoners.includes_jails = raw_prisoners.includes_jails.astype('bool') 


#melt new long-format dataset
prisoners = pd.melt(raw_prisoners, id_vars=['jurisdiction', 'includes_jails'], 
    value_vars=['2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016'],
    var_name='year',
    value_name='prisoner_count'
    )

#inspect final results for prisoners dataset
prisoners.year = pd.to_numeric(prisoners.year, downcast='integer')
prisoners.head()


#raw crime dataset is already in long format
crime.head()

#cast boolean indicators
crime.crime_reporting_change = crime.crime_reporting_change.astype('bool') 
crime.crimes_estimated = crime.crimes_estimated.astype('bool')

#cast year in int format
crime.year = pd.to_numeric(crime.year, downcast='integer')

#inspect missing values in the crime dataframe
print(crime.isnull().sum())

#drop extraneous columns
cols_to_drop = ['Unnamed: 15','Unnamed: 16','Unnamed: 17','Unnamed: 18','Unnamed: 19','Unnamed: 20']
crime = crime.drop(cols_to_drop, axis=1)
crime.head()

#only include rows not missing state values
crime = crime[crime.jurisdiction.notnull()]

#Looks OK now
print(crime.isnull().sum())

#standardize the jurisdiction fields for joining
crime['jurisdiction'] = crime['jurisdiction'].map(lambda x: x.strip().upper())
prisoners['jurisdiction'] = prisoners['jurisdiction'].map(lambda x: x.strip().upper())


#merge the crime and prisoners dataset
crime_and_incarceration = pd.merge(prisoners,crime,how="left", on=['jurisdiction','year'])

#clean numeric columns
def string_to_int(val):
    val = str(val).replace('nan','')
    if val:
        val = val.replace(',','').strip()
    try:
        val = int(val)
    except ValueError:
        val = None
    return val

cols_to_clean = ['prisoner_count','state_population','violent_crime_total', 'murder_manslaughter', 'rape_legacy','rape_revised', 'robbery', 'agg_assault', 'property_crime_total','burglary', 'larceny', 'vehicle_theft']
for col in cols_to_clean:
    crime_and_incarceration[col] = crime_and_incarceration[col].apply(lambda x: string_to_int(x))

#save merged dataset
crime_and_incarceration.to_csv("crime_and_incarceration_by_state.csv", header=True, index=False)












