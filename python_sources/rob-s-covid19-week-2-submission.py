#!/usr/bin/env python
# coding: utf-8

# # Rob's Baseline Model

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import seaborn as sns
import os, gc, pickle, copy, datetime, warnings
import pycountry

pd.set_option('max_columns', 500)
pd.set_option('max_rows', 500)
pd.options.display.float_format = '{:.2f}'.format


# In[ ]:


get_ipython().system('ls -l ../input/covid19-global-forecasting-week-2/')


# # Data Preprocessing
# https://www.kaggle.com/osciiart/covid19-lightgbm

# In[ ]:


# Read in data
train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")
test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")

tt = pd.concat([train, test], sort=False)
tt = train.merge(test, on=['Province_State','Country_Region','Date'], how='outer')

# concat Country/Region and Province/State
def name_place(x):
    try:
        x_new = x['Country_Region'] + "_" + x['Province_State']
    except:
        x_new = x['Country_Region']
    return x_new
tt['Place'] = tt.apply(lambda x: name_place(x), axis=1)
# tt = tt.drop(['Province_State','Country_Region'], axis=1)
tt['Date'] = pd.to_datetime(tt['Date'])
tt['doy'] = tt['Date'].dt.dayofyear
tt['dow'] = tt['Date'].dt.dayofweek
tt['hasProvidence'] = ~tt['Province_State'].isna()


country_meta = pd.read_csv('../input/covid19-forecasting-metadata/region_metadata.csv')
tt = tt.merge(country_meta, how='left')

country_date_meta = pd.read_csv('../input/covid19-forecasting-metadata/region_date_metadata.csv')
#tt = tt.merge(country_meta, how='left')

tt['HasFatality'] = tt.groupby('Place')['Fatalities'].transform(lambda x: x.max() > 0)
tt['HasCases'] = tt.groupby('Place')['ConfirmedCases'].transform(lambda x: x.max() > 0)

first_case_date = tt.query('ConfirmedCases >= 1').groupby('Place')['Date'].min().to_dict()
ten_case_date = tt.query('ConfirmedCases >= 10').groupby('Place')['Date'].min().to_dict()
hundred_case_date = tt.query('ConfirmedCases >= 100').groupby('Place')['Date'].min().to_dict()
first_fatal_date = tt.query('Fatalities >= 1').groupby('Place')['Date'].min().to_dict()
ten_fatal_date = tt.query('Fatalities >= 10').groupby('Place')['Date'].min().to_dict()
hundred_fatal_date = tt.query('Fatalities >= 100').groupby('Place')['Date'].min().to_dict()

tt['First_Case_Date'] = tt['Place'].map(first_case_date)
tt['Ten_Case_Date'] = tt['Place'].map(ten_case_date)
tt['Hundred_Case_Date'] = tt['Place'].map(hundred_case_date)
tt['First_Fatal_Date'] = tt['Place'].map(first_fatal_date)
tt['Ten_Fatal_Date'] = tt['Place'].map(ten_fatal_date)
tt['Hundred_Fatal_Date'] = tt['Place'].map(hundred_fatal_date)

tt['Days_Since_First_Case'] = (tt['Date'] - tt['First_Case_Date']).dt.days
tt['Days_Since_Ten_Cases'] = (tt['Date'] - tt['Ten_Case_Date']).dt.days
tt['Days_Since_Hundred_Cases'] = (tt['Date'] - tt['Hundred_Case_Date']).dt.days
tt['Days_Since_First_Fatal'] = (tt['Date'] - tt['First_Fatal_Date']).dt.days
tt['Days_Since_Ten_Fatal'] = (tt['Date'] - tt['Ten_Fatal_Date']).dt.days
tt['Days_Since_Hundred_Fatal'] = (tt['Date'] - tt['Hundred_Fatal_Date']).dt.days

# Merge smoking data
smoking = pd.read_csv("../input/smokingstats/share-of-adults-who-smoke.csv")
smoking = smoking.rename(columns={'Smoking prevalence, total (ages 15+) (% of adults)': 'Smoking_Rate'})
smoking_dict = smoking.groupby('Entity')['Year'].max().to_dict()
smoking['LastYear'] = smoking['Entity'].map(smoking_dict)
smoking = smoking.query('Year == LastYear').reset_index()
smoking['Entity'] = smoking['Entity'].str.replace('United States', 'US')

tt = tt.merge(smoking[['Entity','Smoking_Rate']],
         left_on='Country_Region',
         right_on='Entity',
         how='left',
         validate='m:1') \
    .drop('Entity', axis=1)

# Country data
country_info = pd.read_csv('../input/countryinfo/covid19countryinfo.csv')


tt = tt.merge(country_info, left_on=['Country_Region','Province_State'],
              right_on=['country','region'],
              how='left',
              validate='m:1')

# State info from wikipedia
us_state_info = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states_by_population')[0]     [['State','Population estimate, July 1, 2019[2]']]     .rename(columns={'Population estimate, July 1, 2019[2]' : 'Population'})
#us_state_info['2019 population'] = pd.to_numeric(us_state_info['2019 population'].str.replace('[note 1]','').replace('[]',''))

tt = tt.merge(us_state_info[['State','Population']],
         left_on='Province_State',
         right_on='State',
         how='left')

tt['pop'] = pd.to_numeric(tt['pop'].str.replace(',',''))
tt['pop'] = tt['pop'].fillna(tt['Population'])
tt['pop'] = pd.to_numeric(tt['pop'])

tt['pop_diff'] = tt['pop'] - tt['Population']
tt['Population_final'] = tt['Population']
tt.loc[~tt['hasProvidence'], 'Population_final'] = tt.loc[~tt['hasProvidence']]['pop']

tt['Confirmed_Cases_Diff'] = tt.groupby('Place')['ConfirmedCases'].diff()
tt['Fatailities_Diff'] = tt.groupby('Place')['Fatalities'].diff()
max_date = tt.dropna(subset=['ConfirmedCases'])['Date'].max()
tt['gdp2019'] = pd.to_numeric(tt['gdp2019'].str.replace(',',''))


# In[ ]:


# Correcting population for missing countries
# Googled their names and copied the numbers here
pop_dict = {'Angola': int(29.78 * 10**6),
            'Australia_Australian Capital Territory': 423_800,
            'Australia_New South Wales': int(7.544 * 10**6),
            'Australia_Northern Territory': 244_300,
            'Australia_Queensland' : int(5.071 * 10**6),
            'Australia_South Australia' : int(1.677 * 10**6),
            'Australia_Tasmania': 515_000,
            'Australia_Victoria': int(6.359 * 10**6),
            'Australia_Western Australia': int(2.589 * 10**6),
            'Brazil': int(209.3 * 10**6),
            'Canada_Alberta' : int(4.371 * 10**6),
            'Canada_British Columbia' : int(5.071 * 10**6),
            'Canada_Manitoba' : int(1.369 * 10**6),
            'Canada_New Brunswick' : 776_827,
            'Canada_Newfoundland and Labrador' : 521_542,
            'Canada_Nova Scotia' : 971_395,
            'Canada_Ontario' : int(14.57 * 10**6),
            'Canada_Prince Edward Island' : 156_947,
            'Canada_Quebec' : int(8.485 * 10**6),
            'Canada_Saskatchewan': int(1.174 * 10**6),
            'China_Anhui': int(62 * 10**6),
            'China_Beijing': int(21.54 * 10**6),
            'China_Chongqing': int(30.48 * 10**6),
            'China_Fujian' :  int(38.56 * 10**6),
            'China_Gansu' : int(25.58 * 10**6),
            'China_Guangdong' : int(113.46 * 10**6),
            'China_Guangxi' : int(48.38 * 10**6),
            'China_Guizhou' : int(34.75 * 10**6),
            'China_Hainan' : int(9.258 * 10**6),
            'China_Hebei' : int(74.7 * 10**6),
            'China_Heilongjiang' : int(38.31 * 10**6),
            'China_Henan' : int(94 * 10**6),
            'China_Hong Kong' : int(7.392 * 10**6),
            'China_Hubei' : int(58.5 * 10**6),
            'China_Hunan' : int(67.37 * 10**6),
            'China_Inner Mongolia' :  int(24.71 * 10**6),
            'China_Jiangsu' : int(80.4 * 10**6),
            'China_Jiangxi' : int(45.2 * 10**6),
            'China_Jilin' : int(27.3 * 10**6),
            'China_Liaoning' : int(43.9 * 10**6),
            'China_Macau' : 622_567,
            'China_Ningxia' : int(6.301 * 10**6),
            'China_Qinghai' : int(5.627 * 10**6),
            'China_Shaanxi' : int(37.33 * 10**6),
            'China_Shandong' : int(92.48 * 10**6),
            'China_Shanghai' : int(24.28 * 10**6),
            'China_Shanxi' : int(36.5 * 10**6),
            'China_Sichuan' : int(81.1 * 10**6),
            'China_Tianjin' : int(15 * 10**6),
            'China_Tibet' : int(3.18 * 10**6),
            'China_Xinjiang' : int(21.81 * 10**6),
            'China_Yunnan' : int(45.97 * 10**6),
            'China_Zhejiang' : int(57.37 * 10**6),
            'Denmark_Faroe Islands' : 51_783,
            'Denmark_Greenland' : 56_171,
            'France_French Guiana' : 290_691,
            'France_French Polynesia' : 283_007,
            'France_Guadeloupe' : 395_700,
            'France_Martinique' : 376_480,
            'France_Mayotte' : 270_372,
            'France_New Caledonia' : 99_926,
            'France_Reunion' : 859_959,
            'France_Saint Barthelemy' : 9_131,
            'France_St Martin' : 32_125,
            'Netherlands_Aruba' : 105_264,
            'Netherlands_Curacao' : 161_014,
            'Netherlands_Sint Maarten' : 41_109,
            'Papua New Guinea' : int(8.251 * 10**6),
            'US_Guam' : 164_229,
            'US_Virgin Islands' : 107_268,
            'United Kingdom_Bermuda' : 65_441,
            'United Kingdom_Cayman Islands' : 61_559,
            'United Kingdom_Channel Islands' : 170_499,
            'United Kingdom_Gibraltar' : 34_571,
            'United Kingdom_Isle of Man' : 84_287,
            'United Kingdom_Montserrat' : 4_922
           }

tt['Population_final'] = tt['Population_final'].fillna(tt['Place'].map(pop_dict))


# In[ ]:


tt.loc[tt['Place'] == 'Diamond Princess', 'Population final'] = 2_670


# In[ ]:


tt['ConfirmedCases_Log'] = tt['ConfirmedCases'].apply(np.log1p)
tt['Fatalities_Log'] = tt['Fatalities'].apply(np.log1p)


# In[ ]:


tt['Population_final'] = tt['Population_final'].astype('int')
tt['Cases_Per_100kPop'] = (tt['ConfirmedCases'] / tt['Population_final']) * 100000
tt['Fatalities_Per_100kPop'] = (tt['Fatalities'] / tt['Population_final']) * 100000

tt['Cases_Percent_Pop'] = ((tt['ConfirmedCases'] / tt['Population_final']) * 100)
tt['Fatalities_Percent_Pop'] = ((tt['Fatalities'] / tt['Population_final']) * 100)

tt['Cases_Log_Percent_Pop'] = ((tt['ConfirmedCases'] / tt['Population_final']) * 100).apply(np.log1p)
tt['Fatalities_Log_Percent_Pop'] = ((tt['Fatalities'] / tt['Population_final']) * 100).apply(np.log1p)


tt['Max_Confirmed_Cases'] = tt.groupby('Place')['ConfirmedCases'].transform(max)
tt['Max_Fatalities'] = tt.groupby('Place')['Fatalities'].transform(max)

tt['Max_Cases_Per_100kPop'] = tt.groupby('Place')['Cases_Per_100kPop'].transform(max)
tt['Max_Fatalities_Per_100kPop'] = tt.groupby('Place')['Fatalities_Per_100kPop'].transform(max)


# In[ ]:


tt.query('Date == @max_date')     .query('Place != "Diamond Princess"')     .query('Cases_Log_Percent_Pop > -10000')     ['Cases_Log_Percent_Pop'].plot(kind='hist', bins=500)
plt.show()


# In[ ]:


tt.query('Days_Since_Ten_Cases > 0')     .query('Place != "Diamond Princess"')     .dropna(subset=['Cases_Percent_Pop'])     .query('Days_Since_Ten_Cases < 40')     .plot(x='Days_Since_Ten_Cases', y='Cases_Log_Percent_Pop', style='.', figsize=(15, 5), alpha=0.2)
plt.show()


# In[ ]:


PLOT = False
if PLOT:
    for x in tt['Place'].unique():
        try:
            fig, ax = plt.subplots(1, 4, figsize=(15, 2))
            tt.query('Place == @x')                 .query('ConfirmedCases > 0')                 .set_index('Date')['Cases_Log_Percent_Pop']                 .plot(title=f'{x} confirmed log pct pop', ax=ax[0])
            tt.query('Place == @x')                 .query('ConfirmedCases > 0')                 .set_index('Date')['Cases_Percent_Pop']                 .plot(title=f'{x} confirmed cases', ax=ax[1])
            tt.query('Place == @x')                 .query('Fatalities > 0')                 .set_index('Date')['Fatalities_Log_Percent_Pop']                 .plot(title=f'{x} confirmed log pct pop', ax=ax[2])
            tt.query('Place == @x')                 .query('Fatalities > 0')                 .set_index('Date')['Fatalities_Percent_Pop']                 .plot(title=f'{x} confirmed cases', ax=ax[3])
        except:
            pass
        plt.show()


# In[ ]:


tt.query('Date == @max_date')[['Place','Max_Cases_Per_100kPop',
                               'Max_Fatalities_Per_100kPop','Max_Confirmed_Cases',
                               'Population_final',
                              'Days_Since_First_Case',
                              'Confirmed_Cases_Diff']] \
    .drop_duplicates() \
    .sort_values('Max_Cases_Per_100kPop', ascending=False)


# # Features about cases since first case/ fatality

# In[ ]:


tt['Past_7Days_ConfirmedCases_Std'] = tt['Place'].map(tt.dropna(subset=['ConfirmedCases']).query('Date >= 20200324').groupby('Place')['ConfirmedCases'].std().to_dict())
tt['Past_7Days_Fatalities_Std'] = tt['Place'].map(tt.dropna(subset=['ConfirmedCases']).query('Date >= 20200324').groupby('Place')['Fatalities'].std().to_dict())

tt['Past_7Days_ConfirmedCases_Min'] = tt['Place'].map(tt.dropna(subset=['ConfirmedCases']).query('Date >= 20200324').groupby('Place')['ConfirmedCases'].min().to_dict())
tt['Past_7Days_Fatalities_Min'] = tt['Place'].map(tt.dropna(subset=['Fatalities']).query('Date >= 20200324').groupby('Place')['Fatalities'].min().to_dict())

tt['Past_7Days_ConfirmedCases_Max'] = tt['Place'].map(tt.dropna(subset=['ConfirmedCases']).query('Date >= 20200324').groupby('Place')['ConfirmedCases'].max().to_dict())
tt['Past_7Days_Fatalities_Max'] = tt['Place'].map(tt.dropna(subset=['Fatalities']).query('Date >= 20200324').groupby('Place')['Fatalities'].max().to_dict())

tt['Past_7Days_Confirmed_Change_of_Total'] = (tt['Past_7Days_ConfirmedCases_Max'] - tt['Past_7Days_ConfirmedCases_Min']) / (tt['Past_7Days_ConfirmedCases_Max'])
tt['Past_7Days_Fatalities_Change_of_Total'] = (tt['Past_7Days_Fatalities_Max'] - tt['Past_7Days_Fatalities_Min']) / (tt['Past_7Days_Fatalities_Max'])


# In[ ]:


tt['Past_21Days_ConfirmedCases_Std'] = tt['Place'].map(tt.dropna(subset=['ConfirmedCases']).query('Date >= 20200310').groupby('Place')['ConfirmedCases'].std().to_dict())
tt['Past_21Days_Fatalities_Std'] = tt['Place'].map(tt.dropna(subset=['ConfirmedCases']).query('Date >= 20200310').groupby('Place')['Fatalities'].std().to_dict())

tt['Past_21Days_ConfirmedCases_Min'] = tt['Place'].map(tt.dropna(subset=['ConfirmedCases']).query('Date >= 20200310').groupby('Place')['ConfirmedCases'].min().to_dict())
tt['Past_21Days_Fatalities_Min'] = tt['Place'].map(tt.dropna(subset=['Fatalities']).query('Date >= 20200324').groupby('Place')['Fatalities'].min().to_dict())

tt['Past_21Days_ConfirmedCases_Max'] = tt['Place'].map(tt.dropna(subset=['ConfirmedCases']).query('Date >= 20200310').groupby('Place')['ConfirmedCases'].max().to_dict())
tt['Past_21Days_Fatalities_Max'] = tt['Place'].map(tt.dropna(subset=['Fatalities']).query('Date >= 20200310').groupby('Place')['Fatalities'].max().to_dict())

tt['Past_21Days_Confirmed_Change_of_Total'] = (tt['Past_21Days_ConfirmedCases_Max'] - tt['Past_21Days_ConfirmedCases_Min']) / (tt['Past_21Days_ConfirmedCases_Max'])
tt['Past_21Days_Fatalities_Change_of_Total'] = (tt['Past_21Days_Fatalities_Max'] - tt['Past_21Days_Fatalities_Min']) / (tt['Past_21Days_Fatalities_Max'])

tt['Past_7Days_Fatalities_Change_of_Total'] = tt['Past_7Days_Fatalities_Change_of_Total'].fillna(0)
tt['Past_21Days_Fatalities_Change_of_Total'] = tt['Past_21Days_Fatalities_Change_of_Total'].fillna(0)


# In[ ]:


tt['Date_7Days_Since_First_Case'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Case'] == 7]     .set_index('Place')['Date']     .to_dict())
tt['Date_14Days_Since_First_Case'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Case'] == 14]     .set_index('Place')['Date']     .to_dict())
tt['Date_21Days_Since_First_Case'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Case'] == 21]     .set_index('Place')['Date']     .to_dict())
tt['Date_28Days_Since_First_Case'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Case'] == 28]     .set_index('Place')['Date']     .to_dict())
tt['Date_35Days_Since_First_Case'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Case'] == 35]     .set_index('Place')['Date']     .to_dict())
tt['Date_60Days_Since_First_Case'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Case'] == 60]     .set_index('Place')['Date']     .to_dict())

tt['Date_7Days_Since_Ten_Cases'] = tt['Place'].map(tt.loc[tt['Days_Since_Ten_Cases'] == 7]     .set_index('Place')['Date']     .to_dict())
tt['Date_14Days_Since_Ten_Cases'] = tt['Place'].map(tt.loc[tt['Days_Since_Ten_Cases'] == 14]     .set_index('Place')['Date']     .to_dict())
tt['Date_21Days_Since_Ten_Cases'] = tt['Place'].map(tt.loc[tt['Days_Since_Ten_Cases'] == 21]     .set_index('Place')['Date']     .to_dict())
tt['Date_28Days_Since_Ten_Cases'] = tt['Place'].map(tt.loc[tt['Days_Since_Ten_Cases'] == 28]     .set_index('Place')['Date']     .to_dict())
tt['Date_35Days_Since_Ten_Cases'] = tt['Place'].map(tt.loc[tt['Days_Since_Ten_Cases'] == 35]     .set_index('Place')['Date']     .to_dict())
tt['Date_60Days_Since_Ten_Cases'] = tt['Place'].map(tt.loc[tt['Days_Since_Ten_Cases'] == 60]     .set_index('Place')['Date']     .to_dict())


tt['Date_7Days_Since_First_Fatal'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Fatal'] == 7]     .set_index('Place')['Date']     .to_dict())
tt['Date_14Days_Since_First_Fatal'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Fatal'] == 14]     .set_index('Place')['Date']     .to_dict())
tt['Date_21Days_Since_First_Fatal'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Fatal'] == 21]     .set_index('Place')['Date']     .to_dict())
tt['Date_28Days_Since_First_Fatal'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Fatal'] == 28]     .set_index('Place')['Date']     .to_dict())
tt['Date_35Days_Since_First_Fatal'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Fatal'] == 35]     .set_index('Place')['Date']     .to_dict())
tt['Date_60Days_Since_First_Fatal'] = tt['Place'].map(tt.loc[tt['Days_Since_First_Fatal'] == 60]     .set_index('Place')['Date']     .to_dict())


# In[ ]:


tt['CC_7D_1stCase'] = tt.loc[tt['Date_7Days_Since_First_Case'] == tt['Date']]['ConfirmedCases']
tt['CC_14D_1stCase'] = tt.loc[tt['Date_14Days_Since_First_Case'] == tt['Date']]['ConfirmedCases']
tt['CC_21D_1stCase'] = tt.loc[tt['Date_21Days_Since_First_Case'] == tt['Date']]['ConfirmedCases']
tt['CC_28D_1stCase'] = tt.loc[tt['Date_28Days_Since_First_Case'] == tt['Date']]['ConfirmedCases']
tt['CC_35D_1stCase'] = tt.loc[tt['Date_35Days_Since_First_Case'] == tt['Date']]['ConfirmedCases']
tt['CC_60D_1stCase'] = tt.loc[tt['Date_60Days_Since_First_Case'] == tt['Date']]['ConfirmedCases']

tt['F_7D_1stCase'] = tt.loc[tt['Date_7Days_Since_First_Case'] == tt['Date']]['Fatalities']
tt['F_14D_1stCase'] = tt.loc[tt['Date_14Days_Since_First_Case'] == tt['Date']]['Fatalities']
tt['F_21D_1stCase'] = tt.loc[tt['Date_21Days_Since_First_Case'] == tt['Date']]['Fatalities']
tt['F_28D_1stCase'] = tt.loc[tt['Date_28Days_Since_First_Case'] == tt['Date']]['Fatalities']
tt['F_35D_1stCase'] = tt.loc[tt['Date_35Days_Since_First_Case'] == tt['Date']]['Fatalities']
tt['F_60D_1stCase'] = tt.loc[tt['Date_60Days_Since_First_Case'] == tt['Date']]['Fatalities']

tt['CC_7D_10Case'] = tt.loc[tt['Date_7Days_Since_Ten_Cases'] == tt['Date']]['ConfirmedCases']
tt['CC_14D_10Case'] = tt.loc[tt['Date_14Days_Since_Ten_Cases'] == tt['Date']]['ConfirmedCases']
tt['CC_21D_10Case'] = tt.loc[tt['Date_21Days_Since_Ten_Cases'] == tt['Date']]['ConfirmedCases']
tt['CC_28D_10Case'] = tt.loc[tt['Date_28Days_Since_Ten_Cases'] == tt['Date']]['ConfirmedCases']
tt['CC_35D_10Case'] = tt.loc[tt['Date_35Days_Since_Ten_Cases'] == tt['Date']]['ConfirmedCases']
tt['CC_60D_10Case'] = tt.loc[tt['Date_60Days_Since_Ten_Cases'] == tt['Date']]['ConfirmedCases']

tt['F_7D_10Case'] = tt.loc[tt['Date_7Days_Since_Ten_Cases'] == tt['Date']]['Fatalities']
tt['F_14D_10Case'] = tt.loc[tt['Date_14Days_Since_Ten_Cases'] == tt['Date']]['Fatalities']
tt['F_21D_10Case'] = tt.loc[tt['Date_21Days_Since_Ten_Cases'] == tt['Date']]['Fatalities']
tt['F_28D_10Case'] = tt.loc[tt['Date_28Days_Since_Ten_Cases'] == tt['Date']]['Fatalities']
tt['F_35D_10Case'] = tt.loc[tt['Date_35Days_Since_Ten_Cases'] == tt['Date']]['Fatalities']
tt['F_60D_10Case'] = tt.loc[tt['Date_60Days_Since_Ten_Cases'] == tt['Date']]['Fatalities']

tt['CC_7D_1Fatal'] = tt.loc[tt['Date_7Days_Since_First_Fatal'] == tt['Date']]['ConfirmedCases']
tt['CC_14D_1Fatal'] = tt.loc[tt['Date_14Days_Since_First_Fatal'] == tt['Date']]['ConfirmedCases']
tt['CC_21D_1Fatal'] = tt.loc[tt['Date_21Days_Since_First_Fatal'] == tt['Date']]['ConfirmedCases']
tt['CC_28D_1Fatal'] = tt.loc[tt['Date_28Days_Since_First_Fatal'] == tt['Date']]['ConfirmedCases']
tt['CC_35D_1Fatal'] = tt.loc[tt['Date_35Days_Since_First_Fatal'] == tt['Date']]['ConfirmedCases']
tt['CC_60D_1Fatal'] = tt.loc[tt['Date_60Days_Since_First_Fatal'] == tt['Date']]['ConfirmedCases']

tt['F_7D_1Fatal'] = tt.loc[tt['Date_7Days_Since_First_Fatal'] == tt['Date']]['Fatalities']
tt['F_14D_1Fatal'] = tt.loc[tt['Date_14Days_Since_First_Fatal'] == tt['Date']]['Fatalities']
tt['F_21D_1Fatal'] = tt.loc[tt['Date_21Days_Since_First_Fatal'] == tt['Date']]['Fatalities']
tt['F_28D_1Fatal'] = tt.loc[tt['Date_28Days_Since_First_Fatal'] == tt['Date']]['Fatalities']
tt['F_35D_1Fatal'] = tt.loc[tt['Date_35Days_Since_First_Fatal'] == tt['Date']]['Fatalities']
tt['F_60D_1Fatal'] = tt.loc[tt['Date_60Days_Since_First_Fatal'] == tt['Date']]['Fatalities']


# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(15, 5))
tt[['Place','Past_7Days_Confirmed_Change_of_Total','Past_7Days_Fatalities_Change_of_Total',
    'Past_7Days_ConfirmedCases_Max','Past_7Days_ConfirmedCases_Min',
   'Past_7Days_Fatalities_Max','Past_7Days_Fatalities_Min']] \
    .drop_duplicates() \
    .sort_values('Past_7Days_Confirmed_Change_of_Total')['Past_7Days_Confirmed_Change_of_Total'] \
    .plot(kind='hist', bins=50, title='Distribution of Pct change confirmed past 7 days', ax=axs[0])
tt[['Place','Past_21Days_Confirmed_Change_of_Total','Past_21Days_Fatalities_Change_of_Total',
    'Past_21Days_ConfirmedCases_Max','Past_21Days_ConfirmedCases_Min',
   'Past_21Days_Fatalities_Max','Past_21Days_Fatalities_Min']] \
    .drop_duplicates() \
    .sort_values('Past_21Days_Confirmed_Change_of_Total')['Past_21Days_Confirmed_Change_of_Total'] \
    .plot(kind='hist', bins=50, title='Distribution of Pct change confirmed past 21 days', ax=axs[1])
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
tt[['Place','Past_7Days_Fatalities_Change_of_Total']]     .drop_duplicates()['Past_7Days_Fatalities_Change_of_Total']     .plot(kind='hist', bins=50, title='Distribution of Pct change confirmed past 7 days', ax=axs[0])
tt[['Place', 'Past_21Days_Fatalities_Change_of_Total']]     .drop_duplicates()['Past_21Days_Fatalities_Change_of_Total']     .plot(kind='hist', bins=50, title='Distribution of Pct change confirmed past 21 days', ax=axs[1])
plt.show()


# In[ ]:


# Example of flat prop
tt.query("Place == 'China_Chongqing'").set_index('Date')['ConfirmedCases'].dropna().plot(figsize=(15, 5))
plt.show()


# # List of Places to keep cases constant

# In[ ]:


# Assume the places with small rate of change will continue slow down of virus spread
constant_case_places = tt.loc[(tt['Past_21Days_Confirmed_Change_of_Total'] < 0.01) & (tt['ConfirmedCases'] > 10)]['Place'].unique()
constant_case_places


# # List of Places to keep fatailities constant

# In[ ]:


# Assume the places with small rate of change will continue slow down of virus spread
constant_fatal_places = tt.loc[(tt['Past_21Days_Fatalities_Change_of_Total'] < 0.01) & (tt['Fatalities'] > 1)]['Place'].unique()
constant_fatal_places


# # Other Data Prep

# In[ ]:


# Example of flat prop
tt.query("Place == 'Italy'").set_index('Date')[['ConfirmedCases']].dropna().plot(figsize=(15, 5))
plt.show()
tt.query("Place == 'Italy'").set_index('Date')[['ConfirmedCases_Log']].dropna().plot(figsize=(15, 5))
plt.show()


# In[ ]:


latest_summary_stats = tt.query('Date == @max_date')     [['Country_Region',
      'Place',
      'Max_Cases_Per_100kPop',
      'Max_Fatalities_Per_100kPop',
      'Max_Confirmed_Cases',
      'Population_final',
      'Days_Since_First_Case',
      'Days_Since_Ten_Cases']] \
    .drop_duplicates()


# In[ ]:


latest_summary_stats.query('Place != "Diamond Princess"')     .query('Country_Region != "China"')     .plot(y='Max_Cases_Per_100kPop',
          x='Days_Since_Ten_Cases',
          style='.',
          figsize=(15, 5))


# In[ ]:


tt.query('Province_State == "Maryland"')[['ConfirmedCases','Confirmed_Cases_Diff']]


# # Create Models

# # Falalities per day

# In[ ]:


# train model to predict fatalities/day
# params
SEED = 42
params = {'num_leaves': 8,
          'min_data_in_leaf': 5,  # 42,
          'objective': 'regression',
          'max_depth': 8,
          'learning_rate': 0.02,
      #    'boosting': 'gbdt',
          'bagging_freq': 5,  # 5
          'bagging_fraction': 0.8,  # 0.5,
          'feature_fraction': 0.8201,
          'bagging_seed': SEED,
          'reg_alpha': 1,  # 1.728910519108444,
          'reg_lambda': 4.9847051755586085,
          'random_state': SEED,
          'metric': 'mse',
          'verbosity': 100,
         'n_estimators' : 500,
          'min_gain_to_split': 0.02,  # 0.01077313523861969,
          'min_child_weight': 5,  # 19.428902804238373,
          'num_threads': 6,
          }


# In[ ]:


import lightgbm as lgb
pd.set_option('max_columns', 500)
try:
    tt['healthexp'] = pd.to_numeric(tt['healthexp'].str.replace(',',''))
except:
    pass
train = tt.loc[tt['Date'] < '25-March-2020']
test = tt.loc[tt['Date'] >= '25-March-2020']


# In[ ]:


# [c for c in tt.columns]


# In[ ]:


FEATURES = ['doy', 'dow', 'hasProvidence',
       'lat', 'lon', 'Days_Since_First_Case', 'Days_Since_Ten_Cases',
       'Days_Since_Hundred_Cases', 'Days_Since_First_Fatal',
       'Days_Since_Ten_Fatal', 'Days_Since_Hundred_Fatal', 'Smoking_Rate',
            'Population_final',
#             'tests',
#        'testpop', 'density', 'medianage', 'urbanpop',
#             'quarantine', 'schools',
#        'publicplace',
            #'gatheringlimit', 'gathering', 'nonessential',
       'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54', 'sex64',
       'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung', 'gdp2019',
       'healthexp', 'healthperpop', 'fertility', 
            #'firstcase', 'totalcases',
       #'activecases', 'newcases', 'deaths', 'newdeaths', 'recovered',
       #'critical', 'casediv1m', 'deathdiv1m',
             'CC_7D_1stCase',
 'CC_14D_1stCase',
 'CC_21D_1stCase',
 'CC_28D_1stCase',
 'CC_35D_1stCase',
 'CC_60D_1stCase',
 'F_7D_1stCase',
 'F_14D_1stCase',
 'F_21D_1stCase',
 'F_28D_1stCase',
 'F_35D_1stCase',
 'F_60D_1stCase',
 'CC_7D_10Case',
 'CC_14D_10Case',
 'CC_21D_10Case',
 'CC_28D_10Case',
 'CC_35D_10Case',
 'CC_60D_10Case',
 'F_7D_10Case',
 'F_14D_10Case',
 'F_21D_10Case',
 'F_28D_10Case',
 'F_35D_10Case',
 'F_60D_10Case',
 'CC_7D_1Fatal',
 'CC_14D_1Fatal',
 'CC_21D_1Fatal',
 'CC_28D_1Fatal',
 'CC_35D_1Fatal',
 'CC_60D_1Fatal',
 'F_7D_1Fatal',
 'F_14D_1Fatal',
 'F_21D_1Fatal',
 'F_28D_1Fatal',
 'F_35D_1Fatal',
 'F_60D_1Fatal'
           ]
TARGET = ['ConfirmedCases_Log']


# In[ ]:


reg = lgb.LGBMRegressor(**params)
reg.fit(train[FEATURES], train[TARGET])


# In[ ]:


test['ConfirmedCases_Log_Pred'] = reg.predict(test[FEATURES])
test['ConfirmedCases_pred'] = test['ConfirmedCases_Log_Pred'].apply(np.expm1)


# # Results dont look good

# In[ ]:


test.set_index('Date')[['Place','ConfirmedCases_pred','ConfirmedCases']].query('Place == "US_Maryland"').plot(figsize=(15, 5))
plt.show()

test.set_index('Date')[['Place','ConfirmedCases_pred','ConfirmedCases']].query('Place == "Bhutan"').plot(figsize=(15, 5))
plt.show()


# # Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression, ElasticNet


# In[ ]:


us_states = tt.query('Country_Region == "US"')['Place'].unique()
us_states


# # US States

# In[ ]:


myplace = "US_Florida"
tt.query('Place == @myplace and Days_Since_First_Fatal >= 0')[['Days_Since_First_Fatal','Fatalities_Log']]


# In[ ]:


for myplace in us_states:
    try:
        # Confirmed Cases
        fig, axs = plt.subplots(1, 2, figsize=(15, 3))
        dat = tt.query('Place == @myplace and Days_Since_Ten_Cases >= 0')[['Days_Since_Ten_Cases','ConfirmedCases_Log']].dropna()
        X = dat['Days_Since_Ten_Cases']
        y = dat['ConfirmedCases_Log']
        y = y.cummax()
        dat_all = tt.query('Place == @myplace and Days_Since_Ten_Cases >= 0')[['Days_Since_Ten_Cases','ConfirmedCases_Log']]
        X_pred = dat_all['Days_Since_Ten_Cases']
        en = ElasticNet()
        en.fit(X.values.reshape(-1, 1), y.values)
        preds = en.predict(X_pred.values.reshape(-1, 1))
        tt.loc[(tt['Place'] == myplace) & (tt['Days_Since_Ten_Cases'] >= 0), 'ConfirmedCases_Log_Pred1'] = preds
        tt.loc[(tt['Place'] == myplace), 'ConfirmedCases_Pred1'] = tt['ConfirmedCases_Log_Pred1'].apply(np.expm1)
        # Cap at 10 % Population
        pop_myplace = tt.query('Place == @myplace')['Population_final'].values[0]
        tt.loc[(tt['Place'] == myplace) & (tt['ConfirmedCases_Pred1'] > (0.1 * pop_myplace)), 'ConfirmedCases_Pred1'] = (0.1 * pop_myplace)
        ax = tt.query('Place == @myplace').set_index('Date')[['ConfirmedCases','ConfirmedCases_Pred1']].plot(figsize=(15, 5), title=myplace, ax=axs[0])
        # Fatalities
        # If low count then do percent of confirmed:
        dat = tt.query('Place == @myplace and Days_Since_Ten_Cases >= 0')[['Days_Since_Ten_Cases','Fatalities_Log']].dropna()
        if len(dat) < 5:
            tt.loc[(tt['Place'] == myplace), 'Fatalities_Pred1'] = tt.loc[(tt['Place'] == myplace)]['ConfirmedCases_Pred1'] * 0.001
        elif tt.query('Place == @myplace')['Fatalities'].max() < 5:
            tt.loc[(tt['Place'] == myplace), 'Fatalities_Pred1'] = tt.loc[(tt['Place'] == myplace)]['ConfirmedCases_Pred1'] * 0.001
        else:
            X = dat['Days_Since_Ten_Cases']
            y = dat['Fatalities_Log']
            y = y.cummax()
            dat_all = tt.query('Place == @myplace and Days_Since_Ten_Cases >= 0')[['Days_Since_Ten_Cases','Fatalities_Log']]
            X_pred = dat_all['Days_Since_Ten_Cases']
            en = ElasticNet()
            en.fit(X.values.reshape(-1, 1), y.values)
            preds = en.predict(X_pred.values.reshape(-1, 1))
            tt.loc[(tt['Place'] == myplace) & (tt['Days_Since_Ten_Cases'] >= 0), 'Fatalities_Log_Pred1'] = preds
            tt.loc[(tt['Place'] == myplace), 'Fatalities_Pred1'] = tt['Fatalities_Log_Pred1'].apply(np.expm1)

            # Cap at 0.0001 Population
            pop_myplace = tt.query('Place == @myplace')['Population_final'].values[0]
            tt.loc[(tt['Place'] == myplace) & (tt['Fatalities_Pred1'] > (0.0005 * pop_myplace)), 'Fatalities_Pred1'] = (0.0005 * pop_myplace)

        ax = tt.query('Place == @myplace').set_index('Date')[['Fatalities','Fatalities_Pred1']].plot(figsize=(15, 5), title=myplace, ax=axs[1])
        plt.show()
    except:
        print(f'============= FAILED FOR {myplace} =============')


# In[ ]:


myplace = 'US_Virgin Islands'
dat = tt.query('Place == @myplace and Days_Since_Ten_Cases >= 0')[['Place','Days_Since_Ten_Cases','ConfirmedCases']].dropna()


# In[ ]:


tt.loc[tt['Place'].isin(us_states)].groupby('Place')['Fatalities_Pred1'].max().sum()


# # Deal with Flattened location

# In[ ]:


constant_fatal_places


# In[ ]:


tt.loc[tt['Place'].isin(constant_fatal_places), 'ConfirmedCases_Pred1'] = tt.loc[tt['Place'].isin(constant_fatal_places)]['Place'].map(tt.loc[tt['Place'].isin(constant_fatal_places)].groupby('Place')['ConfirmedCases'].max())
tt.loc[tt['Place'].isin(constant_fatal_places), 'Fatalities_Pred1'] = tt.loc[tt['Place'].isin(constant_fatal_places)]['Place'].map(tt.loc[tt['Place'].isin(constant_fatal_places)].groupby('Place')['Fatalities'].max())


# In[ ]:


for myplace in constant_fatal_places:
    fig, axs = plt.subplots(1, 2, figsize=(15, 3))
    ax = tt.query('Place == @myplace').set_index('Date')[['ConfirmedCases','ConfirmedCases_Pred1']].plot(figsize=(15, 5), title=myplace, ax=axs[0])
    ax = tt.query('Place == @myplace').set_index('Date')[['Fatalities','Fatalities_Pred1']].plot(figsize=(15, 5), title=myplace, ax=axs[1])
    plt.show()


# # Remaining Locations
# ## 217 left

# In[ ]:


remaining_places = pd.DataFrame(tt.groupby('Place')['ConfirmedCases_Pred1'].max().isna()).query('ConfirmedCases_Pred1').index.values
print(remaining_places)
print(len(remaining_places))


# In[ ]:


for myplace in remaining_places:
    try:
        # Confirmed Cases
        fig, axs = plt.subplots(1, 2, figsize=(15, 3))
        dat = tt.query('Place == @myplace and Days_Since_Ten_Cases >= 0')[['Days_Since_Ten_Cases','ConfirmedCases_Log']].dropna()
        X = dat['Days_Since_Ten_Cases']
        y = dat['ConfirmedCases_Log']
        y = y.cummax()
        dat_all = tt.query('Place == @myplace and Days_Since_Ten_Cases >= 0')[['Days_Since_Ten_Cases','ConfirmedCases_Log']]
        X_pred = dat_all['Days_Since_Ten_Cases']
        en = ElasticNet()
        en.fit(X.values.reshape(-1, 1), y.values)
        preds = en.predict(X_pred.values.reshape(-1, 1))
        tt.loc[(tt['Place'] == myplace) & (tt['Days_Since_Ten_Cases'] >= 0), 'ConfirmedCases_Log_Pred1'] = preds
        tt.loc[(tt['Place'] == myplace), 'ConfirmedCases_Pred1'] = tt['ConfirmedCases_Log_Pred1'].apply(np.expm1)
        # Cap at 10 % Population
        pop_myplace = tt.query('Place == @myplace')['Population_final'].values[0]
        tt.loc[(tt['Place'] == myplace) & (tt['ConfirmedCases_Pred1'] > (0.1 * pop_myplace)), 'ConfirmedCases_Pred1'] = (0.1 * pop_myplace)
        ax = tt.query('Place == @myplace').set_index('Date')[['ConfirmedCases','ConfirmedCases_Pred1']].plot(figsize=(15, 5), title=myplace, ax=axs[0])
        # Fatalities
        # If low count then do percent of confirmed:
        dat = tt.query('Place == @myplace and Days_Since_Ten_Cases >= 0')[['Days_Since_Ten_Cases','Fatalities_Log']].dropna()
        if len(dat) < 5:
            tt.loc[(tt['Place'] == myplace), 'Fatalities_Pred1'] = tt.loc[(tt['Place'] == myplace)]['ConfirmedCases_Pred1'] * 0.001
        elif tt.query('Place == @myplace')['Fatalities'].max() < 5:
            tt.loc[(tt['Place'] == myplace), 'Fatalities_Pred1'] = tt.loc[(tt['Place'] == myplace)]['ConfirmedCases_Pred1'] * 0.001
        else:
            X = dat['Days_Since_Ten_Cases']
            y = dat['Fatalities_Log']
            y = y.cummax()
            dat_all = tt.query('Place == @myplace and Days_Since_Ten_Cases >= 0')[['Days_Since_Ten_Cases','Fatalities_Log']]
            X_pred = dat_all['Days_Since_Ten_Cases']
            en = ElasticNet()
            en.fit(X.values.reshape(-1, 1), y.values)
            preds = en.predict(X_pred.values.reshape(-1, 1))
            tt.loc[(tt['Place'] == myplace) & (tt['Days_Since_Ten_Cases'] >= 0), 'Fatalities_Log_Pred1'] = preds
            tt.loc[(tt['Place'] == myplace), 'Fatalities_Pred1'] = tt['Fatalities_Log_Pred1'].apply(np.expm1)

            # Cap at 0.0001 Population
            pop_myplace = tt.query('Place == @myplace')['Population_final'].values[0]
            tt.loc[(tt['Place'] == myplace) & (tt['Fatalities_Pred1'] > (0.0005 * pop_myplace)), 'Fatalities_Pred1'] = (0.0005 * pop_myplace)

        ax = tt.query('Place == @myplace').set_index('Date')[['Fatalities','Fatalities_Pred1']].plot(figsize=(15, 5), title=myplace, ax=axs[1])
        plt.show()
    except:
        print(f'============= FAILED FOR {myplace} =============')
        ax = tt.query('Place == @myplace').set_index('Date')[['ConfirmedCases','ConfirmedCases_Pred1']].plot(figsize=(15, 5), title=myplace, ax=axs[0])
        ax = tt.query('Place == @myplace').set_index('Date')[['Fatalities','Fatalities_Pred1']].plot(figsize=(15, 5), title=myplace, ax=axs[1])


# In[ ]:


# Estimated total
tt.groupby('Place')['Fatalities_Pred1'].max().sum()


# In[ ]:


# Clean Up any time the actual is less than the real
tt['ConfirmedCases_Pred1'] = tt[['ConfirmedCases','ConfirmedCases_Pred1']].max(axis=1)
tt['Fatalities_Pred1'] = tt[['Fatalities','Fatalities_Pred1']].max(axis=1)

tt['ConfirmedCases_Pred1'] = tt['ConfirmedCases_Pred1'].fillna(0)
tt['Fatalities_Pred1'] = tt['Fatalities_Pred1'].fillna(0)

# Fill pred with
tt.loc[~tt['ConfirmedCases'].isna(), 'ConfirmedCases_Pred1'] = tt.loc[~tt['ConfirmedCases'].isna()]['ConfirmedCases']
tt.loc[~tt['Fatalities'].isna(), 'Fatalities_Pred1'] = tt.loc[~tt['Fatalities'].isna()]['Fatalities']

tt['ConfirmedCases_Pred1'] = tt.groupby('Place')['ConfirmedCases_Pred1'].transform('cummax')
tt['Fatalities_Pred1'] = tt.groupby('Place')['Fatalities_Pred1'].transform('cummax')


# # Plot them all!!

# In[ ]:


for myplace in tt['Place'].unique():
    try:
        # Confirmed Cases
        fig, axs = plt.subplots(1, 2, figsize=(15, 3))
        ax = tt.query('Place == @myplace').set_index('Date')[['ConfirmedCases','ConfirmedCases_Pred1']].plot(title=myplace, ax=axs[0])
        ax = tt.query('Place == @myplace').set_index('Date')[['Fatalities','Fatalities_Pred1']].plot(title=myplace, ax=axs[1])
        plt.show()
    except:
        print(f'============= FAILED FOR {myplace} =============')


# In[ ]:


tt.groupby('Place')['Fatalities_Pred1'].max().sort_values()


# In[ ]:


# Questionable numbers
tt.query('Place == "Iran"').set_index('Date')[['ConfirmedCases',
                                               'ConfirmedCases_Pred1',]].plot(figsize=(15, 5))
# Make Iran's Predictions Linear

tt.query('Place == "Iran"').set_index('Date')[['Fatalities','Fatalities_Pred1']].plot(figsize=(15, 5))


# # Make Iran's Predictions Linear

# In[ ]:


dat.iloc[-10:]


# In[ ]:


for myplace in ['Iran']:

    # Confirmed Cases
    fig, axs = plt.subplots(1, 2, figsize=(15, 3))
    dat = tt.query('Place == @myplace and Days_Since_Ten_Cases >= 0')[['Days_Since_Ten_Cases','ConfirmedCases']].dropna()
    dat = dat.iloc[-10:]
    X = dat['Days_Since_Ten_Cases']
    y = dat['ConfirmedCases']
    y = y.cummax()
    dat_all = tt.query('Place == @myplace and Days_Since_Ten_Cases >= 0')[['Days_Since_Ten_Cases','ConfirmedCases']]
    X_pred = dat_all['Days_Since_Ten_Cases']
    en = ElasticNet()
    en.fit(X.values.reshape(-1, 1), y.values)
    preds = en.predict(X_pred.values.reshape(-1, 1))
    tt.loc[(tt['Place'] == myplace) & (tt['Days_Since_Ten_Cases'] >= 0), 'ConfirmedCases_Pred1'] = preds
    # Cap at 10 % Population
    pop_myplace = tt.query('Place == @myplace')['Population_final'].values[0]
    tt.loc[(tt['Place'] == myplace) & (tt['ConfirmedCases_Pred1'] > (0.1 * pop_myplace)), 'ConfirmedCases_Pred1'] = (0.1 * pop_myplace)
    ax = tt.query('Place == @myplace').set_index('Date')[['ConfirmedCases','ConfirmedCases_Pred1']].plot(figsize=(15, 5), title=myplace, ax=axs[0])
    # Fatalities
    # If low count then do percent of confirmed:
    dat = tt.query('Place == @myplace and Days_Since_Ten_Cases >= 0')[['Days_Since_Ten_Cases','Fatalities']].dropna()
    dat = dat.iloc[-10:]
    if len(dat) < 5:
        tt.loc[(tt['Place'] == myplace), 'Fatalities_Pred1'] = tt.loc[(tt['Place'] == myplace)]['ConfirmedCases_Pred1'] * 0.001
    elif tt.query('Place == @myplace')['Fatalities'].max() < 5:
        tt.loc[(tt['Place'] == myplace), 'Fatalities_Pred1'] = tt.loc[(tt['Place'] == myplace)]['ConfirmedCases_Pred1'] * 0.001
    else:
        X = dat['Days_Since_Ten_Cases']
        y = dat['Fatalities']
        y = y.cummax()
        dat_all = tt.query('Place == @myplace and Days_Since_Ten_Cases >= 0')[['Days_Since_Ten_Cases','Fatalities']]
        X_pred = dat_all['Days_Since_Ten_Cases']
        en = ElasticNet()
        en.fit(X.values.reshape(-1, 1), y.values)
        preds = en.predict(X_pred.values.reshape(-1, 1))
        tt.loc[(tt['Place'] == myplace) & (tt['Days_Since_Ten_Cases'] >= 0), 'Fatalities_Pred1'] = preds

        # Cap at 0.0001 Population
        pop_myplace = tt.query('Place == @myplace')['Population_final'].values[0]
        tt.loc[(tt['Place'] == myplace) & (tt['Fatalities_Pred1'] > (0.0005 * pop_myplace)), 'Fatalities_Pred1'] = (0.0005 * pop_myplace)

    ax = tt.query('Place == @myplace').set_index('Date')[['Fatalities','Fatalities_Pred1']].plot(figsize=(15, 5), title=myplace, ax=axs[1])
    plt.show()


# In[ ]:


# Clean Up any time the actual is less than the real
tt['ConfirmedCases_Pred1'] = tt[['ConfirmedCases','ConfirmedCases_Pred1']].max(axis=1)
tt['Fatalities_Pred1'] = tt[['Fatalities','Fatalities_Pred1']].max(axis=1)

tt['ConfirmedCases_Pred1'] = tt['ConfirmedCases_Pred1'].fillna(0)
tt['Fatalities_Pred1'] = tt['Fatalities_Pred1'].fillna(0)

# Fill pred with
tt.loc[~tt['ConfirmedCases'].isna(), 'ConfirmedCases_Pred1'] = tt.loc[~tt['ConfirmedCases'].isna()]['ConfirmedCases']
tt.loc[~tt['Fatalities'].isna(), 'Fatalities_Pred1'] = tt.loc[~tt['Fatalities'].isna()]['Fatalities']

tt['ConfirmedCases_Pred1'] = tt.groupby('Place')['ConfirmedCases_Pred1'].transform('cummax')
tt['Fatalities_Pred1'] = tt.groupby('Place')['Fatalities_Pred1'].transform('cummax')


# In[ ]:


# Questionable numbers
tt.query('Place == "Iran"').set_index('Date')[['ConfirmedCases',
                                               'ConfirmedCases_Pred1',]].plot(figsize=(15, 5))
# Make Iran's Predictions Linear

tt.query('Place == "Iran"').set_index('Date')[['Fatalities','Fatalities_Pred1']].plot(figsize=(15, 5))


# # Make Submission

# In[ ]:


ss = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')


# In[ ]:


print(ss.shape)
ss.head()


# In[ ]:


mysub = tt.dropna(subset=['ForecastId'])[['ForecastId','ConfirmedCases_Pred1','Fatalities_Pred1']]
mysub['ForecastId'] = mysub['ForecastId'].astype('int')
mysub = mysub.rename(columns={'ConfirmedCases_Pred1':'ConfirmedCases',
                      'Fatalities_Pred1': 'Fatalities'})
mysub.to_csv('submission.csv', index=False)


# # Summary Stats

# In[ ]:


# Questionable numbers
tt.groupby('Date').sum()[['ConfirmedCases',
                          'ConfirmedCases_Pred1',]].plot(figsize=(15, 5))
# Make Iran's Predictions Linear

tt.groupby('Date').sum()[['Fatalities',
                          'Fatalities_Pred1']].plot(figsize=(15, 5))

