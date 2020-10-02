#!/usr/bin/env python
# coding: utf-8

# # Lunar Lunacy: Myth or Reality?
# ![Lunatic Etymology](https://drive.google.com/uc?id=1R7TCE4gas5Hs1xIJL-YMi-_1Gy1metii)
# 
# Do you know that Lunar Cycle induces psychological changes in living beings, making humans more violent or losing their sanity on Full Moon night? Even hospital staff claims that there is more number of admissions on a Full Moon night!
# 
# This Lunar-Lunacy connection appears in multiple contexts ranging from increased birth rates to more number of suicides, homicides, traffic accidents and what not! There is an entire [Wikipedia article](https://en.wikipedia.org/wiki/Lunar_effect) devoted to same and [this Scientific American article](https://www.scientificamerican.com/article/lunacy-and-the-full-moon/) sums it up really well. **_But, if you don't want any SPOILERS, I suggest NOT to go over there!_**
# 
# In this notebook, I have tested the Lunar Lunacy effect on road accidents in UK. Does Full Moon night really affect no. of accidents & casualities? Let's analyse...

# In[ ]:


import numpy as np
import pandas as pd


# # Preparing Accident Data

# In[ ]:


accident_data = pd.read_csv('../input/uk-road-safety-accidents-and-vehicles/Accident_Information.csv')
accident_data


# In[ ]:


accident_data.columns


# In[ ]:


adata = accident_data[['Accident_Index', 'Local_Authority_(District)', 'Number_of_Casualties', 'Date', 'Time']].set_index('Accident_Index')
adata['DateTime'] = pd.to_datetime(adata['Date'] + ' ' + adata['Time'])  # Date and Time combined as datetime64 type
adata


# Now, select only London area (or Greater London) records, since we have lunar data only for it (read data description to know more).
# 
# There are total 33 districts in Greater London as given in https://en.wikipedia.org/wiki/List_of_London_boroughs, following is a list of them:

# In[ ]:


london_dist = [
'Barking and Dagenham',
'Barnet',
'Bexley',
'Brent',
'Bromley',
'Camden',
'Croydon',
'Ealing',
'Enfield',
'Greenwich',
'Hackney',
'Hammersmith and Fulham',
'Haringey',
'Harrow',
'Havering',
'Hillingdon',
'Hounslow',
'Islington',
'Kensington and Chelsea',
'Kingston upon Thames',
'Lambeth',
'Lewisham',
'Merton',
'Newham',
'Redbridge',
'Richmond upon Thames',
'Southwark',
'Sutton',
'Tower Hamlets',
'Waltham Forest',
'Wandsworth',
'Westminster',
'City of London'
]
len(london_dist)


# In[ ]:


adata = adata[adata['Local_Authority_(District)'].isin(london_dist)]
adata


# # Preparing Lunar Data
# Technically, full moon occurs only at an instant of time (i.e. given by PhaseTime). But as per the myths of lunar lunacy, the effect happens thoughout the night of full moon. So our task is to find those nights for which moon was *nearly full*.
# 
# And keep in mind that, this **full moon night** will span two dates because by midnight, new date starts.

# In[ ]:


lunar_data = pd.read_csv('../input/moonrise-moonset-and-phase-uk-2005-2017/UK_Lunar_Data.csv')
lunar_data


# Since we are interested in only full moon nights, let's select all the dates when the full moon occured (`Phase` = `FullMoon`) as well as all dates adjacent to it. Also, fill `Phase` as `FM_Previous` in dates previous to it and `FM_Next` in dates next to it, so that we can refer such dates easily in next steps.

# In[ ]:


ldata = lunar_data[lunar_data['Phase']=='Full Moon']
lunar_data.loc[ldata.index-1, 'Phase'] = 'FM_Previous'
lunar_data.loc[ldata.index+1, 'Phase'] = 'FM_Next'
ldata = pd.concat([ldata, lunar_data[(lunar_data['Phase']=='FM_Previous') | (lunar_data['Phase']=='FM_Next')]]).sort_index()
ldata


# In[ ]:


ldata.info()


# Since all values of `MoonriseEarly` is null, we can conclude that moon always rise later than when it rises, on the dates of full moon and its adjacent dates. So let's drop `MoonriseEarly` and also convert date and time columns in their respective datatypes.

# In[ ]:


ldata['Date'] = pd.to_datetime(ldata['Date'], dayfirst=True)
ldata['Moonset'] = pd.to_timedelta(ldata['Moonset']+':00')
ldata['Moonrise'] = pd.to_timedelta(ldata['MoonriseLate']+':00')
ldata['PhaseTime'] = pd.to_timedelta(ldata['PhaseTime']+':00')
ldata['Dusk'] = pd.to_timedelta(ldata['CivilDusk']+':00')
ldata['Dawn'] = pd.to_timedelta(ldata['CivilDawn']+':00')
ldata.drop(['MoonriseEarly', 'MoonriseLate', 'CivilDusk', 'CivilDawn'], axis=1, inplace=True)
ldata.reset_index(drop=True, inplace=True)
ldata


# Now, Let's explore this data by Plotly - beautiful & interactive!
# 
# For sake of plotting, times have been converted from timedelta64 to hours (in decimals). And notice that,
# - timings (hours) < 0 represent **Previous date's events**, and
# - timings (hours) > 24 represent **Next date's events**.

# In[ ]:


import plotly.graph_objects as go
fig = go.Figure()

# Full Moon Date (Current)
fm_ldata = ldata[ldata['Phase']=='Full Moon']
fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=fm_ldata['Moonrise']/np.timedelta64(1, 'h'),
                         name='Moonrise', line=dict(color='darkviolet')))
fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=fm_ldata['Moonset']/np.timedelta64(1, 'h'),
                         name='Moonset', line=dict(color='darkred')))
fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=fm_ldata['PhaseTime']/np.timedelta64(1, 'h'),
                         name='PhaseTime', line=dict(color='limegreen')))
fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=fm_ldata['Dusk']/np.timedelta64(1, 'h'),
                         name='Dusk', line=dict(color='deepskyblue')))
fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=fm_ldata['Dawn']/np.timedelta64(1, 'h'),
                         name='Dawn', line=dict(color='darkorange')))

# Previous Date
prev_ldata = ldata[ldata['Phase']=='FM_Previous']
fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=prev_ldata['Moonrise']/np.timedelta64(1, 'h') - 24,
                         name='Moonrise (Previous)', line=dict(color='violet')))
fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=prev_ldata['Dusk']/np.timedelta64(1, 'h') - 24,
                         name='Dusk (Previous)', line=dict(color='lightskyblue')))

# Next Date
next_ldata = ldata[ldata['Phase']=='FM_Next']
fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=next_ldata['Moonset'] / np.timedelta64(1, 'h') + 24,
                         name='Moonset (Next)', line=dict(color='orangered')))
fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=next_ldata['Dawn'] / np.timedelta64(1, 'h') + 24,
                         name='Dawn (Next)', line=dict(color='orange')))

fig.update_layout(xaxis=dict(title_text='Dates when Full Moon Instance occured'),
                  yaxis=dict(tick0=0, dtick=4, title_text='Timing (in Hours)'),
                  title_text='Comparison of several Astronomical event timings on Full Moon Dates',
                  xaxis_rangeslider_visible=True)
fig.show()


# OK, so we have very interesting observations from this plot:
# 1. Moonrise almost always happens just before Dusk and Moonset almost always happens just after Dawn, for full moon dates. Therefore, approximate full moon (since moon is completely full only for a time instant) will always be available after dusk until dawn. So now ownwards, we will only consider Dusk and Dawn times when finding full moon night period.
# 
# 2. We have 3 cases for full moon PhaseTime:
#     1. **FullMoon < Dawn** (when green line comes below dark-orange, in graph) - Dates when fullmoon instance was before dawn i.e. after 00:00 of previous date's night (<0 in graph).
#     2. **Dawn < FullMoon < Dusk** (when green line lies in between sky-blue and dark-orange, in graph) - Dates when fullmoon instance was after dawn, before dusk i.e. in daylight.
#     3. **Dusk < FullMoon** (when green line comes above blue, in graph) - Dates when fullmoon instance was after dusk i.e. before 00:00 of current date's night (>24 in graph).
# 
# Building upon this information, we can reduce above plot in such a way that we can better deduce the **full moon night period** from **full moon instance** (PhaseTime).

# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=fm_ldata['PhaseTime']/np.timedelta64(1, 'h'),
                         name='Full Moon Instance', mode='markers', line=dict(color='limegreen')))

fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=fm_ldata['Dawn']/np.timedelta64(1, 'h'),
                         name='Dawn', line=dict(color='darkorange')))

fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=prev_ldata['Dusk']/np.timedelta64(1, 'h') - 24,
                         name='Dusk (Previous)', fill='tonexty', line=dict(color='lightskyblue')))

fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=next_ldata['Dawn'] / np.timedelta64(1, 'h') + 24,
                         name='Dawn (Next)', line=dict(color='orange')))

fig.add_trace(go.Scatter(x=fm_ldata['Date'], y=fm_ldata['Dusk']/np.timedelta64(1, 'h'),
                         name='Dusk', fill='tonexty', line=dict(color='deepskyblue')))

fig.update_layout(xaxis=dict(title_text='Dates when Full Moon Instance occured'),
                  yaxis=dict(tick0=0, dtick=4, title_text='Timing (in Hours)'),
                  title_text='Occurance of FullMoon Instance in two Nights',
                  xaxis_rangeslider_visible=True)
fig.show()


# > The shaded region here indicates the night period.
# 
# Now, coming back to 3 cases mentioned above:
# 
# - **Case-1:** Full Moon night period becomes the time span between dusk of previous date and dawn of current date (i.e. coming night).
# 
# - **Case-3:** Similarly, Full Moon night period becomes the time span between dusk of current date and dawn of next date (i.e last night).
# 
# - **Case-2:** It's quite trickier! Let's split the time span between dawn and dusk of current date in 2 halves. Then:
#     - If full moon instance lies in lower half, we will take full moon night period same as in case 1 (since full moon instance is closer to last night).
#     - Else, we will take full moon night period same as in case 3 (since full moon instance is closer to coming night).

# In[ ]:


def get_last_night(row):
    night_start = ldata.loc[row.name-1, 'Date'] + ldata.loc[row.name-1, 'Dusk']  # Previous Dusk
    night_end = row['Date'] + row['Dawn']  # Dawn
    return (night_start, night_end)

def get_coming_night(row):
    night_start = row['Date'] + row['Dusk']  # Dusk
    night_end = ldata.loc[row.name+1, 'Date'] + ldata.loc[row.name+1, 'Dawn']  # Next Dawn
    return (night_start, night_end)

def get_full_moon_nights(row):
    if row['PhaseTime'] < row['Dawn']:
        return get_last_night(row)
        
    elif row['Dusk'] < row['PhaseTime']:
        return get_coming_night(row)
        
    else:
        mid = (row['Dawn']+row['Dusk'])/2
        if row['PhaseTime'] < mid:
            return get_last_night(row)
        else:
            return get_coming_night(row)


# In[ ]:


# Full Moon (fm) Nights
fm_nights = pd.DataFrame(list(fm_ldata.apply(get_full_moon_nights, axis=1)), columns=['NightStart', 'NightEnd'])
fm_nights

Hence, we obtain full moon night periods as `fm_nights`.

But, we also need to find ordinary night periods to contrast the two groups for road accidents.
# # Finding all Night Periods
# So that we can filter out accident data, for nights!

# In[ ]:


all_ldata = lunar_data[['Date', 'CivilDusk', 'CivilDawn']].rename(columns={'CivilDusk': 'Dusk', 'CivilDawn': 'Dawn'})
all_ldata['Date'] = pd.to_datetime(all_ldata['Date'], dayfirst=True)
all_ldata['Dusk'] = pd.to_timedelta(all_ldata['Dusk']+':00')
all_ldata['Dawn'] = pd.to_timedelta(all_ldata['Dawn']+':00')

def get_all_nights(row):
    night_start = row['Date'] + row['Dusk']
    night_end = all_ldata.loc[row.name+1, 'Date'] + all_ldata.loc[row.name+1, 'Dawn']  # Using next date's Dawn
    return (night_start, night_end)

all_nights = pd.DataFrame(list(all_ldata.iloc[:-1].apply(get_all_nights, axis=1)), columns=['NightStart', 'NightEnd'])
all_nights


# Label the full moon nights as 1 and ordinary nights as 0

# In[ ]:


all_nights['FullMoon'] = 0
all_nights.loc[all_nights['NightStart'].isin(fm_nights['NightStart']), 'FullMoon'] = 1
all_nights


# In[ ]:


all_nights[all_nights['FullMoon']==1]


# # Finding records in Accidents data that fall in Night Periods

# In[ ]:


# Interval Index formed from all_nights 
all_nights_idx = pd.IntervalIndex.from_arrays(all_nights['NightStart'],all_nights['NightEnd'],closed='both')

# Creating a mapping of adata indices with all_nights_idx, where adata['DateTime'] falls in all_nights_idx
adata_nights_idx_map = pd.Series(all_nights_idx.get_indexer(adata['DateTime']), index=adata.index)
adata_nights_idx_map


# In[ ]:


# Dropping rows where mapped value is -1 since those accidents didn't happen in night
adata_nights_idx_map = adata_nights_idx_map[adata_nights_idx_map != -1]

# Selecting all rows from adata where accidents happened in night (non -1)
adata = adata.loc[adata_nights_idx_map.index]
adata


# In[ ]:


# Finding all night periods corresonding to accidents that happned in night
nights_in_adata = all_nights.loc[adata_nights_idx_map]
nights_in_adata.index = adata_nights_idx_map.index
nights_in_adata


# In[ ]:


# Combining night periods data in adata
adata = pd.concat([adata, nights_in_adata], axis=1)
adata


# # Grouping Accident data by Night Periods
# 
# Now to analyse accidents data in context of a particular night, we will group adata by night period

# In[ ]:


# Night periods as Intervals
adata_night_idx = pd.IntervalIndex.from_arrays(adata['NightStart'],adata['NightEnd'],closed='both')


# In[ ]:


# Summarized adata
summ_adata = adata.groupby(adata_night_idx).agg(Accidents_Count=('DateTime', 'size'),
                                   Total_Casualities=('Number_of_Casualties', 'sum'),
                                   Full_Moon=('FullMoon', 'max'))  # FullMoon value is same for each night group
summ_adata


# In[ ]:


# To get adata for Full Moon Nights
summ_adata[summ_adata['Full_Moon']==1]


# Also, we can explore this data by following plots:

# In[ ]:


def plot_summ_adata(column_name):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=summ_adata.index.left.date, y=summ_adata[column_name],
                             name='Ordinary Night', line=dict(color='mediumpurple')))

    fm_summ_adata = summ_adata[summ_adata['Full_Moon']==1]
    fig.add_trace(go.Scatter(x=fm_summ_adata.index.left.date, y=fm_summ_adata[column_name],
                             name='Full Moon Night', mode='markers', line=dict(color='darkorange')))

    fig.update_layout(xaxis=dict(title_text='Date of Night\'s beginning',
                                 rangeselector=dict(buttons=list([
                                  dict(count=1, label="1m", step="month", stepmode="backward"),
                                  dict(count=6, label="6m", step="month", stepmode="backward"),
                                  dict(count=1, label="1y", step="year", stepmode="backward"),
                                  dict(count=4, label="4y", step="year", stepmode="backward"),
                                  dict(step="all")])),
                                 rangeslider=dict(visible=True,
                                                  range=['2005-01-01', '2017-12-31']),  # Span rangeslider for Date range of entire adata
                                 range=['2017-07-01', '2017-12-31'],  # Show on x-axis only date Range of last half year in adata
                                 type='date'
                                ),
                      yaxis=dict(title_text=column_name),
                      title_text='{} per Night'.format(column_name))
    fig.show()


# In[ ]:


plot_summ_adata('Accidents_Count')


# In[ ]:


plot_summ_adata('Total_Casualities')


# # Performing independent t-test
# 
# Now we have our data in a form, ready to apply the hypothesis testing! Let's define our hypotheses first:

# ### Alternate Hypothesis (Ha):
# There is increase (difference) in no. of accidents (and casualities) on full moon nights as compared to ordinary nights.
# ### Null Hypothesis (H0):
# There is **no** difference in no. of accidents (and casualities) on full moon nights as compared to ordinary nights.

# In[ ]:


# To test using scipy ttest_ind
from scipy import stats

# Full Moon nights
fm_summ_adata = summ_adata[summ_adata['Full_Moon']==1]

# Ordinary Nights
non_fm_summ_adata = summ_adata[summ_adata['Full_Moon']==0]


# In[ ]:


stats.ttest_ind(fm_summ_adata['Accidents_Count'], non_fm_summ_adata['Accidents_Count'])


# In[ ]:


stats.ttest_ind(fm_summ_adata['Total_Casualities'], non_fm_summ_adata['Total_Casualities'])


# Since pvalue is much greater than 0.1, therefore null hypothesis fails to be rejeccted. 
# 
# ### Hence we can conclude that no. of accidents and casualities on full moon nights is no different than that on ordinary nights.

# # Comparing Averages
# We can further verify this by comparing average number of accidents and casualities for full moonnights and ordinary nights.

# In[ ]:


fig = go.Figure(data=[
    go.Bar(name='Ordinary Night', x=summ_adata.columns[:-1], y=non_fm_summ_adata.mean().iloc[:-1], marker_color='purple'),
    go.Bar(name='Full Moon Night', x=summ_adata.columns[:-1], y=fm_summ_adata.mean().iloc[:-1], marker_color='orange')
])

fig.update_layout(
    title='Comparison of Averages',
    yaxis=dict(title='Mean'),
    bargap=0.3,
    bargroupgap=0.05
)
fig.show()


# Clearly, there is no much difference in averages on full moon night when compared with ordinary night. Even we can see averages are quite low on full moon nights, if we look closely. This also indicates that **no. of accidents or casualities are unaffected by full moon.**

# <hr>
# <br>
# ### Pointers for more things to analyse
# - Check especially with motorcyclists (from vehicles dataset) as they're directly exposed to moonlight
# - Can test for accident severity (although majority is slight)
# - Check especially for Friday 13th nights!
# - Can do some plotting on map using longitutde & latitude of road accidents in dataset
# - Anyhow prove the null hypothesis wrong by analysing what piques your curiosity!
