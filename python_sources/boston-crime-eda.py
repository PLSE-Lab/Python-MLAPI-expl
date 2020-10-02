#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import folium


# There are two input files provided crime.csv and offense_codes.csv; all the information which is required is present in the crime.csv file the other file is redundant.
# 
# For, a quick test use the below two commands to check the data.
# *input_data_offense_code.loc[input_data_offense_code['CODE'].isin([3410,1402,3114])]* - this dataframe will have values from offense_codes.csv
# *input_data_crime.loc[input_data_crime['OFFENSE_CODE'].isin([3410,1402,3114])]*

# In[ ]:


input_data_crime=pd.read_csv('../input/crimes-in-boston/crime.csv',encoding='latin-1')
input_data_crime.head()


# The data is giving us some very interesting details like Date, Time, Offense, Location.
# 
# Let's jump into it and visulize the data.

# In[ ]:


year_data=input_data_crime.groupby('YEAR').size().reset_index(name='number of crimes')
fig=go.Figure([go.Bar(x=year_data['YEAR'],y=year_data['number of crimes'],
                     text=year_data['number of crimes'], textposition='auto',
                     marker_color=['lightsteelblue','darkseagreen','darkseagreen','lightsteelblue'])])
fig.update_layout(
    autosize=False,
    title_text='Yearwise number of crimes',
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text='Year',
            font=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text='Number of Crimes',
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    )
)
fig.show()


# Based on the 'Yearwise number of crimes' chart there is a high number of crimes committed in 2016 and 2017, out of the 4 years data provided now lets find out the top 5 months when the highest number of crimes were committed.

# In[ ]:


year_month_data=input_data_crime.groupby(['YEAR','MONTH']).size().reset_index(name='number of crimes')
month_map={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
year_month_data['month_name']=year_month_data['MONTH'].apply(lambda x:month_map[x])

year_month_data_sample1=year_month_data[year_month_data['YEAR']==2015].nlargest(5,'number of crimes')
year_month_data_sample1=year_month_data_sample1.sort_values("MONTH")
year_month_data_sample2=year_month_data[year_month_data['YEAR']==2016].nlargest(5,'number of crimes')
year_month_data_sample2=year_month_data_sample2.sort_values("MONTH")
year_month_data_sample3=year_month_data[year_month_data['YEAR']==2017].nlargest(5,'number of crimes')
year_month_data_sample3=year_month_data_sample3.sort_values("MONTH")
year_month_data_sample4=year_month_data[year_month_data['YEAR']==2018].nlargest(5,'number of crimes')
year_month_data_sample4=year_month_data_sample4.sort_values("MONTH")

fig = make_subplots(
    rows=2, cols=2, subplot_titles=("2015", "2016", "2017", "2018")
)

fig.add_trace(go.Bar(x=year_month_data_sample1['month_name'], y=year_month_data_sample1['number of crimes'],
                    marker_color=['darkseagreen','darkseagreen','lightsteelblue','lightsteelblue','lightsteelblue']),row=1, col=1)
fig.add_trace(go.Bar(x=year_month_data_sample2['month_name'],y=year_month_data_sample1['number of crimes'],
                    marker_color=['lightsteelblue','lightsteelblue','darkseagreen','darkseagreen','lightsteelblue']),row=1,col=2)
fig.add_trace(go.Bar(x=year_month_data_sample3['month_name'],y=year_month_data_sample1['number of crimes'],
                    marker_color=['lightsteelblue','darkseagreen','darkseagreen','lightsteelblue','lightsteelblue']),row=2,col=1)
fig.add_trace(go.Bar(x=year_month_data_sample4['month_name'],y=year_month_data_sample1['number of crimes'],
                    marker_color=['lightsteelblue','lightsteelblue','lightsteelblue','darkseagreen','darkseagreen']),row=2,col=2)

fig.update_layout(
    showlegend=False,
    title_text='Year on year - Top 5 months crime comitted'
)

fig.show()


# In the above graph we can easily see that year on year July and August recorded the top 5 most crime commiting months. As I am focusing on the year 2016 and 2017, the graph for 2015 and 2018 is redundant but there is no harm in keeping it.
# 
# Focusing only on 2016 and 2017 lets find out the days on July and August when there was maximum number of crimes reported.

# In[ ]:


year_month_day_data=input_data_crime.groupby(['YEAR','MONTH','DAY_OF_WEEK']).size().reset_index(name='number of crimes')

year_month_day_data_sample1=year_month_day_data[year_month_day_data['YEAR']==2016][year_month_day_data['MONTH']==7].sort_values('number of crimes')
year_month_day_data_sample2=year_month_day_data[year_month_day_data['YEAR']==2016][year_month_day_data['MONTH']==8].sort_values('number of crimes')
year_month_day_data_sample3=year_month_day_data[year_month_day_data['YEAR']==2017][year_month_day_data['MONTH']==7].sort_values('number of crimes')
year_month_day_data_sample4=year_month_day_data[year_month_day_data['YEAR']==2017][year_month_day_data['MONTH']==8].sort_values('number of crimes')


# In[ ]:


fig = make_subplots(
    rows=2, cols=2, subplot_titles=("2016", "2016","2017","2017")
)

fig.add_trace(go.Bar(x=year_month_day_data_sample1['DAY_OF_WEEK'], y=year_month_day_data_sample1['number of crimes'],
                    marker_color=['lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue','darkseagreen']),row=1, col=1)
fig.add_trace(go.Bar(x=year_month_day_data_sample2['DAY_OF_WEEK'],y=year_month_day_data_sample2['number of crimes'],
                    marker_color=['lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue','darkseagreen']),row=1,col=2)
fig.add_trace(go.Bar(x=year_month_day_data_sample3['DAY_OF_WEEK'],y=year_month_day_data_sample3['number of crimes'],
                    marker_color=['lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue','darkseagreen']),row=2,col=1)
fig.add_trace(go.Bar(x=year_month_day_data_sample4['DAY_OF_WEEK'],y=year_month_day_data_sample4['number of crimes'],
                    marker_color=['lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue','darkseagreen']),row=2,col=2)


fig.update_xaxes(title_text="July",row=1,col=1)
fig.update_xaxes(title_text="August",row=1,col=2)
fig.update_xaxes(title_text="July",row=2,col=1)
fig.update_xaxes(title_text="August",row=2,col=2)

fig.update_yaxes(title_text="Number of crimes",row=1,col=1)
fig.update_yaxes(title_text="Number of crimes",row=1,col=2)
fig.update_yaxes(title_text="Number of crimes",row=2,col=1)
fig.update_yaxes(title_text="Number of crimes",row=2,col=2)

fig.update_layout(
    showlegend=False,
    title_text='Daywise breakdown number of crimes commited',
    height=700
)

fig.show()


# In[ ]:


year_month_day_hour_data=input_data_crime.groupby(['YEAR','MONTH','DAY_OF_WEEK','HOUR']).size().reset_index(name='number of crimes')

year_month_day_hour_data_sample1=year_month_day_hour_data[year_month_day_hour_data['YEAR']==2016][year_month_day_hour_data['MONTH']==7][year_month_day_hour_data['DAY_OF_WEEK']=='Friday']
year_month_day_hour_data_sample2=year_month_day_hour_data[year_month_day_hour_data['YEAR']==2016][year_month_day_hour_data['MONTH']==8][year_month_day_hour_data['DAY_OF_WEEK']=='Monday']
year_month_day_hour_data_sample3=year_month_day_hour_data[year_month_day_hour_data['YEAR']==2017][year_month_day_hour_data['MONTH']==7][year_month_day_hour_data['DAY_OF_WEEK']=='Monday']
year_month_day_hour_data_sample4=year_month_day_hour_data[year_month_day_hour_data['YEAR']==2017][year_month_day_hour_data['MONTH']==8][year_month_day_hour_data['DAY_OF_WEEK']=='Thursday']


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=year_month_day_hour_data_sample1['HOUR'],y=year_month_day_hour_data_sample1['number of crimes'],name='2016 - July, Friday',line_shape='linear',line_color='#DC143C'))
fig.add_trace(go.Scatter(x=year_month_day_hour_data_sample2['HOUR'],y=year_month_day_hour_data_sample2['number of crimes'],name='2016 - August, Monday',line_shape='linear',line_color='#BA55D3'))
fig.add_trace(go.Scatter(x=year_month_day_hour_data_sample3['HOUR'],y=year_month_day_hour_data_sample3['number of crimes'],name='2017 - July, Monday',line_shape='linear',line_color='#228B22'))
fig.add_trace(go.Scatter(x=year_month_day_hour_data_sample4['HOUR'],y=year_month_day_hour_data_sample4['number of crimes'],name='2017 - August, Thursday',line_shape='linear',line_color='#4169E1'))

fig.update_layout(
    showlegend=True,
    title_text='Time of the day when the crimes were committed.',
    font_size=10
)

fig.show()


# From the above two graphs we can see that on Friday there was maximum number of crimes commited in the month of July while Monday is the day in August for the year 2016.
# 
# Similarly, mondays are the most crime commited day in July 2017 and thursday in August 2017.
# 
# Taking the friday of July 2016, august of 2016 and 2017 and thursday of 2017 we can see the maximum number of crimes commited between 17:00 and 18:00 hours.
# 
# We now have the following details.
# - 2016 and 2017 saw the highest crimes committed
# - Summer (July, August) is the time when crimes are more active
# - Monday, Thrusday, Friday are the days of the week
# - 17:00 and 18:00 are the timne when crimes are committed
# 
# With the above data lets find out the area where the crimes were committed the most and the type of crimes which were committed the most.

# In[ ]:


year_month_day_hour_district_data_sample=input_data_crime[input_data_crime['YEAR']==2016][input_data_crime['MONTH']==7][input_data_crime['DAY_OF_WEEK']=='Friday'][input_data_crime['HOUR']==17]
year_month_day_hour_district_data_sample=year_month_day_hour_district_data_sample.append(input_data_crime[input_data_crime['YEAR']==2016][input_data_crime['MONTH']==7][input_data_crime['DAY_OF_WEEK']=='Friday'][input_data_crime['HOUR']==18])
year_month_day_hour_district_data_sample=year_month_day_hour_district_data_sample.append(input_data_crime[input_data_crime['YEAR']==2016][input_data_crime['MONTH']==8][input_data_crime['DAY_OF_WEEK']=='Monday'][input_data_crime['HOUR']==17])
year_month_day_hour_district_data_sample=year_month_day_hour_district_data_sample.append(input_data_crime[input_data_crime['YEAR']==2016][input_data_crime['MONTH']==8][input_data_crime['DAY_OF_WEEK']=='Monday'][input_data_crime['HOUR']==18])
year_month_day_hour_district_data_sample=year_month_day_hour_district_data_sample.append(input_data_crime[input_data_crime['YEAR']==2017][input_data_crime['MONTH']==7][input_data_crime['DAY_OF_WEEK']=='Monday'][input_data_crime['HOUR']==17])
year_month_day_hour_district_data_sample=year_month_day_hour_district_data_sample.append(input_data_crime[input_data_crime['YEAR']==2017][input_data_crime['MONTH']==7][input_data_crime['DAY_OF_WEEK']=='Monday'][input_data_crime['HOUR']==18])
year_month_day_hour_district_data_sample=year_month_day_hour_district_data_sample.append(input_data_crime[input_data_crime['YEAR']==2017][input_data_crime['MONTH']==8][input_data_crime['DAY_OF_WEEK']=='Thursday'][input_data_crime['HOUR']==17])
year_month_day_hour_district_data_sample=year_month_day_hour_district_data_sample.append(input_data_crime[input_data_crime['YEAR']==2017][input_data_crime['MONTH']==8][input_data_crime['DAY_OF_WEEK']=='Thursday'][input_data_crime['HOUR']==18])

year_month_day_hour_district_data_sample=year_month_day_hour_district_data_sample.drop(['SHOOTING'],axis=1)
year_month_day_hour_district_data_sample=year_month_day_hour_district_data_sample.dropna(how='any')

year_month_day_hour_district_data_sample1=year_month_day_hour_district_data_sample[year_month_day_hour_district_data_sample['OFFENSE_CODE']==3115]
year_month_day_hour_district_data_sample1=year_month_day_hour_district_data_sample1.append(year_month_day_hour_district_data_sample[year_month_day_hour_district_data_sample['OFFENSE_CODE']==3006])
year_month_day_hour_district_data_sample1=year_month_day_hour_district_data_sample1.append(year_month_day_hour_district_data_sample[year_month_day_hour_district_data_sample['OFFENSE_CODE']==3301])
year_month_day_hour_district_data_sample1=year_month_day_hour_district_data_sample1.append(year_month_day_hour_district_data_sample[year_month_day_hour_district_data_sample['OFFENSE_CODE']==3125])
year_month_day_hour_district_data_sample1=year_month_day_hour_district_data_sample1.append(year_month_day_hour_district_data_sample[year_month_day_hour_district_data_sample['OFFENSE_CODE']==802])


# In[ ]:


crime_map = folium.Map(location=[year_month_day_hour_district_data_sample['Lat'].mean(),year_month_day_hour_district_data_sample['Long'].mean()],zoom_start=12,tiles='OpenStreetMap')

for i in range(len(year_month_day_hour_district_data_sample1)):
    crime_map.add_child(folium.Marker(location=[year_month_day_hour_district_data_sample1.Lat.iloc[i],year_month_day_hour_district_data_sample1.Long.iloc[i]], 
        icon=folium.Icon("red" if year_month_day_hour_district_data_sample1.OFFENSE_CODE.iloc[i] == 3115
                        else 'orange' if year_month_day_hour_district_data_sample1.OFFENSE_CODE.iloc[i] == 3006
                        else 'blue' if year_month_day_hour_district_data_sample1.OFFENSE_CODE.iloc[i] == 3301
                        else 'gray' if year_month_day_hour_district_data_sample1.OFFENSE_CODE.iloc[i] == 3125
                        else 'green', 
        prefix='fa', icon='circle')))
    

crime_map


# The legend in the map is getting overlapped and combersium adding the text for each color<br>
# RED - INVESTIGATE PERSON<br>
# ORANGE - SICK/INJURED/MEDICAL - PERSON<br>
# BLUE - VERBAL DISPUTE<br>
# GRAY - WARRANT ARREST<br>
# GREEN - ASSAULT SIMPLE - BATTERY<br>
# 

# In[ ]:




