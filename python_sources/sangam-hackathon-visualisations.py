#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
from folium.plugins import TimestampedGeoJson,MarkerCluster 
data=pd.read_csv('../input/Sangam_2019_Hackathon_Data.csv')


# In[ ]:


import folium
from folium.plugins import TimestampedGeoJson
import branca

def make_map(features,color_man,caption='Temperature For Stations',period='H',f_name='temperature'):
    print('> Making map...')
    m = folium.Map(
    location=[28.457912,77.033994],
    zoom_start=9,
    tiles='openstreetmap')
    if period=='H':
        TimestampedGeoJson(
        {'type': 'FeatureCollection',
        'features': features}
        , period='PT1H'
        , add_last_point=True
        , auto_play=True
        , loop=False
        , max_speed=1000
        , loop_button=True
        , date_options='YYYY/MM/DD HH:MM'
        , time_slider_drag_update=True
        ).add_to(m)
    if period=='D':
        TimestampedGeoJson(
        {'type': 'FeatureCollection',
        'features': features}
        , period='P1D'
        , add_last_point=True
        , auto_play=True
        , loop=True
        , max_speed=1000
        , loop_button=True
        , date_options='YYYY/MM/DD'
        , time_slider_drag_update=True
        ).add_to(m)
        
        
        
    m.fit_bounds(m.get_bounds())
    
    color_man.caption = "{} ".format(f_name)
    m.add_child(color_man)    
    print('> Done.')
    return m



def create_geojson_features(df,colorer,feature_name='temperature'):
    print('> Creating GeoJSON features...')
    
    features = []
    print(colorer)
    for _, row in df.iterrows():
        feature = {
            'type': 'Feature',
            'geometry': {
                'type':'Point', 
                'coordinates':[row['longitude'],row['latitude']]
            },
            'properties': {
                'time': row['date_str'],
                'style': {'color' : colorer(row[feature_name])},
                'icon': 'circle',
                'iconstyle':{
                    'fillColor': colorer(row[feature_name]),
                    'fillOpacity': 0.8,
                    'stroke': 'true',
                    'radius': 16
                }
            }
        }
        features.append(feature)
    return features


def questionb1(data=None,frame_level='H',               device_list=['S3','S5', 'S3', 'S4', 'S1','S10', 'S8', 'S9', 'S6', 'S7'],               feature_col_name='temperature',              min_value=25,max_value=38,
              c_platter=None 
              ):
    
    if c_platter is None:
        colorscale_paletter = branca.colormap.linear.RdBu_09.scale(vmin=min_value,vmax=max_value)
    else :
        colorscale_paletter=c_platter
    if frame_level=='H':
        data['date_str']=data.svrtime.apply(lambda x :x.split(':')[0]+':00')
        date_format='YYYY/MM/DD HH:MM'
    if frame_level=='D':
        data['date_str']=data.svrtime.apply(lambda x :x.split(' ')[0])
        date_format='YYYY/MM/DD HH:MM'
    temp=data.loc[data.device_id.isin(device_list) & (data.latitude>10) & (data.longitude>50) &(data.latitude<39) ]    .groupby(by=['device_id','date_str'],as_index=False).agg({'latitude':'mean','longitude':'mean',feature_col_name:'mean'})
    print("Make sure the min and max are proptional for the series {} , the maxium minimum after aggregation is {} , {} we are using vmin {} vmax={} \n quantiles are {}"          .format(feature_col_name,temp[feature_col_name].max(),temp[feature_col_name].min(),min_value,max_value,temp[feature_col_name].quantile([0,0.25,0.5,0.75,0.9])))
    
    #print(temp.head())
    features_collected=create_geojson_features(temp,feature_name=feature_col_name,colorer=colorscale_paletter)
    #print(features_collected[0:100])
    mapped=make_map(features=features_collected,color_man=colorscale_paletter,period=frame_level,f_name=feature_col_name)
    return mapped
    
    


# # Geo Spatial Temporal trend - Temperature

# In[ ]:


test_map=questionb1(data,'H',feature_col_name='temperature')
test_map


# # Spatial Temporal Variation of Humidity

# In[ ]:


test_map2=questionb1(data,'H',device_list=['M1','M2'],feature_col_name='humidity',max_value=34)
test_map2


# # Spatial Temporal variation of PM01

# In[ ]:


test_map3=questionb1(data,'H',device_list=['M1','M2'],feature_col_name='pm01',max_value=44,min_value=13)
test_map3


# # Spatial Temporal Variation of PM2.5
# 
# 
# 

# In[ ]:


test_map4=questionb1(data,'H',device_list=['M1','M2'],feature_col_name='pm25',max_value=55,min_value=8)
test_map4


# # Spatial Temporal variation of PM10

# In[ ]:


test_map4=questionb1(data,'H',device_list=['S1','S2'],feature_col_name='pm01',max_value=44,min_value=13)
test_map4


# ### AQI

# In[ ]:


data.head().T


# In[ ]:


data['date_str']= data.svrtime.apply(lambda x:x.split(' ')[0])


# In[ ]:


data.columns


# ### O3 concerntration is missing which is needed to determine the air quality index 
# ### So2 concerntration should be calculated via s02_gas paramter

# In[ ]:



def get_aqi(row):
    ##
    #print(row)
    pm10,pm25,no2,co,nh3=row['pm10'],row['pm25'],row['no2'],row['co'],row['nh3']
    pm10_res=0
    pm25_res=0
    no2_res=0
    co_res=0
    nh3_res=0
    
    
    if(pm10<=50):
        pm10_res=0
    elif (pm10<=100):
        pm10_res=1
    elif (pm10<=250):
        pm10_res=2
    elif (pm10<=350):
        pm10_res=3
    elif (pm10<=430):
        pm10_res=4
    else:
        pm10_res=5
    
    
    if(pm25<=30):
        pm25_res=0
    elif (pm25<=60):
        pm25_res=1
    elif (pm25<=90):
        pm25_res=2
    elif (pm25<=120):
        pm25_res=3
    elif (pm25<=250):
        pm25_res=4
    else:
        pm25_res=5
    if(no2<=40):
        no2_res=0
    elif (no2<=80):
        no2_res=1
    elif (no2<=180):
        no2_res=2
    elif (no2<=280):
        no2_res=3
    elif (no2<=400):
        no2_res=4
    else:
        no2_res=5

    if(co<=1):
        co_res=0
    elif (co<=2):
        co_res=1
    elif (co<=10):
        co_res=2
    elif (co<=17):
        co_res=3
    elif (co<=34):
        co_res=4
    else:
        co_res=5
    if(nh3<=200):
        nh3_res=0
    elif (nh3<=400):
        nh3_res=1
    elif (nh3<=800):
        nh3_res=2
    elif (nh3<=1200):
        nh3_res=3
    elif (nh3<=1800):
        nh3_res=4
    else:
        nh3_res=5

    aqi=np.max(np.array([nh3_res,co_res,no2_res,pm25_res,pm10_res]))
    result={0:'Good',1:'Satisfactory',2:'Moderately polluted',3:'Poor',4:'Very Poor',5:'Severe'}
    #print(aqi)
    return aqi
    return result[aqi]





def aqi_plot(df=data,dev_list=['S3','S5', 'S3', 'S4', 'S1','S10', 'S8', 'S9', 'S6', 'S7']):
    temp=df.loc[data.device_id.isin(dev_list) & (data.latitude>10) & (data.longitude>50) &(data.latitude<39)]    .groupby(by=['date_str','device_id','latitude','longitude'],as_index=False).agg(    {'pm10':'mean','pm25':'mean','no2':'mean','co':'mean','nh3':'mean'})
    temp['date_str']=temp['date_str'].apply(lambda x:x.split(' ')[0])
    temp['aqi']=temp[['pm10', 'pm25', 'no2','co', 'nh3']].apply(lambda row:get_aqi(row).astype(float),axis=1)
    colorscale = branca.colormap.linear.YlOrRd_05.to_step(n=6,data=[0,1,2,3,4,5])
    feat=create_geojson_features(temp,colorer=colorscale,feature_name='no2')
    temp['svrtime']=temp.date_str.apply(lambda x : '{} 00:00:00'.format(x))
    return questionb1(data=temp,frame_level='D',device_list=dev_list,feature_col_name='aqi',min_value=0,max_value=5,c_platter=colorscale)
    


# # Air QUALITY INDEX PLOT

# In[ ]:


aqi_plot()


# In[ ]:





# # Clustering close by GPS sensors on spatial data

# In[ ]:


from sklearn.preprocessing import LabelEncoder
def make_clust_plot(df=data,dev_list=['S3','S5', 'S3', 'S4', 'S1','S10', 'S8', 'S9', 'S6', 'S7'],max_value=40,min_value=24,feature_name='temperature'):
    temp=df.loc[data.device_id.isin(dev_list) & (data.latitude>10) & (data.longitude>50) &(data.latitude<39)]    .groupby(by=['date_str','device_id','latitude','longitude'],as_index=False).agg(    {feature_name:'mean'})
    temp['date_str']=temp['date_str'].apply(lambda x:x.split(' ')[0])
    colorscale = branca.colormap.linear.YlOrRd_05.scale(min_value,max_value)
    device_id_encode=LabelEncoder().fit_transform(temp.device_id)
    print(temp.device_id.nunique())
    colorscale_rind = branca.colormap.linear.GnBu_05.to_step(n=temp.device_id.nunique()+1,data=device_id_encode)

    print('> >  >> > > > >> > > Making map > > > > > >> > > >> >> > >')
    print("Make sure the min and max are proptional for the series {} , the maxium minimum after aggregation is {} , {} we are using vmin {} vmax={} \n quantiles are {}"          .format(feature_name,temp[feature_name].max(),temp[feature_name].min(),min_value,max_value,temp[feature_name].quantile([0,0.25,0.5,0.75,0.9])))
    
    
    m = folium.Map(
    location=[28.457912,77.033994],
    zoom_start=9,
    tiles='openstreetmap')##tiles='Cartodb Positron'
    
    marker_cluster = MarkerCluster(
        name='Stations',
        overlay=True,
        control=False,
        icon_create_function=None
    )
    print(temp.shape)
    for k in range(temp.shape[0]):
        location = temp.latitude[k], temp.longitude[k]
        
        
        marker = folium.CircleMarker(location=location,color=colorscale_rind(device_id_encode[k]) ,                                     fill_opacity=0.7,                                     fill_color=colorscale(temp.values[k,4]),fill=True)
        popup = 'lon:{} lat:{} <br> {} = {} <br>device_id {}'.format(location[1], location[0],feature_name,temp.values[k,4],temp.device_id[k])
        folium.Popup(popup).add_to(marker)
        marker_cluster.add_child(marker)
        
    marker_cluster.add_to(m)
    
    #folium.LayerControl().add_to(m)
    colorscale.caption = "Geo Spatial clustering for {} ".format(feature_name)
    m.add_child(colorscale)  
    return m


# In[ ]:


r=make_clust_plot(feature_name='pm25',dev_list=['S3','S2'],max_value=36,min_value=30)
r


# In[ ]:


q=make_clust_plot(feature_name='humidity',dev_list=['M2'],max_value=36,min_value=30)
q

