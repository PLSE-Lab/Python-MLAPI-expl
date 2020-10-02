#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings; warnings.filterwarnings('ignore')
import geopandas,pandas,numpy,seaborn,folium
import sqlite3,os,sympy,pylab as plt
from matplotlib import cm
import matplotlib.patches as mpatches
plt.style.use('seaborn-whitegrid')
os.listdir('../input')


# In[ ]:


from IPython.core.display import display
from IPython.core.magic import register_line_magic
@register_line_magic
def get_query(q):
    sympy.pprint(r'SQL Queries')
    tr=[]; cursor.execute(q)
    result=cursor.fetchall()
    for r in result: tr+=[r]
    display(pandas.DataFrame.from_records(tr)            .style.set_table_styles(style_dict))
def connect_to_db(dbf):
    sqlconn=None
    try:
        sqlconn=sqlite3.connect(dbf)
        return sqlconn
    except Error as err:
        print(err)
        if sqlconn is not None:
            sqlconn.close()
connection=connect_to_db('example.db')
if connection is not None:
    cursor=connection.cursor()
thp=[('font-size','15px'),('text-align','center'),
     ('font-weight','bold'),('padding','5px 5px'),
     ('color','white'),('background-color','slategray')]
tdp=[('font-size','14px'),('padding','5px 5px'),
     ('text-align','center'),('color','darkblue'),
     ('background-color','silver')]
style_dict=[dict(selector='th',props=thp),
            dict(selector='td',props=tdp)]
os.listdir()


# `map.geojson`

# In[ ]:


df=geopandas.read_file('../input/map.geojson')
fl=['addr:city','addr:country','addr:housenumber','addr:housenumber2',
    'addr:place','addr:postcode','addr:street','addr:street2', 
    'admin_level','amenity','area','artist_name','artwork_type','atm',
    'boundary','building','contact:fax','contact:phone','contact:website',
    'education','footway','government','height','heritage',
    'highway','historic','name','office','official_status','omkmo:code',
    'omkte:code','opening_hours','phone','place','public_transport',
    'religion', 'reservation', 'residential','room','service',
    'short_name','surface','timestamp','uid','user','version']
df[fl].to_sql('map',con=connection,if_exists='replace')
df.shape


# In[ ]:


print(sorted(df.columns)[:180])


# In[ ]:


print(sorted(df.columns)[180:360])


# In[ ]:


users=set(df['user'])
print(sorted(users)[:20])


# In[ ]:


df.boundary.plot(figsize=(12,12),color='gray');


# In[ ]:


print(set(df['highway']))


# In[ ]:


ax=df[df['highway']=='footway'].plot(figsize=(12,7),
                                     color='gray',label='footways')
df[df['highway']=='street_lamp'].plot(ax=ax,color='r',label='lamps')
df[df['highway']=='traffic_signals'].plot(ax=ax,color='g',
                                          label='traffic signals')
plt.legend();


# In[ ]:


print(set(df['building']))


# In[ ]:


ax1=df[df['building']=='office'].plot(figsize=(12,7))
df[df['building']=='university'].plot(ax=ax1,color='r')
df[df['building']=='hotel'].plot(ax=ax1,color='g')
df[df['highway']=='footway'].plot(color='gray',ax=ax1)
blue_patch=mpatches.Patch(label='office')
red_patch=mpatches.Patch(color='r',label='university')
green_patch=mpatches.Patch(color='g',label='hotel')
plt.legend(handles=[blue_patch,red_patch,green_patch],loc=4);


# `moscow_region_admin.geojson`, `moscow_region_roads.geojson`, `moscow_region_waterareas.geojson`

# In[ ]:


df_admin=geopandas.read_file('../input/moscow_region_admin.geojson')
df_roads=geopandas.read_file('../input/moscow_region_roads.geojson')
df_waterareas=geopandas.read_file('../input/moscow_region_waterareas.geojson')
fl=['id','osm_id','name','type','admin_level','geometry']
df_admin.columns=fl
df_admin[fl[:-1]].to_sql('admin',con=connection,if_exists='replace')
df_roads.iloc[:,:-1].to_sql('roads',con=connection,if_exists='replace')
df_waterareas.iloc[:,:-1].to_sql('waterareas',con=connection,if_exists='replace')
df_admin.shape,df_roads.shape,df_waterareas.shape


# In[ ]:


df_roads.head(3).T.style.set_table_styles(style_dict)


# In[ ]:


df_admin.geometry.plot(figsize=(12,12),alpha=.7,
                       edgecolor='slategray',cmap=cm.prism);


# `sql`

# In[ ]:


get_ipython().run_line_magic('get_query', 'SELECT * FROM sqlite_master;')


# In[ ]:


get_ipython().run_line_magic('get_query', 'PRAGMA table_info("waterareas")')


# In[ ]:


get_ipython().run_line_magic('get_query', 'SELECT DISTINCT name FROM roads WHERE type="residential";')


# In[ ]:


if connection is not None:
    connection.close()
if os.path.exists('example.db'):
    os.remove('example.db')
else:
    print('The file does not exist')
os.listdir()


# To be continued...
