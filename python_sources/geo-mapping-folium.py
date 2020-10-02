#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import folium
import codecs
with codecs.open("../input/police-killings/police_killings.csv", "r", "Shift-JIS", "ignore") as fp:
    df = pd.read_csv(fp)
mapobj = folium.Map(location=[32.529577, -86.362829], zoom_start=4)


# In[ ]:


colord = dict()
colord["Gunshot"] = "blue"
colord["Taser"] = "red"
colord["Death in custody"] = "green"
colord["Struck by vehicle"] = "black"

for i, row in df.iterrows():
    folium.Marker(location=[row["latitude"], row["longitude"]],
                  popup=row["cause"],
                  icon=folium.Icon(
                      icon="info-sign",
                      color=colord.get(row["cause"], "gray")
                  )).add_to(mapobj)


# ```
# Blue: Gunshot
# Red: Taser
# Green: Death in custody
# Black: Struck by vehicle
# Gray: Other
# ```

# In[ ]:


mapobj


# In[ ]:




