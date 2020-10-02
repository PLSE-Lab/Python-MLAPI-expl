#!/usr/bin/env python
# coding: utf-8

# ![](https://i.imgur.com/rTGViJh.png)
# 
# # Exploring Blue Plaques
# 
# "Blue plaques" is a historical term refering to a placemarker. It originated from by far the most famous such scheme, the blue plaque scheme in the United Kingdom (originally London). However, historic markers exist all over the world today. This dataset tabulates many of them. In this notebook we will briefly explore what and where they are, according to the dataset.
# 
# Hopefully after reading this notebook you will have a decent idea of what this dataset is, and how to progress to doing some more interesting things, like looking at the dataset's NLP content, from there!

# In[ ]:


import pandas as pd
plaques = pd.read_csv("../input/blue-plaques/open-plaques-all-2017-06-19.csv", index_col=0)
pd.set_option('max_columns', None)
plaques.head(2)


# ## Place, time, color, meta

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")

f, axarr = plt.subplots(2, 2, figsize=(12, 11))
f.subplots_adjust(hspace=0.75)
plt.suptitle('Plaque Locations, Times Erected, and Colors', fontsize=18)

plaques.country.value_counts().head(10).plot.bar(ax=axarr[0][0])
axarr[0][0].set_title("Plaques per Country (n=10)")

plaques.area.value_counts().head(20).plot.bar(ax=axarr[0][1])
axarr[0][1].set_title("Plaques per City (n=20)")

plaques.erected.value_counts().sort_index().tail(100).plot.line(ax=axarr[1][0])
axarr[1][0].set_title("Plaques erected over Time (t > 1910)")

plaques.colour.value_counts().head(10).plot.bar(ax=axarr[1][1], color='darkgray')
axarr[1][1].set_title("Plaque colors (n=10)")


# The vast majority of the plaques listed in this dataset are in the United States, the United Kingdom, or Germany. The number of plaques listed for other countries is vanishingly small. This points to signficant localization bias in the way this dataset was generated; while plaques in these three countries are indeed numerous, it is hard to believe that they are so overwhelmingly numerous, while similar countries liek France and Canada have barely any!
# 
# We can see that, furthermore, the data is heavily biased towards individual cities. Berlin leads the pack in terms of the number of plaques in this dataset; in fact almost every plaque in Germany in this dataset is in Berlin! London and Paris follow. Many of the rest of the top cities are either British towns or towns in the United States. Overall, this dataset gives me the impression that it was sampled primarily by combining several other easy-to-access datasets on plaque locations.
# 
# I suspect that the datasets for United States, UK, and Berlin plaques are relatively completely, but the data is far more sparse elsewhere. Most of the rest of the data was likely gathered by volunteers.
# 
# Although blue is the most famous color, a plurality of plaques are actually black. Almost all plaques are either black, blue, or brass, with exceptions being very rare.
# 
# Plaque erection has quadrupled since the 1980s, and almost doubled since the year 2000. Clearly plaques are very popular!

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")

f, axarr = plt.subplots(1, 2, figsize=(12, 4))
plaques['geolocated?'].value_counts(dropna=False).sort_index()[::-1].plot.bar(ax=axarr[0])
axarr[0].set_title("geolocated?")

plaques['photographed?'].value_counts(dropna=False).sort_index()[::-1].plot.bar(ax=axarr[1])
axarr[1].set_title("photographed?")
axarr[1].set_ylim([0, 35000])


# Metadata about the plaques is present in the vast majority of cases in terms of geolocation, but while a decent chunk of the plauqes are also linked to photographs, not all are.

# ## Subjects
# 
# Who gets historical plauqes?

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")

f, axarr = plt.subplots(2, 2, figsize=(12, 11))
f.subplots_adjust(hspace=0.75)
# plt.suptitle('Plaque Locations, Times Erected, and Colors', fontsize=18)

plaques.lead_subject_type.value_counts(dropna=False).plot.bar(ax=axarr[0][0])
axarr[0][0].set_title("Subject Kind")

plaques.number_of_subjects.value_counts(dropna=False).sort_index().plot.bar(ax=axarr[0][1])
axarr[0][1].set_title("# of Subjects")

plaques.lead_subject_born_in.plot.hist(ax=axarr[1][0], bins=200)
axarr[1][0].set_title("Lead Subject Born In")
axarr[1][0].set_xlim([1500, 2020])

plaques.lead_subject_died_in.plot.hist(ax=axarr[1][1], bins=200)
axarr[1][1].set_title("Lead Subject Died In")
axarr[1][1].set_xlim([1500, 2020])


# The majority of plaques are actually listed as `nan` in terms of subject kind, but I think we can safely assume that most are men. Men lead the tagged plaques by quite a lot. The handful of plaques dedicated to animls are interesting:

# In[ ]:


pd.set_option('max_colwidth',1000)
plaques.query('lead_subject_type == "animal"')[['inscription', 'country']]


# In[ ]:


pd.reset_option('max_colwidth')


# Peak subject births and deaths are around 1850 and 1950, respectively. We can also take a quick look at subject lifetimes:

# In[ ]:


(plaques.lead_subject_died_in - plaques.lead_subject_born_in).where(lambda v: (v < 100) & (0 < v)).dropna().plot.hist(bins=30)
plt.suptitle('Subject Lifetimes')


# Peak subject age at death is 70 years. A few subjects lived only a very small number of years; these are likely the animals!

# ## Words
# 
# What words show up in the plaque inscriptions?

# In[ ]:


from wordcloud import WordCloud
w = WordCloud(width=800, height=400)
w.generate(" ".join(list(plaques.inscription.values.astype(str))))


# In[ ]:


w.to_image()


# Texas seems to be a big theme, as does school, family, and home. A good chunk of historical markers seem to have to do with the Civil War.

# ## Geographic coverage
# 
# Let's get a clearer picture of where these plaques are, exactly. First of all, let's map out where the points are.

# In[ ]:


latlongs = plaques[['latitude', 'longitude']].dropna()
from shapely.geometry import Point
points = latlongs.apply(lambda srs: Point(srs.longitude, srs.latitude), axis='columns')
import geopandas as gpd
gplaques = gpd.GeoDataFrame(plaques, geometry=points)


# In[ ]:


world_countres = gpd.read_file("../input/countries-shape-files/ne_10m_admin_0_countries.shp")


# In[ ]:


import geoplot as gplt
import geoplot.crs as gcrs

ax = gplt.polyplot(world_countres, projection=gcrs.PlateCarree(), linewidth=0.5, 
                   figsize=(14, 8))
gplt.pointplot(gplaques.loc[pd.notnull(gplaques.geometry)], projection=gcrs.PlateCarree(), 
               edgecolor='black', alpha=1, s=69,
               ax=ax)


# Notice that there are seemingly just 2 markers in all of China, according to this dataset! This data is about as Westernly biased as possible...
# 
# Notice also that there are a number of points which have their y coordinate on the equator. This indicates that these pointsa re incorrectly geospatially coded; there isn't actually a line of plaques in the Atlantic Occean, below the coast of Africa, at all!
# 
# Because there are so many points, this view doesn't quite capture how strongly biased the same. Here is the same data presented in KDE form, which shows just how strongly the data is oriented towards the UK and towards Texas:

# In[ ]:


ax = gplt.polyplot(world_countres, projection=gcrs.PlateCarree(), linewidth=0.5, 
                   figsize=(14, 8))
gplt.kdeplot(gplaques.loc[pd.notnull(gplaques.geometry)], projection=gcrs.PlateCarree(), 
             ax=ax, clipped=True)


# If you're interested in using this data or extending an analysis, I hope this contextual notebook has been helpful. Try forking it to get your own analysis going! Otherwise, if you enjoyed this kernel, be sure to upvote!
# 
# That's all folks!
