#!/usr/bin/env python
# coding: utf-8

# This study explores the spatial trends in interest level across NYC neighbourhoods.

#     %matplotlib inline
#     import numpy as np
#     import json
#     import pandas as pd
#     import geopandas
#     import geocoder
#     from geopandas.tools import sjoin
#     from shapely.geometry import Point
#     import seaborn as sns
#     color = sns.color_palette()
#     import matplotlib.pyplot as plt

#     dat = pd.read_json('train.json')

# Missing values found in latitude/ longitude columns are replaced with geocoded values obtained from their street address.

#     missingCoords = dat[(dat.longitude == 0) | (dat.latitude == 0)]
#     missingGeoms = (missingCoords.street_address + ' New York').apply(geocoder.google)

#     dat.loc[(dat.longitude == 0) | (dat.latitude == 0), 'latitude'] = missingGeoms.apply(lambda x: x.lat)
#     dat.loc[(dat.longitude == 0) | (dat.latitude == 0), 'longitude'] = missingGeoms.apply(lambda x: x.lng)

# Latitude and longitude columns are used to create the geometry column.

#     dat['geometry'] = dat.apply(lambda x: Point((float(x.longitude), float(x.latitude))), axis=1)

# Point geometries created in the step above are then compared with a polygon shape file describing NYC neighbourhood areas to fill neighbourhood information. The observations are then grouped by interest level and neighbourhood area, and aggregated. The maximum of low, medium and high interest level counts is used to fill the column 'type', which is regarded as the sentiment expressed by the corresponding neighbourhood area.

#     poly = geopandas.GeoDataFrame.from_file('nynta_1/nynta_1.shp')
#     gdat = geopandas.GeoDataFrame(dat, crs = poly.crs, geometry='geometry')
#     pointInPolys = sjoin(poly, gdat, how='left', op='contains')
#     lowPolys = pointInPolys[pointInPolys['interest_level'] == 'low']
#     mediumPolys = pointInPolys[pointInPolys['interest_level'] == 'medium']
#     highPolys = pointInPolys[pointInPolys['interest_level'] == 'high']

#     lows = poly.merge(pd.DataFrame({'locount' : lowPolys.groupby('ntacode').size()}).reset_index(), on="ntacode", how="left").fillna(0)

#     mediums = poly.merge(pd.DataFrame({'medcount' : mediumPolys.groupby('ntacode').size()}).reset_index(), on="ntacode", how="left").fillna(0)

#     highs = poly.merge(pd.DataFrame({'hicount' : highPolys.groupby('ntacode').size()}).reset_index(), on="ntacode", how="left").fillna(0)

#     lows.plot(column='locount', scheme='QUANTILES', k=5, cmap='OrRd', legend=True, figsize=(8, 6))

# ![enter image description here][1]
# 
# 
#   [1]: http://i68.tinypic.com/2wmmkk4.jpg

#     mediums.plot(column='medcount', scheme='QUANTILES', k=5, cmap='OrRd', legend=True, figsize=(8, 6))

# ![enter image description here][1]
# 
# 
#   [1]: http://i66.tinypic.com/33c6oft.jpg

#     all = lows.merge(pd.DataFrame({'medcount' : mediumPolys.groupby('ntacode').size()}).reset_index(), on="ntacode", how="left").merge(pd.DataFrame({'hicount' : highPolys.groupby('ntacode').size()}).reset_index(), on="ntacode", how="left").fillna(0)
# 

#     all.plot(column='type', cmap='RdYlGn', legend=True, figsize=(8, 6))

# ![enter image description here][1]
# 
# 
#   [1]: http://i63.tinypic.com/20rl56g.png

# ## Findings ##
# It can be seen that Manhattan and Brooklyn apartments's common interest level is low. Bronx, Staten Island and Queens have medium and high interest level apartments, although they have fewer property listings compared to Manhattan and Brooklyn. Since Manhattan and Brooklyn have more listings than other boroughs, neighborhood wise box plots on price per bedroom will help reveal the spread in each of them.

#     ulimit = np.percentile(pointInPolys.pricePerBR.values, 99)
#     pointInPolys['pricePerBR'].ix[pointInPolys['pricePerBR']>ulimit] = ulimit
#     
#     plt.figure(figsize=(8,4))
#     sns.violinplot(x='interest_level', y='pricePerBR', data=pointInPolys[pointInPolys.boroname=='Manhattan'], order =['low','medium','high'])
#     plt.xlabel('Interest Level', fontsize=12)
#     plt.ylabel('Price per BR', fontsize=12)
#     plt.title('Manhattan')
#     plt.show()

# ![enter image description here][1]
# 
# 
#   [1]: http://i64.tinypic.com/1z4agdz.png

#     plt.figure(figsize=(8,4))
#     sns.violinplot(x='interest_level', y='pricePerBR', data=pointInPolys[pointInPolys.boroname=='Brooklyn'], order =['low','medium','high'])
#     plt.xlabel('Interest Level', fontsize=12)
#     plt.ylabel('Price per BR', fontsize=12)
#     plt.title('Brooklyn')
#     plt.show()
#     
#     plt.figure(figsize=(8,4))
#     sns.violinplot(x='interest_level', y='pricePerBR', data=pointInPolys[pointInPolys.boroname=='Queens'], order =['low','medium','high'])
#     plt.xlabel('Interest Level', fontsize=12)
#     plt.ylabel('Price per BR', fontsize=12)
#     plt.title('Queens')
#     plt.show()
#     
#     plt.figure(figsize=(8,4))
#     sns.violinplot(x='interest_level', y='pricePerBR', data=pointInPolys[pointInPolys.boroname=='Bronx'], order =['low','medium','high'])
#     plt.xlabel('Interest Level', fontsize=12)
#     plt.ylabel('Price per BR', fontsize=12)
#     plt.title('Bronx')
#     plt.show()
#     
#     plt.figure(figsize=(8,4))
#     sns.violinplot(x='interest_level', y='pricePerBR', data=pointInPolys[pointInPolys.boroname=='Staten Island'], order =['low','medium','high'])
#     plt.xlabel('Interest Level', fontsize=12)
#     plt.ylabel('Price per BR', fontsize=12)
#     plt.title('Staten Island')
#     plt.show()

# ![enter image description here][1]
# 
# 
#   [1]: http://i63.tinypic.com/1zz6y3o.png

# ![enter image description here][1]
# 
# 
#   [1]: http://i66.tinypic.com/9u6yc6.png

# ![enter image description here][1]
# 
# 
#   [1]: http://i67.tinypic.com/nearev.png

# ![enter image description here][1]
# 
# 
#   [1]: http://i63.tinypic.com/33eiu0g.png

# The plots reveal a decreasing price per bedroom trend in the direction from Manhattan to Staten Island, and gap between price levels across the interest levels too become less pronounced.
