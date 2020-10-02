#!/usr/bin/env python
# coding: utf-8

# # Implementation Of The Example Workflow
# 
# To get used to the data of this challenge, and to catch a view of where this challenge will challenge us, I came up with a simple implementation of the example workflow proposed by Chris from Kaggle [here][1].
# 
# May be a bit late for the party, but, if anyone has suggestions or concerns, I'd be more than happy to listen.
# 
# [1]: https://www.kaggle.com/center-for-policing-equity/data-science-for-good/discussion/67450

# In[ ]:


# imports

import fiona
import matplotlib.pyplot as plt
import pandas as pd
from geopandas.plotting import plot_polygon_collection
from shapely.geometry import shape


# In[ ]:


# tools

# EEPSG:102003 USA_Contiguous_Albers_Equal_Area_Conic
equal_area_proj = ('+proj=aea'
                   ' +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96'
                   ' +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs ')

def project(geom, p1, p2=equal_area_proj):
    """Convert geom from p1 to p2
    
    Parameters
    ----------
    geom: shapely geometry object
    p1: str or dict
        Parameters for the original projection
    p2: str or dict
        Parameters for the desired projection
        
    Returns
    -------
    shapely geometry object
        An object equivalent to geom, but
        projected into p2 instead
    """
    import pyproj
    from functools import partial    
    from shapely.ops import transform
    
    p1 = pyproj.Proj(p1, preserve_units=True)
    p2 = pyproj.Proj(p2, preserve_units=True)
    project = partial(pyproj.transform, p1, p2)
    transformed = transform(project, geom)
    return transformed


# # Police District
# 
# First, we need to retrieve the geometry for the police district.
# 
# Since we have no data for Providence, let's use Boston (arbitrarily) instead.
# 
# ![](https://upload.wikimedia.org/wikipedia/en/b/b7/Boston_Police_patch.jpg)

# In[ ]:


get_ipython().system('ls ../input/data-science-for-good/')


# In[ ]:


# retrieve district geometry and properties

loc = '../input/data-science-for-good/cpe-data/Dept_11-00091/11-00091_Shapefiles'
with fiona.open(loc) as c:
    rec = next(iter(c))  # choose first district of the list
    crs = c.crs  # coordinate reference system


# In[ ]:


rec['properties']


# We chose district 14. Let's plot its shape using shapely.

# In[ ]:


d14_shape = shape(rec['geometry'])
d14_shape = project(d14_shape, crs)  # project into equal-area
d14_shape


# Nice...
# 
# Now that we have the shape for the district, let's focus on the census tracts.

# # Census tracts
# 
# The census tracts (from the ACS) were retrieved in [this location][1]. It is a simplified version, so and we will probably want to use the [TIGER][2] later.
# 
# [1]: https://www.census.gov/geo/maps-data/data/tiger-cart-boundary.html
# [2]: https://www.census.gov/geo/maps-data/data/tiger.html

# In[ ]:


# retrieve census tracts shapes and properties

loc = '../input/01-example-workflow/ma_simplified'
with fiona.open(loc) as c:
    records = list(c)  # retrieve all tracts from shapefile
    crs = c.crs  # coordinate reference system
print(f"{len(records)} census tracts available")


# In[ ]:


# set shapely ``shape`` in each record

for record in records:
    record['shape'] = shape(record['geometry'])
    # project into equal-area
    record['shape'] = project(record['shape'], crs)


# In[ ]:


# plot census tracts

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

polygons = [r['shape'] for r in records]
plot_polygon_collection(ax, polygons);


# # Combining the shapes
# 
# Now let's combine the shapes from the tracts with the shape from the district. We want to know what percentage of each census tract is part of the specified district.
# 
# To store this data, we will use a vanilla dictionary.

# In[ ]:


percentages = {}

for record in records:
    id = record['properties']['AFFGEOID']
    intersection = record['shape'].intersection(d14_shape)
    percentage = intersection.area / record['shape'].area
    percentages[id] = percentage

list(percentages.items())[:5]


# In[ ]:


# plot the intercepting census tracts

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

tracts = [r['shape'] for r in records
          if percentages[r['properties']['AFFGEOID']] > 0]
plot_polygon_collection(ax, tracts, edgecolor='white', alpha=0.5)
plot_polygon_collection(ax, [d14_shape], color='red', alpha=0.5)


# # Final steps
# 
# Now we calculate the total amount of people and the total amount of black people in the district.
# 
# We start by loading the census data:

# In[ ]:


loc = ('../input/data-science-for-good/cpe-data/Dept_11-00091/11-00091_ACS_data'
       '/11-00091_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv')
df = pd.read_csv(loc, skiprows=[1])  # there are 2 header rows
df = df.set_index('GEO.id')  # prepare for joining
df.head()


# Then, we estimate the population contribution from each census tract (both for black and total people):

# In[ ]:


###
# Estimate; RACE - Race alone or in combination with one or more other races - Total population - Black or African American
black_varname = 'HC01_VC79'

# Estimate; RACE - Race alone or in combination with one or more other races - Total population
total_varname = 'HC01_VC77'
###

# join percentage values
pct_series = pd.Series(percentages)
pct_series.name = 'percentage'
small = df[[black_varname, total_varname]].join(pct_series, how='left')

# estimate populations inside police district
small['black_pop'] = small[black_varname] * small['percentage']
small['total_pop'] = small[total_varname] * small['percentage']

small.head()


# Summing the contributions, we arrive at estimates of the amount of people:

# In[ ]:


black_pop = small['black_pop'].sum()
total_pop = small['total_pop'].sum()


# In[ ]:


print(f"Estimated black population: {black_pop:.1f}")
print(f"Estimated total population: {total_pop:.1f}")
print(f"Estimated percentage: {black_pop / total_pop:.1%}")

