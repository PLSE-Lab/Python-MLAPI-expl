#!/usr/bin/env python
# coding: utf-8

# #This notebook explores the forest cover dataset.
# 
# **There are seven tree types, each represented by an integer variable:**
# 
#  1. Spruce/Fir
#  2. Lodgepole Pine
#  3. Ponderosa Pine
#  4. Cottonwood/Willow
#  5. Aspen
#  6. Douglas-fir
#  7. Krummholz
# 
# **Remaining data fields include:**
# 
#  - Elevation: Elevation in meters
#  - Aspect: Aspect in degrees azimuth
#  - Slope: Slope in degrees
#  - Horizontal Distance To Hydrology: Horz Dist to nearest surface water
#    features
#  - Vertical Distance To Hydrology: Vert Dist to nearest surface water
#    features
#  - Horizontal Distance To Roadways: Horz Dist to nearest roadway
#  - Hillshade 9am (0 to 255 index): Hillshade index at 9am, summer
#    solstice
#  - Hillshade Noon (0 to 255 index): Hillshade index at noon, summer
#    solstice
#  - Hillshade 3pm (0 to 255 index): Hillshade index at 3pm, summer
#    solstice
#  - Horizontal Distance To Fire Points: Horz Dist to nearest wildfire
#    ignition points
#  - Wilderness Area (4 binary columns, 0 = absence or 1 = presence):
#    Wilderness area designation
#  - Soil Type (40 binary columns, 0 = absence or 1 = presence): Soil Type
#    designation
#  - Cover Type (7 types, integers 1 to 7): Forest Cover Type designation

# In[ ]:


# import the needed libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/covtype.csv')


# In[ ]:


df.describe()


# In[ ]:


df.columns


# #Wilderness Areas
# Let's start off by sorting the data by cover type, specifically looking at those 4 wilderness areas. If the tree was observed in a particular area, it got a "1" in that column. So, if the mean, max and min are all zeroes for a cover type, that tells us that cover type was not observed in that area. 
# 
# **Look at the stats below for the Wilderness Areas:**

# In[ ]:


wild1 = df['Wilderness_Area1'].groupby(df['Cover_Type'])
totals = []
for value in wild1.sum():
    totals.append(value)
print(totals)
total_sum = sum(totals)
print("Total Trees in Area: %d" % total_sum)
percentages = [ (total*100 / total_sum) for total in totals]
print(percentages)


# **Types 3, 4, and 6 do not grow in Wilderness Area 1.**  Lodgepole Pine and Spruce/Fir make over 96% of the cover types observed.

# In[ ]:


trees = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow',
         'Aspen','Douglas-fir', 'Krummholz']
xs = [i + 0.1 for i, _ in enumerate(trees)]
plt.bar(xs, totals)
plt.ylabel("# of Trees")
plt.title("Cover Types in Wilderness Area 1")
plt.xticks([i + 0.5 for i, _ in enumerate(trees)], trees, rotation='vertical');


# In[ ]:


wild2 = df['Wilderness_Area2'].groupby(df['Cover_Type'])
totals = []
for value in wild2.sum():
    totals.append(value)
print(totals)
total_sum = sum(totals)
print("Total Trees in Area: %d" % total_sum)
percentages = [ (total*100 / total_sum) for total in totals]
print(percentages)


# **Types 3, 4, 5, and 6 do not grow in Wilderness Area 2.** Spruce/Fir, Lodgepole Pine, and Krummholz make up all of the observations.

# In[ ]:


trees = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow',
         'Aspen','Douglas-fir', 'Krummholz']
xs = [i + 0.1 for i, _ in enumerate(trees)]
plt.bar(xs, totals)
plt.ylabel("# of Trees")
plt.title("Cover Types in Wilderness Area 2")
plt.xticks([i + 0.5 for i, _ in enumerate(trees)], trees, rotation='vertical');


# In[ ]:


wild3 = df['Wilderness_Area3'].groupby(df['Cover_Type'])
totals = []
for value in wild3.sum():
    totals.append(value)
print(totals)
total_sum = sum(totals)
print("Total Trees in Area: %d" % total_sum)
percentages = [ (total*100 / total_sum) for total in totals]
print(percentages)


# **Type 4 does not grow in Wilderness Area 3.** Spruce/Fir and Lodgepole Pine make up 85% of the observed cover types.

# In[ ]:


trees = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow',
         'Aspen','Douglas-fir', 'Krummholz']
xs = [i + 0.1 for i, _ in enumerate(trees)]
plt.bar(xs, totals)
plt.ylabel("# of Trees")
plt.title("Cover Types in Wilderness Area 3")
plt.xticks([i + 0.5 for i, _ in enumerate(trees)], trees, rotation='vertical');


# In[ ]:


wild4 = df['Wilderness_Area4'].groupby(df['Cover_Type'])
totals = []
for value in wild4.sum():
    totals.append(value)
print(totals)
total_sum = sum(totals)
print("Total Trees in Area: %d" % total_sum)
percentages = [ (total*100 / total_sum) for total in totals]
print(percentages)


# **Types 1, 5, and 7 do not grow in Wilderness Area 4.**  Cottonwood/Willow and Douglas Fir make up almost 85% of observed cover types here. 
# 
# **This area stands out from the others rather noticeably.** The other three wilderness areas are heavy on the Spruce/Fir and Lodgepole Pine, while this area is not. Conversely, area four is heavy on Ponderosa Pine, Cottonwood/Willow, and Douglas Fir, while the other three areas are not.

# In[ ]:


trees = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow',
         'Aspen','Douglas-fir', 'Krummholz']
xs = [i + 0.1 for i, _ in enumerate(trees)]
plt.bar(xs, totals)
plt.ylabel("# of Trees")
plt.title("Cover Types in Wilderness Area 4")
plt.xticks([i + 0.5 for i, _ in enumerate(trees)], trees, rotation='vertical');


# #Total Cover Types Observed
# 
# Now that we know how those 4 areas shake out, let's step back and look at the total makeup of the Roosevelt National Forest.

# In[ ]:


# Group elevation data by cover type
coverSets = df['Wilderness_Area1'].groupby(df['Cover_Type'])


# In[ ]:


totals = []
counter = 0
for total in coverSets.count():
    #counter = counter + 1
    totals.append(total)
totals_zip = zip(totals, [1, 2, 3, 4, 5, 6, 7])
print(list(totals_zip))


# With the counts for each type of cover, let's see how they compare over all the wilderness areas.

# In[ ]:


total_trees = sum(totals)
percentages = [(total*100 / total_trees) for total in totals]
print(percentages)


# In[ ]:


trees = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow',
         'Aspen','Douglas-fir', 'Krummholz']
xs = [i + 0.1 for i, _ in enumerate(trees)]
plt.bar(xs, totals)
plt.ylabel("# of Trees")
plt.title("Tree types in Roosevelt National Forest, Colorado")
plt.xticks([i + 0.5 for i, _ in enumerate(trees)], trees, rotation='vertical');


# #Elevation
# 
# Elevation and horizontal distance to hydrology might be good factors to explore, because we expect there to be differences across the species. Different strokes for different folks.
# 
# First, we'll graph box plots of the elevation for each cover type:

# In[ ]:


#print(elevations.describe())
spruce = df[df.Cover_Type == 1]
lodgepole = df[df.Cover_Type == 2]
ponderosa = df[df.Cover_Type == 3]
willow = df[df.Cover_Type == 4]
aspen = df[df.Cover_Type == 5]
douglas = df[df.Cover_Type == 6]
krummholz = df[df.Cover_Type == 7]
plt.figure()
plt.title('Elevation of Cover Types')
plt.ylabel('Elevation (in meters)')
data = [spruce.Elevation, lodgepole.Elevation, ponderosa.Elevation, willow.Elevation,
aspen.Elevation, douglas.Elevation, krummholz.Elevation]
plt.xticks([1, 2, 3, 4, 5, 6, 7])
plt.boxplot(data)
plt.show() 


# Based on elevation, only the Cottonwood/Willow and the Krumholz cover types do not overlap. Interestingly, the cottonwood/willows and the aspens have relatively narrow elevation ranges compared to the other five cover types.

# #Horizontal Distance to Hydrology
# Now we'll graph the box plots for the cover types based on horizontal distance to hydrology:

# In[ ]:


plt.figure()
plt.title('Horizontal_Distance_To_Hydrology of Cover Types')
plt.ylabel('Distance (in meters)')
data = [spruce.Horizontal_Distance_To_Hydrology, lodgepole.Horizontal_Distance_To_Hydrology,
        ponderosa.Horizontal_Distance_To_Hydrology, willow.Horizontal_Distance_To_Hydrology,
        aspen.Horizontal_Distance_To_Hydrology, douglas.Horizontal_Distance_To_Hydrology, 
        krummholz.Horizontal_Distance_To_Hydrology]
plt.xticks([1, 2, 3, 4, 5, 6, 7])
plt.boxplot(data)
plt.show() 


# All the tree types love their water. The order is a little different from the elevation order. The pines that grow higher up can handle a bit more horizontal distance from the water. The maxima emphasize the distinction.

# #Vertical Distance to hydrology

# In[ ]:


plt.figure()
plt.title('Vertical_Distance_To_Hydrology of Cover Types')
plt.ylabel('Distance (in meters)')
data = [spruce.Vertical_Distance_To_Hydrology, lodgepole.Vertical_Distance_To_Hydrology,
        ponderosa.Vertical_Distance_To_Hydrology, willow.Vertical_Distance_To_Hydrology,
        aspen.Vertical_Distance_To_Hydrology, douglas.Vertical_Distance_To_Hydrology, 
        krummholz.Vertical_Distance_To_Hydrology]
plt.xticks([1, 2, 3, 4, 5, 6, 7])
plt.boxplot(data)
plt.show() 


# Meh.

# #Slope

# In[ ]:


plt.figure()
plt.title('Slope of Cover Types')
plt.ylabel('Angle (in degrees)')
data = [spruce.Slope, lodgepole.Slope,
        ponderosa.Slope, willow.Slope,
        aspen.Slope, douglas.Slope, 
        krummholz.Slope]
plt.xticks([1, 2, 3, 4, 5, 6, 7])
plt.boxplot(data)
plt.show() 


# #Horizontal Distance to Roadways

# In[ ]:


plt.figure()
plt.title('Horizontal Distance to Roadways')
plt.ylabel('Distance (in meters)')
roadway_data = [spruce.Horizontal_Distance_To_Roadways, lodgepole.Horizontal_Distance_To_Roadways,
        ponderosa.Horizontal_Distance_To_Roadways, willow.Horizontal_Distance_To_Roadways,
        aspen.Horizontal_Distance_To_Roadways, douglas.Horizontal_Distance_To_Roadways, 
        krummholz.Horizontal_Distance_To_Roadways]
plt.xticks([1, 2, 3, 4, 5, 6, 7])
plt.boxplot(roadway_data)
plt.show() 


# #Horizontal Distance to Fire Points

# In[ ]:


plt.figure()
plt.title('Horizontal Distance to Fire Points')
plt.ylabel('Distance (in meters)')
firepoint_data = [spruce.Horizontal_Distance_To_Fire_Points, lodgepole.Horizontal_Distance_To_Fire_Points,
        ponderosa.Horizontal_Distance_To_Fire_Points, willow.Horizontal_Distance_To_Fire_Points,
        aspen.Horizontal_Distance_To_Fire_Points, douglas.Horizontal_Distance_To_Fire_Points, 
        krummholz.Horizontal_Distance_To_Fire_Points]
plt.xticks([1, 2, 3, 4, 5, 6, 7])
plt.boxplot(firepoint_data)
plt.show()


# **The roadways and fire points data look similar, but a straight-up correlation is rather weak: 0.33158**

# In[ ]:


road_fire = df[['Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']]
road_fire.corr()


# But what if we only correlated the medians? Those look like they would give us a higher r-value. i.e., emphasize the correlation. Let's plot them and throw a best-fit line in there.

# In[ ]:


x = [spruce.Horizontal_Distance_To_Roadways.median(), lodgepole.Horizontal_Distance_To_Roadways.median(),
        ponderosa.Horizontal_Distance_To_Roadways.median(), willow.Horizontal_Distance_To_Roadways.median(),
        aspen.Horizontal_Distance_To_Roadways.median(), douglas.Horizontal_Distance_To_Roadways.median(), 
        krummholz.Horizontal_Distance_To_Roadways.median()]
y = [spruce.Horizontal_Distance_To_Fire_Points.median(), lodgepole.Horizontal_Distance_To_Fire_Points.median(),
        ponderosa.Horizontal_Distance_To_Fire_Points.median(), willow.Horizontal_Distance_To_Fire_Points.median(),
        aspen.Horizontal_Distance_To_Fire_Points.median(), douglas.Horizontal_Distance_To_Fire_Points.median(), 
        krummholz.Horizontal_Distance_To_Fire_Points.median()]
print(x)
print(y)
plt.figure()
plt.title("Roadway Dist Medians vs. Fire Point Dist Medians")
plt.xlabel("Distance to Roadways (in meters)")
plt.ylabel("Distance to Fire Points (in meters)")
plt.scatter(x, y)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.show()


# **Focusing on the medians cranks up the correlation to 0.94777324:**

# In[ ]:


print(np.corrcoef(x,y))


# But what does that mean? That fires typically start near the roads? "Man is in the forest"? Somebody help me out here. 

# #Soil Type

# In[ ]:


soil_counts = []
for num in range(1,41):
    col = ('Soil_Type' + str(num))
    this_soil = df[col].groupby(df['Cover_Type'])
    totals = []
    for value in this_soil.sum():
        totals.append(value)
    total_sum = sum(totals)
    soil_counts.append(total_sum)
    print("Total Trees in Soil Type {0}: {1}".format(num, total_sum))
    percentages = [ (total*100 / total_sum) for total in totals]
    print("{0}\n".format(percentages))
print("Number of trees in each soil type:\n{0}".format(soil_counts))
        


# In[ ]:


soil_types = range(1,41)
xs = [i + 0.2 for i, _ in enumerate(soil_types)]
plt.bar(xs, soil_counts)
plt.ylabel("# of Trees")
plt.title("Soil Types in Roosevelt National Forest, Colorado")
plt.xticks([i + 0.75 for i, _ in enumerate(soil_types)], soil_types, rotation='vertical');


# In[ ]:




soil_counts = []
for num in range(1,41):
    col = ('Soil_Type' + str(num))
    this_soil = df[col].groupby(df['Cover_Type'])
    totals = []
    for value in this_soil.sum():
        totals.append(value)
    total_sum = sum(totals)
    soil_counts.append(total_sum)
    print("Total Trees in Soil Type {0}: {1}".format(num, total_sum))
    percentages = [ (total*100 / total_sum) for total in totals]
    print("{0}\n".format(percentages))
print("Number of trees in each soil type:\n{0}".format(soil_counts))
        

