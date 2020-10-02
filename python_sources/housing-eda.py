#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from folium import plugins
from IPython.display import display
from scipy import stats

housing = pd.read_csv("../input/usa-housing-listings/housing.csv")
housing.dtypes


# ## Basic data layout

# In[ ]:


pd.set_option('display.max_columns', None)
display(housing.head(n=5))


# ## Draw heatmap
# Lets take a random sample of roughly 10% of the data to visualize where listings are located

# In[ ]:


heatMap = folium.Map([41, -96], zoom_start = 4)
heatMapCleanLoc = housing[np.isfinite(housing['lat'])].sample(n=50000)

heatArr = heatMapCleanLoc[["lat", "long"]].as_matrix()
heatMap.add_child(plugins.HeatMap(heatArr, radius = 15))


# ## Correlation Matrix
# Lets build a simple correlation matrix to check for relationships between variables

# In[ ]:


corrMatrix = housing.corr()
fig, ax = plt.subplots(figsize=(12,10))
sn.heatmap(corrMatrix)


# There doesn't appear to be any usefull correlations, with baths and beds having an (obvious) positive correlation, along with cats_allowed and dogs_allowed having a very strong positive correlation which isn't insightful in the slightest.

# ## Describe a few critical columns
# 

# In[ ]:


housing.price.describe().apply(lambda x: format(x, 'f'))


# In[ ]:


housing.sqfeet.describe().apply(lambda x: format(x, 'f'))


# In[ ]:


housing.beds.describe().apply(lambda x: format(x, 'f'))


# In[ ]:


housing.baths.describe().apply(lambda x: format(x, 'f'))


# ## Remove outliers
# There are clearly some major outliers unless somebody listed a magic house with 1,100 bedrooms and 75 baths. The following code removes any rows with outliers with a price lying beyond $5000, square footage beyond 4000 feet, and beyond 3 standard deviations of the mean for beds and baths.

# In[ ]:


housingReduced = housing[housing.price <= 5000]
housingReduced = housingReduced[housingReduced.sqfeet <= 4000]
housingReduced = housingReduced[np.abs(housingReduced.beds-housing.beds.mean()) <= (3 * housingReduced.beds.std())]
housingReduced = housingReduced[np.abs(housingReduced.baths-housing.baths.mean()) <= (3 * housingReduced.baths.std())]

print(f"{housing.shape[0] - housingReduced.shape[0]} rows removed")


# ## Histograms
# Lets visualize the makeup of price, sqfeet, beds, and baths

# In[ ]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
n, bins, patches = plt.hist(housingReduced.price, bins=90, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)
n = n.astype('int') # it MUST be integer
for i in range(len(patches)):
    patches[i].set_facecolor(plt.cm.viridis(n[i]/max(n)))
plt.title('Price Distribution', fontsize=20)
plt.xlabel('Price per month', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()


# We see that most homes are listed for between \$700 and \$1300

# In[ ]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
n, bins, patches = plt.hist(housingReduced.sqfeet, bins=90, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)
n = n.astype('int') # it MUST be integer
for i in range(len(patches)):
    patches[i].set_facecolor(plt.cm.viridis(n[i]/max(n)))
plt.title('Square Foootage Distribution', fontsize=20)
plt.xlabel('Square Footage', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()


# Square footage has a very similar distribution to price, with most listings being between 700 and 1500 square feet

# In[ ]:


fig = plt.gcf()
fig.set_size_inches(8, 10)
n, bins, patches = plt.hist(housingReduced.beds, bins=90, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)
n = n.astype('int') # it MUST be integer
for i in range(len(patches)):
    patches[i].set_facecolor(plt.cm.viridis(n[i]/max(n)))
plt.title('Bedrooms Distribution', fontsize=20)
plt.xlabel('Number of Bedrooms', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()


# Most listings have 2 bedrooms, followed by 1 and 3 respectively. Very few have more than 4

# In[ ]:


fig = plt.gcf()
fig.set_size_inches(8, 10)
n, bins, patches = plt.hist(housingReduced.baths, bins=90, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)
n = n.astype('int') # it MUST be integer
for i in range(len(patches)):
    patches[i].set_facecolor(plt.cm.viridis(n[i]/max(n)))
plt.title('Baths Distribution', fontsize=20)
plt.xlabel('Number of Baths', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()


# 1 Bath is the most common amount, followed by 2

# ## Comparing price and square footage
# Lets build a simple line graph to see how the monthly rent of a listing correlates with its price

# In[ ]:


fig, ax = plt.subplots(figsize = (16, 16))
grouped = housingReduced.groupby("sqfeet")["price"].mean()
rollingGrouped = grouped.rolling(25).mean()
sqfeetData = rollingGrouped.index.values
priceData = rollingGrouped.values
ax.set_xlabel("square footage", fontsize = 14)
ax.set_ylabel("price", fontsize = 14)
ax.plot(sqfeetData, priceData)


# We see a (predictable) increase in price as square footage increases

# ## Count boolean values
# Lets check how many housing rentals allow smoking, cats, dogs, have EV chargers, come furnished, and have wheelchair access

# In[ ]:


canSmoke = dict(housing.smoking_allowed.value_counts())
print(f"{canSmoke[1]} listings allow smoking ({round((canSmoke[1] / housingReduced.shape[0]) * 100, 2)}%) while {canSmoke[0]} listings do not ({round((canSmoke[0] / housingReduced.shape[0]) * 100, 2)}%)")

haveCats = dict(housing.cats_allowed.value_counts())
print(f"{haveCats[1]} listings allow cats ({round((haveCats[1] / housingReduced.shape[0]) * 100, 2)}%) while {haveCats[0]} listings do not ({round((haveCats[0] / housingReduced.shape[0]) * 100, 2)}%)")

haveDogs = dict(housing.dogs_allowed.value_counts())
print(f"{haveDogs[1]} listings allow dogs ({round((haveDogs[1] / housingReduced.shape[0]) * 100, 2)}%) while {haveDogs[0]} listings do not ({round((haveDogs[0] / housingReduced.shape[0]) * 100, 2)}%)")

evCharge = dict(housing.electric_vehicle_charge.value_counts())
print(f"{evCharge[1]} listings have electric vehicle charging ({round((evCharge[1] / housingReduced.shape[0]) * 100, 2)}%) while {evCharge[0]} listings do not ({round((evCharge[0] / housingReduced.shape[0]) * 100, 2)}%)")

furnished = dict(housing.comes_furnished.value_counts())
print(f"{furnished[1]} listings come furnished ({round((furnished[1] / housingReduced.shape[0]) * 100, 2)}%) while {furnished[0]} listings do not ({round((furnished[0] / housingReduced.shape[0]) * 100, 2)}%)")

wheelchair = dict(housing.wheelchair_access.value_counts())
print(f"{wheelchair[1]} listings have wheelchair access ({round((wheelchair[1] / housingReduced.shape[0]) * 100, 2)}%) while {wheelchair[0]} listings do not ({round((wheelchair[0] / housingReduced.shape[0]) * 100, 2)}%)")


# ## Pie charts for housing type and laundry/parking options
# Lets wrap it up with pie charts depicting the makeup of housing types and laundry/parking options. Percentages in the legends reflect the overall percentage out of all rows in the dataset, they don't always add up to 100% as some entries are null.

# In[ ]:


fig = plt.gcf()
fig.set_size_inches(12, 12)
baseColors = ["red", "orange", "gold", "yellow", "greenyellow", "mediumseagreen", "turquoise", "blue", "darkblue", "mediumpurple", "purple", "crimson"]
types = dict(housingReduced.type.value_counts())
labels = []
sizes = []
colors = []
count = 0
for k, v in types.items():
    labels.append(f"{k} - {round(((v / housingReduced.shape[0]) * 100), 2)}%")
    sizes.append(v)
    colors.append(baseColors[count])
    count += 1

patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=120)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.title("Distribution of housing types", fontsize = 16)
plt.show()


# Apartment is by far the most common housing type, with house and townhouse also making an appearance.

# In[ ]:


fig = plt.gcf()
fig.set_size_inches(12, 12)
baseColors = ["red", "yellow", "mediumseagreen", "blue", "purple"]
laundry = dict(housingReduced.laundry_options.value_counts())
labels = []
sizes = []
colors = []
count = 0
for k, v in laundry.items():
    labels.append(f"{k} - {round(((v / housingReduced.shape[0]) * 100), 2)}%")
    sizes.append(v)
    colors.append(baseColors[count])
    count += 1

patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=120)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.title("Distribution of laundry options", fontsize = 16)
plt.show()


# Laundry options are pretty evenly distributed, with the exception of no laundry on site representing 1% of listings.

# In[ ]:


fig = plt.gcf()
fig.set_size_inches(12, 12)
baseColors = ["red", "orange", "yellow", "mediumseagreen", "blue", "purple"]
parking = dict(housingReduced.parking_options.value_counts())
labels = []
sizes = []
colors = []
count = 0
for k, v in parking.items():
    labels.append(f"{k} - {round(((v / housingReduced.shape[0]) * 100), 2)}%")
    sizes.append(v)
    colors.append(baseColors[len(baseColors) - 1 - count])
    count += 1

patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=120)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.title("Distribution of parking options", fontsize = 16)
plt.show()


# Most housing options offer off-street parking, followed by attached garage and carport.
