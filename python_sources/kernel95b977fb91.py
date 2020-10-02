#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


automobile_data = pd.read_csv("../input/autodata-akhtar/Automobile_data.csv")


# In[ ]:


automobile_data.head()


# In[ ]:


automobile_data.info()


# In[ ]:


automobile_data.replace('?', np.nan, inplace=True)


# In[ ]:


print(automobile_data.isnull().sum())


# In[ ]:


thresh = len(automobile_data) * .1
automobile_data.dropna(thresh = thresh, axis = 1, inplace = True)


# In[ ]:


print(automobile_data.isnull().sum())


# In[ ]:


def impute_median(series):
    return series.fillna(series.median())

#automobile_data['num-of-doors']=automobile_data['num-of-doors'].transform(impute_median) ---String hence ignored
automobile_data.bore=automobile_data['bore'].transform(impute_median)
automobile_data.stroke=automobile_data['stroke'].transform(impute_median)
automobile_data.horsepower=automobile_data['horsepower'].transform(impute_median)
automobile_data.price=automobile_data['price'].transform(impute_median)


# In[ ]:


automobile_data['num-of-doors'].fillna(str(automobile_data['num-of-doors'].mode().values[0]),inplace=True)
automobile_data['peak-rpm'].fillna(str(automobile_data['peak-rpm'].mode().values[0]),inplace=True)
automobile_data['normalized-losses'].fillna(str(automobile_data['normalized-losses'].mode().values[0]),inplace=True)


# In[ ]:


print(automobile_data.isnull().sum())


# In[ ]:


automobile_data.head()


# In[ ]:


automobile_data.make.value_counts().nlargest(10).plot(kind='bar', figsize=(15,5))
plt.title("Top 10 Number of vehicles by make (Akhtar)")
plt.ylabel('Number of vehicles')
plt.xlabel('Make');


# In[ ]:


automobile_data['price']=pd.to_numeric(automobile_data['price'],errors='coerce')
sns.distplot(automobile_data['price']);


# The distribution is highly skewed towards the left which implies there are lesser vehicles that have a very high price range

# In[ ]:


print("Skewness: %f" % automobile_data['price'].skew())
print("Kurtosis: %f" % automobile_data['price'].kurt())


# In[ ]:


plt.figure(figsize=(20,10))
c=automobile_data.corr()
sns.heatmap(c,cmap="BrBG",annot=True)


# Observation : The above heat map produces a correlation plot between variables of the dataframe.

# In[ ]:


sns.lmplot('engine-size', # Horizontal axis
           'price', # Vertical axis
           data=automobile_data, # Data source
           fit_reg=False, # Don't fix a regression line
           hue="make", # Set color
           palette="Paired",
           scatter_kws={"marker": "D", # Set marker style
                        "s": 100}) # S marker size


# Observation : The price and engine-size is positively correlated, so as the size of the engine increases price also increases as illustrated by the scatter plot above.

# In[ ]:


sns.lmplot('city-mpg', # Horizontal axis
           'price', # Vertical axis
           data=automobile_data, # Data source
           fit_reg=False, # Don't fix a regression line
           hue="body-style", # Set color
           palette="Paired",
           scatter_kws={"marker": "D", # Set marker style
                        "s": 100}) # S marker size


# Observation : Price and city-mpg(mileage) is negatively correlated, so as the city-mpg increases price decreases as illustrated by the scatter plot above.

# In[ ]:


sns.boxplot(x="fuel-type", y="price",data = automobile_data)


# Observation : The diesel automobiles have a larger price range as compared to automobiles using gas as a fuel. But there are many outliers amongst gas type vehicles that are highly expensive cars

# In[ ]:


cars = automobile_data = pd.read_csv("../input/cleaneddata-akhtar/cardata_cleaned.csv")


# In[ ]:


plot_color = "#dd0033"
title_color = "#333333"
y_title_margin = 1.0 # The amount of space above titles
left   =  0.10  # the left side of the subplots of the figure
right  =  0.95    # the right side of the subplots of the figure
bottom =  0.1    # the bottom of the subplots of the figure
top    =  0.5    # the top of the subplots of the figure
wspace =  0.1     # the amount of width reserved for blank space between subplots
hspace = 0.6 # the amount of height reserved for white space between subplots

plt.subplots_adjust(
    left    =  left, 
    bottom  =  bottom, 
    right   =  right, 
    top     =  top, 
    wspace  =  wspace, 
    hspace  =  hspace
)
sns.set_style("whitegrid") #set seaborn style template


# In[ ]:


make_hist=sns.countplot(cars['make'], color=plot_color)
make_hist.set_xticklabels(make_hist.get_xticklabels(), rotation=90)
make_hist.set_xlabel('')
make_hist.set_ylabel('Counts', fontsize=14)

ax = make_hist.axes
ax.patch.set_alpha(0)
ax.set_title('Make - Distribution', fontsize=16, color="#333333")
fig = make_hist.get_figure()
fig.figsize=(10,5)
fig.patch.set_alpha(0.5)
fig.savefig('01make_distribution.png',dpi=fig.dpi,bbox_inches='tight')


# Observation : Toyota has the highest number of models, compared to other manufacturers

# In[ ]:


fig, ax = plt.subplots(figsize=(5,5), ncols=1, nrows=1) # get the figure and axes objects for a 3x2 subplot figure

fig.patch.set_alpha(0.5)
ax.set_title("Symboling - Distribution", y = y_title_margin, color=title_color,fontsize=16)

#Set transparency for individual subplots.
ax.patch.set_alpha(0.5)

symbol_hist=sns.countplot(cars["symboling"], color=plot_color, ax=ax )
#symbol_hist.set_xticklabels(symbol_hist.get_xticklabels(), rotation=90,fontsize=12)
symbol_hist.set_ylabel('Count',fontsize=16 )
symbol_hist.set_xlabel('Symboling - Insurance Risk Factor',fontsize=16)

#plt.show()
fig.savefig('02symboling_distribution.png',dpi=fig.dpi,bbox_inches='tight')


# Observation : Above chart shows how the symboling values are distributed across the data set. -3 is least risky, while +3 is most risky vehicle. majority of car models fall on the riskier side

# In[ ]:


plot_color = "#dd0033"
title_color = "#333333"

fig, ax = plt.subplots(figsize=(20,20), ncols=2, nrows=2)

plt.subplots_adjust(
    left    =  left, 
    bottom  =  bottom, 
    right   =  right, 
    top     =  0.6, 
    wspace  =  0.3, 
    hspace  =  0.5
)

fig.patch.set_alpha(0.5)

ax[0][0].set_title('Body Style Distribution', fontsize=14)
ax[0][0].set_alpha(0)

bstyle_dist=sns.countplot(cars['body-style'],color=plot_color, ax=ax[0][0])
bstyle_dist.set_xticklabels(bstyle_dist.get_xticklabels(),rotation=45, fontsize=14)
bstyle_dist.set_xlabel('')
bstyle_dist.set_ylabel('Counts', fontsize=14)

ax[0][1].set_title('Number of Doors Distribution', fontsize=14)
ax[0][1].set_alpha(0)

numdoors_dist=sns.countplot(cars['num-of-doors'],color=plot_color, ax=ax[0][1])
numdoors_dist.set_xlabel('')
numdoors_dist.set_ylabel('Counts', fontsize=14)

ax[1][0].set_title('Drive Wheels Distribution', fontsize=14)
ax[1][0].set_alpha(0)

drvwheels_dist=sns.countplot(cars['drive-wheels'],color=plot_color, ax=ax[1][0])
drvwheels_dist.set_xticklabels(drvwheels_dist.get_xticklabels(),rotation=45, fontsize=14)
drvwheels_dist.set_xlabel('')
drvwheels_dist.set_ylabel('Counts', fontsize=14)

fig.savefig('03categorical_vars_distribution.png',dpi=fig.dpi,bbox_inches='tight')


# In[ ]:


fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 

fig.patch.set_alpha(0.5)
ax[0].set_title("Normalized Losses - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[0].patch.set_alpha(0)

normloss_hist=sns.distplot(cars['normalized-losses'], color=plot_color, ax=ax[0] )
#symbol_hist.set_xticklabels(symbol_hist.get_xticklabels(), rotation=90,fontsize=12)
#symbol_hist.set_ylabel('Count',fontsize=16 )
symbol_hist.set_xlabel('Normalized Losses',fontsize=16)

ax[1].set_title("Normalized Losses - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[1].patch.set_alpha(0)
normloss_hist=sns.violinplot(cars['normalized-losses'], color=plot_color, ax=ax[1] )

#plt.show()
fig.savefig('04normalized_losses_distribution.png',dpi=fig.dpi,bbox_inches='tight')


# In[ ]:


cars['normalized-losses'].describe()


# Observation : The violin plot shows an outlier on the higher value side

# In[ ]:


fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 

fig.patch.set_alpha(0.5)
ax[0].set_title("Wheel Base - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[0].patch.set_alpha(0)

wbase_hist=sns.distplot(cars["wheel-base"], hist=True, color=plot_color, ax=ax[0] )
#symbol_hist.set_xticklabels(symbol_hist.get_xticklabels(), rotation=90,fontsize=12)
#symbol_hist.set_ylabel('Count',fontsize=16 )
wbase_hist.set_xlabel('Wheel Base',fontsize=16)

ax[1].set_title("Wheel Base - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[1].patch.set_alpha(0)
wbase_box=sns.violinplot(cars["wheel-base"], color=plot_color, ax=ax[1] )

#plt.show()
fig.savefig('05wheelbase_distribution.png',dpi=fig.dpi,bbox_inches='tight')


# In[ ]:


cars['wheel-base'].describe()


# Observations : 1.Positively skewed distribution 2.Majority of models have their wheel bases on or around the mean. The However, there is a sharp fall-off in the distribution just a little left to the mean value. This indicates that a wheel base shorter than ~93 is very rare. 3.The violin plot indicates that there are outliers on the higher side

# In[ ]:


fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 

fig.patch.set_alpha(0.5)
ax[0].set_title("Height - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[0].patch.set_alpha(0)

height_hist=sns.distplot(cars["height"], hist=True, color=plot_color, ax=ax[0] )
height_hist.set_xlabel('Height',fontsize=16)

ax[1].set_title("Height - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[1].patch.set_alpha(0)
height_box=sns.violinplot(cars["height"], color=plot_color, ax=ax[1] )

#plt.show()
fig.savefig('06height_distribution.png',dpi=fig.dpi,bbox_inches='tight')


# In[ ]:


cars['height'].describe()


# Observation : Majority car models have their body height around the mean value. However, there is a sharp drop immediately after the mean.This shows there are not many vehicles significantly taller than the median value.

# In[ ]:


fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 

fig.patch.set_alpha(0.5)
ax[0].set_title("Engine Size - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[0].patch.set_alpha(0)

enginesize_hist=sns.distplot(cars["engine-size"], hist=True, color=plot_color, ax=ax[0] )
enginesize_hist.set_xlabel('Engine Size',fontsize=16)

ax[1].set_title("Engine Size - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[1].patch.set_alpha(0)
enginesize_box=sns.violinplot(cars["engine-size"], color=plot_color, ax=ax[1] )

#plt.show()
fig.savefig('07enginesize_distribution.png',dpi=fig.dpi,bbox_inches='tight')


# In[ ]:


print("Mode:" + str(cars["engine-size"].mode()))
print(cars["engine-size"].describe())


# Observation : The distribution is right (positively) skewed. There is a very high number of smaller sized engines. 1500-2000 cc engines are most common, even though, the mean engine size is 2051 cc. There are a few high capacity outlier engines too.

# In[ ]:


print("Mode:" + str(cars["bore"].mode()))
print(cars["bore"].describe())
#cars["bore"].value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 

fig.patch.set_alpha(0.5)
ax[0].set_title("Bore - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[0].patch.set_alpha(0)

bore_hist=sns.distplot(cars["bore"], hist=True, color=plot_color, ax=ax[0] )
#symbol_hist.set_xticklabels(symbol_hist.get_xticklabels(), rotation=90,fontsize=12)
#symbol_hist.set_ylabel('Count',fontsize=16 )
bore_hist.set_xlabel('Bore',fontsize=16)

ax[1].set_title("Bore - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[1].patch.set_alpha(0)
bore_box=sns.violinplot(cars["bore"], color=plot_color, ax=ax[1] )

#plt.show()
fig.savefig('08bore_distribution.png',dpi=fig.dpi,bbox_inches='tight')


# In[ ]:


print("Mode:" + str(cars["bore"].mode()))
print(cars["bore"].describe())
#cars["bore"].value_counts()


# Observation: There are two peaks. Bore values tend to concentrate near 3.19 and 3.62

# In[ ]:


fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 

fig.patch.set_alpha(0.5)
ax[0].set_title("Stroke - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[0].patch.set_alpha(0)

stroke_hist=sns.distplot(cars["stroke"], hist=True, color=plot_color, ax=ax[0] )
stroke_hist.set_xlabel('stroke',fontsize=16)

ax[1].set_title("Stroke - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[1].patch.set_alpha(0)
stroke_box=sns.violinplot(cars["stroke"], color=plot_color, ax=ax[1] )

#plt.show()
fig.savefig('09stroke_distribution.png',dpi=fig.dpi,bbox_inches='tight')


# In[ ]:


print("Mode:" + str(cars["stroke"].mode()[0]))
print("Median:" + str(cars["stroke"].median()))

print(cars["stroke"].describe())
#cars["stroke"].value_counts()


# Observation : Majority of engine stroke values fall between 3.15 and 3.41. Barring a few outliers, the curve falls off sharply beyond the inter quartile range.

# In[ ]:


fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 

fig.patch.set_alpha(0.5)
ax[0].set_title("horsepower - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[0].patch.set_alpha(0)

hp_hist=sns.distplot(cars["horsepower"], hist=True, color=plot_color, ax=ax[0] )
hp_hist.set_xlabel('horsepower',fontsize=16)

ax[1].set_title("horsepower - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[1].patch.set_alpha(0)
hp_box=sns.boxplot(cars["horsepower"], color=plot_color, ax=ax[1] )

#plt.show()
fig.savefig('10horsepower_distribution.png',dpi=fig.dpi,bbox_inches='tight')


# In[ ]:


print("Mode:" + str(cars["horsepower"].mode()[0]))
print("Median:" + str(cars["horsepower"].median()))

print(cars["horsepower"].describe())
#cars["stroke"].value_counts()


# Observation :Horsepower shows a positively skewed distribution. Majority of engines tend to have a lower power output within the power range of the dataset.

# In[ ]:


fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 

fig.patch.set_alpha(0.5)
ax[0].set_title("Fuel Efficiency - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[0].patch.set_alpha(0)

citympg_hist=sns.distplot(cars["city-mpg"], hist=True, color=plot_color, ax=ax[0] )
citympg_hist.set_xlabel('Fuel Efficiency(City)',fontsize=16)

ax[1].set_title("Fuel Efficiency - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[1].patch.set_alpha(0)
citympg_box=sns.violinplot(cars["city-mpg"], color=plot_color, ax=ax[1] )

#plt.show()
fig.savefig('11citympg_distribution.png',dpi=fig.dpi,bbox_inches='tight')


# In[ ]:


print("Mode:" + str(cars["city-mpg"].mode()[0]))
print("Median:" + str(cars["city-mpg"].median()))

print(cars["city-mpg"].describe())
#cars["city-mpg"].value_counts()


# Observation : The FE figures range from 14 mpg and upwards. At 31 mpg, which is what majority cars output, there is a sharp fall, indicating a practical limit on how high the FE figures can go. Although there are a few outliers - nearing 49 mpg.

# In[ ]:


fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 

fig.patch.set_alpha(0.5)
ax[0].set_title("Price - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[0].patch.set_alpha(0)

price_hist=sns.distplot(cars["price"], hist=True, color=plot_color, ax=ax[0] )
price_hist.set_xlabel('Price',fontsize=16)

ax[1].set_title("Price - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[1].patch.set_alpha(0)
normloss_hist=sns.violinplot(cars["price"], color=plot_color, ax=ax[1] )

#plt.show()
fig.savefig('12price_distribution.png',dpi=fig.dpi,bbox_inches='tight')


# In[ ]:


print("Mode:" + str(cars["price"].mode()[0]))
print("Median:" + str(cars["price"].median()))

print(cars["price"].describe())
#cars["price"].value_counts()


# Observation : Price shows a positively skewed distriution. Most common car prices are below 10K. There are outliers having prices upwards of 30K

# In[ ]:


ncyl_hist=sns.countplot(cars['num_cylinders'], color=plot_color)
ncyl_hist.set_xlabel('Cylinders')
ncyl_hist.set_ylabel('Counts', fontsize=14)

ax = ncyl_hist.axes
ax.patch.set_alpha(0)
ax.set_title('Number of cylinders - Distribution', fontsize=16, color="#333333")
fig = ncyl_hist.get_figure()
fig.figsize=(10,5)
fig.patch.set_alpha(0.5)
fig.savefig('13numcylinders_distribution.png',dpi=fig.dpi,bbox_inches='tight')


# In[ ]:


print("Mode:" + str(cars["num_cylinders"].mode()[0]))
print("Median:" + str(cars["num_cylinders"].median()))

print(cars["num_cylinders"].describe())
#cars["num_cylinders"].value_counts()


# Observation :4 cylinder engines are most common for cars

# In[ ]:


fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1) 

fig.patch.set_alpha(0.5)
ax[0].set_title("Curb Weight - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[0].patch.set_alpha(0)

cweight_hist=sns.distplot(cars["curb-weight"], hist=True, color=plot_color, ax=ax[0] )
cweight_hist.set_xlabel('Curb Weight',fontsize=16)

ax[1].set_title("Curb Weight - Distribution", y = y_title_margin, color=title_color,fontsize=16)
ax[1].patch.set_alpha(0)
cweight_box=sns.violinplot(cars["curb-weight"], color=plot_color, ax=ax[1] )
cweight_box.set_xlabel('Curb Weight',fontsize=16)

#plt.show()
fig.savefig('12curbweight_distribution.png',dpi=fig.dpi,bbox_inches='tight')


# In[ ]:


print("Mode:" + str(cars["curb-weight"].mode()[0]))
print("Median:" + str(cars["curb-weight"].median()))

print(cars["curb-weight"].describe())
#cars["curb-weight"].value_counts()


# 
