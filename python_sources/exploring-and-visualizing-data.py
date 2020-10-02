#!/usr/bin/env python
# coding: utf-8

# # Exploring and Visualizing Data

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# Store start time to check script's runtime
scriptStartTime = time.time()

# Read file
df = pd.read_csv("../input/train.csv")

df["Dates"] = pd.to_datetime(df["Dates"])


# Print the columns to get an idea what kind of data are stored in the dataset.

# In[ ]:


# What are the columns in this dataset
print(df.columns)


# To get an idea of the categories' diversity, print the unique names that appear in 'Category'.

# In[ ]:


# Let's see what the Categories are
print(df["Category"].unique())


# In[ ]:


# Check amount of unique values
print("Uniques:")
for column in df.columns:
    print("Unique in '" + column + "': " + str(df[column].nunique()))


# Interesting information: There are 39 crime categories. So we have a rather fine grouping of crimes.
# There are 879 unique descripts, compared to 389257 dates that means there are a lot of duplicate descripts.
# There are 10 different pd districts. A number low enough that it might help to group data for visualiztion.

# ## Plotting
# Visualize the count for each Category. That helps to see which categories are more relevant or contain a lot of samples.

# In[ ]:


# Amount of crimes per category
groups = df.groupby("Category")["Category"].count()
groups = groups.sort_values(ascending=0)
plt.figure()
groups.plot(kind='bar', title="Category Count")
print(groups)


# We can see, that the counts for the categories have a large variety. Up to a factor of >1000. If we want to compare standard deviation or variance later on, we should use a ratio relative to the overall count or mean for each category to get comparable results. 
# The largest category by far is LARCENY/THEFT. Let's investigate it further by plotting the crimes occurence for each weekday.

# In[ ]:


# Largest category is LARCENY/THEFT, let's investigate it further
dfTheft = df[df["Category"] == "LARCENY/THEFT"]
groups = dfTheft.groupby("DayOfWeek")["Category"].count()
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
groups = groups[weekdays]
plt.figure()
groups.plot(kind="bar", title="LARCENY/THEFT per weekday")
print(groups)


# There seems to be a variance inbetween days. Friday and saturday have higher values. Not by a large margin but there appears to be a difference comapred to the other days. Let's see which group has the highest Coefficient Of Variation per day.

# In[ ]:


# Find the crime group with highest per-day-Coefficient Of Variation
dayOfWeekVars = pd.DataFrame(columns=["Category", "CoefficientOfVariation"])
rows = []
for c in df["Category"].unique():
    dfSubset = df[df["Category"] == c]
    dfSubsetGrouped = dfSubset.groupby("DayOfWeek")["Category"].count()
    std = dfSubsetGrouped.std()
    mean = dfSubsetGrouped.mean()
    cv = std / mean
    
    # Only consider category, if there are enough samples
    if (len(dfSubset) > 300):
        rows.append({'Category': c, 'CoefficientOfVariation': cv})

categoryDayCV = pd.DataFrame(rows).sort_values(by="CoefficientOfVariation", ascending=0)
#plt.figure()
categoryDayCV.plot(x="Category", kind="bar", title="Category Day Coefficient Of Variation")
plt.show()

print("Top 5 Coefficient Of Variation by day:")
print(categoryDayCV["Category"][:5])
print("Bottom 5 Coefficient Of Variation by day:")
print(categoryDayCV["Category"][-5:])


# Some of the categories appear to have a higher variaiton per day than others. To see the differences, plot the Top 5 varying categories per day.

# In[ ]:


for category in categoryDayCV["Category"][:5]:
    dfCategory = df[df["Category"] == category]
    groups = dfCategory.groupby("DayOfWeek")["Category"].count()
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    groups = groups[weekdays]
    plt.figure()
    groups.plot(kind="bar", title=category + " count by day")


# Some crimes seem to occur more often on weekends. For example drunkennes and driving under the influence.

# In[ ]:


# Function to plot data grouped by Dates and Category
def plotTimeGroup(dfGroup, ncols=10, area=False, title=None):
    categoryCV = pd.DataFrame(columns=["Category", "CV"])
    rows = []

    for column in dfGroup.columns:
        col = dfGroup[column]
        # Only consider category, if there are enough samples
        if (col.sum() > 500):
            rows.append({'Category': column, 'CV': col.std() / col.mean()})

    categoryCV = pd.DataFrame(rows).sort_values(by="CV", ascending=0)
    #The graph with all categories is unreadable. Therefore, columns with a
    # high coefficient of variation are extracted:
    topCVCategories = categoryCV[:ncols]["Category"].tolist()


    f = plt.figure(figsize=(13,8))
    ax = f.gca()
    if area:
        dfGroup[topCVCategories].plot.area(ax=ax, title=title, colormap="jet")
    else:
        dfGroup[topCVCategories].plot(ax=ax, title=title, colormap="jet")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=1, fontsize=11)


# Lets take a look at the crime's long variation over the years.
# Each category is counted for each year.

# Plot crime categories counted per year. Only display categories with a high coefficient of variation.

# In[ ]:


# Crime category count per year
dfGroup = df[["Dates", "Category"]]
# Drop year 2015 because it does not contain all months
dfGroup = dfGroup[dfGroup["Dates"].map(lambda x: x.year < 2015)]
dfGroup = dfGroup.groupby([dfGroup["Dates"].map(lambda x: x.year), "Category"])
dfGroup = dfGroup.size().unstack()

plotTimeGroup(dfGroup, title="Crime Categories History")


# In[ ]:


plotTimeGroup(dfGroup, title="Crime Categories History", area=True)


# Group data by month to visualize variance per month.

# In[ ]:


# Crime category count per year
dfGroup = df[["Dates", "Category"]]
# Drop year 2015 because it does not contain all months
dfGroup = dfGroup[dfGroup["Dates"].map(lambda x: x.year < 2015)]
dfGroup = dfGroup.groupby([dfGroup["Dates"].map(lambda x: x.month), "Category"])
dfGroup = dfGroup.size().unstack()

plotTimeGroup(dfGroup, ncols=15, title="Crime Categories Per Month")


# In[ ]:


plotTimeGroup(dfGroup, ncols=15, title="Crime Categories Per Month", area=True)

