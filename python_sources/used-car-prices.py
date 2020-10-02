#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from IPython.core.display import display, HTML

data = pd.read_csv("../input/craigslistVehiclesFull.csv")
data.head()


# In[ ]:


def buildPercentileFrame(data, targetAttribute, groupOnAttribute):
    #Clean outlying data without affecting original dataframe (year category requires cleaning but not as much as other fields)
    tempData = data.copy()
    if groupOnAttribute != "year":
        tempData[groupOnAttribute] = tempData[groupOnAttribute][~((tempData[groupOnAttribute]-tempData[groupOnAttribute].mean()).abs() > 3*tempData[groupOnAttribute].std())]
    else:
        tempData[groupOnAttribute] = tempData[groupOnAttribute][~((tempData[groupOnAttribute]-tempData[groupOnAttribute].mean()).abs() > 10*tempData[groupOnAttribute].std())]
    if targetAttribute != "year":
        tempData[targetAttribute] = tempData[targetAttribute][~((tempData[targetAttribute]-tempData[targetAttribute].mean()).abs() > 3*tempData[targetAttribute].std())]
    else:
        tempData[targetAttribute] = tempData[targetAttribute][~((tempData[targetAttribute]-tempData[targetAttribute].mean()).abs() > 10*tempData[targetAttribute].std())]

    #Build a list of 11 percentiles ranging from 0 through 100 for the group on attribute
    xPercentiles = [tempData[groupOnAttribute].quantile((i+1)/10) for i in range(10)]
    xPercentiles.insert(0, 0)
    
    #percentileDict will later be used to store lists of means for all 10 group by attribute percentiles
    percentileDict = {"mean_between_percentiles": [f"{i}-{i+10}" for i in range(0, 100, 10)]}
    
    #Loop through all percentiles of group by attribute to find subsequent percentiles for target attribute
    for i in range(10):
        #Build a temporary frame of rows between two 'group by attribute' percentiles
        xPercentileFrame = tempData[tempData[groupOnAttribute].between(xPercentiles[i], xPercentiles[i + 1])]
        
        #Build a list of 11 percentiles ranging from 0 through 100 for the target attribute using the temporary group by frame just created
        yPercentiles = [xPercentileFrame[targetAttribute].quantile((j+1)/10) for j in range(10)]
        yPercentiles.insert(0, 0)
        
        #Gather means for target attribute at all 10 percentiles
        yMeans = []
        for j in range(10):
            yPercentileFrame = xPercentileFrame[xPercentileFrame[targetAttribute].between(yPercentiles[j], yPercentiles[j+1])]
            yMeans.append(int(yPercentileFrame[targetAttribute].mean()))
        
        #Finally, add the data to a dictionary
        if len(percentileDict) == 1:
            percentileDict[f"{groupOnAttribute} between {round(xPercentiles[i])} and {round(xPercentiles[i+1])}"] = yMeans
        else:
            percentileDict[f"{round(xPercentiles[i])}-{round(xPercentiles[i+1])}"] = yMeans
            
    del tempData
    
    #Return an html table of the data
    return HTML(pd.DataFrame(percentileDict).set_index("mean_between_percentiles").to_html())


# Say we want to find a reasonable price to pay for a used car. One way to do this is compare prices by other numeric factors, in this case either mileage accumulated or year of manufacturing. A reasonable approach would be to compare these two variables using percentiles in order to see what the going rate for a car in the 20th-30th percentile of miles accumulated is on average. The program above takes this one step further by not only calculating a single mean for each category, but instead displaying another 10 price percentiles per categorical percentile. This sounds somewhat complicated but visually it makes far more sense.

# In[ ]:


display(buildPercentileFrame(data, "price", "odometer"))


# Lets start with something simple. This category compares mean price by odometer for every vehicle in the dataset. Lets analyze a hypothetical car with 55,000 miles (falling within the 46,000 and 69,400 miles category). A fantastic price would be around \$3,250 (mean of cars within 0 and 10th percentile of price). An average price would be around \$15,000, while a poor price would be anything above $20,000
# 
# Also, quick explanation as to why the first row of low odometers is far cheaper than it should be. I've spent lots of time pouring over craigslist cars+trucks over the past months due to this project and several others, and it's pretty common for used car dealerships and even individuals to list the odometer as 0 even though the car is clearly used. Unfortunately it's pretty hard to clean that data as there are plenty of new cars on craigslist as well.

# In[ ]:


fordData = data[data.manufacturer == 'ford']
display(buildPercentileFrame(fordData, "price", "odometer"))


# The previous table is fine, but comparing prices/odometers of every car out of 1.7 million is pretty generic. In order to find results specific to one's desired car, refinement is necessary. Here's a table showing only Fords.

# In[ ]:


midsizeFords = fordData[fordData["size"] == 'mid-size']
midsizeFordSedans = midsizeFords[midsizeFords.type == "sedan"]
execellentMidsizeFordSedans = midsizeFordSedans[midsizeFordSedans.condition == "excellent"]
display(buildPercentileFrame(execellentMidsizeFordSedans, "price", "odometer"))


# The Ford table changes things slightly, but it's still quite similar to the overall table seeing as Ford is one of the top manufacturers on craigslist. Narrowing things down even further to a Ford midsize sedan in 'excellent' condition we see a clear change in price. Price means are slightly more skewed due to the smaller amount of data being processed, but its clear that the \$15,000 average price for the hypothetical 55,000 mile car from earlier is not accurate. If said car is a midsize sedan built by Ford in 'excellent' condition, a fair price would be around \$11,000.

# In[ ]:


newerMidsizeFordSedans = execellentMidsizeFordSedans[execellentMidsizeFordSedans.year >= 2015]
display(buildPercentileFrame(newerMidsizeFordSedans, "price", "odometer"))
newerMidsizeFordSedans.size


# For a final refinement step, we can focus exclusively on cars manufactured in or after 2015. There is a slight increase in price for our lovely 55,000 mile sedan but it's not a huge jump given that mileage is obviously static and difference in wear between a 2010 and a 2015 sedan both driven 55,000 miles presumably  won't surmount to much. The size of the data we're dealing with has also shrunk quite a bit, down to ~10,000 which is certainly more refined than ~1,700,000.
# 
# Overall this is a helpful tool to determine what a fair selling (or buying) price is when hunting for used cars. Feel free to explore the dataset using other refinement combinations!
