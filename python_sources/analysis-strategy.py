#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# <p>
# 
# My approach to this project was to explore the data firstly, then draw out the distinct questions that come to mind a first glance of the data:
# 
# <ul>
#   <li>Has crime changed over the years? Is it decreasing?</li>
#   <li>Which are the highest and lowest crimes?</li> 
#   <li>What Skewed Distribution is this dataset?</li>
#   <li>Is there a particular day(s) of the week where crimes occur more?</li>
#   <li>Is there a trend of crime per year on all crime?</li>
#   <li>Which neighbourhood of the city are the hotspots for crime?</li> 
#   <li>Has crime types change over the years? Have they decreased?</li>
# </ul>    
# 
# The outcome to all this would be to answers the questions that came up in the analysis in the data exploration, identify high crime areas, and predicting crimes from historical data of data-time and location.  
# </p>
# 
# # Selected Techniques
# Acquiring the data from the City of Vancouver Open Data Catalogue website, the data custodian is the Vancouver Police Department where the departments GeoDASH Crime Map is the authoritative source.
# 
# Using Jupyter Notebook and Python is a great and flexible tool to create readable analyses where code, images, comments and plots are kept in one place. I started off by adding some Styling (CSS) to my project. I then proceed with my code to import all the libraries needed. 
# 
# The next approach was to prepare the data and in doing so I needed to identify any anomalies and missing data. Cleaning the data by filling in defaults of the blank rows in each column as well as enriching the data by adding new columns. 
# 
# I then was ready to explore and visualize the dataset, along with implementing the algorithm techniques.
# 
# # Technique Implementation
# In any analysis, a great guide or roadmap keeps things on track and defined the final outcome. 
# A summary list of roadmap goals:
# 
# - Exploring Data set
# - Heatmap
# - Decision Tree

# In[ ]:


#import gmplot
import numpy as np # linear algebra
import pyproj as pp
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import plotly.plotly as py
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Cleaning and Transforming the Dataset 
# <ul>
#          <li> Defaulting records by filling the blank data for "HOUR" column to "00".</li>
#          <li> Blank records for "NEIGHBOURHOOD" as "N/A".</li>
#          <li> Blank records for "HUNDRED_BLOCK" as "N/A".</li>
#          <li> Deleting "MINUTE" column as predicting information to the actual minute is not necessary here.</li>
#          <li> Adding "NeighbourhoodID" column as a category key ID for Neighbourhood.</li>
#          <li> Adding "CrimeTypeID" column as a category key ID for "TYPE" as in type of crime.</li>
#          <li> Adding "Incident" column as a row count to keep track of incident totals per crime type, etc.</li>
#          <li> Combining date fields and adding a column "Date" format.</li>
#          <li> Using "Date" to get weekday name's.</li>
#          <li> Before excluding 2017 data from the main DataFrame, creating another DataFrame just for 2017 data</li>
#          <li> Excluded data for the current year 2017, in order to work with full sets of years.</li>
#     </ul>

# In[ ]:


# Importing Dataset into a DataFrame
dfcrime = pd.read_csv('..//input/crime.csv')

# Cleaning & Transforming the data
dfcrime['HOUR'].fillna(00, inplace = True)
dfcrime['NEIGHBOURHOOD'].fillna('N/A', inplace = True)
dfcrime['HUNDRED_BLOCK'].fillna('N/A', inplace = True)
del dfcrime['MINUTE']
dfcrime['NeighbourhoodID'] = dfcrime.groupby('NEIGHBOURHOOD').ngroup().add(1)
dfcrime['CrimeTypeID'] = dfcrime.groupby('TYPE').ngroup().add(1)
dfcrime['Incident'] = 1
dfcrime['Date'] = pd.to_datetime({'year':dfcrime['YEAR'], 'month':dfcrime['MONTH'], 'day':dfcrime['DAY']})
dfcrime['DayOfWeek'] = dfcrime['Date'].dt.weekday_name
dfcrime['DayOfWeekID'] = dfcrime['Date'].dt.weekday
dfpred = dfcrime[(dfcrime['YEAR'] >= 2017)]
dfcrime = dfcrime[(dfcrime['YEAR'] < 2017)]

# Calling a dataframe results
dfcrime.head()


# ## 1. Exploration and Visualization
# ### Has crime changed over the years? Is it decreasing?
# By the graph below we can determine that crime has decreased over the years with a slight uptake in 2016. Having it been an anomaly or an event that occurred this spike in the data.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Setting plot style for all plots
plt.style.use('seaborn')

# Count all crimes and group by year
dfCrimeYear = pd.pivot_table(dfcrime, values=["Incident"],index = ["YEAR"], aggfunc='count')

# Graph results of Year by Crimes
f, ax = plt.subplots(1,1, figsize = (12, 4), dpi=100)
xdata = dfCrimeYear.index
ydata = dfCrimeYear
ax.plot(xdata, ydata)
ax.set_ylim(ymin=0, ymax=60000)
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.title('Vancouver Crimes from 2003-2017')
plt.show()


# ### Which are the highest and lowest crimes?
# 
# <p>Noticing the bar graph, the evidence is that "Theft from Vehicle" is the highest crime with "Homicide" being the lowest crime from 2003 - 2017. </p>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Pivoting dataframe by Crime Type to calculate Number of Crimes
dfCrimeType = pd.pivot_table(dfcrime, values=["Incident"],index = ["TYPE"], aggfunc='count')

dfCrimeType = dfCrimeType.sort_values(['Incident'], ascending=True)

# Create bar graph for number of crimes by Type of Crime
crimeplot = dfCrimeType.plot(kind='barh',
               figsize = (6,8),
               title='Number of Crimes Committed by Type'
             )

plt.rcParams["figure.dpi"] = 100
plt.legend(loc='lower right')
plt.ylabel('Crime Type')
plt.xlabel('Number of Crimes')
plt.show(crimeplot)


# ### What Skewed Distribution is this dataset?
# 
# This is a right skewed distribution or referred to as a positive skewness. It would be great to see how many arrests are made per year and cased closed to offset the crimes they have per year or per crime. From that theory we could see if we might get a normal distribution from or even a left skewed distribution (more arrests made in some years than crimes).

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Count of Incidents per Year By Type
dfPivYearType = pd.pivot_table(dfcrime, values=["Incident"],index = ["YEAR", "TYPE"], aggfunc='count')

dfCrimeByYear = dfPivYearType.reset_index().sort_values(['YEAR','Incident'], ascending=[1,0]).set_index(["YEAR", "TYPE"])

# Plot data on box whiskers plot
NoOfCrimes = dfCrimeByYear["Incident"]
plt.boxplot(NoOfCrimes)
plt.show()


# ### Is there a particular day(s) of the week where crimes occur more?
# 
# <p>Weekends do look higher than weekdays of when crimes happen most.</p>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Adding Days Lookup
days = ['Monday','Tuesday','Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']

# Grouping dataframe by Day of Week ID and plotting
dfcrime.groupby(dfcrime["DayOfWeekID"]).size().plot(kind='barh')

# Customizing Plot 
plt.ylabel('Days of the week')
plt.yticks(np.arange(7), days)
plt.xlabel('Number of crimes')
plt.title('Number of Crimes by Day of the Week')
plt.show()


# ### Is there a trend of crime per year on all crime?
# 
# <p>
# <ul>
#     <li>At first glance, you can see that crime is increasing as the heapmap gets darker moving to 2016</li>
#     <li>Homicide, Mischef, Vehicle collisions(All collisions) crimes look constant on the heatmap of the years</li>
#     <li>Studying the heatmap, 2010-2013 seem to display the lowest crime years </li>
# </ul>
# </p>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Create a pivot table with month and category. 
dfPivYear = dfcrime.pivot_table(values='Incident', index='TYPE', columns='YEAR', aggfunc=len)

fig, ax = plt.subplots(1, 1, figsize = (12, 6), dpi=300)
plt.title('Type of Crime By Year', fontsize=16)
plt.tick_params(labelsize=10)

sns.heatmap(
    dfPivYear.round(), 
    linecolor='lightgrey',
    linewidths=0.1,
    cmap='viridis', 
    annot=True, 
    fmt=".0f"
);

# Remove labels
ax.set_ylabel('Crime Type')    
ax.set_xlabel('Year')

plt.show()


# ## 2. Heatmap
# ### Which neighbourhood of the city are the hotspots for crime?
# 
# We can't work with the full dataset as not all coordinates were logged correctly but we can still get a good idea which neighbourhoods are the crime hot-spots. Vancouver CBD being the most noticeable, Kitsilando, Kerrisdale, Sunset(South Vancouver) and Marpole. 
# <br></br>
# 
# Attempted to acquire data of population within Vancouver city (no solid dataset is available), I was trying to draw up a correlation of population density to the crime per location. 

# In[ ]:


'''
import gmplot 
# Clean the data of zero points of latitude amd longitude as we can not plot those coordinates
dfCoord = dfcrime[(dfcrime.Latitude != 0.0) & (dfcrime.Longitude != 0.0)]

# Assign datapoints in variables
latitude = dfCoord["Latitude"]
longitude = dfCoord["Longitude"]

# Creating the location we would like to initialize the focus on. 
# Parameters: Lattitude, Longitude, Zoom
gmap = gmplot.GoogleMapPlotter(49.262, -123.135, 11)

# Overlay our datapoints onto the map
gmap.heatmap(latitude, longitude)

# Generate the heatmap into an HTML file
gmap.draw("crime_heatmap.html")
'''


# <i>Code above produces the heatmap shown below in a HTML file and I then took a screen shot of the image</i>

# ![vancouverheatmap.PNG](http://<blockquote class="imgur-embed-pub" lang="en" data-id="a/jm6qq8C"><a href="//imgur.com/jm6qq8C"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>)
# 
# <figure>
#   <img src="" style="width:60%;border-radius: 8px">
#   <figcaption style="text-align: center;">Fig. 1 - Vancouver Crime Heatmap</figcaption>
# </figure>

# ### Has crime types change over the years? Have they decreased? 
# 
# <p>There is a mixture of events per crime. At a glance it can be perceived that the majority of the crimes are decreasing over the years. Studying it closer the crime though to have decrease seems to be on the rise again. To determine this, I would have to look at other variables that influence the each crime from weather, arrests, sales, construction in the area, etc.  
# <br></br>
# 
# Continuing with the data set I can make some theoretical correlations, comparably of when one crime increases another  may decrease. For example, has Police presence decreased, are more people buying bicycles or merely finding it easier to travel to work with a bicycle, thus the drop of vehicle theft has decreased and why Bicycle theft may have increased. Awareness campaigns of people rights has maybe increased over the years and notably "Offence Against a Person" has decreased. </p>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Crime count by Category per year
dfPivCrimeDate = dfcrime.pivot_table(values='Incident'
                                     ,aggfunc=np.size
                                     ,columns='TYPE'
                                     ,index='YEAR'
                                     ,fill_value=0)
plo = dfPivCrimeDate.plot(figsize=(15, 15), subplots=True, layout=(-1, 3), sharex=False, sharey=False)


# ## 3. Decision Tree
# ### Create Training And Test Data Sets
# 
# <p>We can look at the shape of all the data to make sure we did everything correctly. We expect the training features number of columns and observations to match the testing feature number of columns and observations to match for the respective training and testing features and the labels. 
# <br></br>
# <br></br>
# 
# Proceeding with the test and training data to determine a targeted prediction the Decision tree Classifier algorithm will be used.
# </p>
# 

# In[ ]:


# New DataFrame to filter out columns needed
dfRandomF = dfcrime

# Split data for training and testing
#dfRandomF['train'] = np.random.uniform(0, 1, len(dfRandomF)) <= .70

X = dfRandomF[['YEAR', 'MONTH', 'DAY','HOUR', 'NeighbourhoodID']]

Y = dfRandomF[['TYPE']]

# To create a training and testing set, I am splitting the data
# by 70% training and 30% testing
X_train , X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 5)

print('Number of observations and columns in the training data:', X_train.shape, y_train.shape)
print('Number of observations and columns in the testing data:',X_test.shape, y_test.shape)


# ### Decision Tree Classifier with criterion gini index

# In[ ]:


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 5,
                               max_depth=5, min_samples_leaf=8)
clf_gini.fit(X_train, y_train)


# In[ ]:


# Adding prediction test
y_pred_gn = clf_gini.predict(X_test)


# ### Decision Tree Classifier with criterion information gain

# In[ ]:


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 5,
                                    max_depth=5, min_samples_leaf=8)
clf_entropy.fit(X_train, y_train)


# In[ ]:


# Adding prediction test
y_pred_en = clf_entropy.predict(X_test)


# In[ ]:


# Random values for prediction 
clf_gini.predict([[2017,1,5,15.0,12]])


# In[ ]:


# Using the same parameters for predition
dfpred[(dfpred['YEAR'] == 2017) & 
        (dfpred['MONTH'] == 1) & 
        (dfpred['DAY'] == 5) & 
        (dfpred['HOUR'] == 15.0) &
        (dfpred['NeighbourhoodID'] == 12)]


# ### Using the same parameters and excluding the Hour there is a 20% chance of accuracy on my selection.

# In[ ]:


dfpred[(dfpred['YEAR'] == 2017) & 
        (dfpred['MONTH'] == 1) & 
        (dfpred['DAY'] == 5) & 
        (dfpred['NeighbourhoodID'] == 12)]


# In[ ]:


print ('Accuracy is', accuracy_score(y_test,y_pred_gn)*100, '%')


# In[ ]:


print ('Accuracy is', accuracy_score(y_test,y_pred_en)*100, '%')


# ## Reference
# 
# Anon., n.d. Open Data Pilot Project, City of Vancouver. [Online] Available at: http://data.vancouver.ca/datacatalogue/crime-data.htm [Accessed 14 5 2018].
# 
# Osaku, Wilian, 2017. EDA of Crime in Vancouver (2003 - 2017)
# Available at: https://www.kaggle.com/wosaku/eda-of-crime-in-vancouver-2003-2017
# [Accessed 14 5 2018].
# 
# Saxena,Rahul., 2017. Building Decision Tree Algorithm In Python With Scikit Learn. [Online] 
# Available at: http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
# [Accessed 02 7 2018].
# 
# Jonathan P., 2018. How to Generate a Geographical Heatmap with Python. [Online] 
# Available at: https://eatsleepdata.com/data-viz/how-to-generate-a-geographical-heatmap-with-python.html
# [Accessed 02 7 2018].
# 
# Koehrsen, William., 2017. Random Forest in Python. [Online] 
# Available at: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
# [Accessed 02 7 2018].
# 
# A special thanks for Philip Mostert and Leon Van Vuuren for some coding guidance.

# In[ ]:


'''
# Style Report
from IPython.core.display import HTML
css_file = 'style.css'
HTML(open(css_file, 'r').read())
'''

