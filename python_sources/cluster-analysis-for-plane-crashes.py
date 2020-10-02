#!/usr/bin/env python
# coding: utf-8

# # Analysis of Airplane Crashes since 1908

# ## 1. Importing modules

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats as sts
import datetime as dt
import os

print(os.listdir("../input/"))


# ## 2. Loading the data

# In[ ]:


data = pd.read_csv("../input/Airplane_Crashes_and_Fatalities_Since_1908.csv")


# Let's verify if the data was loaded correctly.

# In[ ]:


data.head()


# ## 3. Data Info and Manipulation

# On this dataset, we have info on:
# 
# * Date of the accident
# * Time of the accident
# * Location where the accident took place
# * Operator of the aircraft
# * Flight number
# * Route
# * Type of aircraft
# * Registration code
# * Construction/Serial/Line/Fuselage number
# * Number of people aboard
# * Number of fatalities
# * Number of people killed on the ground
# * ** Textual information on the crash**

# In[ ]:


data.dtypes


# In[ ]:


data.isnull().any()


# As we can see from the outputs above, we have some NaN values. For the numeric columns, we'll replace these values with 0.

# In[ ]:


data['Fatalities'].fillna(0, inplace = True)
data['Aboard'].fillna(0, inplace = True)
data['Ground'].fillna(0, inplace = True)


# Let's convert the 'Date' column to the appropriate format.

# In[ ]:


data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].dt.strftime("%m/%d/%Y")
data['Date'].head(5)


# For visualization purposes, let's create a new column with just the year from the column 'Date'.

# In[ ]:


data['Year'] = pd.DatetimeIndex(data['Date']).year
data['Year'].head(5)


# With data on number of fatalities and people aboard, we can create a new variable with the number of people that survived the crash. We're going to call this variable 'Survived'. Let's also replace any NaN value with 0 on the Survived column.

# In[ ]:


data['Survived'] = data['Aboard'] - data['Fatalities']
data['Survived'].fillna(0, inplace = True)


# Now our dataframe looks like this.

# In[ ]:


data.head(5)


# ## 4. Some statistics on the data 

# In[ ]:


data.describe()


# ## 5. Visualizations

# In[ ]:


matplotlib.rcParams['figure.figsize'] = (20, 10)
sns.set_context('talk')
sns.set_style('whitegrid')
sns.set_palette('tab20')


# ### 5.1 Airplane Crashes per Year 

# First we summarise to get the count of accidents per year

# In[ ]:


total_crashes_year = data[['Year', 'Date']].groupby('Year').count()
total_crashes_year = total_crashes_year.reset_index()
total_crashes_year.columns = ['Year', 'Crashes']


# Then we plot with Seaborn.

# In[ ]:


sns.lineplot(x = 'Year', y = 'Crashes', data = total_crashes_year)
plt.title('Total Airplane Crashes per Year')
plt.xlabel('')


# In[ ]:


total_crashes_year[total_crashes_year['Crashes'] > 80]


# From the 40's, there's a significant increase in airplane crashes, which must likely be because of World War II (1939 - 1945). The highest peaks are between 1960 and 2000. The year with most accidents is 1972, with 104 occurences.

# ### 5.2 Death Toll per Year

# In[ ]:


#summarise
pcdeaths_year = data[['Year', 'Fatalities']].groupby('Year').sum()
pcdeaths_year.reset_index(inplace = True)


# In[ ]:


# Plot
sns.lineplot(x = 'Year', y = 'Fatalities', data = pcdeaths_year)
plt.title('Total Number of Fatalities by Air Plane Crashes per Year')
plt.xlabel('')


# Here we can see the same pattern, the years that had the most accidents are also the ones with the most fatalities

# ### 5.3 People Aboard Airplanes per Year

# In[ ]:


# summarise
abrd_per_year = data[['Year', 'Aboard']].groupby('Year').sum()
abrd_per_year = abrd_per_year.reset_index()


# In[ ]:


# plot
sns.lineplot(x = 'Year', y = 'Aboard', data = abrd_per_year)
plt.title('Total of People Aboard Airplanes per Year')
plt.xlabel('')
plt.ylabel('Count')


# From the 40's, the number of people aboard airplanes starts to increase. From 1960 to 2000 is where we have most people aboard, the same years with most plane crashes and fatalities. Let's calculate Correlation Coefficients for these two variables.

# In[ ]:


sts.pearsonr(data.Fatalities, data.Aboard)


# In[ ]:


sts.spearmanr(data.Fatalities, data.Aboard)


# The coefficients suggests a pretty high correlation between the number of fatalities and people aboard. Let's visualize this relation.

# In[ ]:


sns.regplot(x = 'Aboard', y = 'Fatalities', data = data, scatter_kws=dict(alpha = 0.3), line_kws=dict(color = 'red', alpha = 0.5))
plt.title('Fatalities x People Aboard')


# ### 5.4 Fatalities vs Survived vs Killed on Ground

# Let's now visualize how the number of fatalities compare with the number of survived and those who were killed on the ground.

# In[ ]:


#summarise
FSG_per_year = data[['Year', 'Fatalities', 'Survived', 'Ground']].groupby('Year').sum()
FSG_per_year = FSG_per_year.reset_index()


# In[ ]:


#plot
sns.lineplot(x = 'Year', y = 'Fatalities', data = FSG_per_year)
sns.lineplot(x = 'Year', y = 'Survived', data = FSG_per_year)
sns.lineplot(x = 'Year', y = 'Ground', data = FSG_per_year)
plt.legend(['Fatalities', 'Survival', 'Ground'])
plt.xlabel('')
plt.ylabel('Count')
plt.title('Fatalities vs Survived vs Killed on Ground per Year')


# There's a peak close to 2000, with more than 5000 killed on the ground, way more than the number of fatalities. Let's see it.

# In[ ]:


data['Ground'].max()


# In[ ]:


data[data['Ground'] == 2750]


# This dreadful number of people killed on the ground is due to the tragic event of 9/11, where the Twin Towers were brought down by two planes hijacked by terrorists.

# ## 6. Cluster Analysis

# ### 6.1 Importing needed modules

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# ### 6.2 Data Preparation

# In the 'Summary' column, we have NaN values as well, so we're going to create a new dataframe with the 'Summary' data and dropping all rows with NaN values.

# In[ ]:


text_data = data['Summary'].dropna()
text_data = pd.DataFrame(text_data)


# Now we'll convert this text data to a list, vectorize it and remove any stop words in the data.

# In[ ]:


documents = list(text_data['Summary'])
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)


# ## 6.3 Model Fitting

# And now we fit the model. For this analysis, we'll be using the KMeans algorithm with 7 clusters.

# In[ ]:


true_k = 7
model = KMeans(n_clusters=true_k, max_iter=100, n_init=1)
model.fit(X)


# Now let's see the most common terms for each cluster.

# In[ ]:


print ('Most Common Terms per Cluster:')
order_centroids = model.cluster_centers_.argsort()[:,::-1]
terms = vectorizer.get_feature_names()

for i in range(true_k):
    print('Cluster %d:' % i)
    for ind in order_centroids[i, :10]:
        print ('%s' % terms[ind]),
    print

