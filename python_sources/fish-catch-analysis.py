#!/usr/bin/env python
# coding: utf-8

# #Abstract#
# 
# This document provides an analysis of annual nominal catches of more than 200 species of fish and
# shellfish in the Northeast Atlantic region, which are officially submitted by 20 _[International Council for the Exploration of the Sea (ICES)](http://www.ices.dk/)_
# member countries. The paper analyzises the data between 2006 and 2014 and will be looking into
# questions which countries make the greatest impact on marine world, in which areas and on which
# species.
# 
# #Data Loading#
# 
# The annual nominal catches data was downloaded using the following _[link](http://www.ices.dk/marine-data/Documents/CatchStats/OfficialNominalCatches.zip)_.
# 
# For the purposes of fish species decoding, a dataset from _[Food and Agriculture Organization of the United Nations](http://www.fao.org/)_ is used,
# which contains a list of 12600 species selected according to their interest or relation to fisheries
# and aquaculture. The file was downloaded using the following _[link](ftp://ftp.fao.org/FI/STAT/DATA/ASFIS_sp.zip)_.
# 
# For the purposes of country codes decoding, a dataset of country codes, named **IC Country**, was downloaded from _[this site section](http://vocab.ices.dk/)_.
# 
# For the purposes of fishing areas decoding, a dataset of Northeast Atlantic area subdivions was downloaded
# using the following _[link](http://www.fao.org/fishery/xml/area/Area27/en#FAO-fishing-area-27.7)_.
# 
# All datasets were uploaded to _[GitHub repository](https://github.com/aie0/data)_ under the
# folder **catch22**, as standalone files for retrieval convinience.

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import re
import urllib
from xml.etree import ElementTree

try:
	fishCatchesDS = pd.read_csv(fishCatchesURL)
	fishCodesDS = pd.read_csv(fishCodesURL, sep='\t')
	countryCodesDS = pd.read_csv(countryCodesURL)

	# load area codes from xml
	file = urllib.urlopen(northAtlanticFAOCodesURL)
	data = file.read()
	file.close()
	root = ElementTree.fromstring(data)
	ns = {'role': 'http://www.fao.org/fi/figis/devcon/'}
	codes = {}
	for area in root.findall('./role:Area/role:AreaProfile/role:AreaStruct/role:Area/role:AreaIdent', ns):
	    for i, child in enumerate(area):
	        if i is 0:
	            name = re.sub('\(.*\)', '', child.text)
	            match = re.search(",", name)
	            if (match is not None):
	                name = name[:match.start()]
	        else:
	            code = child.attrib['Code']
	    codes[code] = name
	northAtlanticFAOCodesDS = pd.DataFrame(codes.items(), columns=['Code', 'Name'])
except:
    pass


# #Data Pre-processing#
# As a pre-processing phase, we'll remove all irrelevant data from the datasets as well as any
# rows with missing data. To make things simpler and more concise, we'll also remove all area subdivions
# from fish catch data and focus our attention on main areas only.

# In[ ]:


try:
	# slice ds
	fishCodesDS = fishCodesDS[['3A_CODE', 'English_name']]
	countryCodesDS = countryCodesDS[countryCodesDS.columns[0:2]]
	fishCatchesDS = fishCatchesDS[fishCatchesDS.columns[0:13]]
	fishCatchesDS.drop(['Units'], axis=1, inplace=True)

	# remove all subdivisions
	fishCatchesDS = fishCatchesDS.dropna()
	fishCatchesDS = fishCatchesDS[fishCatchesDS.Area.str.contains('^27\.\d+$')]
except:
	pass


# #Data Processing#
# ##Catch by country##
# Let's see which countries make the greatest impact on the marine world by plotting their respective
# annual catch.

# In[ ]:


try:
	# plot catches by countries
	fishCatchesByCountry = fishCatchesDS.groupby(['Country']).sum()
	fishCatchesByCountry = pd.merge(fishCatchesByCountry, countryCodesDS, right_on='Code', left_index=True)
	plt.figure()
	ax = fishCatchesByCountry.plot(kind='bar', x=10, stacked=True, figsize=(15, 5), title='Nominal catches by country')
	ax.set_xlabel("Country")
except:
	pass


# ![Catch by country](https://raw.githubusercontent.com/aie0/data/master/catch22/catch-by-country.png "Catch by country")
# 
# As we can see the top countries are: Norway, Iceland, Russia and Denmark. Interestingly there is
# no significant variance amoung different years for the same country. Let's take a closer look at the
# overall catch of each country and check if one presents itself there.
# 
# ##Catch by year##

# In[ ]:


try:
	# plot catches by year
	plt.figure()
	fishCatchesByYear = fishCatchesDS.ix[:, 3:11].sum()
	ax = fishCatchesByYear.plot(kind='line', figsize=(10, 5), title='Nominal catches by year', legend=False)
except:
	pass


# ![Catch by year](https://raw.githubusercontent.com/aie0/data/master/catch22/catch-by-year.png "Catch by year")
# 
# We can see the impact of 2008 financial crisis, by a slight decrease of overall catch in 2008 and 2009.
# Also the _[2010 EU fishery regulations](http://www.bbc.com/news/world-europe-13054597)_ haven't gone
# unnoticed, pushing the line down to the bottom between 2010 and 2011. However, as usually with politicians,
# sooner than later business goes back to normal, and already in 2013 we can see the overall amount
# similar to one before the applied restrictions.
# 
# ##Catch by species##
# It will be also curious to know, which species are the most favorite and come to our plates.

# In[ ]:


try:
	# plot catches by fish species
	fishCatchesBySpecies = fishCatchesDS.groupby(['Species']).sum()
	fishCatchesBySpecies = fishCatchesBySpecies.sum(axis=1)
	fishCatchesBySpecies = fishCatchesBySpecies.order(ascending=False).head(10)
	fishCatchesBySpecies = pd.merge(pd.DataFrame(fishCatchesBySpecies), fishCodesDS, how='left', right_on='3A_CODE', left_index=True)
	plt.figure()
	ax = fishCatchesBySpecies.plot(kind='bar', x='English_name', stacked=True, figsize=(10, 5), title='Nominal catches by species', legend=False)
	ax.set_xlabel("Species")
except:
	pass


# ![Catch by species](https://raw.githubusercontent.com/aie0/data/master/catch22/catch-by-species.png "Catch by species")
# 
# And the winners are, or rather loosers if one asks me: hering, cod, whiting and mackerel.
# 
# ##Catch by area##
# Our last topic of interest was the areas distribution, where the fishing occurs.

# In[ ]:


try:
	# plot catches by area
	fishCatchesByArea = fishCatchesDS.groupby(['Area']).sum()
	fishCatchesByArea = fishCatchesByArea.sum(axis=1)
	fishCatchesByArea = fishCatchesByArea.order(ascending=False).head(10)
	fishCatchesByArea = pd.merge(pd.DataFrame(fishCatchesByArea), northAtlanticFAOCodesDS, how='left', right_on='Code', left_index=True)
	plt.figure()
	ax = fishCatchesByArea.plot(kind='bar', x='Name', stacked=True, figsize=(10, 5), title='Nominal catches by area', legend=False)
	ax.set_xlabel("Area")
except:
	pass


# ![Catch by area](https://raw.githubusercontent.com/aie0/data/master/catch22/catch-by-area.png "Catch by area")
# 
# Not surprisingly, the results correlate nicely with the top countries, found previously. Norwegian and
# North Sea being close to Norway, Iceland and Faroes Islands to Iceland and Irish Sea with Skagerrak to Denmark.
# 
# ##Conclusion##
# In this report we've analyzed the data provided by International Council for the Exploration of the
# Sea and observed the impact of different countries on fishery market. We've also discovered the
# impact on the total amount made by global and local events such as financial crisis and EU
# regulations.
