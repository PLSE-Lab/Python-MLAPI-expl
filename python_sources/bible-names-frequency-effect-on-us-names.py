#!/usr/bin/env python
# coding: utf-8

# #Abstract#
# 
# This document provides an analysis of name frequency occurrence in the Bible and it's correlation
# with names give to US children. The paper analyzises the data between 1880 and 2014 and will be
# looking into questions how frequency occurrence of names in Bible correlate with US baby names.
# 
# ##Assumptions##
# 
# 1. Religious people are more affected by the Bible than non-religious.
# 
# 2. The frequency occurance of names in the Bible must correlate with frequency occurance of names
# given by religious people. 
# 
# 3. The bigger correlation of frequency occurance of names in the Bible and in sample population,
# the more religious sampled population is.
# 
# 4. Since Christianity and Judaism has different view on New and Old Testaments, the analysis will
# be performed on both scripts separately.
# 
# ##Limitations##
# The analysis doesn't take into account sentiment of the names in the Bible, for instance the weigth given
# to name _[Judas](https://en.wikipedia.org/wiki/Judas_Iscariot)_ is the same weight given to
# _[Paul](https://en.wikipedia.org/wiki/Paul_the_Apostle)_, even though Judas has a negative
# sentiment and Paul a positive one.
# 
# #Data Loading#
# The US names data was provided by Kaggle and was downloaded using the following _[link](https://www.kaggle.com/kaggle/us-baby-names/downloads/us-baby-names-release-2015-12-18-00-53-48.zip)_.
# 
# A list of Bible names were downloaded from _[MetaV repository](https://github.com/robertrouse/KJV-bible-database-with-metadata-MetaV-)_,
# which links together details on people, places, periods of time, and passages in the Bible at
# word-level detail. The file was downloaded using the following _[link](https://raw.githubusercontent.com/robertrouse/KJV-bible-database-with-metadata-MetaV-/master/CSV/People.csv)_.
# 
# A text of New and Old Testaments was downloaded from _[Bible Text project](https://sites.google.com/site/ruwach/bibletext)_,
# which parses and divides the Bible into separate books and chapters. Both testaments were dowloaded
# using the following links: _[New Testament](http://ruwach.googlepages.com/asv_NewTestament.zip)_,
# _[Old Testament](http://ruwach.googlepages.com/asv_OldTestament.zip)_.
# 
# Since both New and Old Testaments are provided in a distributed form, where each chapter was put in
# a separate file, a concatenation of files was performed using the following Bash script under
# appropriate folder:
# 
# find . -type f -name "*.txt" -print0 | xargs -0 cat > newTestament.txt
# 
# find . -type f -name "*.txt" -print0 | xargs -0 cat > oldTestament.txt
# 
# All datasets were uploaded to _[GitHub repository](https://github.com/aie0/data)_ under the
# folder **bible**, as standalone files for retrieval convinience and reproducibility. Since Github
# imposes 100MB file size restriction, both US national and state names files were archived into a
# single zip file. However, this document uses Kaggle datasets directly.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import RegexpTokenizer

HIGH_CORR_THRESHOLD = 0.85
TOP_BIBLE_NAMES = 20
prefix = ''
bibleNamesURL = prefix + 'People.csv'
newTestamentURL = prefix + 'newTestament.txt'
oldTestamentURL = prefix + 'oldTestament.txt'

prefix = '../input/'
stateNamesURL = prefix + 'StateNames.csv'
nationalNamesURL = prefix + 'NationalNames.csv'

nationalNamesDS = pd.read_csv(nationalNamesURL)
stateNamesDS = pd.read_csv(stateNamesURL)

try:
    bibleNamesDS = pd.read_csv(bibleNamesURL)
    # retrieve all words starting with capital letter and having atleast length of 3
    tokenizer = RegexpTokenizer("[A-Z][a-z]{2,}")
    # load new testament
    file = open(newTestamentURL)
    bibleData = file.read()
    file.close()
    newTestamentWordsCount = pd.DataFrame(tokenizer.tokenize(bibleData)).apply(pd.value_counts)

    # load old testament
    file = open(oldTestamentURL)
    bibleData = file.read()
    file.close()
    oldTestamentWordsCount = pd.DataFrame(tokenizer.tokenize(bibleData)).apply(pd.value_counts)
    bibleData = None
except:
    pass


# #Data Pre-processing#
# As a pre-processing phase, we'll remove all irrelevant data from the datasets, such as 'Gender' and
# 'Id' features.
# 
# Since Bible names dataset includes multiple instances of the same name, a unique set of names
# is extracted from it.

# In[ ]:


try:
    # remove irrelevant columns
    stateNamesDS.drop(['Id', 'Gender'], axis=1, inplace=True)
    nationalNamesDS.drop(['Id', 'Gender'], axis=1, inplace=True)

    # retrieve unique names count of each testament
    bibleNames = pd.Series(bibleNamesDS['Name'].unique())
    # filtering out Bible names
    newTestamentNamesCount = pd.merge(newTestamentWordsCount, pd.DataFrame(bibleNames), right_on=0, left_index=True)
    newTestamentNamesCount = newTestamentNamesCount.ix[:, 0:2]
    newTestamentNamesCount.columns = ['Name', 'BibleCount']

    oldTestamentNamesCount = pd.merge(oldTestamentWordsCount, pd.DataFrame(bibleNames), right_on=0, left_index=True)
    oldTestamentNamesCount = oldTestamentNamesCount.ix[:, 0:2]
    oldTestamentNamesCount.columns = ['Name', 'BibleCount']
except:
    pass


# #Exploratory Analysis of Bible Data#
# As part of preliminary analysis, we'll plot top 20 names in the Old Testament. From it we can see
# that two the most frequent names are 'God' and 'Israel'. 'God' is not really a name, even though
# there is a statistically insignificant number of babies with this name in US. Despite 'Israel'
# being a name, it's also a country, of which Old Testament is all about. Therefore it's frequency
# occurrence is skewed and doesn't assist us with what we want to achieve. We'll remove both names as
# part of pre-processing phase.
# 
# ![Top 20 names in Old Testament](https://raw.githubusercontent.com/aie0/data/master/bible/old-testament-names-count.png)

# In[ ]:


try:
    # plot top TOP_BIBLE_NAMES old testament names
    topOldTestamentNamesCount = oldTestamentNamesCount.sort_values('BibleCount', ascending=False).head(TOP_BIBLE_NAMES)
    topOldTestamentNamesCount.plot(kind='bar', x='Name', legend=False, title='Old Testament names count')

    # remove God/Israel
    oldTestamentNamesCount = oldTestamentNamesCount.drop(oldTestamentNamesCount[(oldTestamentNamesCount.Name == 'God') | (oldTestamentNamesCount.Name == 'Israel')].index)
except:
    pass


# The same goes for the New Testament and again we can see the two most frequent names as
# outliners - 'God' and 'Jesus'. Since the New Testament is a book about Jesus, the name has a skewed
# frequency and should be removed. We'll remove both names as part of pre-processing phase.
# 
# ![Top 20 names in New Testament](https://raw.githubusercontent.com/aie0/data/master/bible/new-testament-names-count.png)

# In[ ]:


try:
    # plot top TOP_BIBLE_NAMES new testament names
    topNewTestamentNamesCount = newTestamentNamesCount.sort_values('BibleCount', ascending=False).head(TOP_BIBLE_NAMES)
    topNewTestamentNamesCount.plot(kind='bar', x='Name', legend=False, title='New Testament names count')

    # remove God/Jesus
    newTestamentNamesCount = newTestamentNamesCount.drop(newTestamentNamesCount[(newTestamentNamesCount.Name == 'God') | (newTestamentNamesCount.Name == 'Jesus')].index)
except:
    pass


# #State Data Pre-processing#
# US state names dataset is filtered to contain only names appearing in the Bible.

# In[ ]:


try:
    # get state data of new testament names
    newTestamentStateNamesCount = pd.merge(newTestamentNamesCount, stateNamesDS, right_on='Name', left_on='Name')

    # get state data of old testament names
    oldTestamentStateNamesCount = pd.merge(oldTestamentNamesCount, stateNamesDS, right_on='Name', left_on='Name')

    # remove name column
    newTestamentStateNamesCount = newTestamentStateNamesCount.ix[:, 1:5]
    oldTestamentStateNamesCount = oldTestamentStateNamesCount.ix[:, 1:5]
except:
    pass


# #State Data Processing#
# Plotting highly correlated (>0.85) name frequency occurrence between US states and the Bible, reveals
# several interesting things:
# 
# 1. High correlation occurs only between 1910 and 1945, whereas correlation with New Testament names
# appears in 1910-1920, and with Old Testament in years 1930-1945.
# 
# 2. There are much more states correlating with Old Testament than with New Testament - 19 vs 3.
# 
# 3. The top US states correlating with Bible names are the same for New and Old Testaments - Alaska and Nevada.
# 
# ![New Testament US state correlation](https://raw.githubusercontent.com/aie0/data/master/bible/new-testament-state-corr.png)   ![Old Testament US state correlation](https://raw.githubusercontent.com/aie0/data/master/bible/old-testament-state-corr.png)

# In[ ]:


# scale and calculate plot states with high corr
def plotStateCorr(stateNamesCount, title):
    stateNamesCount[['Count','BibleCount']] = stateNamesCount[['Count','BibleCount']].apply(lambda x: MinMaxScaler().fit_transform(x))
    stateNamesCount = stateNamesCount.groupby(['Year', 'State']).corr()
    stateNamesCount = stateNamesCount[::2]
    highCorrStateNamesCount = stateNamesCount[stateNamesCount.Count > HIGH_CORR_THRESHOLD]
    highCorrStateNamesCount.drop(['BibleCount'], axis=1, inplace=True)
    highCorrStateNamesCount = highCorrStateNamesCount.unstack()
    highCorrStateNamesCount = highCorrStateNamesCount.reset_index()
    fg = sns.FacetGrid(data=highCorrStateNamesCount, hue='State', size=5)
    fg.map(pyplot.scatter, 'Year', 'Count').add_legend().set_axis_labels('Year', 'Correlation coefficient')
    sns.plt.title(title)

try:
    plotStateCorr(newTestamentStateNamesCount, 'Correlation of New Testament and US state names')
    plotStateCorr(oldTestamentStateNamesCount, 'Correlation of Old Testament and US state names')
    oldTestamentStateNamesCount = None
    newTestamentStateNamesCount = None
    stateNamesDS = None
except:
    pass


# #National Data Pre-processing#
# To answer the question on a national scale, the US national names dataset is similarly pre-processed -
# filtered to contain only names appearing in the Bible.

# In[ ]:


try:
    # get national data of new testament names
    newTestamentNationalNamesCount = pd.merge(newTestamentNamesCount, nationalNamesDS, right_on='Name', left_on='Name')

    # get national data of old testament names
    oldTestamentNationalNamesCount = pd.merge(oldTestamentNamesCount, nationalNamesDS, right_on='Name', left_on='Name')

    # remove name column
    newTestamentNationalNamesCount = newTestamentNationalNamesCount.ix[:, 1:4]
    oldTestamentNationalNamesCount = oldTestamentNationalNamesCount.ix[:, 1:4]
except:
    pass


# #National Data Processing#
# Plotting name frequency occurrence between US and the Bible, reveals additional information:
# 
# 1. Starting from 1960s correlation with Bible names starts to decline until the end of the
# observed timeline - nowdays.
# 
# 2. Nowdays correlation with Old Testament is much higher than with New Testament.
# 
# 3. Correlation peaks in 1930-1950.
# 
# ![New Testament US national correlation](https://raw.githubusercontent.com/aie0/data/master/bible/new-testament-national-corr.png) 
# ![Old Testament US national correlation](https://raw.githubusercontent.com/aie0/data/master/bible/old-testament-national-corr.png)

# In[ ]:


# scale and calculate plot states with high corr
def plotNationalCorr(nationalNamesCount, title):
    nationalNamesCount[['Count','BibleCount']] = nationalNamesCount[['Count','BibleCount']].apply(lambda x: MinMaxScaler().fit_transform(x))
    nationalNamesCount = nationalNamesCount.groupby('Year').corr()
    nationalNamesCount = nationalNamesCount[::2]
    nationalNamesCount.unstack().plot(kind='line', y='Count', legend=False, title=title)

try:    
    plotNationalCorr(newTestamentNationalNamesCount, 'New Testament national correlation')
    plotNationalCorr(oldTestamentNationalNamesCount, 'Old Testament national correlation')
except:
    pass


# #Conclusion#
# 
# In this report we've analyzed the US baby names data and observed a distinct correlation of name
# frequency occurrence between US states and the Bible. Analysis of national data reinforced the
# findings from US state analysis - the highest correlation appears the first half of 20th century.
# 
# We can also conclude the number of US states having a correlation with Old Testament is much higher
# than number of US states having a correlation with New Testament.
# 
# According to the paper assumptions, Nevada and Alaska should be considered as more religious states
# than others. However since there is no relible source from which we could deduce religion
# demographics of US states for the supporting period of time, the assumption cannot be considered
# verified.
