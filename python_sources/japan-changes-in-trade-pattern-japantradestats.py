#!/usr/bin/env python
# coding: utf-8

# **1. Introduction**
# 
# This is my first kernel on Kaggle. I chose the Japan Trade Statistics dataset uploaded by @TadashiNagao. I am looking at Japan's trade relations and how they have changed over the years. Export and Import both count as a trade relation.
# 
# Firstly I am creating a dataset which shows the top 5 trade partners for Japan per year.

# In[ ]:


### Japan:Changes in Trade Pattern(JapanTradeStats)
# Import pandas and numpy
import pandas as pd
import numpy as np

# Load Japan Trade Stats and its auxillaries dataset
df = pd.read_csv('../input/year_1988_2015.csv')
country = pd.read_csv('../input/country_eng.csv')
hs2 = pd.read_csv('../input/hs2_eng.csv')
hs4 = pd.read_csv('../input/hs4_eng.csv')
hs6 = pd.read_csv('../input/hs6_eng.csv')
hs9 = pd.read_csv('../input/hs9_eng.csv')

# Create new dataset Trade with Top 5 Ranks of Value Year (VY) by country
Trade = pd.Series.to_frame(df.groupby(['Year', 'Country'])['VY'].sum())
Trade = Trade.reset_index()
Trade.sort_values('VY', ascending=False)
Trade = Trade.groupby('Year').apply(lambda x: x.sort_values('VY', ascending=False))
Trade.set_index('Year', inplace=True)

# Convert Country code to Country Names
Trade['Country'].replace(to_replace = list(country['Country']), value = country['Country_name'], inplace=True)

# Select top 5 values and reshape to show Rank vs Year
Trade = Trade['Country'].groupby('Year').head().values
Trade = Trade.reshape(28, 5)
Trade = pd.DataFrame(Trade.T, index = np.arange(1, 6), columns = np.arange(1988, 2016))
Trade.index.names = ['Rank']

Trade


# **2. Isolate major changes in Trade partner positions**
# 
# **2.1 Germany falling from second position to fifth position between 1992 & 1993**
# 
# Created a new dataset for Germany with VY_x as Value year of 1992, VY_y as Value year of 1993 and VY_diff as difference between 1992 trade value and 1993 trade value.

# In[ ]:


# Germany 1992 and 1993 fall in VY Dataset
# Create a dataset with VY and hs2 and hs4 information for Germany 1992 & 1993

l = ["213"]

Germany_92 = df[df.Year.isin(["1992"]) & df.Country.isin(l)]
Germany_92 = pd.Series.to_frame(Germany_92.groupby(['exp_imp', 'hs2', 'hs4'])['VY'].sum())
Germany_92 = Germany_92.reset_index()

Germany_93 = df[df.Year.isin(["1993"]) & df.Country.isin(l)]
Germany_93 = pd.Series.to_frame(Germany_93.groupby(['exp_imp', 'hs2', 'hs4'])['VY'].sum())
Germany_93 = Germany_93.reset_index()

# Create a merged dataset of 1992 and 1993 data
Germany_compare = pd.merge(Germany_92, Germany_93, on = ['hs2', 'hs4', 'exp_imp'], how = 'outer')

# Replace hs information from auxillary file
Germany_compare['hs4'].replace(to_replace = list(hs4.hs4), value = hs4['hs4_name'], inplace=True)
Germany_compare['hs2'].replace(to_replace = list(hs2.hs2), value = hs2['hs2_name'], inplace=True)

# View top 10 items which fell in Value
Germany_compare['VY_diff'] = Germany_compare.VY_x - Germany_compare.VY_y
Germany_compare = Germany_compare.sort_values('VY_diff', ascending=False)
Germany_compare.head(10)


# **2.1 China rising from fifth position to second position between 1992 & 1993**
# 
# Created a new dataset for China with VY_x as Value year of 1992, VY_y as Value year of 1993 and VY_diff as difference between 1993 trade value and 1992 trade value.

# In[ ]:


# China 1992 and 1993 rise in VY dataset
# Create a dataset with VY and hs information for China 1992 & 1993

l = ["105"]
y = ["1992"]
China_92 = df[df.Year.isin(y) & df.Country.isin(l)]
China_92 = China_92.drop(['Year', 'Country', 'Unit1', 'Unit2', 'QY1', 'QY2'], axis = 1)

y = ["1993"]
China_93 = df[df.Year.isin(y) & df.Country.isin(l)]
China_93 = China_93.drop(['Year', 'Country', 'Unit1', 'Unit2', 'QY1', 'QY2'], axis = 1)

China_compare = pd.merge(China_92, China_93, on = ['hs2', 'hs4', 'hs6', 'hs9', 'exp_imp'], how = 'outer')

# Replace hs information from auxillary file
China_compare['hs9'].replace(to_replace = list(hs9.hs9), value = hs9['hs9_name'], inplace=True)
China_compare['hs6'].replace(to_replace = list(hs6.hs6), value = hs6['hs6_name'], inplace=True)
China_compare['hs4'].replace(to_replace = list(hs4.hs4), value = hs4['hs4_name'], inplace=True)
China_compare['hs2'].replace(to_replace = list(hs2.hs2), value = hs2['hs2_name'], inplace=True)

# View top 10 items which rose in Value
China_compare['VY_diff'] = China_compare.VY_y - China_compare.VY_x
China_compare = China_compare.sort_values('VY_diff', ascending=False)
print(China_compare.head(10))


# **3. Closer look at various trade areas**
# 
# **3.1 China's rise to second most preferred trade partner**
# 
# The new China dataset reveals few key points which shows how China moved up to the second position :-
# 
# * China in 1993 traded about 1231 new item categories which were not a part of 1993 list. The total trade value of these items were around 50157326 units.
# * In the item category Telephone sets (851730000). Japan exported around 43% of its total Telephone sets (851730000) exports to China in 1992. This share rose to about 59% in 1993.
# * In the item category Flat-rolled iron (720824100). Japan exported around 6% of its total Flat-rolled iron (720824100) exports to China in 1992. This share rose to about 50% in 1993.
# * In the item category Motorcars and other motor vehicles (870323920). Japan exported around 0.9% of its total Motorcars and other motor vehicles (870323920) exports to China in 1992. This share rose to about 2% in 1993 in an environment where the export of the item category overall dropped by 11%.
# * The total trade value for Japan rose about 7.5% between 1992 and 1993.

# In[ ]:


# China: Calculation on Items not listed in 1992 but present in 1993
print("No. of new Items:", len(China_compare[China_compare.VY_x.isnull()]))
print("Total value of the trade gained", China_compare[China_compare.VY_x.isnull()].VY_y.sum())
print("#############################################################################################")

# China: Change in percentage share of Telephone sets (851730000) export
print("China: Share of Telephone sets (851730000) trading in 1992: {:.1%}".format(China_compare.loc[3459, 'VY_x']/df[df.Year.isin(["1992"]) & df.hs9.isin(["851730000"]) & df.exp_imp.isin([1])].VY.sum()))
print("China: Share of Telephone sets (851730000) trading in 1993: {:.1%}".format(China_compare.loc[3459, 'VY_y']/df[df.Year.isin(["1993"]) & df.hs9.isin(["851730000"]) & df.exp_imp.isin([1])].VY.sum()))
print("#############################################################################################")

# China: Change in percentage share of Flat-rolled iron (720824100) export

print("China: Share of Flat-rolled iron (720824100) trading in 1992: {:.1%}".format(China_compare.loc[2133, 'VY_x']/df[df.Year.isin(["1992"]) & df.hs9.isin(["720824100"]) & df.exp_imp.isin([1])].VY.sum()))
print("China: Share of Flat-rolled iron (720824100) trading in 1993: {:.1%}".format(China_compare.loc[2133, 'VY_y']/df[df.Year.isin(["1993"]) & df.hs9.isin(["720824100"]) & df.exp_imp.isin([1])].VY.sum()))
print("#############################################################################################")

# China: Change in percentage share of Motorcars and other motor vehicles (870323920) export
print("China: Share of Motorcars and other motor vehicles (870323920) trading in 1992: {:.1%}".format(China_compare.loc[3740, 'VY_x']/df[df.Year.isin(["1992"]) & df.hs9.isin(["870323920"]) & df.exp_imp.isin([1])].VY.sum()))
print("China: Share of Motorcars and other motor vehicles (870323920) trading in 1993: {:.1%}".format(China_compare.loc[3740, 'VY_y']/df[df.Year.isin(["1993"]) & df.hs9.isin(["870323920"]) & df.exp_imp.isin([1])].VY.sum()))
VY92 = df[df.Year.isin(["1992"]) & df.hs9.isin(["870323920"]) & df.exp_imp.isin([1])].VY.sum()
VY93 = df[df.Year.isin(["1993"]) & df.hs9.isin(["870323920"]) & df.exp_imp.isin([1])].VY.sum()
print("Motorcars and other motor vehicles (870323920) export value: 1992 ->", VY92, "1993->", VY93)
print("Overall percentage drop of Motorcars and other motor vehicles (870323920) value: {:.1%}".format((VY92-VY93)/VY92))
print("#############################################################################################")

# Percentage rise in trade value for Japan between 1992 & 1993
TotalVY92 = df[df.Year.isin(["1992"])].VY.sum()
TotalVY93 = df[df.Year.isin(["1993"])].VY.sum()
print("Japan: Percentage rise in Trade Value (1992 to 1993): {:.1%}".format((TotalVY92-TotalVY93)/TotalVY92))


# **3.2 Germany's drop to fifth most preferred trade partner**
# 
# The new Germany dataset reveals few key points which shows how Germany moved down to the fifth position :-
# 
# * Germany in 1993 drop about 126 item categories as trade with Japan. The total trade value of these items were around 2488568 units.
# * In the item category Motorcars and other motor vehicles (8703) took the most hit. Germany's export share lost about 2% and import share lost about 12.5%
# * Overall Motorcars and other motor vehicles (8703) export and import fell by 14.4% and 11.1% respectively.
# * Japan's Transmission_apparatus (8525) export demand overall went down by about 20%. Specific export to Germany went down by about 2.8%. 

# In[ ]:


# Germany: Calculation on Items not listed in 1993 but present in 1992
print("No. of Items dropped in 1993:", len(Germany_compare[Germany_compare.VY_y.isnull()]))
print("Total value of the Trade lost", Germany_compare[Germany_compare.VY_y.isnull()].VY_x.sum())
print("#############################################################################################")

# Germany: Change in percentage share of Motorcars and other motor vehicles (8703) export
print("Germany: Share of Motorcars 1500cc (8703) trading in 1992: {:.1%}".format(Germany_compare.loc[753, 'VY_x']/df[df.Year.isin(["1992"]) & df.hs4.isin(["8703"]) & df.exp_imp.isin([1])].VY.sum()))
print("Germany: Share of Motorcars 1500cc (8703) trading in 1993: {:.1%}".format(Germany_compare.loc[753, 'VY_y']/df[df.Year.isin(["1993"]) & df.hs4.isin(["8703"]) & df.exp_imp.isin([1])].VY.sum()))
VY92 = df[df.Year.isin(["1992"]) & df.hs4.isin(["8703"]) & df.exp_imp.isin([1])].VY.sum()
VY93 = df[df.Year.isin(["1993"]) & df.hs4.isin(["8703"]) & df.exp_imp.isin([1])].VY.sum()
print("Total demand for Motorcars 1500cc (8703) export {:.1%}".format((VY92-VY93)/VY92))
print("#############################################################################################")

# Germany: Change in percentage share of Motorcars and other motor vehicles (8703) import

print("Germany: Share of Motorcars 1500cc (8703) trading in 1992: {:.1%}".format(Germany_compare.loc[1735, 'VY_x']/df[df.Year.isin(["1992"]) & df.hs4.isin(["8703"]) & df.exp_imp.isin([2])].VY.sum()))
print("Germany: Share of Motorcars 1500cc (8703) trading in 1993: {:.1%}".format(Germany_compare.loc[1735, 'VY_y']/df[df.Year.isin(["1993"]) & df.hs4.isin(["8703"]) & df.exp_imp.isin([2])].VY.sum()))
VY92 = df[df.Year.isin(["1992"]) & df.hs4.isin(["8703"]) & df.exp_imp.isin([2])].VY.sum()
VY93 = df[df.Year.isin(["1993"]) & df.hs4.isin(["8703"]) & df.exp_imp.isin([2])].VY.sum()
print("Total demand for Motorcars 1500cc (8703) import {:.1%}".format((VY92-VY93)/VY92))
print("#############################################################################################")

# Germany: Change in percentage share of Transmission_apparatus (8525) export
print("Germany: Share of Transmission_apparatus (8525) trading in 1992: {:.1%}".format(Germany_compare.loc[726, 'VY_x']/df[df.Year.isin(["1992"]) & df.hs4.isin(["8525"]) & df.exp_imp.isin([1])].VY.sum()))
print("Germany: Share of Transmission_apparatus (8525) trading in 1993: {:.1%}".format(Germany_compare.loc[726, 'VY_y']/df[df.Year.isin(["1993"]) & df.hs4.isin(["8525"]) & df.exp_imp.isin([1])].VY.sum()))
VY92 = df[df.Year.isin(["1992"]) & df.hs4.isin(["8525"]) & df.exp_imp.isin([1])].VY.sum()
VY93 = df[df.Year.isin(["1993"]) & df.hs4.isin(["8525"]) & df.exp_imp.isin([1])].VY.sum()
print("Total demand for Transmission_apparatus (8525) export {:.1%}".format((VY92-VY93)/VY92))

