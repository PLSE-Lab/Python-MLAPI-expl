#!/usr/bin/env python
# coding: utf-8

# **PURPOSE:** 
# As a dark chocolate lover, I enjoy eating chocolate bars in the 60 - 80% cacao range. 
# Recently I tried a 92% bar that I found rather bitter and I'm currently eating a 90% bar that seems light and sweet in comparison.
# I can't believe that just a 2% diference in cacao content can make such a big difference in taste. I think there must be something else.
# So I'll try to use this dataset to answer the question: What makes a chocolate bar "good" is it the cacao content, the origin of the cacao or the hability of the maker?
# 
# The definition of "good" I'll be using is the score given by some experts and compiled by Brady Brelinski, the updated dataset and the rating criteria can be found at (http://flavorsofcacao.com/index.html)
# 
# Flavors of Cacao Rating System:
# 5= Elite (Transcending beyond the ordinary limits)
# 4= Premium (Superior flavor development, character and style)
# 3= Satisfactory(3.0) to praiseworthy(3.75) (well made with special qualities)
# 2= Disappointing (Passable but contains at least one significant flaw)
# 1= Unpleasant (mostly unpalatable)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np                # linear algebra
import pandas as pd               # data frames
import seaborn as sns             # visualizations
import matplotlib.pyplot as plt   # visualizations
import scipy.stats                # statistics
from sklearn import preprocessing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


chocobars = pd.read_csv("../input/flavors_of_cacao.csv")

#Primeros registros
chocobars.head()


# In[ ]:


# No de observaciones y tipo de datos
print(chocobars.info())

# Numero de Observaciones y Columnas
print(chocobars.shape)


# Those column names are too long, it is better to change them to simpler ones
# 

# In[ ]:


chocobars.columns = ['Maker', 'Origin', 'REF', 'Review_Date', 'Cocoa_Percent', 'Maker_Location', 
              'Rating', 'Bean_Type', 'Broad_Origin']


# Also Cocoa_Percent should be a number, so the '%' symbol must be dropped

# In[ ]:


chocobars['Cocoa_Percent']=(chocobars['Cocoa_Percent']).str.replace('%', ' ')
chocobars['Cocoa_Percent']=(chocobars['Cocoa_Percent']).astype(float)


# Now, let's check that data again

# In[ ]:


chocobars.head()


# In[ ]:


# No de observaciones y tipo de datos
print(chocobars.info())

# Numero de Observaciones y Columnas
print(chocobars.shape)


# **EXPLORATORY DATA ANALYSIS**
# We have 1794 records and 9 columns: 4 numeric and the remaining 5 text.
# Let's see what we have here.

# In[ ]:


#First the numerical variables
chocobars.iloc[:,~chocobars.columns.isin(['Maker','Origin','Maker_Location','Bean_Type','Broad_Origin'])].describe()


# "REF" and "Review_Date" refer to the date when the bars were rated. I don't believe there is a reletion between the reviewing date and the rating. Howewver, I'll make a correlation analysis to confirm this. 

# In[ ]:


corr=chocobars.iloc[:,~chocobars.columns.isin(['Maker','Origin','Maker_Location','Bean_Type','Broad_Origin'])].corr()
print (corr)


# The correlation table shows high correlation between "REF" and "Review_Date" which was expected  due to the nature of these variables and  a weak relation with the other two numerical variables. Therefore I won't include ths variables in my analysis.
# 
# Another important result is that the correlation between the Cocoa Percent and the Rating is not very high and negative. 
# 
# So, the data shows that the cacao percent of a chocolate bar is not a strong indicator of its quality. If anything more cacao content seems to have a negative effect on the score recieved by a chocolate bar.

# In[ ]:


#Now the text variables
chocobars.iloc[:,~chocobars.columns.isin(['REF','Review_Date','Cocoa_Percent','Rating'])].describe()


# The column "Origin" has too many unique values and for the column "Bean_Type" out of 1794 almost half (887) are empty. These two variables would not be very useful so I'm going to exclude them from the analysis  

# In[ ]:


#Now how about the country of origin?
chocobars['Broad_Origin'].value_counts()


# In[ ]:


plt.figure(figsize=(20,20))
sns.countplot(y='Broad_Origin', data=chocobars,  order = chocobars['Broad_Origin'].value_counts().index)
plt.ylabel('Broad_Origin', fontsize=15)
plt.xlabel('Number of Bars', fontsize=15)
plt.title('Number of Bars by Origin',fontsize=15)
plt.show()


# So this variable has many different values, but it seems that the more popular origen for the cacao of this bars is Latin America.
# In the top ten of the countries od origin, 8 are located in the American Continent, one in Africa and the other one is empty.

# In[ ]:


#How about the country of the maker?
chocobars['Maker_Location'].value_counts()


# In[ ]:


#How about the country of the maker?
chocobars['Maker_Location'].value_counts()
plt.figure(figsize=(20,20))
sns.countplot(y='Maker_Location', data=chocobars,  order = chocobars['Maker_Location'].value_counts().index)
plt.ylabel('Maker_Location', fontsize=15)
plt.xlabel('Number of Bars', fontsize=15)
plt.title('Number of Bars by Maker Location',fontsize=15)
plt.show()


# While most cacao producers are located in Latin America and Africa, the manufactures of chocolate bars are mostly locates in Europe and North America.
# Ecuador is the only cacao producing conuntry in the top ten of chocolate manufacturing countries.
# Now is time to see how this variables relate to the quality of a Chocolate Bar,

# Since the number of origins is large and the value of the ratings is descrete, a scatterplot and other sorts of graphs would not give information to identify wether or not the origin of the cacao influences the quality of the chocolate bar.
# 
# I've decided to use the coeficient of variation as a measure of the consistency in the quality of the product, A lower CoV means that the ratings of the chocolate bars made with cacao from that country are less disperse.

# In[ ]:


group_Origin = chocobars.groupby('Broad_Origin').agg({'Rating': ['count', 'min','median', 'max', 'mean', 'std' ]})
group_Origin.columns=['Rcount', 'Rmin','Rmedian', 'Rmax', 'Rmean', 'Rstd']
#Coeficient of variation is a standarized measure of the dispersion
group_Origin['coef_var']=group_Origin.Rstd/group_Origin.Rmean
#group_Origin
group_Origin.sort_values(['Rmean','coef_var'], ascending=[False,True])


# Most of the bars with the best average rating are a blend of beans from differen origins, apparently the quality of the chocolate bars is not influenced by the origin of the bean.
# From this table, it seems as if the quality of a bar dependen on the decision of wich cacao beans to mix. Sadly, the Bean_Type variable has too many missing values to be of use in this analysis. 
# 
# Another posibility is that the quality of a chocolate bar relies in the mastery of the manufacturer, as a first approach we will see if there is a significant relation between the country of the manufacturer and the rating received by the chocolate bars.
# 
# However there are some minor corrections to make: Amsterdam is not a country, is a city in the Netherlands and there are a couple of countries misspelled

# In[ ]:


chocobars.loc[chocobars['Maker_Location'] == 'Amsterdam','Maker_Location'] = 'Netherlands'
chocobars.loc[chocobars['Maker_Location'] == 'Niacragua','Maker_Location'] = 'Nicaragua'
chocobars.loc[chocobars['Maker_Location'] == 'Eucador','Maker_Location'] = 'Ecuador'


# I'm going to do a scatterplot just to see if I can see any relation.

# In[ ]:


plt.figure(figsize=(20,20))
sns.boxplot(x="Rating", y="Maker_Location", data=chocobars,  order = chocobars['Maker_Location'].value_counts().index)
sns.swarmplot(x="Rating", y="Maker_Location", data=chocobars,  order = chocobars['Maker_Location'].value_counts().index)
plt.ylabel('Rating', fontsize=15)
plt.xlabel('Maker Location', fontsize=15)
plt.title('Rating by Maker Location',fontsize=15)
plt.show()


# In[ ]:


group_ML = chocobars.groupby('Maker_Location').agg({'Rating': ['count', 'min','median', 'max', 'mean', 'std' ]})
group_ML.columns=['Rcount', 'Rmin','Rmedian', 'Rmax', 'Rmean', 'Rstd']
#Coeficient of variation is a standarized measure of the dispersion
group_ML['coef_var']=group_ML.Rstd/group_ML.Rmean
#group_ML
group_ML.sort_values(['Rmean','coef_var'], ascending=[False,True])


# Many of the countries ranked in the top 10 have a relative low number of bars, probably with only one or two manufacturers.
# 
# Countries with a large number of bars appear far below in the table. More bars mean more manufacturers and more variability in the quality of the bars.
# 
# It seems that the manufacturer has an influence in the quality of the bar. I'm going to use the data from France to see if Ican identify a relationship between the manufacturer and the quality of the chocolate.
# 
# 
# 

# In[ ]:


france=chocobars[chocobars['Maker_Location']=='France']
plt.figure(figsize=(20,20))
sns.boxplot(x="Rating", y="Maker", data=chocobars,  order = chocobars['Maker'].value_counts().index)
sns.swarmplot(x="Rating", y="Maker", data=france,  order = france['Maker'].value_counts().index)
plt.ylabel('Rating', fontsize=15)
plt.xlabel('Maker Location', fontsize=15)
plt.title('Rating by Maker Location',fontsize=15)
plt.show()


# In[ ]:


group_F = france.groupby('Maker').agg({'Rating': ['count', 'min','median', 'max', 'mean', 'std' ]})
group_F.columns=['Rcount', 'Rmin','Rmedian', 'Rmax', 'Rmean', 'Rstd']
#Coeficient of variation is a standarized measure of the dispersion
group_F['coef_var']=group_F.Rstd/group_F.Rmean
group_F.sort_values(['Rmean','coef_var'], ascending=[False,True])


# Even though there are some manufacturers that seem to be more consistent in the quality of their chocolate bars than others. There doesnt seem to be a manufacturer who is consistently "better" or consistently "worse".
# 
# If anything, from these results and the results obtained when compairing the countries it seems as if the fewer chocolate bars a manufacturer produces it can receive a higher rating, probably because they can focus on the quality of one product instead of dividing their attention on several brands.

# **CONCLUSION**
# Out of the variables included in this dataframe. It seems that the quality of a chocolate bar depends more on the number of brands a company manufactures than in the cacao percentange or the origine of the beans used.
