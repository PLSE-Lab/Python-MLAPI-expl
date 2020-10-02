#!/usr/bin/env python
# coding: utf-8

# <h1><b>Analysis and Prediction of Global Gross Agricultural Production</b></h1>
# 

# We'll see how the 200+ countries' and areas' Gross Agricultural Production performance stacked up by finding the top 10 highest producing nations and comparing them to each other.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mp
import seaborn as sb

prodind = "../input/fao_data_production_indices_data.csv"
pdid = pd.read_csv(prodind)

pdid.head()


# In[2]:


pdid.tail()


# We can get rid of the footnote rows at the end of the dataset.

# In[3]:


del pdid['element_code']
del pdid['value_footnotes']
pdid.head()


# Renaming the columns for clarity.

# In[4]:


headers = ["Region", "Production", "Year", "Unit", "Price", "Category"]
pdid.columns = headers
pdid.head()


# Looking for null values throughout the dataset.

# In[5]:


pdid.replace("?", np.nan, inplace = True)
no_data = pdid.isnull()
for column in no_data.columns.values.tolist():
    print(column)
    print (no_data[column].value_counts())
    print("")    


# In[6]:


print(pdid.shape)


# In[7]:


pdid.dropna(inplace=True)
print(pdid.shape)


# In[8]:


pdid.dtypes


# Change the year property from float to object so it is easier to reference later, and price to int for graphing.

# In[9]:


pdid[["Price", "Year"]] = pdid[["Price", "Year"]].astype("int")
pdid[["Year"]] = pdid[["Year"]].astype("object")
pdid.dtypes


# In[10]:


pdid.replace({'Gross Production 1999-2001 (1000 I$)':'Gross Production'}, inplace=True)
pdid.head()


# The year and unit information in the production column's values is redundant (and in case of the timeframe, incorrect). We can rename them for clarity. The focus for now is on Gross agricultural production, but this dataset contains different categories of product as well as measures of production. We'll single out the properties we are interested in without removing the original dataframe's values in case we want to look at other information later.

# In[11]:


pdid.replace({'Net Production 1999-2001 (1000 I$)' : 'Net Production', 'Gross PIN (base 1999-2001)' : 'Gross PIN', 'Grs per capita PIN (base 1999-2001)':'Gross Per Capita PIN', 'Net PIN (base 1999-2001)':'Net PIN', 'Net per capita PIN (base 1999-2001)':'Net Per Capita PIN'}, inplace=True)


# In[12]:


pdid.Category.unique()


# In[13]:


pdid.head()


# In[14]:


pdid1 = pdid.drop(pdid[pdid['Production'] != 'Gross Production'].index)
pdid2 = pdid1.drop(pdid1[pdid1['Category'] != 'agriculture_pin'].index)
pdid2.tail()


# In[15]:


gross_agri_prod = pdid2[['Region', 'Year', 'Price']]
gross_agri_prod.head()


# Now the data is much more focused, but not easy to read. Our index is arbitrary and our first column contains tons of duplicates: our dataframe is long when it should be wide. We'll pivot it to make a wide dataframe with a meaningful index and no repeated values.

# In[16]:


gross_agri_prod2 = gross_agri_prod.pivot(index='Region', columns='Year', values='Price')
gross_agri_prod2.head()


# Now we're getting somewhere. Let's get a total of Gross Production over 1961-2007 for each Region in the dataframe and sort them by that total, then swap the index and columns to prepare for graphing.

# In[17]:


gross_agri_prod2.columns = gross_agri_prod2.columns.astype(str)
years = list(map(str, range(1961, 2008)))
gap = gross_agri_prod2
gap['Total'] = gap.sum(axis=1)
gap.sort_values(['Total'], ascending=False, axis=0, inplace=True)
gap = gap[years].transpose()
gap.fillna(0, inplace=True)
gap.head()


# This dataset contains values such as 'Asia +' and 'Eastern Asia +', which reference some of the same nations. This could lead to a misleading/vague graph, especially if every individual country is overshadowed by the 'World +' column, which is a grand total of the entire world's Gross Agricultural Production. Luckily, they are all marked with a '+', so we can filter them out.

# In[18]:


gap_unwant = [col for col in gap.columns if '+' in col]
gap = gap.drop(gap_unwant, axis=1)
gap.head()


# Now we can seet a list of all individual nations/regions, starting with the most agricultural production and ending with the least. At 220 columns, that's a bit much to graph. Let's get a list of the top 10 Gross Agricultural Producing Nations.

# In[19]:


gap_top10 = gap.transpose()
gap_top10 = gap_top10.head(10)
gap_top10tran = gap_top10.transpose()
gap_top10tran


# Interestingly, the USSR is still in the top 10 after dropping off the map in 1992. Let's get a visual.

# In[20]:


gap_top10tran.fillna(0, inplace=True)
ax = gap_top10tran.plot(kind='area', figsize=(20,8), stacked=False)
ax.set_title('Top Gross Agricultural Producing Nations from 1961 to 2007', fontsize=20, fontweight='bold')
ax.set_ylabel('GAP in Billions (Int $)', fontsize=15)
ax.set_xlabel('')
ax.tick_params(labelsize=13)


# Now we've got an area graph of the top 10 gross agricultural producers from 1961 to 2007. A box plot might show us some more about this dataset.

# In[21]:


ax0 = gap_top10tran.plot(kind='box', figsize=(20,8))
ax0.set_title('Top Gross Agricultural Producing Nations from 1961 to 2007', fontsize=20, fontweight='bold')
ax0.set_ylabel('GAP in Billions (Int $)', fontsize=15)
ax0.tick_params(labelsize=10)


# You can analyze variance in production over time separately for each country with this.
# 
# Now let's use this data to predict what gross agricultural production might be in the future.

# In[22]:


gap = gap.reset_index()
gap_reg = gap[['Year', 'China', 'United States of America', 'India', 'USSR', 'Brazil', 'France', 'Germany', 'Italy', 'Argentina', 'Indonesia']].copy()
gap_reg.head()


# Let's see how our model expects China to perform in the future.

# In[23]:


countries = gap_reg[['China', 'United States of America', 'India', 'USSR', 'Brazil', 'France', 'Germany', 'Italy', 'Argentina', 'Indonesia']]

from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

lm = LinearRegression()

X = gap_reg[['Year']]
Y = gap_reg[['China']]

lm.fit(X,Y)

pred=lm.predict(Y)
pred_china = pred[0:11]
pred_china  


# In[24]:


plt.figure(figsize=(20, 10))
sb.jointplot(x="Year", y="China", data=gap_reg.astype(int), kind="reg")
plt.ylim(0,)


# Now let's get predictions for the other top five countries and see what the model thinks they'll look like from 2008 to 2018. We'll exclude the USSR.

# In[25]:


X = gap_reg[['Year']]
Y = gap_reg[['United States of America']]
lm.fit(X,Y)
pred1=lm.predict(Y)
pred_usa = pred1[0:11]

X = gap_reg[['Year']]
Y = gap_reg[['India']]
lm.fit(X,Y)
pred2=lm.predict(Y)
pred_ind = pred2[0:11]

X = gap_reg[['Year']]
Y = gap_reg[['Brazil']]
lm.fit(X,Y)
pred3=lm.predict(Y)
pred_bra = pred3[0:11]

X = gap_reg[['Year']]
Y = gap_reg[['France']]
lm.fit(X,Y)
pred4=lm.predict(Y)
pred_fra = pred4[0:11]


# In[26]:


pred_col = ['China', 'United States of America', 'India', 'Brazil', 'France']
pred_indx = ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']
preds = pd.DataFrame(index=pred_indx, columns=pred_col)
preds['China'] = pred_china
preds['United States of America'] = pred_usa
preds['India'] = pred_ind
preds['Brazil'] = pred_bra
preds['France'] = pred_fra
preds


# Now let's take a look at these predicted values on an area and box plot.

# In[27]:


ax = preds.plot(kind='area', figsize=(20,8), stacked=False)
ax.set_title('Predicted Top Gross Agricultural Producing Nations from 2008 to 2018', fontsize=20, fontweight='bold')
ax.set_ylabel('GAP in Billions (Int $)', fontsize=15)
ax.set_xlabel('')
ax.tick_params(labelsize=13)


# In[28]:


ax0 = preds.plot(kind='box', figsize=(20,8))
ax0.set_title('Predicted Top Gross Agricultural Producing Nations from 2008 to 2018', fontsize=20, fontweight='bold')
ax0.set_ylabel('GAP in Billions (Int $)', fontsize=15)
ax0.tick_params(labelsize=10)


# There is less variance in this predicted model. While more biased, these data do resemble trends in the original dataset.
# 
# Look at India. Seeing the original dataset could lead one to believe India may have surpassed the United States in Gross Agricultural Production given a few more years, but our model instead predicted stagnation. Maybe comparing regression models of the US to India will yield different results.

# In[29]:


sb.pairplot(gap_reg.astype(int), x_vars=["India", "United States of America"], y_vars=["Year"],
             size=5, aspect=.8, kind="reg");


# Side by side, we can see why our model came to the conclusion it did. India's production starting around 2000 begins dipping below the regression line, whereas the United States' continues on trend with steady gains. Looking at the area plot of our original data did not make this as clear as the pairplot has.

# It can be concluded from the original and predicted data that China's Gross Agricultural Production outpaced all other individual nations significantly. The United States has kept steady yet lower gains in Agricultural Production, securing its spot in second place above India as of 2007, and is predicted to continue doing so.

# I would love to find a similar dataset of Gross Agricultural Production for the years 2008 and up, to compare it with our predictions. Thanks for following along, it was fun and challenging.
