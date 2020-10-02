#!/usr/bin/env python
# coding: utf-8

# **Chicago has started to under-go gentrification in historically crime ridden areas. Hypothesis: The decline in crime in specific neighborhoods is a leading indicator of home price appreciation with specific crimes having a larger impact over others.**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
import statsmodels.api as sm
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# **Load in Crime Data, Home Price Data and Chicago Neighborhood Codes - Merge house and Neighborhood Mapping dfs**

# In[ ]:


#Crime Data
cf1=pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2001_to_2004.csv', sep=',',error_bad_lines=False)
cf2=pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2005_to_2007.csv', sep=',',error_bad_lines=False)
cf3=pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2008_to_2011.csv', sep=',',error_bad_lines=False)
cf4=pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv', sep=',',error_bad_lines=False)

crime = cf1.append(cf2)
crime = crime.append(cf3)
crime = crime.append(cf4)


# In[ ]:


#Community Area Sheet
comAr=pd.read_csv('/kaggle/input/crimehousing/CommAreas.csv', sep=',')
#Housing Data
df=pd.read_csv('/kaggle/input/crimehousing/Neighborhood_Zhvi_AllHomes.csv', sep=',')
df2 = df[df['City']=='Chicago']
df2.loc[:,'RegionName'] = df2.loc[:,'RegionName'].str.upper()


# In[ ]:


#Merge housing and community area sheet
cmb = pd.merge(df2,comAr,left_on='RegionName',right_on='COMMUNITY')
cmb = cmb.drop(['the_geom','PERIMETER','AREA','COMAREA_','COMAREA_ID','COMMUNITY','SHAPE_AREA','SHAPE_LEN'], axis=1)

cmb.head(5)


# **Filter on Oakland. Oakland is located in the South Side of Chicago - known for its crime & lower houeshold incomes.**

# In[ ]:


#Oakland still dangerous but in midst of redevelopment
sthChicago = cmb[cmb['RegionName']=='OAKLAND']
# Area = police code for that community
area = float(sthChicago['AREA_NUMBE'])
print(area)


# In[ ]:


oaklandCrime = crime[crime['Community Area']==area]
oaklandCrime = oaklandCrime[oaklandCrime['Arrest']==True]
oaklandCrime = oaklandCrime.drop(['Unnamed: 0','ID','Case Number','Domestic','Community Area','Block','IUCR','Location Description','Beat','District','Ward','FBI Code','X Coordinate','Y Coordinate','Year','Updated On','Latitude','Longitude','Location'],axis=1)
oaklandCrime['Date'] = pd.to_datetime(oaklandCrime['Date'])
oaklandCrime.rename(columns={'Primary Type':'Crime'}, inplace=True)
oaklandCrime.head()
# len(oaklandCrime)


# Sum crimes for each month. Identify the trend for # crimes vs house price using a regression model. Y = price, x's are type of crime committed.

# In[ ]:


sum_of_crimes = oaklandCrime.groupby([oaklandCrime.Date.dt.year.rename('year'), oaklandCrime.Date.dt.month.rename('month'),oaklandCrime['Crime']]).sum()
sum_of_crimes = sum_of_crimes.reset_index()
sum_of_crimes['Date'] = sum_of_crimes['year'].astype(str) + "-"+sum_of_crimes['month'].astype(str)
sum_of_crimes = sum_of_crimes.drop(['year','month'],axis=1)
sum_of_crimes['Date'] = pd.to_datetime(sum_of_crimes['Date'])
sum_of_crimes = sum_of_crimes.set_index('Date')
sum_of_crimes = sum_of_crimes.sort_index()
sum_of_crimes = sum_of_crimes.loc['2002-01-01':,:]
sum_of_crimes.tail(10)

#pivot so each crime type becomes a feature
pvtCrime = sum_of_crimes.pivot_table(index='Date',columns='Crime',values='Arrest',aggfunc='sum',fill_value=0)


# In[ ]:


plt.figure(figsize=(8,10))
sum_of_crimes.groupby([sum_of_crimes['Crime']]).size().sort_values(ascending=True).plot(kind='barh')
plt.title('Number of Crimes')
plt.ylabel('Crime Type')
plt.xlabel('Crime Count')
plt.show()


# Currently not eliminating any data but would explore eliminating infrequent crimes.

# In[ ]:


housePrices = sthChicago.loc[:,'1996-04':'2019-07'].transpose().dropna()
housePrices.index = pd.to_datetime(housePrices.index)
housePrices = housePrices.loc['2002-01-1':'2016-12-01',:]
housePrices.columns = ['Price']
#make sure same entries of house prices as in crime data
housePrices = housePrices[housePrices.index.isin(pvtCrime.index)]
#scale prices down
housePrices['Price']=housePrices['Price']/10000
# housePrices.tail(10)


# Create the model using an ordinary least squares regression.

# In[ ]:


#80% training - 20% testing
training_size = int(len(pvtCrime)* 0.80)

X = pvtCrime.iloc[:training_size,:]
y = housePrices.iloc[:training_size]
X = sm.add_constant(X)
test = pvtCrime.iloc[training_size:,:]
test =sm.add_constant(test)
y_valid = housePrices.iloc[training_size:]


model = sm.OLS(y,X).fit()
predictions = model.predict(test)


# In[ ]:


model.summary()


# Looking at the p-values - We would want to drop:
# 
# * Narcotics
# * Trespass
# * Weapons Violation
# * Obscenity (very low coefficient)
# * Can also test removing high standard error features

# In[ ]:


from sklearn import metrics
print("RMSE:")
print(np.sqrt(metrics.mean_squared_error(y_valid,predictions)))


# RMSE is $29,000 (when re-scaled). Indicating a somewhat poor predictor.

# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(y_valid.index,y_valid.Price,label='Actual Price')
plt.plot(predictions.index,predictions,'r',label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price ($10,000s)')
plt.title('Predicted Price vs Actual')
plt.legend(loc='upper left')
plt.show()


# Plot showing signs of overfitting. Feature removal may help with performance. Can Rejig hypothesis to eliminate non-violent crimes / crimes with not enough data / only look at total count of crimes per month / non-arrests included in data.
# 
# **Quick Feature Removal Example**

# In[ ]:


X2 = pvtCrime.drop(['WEAPONS VIOLATION','CRIMINAL TRESPASS','NARCOTICS','OBSCENITY'],axis=1)
X3 = X2.iloc[:training_size,:]
X3 = sm.add_constant(X3)
y = housePrices.iloc[:training_size]

test2 = X2.iloc[training_size:,:]
test2 = sm.add_constant(test2)
y_valid = housePrices.iloc[training_size:]


model2 = sm.OLS(y,X3).fit()
predictions2 = model2.predict(test2)
model2.summary()


# In[ ]:


print("RMSE:")
print(np.sqrt(metrics.mean_squared_error(y_valid,predictions2)))


# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(y_valid.index,y_valid.Price,label='Actual Price')
plt.plot(predictions2.index,predictions2,'r',label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price ($10,000s)')
plt.title('Predicted Price vs Actual')
plt.legend(loc='upper left')
plt.show()


# Feature deletion improved graphical performance(less noise), albeit suffered from a lower RSME and R^2 value. More room for improvement with outlier elimination.
