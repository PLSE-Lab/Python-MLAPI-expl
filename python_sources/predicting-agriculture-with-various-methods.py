#!/usr/bin/env python
# coding: utf-8

# # Introduction #
# This Jupyter notebook is an extension of my previous work on agricultural indicators. It continues to use the World Bank's World Development Indicators (WDI) data set in order to determine how certain values correlate and cause major production numbers. For my new project, I wish to use a number of new data analytic techniques and methods that significantly streamlines the analytic process. For this project, I examine how a number of agricultural values, including numbers on land, correlate to Value Added in agriculture, Cereal Production, and a Food Production index. 
# 
# As I continue to learn different statistical methods and machine learning types, I plan on adding to this project with those new techniques. In college I studied history and Chinese and for my honors thesis studied land privatization policies in history and how they affected the output of various industries. As I transition over to government and private sector work following my graduation, I am teaching myself statistical analysis in order to further improve my skills. This project is a demonstration of my abilities. I am using [Introduction to Statistical Learning by James, et al]][1] as my current textbook.
# 
# Updates:
# 06/27/2017: Section 1 with multiple linear regression completed.
# 06/30/2017: Started section 2, clustering.
# 
# I start by importing a number of important packages that allow us to conduct the data processing and visualizations that are the aim of this project. We use SQL as our main data import method and connect it to Pandas. 
# 
# In this first section I also list a number of regions and overarching "countries" that cause the data to double count. Eliminating these tends to decrease our R-Squared scores, but makes the project more accurate by eliminating these extra, double-counted data points. 
# 
# Finally, I'm taking from [Lj Miranda's fantastic visualization series on Philippines' Energy use seen here][2]. He has a great set of color codes and methods for visualization time series data that are worth repeating here.
# 
# 
#   [1]: http://www-bcf.usc.edu/~gareth/ISL/
#   [2]: https://www.kaggle.com/ljvmiranda/philippines-energy-use

# In[ ]:


import numpy as np # lienar algebra
import pandas as pd # data processing
import sqlite3 as sql # sql connector
import seaborn as sns # visualization
import matplotlib.pyplot as plt # visualization
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pylab import fill_between

conn = sql.connect('../input/database.sqlite')
regions = ['ARB','CSS','EAS','EAP','ECS','ECA','EUU','FCS','HPC','HIC','NOC','OEC','LCN',
           'LAC','LDC','LMY','LIC','LMC','MEA','MNA','MIC','NAC','MNP','OED','OSS','PSS',
           'SST','SAS','ZAF','SSF','SSA','UMC','WLD']

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)


# In[ ]:


# This section is used purely to look through the wide number of indicators available to us
# and pick out those that would best correlate for what we want to accomplish
# series = pd.read_sql('''SELECT * FROM Series''', con = conn)
# list(zip(series.IndicatorName,series.SeriesCode,series.LongDefinition))


# In[ ]:


ag_indicators = {
    'ag machinery, tractors':'AG.AGR.TRAC.NO',
    'ag machinery per 100 sq km':'AG.LND.TRAC.ZS',
    'ag value added per worker':'EA.PRD.AGRI.KD',
    'cereal production, tons':'AG.PRD.CREL.MT',
    'cereal yield, kg per hectare':'AG.YLD.CREL.KG',
    'fertilizer consumption, % (quantity used per unit of arable land)':'AG.CON.FERT.PT.ZS',
    'fertilizer consumption, kg per hectare of arable land':'AG.CON.FERT.ZS',
    'land under cereal production, hectare':'AG.LND.CREL.HA',
    'ag irrigated land, %':'AG.LND.IRIG.AG.ZS',
    'ag total land, %':'AG.LND.AGRI.ZS',
    'ag total land, sq km':'AG.LND.AGRI.K2',
    'arable land, %':'AG.LND.ARBL.ZS',
    'arable land, hectares':'AG.LND.ARBL.HA',
    'average precipitation, mm per year':'AG.LND.PRCP.MM',
    'land area, sq km':'AG.LND.TOTL.K2',
    'permanent cropland, %':'AG.LND.CROP.ZS',
    'food production index':'AG.PRD.FOOD.XD',
    'value added agriculture':'NV.AGR.TOTL.KD'
}             


# In[ ]:


# Iterates through the dictionary listed above -- for reference purposes originally -- 
# and reads the designated information from the SQL database into a new dictionary
# that follows the same style.
agDF = {}
for key, value in ag_indicators.items():
    statement = '''SELECT CountryCode, Year, Value FROM Indicators WHERE IndicatorCode IS "{}"'''                 .format(value)
    agDF[key] = (pd.read_sql(statement, con = conn))


# In[ ]:


# A method for cleaning and merging the dataframes established in the previous section.
#
# Using the optional argument *data allows us to incorporate a number of data sets > 1
# that we can then iterate through with a simple for loop.
#
# The method here first changes the value to a separate name dX where X is the number
# of data frames we're adding. Hence starting at n = 0 for our first data set and then
# iterating from there in the first if-then statement, and iterating further in the 
# first for loop, which does the exact same thing as the first if-then statement but
# is done in a new for loop. A better system would likely be to conduct this via 
# recursion, which may be done later on. 
def clean(df1, *data):
    n = 0
    df = df1.copy()
    if 'Value' in df1.columns.values:
        df.rename(columns={'Value':'d{}'.format(n)}, inplace=True)
        n += 1
    for f in data:
        frame = f.copy()
        if 'Value' in frame.columns.values:
            frame.rename(columns={'Value':'d{}'.format(n)}, inplace=True)
            n += 1
        df = pd.merge(df, frame, how='left', on=['CountryCode','Year'])
    # Necessary to remove the double-counting regions noted in the first section
    for name in regions:
        df.drop(df.loc[df.CountryCode == name].index, inplace=True)
    # Removes any 0 values that raise an error when doing logarithms
    for col in df.columns.values:
        df.drop(df.loc[df[col] == 0].index, inplace=True)
    return df


# # 1: Cereal Production #
# For our first set of tests and visualizations we'll work with absolute values only. Doing so instead of working with more-complex values such as cereal yield, fertilizer consumption per hectare, tractors per sq km, etc. we'll be easier to work with off the bat.
# 
# Since we're interested in cereal production, we'll start with our values for land as well as tractors since that is in an absolute number.

# In[ ]:


fertilizer = clean(agDF['fertilizer consumption, kg per hectare of arable land']
                  ,agDF['arable land, hectares'])
fertilizer['Value'] = fertilizer.d0 * fertilizer.d1
fertilizer.drop(['d0','d1'],axis=1,inplace=True)

df = clean(
    agDF['cereal production, tons'],
    agDF['ag machinery, tractors'],
    fertilizer,
    agDF['average precipitation, mm per year'],
    agDF['land under cereal production, hectare'],
    agDF['ag total land, sq km'],
    agDF['arable land, hectares'],
    agDF['land area, sq km']
)
df = df.drop(['CountryCode','Year'], axis=1)
df = np.log(df)
df.info()


# In[ ]:


corr_mean = df.corr()
plt.figure(figsize=(14, 14))
sns.heatmap(corr_mean, cbar=True, square=True, annot=True, fmt='.2f', cmap='PiYG')


# This correlation heat map is not a good visualization. Because of the wide range in correlation numbers, it's hard to discern what actually correlates well with other things. This is, primarily, because we included the values for average precipitation per year. If we remove that section it should be easier to discern the correlation numbers.

# In[ ]:


features = ['d0','d1','d2','d4','d5','d6','d7']
corr_mean = df[features].corr()
plt.figure(figsize=(14, 14))
sns.heatmap(corr_mean, cbar=True, square=True, annot=True, fmt='.2f', cmap='PiYG')


# ## 1.2 ##
# Land under cereal production in hectares (d4 in our set), has one hell of a correlation coefficient with cereal production. Since we know that the calculations for total cereal production don't contain land elements (which cereal yield would since it's cereal production per hectare) we can use that feature for our linear regression. 
# 
# Agricultural land in square kilometers also has a strong correlation with cereal production. However, it has a stronger correlation with land under cereal production. We know from the long definitions of the various series that total agriculture land incorporates land under cereal production, we can use either one. Since the latter has a better correlation with cereal production, we'll use that instead. The same method can be applied to arable land: it incorporates not only agriculture land, but also land under cereal production, so we can eliminate it as well. Finally we have total land area: unsurprisingly, this has the worst correlation among all the land values with cereal production since it is the final value of land, whereas all the others are subsets of it. We can eliminate it as well, leaving us with d0, d1, d2, and d4 -- cereal production in tons, total number of tractors in use, fertilizer, and land under cereal production in hectares. First, however, can we visualize this information in a simple manner? We'll need to calculate our land under cereal production for square kilometers, but that isn't difficult.

# In[ ]:


# Here we change the two variables that are in hectares to sq km by dividing by 100 for each
#
cereal_land = agDF['land under cereal production, hectare'].copy()
cereal_land['Value'] = cereal_land['Value'].map(lambda x: x / 100)
arable_land = agDF['arable land, hectares'].copy()
arable_land['Value'] = arable_land['Value'].map(lambda x: x / 100)
land_df = clean(
    cereal_land,
    agDF['ag total land, sq km'],
    arable_land,
    agDF['land area, sq km']
).dropna(axis=0, how='any')

usa_land = land_df.loc[land_df.CountryCode == 'USA']
fig = plt.figure()
plt.plot(usa_land.Year,usa_land.d3,label='Total Land',color=tableau20[6])
plt.plot(usa_land.Year,usa_land.d1,label='Agricultural Land',color=tableau20[0])
plt.plot(usa_land.Year,usa_land.d2,label='Arable Land',color=tableau20[3])
plt.plot(usa_land.Year,usa_land.d0,label='Land under Cereal Production',color=tableau20[4])

fill_between(usa_land.Year,usa_land.d3,0,alpha=0.5,color=tableau20[6])
fill_between(usa_land.Year,usa_land.d1,0,alpha=0.5,color=tableau20[0])
fill_between(usa_land.Year,usa_land.d2,0,alpha=0.5,color=tableau20[3])
fill_between(usa_land.Year,usa_land.d0,0,alpha=0.5,color=tableau20[4])

plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.xlabel('Years', fontsize=14)


# We can see from this visualization of the United States' land that each descending set of land values contains the next one: land under cereal production is a subset of arable land, arable land a subset of agricultural land, and agricultural land a subset of total land area. Armed with this knowledge, we can reasonably ignore using any of the other values for calculating cereal production since Land under Cereal Production (d1) contains such a strong correlation to total cereal production in tons.
# 
# Cereal production in tons is the value we're attempting to determine the value of, so it will be our dependent variable while d1 and d2 will be our independent features. We list those in our features list in order to subset more easily. Then we make a clean data frame that combines only those values and drops any NaN values from the list. We then drop the CountryCode and Year columns -- which are used as indices in our cleaner method, and normalize via logarithm the remaining values.
# 
# Then we split the set using the method from sklearn. We then divide our training and test sets into training xs, training ys, test xs and test ys, and use them for a multiple linear regression. We run this 100 times and take the average of our scores, coefficients, and intercepts.

# In[ ]:


features = ['d1','d2','d3']
df = clean(
    agDF['cereal production, tons']
    ,agDF['ag machinery, tractors']
    ,fertilizer
    ,agDF['land under cereal production, hectare']
).dropna(axis=0, how='any')
df.drop(['CountryCode','Year'], axis=1, inplace=True)
df = np.log(df)

coef1, coef2, coef3, intercept, score = [],[],[],[],[]
for x in range(1000):
    train, test = train_test_split(df, test_size=0.2)
    train_x = train[features]
    train_y = train.d0
    test_x = test[features]
    test_y = test.d0

    regr = LinearRegression()
    regr.fit(train_x, train_y)
    pred = regr.predict(test_x)
    coef1.append(regr.coef_[0])
    coef2.append(regr.coef_[1])
    coef3.append(regr.coef_[2])
    intercept.append(regr.intercept_)
    score.append(regr.score(test_x, test_y))

coeff_df = pd.DataFrame({'Features':['Tractors','Fertilizer','Land']})
coeff_df['Coefficient'] = pd.Series([np.mean(coef1),np.mean(coef2),np.mean(coef3)])
coeff_df['Intercept'] = np.mean(intercept)
coeff_df['R-Squared'] = np.mean(score)
coeff_df


# ## 1.3 ##
# An R-Squared of 0.952 isn't bad at all. This gives us a good grounding in testing values. Having eliminated a number of variables that were merely subsets all containing land under cereal production, we brought the number of features to use down to 3. In addition to land, we used the total number of tractors as well as the total amount of fertilizer in kilograms. 
# 
# We can relate this model as:
# 
# LN(y) = 0.078 LN(x0) + 0.214 LN (x1) + 0.754 LN (x2) - 0.383
# 
# Where y is cereal production in tons, x0 is the total number of tractors used, and x1 is total amount of fertilizer used, and x2 is the number of hectares of land under cereal production. Now we'll move on to other tests that require more calculations between data frames. 
# 
# This of course has some theoretical limitations: not all fertilizer from the data set was likely used purely on cereal production, as it could be used in other types of agricultural production not found here, and the same can be said of the number of tractors used, since it's highly unlikely that was used purely on cereal production given how much more land is used for agricultural production that is not cereal production. 
# 
# Let's check to see how many values we're working with here, and if simplifying that could increase our data points.

# In[ ]:


df = clean(
    agDF['cereal production, tons']
    ,agDF['ag machinery, tractors']
    ,fertilizer
    ,agDF['land under cereal production, hectare']
).dropna(axis=0, how='any')
df.info()


# 302 data points is not a lot, to say the least. Since we saw that at most we could have nearly 8000 data points, what if we simplify the model to account for the theoretical limitations noted earlier.

# In[ ]:


df = clean(agDF['cereal production, tons']
           ,agDF['ag machinery, tractors']
           ,agDF['land under cereal production, hectare']
).dropna(axis=0, how='any')
df.drop(['CountryCode','Year'],axis=1,inplace=True)
df = np.log(df)
df.info()


# 5000 data points is way better than 302, just from eliminating fertilizer from our inputs. If we run a linear regression off this, let's see what our R-squared is.
# 
# At the same time, let's change that multiple linear regression algorithm used earlier into a method to more easily replicate it throughout this code.

# In[ ]:


features = ['d1','d2']
def multiple_regression(data, features):
    coeff, intercept, score, mse = [],[],[],[]
    for x in range(len(features)):
        coeff.append([])
    for x in range(1000):
        train, test = train_test_split(df, test_size=0.2)
        train_x = train[features]
        train_y = train.d0
        test_x = test[features]
        test_y = test.d0
        
        regr = LinearRegression()
        regr.fit(train_x, train_y)
        pred = regr.predict(test_x)
        for x in range(len(features)):
            coeff[x].append(regr.coef_[x])
        intercept.append(regr.intercept_)
        score.append(regr.score(test_x,test_y))
        mse.append(metrics.mean_squared_error(test_y, pred))

    print("Coefficients: {0} \n"
          "Irreducible:  {1} \n"
          "R-Squared:    {2} \n"
          "Mean-Squared: {3}".format(
              np.array([np.mean(coeff[x]) for x in range(len(features))]),
              np.mean(intercept),np.mean(score),np.mean(mse)))

multiple_regression(df, features)


# Looking at that, simplifying our model for theoretical concerns actually improved our R-squared value by almost 2 percentage points, from 0.951 to 0.969. This means that even if we attempted to over-fit the model with extra variables -- not necessarily land -- that it wouldn't even give us a better score. That changes our model to the following:
# 
# LN( y ) = 0.179 * LN ( x0 ) + 0.891 * LN ( x1 ) + 0.372,
# 
# Where y is our cereal production in tons, x0 is tractors, and x1 is land in hectares.
# 
# Let's see if simplifying it further gives us a better R-Squared.

# In[ ]:


df = clean(agDF['cereal production, tons'],agDF['land under cereal production, hectare'])      .dropna(axis=0,how='any')
df.drop(['CountryCode','Year'],axis=1,inplace=True)
df = np.log(df)
features = ['d1']
multiple_regression(df, features)


# This final model doesn't give us a better R-squared, not too surprisingly. It may, however, be more accurate given the theoretical problems with the model mentioned earlier; specifically that the total number of tractors in each country in a given year are NOT always being used entirely for cereal production, as they might be used in other agricultural production. The same problem occurs with fertilizer consumption.
# 
# Each model does give us a good grounding in how to do multiple linear regression and how to determine good predictor variables from those that may end up being noise in our model. 

# # 2: Clustering #
