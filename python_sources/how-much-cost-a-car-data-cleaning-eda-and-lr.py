#!/usr/bin/env python
# coding: utf-8

# # How much cost a car?

# <img src='https://i.imgur.com/QA8v36F.png' style='width:1000px;height:550px'/>

# ### About Dataset:

# This dataset it's a popular dataset to study regression tecnics from UCI Machine Learning Repository that can be found here: http://archive.ics.uci.edu/ml/datasets/Automobile
# 
# This data set consists of three types of entities: 
# 
# (a) the specification of an auto in terms of various characteristics, 
# 
# (b) its assigned insurance risk rating, 
# 
# (c) its normalized losses in use as compared to other cars.
# 
# The second rating corresponds to the degree to which the auto is more risky than its price indicates. Cars are initially assigned a risk factor symbol associated with its price. Then, if it is more risky (or less), this symbol is adjusted by moving it up (or down) the scale. Actuarians call this process "symboling". A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe.
# 
# The third factor is the relative average loss payment per insured vehicle year.  This value is normalized for all autos within a particular size classification (two-door small, station wagons, sports/speciality, etc...), and represents the average loss per car per year.

# ### Approach:
# 
# How much cost a car? Determined to answer this question let's performe the following steps to understand our data and be capable of create a regression model to evaluate cars of all makers and models.
# 
# - Loading Required Librarys
# - Importing and cleaning our dataset
# - Dealing with missing values
# - Exploratory Analysis
# - Feature Engineering
# - Modeling
# - Conclusion

# ### Loading Required Librarys

# In[ ]:


import matplotlib.pyplot as plt # visualization
import numpy as np
import pandas as pd
import random
import seaborn as sns # visualization
from statsmodels.graphics.gofplots import qqplot # visualization
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

# just to jupyter notebook be capable of execute matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

from pandas.plotting import register_matplotlib_converters # avoid erros between pandas and matplotlib
register_matplotlib_converters()


# ### Importing and cleaning our dataset

# In[ ]:


# Importing dataset from UCI Machine Learning Repository
cars = pd.read_csv('../input/automobile-dataset/Automobile_data.csv')

# Seeing the first 5 rows
cars.head()


# Apparently this dataset have "?" in place of missing values, how to solve that? Let's investigate!

# In[ ]:


# Seeing dataset structure
cars.info()


# As we can see some continuos features are converted to strings, let's deal with that latter

# In[ ]:


# seeing dataset shape
cars.shape


# In[ ]:


# Seeing columns
cars.describe()


# The features above seems to be okay, let's verify missing values:

# In[ ]:


# Looking for total missing values in each column
cars.isna().sum()


# We was right, the missing values was replaced by another characters, let's solve that importing our
# dataset again passing now the argument "na_values" inside pd.read_csv() with the possible characters that
# are replacing missing values
# 
# When we remove this characters from our dataset the problem with continuos as strings will be solved

# In[ ]:


# Creating a list with the possible characters that are replacing missing values
missing_values = ['?','--','-','??','.']

# Importing dataset from UCI Machine Learning Repository
cars = pd.read_csv('../input/automobile-dataset/Automobile_data.csv',na_values = missing_values)

# Seeing the first 5 rows
cars.head()


# In[ ]:


# Verifying continuos variables
cars.describe()


# ### Dealing with missing values

# Apparently everything it's okay with missing values, let's plot a bar graph with na's percentage

# In[ ]:


# Ploting missing values percentage in each dataset
plt.subplots(0,0, figsize = (18,5))
ax = (cars.isnull().sum()).sort_values(ascending = False).plot.bar(color = 'blue')
plt.title('Missing values per column', fontsize = 20);


# Now we can see clearly the missing values proportion, normalized_losses have more missing values, let's deal with this feature first:

# In[ ]:


# Interpolating a linear regression to replace missing values in continuos variables
cars['normalized-losses'] = cars['normalized-losses'].interpolate(method = "linear"
                                      ,limit_direction = "both")

cars['price'] = cars['price'].interpolate(method = "linear"
                                      ,limit_direction = "both")

cars['stroke'] = cars['stroke'].interpolate(method = "linear"
                                      ,limit_direction = "both")

cars['bore'] = cars['bore'].interpolate(method = "linear"
                                      ,limit_direction = "both")

cars['peak-rpm'] = cars['peak-rpm'].interpolate(method = "linear"
                                      ,limit_direction = "both")

cars['horsepower'] = cars['horsepower'].interpolate(method = "linear"
                                      ,limit_direction = "both")


# The feature 'num_of_doors' represent the number of doors(two, four) but are described as a string object, let's investigate more to replace missing values in this feature

# In[ ]:


# Counting number of missing values in num_of_doors
cars['num-of-doors'].isna().sum()


# Only 2 missing values, let's make a simple analysis by cars make and body_style and try to infer how many doors this cars have

# In[ ]:


# Looking what body_style and make our missing values have
cars[['make','body-style']][cars['num-of-doors'].isnull()==True]


# As we can see we have a car that are a sedan from dodge make and we have a sedan from mazda make, let's see how many doors each of this car types have

# In[ ]:


# Seeing how many doors a mazda sedan have
cars['num-of-doors'][(cars['body-style']=='sedan') & (cars['make']=='mazda')]


# In[ ]:


# Seeing how many doors a dodge sedan have
cars['num-of-doors'][(cars['body-style']=='sedan') & (cars['make']=='dodge')]


# As we can see this type of cars have four doors, let's replace missing values in this feature with 'four'

# In[ ]:


# Replacing missing values into num_of_doors
cars['num-of-doors'] = cars['num-of-doors'].fillna('four')


# Now let's verify if missing values was replaced correctly

# In[ ]:


# Verifying missing alues
cars.isna().sum()


# ### Exploratory Analysis

# Now that we already cleaned our data and treated missing values, let's explore our dataset to understand and get insights from it

# First of all let's take a look into our target: 'price'

# In[ ]:


# Plotting a histogram of our feature price
plt.figure(figsize=(8,6)) # creating the figure
plt.hist(cars['price'] # plotting the histogram
         ,bins=30 # defyning number of bars
         ,label='price' # add legend
        ,color='blue') # defyning the color

plt.xlabel('price') # add xlabel
plt.ylabel('frequency') # add ylabel
plt.legend()
plt.title('price distribution');


# Clearly our target have not a normal distribution, let's deal with that latter in feature engginering, let's take a look into all numerical distributions

# In[ ]:


# Saving numerical features
num_var = ['symboling','normalized-losses','wheel-base','length'
          ,'width','height','curb-weight','engine-size','bore'
           ,'stroke','compression-ratio','horsepower','peak-rpm'
           ,'city-mpg','highway-mpg']

# plotting a histogram for each feature
cars[num_var].hist(bins=10
                   , figsize=(50,30)
                   , layout=(4,4));


# None of our numerical features have a normal distribution, let's deal with that latter in feature engginering, now let's a look in each numerical feature correlation with our target and with each other

# In[ ]:


# Numerical variables correlation
corr = cars.corr() # creting the correlation matrix

plt.figure(figsize=(12,12)) # creating the and difyning figure size
ax = sns.heatmap( # plotting correlation matrix
    corr,vmin=-1, vmax=1, center=0,
    annot=True,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels( # adding axes values
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# engine_size, horsepower, curb_weight, length and width have a high correlation with car prices, let's make a scatter plot for each of this features to understand better this relation

# In[ ]:


# Plotting a scatter plot of relation between engine_size and price
plt.figure(figsize=(10,6)) # creating the figure
plt.plot(cars['engine-size'] # plotting
         ,cars['price'],'.'
        ,color='blue')

plt.xlabel('engine-size') # add xlabel
plt.ylabel('price') # add ylabel
plt.title('Relation between engine-size and price');


# A beautiful distribution, with linear relation

# In[ ]:


# Plotting a scatter plot of relation between horsepower and price
plt.figure(figsize=(10,6)) # creating the figure
plt.plot(cars['horsepower'] # plotting
         ,cars['price'],'.'
        ,color='blue')

plt.xlabel('horsepower') # add xlabel
plt.ylabel('price') # add ylabel
plt.title('Relation between horsepower and price');


# A good distribution but we have also some outlayers, let's see now curb_weight vs price

# In[ ]:


# Plotting a scatter plot of relation between curb_weight and price
plt.figure(figsize=(10,6)) # creating the figure
plt.plot(cars['curb-weight'] # plotting
         ,cars['price'],'.'
        ,color='blue')

plt.xlabel('curb-weight')
plt.ylabel('price')
plt.title('Relation between curb-weight and price');


# Let's take a look into length feature vs prices

# In[ ]:


# Plotting a scatter plot of relation between length and price
plt.figure(figsize=(10,6)) # creating the figure
plt.plot(cars['length'] # plotting
         ,cars['price'],'.'
        ,color='blue')

plt.xlabel('length')
plt.ylabel('price')
plt.title('Relation between length and price');


# Very similar with all another high correlated features, i'll need to remove some of this features latter do avoid overfitting  in our model

# In[ ]:


# Plotting a scatter plot of relation between width and price
plt.figure(figsize=(10,6)) # creating the figure
plt.plot(cars['width'] # plotting
         ,cars['price'],'.'
        ,color='blue')

plt.xlabel('width')
plt.ylabel('price')
plt.title('Relation between width and price');


# Not soo linear, latter in feature engginering i'll deal with unnormal distribution of target to solve some problems

# ###### Let's make a bivariate a analysis of each categorical feature with our target to understand better this features

# In[ ]:


# Plotting Distribution of maker into price
plt.figure(figsize=(24,8))
sns.boxplot(x='make',y='price',data=cars, 
                 palette="colorblind")
plt.title('Distribution of make into price');


# Clearly the makers mercedes-benz, porsche, jaguar and bmw have the high prices, with this information we can create a new feature with this makers

# In[ ]:


# Plotting Distribution of num_of_cylinders into price
plt.figure(figsize=(15,6))
sns.boxplot(x='num-of-cylinders',y='price',data=cars, 
                 palette="colorblind")
plt.title('Distribution of num-of-cylinders into price');


# We will convert this feature to continuos latter but we already can see that much more cylinders, more high prices, cars with eight and twelve cylinders shows that in boxplot

# In[ ]:


# Plotting Distribution of fuel_system into price
plt.figure(figsize=(15,6))
sns.boxplot(x='fuel-system',y='price',data=cars, 
                 palette="colorblind")
plt.title('Distribution of fuel-system into price');


# Here we can see that some outlayers cars with mpfi fuel system have high prices but all other fuel system types are all custing less then 25000

# In[ ]:


# Plotting Distribution of engine_type into price
plt.figure(figsize=(15,6))
sns.boxplot(x='engine-type',y='price',data=cars, 
                 palette="colorblind")
plt.title('Distribution of engine-type into price');


# Here if we exclude the outlayers we can clarly see that cars with more high prices have ohcv or dohcv engines

# In[ ]:


# Plotting Distribution of body_style into price
plt.figure(figsize=(12,6))
sns.boxplot(x='body-style',y='price',data=cars, 
                 palette="colorblind")
plt.title('Distribution of body-style into price');


# Clearly we have more sheep cars that are hatchback, sedan or wagon, only a feel outlayers sedan cars have high prices

# In[ ]:


# Plotting Distribution of drive_wheels into price
plt.figure(figsize=(12,6))
sns.boxplot(x='drive-wheels',y='price',data=cars, 
                 palette="colorblind")
plt.title('Distribution of drive-wheels into price');


# Cars with fwd and 4wd drive wheels tends to be more sheep and rwd cars more expensive

# In[ ]:


# Plotting Distribution of engine_location into price
plt.figure(figsize=(10,6))
sns.boxplot(x='engine-location',y='price',data=cars, 
                 palette="colorblind")
plt.title('Distribution of engine-location into price');


# All cars with the engine located into rear have high prices

# In[ ]:


# Plotting Distribution of fuel_type into price
plt.figure(figsize=(10,6))
sns.boxplot(x='fuel-type',y='price',data=cars, 
                 palette="colorblind")
plt.title('Distribution of fuel-type into price');


# Cars with diesel have high prices but exist some outlayers cars with gas that have a high price and we can work with that in feature engineering

# In[ ]:


# Plotting Distribution of aspiration into price
plt.figure(figsize=(10,6))
sns.boxplot(x='aspiration',y='price',data=cars, 
                 palette="colorblind")
plt.title('Distribution of aspiration into price');


# Cars with turbo tends to have more high prices but exist outlayers with std with high prices too

# In[ ]:


# Plotting Distribution of num_of_doors into price
plt.figure(figsize=(10,6))
sns.boxplot(x='num-of-doors',y='price',data=cars, 
                 palette="colorblind")
plt.title('Distribution of num-of-doors into price');


# We have high prices cars with two and four doors, represented into boxplot as outlayers

# ### Feature Engineering

# ###### Dealing with outlayers in numeric features

# We can create a boxplot for each numerical feature to visualize outlayers and use this boxplots to remove then using Boxplot Interquartile Range(IQR) method.

# ###### But what is IQR?

# IQR is simple the diference between Q3(25th percentile) and Q1(75th percentile), values above (Q3+1.5 * IQR) and values below (Q1-1.5 * IQR) are considered outlayers, in a boxplot we can find all this values and use then to remove outlayers from our data.

# <img src='https://i.imgur.com/btxdIWH.png' style='width:1000px;height:550px'/>

# In[ ]:


# Plotting boxplots to numeric features
num_var = ['normalized-losses','wheel-base','length'
          ,'width','height','engine-size','bore'
           ,'stroke','compression-ratio','horsepower'
           ,'city-mpg','highway-mpg']

plt.figure(figsize=(20,10))
sns.boxplot(data=cars[num_var], 
                 palette="colorblind")
plt.title('Numerical features outlayers');


# As we can see many or our numerical features have outlayers, let's remove then

# In[ ]:


# creating a for to replace outlayers using boxplot method
for i in num_var:
    # taking quantiles
    Q1 = cars[i].quantile(0.25)
    Q3 = cars[i].quantile(0.75)
    IQR = Q3 - Q1 # calculating IQR
    cars[i] = np.where(cars[i]>(Q3+1.5*IQR),(Q3+1.5*IQR),cars[i]) # removing outlayers
    cars[i] = np.where(cars[i]<(Q1-1.5*IQR),(Q1-1.5*IQR),cars[i]) # removing outlayers


# Now let's verifying if outlayers was removed

# In[ ]:


# Plotting boxplots to numeric features
num_var = ['normalized-losses','wheel-base','length'
          ,'width','height','engine-size','bore'
           ,'stroke','compression-ratio','horsepower'
           ,'city-mpg','highway-mpg']

plt.figure(figsize=(20,10))
sns.boxplot(data=cars[num_var], 
                 palette="colorblind")
plt.title('Numerical features outlayers');


# Dealing with outlayers in our target

# In[ ]:


# Plotting a boxplot of our target to visualize outlayers
plt.figure(figsize=(6,6))
sns.boxplot(y='price',data=cars, 
                 palette="colorblind")
plt.title('price outlayers');


# Here we can see that cars prices above 30000 are outlayers, let's remove then

# In[ ]:


# Replace outlayers using boxplot method
Q1 = cars['price'].quantile(0.25) # taking Q1
Q3 = cars['price'].quantile(0.75) # taking Q3
IQR = Q3 - Q1 # calculating IQR
cars['price'] = np.where(cars['price']>(Q3+1.5*IQR),(Q3+1.5*IQR),cars['price']) # removing outlayers
cars['price'] = np.where(cars['price']<(Q1-1.5*IQR),(Q1-1.5*IQR),cars['price']) # removing outlayers


# In[ ]:


# Visualizing if outlayers was removed
plt.figure(figsize=(6,6))
sns.boxplot(y='price',data=cars, 
                 palette="colorblind")
plt.title('price outlayers');


# Now its perfect, let's normalize our numerical features using log transformation

# ###### Normalizing our numerical features wigh Log-transformation

# First of all let's treat our target 'price'

# In[ ]:


# Quantile-Quantile Plot to virify normal distributions
qqplot(cars['price'], line='s')
plt.title('Quantile-Quantile Plot')
plt.show()


# Clearly we have not a normal distribution in our target, let's use the numpy fuction log1p which  applies log(1+x) to all elements of the column to handdle with that

# In[ ]:


# Log-transformation of the target variable
cars['price'] = np.log1p(cars['price'])

# Quantile-Quantile Plot to virify normal distributions
qqplot(cars['price'], line='s')
plt.title('Quantile-Quantile Plot')
plt.show()


# Not perfect, but much better, now we have values above 10 and below 9 more close to the central line, let's treat another features with the same tecnics

# num_of_cylinders are represented as categorys but it's a continuos feature, let's convert

# In[ ]:


# Converting num_of_cylinders into a continuos variable
cars['num-of-cylinders'][cars['num-of-cylinders']=='two'] = 2
cars['num-of-cylinders'][cars['num-of-cylinders']=='three'] = 3
cars['num-of-cylinders'][cars['num-of-cylinders']=='four'] = 4
cars['num-of-cylinders'][cars['num-of-cylinders']=='five'] = 5
cars['num-of-cylinders'][cars['num-of-cylinders']=='six'] = 6
cars['num-of-cylinders'][cars['num-of-cylinders']=='eight'] = 8
cars['num-of-cylinders'][cars['num-of-cylinders']=='twelve'] = 12

# converting into integer
cars['num-of-cylinders'] = cars['num-of-cylinders'].astype('int64')


# Now let's see this feature relation with prices

# In[ ]:


# Plotting a scatter plot of relation between num_of_cylinders and price
plt.figure(figsize=(10,6)) # creating the figure
plt.plot(cars['num-of-cylinders'] # plotting
         ,cars['price'],'.'
        ,color='blue')

plt.xlabel('num-of-cylinders')
plt.ylabel('price')
plt.title('Relation between num-of-cylinders and price');


# As we already see above in boxplots, cars with more cylinders aparently tends to be more expensive

# Now let's use log trasformation to solve some problemas with features that are not normal distributed

# In[ ]:


# taking numerical features to log trasformation
num_var = ['normalized-losses','wheel-base','length'
          ,'width','height','curb-weight','engine-size','bore'
           ,'stroke','compression-ratio','horsepower','peak-rpm'
           ,'city-mpg','highway-mpg','num-of-cylinders']

cars[num_var] = np.log1p(cars[num_var]) # log transformation


# Let's visualize the new numerical distributions

# In[ ]:


# plotting a histogram for each feature
cars[num_var].hist(bins=10
                   , figsize=(50,30)
                   , layout=(4,4));


# Now with the first outlayers removed, target treated and our first continuos features normalized let's create new features, treat outlayers, do log transformation and do a correlation analysis to remove high correlated features

# ###### Creating new features

# First of all let's try to remove some of high correlations creating new features with relation between high correlated features

# In[ ]:


# relation between length and width
cars['len_wid'] = cars['length']/cars['width']

# relation between wheel_base and curb_weight
cars['whb_c_wght'] = cars['wheel-base']/cars['curb-weight']

# relation between horsepower and engine_size
cars['hpw_eng_size'] = cars['horsepower']/cars['engine-size']

# relation between highway_mpg and city_mpg
cars['hway_cit_mpg'] = cars['highway-mpg']/cars['city-mpg']


# Let's see distributions of this new features

# In[ ]:


# Saving new features
new_feat1 = ['len_wid','whb_c_wght','hpw_eng_size','hway_cit_mpg']

# plotting a histogram for each feature
cars[new_feat1].hist(bins=15
                   , figsize=(20,4)
                   , layout=(1,4));


# Now let's create some features that have a create relation with each other, like mpg and horsepower

# In[ ]:


# creating a feature to represent the mean mpg
cars['mean_mpg'] = (cars['highway-mpg']+cars['city-mpg'])/2

# creating a feature to represent the horsepower per cylinders
cars['hpw_cylinders'] = cars['horsepower']/cars['num-of-cylinders']

# creating a feature to represent the mean mpg per horsepower
cars['mean_mpg_hpw'] = cars['mean_mpg']/cars['horsepower']


# Seeing new features distributions

# In[ ]:


# Saving new features
new_feat2 = ['mean_mpg','hpw_cylinders','mean_mpg_hpw']

# plotting a histogram for each feature
cars[new_feat2].hist(bins=15
                   , figsize=(20,5)
                   , layout=(1,3));


# Now let's convert some binary categorical features into continuos using boxplot distribution related to car prices

# In[ ]:


# Converting engine_location into binary
cars['engine-location'] = np.where(cars['engine-location']=='front',2,1)

# Converting fuel_type into binary
cars['fuel-type'] = np.where(cars['fuel-type']=='gas',2,1)

# Converting aspiration into binary
cars['aspiration'] = np.where(cars['aspiration']=='std',2,1)

# Converting num_of_doors into binary
cars['num-of-doors'] = np.where(cars['num-of-doors']=='two',2,1)


# Here I use boxplots of categorical features in Exploratory analysis step to convert categorical features into continuos features using boxplot distribution, i use range of prices that this categorys was distributed to give then a level of importance, lower levels of importance means that this category can predict low prices more accurattly and high levels of importance means that this category can predict high prices more accurattly, the importance levels was distributed like as 1,2,3,4,5...

# In[ ]:


# converting maker to continuos based on boxplots
cars['make'][cars['make']=='chevrolet'] = 1
cars['make'][cars['make']=='renault'] = 2
cars['make'][cars['make']=='isuzu'] = 3
cars['make'][cars['make']=='subaru'] = 4
cars['make'][cars['make']=='plymouth'] = 5
cars['make'][cars['make']=='dodge'] = 6
cars['make'][cars['make']=='honda'] = 7
cars['make'][cars['make']=='volkswagen'] = 8
cars['make'][cars['make']=='mitsubishi'] = 9
cars['make'][cars['make']=='alfa-romero'] = 10
cars['make'][cars['make']=='mercury'] = 11
cars['make'][cars['make']=='toyota'] = 12
cars['make'][cars['make']=='peugot'] = 13
cars['make'][cars['make']=='mazda'] = 14
cars['make'][cars['make']=='saab'] = 15
cars['make'][cars['make']=='nissan'] = 16
cars['make'][cars['make']=='volvo'] = 17
cars['make'][cars['make']=='audi'] = 18
cars['make'][cars['make']=='jaguar'] = 19
cars['make'][cars['make']=='porsche'] = 20
cars['make'][cars['make']=='bmw'] = 21
cars['make'][cars['make']=='mercedes-benz'] = 22
cars['make']=cars['make'].astype('int64')


# In[ ]:


# Plotting a scatter plot of relation between width and price
plt.figure(figsize=(10,6)) # creating the figure
plt.plot(cars['make'] # plotting
         ,cars['price'],'.'
        ,color='purple')

plt.xlabel('make')
plt.ylabel('price')
plt.title('Relation between make and price');


# In[ ]:


# converting fuel_system to continuos based on boxplots
cars['fuel-system'][cars['fuel-system']=='1bbl'] = 1
cars['fuel-system'][cars['fuel-system']=='spfi'] = 2
cars['fuel-system'][cars['fuel-system']=='2bbl'] = 3
cars['fuel-system'][cars['fuel-system']=='mfi'] = 4
cars['fuel-system'][cars['fuel-system']=='4bbl'] = 5
cars['fuel-system'][cars['fuel-system']=='spdi'] = 6
cars['fuel-system'][cars['fuel-system']=='idi'] = 7
cars['fuel-system'][cars['fuel-system']=='mpfi'] = 8
cars['fuel-system']=cars['fuel-system'].astype('int64')


# In[ ]:


# Plotting a scatter plot of relation between width and price
plt.figure(figsize=(10,6)) # creating the figure
plt.plot(cars['fuel-system'] # plotting
         ,cars['price'],'.'
        ,color='purple')

plt.xlabel('fuel-system')
plt.ylabel('price')
plt.title('Relation between fuel-system and price');


# In[ ]:


# converting engine_type to continuos based on boxplots
cars['engine-type'][cars['engine-type']=='rotor'] = 1
cars['engine-type'][cars['engine-type']=='l'] = 2
cars['engine-type'][cars['engine-type']=='dohcv'] = 3
cars['engine-type'][cars['engine-type']=='dohc'] = 4
cars['engine-type'][cars['engine-type']=='ohcf'] = 5
cars['engine-type'][cars['engine-type']=='ohc'] = 6
cars['engine-type'][cars['engine-type']=='ohcv'] = 7
cars['engine-type']=cars['engine-type'].astype('int64')


# In[ ]:


# Plotting a scatter plot of relation between width and price
plt.figure(figsize=(10,6)) # creating the figure
plt.plot(cars['engine-type'] # plotting
         ,cars['price'],'.'
        ,color='purple')

plt.xlabel('engine-type')
plt.ylabel('price')
plt.title('Relation between engine_type and price');


# In[ ]:


# converting body_style to continuos based on boxplots
cars['body-style'][cars['body-style']=='hatchback'] = 1
cars['body-style'][cars['body-style']=='wagon'] = 2
cars['body-style'][cars['body-style']=='convertible'] = 4
cars['body-style'][cars['body-style']=='sedan'] = 3
cars['body-style'][cars['body-style']=='hardtop'] = 5
cars['body-style']=cars['body-style'].astype('int64')


# In[ ]:


# Plotting a scatter plot of relation between width and price
plt.figure(figsize=(10,6)) # creating the figure
plt.plot(cars['body-style'] # plotting
         ,cars['price'],'.'
        ,color='purple')

plt.xlabel('body-style')
plt.ylabel('price')
plt.title('Relation between body-style and price');


# In[ ]:


# converting drive_wheels to continuos based on boxplots
cars['drive-wheels'][cars['drive-wheels']=='4wd'] = 1
cars['drive-wheels'][cars['drive-wheels']=='fwd'] = 2
cars['drive-wheels'][cars['drive-wheels']=='rwd'] = 3
cars['drive-wheels']=cars['drive-wheels'].astype('int64')


# In[ ]:


# Plotting a scatter plot of relation between width and price
plt.figure(figsize=(10,6)) # creating the figure
plt.plot(cars['drive-wheels'] # plotting
         ,cars['price'],'.'
        ,color='purple')

plt.xlabel('drive-wheels')
plt.ylabel('price')
plt.title('Relation between drive-wheels and price');


# Maker and fuel_system seens to have a good 'linear' relation with our target, latter let's do a correlation analysis to verify this

# ###### Dealing with outlayers in new features

# Here we need to be sure that new features will have not outlayers, let's do a boxplot for each feature to see this

# In[ ]:


# Plotting boxplots to numeric features
num_var = ['mean_mpg','hpw_cylinders','mean_mpg_hpw'
           ,'len_wid','whb_c_wght','hpw_eng_size','hway_cit_mpg']

plt.figure(figsize=(20,10))
sns.boxplot(data=cars[num_var], 
                 palette="colorblind")
plt.title('Numerical features outlayers');


# Perfect, was we can see hpw_cylinders and another features have outlayes, let's remove then using IQR boxplot method

# In[ ]:


# creating a for to replace outlayers using boxplot method
for i in num_var:
    Q1 = cars[i].quantile(0.25)
    Q3 = cars[i].quantile(0.75)
    IQR = Q3 - Q1
    cars[i] = np.where(cars[i]>(Q3+1.5*IQR),(Q3+1.5*IQR),cars[i])
    cars[i] = np.where(cars[i]<(Q1-1.5*IQR),(Q1-1.5*IQR),cars[i])


# Verifying if outlayers was replaced

# In[ ]:


# Plotting boxplots to numeric features
num_var = ['mean_mpg','hpw_cylinders','mean_mpg_hpw'
           ,'len_wid','whb_c_wght','hpw_eng_size','hway_cit_mpg']

plt.figure(figsize=(20,10))
sns.boxplot(data=cars[num_var], 
                 palette="colorblind")
plt.title('Numerical features outlayers');


# Perfect! Now let's do a log transformation with the new features

# ###### Normalizing new features wigh log transformation

# In[ ]:


# taking numerical features to log trasformation
num_var = ['mean_mpg','hpw_cylinders','mean_mpg_hpw'
           ,'len_wid','whb_c_wght','hpw_eng_size','hway_cit_mpg'
          ,'make','drive-wheels','body-style','engine-type','fuel-system'
          ,'engine-location','fuel-type','aspiration','num-of-doors']

cars[num_var] = np.log1p(cars[num_var]) # log transformation


# ###### Removing high correlated features

# Now let's do a correlation analysis again to remove high correlated features

# In[ ]:


# Numerical variables correlation
cars_noprice = cars.drop('price',axis=1) # removing price column

corr = cars_noprice.corr() # creting the correlation matrix

plt.figure(figsize=(20,20)) # creating the and difyning figure size
ax = sns.heatmap( # plotting correlation matrix
    corr,vmin=-1, vmax=1, center=0,
    annot = True,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels( # adding axes values
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# We clearly have some features that are highly correlated with each other, now we have a great number of features, let's select and remove then using a high correlation matrix

# In[ ]:


# creating the correlation matrix
corr_matrix = cars_noprice.corr().abs()

# creating a mask to apply to our correlation matrix and filter high correlations
mask = np.triu(np.ones_like(corr_matrix,dtype=bool))

# replacing low correlations with NA's
tri_df = corr_matrix.mask(mask)

# selecting features to dropp that have correlation with each other above 0.80
to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.8)]

# dropping high correlated features
cars = cars.drop(to_drop,axis=1)


# Now let's see how this our new correlation matrix but first let's but our target 'price' in the last column

# In[ ]:


price = cars['price'] #saving prices
cars = cars.drop('price',axis=1) # dropping prices from cars dataset
cars['price'] = price # joing prices into our dataset again


# In[ ]:


# Numerical variables correlation
corr = cars.corr() # creting the correlation matrix

plt.figure(figsize=(20,20)) # creating the and difyning figure size
ax = sns.heatmap( # plotting correlation matrix
    corr,vmin=-1, vmax=1, center=0,
    annot = True,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels( # adding axes values
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# Now that we have removed high correlated features we have less features to process and we this we also can remove chances of overfitting

# Now let's put our target in the last column and go to modeling

# In[ ]:


price = cars['price'] #saving prices
cars = cars.drop('price',axis=1) # dropping prices from cars dataset
cars['price'] = price # joing prices into our dataset again


# ### Modeling

# First of all let's **split our dataset into train and test**, we need to do this do avoid overfitting in our model and evaluate our model performance, to do this we need to separaty and target from rest of features

# In[ ]:


# splitting the data with our target into y1 and the rest of data into x1
x1 = cars.drop('price', axis=1)
y1 = cars['price']


# In[ ]:


# splitting our dataset into train and target
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.7,random_state=42) # 70% train and 30% test


# ###### Model Selection

# As we can see in our exploratory analysis many of our features have a linear relation with our target.
# Thinking in that, let's create our first model using **Linear Regression**.

# ###### But how linear regression works?

# Linear regression consists of finding the best-fitting straight line through the points. The best-fitting line is called a regression line. The blue lineis  the regression line and consists of the predicted score on weight for each height. The vertical lines from the points to the regression line represent the errors of prediction. As you can see, the black points is very near the regression line; its error of prediction is small.
# 
# With a linear regression model we can use this regression line that is given by y= a + b * x to make predictions of new weights based on already knew height values.
# 
# In this case we have a simple linear regression, in our model i'll use a multiple linear regression(that have more then 1 variable in x axes) to predict car prices.

# <img src='https://i.imgur.com/PKSoO8R.png' style='width:600px;height:350px'/>

# In[ ]:


# training the model
regression = LinearRegression()
regression.fit(x_train1, y_train1)


# Now let's predict car prices in train dataset and latter in test dataset to verify model stabilitty

# In[ ]:


# predicting on train dataset
y_pred_train1 = regression.predict(x_train1)
y_pred_train1 = np.exp(y_pred_train1)


# In[ ]:


# predicting on test dataset
y_pred_test1 = regression.predict(x_test1)
y_pred_test1 = np.exp(y_pred_test1)


# In[ ]:


# removing log transformation from our target
y_train1_exped = np.exp(y_train1)
y_test1_exped = np.exp(y_test1)


# Now, to evaluate our model we need to use to matrics: Mean Absolute Error(MAE) and Root Mean Squared Error(RMSE), let's see this metrics in train and test dataset predictions

# In[ ]:


# model quality metrics in train dataset prediction
print('Mean Absolute Error:', metrics.mean_absolute_error(y_train1_exped, y_pred_train1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train1_exped, y_pred_train1)))


# Apparently a great result with MAE and RMSE, now let's see if our model was capable of generalize in test dataset

# In[ ]:


# model quality metrics in test dataset prediction
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test1_exped, y_pred_test1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test1_exped, y_pred_test1)))


# Not a great diference from train to test predictions, to be sure that our model are good let's look at the Mae Ratio

# ###### Evaluating our model with Mae Ratio metrics

# Mae Ratio is take the MAE percentage in relation to mean price of our test dataset, this metrics it's called MAE RATIO, with this metric we can measure how far our predictions was from the mean of our target, the lower our MAE RATIO is, better our model is, the best scenario it's a model with a MAE RATIO with less then 10.

# In[ ]:


print("The mean price of our test dataset is: ")
print(y_test1_exped.mean()) # calculating the mean of our car prices
print()
print("The MAE percentage in relation to mean price of our test dataset is: ")
print(round(metrics.mean_absolute_error(y_test1_exped, y_pred_test1)/y_test1_exped.mean()*100,2)) # calculating mae ratio


# ### Conclusion

# To answer our question in this notebook title, let's take a look into scatter plot below, most of our cars in our dataset cost between 5000 and 2000 dollars, only a feel cost between 20000 and 30000 dollars, our model seems to be more accuraty to predict prices of sheep cars, as we can see cars with values above 20000 are more scattered in our plot.

# After cleaning, explore and modelling our data we are finelly here, the conclusion! The model has a Mae Ratio of 15.26, not perfect but a good model, with a medium error and can be used to evaluate cars with a good accuracy, our target have not a good distribution and even after outlayers removal and log transformation still not good enough,certainly we can archieve a better results working more on feature engineering to create feature with more prediction power. Lastly we can also test another regression models like gradient boosting and try to improve results without overfitting.

# In[ ]:


# draw the figure to recieve a plot
plt.figure(figsize=(14,8))

# creating a scatter plot with seaborn
sns.scatterplot(x=y_pred_test1, y=y_test1_exped,hue=y_test1_exped);

