#!/usr/bin/env python
# coding: utf-8

# # Exploratory Analysis: 
# 
# This is a dataset containing consumer's thought and the star rating of car manufacturer/model/type. Currently, this dataset has data of 62 major brands. I am going to select one major major brand from each major region (North America, Japan and Europe). 
# 
# I would be doing descriptive analysis of the data and explore if there are any major anomalies across Car Manufactoring year, Reviewer or Rating.
# 
# To begin this exploratory analysis, first import libraries and define functions for plotting the data using matplotlib. Depending on the data, not all plots will be made.
# 
# I would be loading, cleaning and comparing following two analysis
# 
# 1) How a typical reviewer tends to review a American, European and Japanese car.
# 2) How the review varies for cars by Year Model for American, European and Japanese car.

# In[ ]:


#### Link to Dataset on Kaggle: https://www.kaggle.com/amitranjan01/edmunds-data-analysis-cross-continent-review
    
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import os # accessing directory structure
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
#path for kaggle notebook '/kaggle/input'
import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# List files available Only run on my local NoteBook
print(os.listdir("../input"))


# In[ ]:


#Analyze following three Cars from three different region.
#/kaggle/input/edmundsconsumer-car-ratings-and-reviews/Scrapped_Car_Review_Chevrolet.csv
#/kaggle/input/edmundsconsumer-car-ratings-and-reviews/Scraped_Car_Review_mercedes-benz.csv
#/kaggle/input/edmundsconsumer-car-ratings-and-reviews/Scrapped_Car_Reviews_Toyota.csv

# specify 'None' if want to read whole file
nRowsRead = 10000 

# Scrapped_Car_Reviews_Toyota.csv
df_J = pd.read_csv('/kaggle/input/edmundsconsumer-car-ratings-and-reviews/Scrapped_Car_Reviews_Toyota.csv', delimiter=',', nrows = nRowsRead)
#df_J = pd.read_csv('Scrapped_Car_Reviews_Toyota.csv', delimiter=',', nrows = nRowsRead)
df_J.dataframeName = 'Scrapped_Car_Reviews_Toyota.csv'
nRow, nCol = df_J.shape
df_J_orig = df_J.copy(deep=True)
print(f'There are {nRow} rows and {nCol} columns for Toyota')

# Scrapped_Car_Reviews_Toyota.csv
df_E = pd.read_csv('/kaggle/input/edmundsconsumer-car-ratings-and-reviews/Scraped_Car_Review_mercedes-benz.csv', delimiter=',', nrows = nRowsRead)
#df_E = pd.read_csv('Scraped_Car_Review_mercedes-benz.csv', delimiter=',', nrows = nRowsRead)
df_E.dataframeName = 'Scraped_Car_Review_mercedes-benz.csv'
df_E_orig = df_E.copy(deep=True)
nRow1, nCol1 = df_E.shape
print(f'There are {nRow1} rows and {nCol1} columns for Mercedes')

# Scrapped_Car_Reviews_Toyota.csv
df_A = pd.read_csv('/kaggle/input/edmundsconsumer-car-ratings-and-reviews/Scrapped_Car_Review_Chevrolet.csv', delimiter=',', nrows = nRowsRead)
#df_A = pd.read_csv('Scrapped_Car_Review_Chevrolet.csv', delimiter=',', nrows = nRowsRead)
df_A.dataframeName = 'Scrapped_Car_Review_Chevrolet.csv'
df_A_orig = df_A.copy(deep=True)
nRow2, nCol2 = df_A.shape
print(f'There are {nRow2} rows and {nCol2} columns for Chevrolet')


# In[ ]:


#Analyze the Columns for the datset

print("Columns for Japense Car: ", df_J.columns)
print("Columns for European Car: ", df_E.columns)
print("Columns for American Car: ", df_A.columns)


# In[ ]:


#Analyze the Columns for the datset

print("Info for Japense Car: ", df_J.info())
print("Info for European Car: ", df_E.info())
print("Info for American Car: ", df_A.info())


# Unnamed: Sequence number for the rows
# Review_Date: Datetime when review was given
# Author_Name: Author who gave review. User can give review unimaniously
# Vehicle_Title:
# Review_Title: 
# Review:
# Rating:

# In[ ]:


#Print head to verify the data
print("Head for Japense Car: ", df_J.head())
print("Head for European Car: ", df_E.head())
print("Head for American Car: ", df_A.head())


# #Based on the data displayed below some analysis of data is
# 1) Column: "Unnamed: 0": Is dulicate to index, so we can drop that column
# 2) Column: Review_Date is of Object Type and has some extra character around date information, so it needs transformation
# 3) Column Vehicle_Title: This column includes the car Model year, rest information is not important for my anlysis.
# 4) Column: Rating is a float, for my analysis, I am intrested in distribution of the data, so will need o convert it to int.
# 5) Column: Review_Title Not needed for my analysis
# 6) Column: Review Not needed for my analysis
# 
# As next step we will analyze the data types of column.

# In[ ]:


print("Data Types for Japense Car: ", df_J.dtypes)
print("Data Types for European Car: ", df_E.dtypes)
print("Data Types for American Car: ", df_A.dtypes)


# In[ ]:


#Extract the Vechile Make Year from the Vechile Title

df_J['Make_Year'] = df_J['Vehicle_Title'].str[:4]
df_J['Make_Year'] = df_J['Make_Year'].fillna(method ='ffill')
print(df_J['Make_Year'])  

df_E['Make_Year'] = df_E['Vehicle_Title'].str[:4]
df_E['Make_Year'] = df_E['Make_Year'].fillna(method ='ffill')

print(df_E['Make_Year'])  

df_A['Make_Year'] = df_A['Vehicle_Title'].str[:4]
df_A['Make_Year'] = df_A['Make_Year'].fillna(method ='ffill')

print(df_A['Make_Year'])  


# In[ ]:


# Extract the Review Date from Review
df_J['Review_Date_D'] = df_J['Review_Date'].str[4:12]
df_J['Review_Date_D'] = df_J['Review_Date_D'].fillna(method ='ffill')
df_J['Review_Date_D'] = pd.to_datetime(df_J['Review_Date_D'], errors='coerce')
print(df_J['Review_Date_D'])  

df_E['Review_Date_D'] = df_E['Review_Date'].str[4:13]
df_E['Review_Date_D'] = df_E['Review_Date_D'].fillna(method ='ffill')
df_E['Review_Date_D'] = pd.to_datetime(df_E['Review_Date_D'],  errors='coerce')
print(df_E['Review_Date_D']) 

df_A['Review_Date_D'] = df_A['Review_Date'].str[4:13]
df_A['Review_Date_D'] = df_A['Review_Date_D'].fillna(method ='ffill')
df_A['Review_Date_D'] = pd.to_datetime(df_A['Review_Date_D'], errors='coerce')
print(df_A['Review_Date_D']) 


# Drop Following Columns: They are not Stattistical Important for my Analysis of Rating by Make year or Made in Country
# * Unnamed: 0
# * Make Year
# * Review_Date
# * Vehicle_Title
# * Review_Title
# * Review

# In[ ]:





# In[ ]:


#Drop data not required for my analysis. In Place update to exisiting Dataframes
df_E.drop(['Unnamed: 0', 'Review_Date', 'Vehicle_Title', 'Review_Title', 'Review'], axis=1, inplace=True)
df_A.drop(['Unnamed: 0', 'Review_Date', 'Vehicle_Title', 'Review_Title', 'Review'], axis=1, inplace=True)
df_J.drop(['Unnamed: 0', 'Review_Date', 'Vehicle_Title', 'Review_Title', 'Review'], axis=1, inplace=True)


# In[ ]:


print(df_E.info())
print("99th %tile: ", df_E["Rating"].quantile(0.99))
print(df_E.describe())


# In[ ]:


#Replace the missig entries in Rating column with Mean of Rating
df_E['Rating'].fillna(df_E['Rating'].mean(), inplace=True)
df_A['Rating'].fillna(df_E['Rating'].mean(), inplace=True)
df_J['Rating'].fillna(df_E['Rating'].mean(), inplace=True)


# In[ ]:


#Replace the missing authoer with anonymous as author
df_E['Author_Name'].fillna('anonymous', inplace=True)
df_A['Author_Name'].fillna('anonymous', inplace=True)
df_J['Author_Name'].fillna('anonymous', inplace=True)


# In[ ]:


#Add a column representing region where the car belongs. Required to know which data came from which dataframe
df_E['Origin_Region'] = "North America"
df_A['Origin_Region'] = "Eurpoe"
df_J['Origin_Region'] = "Japan"


# In[ ]:


#After replacing the mean,make sure that the rating is not impacted, we can see 99th %tile:  4.875 is unchanged.
print(df_E.info())
print("99th %tile: ", df_E["Rating"].quantile(0.99))
print(df_E.describe())


# In[ ]:


print(df_A.info())
print("99th %tile: ", df_A["Rating"].quantile(0.99))
print(df_A.describe())


# In[ ]:


print(df_J.info())
print("99th %tile: ", df_J["Rating"].quantile(0.99))
print(df_J.describe())


# In[ ]:


#Append all three dataset
df = df_E.append(df_A).append(df_J)
print(df.shape)
print(df.head(10))
print(df.info())


# In[ ]:


# convert the floating Rating to Integer
df['Rating'] = df['Rating'].round(0)
df.head()


# In[ ]:


#Plot the Rating to see how the data is distributed
df['Rating'].plot(kind = 'hist', bins = 100)
plt.show()
#Below Data shows that 5 rating is so frequent that other ratings are not vsisble on graph, 
#so chaging the scale of the graph


# In[ ]:


#Plot the Rating to see how the data is distributed - Using Log Scale
df['Rating'].plot(kind = 'hist', bins = 100)
plt.yscale('log')
plt.show()


# In[ ]:


df[df['Origin_Region'] == "North America"]['Rating'].plot(kind = 'hist')
plt.yscale('log')
plt.show()


# In[ ]:


df[df['Origin_Region'] == "Eurpoe"]['Rating'].plot(kind = 'hist')
plt.yscale('log')
plt.show()


# In[ ]:


df[df['Origin_Region'] == "Japan"]['Rating'].plot(kind = 'hist')
plt.yscale('log')
plt.show()


# Based on above analysis, we can conclude
# 1) Most of teh Cars are rated 5 across all three regions. Eurpoe and American cars have 99.9% rating of 4.5-4.8 while Japense cars of 99.9 % have rating of 5.
# 2) 

# In[ ]:


plt.figure(figsize=(20,10))
plt.scatter(df['Make_Year'], df['Rating'] )
plt.show()
#Scatter plot doesn't relveal any specific data aspect.


# In[ ]:


df['Rating'].groupby(df['Author_Name']).count().nlargest(10)
#Anomalies found, top three revewer has reviewed more than 2000 reviews each. 
#Where as the avarage review per reviewe is just 4
#If we drop top reviewes, we will be left with very small set of data. So we will analyze the data in different section.


# In[ ]:


#Create a Backup copy of data so that we can process it and analyze it later.
df_copy = df.copy(deep=True)


# Above Analysis shows that almost 21,000 reviews were submitted anonymously and top three revieweres submitted too many review.

# In[ ]:


df.drop(df.loc[df['Author_Name']=='anonymous'].index, inplace=True)
df.drop(df.loc[df['Author_Name']=='HD mike'].index, inplace=True)
df.drop(df.loc[df['Author_Name']=='Dave761'].index, inplace=True)
df.drop(df.loc[df['Author_Name']=='Avalon Driver'].index, inplace=True)


# In[ ]:


#Total dataset is reduced form 30,000 to 348
df.shape


# In[ ]:


df['Rating'].groupby(df['Author_Name']).count().nlargest(10)


# In[ ]:


q = df["Rating"].quantile(0.99)
print(q)
df[df["Rating"] < q].count()
#Total of 84 rating are other than 5. Which is about 20% of total left data


# In[ ]:


df['Rating'].groupby(df['Author_Name']).describe()


# In[ ]:


df_copy['Rating'].groupby(df_copy['Author_Name']).count().nlargest(10)


# In[ ]:


#Plot the Rating to see how the data is distributed - Using Log Scale
df['Rating'].plot(kind = 'hist', bins = 100)
plt.yscale('log')
plt.show()
# The new rating distribution seems very similar to what was in original dataframe.


# In[ ]:


#Now lets analyze the top reviewers rating pattern
print(df_copy[df_copy['Author_Name'] == "Avalon Driver "]['Origin_Region'].count())
df_copy[df_copy['Author_Name'] == "Avalon Driver "]['Origin_Region'].value_counts()
#Seems like Avalon Driver has only reviewd Japanese car and have given all rating of 5


# In[ ]:


#Now lets analyze the top reviewers rating pattern
print(df_copy[df_copy['Author_Name'] == "Avalon Driver "]['Rating'].count())
df_copy[df_copy['Author_Name'] == "Avalon Driver "]['Rating'].value_counts()
#Seems like Avalon Driver has only reviewd Japanese car and have given all rating of 5


# In[ ]:


#Now lets analyze the top reviewers rating pattern
print(df_copy[df_copy['Author_Name'] == "Avalon Driver "]['Make_Year'].count())
df_copy[df_copy['Author_Name'] == "Avalon Driver "]['Make_Year'].value_counts()
#Seems like Avalon Driver has given all reviews in one year whcih is 2002


# In[ ]:


#Now lets analyze the top reviewers rating pattern
print(df_copy[df_copy['Author_Name'] == "Dave761 "]['Origin_Region'].count())
df_copy[df_copy['Author_Name'] == "Dave761 "]['Origin_Region'].value_counts()
#Seems like Avalon Driver has only reviewd Japanese car and have given all rating of 5


# In[ ]:


#Now lets analyze the top reviewers rating pattern
print(df_copy[df_copy['Author_Name'] == "Dave761 "]['Rating'].count())
df_copy[df_copy['Author_Name'] == "Dave761 "]['Rating'].value_counts()
#Seems like Avalon Driver has only reviewd Japanese car and have given all rating of 5


# In[ ]:


#Now lets analyze the top reviewers rating pattern
print(df_copy[df_copy['Author_Name'] == "Dave761 "]['Make_Year'].count())
df_copy[df_copy['Author_Name'] == "Dave761 "]['Make_Year'].value_counts()
#Seems like Avalon Driver has given all reviews in one year whcih is 2002


# In[ ]:





# In[ ]:


#ax = df_copy.plot(x="Make_Year", y="Rating", kind="bar")
#df_copy.plot(x="Make_Year", y="Rating", kind="bar", ax=ax, color="C2")
#plt.show()


# Based on Analysis so far, we have only discovered few intresting facts about data. Now we are going back to original data set and take a different approach. This time, we will
# 1) Not change rating from fraction to integer
# 2) Not fill the missing year to Make Year
# 3) See how much data is there after dropping and is it enough for analysis. We will avoid any filling.

# In[ ]:


#Load Original Data (unmodified)
print(df_A_orig.info())
print(df_J_orig.info())
print(df_E_orig.info())


# In[ ]:


df_A_orig1 = df_A_orig.copy(deep=True)
df_A_orig1.dropna(subset = ['Rating'], inplace=True)
print(df_A_orig1.head())
print(df_A_orig1.shape)
df_A_orig1.info()
#If I drop all NA, then I am only left with 51 rows of data out of 10,000 rows. So that is not best option. 
#Lets take another approach


# In[ ]:


df_E_orig1 = df_E_orig.copy(deep=True)
df_E_orig1.dropna(subset = ['Rating'], inplace=True)
print(df_E_orig1.head())
print(df_E_orig1.shape)
df_E_orig1.info()
#If I drop all NA, then I am only left with 51 rows of data out of 10,000 rows. So that is not best option. 
#Lets take another approach


# In[ ]:


df_J_orig1 = df_J_orig.copy(deep=True)
df_J_orig1.dropna(subset = ['Rating'], inplace=True)
print(df_J_orig1.head())
print(df_J_orig1.shape)
df_J_orig1.info()
#If I drop all NA, then I am only left with 51 rows of data out of 10,000 rows. So that is not best option. 
#Lets take another approach


# In[ ]:


#Create new column for Country Of Origin
#Add a column representing region where the car belongs. Required to know which data came from which dataframe
df_E_orig1['Origin_Region'] = "North America"
df_A_orig1['Origin_Region'] = "Eurpoe"
df_J_orig1['Origin_Region'] = "Japan"


# In[ ]:


# Extract the Review Date from Review
df_J_orig1['Review_Date_D'] = df_J_orig1['Review_Date'].str[4:12]
#df_J_orig1['Review_Date_D'] = df_J_orig1['Review_Date_D'].fillna(method ='ffill')
df_J_orig1['Review_Date_D'] = pd.to_datetime(df_J_orig1['Review_Date_D'], errors='coerce')
print(df_J_orig1['Review_Date_D'])  

df_E_orig1['Review_Date_D'] = df_E_orig1['Review_Date'].str[4:13]
#df_E_orig1['Review_Date_D'] = df_E_orig1['Review_Date_D'].fillna(method ='ffill')
df_E_orig1['Review_Date_D'] = pd.to_datetime(df_E_orig1['Review_Date_D'],  errors='coerce')
print(df_E_orig1['Review_Date_D']) 

df_A_orig1['Review_Date_D'] = df_A_orig1['Review_Date'].str[4:13]
#df_A_orig1['Review_Date_D'] = df_A_orig1['Review_Date_D'].fillna(method ='ffill')
df_A_orig1['Review_Date_D'] = pd.to_datetime(df_A_orig1['Review_Date_D'], errors='coerce')
print(df_A_orig1['Review_Date_D']) 


# In[ ]:


#Extract the Vechile Make Year from the Vechile Title

df_J_orig1['Make_Year'] = df_J_orig1['Vehicle_Title'].str[:4]
#df_J_orig1['Make_Year'] = df_J_orig1['Make_Year'].fillna(method ='ffill')
print(df_J_orig1['Make_Year'])  

df_E_orig1['Make_Year'] = df_E_orig1['Vehicle_Title'].str[:4]
#df_E_orig1['Make_Year'] = df_E_orig1['Make_Year'].fillna(method ='ffill')
print(df_E_orig1['Make_Year'])  

df_A_orig1['Make_Year'] = df_A_orig1['Vehicle_Title'].str[:4]
#df_A_orig1['Make_Year'] = df_A_orig1['Make_Year'].fillna(method ='ffill')

print(df_A_orig1['Make_Year'])  


# In[ ]:


df_tot = df_J_orig1.append(df_A_orig1).append(df_E_orig1)
df_tot.info()


# In[ ]:





# In[ ]:


#Plot the Rating to see how the data is distributed - Using Log Scale
df_tot['Rating'].plot(kind = 'hist', bins = 15)
plt.yscale('log')
plt.show()
# The new rating distribution seems more like a continous distribution.


# In[ ]:


df_tot['Rating'].groupby(df_tot['Author_Name']).count().nlargest(10)
#even authors have good reviews distribution with more even review frqquency


# In[ ]:


df_tot.info()


# In[ ]:


df_tot_new = df_tot[['Review_Date_D','Author_Name','Rating','Make_Year','Origin_Region']]


# In[ ]:


df_tot_new.info()


# In[ ]:


grp = df_tot_new.groupby(['Author_Name','Origin_Region','Make_Year'])['Author_Name','Make_Year','Origin_Region','Rating','Review_Date_D']


# In[ ]:


grp.describe(include='all')


# In[ ]:


df_tot_new['Make_Year'] =  pd.to_datetime(df_tot_new['Make_Year'], errors='coerce')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(8,12))
#df_tot_new = df_tot[['Review_Date_D','Author_Name','Rating','Make_Year','Origin_Region']]

tw = df_tot_new[df_tot_new['Origin_Region'] == 'Japan']['Make_Year']
tw.sort_values()
tm = df_tot_new[df_tot_new['Origin_Region'] == 'Japan']['Rating']

sw = df_tot_new[df_tot_new['Origin_Region'] == 'Eurpoe']['Make_Year']
sm = df_tot_new[df_tot_new['Origin_Region'] == 'Eurpoe']['Rating']
sw.sort_values()

kw = df_tot_new[df_tot_new['Origin_Region'] == 'North America']['Make_Year']
km = df_tot_new[df_tot_new['Origin_Region'] == 'North America']['Rating']
kw.sort_values()

plt.subplot(3, 1, 1)
plt.scatter(tw, tm)
plt.xlabel('Model Year')
plt.ylabel('Rating')
plt.title('Japan car rating over year')
plt.grid(True)


plt.subplot(3, 1, 2)

plt.scatter(sw, sm)
plt.xlabel('Model Year')
plt.ylabel('Rating')
plt.title('Eurpoe car rating over year')
plt.grid(True)

plt.subplot(3, 1, 3)

plt.scatter(kw, km)
plt.xlabel('Model Year')
plt.ylabel('Rating')
plt.title('United States car rating over year')
plt.grid(True)


plt.tight_layout()
plt.show()


# In[ ]:


sm.plot()
ticks,labels = plt.xticks()


# In[ ]:


plt.figure(figsize=(14,8))

X = tw
X1 =sw 
X2 =kw 
Y = tm
Y1 = sm
Y2 = km
plt.scatter(X,Y,  marker = '^', color = 'Green', label ='Japan')
plt.scatter(X1,Y1,  marker = '>', color = 'Red', label ='Eurpoe')
plt.scatter(X2,Y2,  marker = '<', color = 'Blue', label ='North America',)
plt.xlabel('Make Year')
plt.ylabel('Rating')
plt.legend(loc='best')
plt.title('Relationship Between Car make Year and Rating for cars')
plt.ylim(0,6)
plt.show()


# This is a logical end to my descriptive analysis. A further extension of this analysis can be forensic analysis or creating a model which can predict the Rating of a given car based on Make year, Model / Country of origin and may be we can extract more information about car features like Engine capcity etc. So far what we have learnt about the dataset.
# 
# 1) Data has many columns, big chunk of the information in the dataset is in form of descriptive set. Which makes it a good candidates for sentiment analysis.
# 2) There is enough data about 5% of total data where we have enough information to look into ratings and explore how rating were awarded, what things influenced the rating like Where car technology originated e.g. Asia (japan), Europe or USA.
# 3) The Rating dataset (subset of data cleaned up for Rating analysis) has some anolmolies 
#     a) The Rating is not evenly distributed (not a normal distribution). It's negatively skewed
#     b) Lot of review were provided anonymously, so making it difficult to identify reviewer pattern.
#     c) Most of reviewers have reviewed same car for multiple times (1-5), so we cannot predict the reviewer bias about a given car or any comparison for same reviewer reviewing different cars. Which kind of make sense that user in japan would not have multiple cars to use and provide review.
#     d) There were few reviewer anomalies where in one year few reviewers have reviewed 2000-3000 review for same car and all 5 rating.
# 4) Rating density distribution increases from 4 - 5.
# 5) Different cars across region has consistent rating across the decade. Average rating remained same.
# 6) Japan cars have slightly higher 99th percentile (5.0) rating and European / American (4.5 - 4.8)
# 
