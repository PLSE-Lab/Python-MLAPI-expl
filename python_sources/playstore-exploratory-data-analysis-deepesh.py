#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing the requiered packages
import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
import seaborn as sns


# In[ ]:


# reading the playstore CSV and user reviews CSV into a pandas dataframe
data = pd.read_csv('../input/googleplaystore.csv')
print(data.columns)
data.head(3)


# **Understanding the Columns**
# * App - Name of the app
# * Category - Category to which the app belongs
# * Rating - Consolidated User rating
# * Reviews - Number of User reviews
# * Size - Size of the app in MB
# * Installs - Number Installs
# * Type - Free or paid 
# * Price - If the app is a paid app, then there is price in USD else if the app is free this column is 0
# * Content Rating - Age restiction for the app if any
# * Genres - It is an extended version for Category, it can have sub categories too
# * LAst Updates - Date of the latest update
# * Current Ver - Current latest version name/Number
# * Andriod Ver - Minumum Android Version required to install the app.

# **Removing the NULL Values**
# > Lets check for the columns that  have NULL values in them. Removing this will make our life much easier and it will help us in getting the workflow smooth and fluidic.

# In[ ]:


# Lets see how many null values are there in each of the colums
null_values = {}
for i in data.columns:
    null_values[i] = len(data[pd.isnull(data[i])])
print(null_values)


# > **Apart from "data.rating" there aren't much Nan/NULL values in the dataset. lets see the distribution of the apps based on the category**

# In[ ]:


test = data.groupby('Category')['App']
test = pd.DataFrame(test.size().reset_index(name = "Count"))
test.sort_values(by = 'Count',axis=0,ascending=False,inplace=True)

#plotting the top 5 categories on the appstore
plt.figure(figsize=(15,9))
sns.barplot(x=test.Category[:8],y=test.Count[:8],data=test)
plt.show()


# **We can see that the apps in category FAMILY has more number of app in the data. Lets see if the sentiment of the reviews corelate with the installs**

# In[ ]:


#importing playstore_user review data. This data as sentiment for each each of the use reviews for apps
review_data = pd.read_csv('../input/googleplaystore_user_reviews.csv')
review_data = review_data.groupby('App').mean()
review_data.reset_index(inplace=True)
review_data.head()


# In[ ]:


# Merging the sentiment_polarity and sentiment_subjectivity data with the data
final_data = pd.merge(data,review_data,how = 'inner',on='App')

# converting the Installs columns to integer
Installs_array = []
for i in final_data.Installs:
    Installs_array.append(int(i[:-1].replace(",","")))
final_data['Installs'] = np.array(Installs_array)

#removing NaN values from Sentiment_Polarity
final_data = final_data[~pd.isnull(final_data['Sentiment_Polarity'])]
final_data.sort_values(by = ['Installs','Sentiment_Polarity'],ascending=False,inplace=True)
final_data.Sentiment_Polarity = np.round(final_data.Sentiment_Polarity,decimals=1)

#Number of Apps
temp = final_data[['Installs','Sentiment_Polarity']]
temp = temp.groupby('Sentiment_Polarity',as_index=True).count()
temp.reset_index(inplace=True)

# Volumes of Installs
temp_1 = final_data[['Installs','Sentiment_Polarity']]
temp_1 = temp_1.groupby('Sentiment_Polarity',as_index=True).sum()
temp_1.reset_index(inplace=True)

#merging the two dataframes
temp = pd.merge(temp,temp_1,on='Sentiment_Polarity',how='inner')
temp.head()

#Plotting the distribution
f,ax = plt.subplots(figsize=(15,9))

sns.set_color_codes("pastel")
sns.barplot(x=temp.Sentiment_Polarity,y=temp.Installs_x,color='b',label = 'Total No of Apps')

sns.set_color_codes("muted")
sns.barplot(x=temp.Sentiment_Polarity,y=np.log(temp.Installs_y),color="b",label='Sum of Installs')
ax.legend(ncol=2, loc="upper right", frameon=True)
sns.despine(left=True,bottom=True)
plt.show()


# **Based on the above plot we can look at the following Observations**
# * The Sentiment for the reviews for many apps are between the range -0.2 to 0.6
# * The apps where the highest number of installs happend are at a sentiment_polarity of 0.2
# * We can aslo see that the apps at the extreme ends with a sentiment_polarity of -0.5 and apps >0.6 have very less number of installs compared to the reviews. or either the review data is not complete

# **Let's see distiribution of apps based on the app rating**

# In[ ]:


rating_data = data[['Rating','App']]
rating_data['Rating_1'] = np.round(rating_data['Rating'],decimals=0)
rating_data = rating_data.groupby("Rating").count()
rating_data = rating_data.reset_index()
rating_data.sort_values(by = 'App',inplace=True,ascending=False)

plt.figure(figsize=(15,9))
sns.barplot(x="Rating",y="App",data=rating_data,hue="Rating_1",x_bins=50)
# sns.scatterplot()
plt.show()


# In[ ]:




