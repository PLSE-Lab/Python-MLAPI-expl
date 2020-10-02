#!/usr/bin/env python
# coding: utf-8

# # **Objective:**
# Our goal in this notebook is to explore the data provided in Zomato csv and to analyze which restaurants have poor ratings in Zomato and why?

# In[ ]:


import pandas as pd                                 # Importing pandas
import numpy as np                                  # Importing numpy
import matplotlib.pyplot as plt                     # Importing matplotlib for visualization

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns                               # Importing seaborn for visualization

import matplotlib.ticker as ticker
import matplotlib.cm as cm

import pandas_profiling                             # AUtomatic EDA

import ipywidgets as widgets                        # Creating widgets
from IPython.display import display                 # Displaying widgets


# # **Loading the datasets**
# The csv file has encoding ISO-8859-1. Encoding defines what kind of characters(ASCII or non ASCII or something different) can be stored in a file.

# In[ ]:


zomato_df = pd.read_csv('../input/zomato-restaurants-data/zomato.csv',encoding = "ISO-8859-1")    # Reading file zomato.csv
#"utf-8"


# # **Pandas Profiling**

# In[ ]:


import pandas_profiling  
report = pandas_profiling.ProfileReport(zomato_df)                # Perform pandas profiling on the dataset    

report.to_file("zomato_data.html")                                # covert profile report as html file


# In[ ]:


#Our observation oo Profiling 
  # Remove Resto ID --> Everything is unique
  # Resturant Name --> It may be important, if we are predicting rating since some of them might be chains and brands
        #You need to add one more column that adds if the name is a brand or not
        #Column with two catagory Band or non Brand
        #Add a column that says a name is catchy or not catchy
        #Cusine, Votes, other variable except name --> Can i predict rating.
            #There are new resturants which are added and we need to predict their rating for next 12 months
  #Country code
        #Just to analysis on India data which is major
        #May be seperate India and other country
        #only do analysis on outside India Data
        # If the problem is to predict ratings ; maybe sepreating India from other country may make sense. 
        # since it is a discrete column, we need to categorize it.

  # City Column 
      # Assuming we are only taking India 
        # Divide the data into North, Central, East, West South 
        # Take Delhi NCR, OtherIndianCities 
        # Tier1 Cities and Tier2 cities 
      # All Country Data 
        # Take Big Cities in INdia vs Big Cities outside 

 # Address Column , Locality Column , Locality Verbose
    # Unless we are only studying New Delhi, there's no pint of taking this column 
        # You will extract Localities - GK, Sectors etc 

 # Lat & Long -> ideally Ignore - Too many Uniques 
    # YOu can take it if you create nearby zones (range)
    # we can create category like; Posh,Happening,Corporate Areas,MiddleClass, Student,Residential?
    # Ideally, we don't know these datapoints, you have to lookout for these datapoints eitehr from your Ops Team or maybe from Internet 

  # Cusines
    # we need to take it 
    # see how you can address the distributional issues of cusines 
        # we can compeup with higher categorization of cusines 
            # <chinese, japense, thai> -> Asian Food 
    # Can we combine cuisine with locality - might give you prominant culture of food habits 

  # avergae Cost of 2
    # assuming you are studying all countries 
      # Normalize the whole column into single currency 
    # if you studying only India, 
      # then no need of normalization 

# What subset of data is making sense based on initial analysis 
    # Country == India ; Curerncy == INR 
    # For India - You will have individual model and analysis 
    # For other country, unless you have good data - it won't make sense 
        # Models might not be good 
        # You can just do Simple EDA on those data 
        # You need to address the Bias issue in the data 

  # Has Table booking any effect on Ratings 
      # If a rest has a booking facility -- is there any correlation with Rating 

  # Has Online booking any effect on Ratings 
      # If a rest has a Online Order facility -- is there any correlation with Rating 

  # If you have categories -- ideal condition is to have good % distribution (equal)

  # Delivering Now -- Ignore 

  # If all the values are same or if all the values are ditinct --  you ignore/remove them 

  # Any column that needs to be predicted should be ideally normal (close)

  # So, for the Rating -> Need to check who are these 0 rating restro 
      # If these are restro's which are not having any votes 
       # If that is the case, we need to ingnore 0 rating restro since they have not been rated
       # but if they have been rated and still having zero rating, we need to consider them in analysis 

  # we don't need to take rating colour and rating text - both ; since they are the same data 
    # let's just take rating text 

  # Votes - -need to check 11% zero vote restr's - becuase they could be new restru's 


# In[ ]:


zomato_df.head()  


# In[ ]:


country_df = pd.read_excel('../input/zomato-restaurants-data/Country-Code.xlsx')


# In[ ]:


country_df.head()


# In[ ]:


import pandas as pd
pd.merge # --> this function helps us to merge two data set where one of the column is common


# In[ ]:


zomato_data=pd.merge(zomato_df,country_df, on='Country Code')
zomato_data.head()


# ### Interactive Exploration of Data

# In[ ]:


# Function to extract all the unique values
# A column will have multiple values which can be repetative. We are extracting unique values and sorting them.

ALL='ALL'
def unique_sorted_values_plus_ALL(array):
    unique = array.unique().tolist()
    unique.sort()
    unique.insert(0, ALL)
    return unique


# In[ ]:


output = widgets.Output()

# Creating 3 dropdowns

# Country dropdown
dropdown_country = widgets.Dropdown(options = unique_sorted_values_plus_ALL(zomato_data['Country']), description='Country')

# City dropdown
dropdown_city = widgets.Dropdown(options = unique_sorted_values_plus_ALL(zomato_data['City']), description='City')

# Rating dropdown
dropdown_rating = widgets.Dropdown(options = unique_sorted_values_plus_ALL(zomato_data['Rating text']), description='Rating')

# Define a function for common_filtering
# Here we will define what output we can get for different dropdown options
def common_filtering(country, city, rating):
    output.clear_output()
    
    if (country == ALL) & (city == ALL) & (rating == ALL):
        common_filter = zomato_data
    elif(city == ALL) & (rating == ALL):
        common_filter = zomato_data[(zomato_data['Country'] == country)]
    elif(country == ALL) & (rating == ALL):
        common_filter = zomato_data[(zomato_data['City'] == city)]
    elif(country == ALL) & (city == ALL):
        common_filter = zomato_data[(zomato_data['Rating text'] == rating)]
    elif (country == ALL):
        common_filter = zomato_data[(zomato_data['City'] == city) & (zomato_data['Rating text'] == rating)]
    elif (city == ALL):
        common_filter = zomato_data[(zomato_data['Country'] == country) & (zomato_data['Rating text'] == rating)]
    elif (rating == ALL):
        common_filter = zomato_data[(zomato_data['City'] == city) & (zomato_data['Country'] == country)]
    
                                    
    else:
        common_filter = zomato_data[(zomato_data['City'] == city) & 
                                  (zomato_data['Country'] == country) &
                                 (zomato_data['Rating text'] == rating)]
    
    with output:
        display(common_filter)

# Define the event handler

# Country dropdown event handler
def dropdown_country_eventhandler(change):                         
    common_filtering(change.new, dropdown_city.value, dropdown_rating.value)  # Here we need to accept country event change
# City dropdown event handler
def dropdown_city_eventhandler(change):
    common_filtering(dropdown_country.value, change.new, dropdown_rating.value) # Here we need to accept city event change
# Rating Text dropdown event handler
def dropdown_rating_eventhandler(change):
    common_filtering(dropdown_country.value, dropdown_city.value, change.new)  # Here we need to accept rating event change
  

# Bind the event handler to the drop downs

dropdown_country.observe(dropdown_country_eventhandler, names='value')
dropdown_city.observe(dropdown_city_eventhandler, names='value')
dropdown_rating.observe(dropdown_rating_eventhandler, names='value')

# Display the dropdowns
input_widgets = widgets.HBox([dropdown_country, dropdown_city, dropdown_rating])
display(input_widgets)


# In[ ]:


display(output)


# ### Data Exploration
# 
# **Description of the numeric columns of dataset**

# In[ ]:


zomato_data[['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']].describe()


# **Observation:**
# There are 9551 rows in all column and there are no missing data in numeric columns

# **Average cost of two**
# Standard deviation is very high on Average cost for two. It is for different countries having different currency.
# There are 15 countries in the data set currencies are not standardized.
# 
# Max value for avarge cost of two is 800000

# In[ ]:


pip install currencyconverter


# In[ ]:


zomato_data['Currency']


# In[ ]:


zomato_data['Currency'].str.split('(', expand=True)[0]


# In[ ]:


zomato_data['Currency_new'] = zomato_data['Currency'].str.split('(', expand=True)[0]


# In[ ]:


zomato_data['Currency_new_code'] = zomato_data['Currency'].str.split('(', expand=True)[1].str.split(')', expand=True)[0]


# In[ ]:


from currency_converter import CurrencyConverter
c = CurrencyConverter()
c.convert(800000,'IDR','INR')


# In[ ]:


currency_code=pd.read_csv('https://raw.githubusercontent.com/datasets/currency-codes/master/data/codes-all.csv')


# In[ ]:


currency_code.head()


# In[ ]:


from currency_converter import CurrencyConverter
c = CurrencyConverter()
c.convert(800000, currency_code [currency_code['Currency']=='Turkish Lira']['AlphabeticCode'].iloc[0],'INR')


# In[ ]:


zomato_data.info()


# # **Obervation:**
# There is no missing values in any column except in Cuisines. 
# Most of the data types are object i.e. they are categorial 

# **Missing value imputation**
# 

# In[ ]:


#filling missing values in Cuisines

#Missing Value
    # If its a category
    # Mode of the column
    # Fill it with some standard value

zomato_data['Cuisines'].fillna("Other",inplace=True)


# In[ ]:


zomato_data['Aggregate rating'].mean()


# In[ ]:


zomato_data[zomato_data['Aggregate rating']>0]['Aggregate rating'].mean()


# In[ ]:


#Filling any missing value
zomato_data['Aggregate rating'].fillna(zomato_data['Aggregate rating'].mean(),inplace=True)


# In[ ]:


zomato_data.shape


# In[ ]:


zomato_data.info()


# **Correlation plot of Numeric columns**

# In[ ]:


correlation = zomato_data[['Average Cost for two','Price range','Aggregate rating','Votes']].corr()


# In[ ]:


correlation


# In[ ]:


import seaborn as sns
# sns.heatmap ---> used to ideally plot correlations 


# In[ ]:


sns.heatmap(correlation,vmin= -1,vmax=1 )


# # **Observation**
# There is a weak correlation between Aggregate rating and Price range
# Other then this there is no correlation
# 
# 
# If the Price range is higher, people should rate the restaurant higher.

# # **Distribution of aggregate rating**

# In[ ]:


sns.distplot(zomato_data['Aggregate rating'])


# **Observation**:  <br>
# A lot of restaurants are rated 0. After this most of the restaurants have been rated between 3 and 4.

# Below lot shows rating after removing 0 rated entries

# In[ ]:


sns.distplot(zomato_data[zomato_data['Aggregate rating']>0]['Aggregate rating'])


# ### Bivariate Analysis
# 
# Here we check the relationship between two variables.

# In[ ]:


# Relationship between aggregate rating and votes

sns.scatterplot(x=zomato_data['Aggregate rating'], y=zomato_data['Votes'])


#  **Obervation**
#  
#  
#  From the above scatter plot we can see that As the quality of food increases, with as the aggregate rating increases, no of votes also increases

# In[ ]:


# Relationship between Aggregate Ratings and Votes

sns.lineplot(x=zomato_data['Aggregate rating'], y= zomato_data['Votes'])


# In[ ]:


# Relationship between Aggregate Ratings and Votes

sns.lineplot(x=zomato_data['Votes'], y= zomato_data['Aggregate rating'])


# **Obeservation** : Here we can see the same inference. Aggregate Ratings and Votes have an increasing trend.

# In[ ]:


# Relationship between Price range and Aggregate Ratings

sns.violinplot(x='Price range', y='Aggregate rating', data = zomato_data)


# **Observation 1** <br>
# Here we can clearly see that with increase in Price range, the median of ratings also increase.

# **Question: Which countries have the highest number of restaurants in Zomato?**

# In[ ]:


"""
The background is whitegrid
The figure size is 14*6 
The x-axis labels are written with a rotation of 45 degree
setting the title to "# of Restaurants registered in Zomato in different Countries 
"""

sns.set_style('whitegrid')
plt.figure(figsize = (14,6))
sns.countplot(x= 'Country', data=zomato_data)
plt.xticks(rotation=45)
plt.title("# of Restaurants registered in Zomato in different Countries ");


# **Observation**
# 
# 
# India has maximum resturents : 8652
# 
# 
# Rest of the world : 9551

# Zomato India:
# 
# Our goal is to determine which resturent has the poor rating in Zomato and why ?
# 

# The number of restaurants registered in Zomato is highest in India.
# So lets look at the data of these restaurants.

# In[ ]:


zomato_india = zomato_data[zomato_data['Country']=='India']


# In[ ]:


zomato_india.head()


# **We have aggregate ratings and Rating text as two column of interest.**

# In[ ]:


zomato_india.groupby('Rating text').mean()


# **Excellent and Very Good** food ratings are provided in restaurants which are slightly premium cost and high price range.
# They also have huge number of votes. This can be due to high quality food or ambience due to which the price is high and so the ratings are good.

# In[ ]:


#Relationship between avarage cost of two and rating

sns.boxplot(y = 'Average Cost for two', x = 'Rating text', data=zomato_india)


# **Rating improves as avarage cost of two increases**

# In[ ]:


# Relationship between Price range and Rating text

sns.boxplot(y = 'Price range', x = 'Rating text', data = zomato_india)


# **Excellent and Very Good** rating resturents have a higher price range
# 
# **Average and Poor** rated resturent have a lower price range

# ### Lets identify restaurants which have high price range and low ratings

# In[ ]:


zomato_india['Price range'].value_counts()


# In[ ]:


# Lets have a look at the expensive restaurants

exp_india_restaurant = zomato_india[zomato_india['Price range'] == 4]
exp_india_restaurant


# In[ ]:


# Lets check the ratings of these restaurants

exp_india_restaurant['Rating text'].value_counts()


# As the price range is high, most of the ratings are good.
# So if price is high, why will be there be 5 poor ratings?

# In[ ]:


# Low rated expensive restaurants

exp_india_restaurant[exp_india_restaurant['Rating text'] == 'Poor']


# In[ ]:


list_of_cuisines = exp_india_restaurant[exp_india_restaurant['Rating text'] == 'Poor']['Cuisines']
list_of_cuisines.values


# **Observations**: These are 5 restaurants which are really expensive but do not have good ratings. <br>
# **Lets have a look at what is their cuisines.**

# In[ ]:


text = ' '.join([j for j in list_of_cuisines.values])
text


# In[ ]:


# Wordcloud of Cuisine

from wordcloud import WordCloud

plt.figure(figsize = (12, 6))
text = ' '.join([j for j in list_of_cuisines.values])
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top cuisines')
plt.axis("off")
plt.show()


# North Indian is the most popular cuisine. So we can infer that these North Indian restaurants in Gurgaon and Noida which do not provide authentic North Indian dishes and that is why customers are unhappy and rate them poorly.

# In[ ]:


bad_rated_restaurants = zomato_india[zomato_india['Rating text'] == 'Poor']
bad_rated_restaurants


# In[ ]:


bad_rated_restaurants.shape


# In[ ]:


bad_rated_restaurants['Has Online delivery'].value_counts()


# In[ ]:


bad_rated_restaurants['Is delivering now'].value_counts()


# Many of these restaurants are not available for delivery most of the time. Hence people provide poor rating to them.

# # **City wise Analysis of Poor Rated restaurants**

# In[ ]:


sns.countplot(x = 'City', data = bad_rated_restaurants)


# Why are ratings of New Delhi, Noida and Gurgaon bad?

# In[ ]:


# Total no. of bad restaurants

bad_rated_restaurants['City'].value_counts()


# In[ ]:


# Total number of restaurants

top_3_cities = zomato_india['City'].value_counts().head(3)
top_3_cities


# In[ ]:


sns.barplot(y = top_3_cities, x = top_3_cities.index)


# **Hence we cannot conclude that these 3 cities have significantly large number of bad restaurants as the total number of restaurants is also high**

# # Let's make our plots interactive using Plotly-express

# In[ ]:


get_ipython().system('pip install plotly_express')


# In[ ]:


import plotly_express as px


# In[ ]:


# Scatter Plot

px.scatter(zomato_data, x="Average Cost for two", y="Votes", size="Votes", color="Rating text", log_x=True, size_max=60,hover_name='City')


# **Observation**: We can see how the Average Cost for two and the Votes are related and in which cities.

# In[ ]:


# Scatter Plot

px.scatter_matrix(zomato_data, dimensions=['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes'], color='Rating text')


# **Observation** : There is not much correlation between the variables.

# In[ ]:


# Box plot of Rating text and Average Cost for two

px.box(zomato_india, x="Rating text", y="Average Cost for two", color="Price range", notched=True)


# **Observation**: Good and Very Good food have very high cost as compared to excellent and other types of food.

# In[ ]:


# Relationship between Online Delivery and Aggregate rating

px.histogram(zomato_data, x="Has Online delivery", y="Aggregate rating", histfunc="avg")


# **Observations**: Restaurants which have online delivery have better ratings.

# In[ ]:


px.histogram(zomato_data, x="Has Table booking", y="Aggregate rating", histfunc="avg")


# **Observations**: Restaurants which have table booking available have more ratings in general

# In[ ]:


px.histogram(zomato_data, x="Is delivering now", y="Aggregate rating", histfunc="avg")


# **Observation**: Restaurants which are delivering now have better ratings than the one which are not.
