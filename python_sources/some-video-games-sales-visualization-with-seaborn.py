#!/usr/bin/env python
# coding: utf-8

# # Exploring Video Game Sales

# Let's explore the different video game sales dataset

# ## I - Pre analysis work

# ### A - Load dependencies

# We'll be using visualisation libraries

# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd


# ### B - Load dataset

# Let's load the dataset

# In[ ]:


data = pd.read_csv("../input/vgsales.csv")


# ## II - Exploring Work

# ### A - First approach of the data

# #### 1. Head of data

# First, let's get a feel of what our data look like

# In[ ]:


data.head()


# #### 2. Info on columns

# Let's see the type of our data, to better understand the structure behind it

# In[ ]:


data.info()


# #### 3. Numerical informations

# Let's see the informations on the numerical distribution of our dataset

# In[ ]:


data.describe()


# #### 4. Categorical informations

# Finally, let's see the informations on the categorical distribution of our dataset. For visual purpose it is not recommanded to use this though

# In[ ]:


def categories_info(data) :
    print("----------")
    print("Categorical Informations")
    print("----------")
    for col in data.columns :
        if data[col].dtypes == 'object' and col != "Name" : #If you want to see "Name" category, delete 'and col != "Name"'
            print("")
            print("**********")
            print("Values for ", col)
            print("Number of different values : ", len(data[col].value_counts()))
            print("**********")
            print(data[col].value_counts())
            print(data[col].value_counts().describe())

categories_info(data)


# ###### Observation : 
# 
# The dataset contains the following informations : Name of the game, platform and  year of release, as well as its genre and publisher. The sales distributed following the market are also represented, and the rank of the game.
# 
# The same name may appear several times, because of crossplatform games
# 
# The dataset is composed of 16 598 items.
# 
# It seems there are missing values for Year and publisher data. 
# 
# Year of release range from 1980 to 2020 (funny isn't it ?), most games were released between 2000 and 2012.
# 
# The biggest market among NA, JP and EU is NA, 4 times the JP one.
# 
# 16 598 Million games have ever been sold throughout the world, the most sold one is "Wii Sports" with 82.74 Million.
# 
# The mean (0.53) and median (0.17) values for global sales are a bit different. It means there is more games that sold badly than ones that succeeded.
# 
# The most popular platforms ever are DS and PS2 (more than 2000)
# 
# Most popular genre ever is Action, least popular is puzzle (6 times less)

# ### B - Data Exploration

# #### Null Year

# We do understand the structure of the dataset, let's try to know more about the games with unidentified year of release

# In[ ]:


#Let's affect this specific dataframe, as we will use it several times
year_null = data[data['Year'].isnull()]
year_null.head(10)


# Let's try to know more about the distribution

# In[ ]:


year_null.describe()


# Let's also know more about the categories

# In[ ]:


categories_info(year_null)


# ###### Observation : 
# 
# 271 years are missing from the dataset.
# 
# As it turns out, there is no obvious reason as to why these values are missing. 
# 
# There are good games as well as bad ones. 
# 
# A quick view at the name of the games may even give us the year of release in certain cases ("Madden NFL 2004" !). These games don't seem to have year of release in common either. 
# 
# But they all sold pretty bad in Japan.

# #### Data Cleaning

# Let's drop the null values. As there is not much of them (270 for 16 700 in total), it won't make much of a difference.
# 
# We could also drop the games that sold poorly.
# 
# While we're at it, let's leave the games that sold more than 10 Million, because they are way too far from the other games

# In[ ]:


data_cleaned = data.dropna()
data_cleaned2 = data_cleaned[data_cleaned['Global_Sales'] > 1]
data_cleaned3 = data_cleaned[data_cleaned['Global_Sales'] < 10]
data_cleaned4 = data_cleaned2[data_cleaned2['Global_Sales'] < 10]

#Games sold between 1 and 10 Million by publisher for more than 10 games
mask = data_cleaned4['Publisher'].value_counts().sort_values() > 10
long = mask[mask.values==True]
clean = data_cleaned4[data_cleaned4['Publisher'].isin(long.index.values)]

#Games sold between 1 and 10 Million
mask = data_cleaned3['Publisher'].value_counts().sort_values() > 10
long = mask[mask.values==True]
clean2 = data_cleaned3[data_cleaned3['Publisher'].isin(long.index.values)]


# In[ ]:


print(len(data_cleaned2))
print(len(data_cleaned4))


# ## III - Visualization Work

# ### A - Distribution of each data

# Let's visualize all the distributions at once. It is a more efficient way to know more about the data. 
# 
# Let's also visualize the game that sold more than 1 Million on the same plots, because the game that succeed are more interesting because they are rarer

# In[ ]:


plot_col = ['Platform', 'Year', 'Genre', 'Publisher', 'Global_Sales']

def plot_distrib(data, data2) :
    
    #Size the figure that will hold all the subplots
    plt.figure(figsize=(12, 15))
    i = 1
    
    #For all the columns, make a subplot
    for col in plot_col :
        plt.subplot(int("32" + str(i)))
        
        #If this is a numerical type : 
        if data[col].dtypes != object :
            
            #Plot the values
            maxi = data[col].value_counts().sort_values(ascending=False).index[0]
            maxi2 = data2[col].value_counts().sort_values(ascending=False).index[0]
            sns.distplot(data[col], kde=False, label="All games")
            sns.distplot(data2[col], kde=False, label="Sales > 1 M")
            
            #Add information on the most popular and the most successfull parameters
            plt.title("{} = popular {}, {} = successfull".format(maxi, col, maxi2))
            plt.legend()
            
        #If this is a categorical type :
        else :
            
            #Restrain from having too many values on the sample plot. Here only the values that occured more than 100 times
            mask = data[col].value_counts().sort_values() > 100
            long = mask[mask.values==True]
            clean = data[data[col].isin(long.index.values)]
            
            #Plot the values and cleaned values, in descending order of occurence
            maxi = clean[col].value_counts().sort_values(ascending=False).index
            maxi2 = data2[col].value_counts().sort_values(ascending=False).index
            sns.countplot(clean[col], order=maxi, label="All games")
            sns.countplot(data2[col], order=maxi, label="Sales > 1 M", saturation=0.4)
            
            #Add information on the most popular and the most successfull parameters
            plt.title("{} = popular {}, {} = successfull".format(maxi[0], col, maxi2[0]))
            plt.legend()
            
        #Rotate the labels on x-axis
        plt.xticks(rotation=90)
        i += 1
    
    #Espace each subplot to avoid overlap
    plt.subplots_adjust(hspace=0.5)


plot_distrib(clean2, clean)
plt.show()


# ###### Observation
# 
# Most games didn't sale much. Only 2031 games sold much than 1 Million. We could analyze them in order to get insights on why they succeeded.
# 
# Though Electronic Arts sold the most number of games, Nintendo sold more successfull games
# 
# Though DS is the console with the more games, PS2 is the console which has the more
# 
# The game market in general is undergoing a general recess since 2009. Nothing seems to explain it with these data, no console were released in 2009.

# ### B - Crossed Distribution

# Now that the distribution of the features are clear, let's see the correlation between the variables.

# #### 1. Numerical Combinations

# Let's combine Year with global sales. Also we'll add if the game is successfull or not

# In[ ]:


def plot_cross_num(data2) :
    
    #Search for the numerical types
    col_num = []
    for col in plot_col :
        if data2[col].dtypes != object :
            col_num.append(col)
    
    #Size the figure that will contain the subplots
    plt.figure(figsize=(12, 15))
    i = 1
    
    #For each column
    for col in col_num :
        col_num.remove(col)
        for col2 in col_num :
        
            #Plot the values
            plt.subplot(int("32" + str(i)))
            sns.lmplot(x=col, y=col2, data=data2, fit_reg=False, hue="Genre", palette="Set1")            
            sns.kdeplot(data2[col], data2[col2], n_levels=20)

            #Add information
            plt.title("{} co-plotted with {}".format(col, col2))
        
            #Rotate the x-label
            plt.xticks(rotation=90)
            i += 1
        
    #Adjust the subplots so that they don't overlap
    plt.subplots_adjust(hspace=0.5)
    
    
plot_cross_num(clean)
plt.show()


# #### 2. Categorical Combinations

# Let's combine Platform, genre and publisher together. As previously, we'll add successfull or not distinction

# In[ ]:


def plot_cross_cat(data2) :
    
    #Search for the categorical types
    col_cat = []
    for col in plot_col :
        if data2[col].dtypes == object :
            col_cat.append(col)
    
    #Size the figure that will contain the subplots
    plt.figure(figsize=(15, 30))
    i = 1
    
    #For each column
    for col in col_cat :
        col_cat.remove(col)
        for col2 in col_cat :
        
            #Plot the values
            plt.subplot(int("42" + str(i)))
            table_count = pd.pivot_table(data2,values=['Global_Sales'],index=[col],columns=[col2],aggfunc='count',margins=False)
            sns.heatmap(table_count['Global_Sales'],linewidths=.5,annot=True,fmt='2.0f',vmin=0)

            #Add information
            plt.title("{} co-plotted with {}".format(col, col2))
        
            #Rotate the x-label
            plt.xticks(rotation=90)
            i += 1
        
    #Adjust the subplots so that they don't overlap
    plt.subplots_adjust(hspace=0.5)
    
    
plot_cross_cat(clean)
plt.show()


# ###### Observation :
# 
# The most important consoles are PS3, PS2 and Xbox360, which mostly have action and Sports games.
# 
# Electronic arts and Sony mainly develop on Psx consoles, whereas Nintendo develops on his own consoles.
# 
# Electronic Arts specializes in Shooter, Sports and Racing games, whereas Nintendo develops Platform games.

# #### 3. Multivariate Combinations

# Let's plot numerical and categorical values together

# In[ ]:


def plot_cross_cross(data2) :
    
    #Search for the types
    col_cat = []
    col_num = []
    for col in plot_col :
        if data2[col].dtypes == object :
            col_cat.append(col)
        else :
            col_num.append(col)
    
    #Size the figure that will contain the subplots
    plt.figure(figsize=(12, 15))
    i = 1
    j = len(col_cat)
    k = len(col_num)
    #For each column
    for cat in col_cat :
        for num in col_num :
        
            #Plot the values
            plt.subplot(int(str(j) + str(k) + str(i)))
            #mask = data[cat].value_counts().sort_values() > 100
            #long = mask[mask.values==True]
            #clean = data[data[cat].isin(long.index.values)]
            sns.violinplot(x=cat , y=num, data=data2, inner=None) 
            sns.swarmplot(x=cat, y=num, data=data2, alpha=0.7) 
            #table_count = pd.pivot_table(data2,values=['Global_Sales'],index=[num],columns=[cat],aggfunc='count',margins=False)
            #sns.heatmap(table_count['Global_Sales'],linewidths=.5,annot=True,fmt='2.0f',vmin=0)

            #Add information
            plt.title("{} co-plotted with {}".format(cat, num))
        
            #Rotate the x-label
            plt.xticks(rotation=90)
            i += 1
        
    #Adjust the subplots so that they don't overlap
    plt.subplots_adjust(hspace=0.5)
    
    
plot_cross_cross(clean)
plt.show()


# ###### Observation :
# 
# We have a clear overview of the lifetime of the consoles. We could replot them with colonne of console following each other next to them
# 
# 

# # Conclusion

# Here is a summary of all the informations obtained through this study
# 
# NA is the biggest game market, JP the smallest one
# 
# The best selling game ever is "Wii Sports" with 80 Million unit sold
# 
# On 17 000 games ever released, only 2000 made it to the 1 Million cap
# 
# The most popular platform are DS and PS2, genre is Action, publisher is Electronic Arts.
# 
# Electronic Arts mostly publishes on Psx consoles sports oriented games, whereas Nintendo specialises in platform games for his consoles
# 
# The peak of sales is during 2009. Now the game market is in recession
