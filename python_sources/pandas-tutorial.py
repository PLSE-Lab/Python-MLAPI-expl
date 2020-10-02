#!/usr/bin/env python
# coding: utf-8

# ## POKEMON STATS ANALYSIS

# Import the important packages we are going to need

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# List our input files

# In[ ]:


import os
print(os.listdir("../input/"))


# Read the csv file from our input and save it to a dataframe (df)

# In[ ]:


df =pd.read_csv("../input/Pokemon.csv")


# We'll use the **head** method to print the first rows of our data

# In[ ]:


df.head()


# We can also specify the number of rows to retrive using **head**

# In[ ]:


df.head(3)


# We'll use the **columns** attribute to get the columns for our dataframe

# In[ ]:


df.columns


# We can get specific columns from our dataframe

# In[ ]:


df[["Name","Type 1"]]


# using the **set_index** method we can change the index of our dataframe to a specific column[](http://)

# In[ ]:


df = df.set_index('Name')


# In[ ]:


df.head()


# Lets remove the # column, using the **drop** method

# In[ ]:


df=df.drop(['#'],axis=1)
df.head()


# ### CLEANING THE DATAFRAME

# lets get all the names that contain Mega in their name

# In[ ]:


df[df.index.str.contains("Mega")]


# Lets understand what happand:
# The following line give as array with boolean variables,it tells as if the string in the given place fulfill the condition.

# In[ ]:


df.index.str.contains("Mega")


# When putting it in the '[]' it will return us only the places where there is an True in the given array

# In[ ]:


df[df.index.str.contains("Mega")].head()


# The name of these pokemon is formatted a bit strangly, we'll use a regular expression to replace their names

# In[ ]:


df.index = df.index.str.replace(".*(?=Mega)", "")
df.head(10)


# Lets get all the values with mega in their name again to see if it worked

# In[ ]:


df[df.index.str.contains("Mega")].head()


# Lets format the names of our columns, make all of them in capital letters

# In[ ]:


df.columns = df.columns.str.upper().str.replace('_', '')
df.head()


# Filtering to see legendery pokemon

# In[ ]:


df[df['LEGENDARY'] == True].head(20)


# Lets show some more metadata about our dataframe.
# we already have seen the **columns** attribute. the **shape** attribute retrives us the total mesures of our data set, the number of rows and columns as a tuple 

# In[ ]:


print('The columns of the dataset are: ',','.join(list(df.columns)))
print('The shape of the dataframe is: ',df.shape)


# We can also print a summary of our data, which gives us alot of statistical information about it

# In[ ]:


df.describe()


# Lets look at what is a series, how can we get a series and what we can do with it.

# if we say that a dataframe is a two dimensional array, then a series is a one dimensional array. it has an index and values. we can get a series when retreving a single column from our dataframe

# In[ ]:


ser = df['TYPE 1']
ser


# We can access a single element using it's index (similar to loc)

# In[ ]:


ser['Bulbasaur']


# In[ ]:


ser.index


# In[ ]:


ser.values


# We can do alot of interesting things with a series. which we will learn about in this tutorial

# We know pokemon don't allways have a secondery type, but for the sake of this example let's say we need the fill the **TYPE 2** values of all pokemon with some value. every pokemon which already has a secondery type is fine, dor the ones that don't will fill them with the value of their **TYPE 1** column. for this we'll use the fillna method, which replaces NaN values in the dataset

# In[ ]:


df['TYPE 2'].fillna(df['TYPE 1'], inplace=True) #fill NaN values in Type2 with corresponding values of Type
df.head(100)


# The **loc** method is our way to access a specifc item in our dataframe, using our index, since our index is the name of the pokemon it is very easy to search for a specific pokemon

# In[ ]:


df.loc['Bulbasaur'] #retrieves complete row data from index with value Bulbasaur


# the **iloc** method is very similar, instead of using the index as the method of accessing the data it uses the serial index, i.e the number of the row we are trying to access.

# In[ ]:


df.iloc[0] #retrieves complete row date from index 0 ; integer version of loc


# We saw how to filter our dataset when we filtered only legendery pokemons. but what if we want to specify multipule filters?

# The first intuative way will be like this

# In[ ]:


# df[(df['TYPE 1']=='Fire' or df['TYPE 1']=='Dragon') and (df['TYPE 2']=='Dragon' or df['TYPE 2']=='Fire')].head(3)


# But that does not work. to combine filtering conditions in pandas, we need to use bitwise operators instead of normal python ones ('|' instead of **or** and '&' instead of **and**). and we need to surround every condition with brackets

# In[ ]:


df[((df['TYPE 1']=='Fire') | (df['TYPE 1']=='Dragon')) & ((df['TYPE 2']=='Dragon') | (df['TYPE 2']=='Fire'))].head(3)


# we can use the **max** method to get the max item in a series. and the idxmax method the get the id of the max item in a series

# In[ ]:


print(f'The pokemon with the msot HP is {df["HP"].idxmax()} with {df["HP"].max()} HP')


# We can use the **sort_values** method to sort our dataframe by a specific column 

# In[ ]:


df.sort_values('TOTAL',ascending=False).head()


# We can also sort by multipule columns when giving an array, this means that the if two rows have the same value in the first column we sort by, their order will be determined by the value in the second column

# In[ ]:


df.sort_values(['TOTAL','ATTACK'],ascending=False).head()


# We can use the **unique** method to get all the unique values from a series, and the **nunique** to get the number of unique values from a series

# In[ ]:


print('The unique  pokemon types are',df['TYPE 1'].unique()) #shows all the unique types in column
print('The number of unique types are',df['TYPE 1'].nunique()) #shows count of unique values 


# The **values_counts** method is very useful. it returns the count of each value in the series (how many times it appered) in descending order

# In[ ]:


df['TYPE 1'].value_counts()


# Group by is a very useful tool, it lets us group based on a column and ask questions about each group individually

# In[ ]:


df.groupby(['TYPE 1']).size()  #same as value_counts


# In[ ]:


df.groupby(['TYPE 1']).groups # our groups and which pokemon is in which


# In[ ]:


df.groupby(['TYPE 1']).groups.keys() # only the group names


# We can access certin rows from groups

# In[ ]:


df.groupby(['TYPE 1']).first()


# In[ ]:


df.groupby(['TYPE 1'])['SPEED'].max() # Highest speed from each group


# We can also preform some interesting aggregations

# In[ ]:


df.groupby(['TYPE 1'])['HP'].mean() # mean HP per Type 1 group


# In[ ]:


df.groupby(['TYPE 1'])['LEGENDARY'].sum() # Number of legendaries from each type


# We can also group by multipule columns

# In[ ]:


df.groupby(['TYPE 1','TYPE 2']).size()


# Lets see how we change values in a column, based on condition:
# 
# We will add 10 to speed to all Legedary Pokemons.

# In[ ]:


df["LEGENDARY"] == True # the condition


# In[ ]:


df[df["LEGENDARY"] == True].head()


# In[ ]:


df.loc[df["LEGENDARY"] == True,"SPEED"].head()


# In[ ]:


df.loc[df["LEGENDARY"] == True,"SPEED"] += 10


# In[ ]:


df.loc[df["LEGENDARY"] == True,"SPEED"].head()


# Lets revers the changes

# In[ ]:


df.loc[df["LEGENDARY"] == True,"SPEED"] -= 10


# ## VISUALISATIONS

# ### Matplotlib

# Matplotlib is a very easy and intuative library from visualization and ploting of data. let's see what kind of visuallizations we can do with it

# Lets start of with a simple bar plot

# A bar chart of the number of pokemons from each generation

# In[ ]:


# get the data ready
pokemon_per_generation = df["GENERATION"].value_counts().sort_index()
pokemon_per_generation


# If we want now we can easily plot our result in a barplot

# In[ ]:


pokemon_per_generation.plot.bar()


# In[ ]:


# or, alternatively
pokemon_per_generation.plot(kind='bar')


# Lets see some of the options we can use when we plot our data

# In[ ]:


# Add title
plt.title('Number of pokemon in each generation')

# Set our x/y labels
plt.xlabel('Generation')
plt.ylabel('Number of pokemon')

pokemon_per_generation.plot(kind='bar')
plt.show()


# We can also plot a group bar plot

# In[ ]:


legendaries_per_generation = df[df['LEGENDARY'] == True]["GENERATION"].value_counts().sort_index()
non_legendaries_per_generation = df[df['LEGENDARY'] == False]["GENERATION"].value_counts().sort_index()

# Concat 2 series to a dataframe with 2 columns
pd.concat([non_legendaries_per_generation,legendaries_per_generation],axis=1,keys=['non_legendaries','legendaries']).plot.bar()

# Add title
plt.title('Number of pokemon / legendaries in each generation')

# Set our x/y labels
plt.xlabel('Generation')
plt.ylabel('Number of pokemon')

# add legend
plt.legend(('Normal', 'Legendary'))

plt.show()


# Or a stacked bar plot

# In[ ]:


plt.bar(non_legendaries_per_generation.index, non_legendaries_per_generation.values)
# we use the bottom argument to start the legendaries bar from the end of the non legendaries
plt.bar(legendaries_per_generation.index, legendaries_per_generation.values,
             bottom=non_legendaries_per_generation.values)

# Add title
plt.title('Number of pokemon / legendaries in each generation')

# Set our x/y labels
plt.xlabel('Generation')
plt.ylabel('Number of pokemon')

# add legend
plt.legend(('Normal', 'Legendary'))

plt.show()


# We can also create a horizontal bar plot, for example: A horizontal bar chart of the number of pokemons for each type 1

# In[ ]:


df["TYPE 1"].value_counts().plot.barh()


# let's create an histogram, for example: The attack distribution for the pokemons across all the genarations

# In[ ]:


df.hist(column='ATTACK')


# Above is a Histogram showing the distribution of attacks for the Pokemons. The average value is between 75-77

# ### lineplot

# Number of Pokemons by Type And Generation

# In[ ]:


type_gen_total=df.groupby(['GENERATION','TYPE 1']).count().reset_index()
type_gen_total=type_gen_total[['GENERATION','TYPE 1','TOTAL']]
# pivot is a very useful method, it reshapes our data based on certin columns, we specify which columns will be the index, columns and values
# in this example we make our generation the row index, the columns are each pokemon type and the values are their total in each generation
type_gen_total=type_gen_total.pivot(index='GENERATION',columns='TYPE 1',values='TOTAL')
type_gen_total.plot(marker='o')
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()


# We can even specify the colors of our types to match the type

# In[ ]:


type_gen_total[["Fire","Water","Electric","Grass","Ice","Fighting","Poison","Ground","Flying","Psychic","Bug","Rock","Ghost","Dragon","Dark","Steel","Fairy"]].plot(marker='o'
,color=["#EE8130","#6390F0","#F7D02C","#7AC74C","#96D9D6","#C22E28","#A33EA1","#E2BF65","#A98FF3","#F95587","#A6B91A","#B6A136","#735797","#6F35FC","#705746","#B7B7CE","#D685AD"])
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()


# ### scatter plot

# In[ ]:


fire = df[(df['TYPE 1']=='Fire') | ((df['TYPE 2'])=="Fire")] #fire contains all fire pokemon
water = df[(df['TYPE 1']=='Water') | ((df['TYPE 2'])=="Water")]  #all water pokemon
ax = fire.plot.scatter(x='ATTACK', y='DEFENSE', color='Red', label='Fire')
water.plot.scatter(x='ATTACK', y='DEFENSE', color='Blue', label='Water', ax=ax);


# This shows that fire type pokemons have a better attack than water type pokemons but have a lower defence than water type.

# ### Strongest Pokemons By Types

# In[ ]:


strong=df.sort_values(by='TOTAL', ascending=False) #sorting the rows in descending order
strong.drop_duplicates(subset=['TYPE 1'],keep='first') #since the rows are now sorted in descending oredr
#thus we take the first row for every new type of pokemon i.e the table will check TYPE 1 of every pokemon
#The first pokemon of that type is the strongest for that type
#so we just keep the first row


# ### Pie chart

#  Distribution of various pokemon types

# In[ ]:


df['TYPE 1'].value_counts().plot.pie()


# ## Seaborn

# Seaborn is another visualization and plotting package which is based on matplotlib. then why should we use it and not matplotlib?. well the seaborn library usually produces better looking plots, with the ability to configure color, legends and more easily. we'll see how to create all the plots we created with matplotlib with seaborn and others we have not yet created

# Bar plot

# In[ ]:


# Add title
plt.title('Number of pokemon in each generation')

# Set our x/y labels
plt.xlabel('Generation')
plt.ylabel('Number of pokemon')
sns.barplot(pokemon_per_generation.index,pokemon_per_generation.values)

plt.show()


# We can also use a countplot instead of a barplot. a countplot is very simmilar to a histogram, 
# *from the seaborn docs:*
# A count plot can be thought of as a histogram across a categorical, instead of quantitative, variable. 
# in our case, the catagory is the pokemon generation and we mesure the change between each generation, instead of looking at each generation separately

# In[ ]:


# Add title
plt.title('Number of pokemon in each generation')

# Set our x/y labels
plt.xlabel('Generation')
plt.ylabel('Number of pokemon')
#sns.barplot(pokemon_per_generation.index,pokemon_per_generation.values)
sns.countplot('GENERATION',data=df,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))

plt.show()


# In[ ]:


plot_data = df["TYPE 1"].value_counts()
sns.barplot(plot_data.values,plot_data.index,palette='plasma')


# In[ ]:


sns.distplot(df['ATTACK'])


# In[ ]:


plot_data = df.groupby(['GENERATION','TYPE 1']).count().reset_index()
sns.lineplot(x='GENERATION',y='TOTAL',hue='TYPE 1',data=plot_data,marker='o')
fig=plt.gcf()
fig.set_size_inches(15,9)
plt.legend(loc='upper right')
plt.show()


# In[ ]:


fire_and_water = df[(df['TYPE 1'].isin(['Fire','Water']))]
sns.scatterplot(x='ATTACK', y='DEFENSE',data=fire_and_water, hue='TYPE 1')


# Now that we gone through all the plots we learned, lets get into some more complex ones

# ### boxplot

# All stats analysis of the pokemons

# In[ ]:


df2=df.drop(['GENERATION','TOTAL'],axis=1)
sns.boxplot(data=df2)
plt.ylim(0,200)  #change the scale of y axix
plt.show()


# In[ ]:


plt.subplots(figsize = (15,5))
plt.title('Attack by Type1')
sns.boxplot(x = "TYPE 1", y = "ATTACK",data = df)
plt.ylim(0,200)
plt.show()


# #### This shows that the Dragon type pokemons have an edge over the other types as they have a higher attacks compared to the other types. Also since the fire pokemons have lower range of values, but higher attacks, they can be preferred over the grass and water types for attacking.
# 

# In[ ]:


plt.subplots(figsize = (15,5))
plt.title('Attack by Type2')
sns.boxplot(x = "TYPE 2", y = "ATTACK",data=df)
plt.show()


# In[ ]:


plt.subplots(figsize = (15,5))
plt.title('Defence by Type')
sns.boxplot(x = "TYPE 1", y = "DEFENSE",data = df)
plt.show()


# This shows that steel type pokemons have the highest defence but normal type pokemons have the lowest defence

# ### violinplot

# In[ ]:


plt.subplots(figsize = (20,10))
plt.title('Attack by Type1')
sns.violinplot(x = "TYPE 1", y = "ATTACK",data = df)
plt.ylim(0,200)
plt.show()


# What the violinplot actually does is it plots according to the density of a region. This means that the parts of the plot where the width is thicker denotes a region with higher density points whereas regions with thinner area show less densely populated points.

# In[ ]:


plt.subplots(figsize = (20,10))
plt.title('Attack by Type1')
sns.violinplot(x = "TYPE 1", y = "DEFENSE",data = df)
plt.ylim(0,200)
plt.show()


# In[ ]:


plt.subplots(figsize = (15,5))
plt.title('Strongest Genaration')
sns.violinplot(x = "GENERATION", y = "TOTAL",data = df)
plt.show()


# This shows that generation 3  has the better pokemons

# ### Finding any Correlation between the attributes

# In[ ]:


plt.figure(figsize=(10,6)) #manage the size of the plot
sns.heatmap(df.corr(),annot=True) #df.corr() makes a correlation matrix and sns.heatmap is used to show the correlations heatmap
plt.show()


# From the heatmap it can be seen that there is not much correlation between the attributes of the pokemons. The highest we can see is the correlation between Sp.Atk and the Total 
