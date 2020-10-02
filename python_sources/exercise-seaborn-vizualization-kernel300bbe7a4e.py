#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Path of the file to read

fifa_filepath = "../input/data-for-datavis/fifa.csv"


# Read the file into a variable fifa_data
fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)


# # Plot the data  (FIFA)

# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(16,6))

# Line chart showing how FIFA rankings evolved over time
sns.lineplot(data=fifa_data)


# # Select a dataset (Spotify)
# The dataset for this tutorial tracks global daily streams on the music streaming service Spotify. We focus on five popular songs from 2017 and 2018:
# 
# "Shape of You", by Ed Sheeran [(link)](https://en.wikipedia.org/wiki/Shape_of_You)
# "Despacito", by Luis Fonzi ([link](https://en.wikipedia.org/wiki/Despacito))
# "Something Just Like This", by The Chainsmokers and Coldplay ([link](https://en.wikipedia.org/wiki/Something_Just_like_This))
# "HUMBLE.", by Kendrick Lamar ([link](https://en.wikipedia.org/wiki/Humble_(song)))
# "Unforgettable", by French Montana ([link](https://en.wikipedia.org/wiki/Unforgettable_(French_Montana_song)))
# 

# # Load the data

# In[ ]:


# Path of the file to read
spotify_filepath = "../input/data-for-datavis/spotify.csv"

# Read the file into a variable spotify_data
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)


# # Examine the data

# In[ ]:


# Print the first 5 rows of the data
spotify_data.head()


# 

# In[ ]:


# Print the last five rows of the data
spotify_data.tail()


# # Plot the data

# In[ ]:


# Line chart showing daily global streams of each song 
sns.lineplot(data=spotify_data)


# 

# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(14,6))

# Add title
plt.title("Daily Global Streams of Popular Songs in 2017-2018")

# Line chart showing daily global streams of each song 
sns.lineplot(data=spotify_data)


# # Plot a subset of the data
# So far, you've learned how to plot a line for every column in the dataset. In this section, you'll learn how to plot a subset of the columns.
# 
# We'll begin by printing the names of all columns. This is done with one line of code and can be adapted for any dataset by just swapping out the name of the dataset (in this case, spotify_data).

# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(14,6))

# Add title
plt.title("Daily Global Streams of Popular Songs in 2017-2018")

# Line chart showing daily global streams of 'Shape of You'
sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")

# Line chart showing daily global streams of 'Despacito'
sns.lineplot(data=spotify_data['Despacito'], label="Despacito")

# Add label for horizontal axis
plt.xlabel("Date")


# In[ ]:





# 

# 

# In[ ]:


# Path of the file to read
museum_filepath = "../input/data-for-datavis/museum_visitors.csv"

# Fill in the line below to read the file into a variable museum_data
museum_data = pd.read_csv(museum_filepath, index_col="Date", parse_dates=True)


# # Step 2: Review the data
# 
# Use a Python command to print the last 5 rows of the data.

# In[ ]:


# Print the last five rows of the data 
museum_data.tail()


# ## Step 3: Convince the museum board 
# 
# The Firehouse Museum claims they ran an event in 2014 that brought an incredible number of visitors, and that they should get extra budget to run a similar event again.  The other museums think these types of events aren't that important, and budgets should be split purely based on recent visitors on an average day.  
# 
# To show the museum board how the event compared to regular traffic at each museum, create a line chart that shows how the number of visitors to each museum evolved over time.  Your figure should have four lines (one for each museum).
# 
# > **(Optional) Note**: If you have some prior experience with plotting figures in Python, you might be familiar with the `plt.show()` command.  

# In[ ]:


# Line chart showing the number of visitors to each museum over time
# Set the width and height of the figure
plt.figure(figsize=(12,6))
# Line chart showing the number of visitors to each museum over time
sns.lineplot(data=museum_data)
# Add title
plt.title("Monthly Visitors to Los Angeles City Museums")


# ## Step 4: Assess seasonality
# 
# When meeting with the employees at Avila Adobe, you hear that one major pain point is that the number of museum visitors varies greatly with the seasons, with low seasons (when the employees are perfectly staffed and happy) and also high seasons (when the employees are understaffed and stressed).  You realize that if you can predict these high and low seasons, you can plan ahead to hire some additional seasonal employees to help out with the extra work.
# 
# #### Part A
# Create a line chart that shows how the number of visitors to Avila Adobe has evolved over time.  (_If your code returns an error, the first thing that you should check is that you've spelled the name of the column correctly!  You must write the name of the column exactly as it appears in the dataset._)

# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(12,6))
# Add title
plt.title("Monthly Visitors to Avila Adobe")
# Line chart showing the number of visitors to Avila Adobe over time
sns.lineplot(data=museum_data['Avila Adobe'])
# Add label for horizontal axis
plt.xlabel("Date")


# #### Part B
# 
# Does Avila Adobe get more visitors:
# - in September-February (in LA, the fall and winter months), or 
# - in March-August (in LA, the spring and summer)?  
# 
# Using this information, when should the museum staff additional seasonal employees?

# In[ ]:





# # Select a Dataset (Flight delays)

# n this tutorial, we'll work with a dataset from the US Department of Transportation that tracks flight delays.
# 
# Opening this CSV file in Excel shows a row for each month (where 1 = January, 2 = February, etc) and a column for each airline code.
# 
# 
# Each entry shows the average arrival delay (in minutes) for a different airline and month (all in year 2015). Negative entries denote flights that (on average) tended to arrive early. For instance, the average American Airlines flight (airline code: AA) in January arrived roughly 7 minutes late, and the average Alaska Airlines flight (airline code: AS) in April arrived roughly 3 minutes early.

# # Load data

# In[ ]:


# Path of the file to read
flight_filepath = "../input/data-for-datavis/flight_delays.csv"

# Read the file into a variable flight_data
flight_data = pd.read_csv(flight_filepath, index_col="Month")


# # Examine the data

# In[ ]:


# Print the data
flight_data


# # Bar chart
# Say we'd like to create a bar chart showing the average arrival delay for Spirit Airlines (airline code: NK) flights, by month.

# In[ ]:



# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=flight_data.index, y=flight_data['NK'])

# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")


# In[ ]:


# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=flight_data.index, y=flight_data['NK'])


# #It has three main components:
# 
# sns.barplot - This tells the notebook that we want to create a bar chart.
# Remember that sns refers to the seaborn package, and all of the commands that you use to create charts in this course will start with this prefix.
# x=flight_data.index - This determines what to use on the horizontal axis. In this case, we have selected the column that indexes the rows (in this case, the column containing the months).
# y=flight_data['NK'] - This sets the column in the data that will be used to determine the height of each bar. In this case, we select the 'NK' column.
# Important Note: You must select the indexing column with flight_data.index, and it is not possible to use flight_data['Month'] (which will return an error). This is because when we loaded the dataset, the "Month" column was used to index the rows. We always have to use this special notation to select the indexing column.
# 

# # Heatmap
# We have one more plot type to learn about: heatmaps!
# 
# In the code cell below, we create a heatmap to quickly visualize patterns in flight_data. Each cell is color-coded according to its corresponding value.

# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(14,7))

# Add title
plt.title("Average Arrival Delay for Each Airline, by Month")

# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=flight_data, annot=True)

# Add label for horizontal axis
plt.xlabel("Airline")


# In[ ]:


#The relevant code to create the heatmap is as follows:

# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=flight_data, annot=True)


# This code has three main components:
# 
# sns.heatmap - This tells the notebook that we want to create a heatmap.
# data=flight_data - This tells the notebook to use all of the entries in flight_data to create the heatmap.
# annot=True - This ensures that the values for each cell appear on the chart. (Leaving this out removes the numbers from each of the cells!)
# What patterns can you detect in the table? For instance, if you look closely, the months toward the end of the year (especially months 9-11) appear relatively dark for all airlines. This suggests that airlines are better (on average) at keeping schedule during these months!

# 

# 
# In this exercise, you will use your new knowledge to propose a solution to a real-world scenario.  To succeed, you will need to import data into Python, answer questions using the data, and generate **bar charts** and **heatmaps** to understand patterns in the data.
# 
# ## Scenario
# 
# You've recently decided to create your very own video game!  As an avid reader of [IGN Game Reviews](https://www.ign.com/reviews/games), you hear about all of the most recent game releases, along with the ranking they've received from experts, ranging from 0 (_Disaster_) to 10 (_Masterpiece_).
# 
# ![ex2_ign](https://i.imgur.com/Oh06Fu1.png)
# 
# You're interested in using [IGN reviews](https://www.ign.com/reviews/games) to guide the design of your upcoming game.  Thankfully, someone has summarized the rankings in a really useful CSV file that you can use to guide your analysis.
# 
# ## Setup
# 
# Run the next cell to import and configure the Python libraries that you need to complete the exercise.
# 
# 

# ## Step 1: Load the data
# 
# Read the IGN data file into `ign_data`.  Use the `"Platform"` column to label the rows.

# In[ ]:


# Path of the file to read
ign_filepath = "../input/data-for-datavis/ign_scores.csv"

# Fill in the line below to read the file into a variable ign_data
ign_data = pd.read_csv(ign_filepath, index_col="Platform")


# ## Step 2: Review the data
# 
# Use a Python command to print the entire dataset.

# In[ ]:


# Print the data
ign_data


# In[ ]:


# Fill in the line below: What is the highest average score received by PC games,
# for any platform?
high_score = 7.759930

# On the Playstation Vita platform, which genre has the 
# lowest average score? Please provide the name of the column, and put your answer 
# in single quotes (e.g., 'Action', 'Adventure', 'Fighting', etc.)
worst_genre = 'Simulation'


# ## Step 3: Which platform is best?
# 
# Since you can remember, your favorite video game has been [**Mario Kart Wii**](https://www.ign.com/games/mario-kart-wii), a racing game released for the Wii platform in 2008.  And, IGN agrees with you that it is a great game -- their rating for this game is a whopping 8.9!  Inspired by the success of this game, you're considering creating your very own racing game for the Wii platform.
# 
# #### Part A
# 
# Create a bar chart that shows the average score for **racing** games, for each platform.  Your chart should have one bar for each platform. 

# In[ ]:


# Bar chart showing average score for racing games by platform

# Set the width and height of the figure
plt.figure(figsize=(8, 6))
# Bar chart showing average score for racing games by platform
sns.barplot(x=ign_data['Racing'], y=ign_data.index)
# Add label for horizontal axis
plt.xlabel("")
# Add label for vertical axis
plt.title("Average Score for Racing Games, by Platform")


# #### Part B
# 
# Based on the bar chart, do you expect a racing game for the **Wii** platform to receive a high rating?  If not, what gaming platform seems to be the best alternative?

# ## Step 4: All possible combinations!
# 
# Eventually, you decide against creating a racing game for Wii, but you're still committed to creating your own video game!  Since your gaming interests are pretty broad (_... you generally love most video games_), you decide to use the IGN data to inform your new choice of genre and platform.
# 
# #### Part A
# 
# Use the data to create a heatmap of average score by genre and platform.  

# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(10,10))
# Heatmap showing average game score by platform and genre
sns.heatmap(ign_data, annot=True)
# Add label for horizontal axis
plt.xlabel("Genre")
# Add label for vertical axis
plt.title("Average Game Score, by Platform and Genre")


# #### Part B
# 
# Which combination of genre and platform receives the highest average ratings?  Which combination receives the lowest average rankings?

# Simulation games for Playstation 4 receive the highest average ratings (9.2). Shooting and Fighting games for Game Boy Color receive the lowest average rankings (4.5).

# # how to create advanced scatter plots.

# # Load and examine the data
# We'll work with a (synthetic) dataset of **insurance charges, to see if we can understand why some customers pay more than others.
# 
# tut3_insurance
# 
# If you like, you can read more about the dataset [here](https://www.kaggle.com/mirichoi0218/insurance/home).

# In[ ]:


# Path of the file to read
insurance_filepath = "../input/data-for-datavis/insurance.csv"

# Read the file into a variable insurance_data
insurance_data = pd.read_csv(insurance_filepath)


# In[ ]:


insurance_data.head()


# # Scatter plots
# To create a simple scatter plot, we use the sns.scatterplot command and specify the values for:
# 
# the horizontal x-axis (x=insurance_data['bmi']), and
# the vertical y-axis (y=insurance_data['charges']).

# In[ ]:


sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])


# The scatterplot above suggests that body mass index (BMI) and insurance charges are positively correlated, where customers with higher BMI typically also tend to pay more in insurance costs. (This pattern makes sense, since high BMI is typically associated with higher risk of chronic disease.)
# 
# To double-check the strength of this relationship, you might like to add a regression line, or the line that best fits the data. We do this by changing the command to sns.regplot.

# In[ ]:


sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])


# # Color-coded scatter plots
# We can use scatter plots to display the relationships between (not two, but...) three variables! One way of doing this is by color-coding the points.
# 
# For instance, to understand how smoking affects the relationship between BMI and insurance costs, we can color-code the points by 'smoker', and plot the other two columns ('bmi', 'charges') on the axes.

# In[ ]:


sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])


# The sns.lmplot command above works slightly differently than the commands you have learned about so far:
# 
# Instead of setting x=insurance_data['bmi'] to select the 'bmi' column in insurance_data, we set x="bmi" to specify the name of the column only.
# Similarly, y="charges" and hue="smoker" also contain the names of columns.
# We specify the dataset with data=insurance_data.
# Finally, there's one more plot that you'll learn about, that might look slightly different from how you're used to seeing scatter plots. Usually, we use scatter plots to highlight the relationship between two continuous variables (like "bmi" and "charges"). However, we can adapt the design of the scatter plot to feature a categorical variable (like "smoker") on one of the main axes. We'll refer to this plot type as a categorical scatter plot, and we build it with the sns.swarmplot command.

# In[ ]:


sns.swarmplot(x=insurance_data['smoker'],
              y=insurance_data['charges'])


# Among other things, this plot shows us that:
# 
# on average, non-smokers are charged less than smokers, and
# the customers who pay the most are smokers; whereas the customers who pay the least are non-smokers.

# 

# In this exercise, you will use your new knowledge to propose a solution to a real-world scenario.  To succeed, you will need to import data into Python, answer questions using the data, and generate **scatter plots** to understand patterns in the data.
# 
# ## Scenario
# 
# You work for a major candy producer, and your goal is to write a report that your company can use to guide the design of its next product.  Soon after starting your research, you stumble across this [very interesting dataset](https://fivethirtyeight.com/features/the-ultimate-halloween-candy-power-ranking/) containing results from a fun survey to crowdsource favorite candies.
# 
# ## Setup
# 
# 

# ## Step 1: Load the Data
# 
# Read the candy data file into `candy_data`.  Use the `"id"` column to label the rows.

# In[ ]:


# Path of the file to read
candy_filepath = "../input/data-for-datavis/candy.csv"

# Fill in the line below to read the file into a variable candy_data
candy_data = pd.read_csv(candy_filepath, index_col="id")


# ## Step 2: Review the data
# 
# Use a Python command to print the first five rows of the data.

# In[ ]:


# Print the first five rows of the data
candy_data.head() 


# The dataset contains 83 rows, where each corresponds to a different candy bar.  There are 13 columns:
# - `'competitorname'` contains the name of the candy bar. 
# - the next **9** columns (from `'chocolate'` to `'pluribus'`) describe the candy.  For instance, rows with chocolate candies have `"Yes"` in the `'chocolate'` column (and candies without chocolate have `"No"` in the same column).
# - `'sugarpercent'` provides some indication of the amount of sugar, where higher values signify higher sugar content.
# - `'pricepercent'` shows the price per unit, relative to the other candies in the dataset.
# - `'winpercent'` is calculated from the survey results; higher values indicate that the candy was more popular with survey respondents.
# 
# Use the first five rows of the data to answer the questions below.

# In[ ]:


# Fill in the line below: Which candy was more popular with survey respondents:
# '3 Musketeers' or 'Almond Joy'?  (Please enclose your answer in single quotes.)
more_popular = '3 Musketeers'
# Fill in the line below: Which candy has higher sugar content: 'Air Heads'
# or 'Baby Ruth'? (Please enclose your answer in single quotes.)
more_sugar = 'Air Heads'


# ## Step 3: The role of sugar
# 
# Do people tend to prefer candies with higher sugar content?  
# 
# #### Part A
# 
# Create a scatter plot that shows the relationship between `'sugarpercent'` (on the horizontal x-axis) and `'winpercent'` (on the vertical y-axis).  _Don't add a regression line just yet -- you'll do that in the next step!_

# In[ ]:


# Scatter plot showing the relationship between 'sugarpercent' and 'winpercent'
sns.scatterplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])


# #### Part B
# 
# Does the scatter plot show a **strong** correlation between the two variables?  If so, are candies with more sugar relatively more or less popular with the survey respondents?

# *The scatter plot does not show a strong correlation between the two variables. Since there is no clear relationship between the two variables, this tells us that sugar content does not play a strong role in candy popularity.
# *

# ## Step 4: Take a closer look
# 
# #### Part A
# 
# Create the same scatter plot you created in **Step 3**, but now with a regression line!

# In[ ]:


# Scatter plot w/ regression line showing the relationship between 'sugarpercent' and 'winpercent'
sns.regplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])


# #### Part B
# 
# According to the plot above, is there a **slight** correlation between `'winpercent'` and `'sugarpercent'`?  What does this tell you about the candy that people tend to prefer?

# **Since the regression line has a slightly positive slope, this tells us that there is a slightly positive correlation between 'winpercent' and 'sugarpercent'. Thus, people have a slight preference for candies containing relatively more sugar.**

# ## Step 5: Chocolate!
# 
# In the code cell below, create a scatter plot to show the relationship between `'pricepercent'` (on the horizontal x-axis) and `'winpercent'` (on the vertical y-axis). Use the `'chocolate'` column to color-code the points.  _Don't add any regression lines just yet -- you'll do that in the next step!_
# 
# 

# Can you see any interesting patterns in the scatter plot?  We'll investigate this plot further  by adding regression lines in the next step!
# 
# ## Step 6: Investigate chocolate
# 
# #### Part A
# 
# Create the same scatter plot you created in **Step 5**, but now with two regression lines, corresponding to (1) chocolate candies and (2) candies without chocolate.

# In[ ]:


# Color-coded scatter plot w/ regression lines
sns.lmplot(x="pricepercent", y="winpercent", hue="chocolate", data=candy_data)


# # Part B
# Using the regression lines, what conclusions can you draw about the effects of chocolate and price on candy popularity?
# 
# 
# 
# 
# 
# 
# We'll begin with the regression line for chocolate candies. Since this line has a slightly positive slope, we can say that more expensive chocolate candies tend to be more popular (than relatively cheaper chocolate candies). Likewise, since the regression line for candies without chocolate has a negative slope, we can say that if candies don't contain chocolate, they tend to be more popular when they are cheaper. One important note, however, is that the dataset is quite small -- so we shouldn't invest too much trust in these patterns! To inspire more confidence in the results, we should add more candies to the dataset.

# 

# ## Step 7: Everybody loves chocolate.
# 
# #### Part A
# 
# Create a categorical scatter plot to highlight the relationship between `'chocolate'` and `'winpercent'`.  Put `'chocolate'` on the (horizontal) x-axis, and `'winpercent'` on the (vertical) y-axis.

# In[ ]:


# Scatter plot showing the relationship between 'chocolate' and 'winpercent'
sns.swarmplot(x=candy_data['chocolate'], y=candy_data['winpercent'])


# #### Part B
# 
# You decide to dedicate a section of your report to the fact that chocolate candies tend to be more popular than candies without chocolate.  Which plot is more appropriate to tell this story: the plot from **Step 6**, or the plot from **Step 7**?

# **In this case, the categorical scatter plot from Step 7 is the more appropriate plot. While both plots tell the desired story, the plot from Step 6 conveys far more information that could distract from the main point.\**

# # histograms and density plots. (Distributions)

# ## Set up the notebook

# # Select a dataset
# We'll work with a dataset of 150 different flowers, or 50 each from three different species of iris (Iris setosa, Iris versicolor, and Iris virginica).
# 
# tut4_iris
# 
# # Load and examine the data
# Each row in the dataset corresponds to a different flower. There are four measurements: the sepal length and width, along with the petal length and width. We also keep track of the corresponding species.

# In[ ]:


# Path of the file to read
iris_filepath = "../input/data-for-datavis/iris.csv"

# Read the file into a variable iris_data
iris_data = pd.read_csv(iris_filepath, index_col="Id")

# Print the first 5 rows of the data
iris_data.head()


# # Histograms
# Say we would like to create a histogram to see how petal length varies in iris flowers. We can do this with the sns.distplot command.

# In[ ]:


# Histogram 
sns.distplot(a=iris_data['Petal Length (cm)'], kde=False)


# We customize the behavior of the command with two additional pieces of information:
# 
# a= chooses the column we'd like to plot (in this case, we chose 'Petal Length (cm)').
# kde=False is something we'll always provide when creating a histogram, as leaving it out will create a slightly different plot.
# 
# # Density plots
# 
# The next type of plot is a kernel density estimate (KDE) plot. In case you're not familiar with KDE plots, you can think of it as a smoothed histogram.
# 
# To make a KDE plot, we use the sns.kdeplot command. Setting shade=True colors the area below the curve (and data= has identical functionality as when we made the histogram above).
# 

# In[ ]:


# KDE plot 
sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)


# # 2D KDE plots
# We're not restricted to a single column when creating a KDE plot. We can create a two-dimensional (2D) KDE plot with the sns.jointplot command.
# 
# In the plot below, the color-coding shows us how likely we are to see different combinations of sepal width and petal length, where darker parts of the figure are more likely.

# In[ ]:


# 2D KDE plot
sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")


# Note that in addition to the 2D KDE plot in the center,
# 
# the curve at the top of the figure is a KDE plot for the data on the x-axis (in this case, iris_data['Petal Length (cm)']), and
# the curve on the right of the figure is a KDE plot for the data on the y-axis (in this case, iris_data['Sepal Width (cm)']).
# 
# # Color-coded plots
# 
# For the next part of the tutorial, we'll create plots to understand differences between the species. To accomplish this, we begin by breaking the dataset into three separate files, with one for each species.

# In[ ]:


# Paths of the files to read
iris_set_filepath = "../input/data-for-datavis/iris_setosa.csv"
iris_ver_filepath = "../input/data-for-datavis/iris_versicolor.csv"
iris_vir_filepath = "../input/data-for-datavis/iris_virginica.csv"

# Read the files into variables 
iris_set_data = pd.read_csv(iris_set_filepath, index_col="Id")
iris_ver_data = pd.read_csv(iris_ver_filepath, index_col="Id")
iris_vir_data = pd.read_csv(iris_vir_filepath, index_col="Id")

# Print the first 5 rows of the Iris versicolor data
iris_ver_data.head()


# In the code cell below, we create a different histogram for each species by using the sns.distplot command (as above) three times. We use label= to set how each histogram will appear in the legend.

# In[ ]:


# Histograms for each species
sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris-setosa", kde=False)
sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", kde=False)
sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", kde=False)

# Add title
plt.title("Histogram of Petal Lengths, by Species")

# Force legend to appear
plt.legend()


# In this case, the legend does not automatically appear on the plot. To force it to show (for any plot type), we can always use plt.legend().
# 
# We can also create a KDE plot for each species by using sns.kdeplot (as above). Again, label= is used to set the values in the legend.

# In[ ]:


# KDE plots for each species
sns.kdeplot(data=iris_set_data['Petal Length (cm)'], label="Iris-setosa", shade=True)
sns.kdeplot(data=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", shade=True)
sns.kdeplot(data=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", shade=True)

# Add title
plt.title("Distribution of Petal Lengths, by Species")


# One interesting pattern that can be seen in plots is that the plants seem to belong to one of two groups, where Iris versicolor and Iris virginica seem to have similar values for petal length, while Iris setosa belongs in a category all by itself.
# 
# In fact, according to this dataset, we might even be able to classify any iris plant as Iris setosa (as opposed to Iris versicolor or Iris virginica) just by looking at the petal length: if the petal length of an iris flower is less than 2 cm, it's most likely to be Iris setosa!

# # (Cancer data)

# In this exercise, you will use your new knowledge to propose a solution to a real-world scenario.  To succeed, you will need to import data into Python, answer questions using the data, and generate **histograms** and **density plots** to understand patterns in the data.
# 
# ## Scenario
# 
# You'll work with a real-world dataset containing information collected from microscopic images of breast cancer tumors, similar to the image below.
# 
# ![ex4_cancer_image](https://i.imgur.com/qUESsJe.png)
# 
# Each tumor has been labeled as either [**benign**](https://en.wikipedia.org/wiki/Benign_tumor) (_noncancerous_) or **malignant** (_cancerous_).
# 
# To learn more about how this kind of data is used to create intelligent algorithms to classify tumors in medical settings, **watch the short video [at this link](https://www.youtube.com/watch?v=9Mz84cwVmS0)**!

# 
# 
# ## Setup
# 
# (Run the next cell to import and configure the Python libraries that you need to complete the exercise.)

# ## Step 1: Load the data
# 
# In this step, you will load two data files.
# - Load the data file corresponding to **benign** tumors into a DataFrame called `cancer_b_data`.  The corresponding filepath is `cancer_b_filepath`.  Use the `"Id"` column to label the rows.
# - Load the data file corresponding to **malignant** tumors into a DataFrame called `cancer_m_data`.  The corresponding filepath is `cancer_m_filepath`.  Use the `"Id"` column to label the rows.

# In[ ]:


# Paths of the files to read
cancer_b_filepath = "../input/data-for-datavis/cancer_b.csv"
cancer_m_filepath = "../input/data-for-datavis/cancer_m.csv"

# Fill in the line below to read the (benign) file into a variable cancer_b_data
cancer_b_data = pd.read_csv(cancer_b_filepath, index_col="Id")

# Fill in the line below to read the (malignant) file into a variable cancer_m_data
cancer_m_data = pd.read_csv(cancer_m_filepath, index_col="Id")


# ## Step 2: Review the data
# 
# Use a Python command to print the first 5 rows of the data for benign tumors.

# In[ ]:


# Print the first five rows of the (benign) data
cancer_b_data.head()


# Use a Python command to print the first 5 rows of the data for malignant tumors.

# In[ ]:


# Print the first five rows of the (malignant) data
cancer_m_data.head()


# In the datasets, each row corresponds to a different image.  Each dataset has 31 different columns, corresponding to:
# - 1 column (`'Diagnosis'`) that classifies tumors as either benign (which appears in the dataset as **`B`**) or malignant (__`M`__), and
# - 30 columns containing different measurements collected from the images.
# 
# Use the first 5 rows of the data (for benign and malignant tumors) to answer the questions below.

# In[ ]:


# Fill in the line below: In the first five rows of the data for benign tumors, what is the
# largest value for 'Perimeter (mean)'?
max_perim = 87.46

# Fill in the line below: What is the value for 'Radius (mean)' for the tumor with Id 842517?
mean_radius = 20.57


# # Step 3: Investigating differences
# 
# Part A
# Use the code cell below to create two histograms that show the distribution in values for 'Area (mean)' for both benign and malignant tumors. (To permit easy comparison, create a single figure containing both histograms in the code cell below.)

# In[ ]:


# Histograms for benign and maligant tumors
sns.distplot(a=cancer_b_data['Area (mean)'], label="Benign", kde=False)
sns.distplot(a=cancer_m_data['Area (mean)'], label="Malignant", kde=False)
plt.legend()


# #### Part B
# 
# A researcher approaches you for help with identifying how the `'Area (mean)'` column can be used to understand the difference between benign and malignant tumors.  Based on the histograms above, 
# - Do malignant tumors have higher or lower values for `'Area (mean)'` (relative to benign tumors), on average?
# - Which tumor type seems to have a larger range of potential values?

# *Malignant tumors have higher values for 'Area (mean)', on average. Malignant tumors have a larger range of potential values.*

# ## Step 4: A very useful column
# 
# #### Part A
# 
# Use the code cell below to create two KDE plots that show the distribution in values for `'Radius (worst)'` for both benign and malignant tumors.  (_To permit easy comparison, create a single figure containing both KDE plots in the code cell below._)

# In[ ]:


# KDE plots for benign and malignant tumors
sns.kdeplot(data=cancer_b_data['Radius (worst)'], shade=True, label="Benign")
sns.kdeplot(data=cancer_m_data['Radius (worst)'], shade=True, label="Malignant")


# #### Part B
# 
# A hospital has recently started using an algorithm that can diagnose tumors with high accuracy.  Given a tumor with a value for `'Radius (worst)'` of 25, do you think the algorithm is more likely to classify the tumor as benign or malignant?

# *The algorithm is more likely to classify the tumor as malignant. This is because the curve for malignant tumors is much higher than the curve for benign tumors around a value of 25 -- and an algorithm that gets high accuracy is likely to make decisions based on this pattern in the data.*

# 

# # What have you learned?
# 
# 
# Since it's not always easy to decide how to best tell the story behind your data, we've broken the chart types into three broad categories to help with this.
# 
# Trends - A trend is defined as a pattern of change.
# sns.lineplot - Line charts are best to show trends over a period of time, and multiple lines can be used to show trends in more than one group.
# Relationship - There are many different chart types that you can use to understand relationships between variables in your data.
# sns.barplot - Bar charts are useful for comparing quantities corresponding to different groups.
# sns.heatmap - Heatmaps can be used to find color-coded patterns in tables of numbers.
# sns.scatterplot - Scatter plots show the relationship between two continuous variables; if color-coded, we can also show the relationship with a third categorical variable.
# sns.regplot - Including a regression line in the scatter plot makes it easier to see any linear relationship between two variables.
# sns.lmplot - This command is useful for drawing multiple regression lines, if the scatter plot contains multiple, color-coded groups.
# sns.swarmplot - Categorical scatter plots show the relationship between a continuous variable and a categorical variable.
# Distribution - We visualize distributions to show the possible values that we can expect to see in a variable, along with how likely they are.
# sns.distplot - Histograms show the distribution of a single numerical variable.
# sns.kdeplot - KDE plots (or 2D KDE plots) show an estimated, smooth distribution of a single numerical variable (or two numerical variables).
# sns.jointplot - This command is useful for simultaneously displaying a 2D KDE plot with the corresponding KDE plots for each individual variable.
# 
# # Changing styles with seaborn
# 
# All of the commands have provided a nice, default style to each of the plots. However, you may find it useful to customize how your plots look, and thankfully, this can be accomplished by just adding one more line of code!
# 
# As always, we need to begin by setting up the coding environment. (This code is hidden, but you can un-hide it by clicking on the "Code" button immediately below this text, on the right.)
# 
# 

# In[ ]:


# Path of the file to read
spotify_filepath = "../input/data-for-datavis/spotify.csv"

# Read the file into a variable spotify_data
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)

# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)


# We can quickly change the style of the figure to a different theme with only a single line of code.

# In[ ]:


# Change the style of the figure to the "dark" theme
sns.set_style("dark")

# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)


# Seaborn has five different themes: (1)"darkgrid", (2)"whitegrid", (3)"dark", (4)"white", and (5)"ticks", and you need only use a command similar to the one in the code cell above (with the chosen theme filled in) to change it.
# 
# In the upcoming exercise, you'll experiment with these themes to see which one you like most!

# 

# 

# In[ ]:


# Path of the file to read
spotify_filepath = "../input/data-for-datavis/spotify.csv"

# Read the file into a variable spotify_data
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)


# # Try out seaborn styles
# 
# Run the command below to try out the `"dark"` theme.

# In[ ]:


# Change the style of the figure
sns.set_style("dark")

# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)


# In[ ]:


# Change the style of the figure
sns.set_style("darkgrid")

# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)


# In[ ]:


# Change the style of the figure
sns.set_style("whitegrid")

# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)


# In[ ]:


# Change the style of the figure
sns.set_style("white")

# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)


# In[ ]:


# Change the style of the figure
sns.set_style("ticks")

# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)

