#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## 1. General Information
# 
# The `dodgers.csv` file displays information of Dodger home gamedays. Each row represents a home game in the season. Each column displays a piece of information about the home game. 
# 
# **General Information:**
# * **month** = Month the game was held.
# * **day** = Date the game was held.
# * **attend** = Number of attendees at the game.
# * **day_of_week** = Day the game was held.
# * **opponent** = Name of the opposing team.
# * **temp** = Temperature during the game.
# * **skies** = Weather conditions of the game. 
# * **day_night** = Whether the game was played in the day or night. 
# 
# **Promotion:**
# * **cap** = Yes, if cap was given as a promotion. No, if otherwise. 
# * **shirt** = Yes, if shirt was given as a promotion. No, if otherwise. 
# * **firweworks** = Yes, if fireworks was given as a promotion. No, if otherwise. 
# * **bobblehead** = Yes, if bobblehead was given as a promotion. No, if otherwise. 

# ### a. Load the dataset and display the first 10 rows. Name the dataset `dodgers`.
# 
# > Use `pd.read_csv()` to upload the dataset from a .csv file to kaggle. Use `.head()` to display the top rows of the dataset. By default, `.head()` will display the top 5 rows. Enter `10` inside the paranthesis to display the top 10 rows. 

# In[ ]:


dodgers = pd.read_csv('/kaggle/input/dodgers-game-day-information/dodgers.csv')
dodgers.head(10)


# ### b. How many games did the Dodgers play at home?
# 
# > Use `.shape` to determine the number of rows and columns in the dataframe. Using `.shape` will print (number of rows, number of columns). To print only the number of rows, use `.shape[0]` and to print only the number of columns use `.shape[1]`. 
# 
# 

# In[ ]:


dodgers.shape[0]


# *The Dodgers played 81 games at home.* 
# 
# *This is useful information because we know that the data displays information from an entire season. An MLB season consist of 162 games, and half of the season games (81) are played at home.*

# ### c. Who did the Dodgers play against in the last 5 home games of the season?
# 
# Display the rows of the last 5 games of the season. What insights can we draw from this subset?
# 
# > Use `.iloc[]` to select rows and columns by an integer, in the order that they appear in the data frame. For example, `iloc[0:5]` will select the first 5 rows of the dataset. To select the last 5 rows, use `[-5:]`.

# In[ ]:


dodgers.iloc[-5:]


# *The last 5 home games of the season were played against the Rockies and the Giants.*
# 
# *Furthermore, there were no promotions given in the last 5 games of the season. This is probably because the Dodgers believed that they did not need to give a promotion to attract an audience in their final games. However, interestingly, attendance was relatively low in the last home game of the season at ~34,000 compared to ~42,000 in the game before. Thus, we can conclude that the Dodgers may have not had a good season since attendance was not at its highest in the final game.*

# ## 2. Attendance at Dodger Home Games

# ### a. What was the average attendance of the Dodger home games?
# 
# > To determine the average attendance, use `.attend` to select the attendance column (attend) of the data frame, followed by `.mean()` to determine the average of the attendance column. 

# In[ ]:


dodgers.attend.mean()


# *The average attendance per Dodger home game was 41,040 people. The average attendance can be helpful in determining the average profit per game. For example, if fixed costs are 400,000 dollars, variable costs are 10  dollars per person, and revenue per ticket is 25 dollars, what is the average profit of Dodger home games? *
# 
# 41040 * (25-10) - 400000 = 215600
# 
# *Thus, average profit of Dodger home games is 215,600 dollars.*
# 

# ### b. What was the highest attendance of Dodger home games this season?
# 
# > As in the prior question, use `.attend` to select the attendance column. Use `.max()` to return the highest value in the attendance column.

# In[ ]:


dodgers.attend.max()


# *The highest attendance of a Dodger home game this season was 56,000 people.*
# 
# *This is nearly 15,000 people more than the average attendance. Based on the assumptions about revenue and costs from the previous question, this could generate approximately 225,000 dollars more in revenue. Are there any correlations or causes of this high attendance? Lets look further in the next question.*

# ### c. What information and conclusions can we draw about the games with the highest attendance?
# 
# Display the row(s) that have the highest attendance, as found in the previous question.
# 
# > Use `.query()` to make a selection based on a condition or expression. We are looking for rows that have an attendance of 56000. Thus, use `'attend == 56000'` inside the `.query()` method. 

# In[ ]:


dodgers.query('attend == 56000')


# *The games with the highest attendance was the 1st and the 60th home game of the season, as indicated by the row number to the left. It makes sense that the first game had the highest attendance since it was the season opener, which most fans like to attend. There were no promotions offered in the first game, so we know that attendees wanted to be at the game and were not influenced by a promotion. The weather was also nice during the day at 67 degrees Farenheit and clear skies, which may have also influenced the attendance.* 
# 
# *For the 60th game, the Dodgers played against the (New York) Giants, which may have influenced attendance, since it is New York, which tends to be a rival to Los Angeles. Additionally, the bobblehead promotion may have been another influencing factor, considering that dedicated fans collect Dodger bobbleheads. Moreover, the weather was nice for a summer night (AUG 21) at 75 degrees Farenheit and clear skies.* 
# 
# *Similarities in both games include relatively nice weather and day of the week--Tuesday. It is interesting however, that both games had high attendance on a Tuesday, which is a weekday and most would expect there to be lower attendance.*
# 
# *Thus, based on the data we can conclude that high attendance may be infleunced by these factors: the season opener home game, relatively nice weather, the opponent, and offering a  bobblehead promotion. Additionally, we can assume that Tuesday game days will not negatively impact attendance.*

# ### d. Which promotion type drives the highest attendance on average?
# 
# Create a bar chart illustrating the average attendance by promotion type, including cap, shirt, fireworks, bobbleheads, and none (no promotion). What further insights can we make from this analysis?
# 
# > Use `.groupby()` to split the data into groups based on the specified criteria. In this case, we want to split the data into the various promotion types, `['cap', 'shirt', 'fireworks', 'bobblehead']`. To view the attendance based on the criteria, use `.attend`, which will organize the attendance column by the promotion type. To find the average of the attendance use `.mean()`. Sort the data to determine the highest average attendance based on the promotion, using `.sort_values()`. Using `.sort_values()` will sort the data in ascending order, but we want to organize in descending order to view the highest value at the beginning. Thus, we will enter `.sort_values(ascending=False)`. Lastly, to create the bar chart, use `.plot.bar()`.
# 
# > For visualization purposes, I have renamed the labels on the x and y-axis.
# 
# 

# In[ ]:


lad = dodgers.groupby(['cap', 'shirt', 'fireworks', 'bobblehead']).attend.mean().sort_values(ascending=False).plot.bar()
lad.set_xticklabels(['bobblehead', 'shirt', 'fireworks', 'none', 'cap'], rotation=0, fontsize=12)
lad.set_xlabel("Promotion Type", fontsize=12)
lad.set_ylabel("Attendance", fontsize=12)


# *The bobblehead promotion drives the highest attendance on average with over 50,000 attendees. This is followed by the shirts, fireworks, no promotion, and cap. Considering that bobbleheads attract several fans, it may be beneficial for the Dodgers to continue offering and expanding the bobblehead promotion. Furthermore, since having no promotion drives more attendance than the cap promotion, the Dodgers should consider potentially removing the cap promotion from their offerings. *

# ### e. How many times did the Dodgers offer the bobblehead promotion?
# 
# > To select the bobblehead column, use `.bobblehead`. To count the number of times the promotion was given, use `.value_counts()` which will count the number of unique instances in the column. In this case, it will count the number of 'YES' and 'NO' in the bobblehead column.

# In[ ]:


dodgers.bobblehead.value_counts()


# *The dodgers offered the bobblehead promotion in 11 home games. Considering this insight, we can now pose further questions: Does the bobblehead promotion drive high attendance because it is a rare and limited offering? Or should the Dodgers instead be promoting bobbleheads more often because we notice that it results in high attendance?*
