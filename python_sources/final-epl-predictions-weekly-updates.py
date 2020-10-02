#!/usr/bin/env python
# coding: utf-8

# # Final EPL Predictions - weekly updates

# Following on from [this piece](https://www.kaggle.com/mattharrison79/how-to-get-into-the-epl-top-4), this notebook will include all the weekly table predictions. I will update as we go. 

# # Week 2

# In[ ]:


import pandas as pd
Week_2_table = pd.read_csv('../input/predicted-tables/predicted_table_week2.csv')
Week_2_table = Week_2_table.reset_index(drop=True)
Week_2_table = Week_2_table.drop(['Positional difference'], axis=1)
print(Week_2_table)


# # Week 3

# In[ ]:


Week_3_table = pd.read_csv('../input/week3table/predicted_table_week3.csv')
Week_3_table = Week_3_table.reset_index(drop=True)
Week_3_table = Week_3_table.drop(['Positional difference'], axis=1)
Week_3_table = Week_3_table.round(3)
print(Week_3_table)


# # Week 4

# In[ ]:


Week_4_table = pd.read_csv('../input/week4table/predicted_table_week4.csv')
Week_4_table = Week_4_table.reset_index(drop=True)
Week_4_table = Week_4_table.drop(['Positional difference'], axis=1)
Week_4_table = Week_4_table.round(3)
print(Week_4_table)


# # Week 5

# In[ ]:


Week_5_table = pd.read_csv('../input/week5table2/predicted_table_week5.csv')
Week_5_table = Week_5_table.reset_index(drop=True)
Week_5_table = Week_5_table.drop(['Positional difference'], axis=1)
Week_5_table = Week_5_table.round(3)
print(Week_5_table)


# # Week 6

# In[ ]:


Week_6_table = pd.read_csv('../input/week6table/predicted_table_week6.csv')
Week_6_table = Week_6_table.reset_index(drop=True)
Week_6_table = Week_6_table.drop(['Positional difference'], axis=1)
Week_6_table = Week_6_table.round(3)
print(Week_6_table)


# # Week 7

# In[ ]:


Week_7_table = pd.read_csv('../input/week7table/predicted_table_week7.csv')
Week_7_table = Week_7_table.reset_index(drop=True)
Week_7_table = Week_7_table.drop(['Positional difference'], axis=1)
Week_7_table = Week_7_table.round(3)
print(Week_7_table)


# # Week 8

# A slight change this week due to Liverpool's 100% record so far in the league but Man City's 8 goals against Watford.
# 
# Despite the 8 point gap between Liverpool and City, the model was still predicting City top due to 7 more goals scored. I knew this model would be limited due to the small amount of variables but this seemed to be completely pointless with the current actual difference in points.
# 
# To attempt to address this, I've added a 3rd variable. Initially I was going to use "wins" but as I already had two variables that I was extrapolating out to the end of the season, I thought adding a third in this way would not add much to the model. Instead I have used win percentage as it is a current figure that does not need to be extrapolated to the end of season. 
# 
# Therefore the model now considers a team's goals score and conceded, projects them to the end of the season but also includes their current win percentage to estimate the amount of games they may win. I aim to return to this metric at some point to cross check it's predictive validity against simply the number of wins. 

# In[ ]:


Week_8_table = pd.read_csv('../input/week8table/predicted_table_week8.csv')
Week_8_table = Week_8_table.reset_index(drop=True)
Week_8_table = Week_8_table.round(3)
print(Week_8_table)


# # Week 9

# In[ ]:


Week_9_table = pd.read_csv("../input/week9table/predicted_table_week9.csv")
Week_9_table = Week_9_table.reset_index(drop=True)
Week_9_table = Week_9_table.round(3)
print(Week_9_table)


# # Week 10

# In[ ]:


Week_10_table = pd.read_csv("../input/week10to12table/predicted_table_week10.csv")
Week_10_table = Week_10_table.reset_index(drop=True)
Week_10_table = Week_10_table.round(3)
print(Week_10_table)


# # Week 11
