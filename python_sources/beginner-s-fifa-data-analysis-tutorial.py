#!/usr/bin/env python
# coding: utf-8

# If you want to work with any form of datasets in python, you most likely want to use pandas package to do so:

# In[ ]:


import pandas as pd


# The read_csv function gets the file and reads it into a DataFrame. In reality if you weren't using kaggle you would need to use the file location instead of "../input/file_name.csv"
# ![](https://images-eds-ssl.xboxlive.com/image?url=8Oaj9Ryq1G1_p3lLnXlsaZgGzAie6Mnu24_PawYuDYIoH77pJ.X5Z.MqQPibUVTc5o.Gms14yKJcD4Yna5HIccFnKcFU7N.HfH0oe2YIHkHcSP3.uPm9yD_U26aVhyibHwYPIMI_D5o2i5AH_b8Jso8hBAvD4NO6ly.R7XqXacvDtHysXxVzDWFY4.ArFKbYZZI.4fD_m8P_ZCC_qaBOWjxH8fKr9tKaDFQwyFm5_.8-&w=200&h=300&format=jpg)
# Next we will read the FIFA 19 player dataset:

# In[ ]:


player_dataset = pd.read_csv("../input/data.csv", index_col = 0) #save dataset as pd DataFrame object


# To look at the dataset  we use the head function:

# In[ ]:


player_dataset.head(5)


# DataFrame's columns are Series, in which the column names are 'attributes' that can be accessed with a dot operator. For ex:

# In[ ]:


player_dataset.Nationality.head(5)


# We want to rank by nationality (top 10), counting how many players are from that nationality

# In[ ]:


nationality_count = player_dataset.sort_values(by='Nationality').Nationality.value_counts()
nationality_count.head(10)


# Let's say that we are interested in how many Brazilian players there are:

# In[ ]:


brazilian_players = player_dataset.loc[(player_dataset.Nationality == "Brazil")]
brazilian_players.head(5)


# This is just a basic data analysis. You can do more cool indexing, boolean sorting, etc. with all the different attributes available to manipulate the player dataset. If you play FIFA 19 maybe this could help you search up players and do interesting research. In terms of doing machine learning all you need to really know is how to import files and process them so you can train and test your models, you won't need to exactly analyze the data in a complex manner unless you want to.
# 
# [Pandas Cheetsheet](https://assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf)
# 
# created by rae385, for Coppell High School AI Club

# In[ ]:




