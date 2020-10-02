#!/usr/bin/env python
# coding: utf-8

# ## This notebook is for beginners on how to get started using the dataset and is meant to showcase the significance of each column in the dataset. There is more to explore using this dataset, so, I've kept it simple for beginners.
# 
# ### Feel free to create your own kernels!
# 
# ### Share this dataset with your friends who can make the most of out of it!
# 
# ### **Thank you!**

# ## Importing Libraries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# # 1.1 Loading the dataframe

# In[ ]:


df = pd.read_csv("/kaggle/input/calories-burned-during-exercise-and-activities/exercise_dataset.csv")
df.head()


# # 1.2 Calculating calories burned according to weight
# 
# We can calculate the calories burned using the **"Calories per kg"** column. We can use the following piece of code:
# 
# > weight * df.iloc[exercise_index]["Calories per kg"]
# 
# * exercise_index : Index of the exercise in the dataframe

# In[ ]:


# Here the weight is 190 lb, you change the weight
weight = 190

# Let us assume that the exercise is cycling using a mountain bike (BMX) for 1 hour
# This exercise is at index 0 in df

print("Calories burned:", weight*df.iloc[0]["Calories per kg"] , "kcal")


# # 1.3 Visualizing the calories burned by each exercise

# In[ ]:


plt.scatter(df.index, df["130 lb"])
plt.xlabel("Index (exercise)")
plt.ylabel("Calories burned")
plt.title("Scatter plot for calories burned vs exercise index (130 lb)")
plt.show()


# # 1.4 Getting information about the most intensive exercises/activities
# 
# Here you can see the most intensive exercises sorted on the basis of the calories burned per hour.

# In[ ]:


df.sort_values('Calories per kg', ascending=False)

