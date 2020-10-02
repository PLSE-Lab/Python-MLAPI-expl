#!/usr/bin/env python
# coding: utf-8

# # Introduction 
# As humans, we always cared and will care about our freedom. 
# To understand what makes us really **free**, I will analyze this dataset and share my observations. 
# 
# # Understanding the Data
# In short, this dataset involves around a research based on some important values that are effecting the human freedom on every country that researches took data. 
# 
# Most important areas and their corresponding column names of this dataset mentioned as follows:
# 
# 1. Rule of Law (#pf_rol)                                                                  
# 2. Security and Safety (#pf_ss)
# 3. Movement (#pf_movement)                                                                  
# 4. Religion (#pf_religion)
# 5. Association, Assembly, and Civil Society (#pf_association)            
# 6. Expression and Information (#pf_expression)
# 7. Identity and Relationships (#pf_identity)                                   
# 8. Size of Government (#ef_government)
# 9. Legal System and Property Rights (#ef_legal)                      
# 10. Access to Sound Money (#ef_money)
# 11. Freedom to Trade Internationally (#ef_trade)                       
# 12. Regulation of Credit, Labor, and Business (#ef_regulation)
# 
# 
# ### On this Notebook, you will find observations based mostly on the 12 fields above.
# ### After field related observations, I will also do an analysis for few Most, Average and Least Scored Countries on 2018.
# 
# For More information about the Data, You can check the following links.
# 
# The dataset overwiev can be found here: <a href="https://www.kaggle.com/gsutters/the-human-freedom-index/home" target="_blank"> The Human Freedom Index (Kaggle) </a>
# 
# Original Research can be found here: <a href="https://www.cato.org/human-freedom-index-new" target="_blank"> The Human Freedom Index (CATO) </a>

# # So Let's Start !

# Importing the required libraries and loading the Dataset

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

OriginalDataSet = pd.read_csv("../input/hfi_cc_2018.csv")


# # EDA

# In[ ]:


OriginalDataSet.info()  # more columns than I desired to work for


# In[ ]:


OriginalDataSet.head(3)  # Looking to the Columns


# Reducing the amount of Columns. (I want to work on only mentioned fields at the list above )

# ## New Simple DataSet + EDA

# In[ ]:


SimplerDataSet = pd.DataFrame() # creating a empty dataset

SimplerDataSet["pf_rule_of_law"] = OriginalDataSet.pf_rol            # Rule Of Law
SimplerDataSet["pf_security_safety"] = OriginalDataSet.pf_ss         # Security and Safety
SimplerDataSet["pf_movement"] = OriginalDataSet.pf_movement          # Movement
SimplerDataSet["pf_religion"] = OriginalDataSet.pf_religion          # Religion
SimplerDataSet["pf_association"] = OriginalDataSet.pf_association    # Association, Assembly, and Civil Society
SimplerDataSet["pf_expression"] = OriginalDataSet.pf_expression      # Expression and Information
SimplerDataSet["pf_identity"] = OriginalDataSet.pf_identity          # Identity and Relationships
SimplerDataSet["ef_government"] = OriginalDataSet.ef_government      # Size of Government
SimplerDataSet["ef_legal"] = OriginalDataSet.ef_legal                # Legal System and Property Rights
SimplerDataSet["ef_money_access"] = OriginalDataSet.ef_money         # Access to Sound Money
SimplerDataSet["ef_trade"] = OriginalDataSet.ef_trade                # Freedom to Trade Internationally
SimplerDataSet["ef_regulation"] = OriginalDataSet.ef_regulation      # Regulation of Credit, Labor, and Business

SimplerDataSet["country"] = OriginalDataSet.countries                # Name of the Country
SimplerDataSet["year"] = OriginalDataSet.year                        # Year of Observation
SimplerDataSet["eco_free_score"] = OriginalDataSet.ef_score          # Economical Freedom Score
SimplerDataSet["eco_free_rank"] = OriginalDataSet.ef_rank            # Economical Freedom Rank
SimplerDataSet["free_score"] = OriginalDataSet.hf_score              # Human Freedom Score
SimplerDataSet["free_rank"] = OriginalDataSet.hf_rank                # Human Freedom Rank


# In[ ]:


SimplerDataSet.head()  # first few rows


# In[ ]:


SimplerDataSet.info()


# In[ ]:


SimplerDataSet.describe()  # statitical description of new dataset


# # Visualizations

# In[ ]:


plt.figure(figsize=(14,10))
plt.title("CORRELATION HEATMAP",fontsize=20)
sns.heatmap(data=SimplerDataSet.corr(),cmap="PRGn_r",annot=True, fmt='.2f', linewidths=1)
plt.show()


# info:
# * Dark Green Cells are showing High amount of **Negative** Correlation
# * Dark Purple Cells are showing High amount of **Positive** Correlation
# * Bright Colored Cells has (no / little) correlation.
# 
# Some interesting Observations from the Heatmap Above:
# * Size of Government Doesn't effect the freedom
# * Religion Doesn't effect the freedom as much as people would think
# * Rule Of Law has very big impact on both Economic and Human Freedom
# * International Trade has bigger correlation than Access to Money on Economic Freedom Score

# In[ ]:


freedom_class = SimplerDataSet.free_score.round(decimals=0)
economy_class = SimplerDataSet.eco_free_score.round(decimals=0)

plt.figure(figsize=(28,28))

plt.suptitle("StoryTeller Scatter Plots",fontsize=20)

plt.subplot(2,2,1)
sns.scatterplot(data=SimplerDataSet,x="pf_rule_of_law",y="ef_legal",hue=freedom_class, palette="RdYlGn",alpha=0.7,size=economy_class, sizes=(10,200))
plt.xlabel("Rule Of Law")
plt.ylabel("Legal System and Property Rights")
plt.grid()

plt.subplot(2,2,2)
sns.scatterplot(data=SimplerDataSet,x="ef_trade",y="ef_money_access",hue=freedom_class, palette="RdYlGn",alpha=0.7,size=economy_class, sizes=(10,200))
plt.xlabel("Freedom to Trade Internationally")
plt.ylabel("Access to Sound Money")
plt.grid()

plt.subplot(2,2,3)
sns.scatterplot(data=SimplerDataSet,x="pf_security_safety",y="pf_expression",hue=freedom_class, palette="RdYlGn",alpha=0.7,size=economy_class, sizes=(10,200))
plt.xlabel("Security and Safety")
plt.ylabel("Expression and Information")
plt.grid()

plt.subplot(2,2,4)
sns.scatterplot(data=SimplerDataSet,x="pf_association",y="ef_regulation",hue=freedom_class, palette="RdYlGn",alpha=0.7,size=economy_class, sizes=(10,200))
plt.xlabel("Association, Assembly, and Civil Society")
plt.ylabel("Regulation of Credit, Labor, and Business")
plt.grid()

plt.show()


# * Size of the each scatter is equals to size Economical Freedom Score.
# * Color of the each scatter is defined by their Human Freedom Score.
# 
# ** With the X Coordinate, Y Coordinate, Size and Color of scatters, I tried to show 4 different information on each plot. **

# # Worst and Best Countries (in terms of freedom)

# In[ ]:


Best_3_free = SimplerDataSet[SimplerDataSet["free_rank"] <= 5]
Best_3_eco = SimplerDataSet[SimplerDataSet["eco_free_rank"] <= 5]

Worst_3_free = SimplerDataSet[SimplerDataSet["free_rank"] >= 158]
Worst_3_eco = SimplerDataSet[SimplerDataSet["eco_free_rank"] >= 158]


# In[ ]:


plt.figure(figsize=(20,16))

plt.suptitle("Amount Of Top 5 and Bottom 5 Appearencies",fontsize=16)

plt.subplot(2,2,1)
plt.title("Top 5 Appearencies on Freedom Rank")
Best_3_free.country.value_counts().plot.bar()


plt.subplot(2,2,2)
plt.title("Top 5 Appearencies on Economical Freedom Rank")
Best_3_eco.country.value_counts().plot.bar()


plt.subplot(2,2,3)
plt.title("Worst 5 Appearencies on Freedom Rank")
Worst_3_free.country.value_counts().plot.bar()


plt.subplot(2,2,4)
plt.title("Worst 5 Appearencies on Economical Freedom Rank")
Worst_3_eco.country.value_counts().plot.bar()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()


# ## Not Complete yet! open for your Comments...

# In[ ]:




