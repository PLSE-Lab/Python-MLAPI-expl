#!/usr/bin/env python
# coding: utf-8

# # **An Exploration on the Diversity of Silicon Valley Companies **
# ## **A visualization on the diversity of the top tech companies in Silicon Valley in 2016**
# By: Joe Akanesuvan
# 
# The lack of a diverse community has been an issue in Silicon Valley for the longest time. Despite the many tech companies' attempt to crack the minority ceiling, the gender and race problem within Silicon Valley seems be getting worse. The goal of the analysis of this dataset is to analyze this particular situation within the top tech companies of Silicon Valley, and to, ultimately, see whether these companies are doing their best to fulfill their pledges to create a more diverse work environment in Silicon Valley.
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import warnings; warnings.simplefilter('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv("../input/Reveal_EEO1_for_2016.csv")


# ### **Understanding the Dataset**
# 
# For this analyze, we have been given the race and gender for the employee of each company. We will first conduct a basic analysis by comparing the amount of male and female, and races of the employees in the companies listed in the dataset. After an initial understanding, we can then look into the roles of these employees within the company.

# In[ ]:


data.head()


# In[ ]:


data.info()


# ### **Gender Diversity in Silicon Valley**

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'data["count"] = data["count"].convert_objects(convert_numeric=True)')


# In[ ]:


companyGender = pd.DataFrame({"count": data.groupby(["company", "gender"])["count"].sum()}).reset_index()


# In[ ]:


genderData = companyGender[companyGender["gender"] == "male"]
genderData.drop(["gender"], inplace=True, axis=1)
genderData.columns.values[1] = "male_count"

femaleData = companyGender[companyGender["gender"] == "female"]
femaleData.columns = ["company", "gender", "female_count"]

genderData = genderData.assign(female_count = femaleData["female_count"].values)

male_prop = genderData["male_count"] / (genderData["male_count"] + genderData["female_count"])
genderData = genderData.assign(male_proportion = male_prop)

genderData = genderData.assign(female_proportion = (1 - male_prop))

genderScore = 1 - (genderData["male_proportion"]**2 + genderData["female_proportion"]**2) 
genderData = genderData.assign(gender_score = genderScore)
genderData.sort_values(["gender_score"], ascending=False)


# To analyze the gender diversity of the tech companies, we will create a scoring index which will be use to capture how diverse a company is gender-wised. The scoring scheme expresses the probability that any two randomly chosen employees will be of a different gender, and can be easily calculated by this equation
# 
# **P(Different Gender) = 1 - P(Same Gender)**, where P(Same Gender) = P(Male)<sup>2</sup> +  P(Female)<sup>2</sup>

# In[ ]:


np.unique(data["race"])


# In[ ]:


companyRace = pd.DataFrame({"count": data.groupby(["company", "race"])["count"].sum()}).reset_index()


# In[ ]:


asianData = companyRace[companyRace["race"] == "Asian"]
asianData.drop(["race"], inplace=True, axis=1)
asianData.columns.values[1] = "asian_count"

# femaleData = companyGender[companyGender["gender"] == "female"]
# femaleData.columns = ["company", "gender", "female_count"]

# genderData = genderData.assign(female_count = femaleData["female_count"].values)

# male_prop = genderData["male_count"] / (genderData["male_count"] + genderData["female_count"])
# genderData = genderData.assign(male_proportion = male_prop)

# genderData = genderData.assign(female_proportion = (1 - male_prop))

# genderScore = 1 - (genderData["male_proportion"]**2 + genderData["female_proportion"]**2) 
# genderData = genderData.assign(gender_score = genderScore)
# genderData.sort_values(["gender_score"], ascending=False)

