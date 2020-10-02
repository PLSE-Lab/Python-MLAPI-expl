#!/usr/bin/env python
# coding: utf-8

# # EDA Project on Suicide Rates in India between 2001-2012
# 
# <img src="https://user-images.githubusercontent.com/53637541/83379224-a691a080-a3f8-11ea-833e-c3f6bdea209a.jpg">

# ## Problem Statement
# 
# This dataset contains yearly suicide rate details of all the states and union territories of India by various categories:suicide causes, education status, by means adopted, professional status, social status between 2001 to 2012.This project aims to identify what are the major causes for suicide and provides insights to help people avoid commiting suicide. 
# 
# National Crime Records Bureau (NCRB), Govt of India has shared this dataset under Govt. Open Data License - India.
# 

# ## Choosing Right Tools

# In[ ]:


import numpy as np
import pandas as pd
pd.options.display.max_columns = 30
pd.options.display.max_rows = 30
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows",None)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ## Loading the Data

# In[ ]:


data = pd.read_csv("/kaggle/input/suicides-in-india/Suicides in India 2001-2012.csv")
data.head(3)


# 
# ### Description of the Dataset
# 
# | Column Name                   | Description                                                                                |
# | ------------------------------|:------------------------------------------------------------------------------------------:| 
# |  State                        | Includes all states and union territories
# |  Year                         | Period from 2001 to 2012
# |  Type_code                    | Various parameters for commiting suicide (Major types)
# |  Type                         | Various causes for commiting suicide (Minor types)
# |  Gender                       | Represents either Male or Female   
# |  Age_group                    | Represents different age groups
# |  Total                        | Total number of suicides
# 
# 

# In[ ]:


data.info()


# **Observation**
# - There are a total of __237519 samples (rows)__ and __7 columns__ in the dataframe.
# - There are __2 columns__ with a **numeric** datatype and __5 columns__ with an **object** datatype.
# - __Year__ column has to be changed to __datetime datatype__.
# 

# ## Data Processing

# In[ ]:


df = data.copy()
df.rename({"Type":"TypeofCauses","Type_code":"Category"}, axis = "columns", inplace = True)
df.head(5)


# In[ ]:


df.State.value_counts()


# In[ ]:


#Replace Delhi(ut) with Delhi
df.replace("Delhi (Ut)", "Delhi", inplace = True)


# In[ ]:


#Removing Total(States), Total(Uts), Total (All India)
df = df.drop(df[(df.State == "Total (States)") |(df.State == "Total (Uts)") | (df.State == "Total (All India)") ].index)
df.State.value_counts()


# ## Exploratory Data Analysis

# ### Analysis based on State

# #### Which States has the highest and the lowest suicide rate?

# In[ ]:


f, ax = plt.subplots(1,2, figsize = (15,7))
df.groupby("State")["Total"].sum().sort_values(ascending = False)[:10].plot(kind = "bar", color = "red", ax= ax[0])                                                                           
plt.ylabel("No.of Suicides")
df.groupby("State")["Total"].sum().sort_values(ascending = True)[:10].plot(kind = "bar",  color = "blue", ax= ax[1])
plt.ylabel("No.of Suicides")                                                                          
ax[0].title.set_text('Top 10 States with highest suicides')
ax[1].title.set_text('Top 10 States with lowest suicides')


# **Observations**
# - Maharashtra, West Bengal & Tamil Nadu are the states with highest suicide rates.
# - Nagaland, Manipur & Mizoram are the states with lowest suicide rates.
# - Union Territories has the lowest suicide rates (Lakshadweep, Daman & Diu).

# #### What are the most common reasons for committing suicide?

# In[ ]:


df.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:10].plot(kind = "bar", color = "y", figsize = (15,7))
plt.ylabel("No. of suicides")
plt.title("Most common reasons for committing suicide")


# **Observations**
# - Married people commit more suicide.
# - No education as well as less education are also the reason for committing suicide. 

# #### What are the reasons for committing suicide in the top 3 states which has highest suicide rate?

# In[ ]:


f, ax = plt.subplots(2,2, figsize = (15,12), constrained_layout = True)
mh = df[df["State"] == "Maharashtra"]
mh_cause = mh[mh["Category"] == "Causes"]
mh_edu = mh[mh["Category"] == "Education_Status"]
mh_adop = mh[mh["Category"] == "Means_adopted"]
mh_prof = mh[mh["Category"] == "Professional_Profile"]
mh_socio = mh[mh["Category"] == "Social_Status"]

mh_edu.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[0,0], color = "b", title = "Reason for suicide based on Education")
mh_adop.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[0,1], color = "r",title ="Reason for suicide based on Means adopted")
mh_prof.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[1,0], color = "g",title = "Reason for suicide based on Professional Profile")
mh_socio.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[1,1], color = "c",title = "Reason for suicide based on Social Status")
plt.setp(ax[0,0], xlabel="")
plt.setp(ax[0,1], xlabel="")
plt.setp(ax[1,0], xlabel="")
plt.setp(ax[1,1], xlabel="")
f.suptitle("Most common Reasons for committing suicide in Maharashtra" + "\n")


# **Observations for Maharashtra**
# - In terms of Education status, people with less education (Primary, Middle) commit more suicide.
# - In terms of Means Adopted, people commit more suicides by hanging and consuming insectidies. 
# - In terms of Professional status, farmers commit more suicide.
# - In terms of Social status, married people are committing more suicide. 

# In[ ]:


f, ax = plt.subplots(2,2, figsize = (15,12), constrained_layout = True)
wb = df[df["State"] == "West Bengal"]
wb_cause = wb[wb["Category"] == "Causes"]
wb_edu = wb[wb["Category"] == "Education_Status"]
wb_adop = wb[wb["Category"] == "Means_adopted"]
wb_prof = wb[wb["Category"] == "Professional_Profile"]
wb_socio = wb[wb["Category"] == "Social_Status"]

wb_edu.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[0,0],color = "b", title = "Reason for suicide based on Education")
wb_adop.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[0,1],color = "r", title ="Reason for suicide based on Means adopted")
wb_prof.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[1,0],color = "g", title = "Reason for suicide based on Professional Profile")
wb_socio.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[1,1],color = "c", title = "Reason for suicide based on Social Status")
plt.setp(ax[0,0], xlabel="")
plt.setp(ax[0,1], xlabel="")
plt.setp(ax[1,0], xlabel="")
plt.setp(ax[1,1], xlabel="")
f.suptitle("Most common reasons for committing suicide in West Bengal" + "\n")


# **Observations for West Bengal**
# - In terms of Education status, people with less education (Primary, Middle) commit more suicide.
# - In terms of Means Adopted, people commit more suicides by hanging and consuming poison. 
# - In terms of Professional status, housewives commit more suicide.
# - In terms of Social status, married people are committing more suicide. 

# In[ ]:


f, ax = plt.subplots(2,2, figsize = (15,12), constrained_layout = True)
tn = df[df["State"] == "Tamil Nadu"]
tn_cause = tn[tn["Category"] == "Causes"]
tn_edu = tn[tn["Category"] == "Education_Status"]
tn_adop = tn[tn["Category"] == "Means_adopted"]
tn_prof = tn[tn["Category"] == "Professional_Profile"]
tn_socio = tn[tn["Category"] == "Social_Status"]

tn_edu.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[0,0],color = "b", title = "Reason for suicide based on Education")
tn_adop.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[0,1],color = "r", title ="Reason for suicide based on Means adopted")
tn_prof.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[1,0],color = "g", title = "Reason for suicide based on Professional Profile")
tn_socio.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[1,1],color = "c", title = "Reason for suicide based on Social Status")
plt.setp(ax[0,0], xlabel="")
plt.setp(ax[0,1], xlabel="")
plt.setp(ax[1,0], xlabel="")
plt.setp(ax[1,1], xlabel="")
f.suptitle("Most common reasons for committing suicide in Tamil Nadu" + "\n")


# **Observations for Tamil Nadu**
# - In terms of Education status, people with less education as well as no education commit more suicide.
# - In terms of Means Adopted, people commit more suicides by hanging and consuming poison. 
# - In terms of Professional status, housewives commit more suicide (Even though Others has highest count of suicide rate, there is no specific reason)
# - In terms of Social status, married people are committing more suicide. 

# ### Analysis based on Age

# #### Which age group commits more suicide?

# In[ ]:


df_Age = df[df["Age_group"] != "0-100+"]
df_nonzero = df_Age[df_Age["Total"] != 0]
df_Age.groupby("Age_group")["Total"].sum().sort_values(ascending = False).plot(kind = "pie", explode = [0.02,0.02,0.02,0.02,0.02],
                                                                              autopct = "%3.1f%%", figsize = (15,7), shadow = False)
plt.title("Suicide rate based on Age group")
plt.ylabel("Different Age groups")


# **Observations**
# - People who are aged between 15 to 44 are committing more suicide.
# - It's quite shocking that even kids also commit suicide.

# #### What is the reason for committing suicide among childrens?

# In[ ]:


child = df[df["Age_group"] == "0-14"]
child.groupby(["TypeofCauses", "State"])["Total"].max().sort_values(ascending = False)[:10].plot(kind = "bar", figsize = (15,7), color = "m")
plt.title("Reasons for children who commit suicide")
plt.ylabel("No. of suicides")


# **Observation**
# - Students are committing more suicides, particularly in West Bengal with a maximum of 184.
# 

# #### What is the reason for committing suicide among different age groups?

# In[ ]:


f, ax = plt.subplots(2,2, figsize = (20,18), constrained_layout = True)
df[df["Age_group"] == "15-29"].groupby("TypeofCauses")["Total"].sum().sort_values(ascending = False)[:10].plot(kind = "bar", fontsize = 10, ax = ax[0,0], color = "g", title = "Age 15 to 29")
df[df["Age_group"] == "30-44"].groupby("TypeofCauses")["Total"].sum().sort_values(ascending = False)[:10].plot(kind = "bar", fontsize = 10, ax = ax[0,1], color = "c", title = "Age 30 to 44")
df[df["Age_group"] == "45-59"].groupby("TypeofCauses")["Total"].sum().sort_values(ascending = False)[:10].plot(kind = "bar", fontsize = 10, ax = ax[1,0], color = "b", title = "Age 45 to 59")
df[df["Age_group"] == "60+"].groupby("TypeofCauses")["Total"].sum().sort_values(ascending = False)[:10].plot(kind = "bar", fontsize = 10, ax = ax[1,1], color = "r", title = "Age 60+")
plt.setp(ax[0,0], xlabel = "")
plt.setp(ax[0,1], xlabel = "")
plt.setp(ax[1,0], xlabel = "")
plt.setp(ax[1,1], xlabel = "")
f.suptitle("Reasons for committing suicide among different Age groups" + "\n")


# **Observations**
# - Age group 15 to 29 & 30 to 44 commit most suicides by hanging.
# - Age group 45 to 59 commit most suicides by hanging and due to family problems. 
# - Age group 60+ commit most suicides by hanging and due to prolonged illness.

# ### Analysis based on gender

# #### Under what category does gender based suicides are higher?

# In[ ]:


sns.barplot(data = df, x = "Category", y = "Total", hue = "Gender", palette = "viridis")
plt.xticks(rotation = 45)
plt.figure(figsize = (15,7))


# **Observation**
# - It is clear that the suicide rate is higher among male when compared to female.
# - Education status and Social status are contributing to an increase in suicide rates amongst both men and women.

# #### How does Education status affect men and women in committing suicide?

# In[ ]:


edu = df[df["Category"] == "Education_Status"]

edu = edu[["TypeofCauses","Gender","Total"]]
edu_sort = edu.groupby(["TypeofCauses","Gender"],as_index = False).sum().sort_values(by="Total", ascending = False)
plt.figure(figsize=(15,7))
sns.barplot(data=edu_sort,x="TypeofCauses",y="Total",hue="Gender",palette = "viridis")
plt.xticks(rotation=45)


# **Observations**
# - Men outgrow women in committing suicide in terms of Education status since they only attain little education.
# - The suicide rate among both men and women are very less when they are well educated (Graduate, Diploma, Post Graduate)

# #### How does Social status affect men and women in committing suicide?

# In[ ]:


socio = df[df["Category"] == "Social_Status"]

socio = socio[["TypeofCauses", "Gender", "Total"]]
socio_sort = socio.groupby(["TypeofCauses","Gender"], as_index = False).sum().sort_values(by = "Total", ascending = False)
plt.figure(figsize = (15,7))
sns.barplot(data = socio_sort, x = "TypeofCauses", y = "Total", hue = "Gender", palette = "summer")
plt.xticks(rotation = 45)


# **Observation**
# - Married men commit suicide almost twice that of married women.
# - The suicide rate is very less when people are living independent (widowed, divorce)

# #### What is the trend of the suicide rate between 2001 to 2012?

# In[ ]:


df.groupby("Year")["Total"].sum().plot( kind = "line", figsize = (15,7))


# **Observation**
# - There is a continuos upward trend in the suicide rate from 2001 to 2011, however after 2011 there is a considerable decrease till 2012.

# ## Conclusion

# - Education status and Social status are the most contributing factors towards committing suicide.
# - Adults aged between 15 to 44 commit more suicides constituting almost around 40.06%.
# - The most common means of committing suicide are by hanging, by consuming insecticides and poison.
# - Maharashtra, West Bengal and Tamil Nadu are the states with highest number of suicides.
# - The number of suicides committed by married men are almost double than that of married women.
# - Suicide rates are increasing since 2001 and there is no drastical decrease found in the following years.

# ## Actionable Insights

# - Uneducation being the most important reason for suicide, people should start educating themselves and thier children rather than being ignorant.
# - Government & NGO's can conduct awareness campaign amongst people especially those who are married for suicide prevention.
# - People can use self-affirmation technique to keep themselves positive and stay away from the thought of suicide.

# In[ ]:




