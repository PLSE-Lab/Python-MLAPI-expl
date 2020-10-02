#!/usr/bin/env python
# coding: utf-8

# # HR Analytics EDA

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

#constants
sns.set_style("dark")
sigLev = 3
pd.set_option("display.precision",sigLev)


# In[2]:


#load in dataset
hrFrame = pd.read_csv("../input/HR_comma_sep.csv")


# # Metadata Analysis

# In[3]:


hrFrame.shape


# We see that in this context we are observation rich, but poor in the number of variables. Thus, we won't be interested in using models that are best suited for high-dimensional cases.

# In[4]:


hrFrame.info()


# _Table 1: Info on the number of observations per variable._
# 
# Thankfully, it looks like there are no missing values within this dataset.

# # Summary Statistics

# In[5]:


leftCountFrame = hrFrame.groupby("left",as_index = False)["satisfaction_level"].count()
leftCountFrame = leftCountFrame.rename(columns = {"satisfaction_level":"count"})
#then plot
sns.barplot(x = "left",y = "count",data = leftCountFrame)
plt.xlabel("Left")
plt.ylabel("Count")
plt.title("Distribution of Left")


# _Figure 1: Distribution of Left._
# 
# It is apparent that we have many more individuals who did not leave than individuals who did leave. This will likely take the shape of an imbalanced classes problem in our context.

# In[6]:


plt.hist(hrFrame["satisfaction_level"])
plt.xlabel("Satisfaction Level")
plt.ylabel("Count")
plt.title("Distribution of Satisfaction Level")


# _Figure 2: Distribution of Satisfaction Level._
# 
# It is apparent that there is a  slight bimodality to this distribution. In our context, we see a large number of individuals who are not satisfied with their work, and then a lump of individuals who are moderately to completely satisfied with their work.

# In[7]:


plt.hist(hrFrame["last_evaluation"])
plt.xlabel("Last Evaluation")
plt.ylabel("Count")
plt.title("Distribution of Last Evaluation")


# _Figure 3: Distribution of Last Evaluation._
# 
# Currently, it is difficult to interpret this parameter as it looks like a normalization of a date parameter. It's not clear which end of the scale represents a more recent evaluation versus a less recent evaluation.

# In[8]:


plt.hist(hrFrame["number_project"])
plt.xlabel("Number of Projects")
plt.ylabel("Count")
plt.title("Distribution of Number of Projects")


# _Figure 4: Distribution of Number of Projects._
# 
# We see that most individuals had between 3 and 5 projects, with none having over 7 projects.

# In[9]:


plt.hist(hrFrame["average_montly_hours"])
plt.xlabel("Average Monthly Hours")
plt.ylabel("Count")
plt.title("Distribution of Average Monthly Hours")


# _Figure 5: Distribution of Average Monthly Hours._
# 
# We see that there is a slight bimodality occurring. There is a sizable number of individuals who work around  $$150 \cdot 12 = 1800$$ hours a year but there is also a sizable number that work close to $$250 \cdot 12 = 3000$$ hours a year. I would argue that working above 2000 hours a year is likely considered a form of overworking. Hence, this may be a key driver that is informing the distribution of leaving.

# In[10]:


plt.hist(hrFrame["time_spend_company"])
plt.xlabel("Time Spent at the Company")
plt.ylabel("Count")
plt.title("Distribution of Time Spent at the Company")


# We see a right-skewed distribution with this variable. This may suggest that we will need to do some transformations on this variable.
# 
# It is not clear whether this variable represents years, months, or some other form of time point.

# In[11]:


waCountFrame = hrFrame.groupby("Work_accident",as_index = False)["left"].count()
waCountFrame = waCountFrame.rename(columns = {"left":"count"})
#then plot
sns.barplot(x = "Work_accident",y = "count",data = waCountFrame)
plt.xlabel("Work Accident")
plt.ylabel("Count")
plt.title("Distribution of Work Accident")


# _Figure 6: Distribution of Work Accident._
# 
# We see that most individuals have not had a work accident. However, a substantial number have had a work accident. This could very likely be informing our leaving variable, since a work accident puts an individual out of commission on projects for a substantial amount of time.

# In[12]:


promoteCountFrame = hrFrame.groupby("promotion_last_5years",as_index = False)["left"].count()
promoteCountFrame = promoteCountFrame.rename(columns = {"left":"count"})
#then plot
sns.barplot(x = "promotion_last_5years",y = "count",data = promoteCountFrame)
plt.xlabel("Promotion in the Last 5 Years")
plt.ylabel("Count")
plt.title("Distribution of Promotion in the Last 5 Years")


# _Figure 7: Distribution of Promotion._
# 
# We see that most people were not promoted in the last 5 years. Thus, it is possible that this variable doesn't contain enough variation to be informing our process. I will thus not consider this variable for the rest of my analysis since it looks to be an extremely low-variance feature.

# In[13]:


deptCountFrame = hrFrame.groupby("sales",as_index = False)["left"].count()
deptCountFrame = deptCountFrame.rename(columns = {"left":"count"})
#then plot
sns.barplot(x = "sales",y = "count",data = deptCountFrame)
plt.xlabel("Department")
plt.ylabel("Count")
plt.title("Distribution of Department")


# _Figure 8: Distribution of Department Across Employees._
# 
# We see that we have many levels of department across employees. For the sake of analysis, we will likely need to turn this variable into a series of sparse indicators. They will likely be sparse since some of these levels occur less than 1000 times in the dataset.

# In[14]:


salaryCountFrame = hrFrame.groupby("salary",as_index = False)["left"].count()
salaryCountFrame = salaryCountFrame.rename(columns = {"left":"count"})
#then plot
sns.barplot(x = "salary",y = "count",data = salaryCountFrame)
plt.xlabel("Salary Level")
plt.ylabel("Count")
plt.title("Distribution of Salary Level")


# _Figure 9: Distribution of Salary Level._
# 
# We see that most individuals tend to be in the lower and medium brackets of salary relative to that of higher salaries. This variable may become important, since it is possible that a desire for higher pay could be a strong informer of the decision to leave an institution.

# # Bivariate Relationships

# We will now turn our attention to the bivariate relationship of our predictors on the choice to leave.

# In[15]:


sns.boxplot(x = "left",y = "satisfaction_level",data = hrFrame)
plt.xlabel("Left")
plt.ylabel("Satisfaction Level")
plt.title("Satisfaction Level on Left")


# _Figure 10: Satisfaction Level on the decision to leave._
# 
# We see that those who leave generally have lower satisfaction levels than those who do not leave. However, the variance is larger for those who choose to leave, which may suggest that this effect is noisy and may be weakened when predicting out-of-sample.

# In[17]:


sns.boxplot(x = "left",y = "last_evaluation",data = hrFrame)
plt.xlabel("Left")
plt.ylabel("Last Evaluation")
plt.title("Last Evaluation on Left")


# _Figure 11: Last Evaluation on Left._
# 
# We see that little is going on here.

# In[18]:


sns.boxplot(x = "left",y = "number_project",data = hrFrame)
plt.xlabel("Left")
plt.ylabel("Number of Projects")
plt.title("Number of Projects on Left")


# _Figure 12: Number of

# In[ ]:





# In[ ]:





# FIX:
# 
# * Finish bivariate relationships
# 
# * Finish interaction effects

# In[ ]:




