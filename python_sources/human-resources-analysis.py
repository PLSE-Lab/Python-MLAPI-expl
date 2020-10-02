#!/usr/bin/env python
# coding: utf-8

# # **Why are the employees leaving ?** #

# *"Why are our best and most experienced employees leaving prematurely?"*
# 
# This is what we will try to answer during this notebook, by having a look at the human resources dataset, exploring its characteristics and analysing its content. We will realise this human resources analysis in 4 steps :
# 
#  - Dataset initialisation & exploration
#  - Single parameter analysis
#  - Multiple parameters analysis
#  - Conclusions

# ----------
# 
# ## *1 - Dataset initialisation & exploration* ##
# 
# ----------

# ### *A) Loading the dataset* ###

# Before making deep analysis of our employees and finding the most relevant reasons of their leaving, we have to import the human resources dataset and briefly analyze its content. We do this first step thanks to the **Pandas** library.

# In[ ]:


# Include pandas to manipulate dataframes and series
import pandas as pd
# Include matplotlib to plot diagrams
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Read human resources data from CSV data file
hr_data = pd.read_csv("../input/HR_comma_sep.csv")


# Now the data is loaded in memory, we are ready to work and study what's wrong with the employees, and maybe find the reasons they are leaving !

# ### *B) Exploring the dataset* ###

# Let us start with a glance on the data, with its size and features.

# In[ ]:


# Print the shape of the dataset (number of samples and features)
print(hr_data.shape)
# Print the features of the dataset
features_name = hr_data.columns.tolist()
print(features_name)


# As we can see, our human resources dataset is composed of **14999 samples of 10 variables** (including the one we are interested in : **left**, that indicates the employee left the company or not).

# In[ ]:


# Reorganise the dataset to put the 'left' feature at the end
features_name.remove('left')
organized_features_name = features_name + ['left']
hr_data = hr_data[organized_features_name]
# Print the reorganised dataset first values
hr_data.head()


# We now have a quick idea of the content of the dataset and the meaning of its features. In order to find the reasons employees are leaving, we shall analyse separately those samples of people. This is the analysis preparation phase.

# ### *C) Preparing for analysis* ###

# It would be interesting to **prepare sub-datasets** to analyse separately people who left the company and people who stayed. We do this by defining two different sub-datasets from the original one, filtering on the **left** feature.

# In[ ]:


# Define sub-dataset with people who stayed and those who left
present_people_data = hr_data[hr_data['left'] == 0]
left_people_data = hr_data[hr_data['left'] == 1]


# With these clean datasets, we are now ready to study the behaviour of these two characteristic samples of people of the company.

# ----------
# 
# ## *2 - Simple parameter analysis* ##
# 
# ----------

# ### *A) Linear correlation coefficient* ###

# The first thing to do so as to analyse the dataset is to draw the **correlation matrix**. This will give us a first idea of the correlation between features of the dataset, especially the ones related to the **left** feature.

# In[ ]:


# Print the correlation matrix related to the 'left' feature of the dataset
hr_data.corr()['left']


# The correlation matrix we drew gave us two interesting information : **work accidents** and **time spent in company** are the most representative features of the dataset explaining the employees leaving (taken one by one), and **there is no feature that explains all alone this phenomenon** in a clear way (no correlation factor above 0.5 in absolute value).

# ### *B) Work accidents* ###

# Let us then analyse the first feature of this dataset : **work accidents**. We will plot their values among the dataset, and compare their repartition between people who left and people who stayed in the company.

# In[ ]:


# Define a figure with one diagram
fig = plt.figure(figsize=(6, 3))
present_people_plot = fig.add_subplot(121)
left_people_plot = fig.add_subplot(122)

# Draw an histogram of the present employees work accidents
present_people_plot.hist(present_people_data.Work_accident, 50, facecolor='blue')
present_people_plot.set_xlabel('Work accidents')
present_people_plot.set_title("Employees who stayed")
# Add a vertical line representing the mean work accidents for this sample
present_people_plot.axvline(present_people_data.Work_accident.mean(), color='r',
                            linestyle='dashed', linewidth=2)

# Draw an histogram of the left employees work accidents
left_people_plot.hist(left_people_data.Work_accident, 50, facecolor='red')
left_people_plot.set_xlabel('Work accidents')
left_people_plot.set_title("Employees who left")
# Add a vertical line representing the mean work accidents for this sample
left_people_plot.axvline(left_people_data.Work_accident.mean(), color='b',
                         linestyle='dashed', linewidth=2)

plt.show()


# The first result we can extract from these diagrams is that people who left and had work accidents are less than people who stayed and had work accidents. As we could suspect, **work accidents all alone cannot explain** people leaving the company.

# ### *C) Time spent in the company* ###

# Let us watch the other feature : **time spent in company**. The correlation matrix gave us a score under the work accidents' one, but it is worth having a look, just in case.

# In[ ]:


# Define a figure with one diagram
fig = plt.figure(figsize=(6, 3))
present_people_plot = fig.add_subplot(121)
left_people_plot = fig.add_subplot(122)

# Draw an histogram of the present employees time spent in company
present_people_plot.hist(present_people_data.time_spend_company, 50, facecolor='blue')
present_people_plot.set_xlabel('Time spent in company')
present_people_plot.set_title("Employees who stayed")
# Add a vertical line representing the mean time spent in company for this sample
present_people_plot.axvline(present_people_data.time_spend_company.mean(), color='r',
                            linestyle='dashed', linewidth=2)

# Draw an histogram of the left employees time spent in company
left_people_plot.hist(left_people_data.time_spend_company, 50, facecolor='red')
left_people_plot.set_xlabel('Time spent in company')
left_people_plot.set_title("Employees who left")
# Add a vertical line representing the mean time spent in company for this sample
left_people_plot.axvline(left_people_data.time_spend_company.mean(), color='b',
                         linestyle='dashed', linewidth=2)

plt.show()


# This time, there are two things we can deduce from these drawings : **not a single employee left** after being in the company for **more than 6 years**, and **the probability for an employee to leave is high** when he has been there **between 3 and 5 years**.

# ----------
# 
# ## *3) Multiple parameters analysis* ##
# 
# ----------

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

caracs_columns = hr_data[:][hr_data.columns[0:6]]
result = hr_data[:][hr_data.columns[9:]]

res = hr_data['left']
tt = hr_data.drop(['left', 'sales', 'salary'], axis=1)

random_forest_model = RandomForestClassifier(n_estimators=10)
random_forest_model.fit(tt, res)

importances = random_forest_model.feature_importances_
for f in range(tt.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, importances[f], importances[importances[f]]))


# ----------
# 
# ## *4) Conclusions* ##
# 
# ----------
