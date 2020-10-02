#!/usr/bin/env python
# coding: utf-8

# **************************************************************************************************************
# **NB: There is an error in the code coming from the backend I guess, but to see the complete work please refer to the version 7 of this kernel** 
# 
# Update: This is a known issue in seaborn: https://github.com/mwaskom/seaborn/issues/1092
# 
# Update 2: I have found a WONDERFUL workaround on https://stackoverflow.com/a/44085840 . Amazing!
# ************************************************************************************************************
# <br>
# This report works on the fundamental question of this dataset: Why employees leave? It also explores two other interesting questions:<br>
#  - How does the company evaluates its employees?<br>
#  - What makes an employee happy?<br>
# 
# The report starts by exploring the data (Both univariate and multivariate analysis) then it goes on to explore the questions of interest. Finally, A random forest model is built to model those who left.

# # Imports

# In[ ]:


import numpy as np
import pandas as pd

import scipy.stats
from scipy.stats.stats import pearsonr

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


from IPython.display import display_html


# # Helper Functions

# In[ ]:


#A function that returns the order of a group_by object according to the average of certain parameter param.
def get_ordered_group_index(df, group_by, param, ascending=False):
    return df.groupby(group_by)[param].mean().sort_values(ascending=ascending).index

def group_by_2_level_perc(df, level1, level2, level1_index_order = None, level2_index_order = None):
    #http://stackoverflow.com/questions/23377108/pandas-percentage-of-total-with-groupby
    df_by_lvl1_lvl2 = df.groupby([level1, level2]).agg({level1: 'count'})
    df_by_lvl1_lvl2_perc = df_by_lvl1_lvl2.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
    #Reorder them in logical ascending order, but first make sure it is not an empty input
    if level1_index_order:
        df_by_lvl1_lvl2_perc = df_by_lvl1_lvl2_perc.reindex_axis(level1_index_order, axis=0, level=0)
    #If a second level order is passed, apply it, else use the default
    if level2_index_order:
        df_by_lvl1_lvl2_perc = df_by_lvl1_lvl2_perc.reindex_axis(level2_index_order, axis=0, level=1)
    return df_by_lvl1_lvl2_perc

#A function that adds some styling to the graphs, like custom ticks for axes, axes labels and a grid
def customise_2lvl_perc_area_graph(p, legend_lst, xtick_label = "", x_label="", y_label=""):
    #If custom ticks are passed, spread them on the axe and write the tick values
    if xtick_label:
        p.set_xticks(range(0,len(xtick_label)))
        p.set_xticklabels(xtick_label)
    #Create y ticks for grid. It will always be a percentage, so it is not customisable
    p.set_yticks(range(0,110,10)) 
    p.set_yticklabels(['{:3.0f}%'.format(x) for x in range(0,110,10)])
    p.set_yticks(range(0,110,5), minor=True) 

    #Draw grid and set y limit to be only 100 (By default it had an empty area at the top of the graph)
    p.xaxis.grid('on', which='major', zorder=1, color='gray', linestyle='dashed')
    p.yaxis.grid('on', which='major', zorder=1, color='gray', alpha=0.2)
    p.yaxis.grid('on', which='minor', zorder=1, color='gray', linestyle='dashed', alpha=0.2)
    p.set(ylim=(0,100))

    #Customise legend
    p.legend(labels=legend_lst, frameon=True).get_frame().set_alpha(0.2)

    #Put the axes labels
    if x_label:
        p.set_xlabel(x_label)
    if y_label:
        p.set_ylabel(y_label);


# ### Test that the Workaround Works!

# In[ ]:


import seaborn as sns

s = pd.Series(data=[5850000, 6000000, 5700000, 13100000, 16331452], name='price_doc')
print(statsmodels.__version__)
print(sns.__version__)
_ = sns.distplot(s, bins=2, kde=True)


# # Import Data and Quick Overview

# In[ ]:


hr_df = pd.read_csv('../input/HR_comma_sep.csv')


# In[ ]:


print("The dataset has", hr_df.shape[1], "features and ", hr_df.shape[0], "entries")


# In[ ]:


hr_df.describe()


# In[ ]:


hr_df.head()


# # Univariate Analysis

# Since the theme of this dataset is about the emplyees who left, I will - beside all the data groups - plot the distributions for those who left and those who stayed

# In[ ]:


hr_by_left = hr_df.groupby('left')
employees_left = hr_by_left.get_group(1)
employees_stayed = hr_by_left.get_group(0)


# ## Satisfaction Level

# ***************************
# **NB: There is an error in the code coming from the backend I guess, but to see the complete work please refer to version 7 of this kernel** 
# *****************************

# In[ ]:


fig, axs = plt.subplots(nrows= 3, figsize=(13, 5))

sns.kdeplot(employees_left.satisfaction_level, ax=axs[0], shade=True, color="r")
kde_plot = sns.kdeplot(employees_stayed.satisfaction_level, ax=axs[0], shade=True, color="g")
kde_plot.legend(labels=['Left', 'Stayed'])

hist_plot = sns.distplot(hr_df.satisfaction_level, ax=axs[1])
box_plot = sns.boxplot(hr_df.satisfaction_level, ax=axs[2])

kde_plot.set(xlim=(0,1.1))
hist_plot.set(xlim=(0,1.1))
box_plot.set(xlim=(0,1.1));


# A bimodal shape (Well, more or less), one peak for the bitter employees at the lower end of satisfaction and another one well spread from medium to high satisfaction.

# ## Last Evaluated

# **NB: There is an error in the code coming from the backend I guess, but to see the complete work please refer to the previous version of this kernel** 

# This is the last evaluation for the employee, i.e. how he was appraised by his company.

# In[ ]:


fig, axs = plt.subplots(nrows= 3, figsize=(13, 5))

sns.kdeplot(employees_left.last_evaluation, ax=axs[0], shade=True, color="r")
kde_plot = sns.kdeplot(employees_stayed.last_evaluation, ax=axs[0], shade=True, color="g")
kde_plot.legend(labels=['Left', 'Stayed'])

hist_plot = sns.distplot(hr_df.last_evaluation, ax=axs[1])
box_plot = sns.boxplot(hr_df.last_evaluation, ax=axs[2])

kde_plot.set(xlim=(0,1.1))
hist_plot.set(xlim=(0,1.1))
box_plot.set(xlim=(0,1.1));


# Another bimodal. very little employees were evaluated below 0.5. A more is around the middle mark and another near the high mark. The other thing to note here is that the evluation has a finer resolution than I thought it would, it is not just a 1..2..3 discrete values, it is an evaluation with 100 steps.

# ## Number of Projects

# **NB: There is an error in the code coming from the backend I guess, but to see the complete work please refer to the previous version of this kernel** 

# In[ ]:


fig, axs = plt.subplots(nrows= 3, figsize=(13, 5))

sns.kdeplot(employees_left.number_project, ax=axs[0], shade=True, color="r")
kde_plot = sns.kdeplot(employees_stayed.number_project, ax=axs[0], shade=True, color="g")
kde_plot.legend(labels=['Left', 'Stayed'])

hist_plot = sns.distplot(hr_df.number_project, ax=axs[1], kde=False)
box_plot = sns.boxplot(hr_df.number_project, ax=axs[2])

kde_plot.set(xlim=(0,8))
hist_plot.set(xlim=(0,8))
box_plot.set(xlim=(0,8));


# ## Average Monthly Hours

# **NB: There is an error in the code coming from the backend I guess, but to see the complete work please refer to the previous version of this kernel** 

# In[ ]:


fig, axs = plt.subplots(nrows=3, figsize=(13, 4))

sns.kdeplot(employees_left.average_montly_hours, ax=axs[0], shade=True, color="r")
kde_plot = sns.kdeplot(employees_stayed.average_montly_hours, ax=axs[0], shade=True, color="g")
kde_plot.legend(labels=['Left', 'Stayed'])

hist_plot = sns.distplot(hr_df.average_montly_hours, ax=axs[1])
box_plot = sns.boxplot(hr_df.average_montly_hours, ax=axs[2])

kde_plot.set(xlim=(0,350))
hist_plot.set(xlim=(0,350))
box_plot.set(xlim=(0,350));


# Another bimodal shape, one at around 150 hours a month and the other is a little over 250 hours a month. Some very high values, 300 (?!) hours a month. That means if this employee never takes any days off, they work 10 hours a day. If they take one day off, it would be 11.5 hours of work, and two days weekend would mean they work for 13 hours a day.

# ## Number of Years Working for the Company

# 

# In[ ]:


fig, axs = plt.subplots(nrows= 3, figsize=(13, 5))

sns.kdeplot(employees_left.time_spend_company, ax=axs[0], shade=True, color="r")
kde_plot = sns.kdeplot(employees_stayed.time_spend_company, ax=axs[0], shade=True, color="g")
kde_plot.legend(labels=['Left', 'Stayed'])

hist_plot = sns.distplot(hr_df.time_spend_company, ax=axs[1], kde=False)
box_plot = sns.boxplot(hr_df.time_spend_company, ax=axs[2])

kde_plot.set(xlim=(0,12))
hist_plot.set(xlim=(0,12))
box_plot.set(xlim=(0,12));


# Interestingly, none of the surveyed employees worked for less than two years. That raises some concerns about the randomness of picking the subjects, there was a certain bias for those who stayed longer. We have to take that into account, to start with. Going back to the dataset's welcome page, it was clearly stated that this is a simulated dataset, so maybe this is why.
# 
# But disregarding that previous point the employees were mostly centered around 3 years.

# ## Employees Who Suffered Work Related Accidents

# In[ ]:


#TODO: CLEAN ME UP! Remove all the commented code
def annotate_bars(bar_plt, bar_plt_var, by=None, x_offset=0, y_offset=0, txt_color="white", fnt_size=12, fnt_weight='bold'):
    if by is None:
        for p in bar_plt.patches:
            bar_plt.annotate(str( int(p.get_height()) ) + "\n" + str(round( (100.0* p.get_height()) /bar_plt_var.count(), 1) )+ "%", 
                             (p.get_x() + x_offset, p.get_height()-y_offset),
                             color=txt_color, fontsize=fnt_size, fontweight=fnt_weight)
    else:
        grouped = bar_plt_var.groupby(by)
        for p in bar_plt.patches:            
            #This part is tricky. The problem is that not each x-tick gets drawn in order, i.e. yes/no of the first group 
            #then yes/no of the second group located on the next tick, but rather all the yes on all the x-ticks get drawn first
            # then all the nos next. So we need to know we are using a patch that belongs to which tick (the x-tick) ultimately
            #refers to one of the groups. So, we get the x absolute coordinate, round it to know this patch is closest to which tick
            #(Assuming that it will always belong to its closest tick), then get the group count of that tick and use it as a total
            #to compute the percentage.
            total = grouped.get_group(bar_plot.get_xticks()[int(round(p.get_x()))]).count()
            bar_plt.annotate(str( int(p.get_height()) ) + "\n" + str(round( (100.0* p.get_height()) /total, 1) )+ "%", 
                             (p.get_x() + x_offset, p.get_height()-y_offset),
                             color=txt_color, fontsize=fnt_size, fontweight=fnt_weight)


# This is a boolean field, indicating whither the employee had an accident at work or not.

# In[ ]:


fig, axs = plt.subplots(ncols= 2, figsize=(13, 5))

work_accidents_plt = sns.countplot(hr_df.Work_accident, ax=axs[0]);
annotate_bars(bar_plt=work_accidents_plt, bar_plt_var=hr_df.Work_accident, x_offset=0.3, y_offset=1100)
    
bar_plot = sns.countplot(x=hr_df.Work_accident, hue=hr_df.left, ax=axs[1])
annotate_bars(bar_plt=bar_plot, by=hr_df.Work_accident, bar_plt_var=hr_df.Work_accident, x_offset=0.1, txt_color="black")
bar_plot.set(ylim=(0,14000));


# Mostly no, fortunatey.

# ## Employees who Left their Jobs

# This is the variable of question, it will be interesting to examine the ratio:

# In[ ]:


employees_left_plt = sns.countplot(hr_df.left);

for p in employees_left_plt.patches:
    employees_left_plt.annotate(str( int(p.get_height()) ) + "\n" + str(round( (100.0* p.get_height()) /hr_df.left.count(), 1) )+ "%", 
                                (p.get_x() + 0.3, p.get_height()-1100),
                                color='white', fontsize=12, fontweight='bold')


# ## Who Got Promoted Within the Past 5 Years

# In[ ]:


fig, axs = plt.subplots(ncols= 2, figsize=(13, 5))

promoted_5years_plt = sns.countplot(hr_df.promotion_last_5years, ax=axs[0]);
annotate_bars(bar_plt=promoted_5years_plt, bar_plt_var=hr_df.promotion_last_5years, x_offset=0.3, txt_color="black")
    
bar_plot = sns.countplot(x=hr_df.promotion_last_5years, hue=hr_df.left, ax=axs[1])
annotate_bars(bar_plt=bar_plot, by=hr_df.promotion_last_5years, bar_plt_var=hr_df.promotion_last_5years, x_offset=0.1, txt_color="black")
bar_plot.set(ylim=(0,16000));


# The vast majority did not get promoted. But if they did, they are most likely going to stay. I think it is a good time for a quick chi-square test to verify this:

# In[ ]:


#Create groups
employees_by_promotion = hr_df.groupby("promotion_last_5years")
employees_promoted = employees_by_promotion.get_group(1)
employees_not_promoted = employees_by_promotion.get_group(0)

#Get counts
employees_promoted_stayed = employees_promoted.groupby("left").get_group(0).left.count()
employees_promoted_left = employees_promoted.groupby("left").get_group(1).left.count()

employees_not_promoted_stayed = employees_not_promoted.groupby("left").get_group(0).left.count()
employees_not_promoted_left = employees_not_promoted.groupby("left").get_group(1).left.count()

#Create rows that makeup the contingency table
promoted_row = [employees_promoted_stayed, employees_promoted_left, employees_promoted_stayed + employees_promoted_left]
not_promoted_row = [employees_not_promoted_stayed, employees_not_promoted_left, employees_not_promoted_stayed + employees_not_promoted_left]
total_row = [employees_promoted_stayed+employees_not_promoted_stayed,
             employees_promoted_left+employees_not_promoted_left,
             hr_df.left.count()]

#Create the contingency table
contingency_table = pd.DataFrame({'Promoted': promoted_row ,
                                  'Not Promoted': not_promoted_row ,
                                  'Total, By Left': total_row},
                                 index = ['Stayed', 'Left', 'Total, by Promotion'], 
                                 columns = [ 'Promoted', 'Not Promoted', 'Total, By Left'])

display_html(contingency_table)


# In[ ]:


chi_squared, p, degrees_of_freedom, expected_frequency = scipy.stats.chi2_contingency( contingency_table )

print("Chi Squared: ", chi_squared)
print("p value: ", p)
print("Degrees of Freedom", degrees_of_freedom)
print("Expected Frequency for The Not Promoted Employees:", expected_frequency[0])
print("Expected Frequency for The Promoted Employees:", expected_frequency[1])


# So, this is a pretty significant result. Promotion within the last 5 years is definitely something the company must do for the employees that they want to retain.

# ## The Department at Which they Worked

# In[ ]:


fig, axs = plt.subplots(figsize=(13, 4))

department_plt = sns.countplot(hr_df.sales, order = hr_df.sales.value_counts().index);

annotate_bars(bar_plt=department_plt, bar_plt_var=hr_df.sales, x_offset=0.2, y_offset=450, txt_color="black")


# Sales department has the highest count, maybe that was the inspiration to name the column as sales (Which I think should be renamed by the way).

# ## Salaries

# In[ ]:


department_plt = sns.countplot(hr_df.salary, order = hr_df.salary.value_counts().index);

for p in department_plt.patches:
    department_plt.annotate(str( int(p.get_height()) ) + "\n" + str(round( (100.0* p.get_height()) /hr_df.salary.count(), 1) )+ "%", 
                                (p.get_x() + 0.3, p.get_height()-800),
                                color='white', fontsize=12, fontweight='bold')


# Around half the employees had a low salary. The high salary employees made up about 8%

# ## Univariate Conclusion

# There was a few graphs that had a bimodal shape. If these modes correlate together, maybe we can have two easily recognizable groups of emplyees.
# 
# The boolean fields are not balanced, the absence (value = 0) is always dominant.
# 
# The first thing we know about those who leave the company is that the lack of promotion is a significant factor.

# # Bivariate Analysis

# ## Quick Overview

# ### Correlation Heatmap

# In[ ]:


plt.figure(figsize=(12, 8))

hr_corr = hr_df.corr()
sns.heatmap(hr_corr, 
            xticklabels = hr_corr.columns.values,
            yticklabels = hr_corr.columns.values,
            annot = True);


# For the "left" parameter, there is a moderate(ish) correlation with the satisfaction level. There is a negative correlation with the number of work accidents and the time spent at the company, but the values are too low, it may be noise.
# 
# The other interesting correlation is the red blob around last evaluation, average monthly hours and the number of projects. They seem to be related in a moderate linear way.

# ### CrossPlots

# Since the theme about this dataset is who will leave next, it only makes sense to colour the scatter plots by the 'left' parameter:

# In[ ]:


plt.figure(figsize=(10, 10))

sns.pairplot(hr_df,  hue="left");


# There are some **very** interesting green patches in the scatters involving the satisfaction level, last evaluation, average monthly hours and time spent at the company.
# 
# It seems that the lowest evaluated employees do not leave.
# 
# Employees who received a promotion within the past 5 years seem to be less likely to leave, their columns are much "blue-er" than those who did not.
# 
# There is a threshold of satisfaction, after which none leaves, in the dataset.
# 
# Employees who remain long enough in the company (over 6 years) are less likely to leave
# 
# The average monthly hours of those who left is higher. Too much work, and maybe too little in return?

# Before going-on with modeling leaving the company, I want to explore a few things:<br>
# 1- Company departments: Are some better than the others or there is a unified process throughout the company?<br>
# 2- Salaries: Are there any decisive factors within the dataset that can make us understand how an employee is paid?<br>
# 3- Employee evaluation: What are the most important factors that makes one employee better than the other?<br>
# 4- What makes and employee happy?

# ## Departments

# **The following graphs are sorted descendingly, so pay attention to the x-axis labels, their order differ from one graph to another.**

# ### How Hard does Each Department Work?

# In[ ]:


fig, axs = plt.subplots(figsize=(13, 4))

bar_plot = sns.barplot(x=hr_df.sales, y=hr_df.average_montly_hours, order=get_ordered_group_index(hr_df, 'sales', 'average_montly_hours') )


# Seems that they all work equally hard, around 10 hours a day on average (Assuming two days off each week)

# ### How Satisfied is Each Department?

# In[ ]:


fig, axs = plt.subplots(figsize=(13, 4))

bar_plot = sns.barplot(x=hr_df.sales, y=hr_df.satisfaction_level, order=get_ordered_group_index(hr_df, 'sales', 'satisfaction_level'))


# More or less similar, except for the accounting and maybe the HR departments.

# ### How Many Projects are Assigned on Average per Employee?

# In[ ]:


fig, axs = plt.subplots(figsize=(13, 4))

bar_plot = sns.barplot(x=hr_df.sales, y=hr_df.number_project, order=get_ordered_group_index(hr_df, 'sales', 'number_project'))


# ### Where are the Promotions?

# In[ ]:


fig, axs = plt.subplots(figsize=(13, 4))

bar_plot = sns.barplot(x=hr_df.sales, y=hr_df.promotion_last_5years, order=get_ordered_group_index(hr_df, 'sales', 'promotion_last_5years'))


# I want to stop a little bit on this plot and contemplate the result. The management department is getting a considerably higher percentage of promotion than the rest of the company. Is this fair? Do they work harder? Is the company doing a good job? Let us explore this more:

# ### Evaluation: Is the Management Department Really Outperforming the Rest?

# In[ ]:


fig, axs = plt.subplots(figsize=(13, 4))

bar_plot = sns.barplot(x=hr_df.sales, y=hr_df.last_evaluation, order=get_ordered_group_index(hr_df, 'sales', 'last_evaluation'))


# Not really, they are the highest but the rest are extremely close

# ### Maybe they Stayed Longer?

# In[ ]:


fig, axs = plt.subplots(figsize=(13, 4))

bar_plot = sns.barplot(x=hr_df.sales, y=hr_df.time_spend_company, order=get_ordered_group_index(hr_df, 'sales', 'time_spend_company'))


# Relatively yes, and the rest seem to be around the same range. Maybe seniority within the company have an impact on the promotion process, but it is definitely not a good explanation for the high difference in the promotion rate.

# ### Work Related Accidents

# In[ ]:


fig, axs = plt.subplots(figsize=(13, 4))

bar_plot = sns.barplot(x=hr_df.sales, y=hr_df.Work_accident, order=get_ordered_group_index(hr_df, 'sales', 'Work_accident'))


# Oh well, the management suffered more accidents on average than the technical department. I find it strange to be honest.

# The department exploration was interesting, and maybe further exploration for the departments from the "who left?" point of view would be beneficial:

# ### Departments & Who Left

# #### Left: Is there a Pattern Within Departments?

# In[ ]:


fig, axs = plt.subplots(figsize=(13, 4))

#Order the bars descendingly according to the PERCENTAGE % of those who left in each department
total_employees_by_dept = hr_df.groupby(["sales"]).satisfaction_level.count()
left_count_by_dept = hr_df[hr_df["left"] == 1].groupby(["sales"]).satisfaction_level.count()
percentages_left_by_dept = (left_count_by_dept / total_employees_by_dept).sort_values(ascending=False)
axe_name_order = percentages_left_by_dept.index

department_plt = sns.countplot(hr_df.sales, order = axe_name_order, color='g');
sns.countplot(employees_left.sales, order = axe_name_order, color='r');

department_plt.legend(labels=['Stayed', 'Left'])
department_plt.set(xlabel='Department\n Sorted for "Left" Percentage')

#Annotate the percentages of those who stayed. It was more straightforward to loop for each category (left, stayed) than
#doing all the work in one loop
#The zip creates an output that is equal to the shortest parameter, so we do not need to adjust the patches length, since
#the loop will stop after finishing the columns of those who stayed
for p, current_column in zip(department_plt.patches, axe_name_order):
    current_column_total = hr_df[hr_df['sales'] == current_column].sales.count()
    stayed_count = p.get_height() - employees_left[employees_left['sales'] == current_column].sales.count()
    department_plt.annotate(str(round( (100.0* stayed_count) /current_column_total, 1) )+ "%", 
                                (p.get_x() + 0.2, p.get_height()-10),
                                color='black', fontsize=12)
    
#In this loop, we want to use the patches located on the second half of patches list, which are the bars for those who left.
for p, current_column in zip(department_plt.patches[int(len(department_plt.patches)/2):], axe_name_order):
    current_column_total = hr_df[hr_df['sales'] == current_column].sales.count()
    left_count = p.get_height()
    department_plt.annotate(str(round( (100.0* left_count) /current_column_total, 1) )+ "%", 
                                (p.get_x() + 0.2, p.get_height()-10),
                                color='black', fontsize=12)


# The HR is a little high, R&D and management are low.

# ### Average Performance

# The horizontal line represents the company's average.

# In[ ]:


fig, axs = plt.subplots(figsize=(13, 4))

colours = ['green', 'red']
bar_plot = sns.barplot(x=hr_df.sales, y=hr_df.last_evaluation, hue=hr_df.left, palette=colours, order=hr_df.sales.value_counts().index)
bar_plot.set(xlabel='Department', ylabel='Average Evaluation')

plt.plot([-1, 10], [hr_df.last_evaluation.mean(), hr_df.last_evaluation.mean()], linewidth=1);

bar_plot.set(ylim=(0,1));


# Those who left in all departments except marketing, accounting and HR have a better average evalution than those who stayed.

# ### By Average Working Time

# The horizontal line represents the company's average.

# In[ ]:


fig, axs = plt.subplots(figsize=(13, 4))

colours = ['green', 'red']
bar_plot = sns.barplot(x=hr_df.sales, y=hr_df.average_montly_hours, hue=hr_df.left, palette=colours, order=hr_df.sales.value_counts().index)
bar_plot.set(xlabel='Department', ylabel='Average Monthly Hours')

plt.plot([-1, 10], [hr_df.average_montly_hours.mean(), hr_df.average_montly_hours.mean()], linewidth=1);


# Those who left were staying more time than those who stayed on average, except for, again, the HR department.

# ### Salaries

# It will be good to see the makeup of each department in terms of salary, and then seeing the difference of salaries between those who stayed and those who left.

# In[ ]:


#Group them according to salary range
low_salaried = hr_df[hr_df.salary == 'low']
mid_salaried = hr_df[hr_df.salary == 'medium']
high_salaried = hr_df[hr_df.salary == 'high']


# In[ ]:


#Group them then get the percentages
left_percent_dept = (hr_by_left.get_group(1).groupby(['sales']).salary.value_counts(normalize=True))*100
salary_percent_dept = (hr_df.groupby(['sales']).salary.value_counts(normalize=True))*100

fig, axs = plt.subplots(nrows= 2, figsize=(13, 7))

#Draw an area ploy for each group
salary_percent_dept.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', ax=axs[0])
left_percent_dept.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', ax=axs[1])

axs[0].set_title('All Employees')
axs[0].set_xlabel('')
axs[0].set_ylabel('Salary Percentage Makeup\n Stayed')

axs[1].set_title('Employees Who Left')
axs[1].set_xlabel('Department')
axs[1].set_ylabel('Salary Percentage Makeup \n Left')

plt.subplots_adjust(hspace=0.2)


# The management has clearly a higher ratio of high salary. The rest seem to be close to each other in their makeup (accounting has a slightly higher high salary ratio, HR and R&D have a slightly higher ratio of medium salaries).
# 
# In terms of who left, I don't want to jump for conclusions just yet, since the makeup is not very close in all departments, but in IT, sales and support, lower salaried employees have higher peaks.
# 
# Now, I want to see what is the ratio of those how each salary range differed in terms of staying\leaving the company.

# In[ ]:


low_salaried_leave_perc = 100*low_salaried.groupby(['sales', 'left']).salary.agg({'sales': 'count'})/low_salaried.groupby(['sales']).salary.agg({'sales': 'count'})
mid_salaried_leave_perc = 100*mid_salaried.groupby(['sales', 'left']).salary.agg({'sales': 'count'})/mid_salaried.groupby(['sales']).salary.agg({'sales': 'count'})
high_salaried_leave_perc = 100*high_salaried.groupby(['sales', 'left']).salary.agg({'sales': 'count'})/high_salaried.groupby(['sales']).salary.agg({'sales': 'count'})

fig, axs = plt.subplots(nrows= 3, figsize=(13, 9))

low_salaried_leave_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', ax=axs[0])
mid_salaried_leave_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', ax=axs[1])
high_salaried_leave_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', ax=axs[2])

axs[0].set_title('Low Salaried Employees')
axs[1].set_title('Medium Salaried Employees')
axs[2].set_title('High Salaried Employees')

axs[0].set_xlabel('')
axs[1].set_xlabel('')
axs[2].set_xlabel('Department')

axs[0].set_ylabel('Left to Stayed Makeup')
axs[1].set_ylabel('Left to Stayed Makeup')
axs[2].set_ylabel('Left to Stayed Makeup')

plt.subplots_adjust(hspace=0.3);


# The last thing I want to visually examine about the relationship between salary, department and leaving is the area graph but by department, i.e. each department will have its own graph and see if there is a department that doesn't look like the others:

# In[ ]:


departments = list(set(hr_df.sales.values))
number_of_departments = len(departments)

fig, axs = plt.subplots(nrows= int(number_of_departments/2), ncols=2, figsize=(13, 20))

for i in range(number_of_departments):
    current_dep = departments[i]
    
    ratio_df = 100*hr_df[hr_df.sales == current_dep].groupby(['salary', 'left']).agg({'salary': 'count'})/hr_df[hr_df.sales == current_dep].groupby(['salary']).agg({'salary': 'count'})
    
    #plot the department
    ratio_df.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', ax=axs[int(i/2),i%2])
    axs[int(i/2),i%2].set_title(current_dep)
    axs[int(i/2),i%2].set_xlabel("")
    
axs[int(i/2),i%2].set_xlabel("Salary")
plt.subplots_adjust(hspace=0.3);


# I see two distinct shapes here:
# 
# 1- Those who have the low and medium salary leaving rate are almost equal, with high salary leaving rate higher. These departments are Product Management, HR, R&D, Technical and Accounting.
# 
# 2- Those who have a high low-salaried leaving, with high and medium leaving rate lower. This shape looks like a letter "V" in the graphs. These departments are: Management, Marketing, Support, Sales and IT
# 
# The main difference between these two shapes is in the medium-salaried leaving rate. Not to jump into conclusions, but I think one explanation is that in certain departments, having a medium salary may be convincing to the employee that they have a good deal, while in other departments this is not the case. Maybe it is because getting a medium-salary in these fields in another company is not that hard?

# ## Salaries: A More General Overview

# In[ ]:


fig, axs = plt.subplots(figsize=(13, 4))

axe_name_order = hr_df.salary.value_counts().index

salary_plt = sns.countplot(hr_df.salary, order = axe_name_order, color='g');
sns.countplot(employees_left.salary, order = axe_name_order, color='r');

salary_plt.legend(labels=['Stayed', 'Left'])

#Annotate the percentages of those who stayed. It was more straightforward to loop for each category (left, stayed) than
#doing all the work in one loop
#The zip creates an output that is equal to the shortest parameter, so we do not need to adjust the patches length, since
#the loop will stop after finishing the columns of those who stayed
for p, current_column in zip(salary_plt.patches, axe_name_order):
    current_column_total = hr_df[hr_df['salary'] == current_column].salary.count()
    stayed_count = p.get_height() - employees_left[employees_left['salary'] == current_column].salary.count()
    salary_plt.annotate(str(round( (100.0* stayed_count) /current_column_total, 1) )+ "%", 
                                (p.get_x() + 0.35, p.get_height()-10),
                                color='black', fontsize=12)
    
#In this loop, we want to use the patches located on the second half of patches list, which are the bars for those who left.
for p, current_column in zip(salary_plt.patches[int(len(salary_plt.patches)/2):], axe_name_order):
    current_column_total = hr_df[hr_df['salary'] == current_column].salary.count()
    left_count = p.get_height()
    salary_plt.annotate(str(round( (100.0* left_count) /current_column_total, 1) )+ "%", 
                                (p.get_x() + 0.35, p.get_height()-10),
                                color='black', fontsize=12)


# This makes sense, people with high salaries are less motivated to leave.

# ### Salary and Work-Load

# In[ ]:


fig, axs = plt.subplots(figsize=(16, 4))

sns.stripplot(y = 'salary', x='average_montly_hours', hue='left', data=hr_df);


# We can have here a closer look to the clusters seen in the cross-plot. For the low and medium salaries, too much work or just below the standard 40 hours\week work have a high concentration of leaving. For the high salaries, the rate of leaving the company is low anyway, so although that the green dots seem to be at around the same values, they're just too little to catch one's eyes.

# If we divide working hours into four categories:
# 
# 1- Those who work a regular 8 hours a day or less (< 168 a month, assuming that a calendar month have on average 21 days of work)<br>
# 2- Those who work between 8 and 10 hours a day ( 168 < average_montly_hours < 210 a month)<br>
# 3- Those who work between 10 and 12 hours a day ( 210 < average_montly_hours < 252 a month)<br>
# 4- Those who work over 12 hours a day.<br>
# 
# and see how the salary goes with this effort:

# In[ ]:


#A function to bin the average monthly hours into the categories described above
def work_load_cat(avg_mnthly_hrs):
    work_load = "unknown"
    if avg_mnthly_hrs < 168:
        work_load = "low"
    elif (avg_mnthly_hrs >= 168) & (avg_mnthly_hrs < 210):
        work_load = "average"
    elif (avg_mnthly_hrs >= 210) & (avg_mnthly_hrs < 252):
        work_load = "above_average"
    elif avg_mnthly_hrs >= 252:
        work_load = "workoholic"
        
    return work_load


# In[ ]:


hr_df['work_load'] = hr_df.average_montly_hours.apply(work_load_cat)

sns.countplot(x='work_load', hue='left', data=hr_df, order = ['low', 'average', 'above_average', 'workoholic']);


# The average zone (8~10 hours a day at work) have a very low record of leaving.

# In[ ]:



#Normalised stacked
departments = list(set(hr_df.sales.values))
number_of_departments = len(departments)

fig, axs = plt.subplots(nrows= int(number_of_departments/2), ncols=2, figsize=(13, 20))

for i in range(number_of_departments):
    current_dep = departments[i]
    
    ratio_df = 100*hr_df[hr_df.sales == current_dep].groupby(['work_load', 'left']).agg({'work_load': 'count'})/hr_df[hr_df.sales == current_dep].groupby(['work_load']).agg({'work_load': 'count'})
    ratio_df = ratio_df.reindex_axis(["low", "average", "above_average", "workoholic"], axis=0, level=0)
    #plot the department
    ratio_df.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', ax=axs[int(i/2),i%2])
    axs[int(i/2),i%2].set_title(current_dep)
    axs[int(i/2),i%2].set_xlabel("")
    
axs[int(i/2),i%2].set_xlabel("work_load")
plt.subplots_adjust(hspace=0.3);


# They have similar patterns. 

# ## Understanding how the Company Evaluates its Employees

# I believe that categorising the employees evaluation would yield better visualisations, so let's do that:

# In[ ]:


#A function to bin last evaluation into one of 5 categories
def last_evaluation_cat(last_evaluation):
    evaluation = "unknown"
    if last_evaluation < 0.45:
        evaluation = "very_low"
    elif (last_evaluation >= 0.45) & (last_evaluation < 0.55):
        evaluation = "mediocre"
    elif (last_evaluation >= 0.55) & (last_evaluation < 0.8):
        evaluation = "average"
    elif (last_evaluation >= 0.8) & (last_evaluation < 0.9):
        evaluation = "very_good"
    elif last_evaluation >= 0.9:
        evaluation = "excellent"
        
    return evaluation


# In[ ]:


hr_df['evaluation'] = hr_df.last_evaluation.apply(last_evaluation_cat)


# ### Evaluation Categories Across the Company

# In[ ]:


sns.countplot(x='evaluation',  data=hr_df, order = ["unknown", 'very_low', 'mediocre', 'average', 'very_good', 'excellent']);


# Average is definitely the dominant, with mediocre, very good and excellent having close counts. Can there be a pattern if we visualise them in terms of who left?

# In[ ]:


sns.countplot(x='evaluation',  hue = 'left', data=hr_df, order = ["unknown", 'very_low', 'mediocre', 'average', 'very_good', 'excellent']);


# 1- The very low performance did not leave the company<br>
# 2- Mediocre has a leaving rate almost as high as those who have stayed<br>
# 3- The average here is similar to that of monthly-time-spent, it has the lowest leaving rate.<br>
# 4- very good and excellent emplyees remain sort of similar.<br>
# 
# Let's explore it further:

# In[ ]:


evaluation_index_order = ["unknown", 'very_low', 'mediocre', 'average', 'very_good', 'excellent']
evaluation_xticks = ['Very Low\n (eval < .45)', 'Mediocre\n ( .45 < eval < .55 )', 'Average\n ( .55 < eval < .8 )', 'Very Good\n ( .8 < eval < .9 )', 'Excellent\n ( .9 < eval)']
evaluation_x_label = "Company Evaluation for the Employee"


# ### Performance Categories Makeup in Terms of Working Hours

# In[ ]:


employees_by_eval_and_workload = group_by_2_level_perc(hr_df, 
                                                       'evaluation', 'work_load',
                                                       evaluation_index_order, ['low','average','above_average', 'workoholic'])#Index Order

workload_legend = ['Low Workload (< 40hrs/week)', 'Average Workload (40 < wl < 50 hrs/week)', 'Above Average Workload (50 < wl < 60hrs/week)', 'Workoholic Workload (wl > 60hrs/week)']

#Plot the Graph
p=employees_by_eval_and_workload.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)

customise_2lvl_perc_area_graph(p, workload_legend, 
                               xtick_label=evaluation_xticks, x_label=evaluation_x_label, #Company Evaluation Graph
                               y_label="Percentage of Monthly Workload")


# Well, it is not an impossibility to work little hours and still get excellent ratings, hopefully it is because they work smarter.

# ### Number of Years with the Company

# In[ ]:


employees_by_eval_and_time_in_company_perc = group_by_2_level_perc(hr_df, 
                                                                   'evaluation', 'time_spend_company',
                                                                   evaluation_index_order)#Index Order

#Plot the Graph
p=employees_by_eval_and_time_in_company_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)

time_spent_legend = [str(x) + " years" for x in range(2,9)] + ['10 years']

customise_2lvl_perc_area_graph(p, time_spent_legend, 
                               xtick_label=evaluation_xticks, x_label=evaluation_x_label, #Company Evaluation Graph
                               y_label="Percentage of Years in Company")


# ### Number of Projects

# In[ ]:


employees_by_eval_and_time_in_company_perc = group_by_2_level_perc(hr_df, 
                                                                   'evaluation', 'number_project',#Variables to groupby
                                                                   evaluation_index_order)#Index Order

#Plot the Graph
p=employees_by_eval_and_time_in_company_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)

num_projects_legend = [str(x) + " projects" for x in range(2,8)]

customise_2lvl_perc_area_graph(p, num_projects_legend, 
                               xtick_label=evaluation_xticks, x_label=evaluation_x_label, #Company Evaluation Graph
                               y_label="Percentage of Number of Projects Assigned")


# ### Salary

# In[ ]:


employees_by_eval_and_salary_perc = group_by_2_level_perc(hr_df, 
                                                          'evaluation', 'salary', 
                                                          evaluation_index_order, ['low', 'medium', 'high'])

#Plot the Graph
p=employees_by_eval_and_salary_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)

num_projects_legend = ['Low', 'Medium', 'High']

customise_2lvl_perc_area_graph(p, num_projects_legend, 
                               xtick_label=evaluation_xticks, x_label=evaluation_x_label, #Company Evaluation Graph
                               y_label="Percentage of Salary Range")


# ### satisfaction level

# In[ ]:


#Create a satisfaction categories
#Arbitrary boundaries:
# < 4.5 low
# 4.5 < < 7.5 medium
# 7.5 < high
def rank_satisfaction(employee):
    level = "unknown"
    if employee.satisfaction_level < 0.45:
        level='low'
    elif employee.satisfaction_level < 0.75:
        level = 'medium'
    else:
        level = 'high'
    return level


# In[ ]:


hr_df['satisfaction'] = hr_df.apply(rank_satisfaction, axis=1)


# In[ ]:


employees_by_eval_and_satisfaction_perc = group_by_2_level_perc(hr_df, 
                                                                'evaluation', 'satisfaction', 
                                                                evaluation_index_order, ['low', 'medium', 'high'])

#Plot the Graph
p=employees_by_eval_and_satisfaction_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)


satisfaction_lvl_legend = ['Low', 'Medium', 'High']

customise_2lvl_perc_area_graph(p, satisfaction_lvl_legend, 
                               xtick_label=evaluation_xticks, x_label=evaluation_x_label, #Company Evaluation Graph
                               y_label="Percentage of Employee's Satisfaction Level")


# ### Using a Decision Tree to Rank Which Variables Contribute Most to an Employee's Evaluation

# I will use the categorised evaluation variable rather than the continuous one, since a decision tree\random forest can be easier to interpret than a regression coefficient.

# In[ ]:


y = hr_df.evaluation.copy()
X = hr_df.copy()
#Remove the label and other columns we have created that categorized continuous variables
X = X.drop(["left", "satisfaction", "last_evaluation", "evaluation", "work_load"], axis=1)


# In[ ]:


X.head()


# ### Sanity Check: Is there any Columns with Missing Data?

# In[ ]:


print(y.isnull().any())
print(X.isnull().any())


# ### Preprocessing

# ***Encode Categorical Columns into Numeric Ones***

# In[ ]:


#Use the label encoder from Scikit-learn to do the conversion
le_sales = LabelEncoder()
le_salary = LabelEncoder()
le_evaluation = LabelEncoder()

#Fit\Transform the data
le_sales.fit(X.sales)
le_salary.fit(X.salary)
le_evaluation.fit(y)

X.sales = le_sales.transform(X.sales)
X.salary = le_salary.transform(X.salary)
y = le_evaluation.transform(y)

#Convert the labels from integers to np so that the estimator wouldn't complain
y = np.float32(y)

min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)


# ### Train\Test Split

# In[ ]:


#train, test = train_test_split(hr_df, test_size= 0.2)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)


# ### Fit the Estimator

# In[ ]:


evaluation_decision_tree = DecisionTreeClassifier()

#Stratify split and train on 5 folds
skf = StratifiedKFold(n_splits=5)
counter = 1
for train_fold, test_fold in skf.split(X_train, y_train):
    evaluation_decision_tree.fit( X_train[train_fold], y_train[train_fold])
    print( str(counter) + ": ", evaluation_decision_tree.score(X_train[test_fold], y_train[test_fold]))
    counter += 1


# In[ ]:


features_order = ['satisfaction_level', 'number_project', 'average_montly_hours', 
                  'time_spend_company', 'Work_accident', 'promotion_last_5years', 'sales', 'salary']
feature_importance_dict = {key: val for key, val in zip(features_order, evaluation_decision_tree.feature_importances_)}

#http://stackoverflow.com/questions/20944483/python-3-sort-a-dict-by-its-values
print([(k, feature_importance_dict[k]) for k in sorted(feature_importance_dict, key=feature_importance_dict.get, reverse=True)])


# So, the top features importance for evaluation is as follows:<br>
# 1- Average Monthly Hours<br>
# 2- Satisfaction Level (Maybe this one is cause by being evaluated higher? Or is it just that satisfied employees give their best?)<br>
# 3- Department<br>
# 4- Number of Projects<br>

# Department is actually the third most important feature. I did not check how the departments were made-up in terms of performance, so let us revisit that:

# In[ ]:


dept_eval_perc = 100*hr_df.groupby(['sales', 'evaluation']).salary.agg({'sales': 'count'})/hr_df.groupby(['sales']).salary.agg({'sales': 'count'})
dept_eval_perc = dept_eval_perc.reindex_axis(['very_low','mediocre', 'average','very_good', 'excellent'], axis=0, level=1)
fig, axs = plt.subplots(figsize=(15, 6))

dept_eval_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', ax=axs, ylim=(0,100))

axs.set_title('Evaluation By Department')
axs.set_xlabel('Department')
axs.set_ylabel('Evaluation % Makeup')

plt.subplots_adjust(hspace=0.3);


# The Management department has a pronounced dip in the lower categories, with the space left seems to be filled by average employees. R&D has a similar but less dramatic pattern.

# ## Understanding what Makes an Employee Happy

# Employees happiness was important in terms of their performance. There is no way to know whichever is the cause for the other, but exploring this aspect is important. I think that any recommendations to be given to the company's upper management will have a lot to draw from this section.

# In[ ]:


satisfaction_index_order = ["unknown", 'low', 'medium', 'high']
satisfaction_xticks = ['Low\n (satisf. < .45)', 'Medium\n ( .45 < satisf. < .75 )', 'High\n ( .9 < satisf.)']
satisfaction_x_label = "Employee's Satisfaction"


# ### Working for a Certain Department?

# In[ ]:


dept_legend = set(hr_df.sales.values)
satisfaction_legend = ['low', 'medium', 'high']

dept_low_satisf_order = (hr_df[hr_df['satisfaction'] == 'low'].groupby('sales').satisfaction_level.count()/hr_df.groupby('sales').satisfaction_level.count()).sort_values(ascending=False).index
dept_low_satisf_order = list(dept_low_satisf_order)

employees_by_satisf_and_department_perc = group_by_2_level_perc(hr_df, 
                                                                'sales', 'satisfaction', 
                                                                dept_low_satisf_order, satisfaction_index_order)

#Plot the Graph
p=employees_by_satisf_and_department_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)



customise_2lvl_perc_area_graph(p, satisfaction_legend, 
                               xtick_label=dept_legend, x_label='Department\n Ordered Descendigly for "Low Happiness" Percentage',
                               y_label="Percentage of Happiness Level")


# The support department is the least "sad" one, while marketing is the highest.

# ### Is Money Everything?

# In[ ]:


employees_by_satisf_and_salary_perc = group_by_2_level_perc(hr_df, 
                                                                'satisfaction', 'salary', 
                                                                satisfaction_index_order, ['low', 'medium', 'high'])

salary_legend = ['Low Salary', 'Medium Salary', 'High Salary']
#Plot the Graph
p=employees_by_satisf_and_salary_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)

customise_2lvl_perc_area_graph(p, salary_legend, 
                               xtick_label=satisfaction_xticks, x_label=satisfaction_x_label, #Employee Satisfaction Graph
                               y_label="Percentage of Employee's Salary Range")


# Not necessarily.

# ### Time Spent Working for the Company

# **Note: The graph starts at 2 years and no values for 9 years**

# In[ ]:


employees_by_satisf_and_salary_perc = group_by_2_level_perc(hr_df, 
                                                                'time_spend_company', 'satisfaction', list(range(2,9)) + [10],
                                                                satisfaction_index_order )


#Plot the Graph
p=employees_by_satisf_and_salary_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)

satisfaction_index_order = ["unknown", 'low', 'medium', 'high']
satisfaction_xticks = ['Low\n (satisf. < .45)', 'Medium\n ( .45 < eval < .75 )', 'High\n ( .9 < satisf.)']

customise_2lvl_perc_area_graph(p, ["Low Satisfaction", "Medium Satisfaction", "High  Satisfaction"], 
                               xtick_label=[0, 1,"2\n years", '3\n years', '4\n years', '5\n years', '6\n years', '7\n years', '8\n years', '10\n years'], 
                               x_label="Time Working for The Company",
                               y_label="Percentage of Employee's Salary Range")

p.set(xlim=(2,9))


# As we progress from year 2 to year 4, the rate of unhappy employees increases at the expense of the two other categories. Then, the rate of highly satisfied employees sharply changes direction. The first 4 years of an employees time working for the company seem to be the critical period.

# ### Workload

# In[ ]:


employees_by_satisf_and_workload_perc = group_by_2_level_perc(hr_df, 
                                                                'satisfaction', 'work_load', 
                                                                satisfaction_index_order, ['low','average','above_average', 'workoholic'])

salary_legend = ['low','average','above_average', 'workoholic']
#Plot the Graph
p=employees_by_satisf_and_workload_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)

customise_2lvl_perc_area_graph(p, salary_legend, 
                               xtick_label=satisfaction_xticks, x_label=satisfaction_x_label, #Employee Satisfaction Graph
                               y_label="Percentage of Employee's Salary Range")


# Almost half of unsatisfied employees don't work very much, maybe they're being passive aggressive? The other interesting thing is that for those who are not satisfied, around 80% work either too much or too little. As we move forward in satisfaction, the percentage of those who work in the mid-ranges increase (They doubled from going low to medium satisfaction at the expense of those who work too hard or too little).
# 
# One may think that there is a correlation between the satisfaction level and the amount of work that an employee puts into work. I would like to see of this is (well, statistically...) true. I will check the correlation of working low hours and an employee's satisfation:

# In[ ]:


low_work_load_employees = hr_df[hr_df['work_load'] == 'low']
pearsonr(low_work_load_employees.satisfaction_level, low_work_load_employees.average_montly_hours)[0]


# In[ ]:


sns.lmplot("satisfaction_level", "average_montly_hours", data=low_work_load_employees);


# A weak correlation. I think the abnormal clusters in this dataset are a reason why we get this result, take a look at the same graph with the colouring according to who left:

# In[ ]:


sns.lmplot("satisfaction_level", "average_montly_hours", hue="left", data=low_work_load_employees);


# ### Number of projects

# In[ ]:


employees_by_satisf_and_proj_num_perc = group_by_2_level_perc(hr_df, 
                                                                'satisfaction', 'number_project', 
                                                                satisfaction_index_order,)

salary_legend = ["2 projects", '3 projects', '4 projects', '5 projects', '6 projects', '7 projects']#['low','average','above_average', 'workoholic']
#Plot the Graph
p=employees_by_satisf_and_proj_num_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)

customise_2lvl_perc_area_graph(p, salary_legend, 
                               xtick_label=satisfaction_xticks, x_label=satisfaction_x_label, #Employee Satisfaction Graph
                               y_label="Percentage of Employee's Salary Range")


# It seems that too many or too little projects are more present in the less satisfied employees. A good hypothesis to work on would be that an employee is happier when they work over 3 to 5 projects. There seems to be a (intuitive) relationship between the work load and the number of projects. Let us check how do they correlate together:

# In[ ]:


#low_work_load_employees = hr_df[hr_df['work_load'] == 'low']
pearsonr(hr_df.number_project, hr_df.average_montly_hours)[0]


# There is a medium correlation.

# ### Using a Decision Tree to Rank Which Variables Contribute Most to an Employee's Satisfaction

# Basically the same process as the one in the previous section

# In[ ]:


y = hr_df.satisfaction.copy()
X = hr_df.copy()
X = X.drop(["left", "satisfaction_level", "satisfaction", "evaluation", "work_load"], axis=1)


# In[ ]:


le_sales = LabelEncoder()
le_salary = LabelEncoder()
le_satisfaction = LabelEncoder()

le_sales.fit(X.sales)
le_salary.fit(X.salary)
le_satisfaction.fit(y)

X.sales = le_sales.transform(X.sales)
X.salary = le_salary.transform(X.salary)
y = le_salary.transform(y)


# In[ ]:


y = np.float32(y)


# In[ ]:


min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)


# In[ ]:


happiness_decision_tree = DecisionTreeClassifier()

#Stratify split and train on 5 folds
skf = StratifiedKFold(n_splits=5)
counter = 1
for train_fold, test_fold in skf.split(X_train, y_train):
    happiness_decision_tree.fit( X_train[train_fold], y_train[train_fold])
    print( str(counter) + ": ", happiness_decision_tree.score(X_train[test_fold], y_train[test_fold]))
    counter += 1


# In[ ]:


features_order = ['last_evaluation', 'number_project', 'average_montly_hours', 
                  'time_spend_company', 'Work_accident', 'promotion_last_5years', 'sales', 'salary']
feature_importance_dict = {key: val for key, val in zip(features_order, happiness_decision_tree.feature_importances_)}
print(feature_importance_dict)


# Features importance for satisfaction:<br>
# 1- Average Monthly Hours<br>
# 2- Last evaluation<br>
# 3- Department<br>
# 4- Number of Projects<br>

# Note that this is almost the same order as the feature importance for employees evaluation, with swapping the second place between both parameters (I mean last evaluation and satisfaction).

# ## Bi\MultiVariate Analysis Conclusions

# For the employees that the company may want to retain, there are a few red flags that they might leave, like:<br>
# 1- They are not satisfied with their job. This one is affected by a few things like:<br>
# &nbsp;&nbsp;&nbsp;&nbsp;a- They work too much or too little. <br>
# &nbsp;&nbsp;&nbsp;&nbsp;b- They are not working on diversified projects (2 projects) or on too many projects (Maybe these are reflected from working too much or too little, but we can think also about not drowning into too much routine or doing excessive multitasking)
# 
# 2- They have been with company about 3 or 4 years
# 
# 3- There seems to be a relationship between an emplyee's performance and their happiness. A constant review and updating for the process of evaluation will likely benefit the company in retaining its employees.

# ### Concerns

# 1- Why does the management department get that much promotions compared to the other departments? They stay a little bit longer with the company, but this may be irrelevant.
# 
# 2- Why 15% of the employees have suffered work-related accidents? More safety measures\training is probablu needed.
# 
# 3- Why does the management department suffer all of these work related injuries? 
# 
# 4- The HR and accoundting departments seem to be more inclined to leave the company, maybe a review is required. But on the other hand, these are also the one that seem to retain higher evaluated employees.

# # Modeling Employees Leaving the Company

# There is no free lunch, but I will just continue on using only a decision tree due to its interpretability, unless in the unlikely situation that it would not yield a high enough accuracy (Unlikely because of all of these cluster of those who left)

# In[ ]:


#Get the label
y = hr_df.left
X = hr_df.copy()
#Drop unnecessary\duplicate columns
X = X.drop(["left", "satisfaction", "evaluation", "work_load"], axis=1)

#Prepocess the data:
#Encode categorical variables into numeric representations
le_sales = LabelEncoder()
le_salary = LabelEncoder()

le_sales.fit(X.sales)
le_salary.fit(X.salary)

X.sales = le_sales.transform(X.sales)
X.salary = le_salary.transform(X.salary)

#Zero mean, 1 std
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

y = np.float32(y)

#Train\Test split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.35, random_state=42, stratify=y)

leaving_random_forest = RandomForestClassifier(n_estimators=100)

#Stratify split and train on 5 folds
skf = StratifiedKFold(n_splits=5)
counter = 1
for train_fold, test_fold in skf.split(X_train, y_train):
    leaving_random_forest.fit( X_train[train_fold], y_train[train_fold])
    print( str(counter) + ": ", leaving_random_forest.score(X_train[test_fold], y_train[test_fold]))
    counter += 1


# In[ ]:


features_order = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 
                  'time_spend_company', 'Work_accident', 'promotion_last_5years', 'sales', 'salary']
feature_importance_dict = {key: val for key, val in zip(features_order, leaving_random_forest.feature_importances_)}


# In[ ]:


#http://stackoverflow.com/questions/20944483/python-3-sort-a-dict-by-its-values
print([(k, feature_importance_dict[k]) for k in sorted(feature_importance_dict, key=feature_importance_dict.get, reverse=True)])


# Features importance for satisfaction:<br>
# 1- Satisfaction Level<br>
# 2- Number of Projects<br>
# 3- Time Spent Working for the Company<br>
# 4- Average Monthly Hours<br>
# 5- Last Evaluation
# 
# The model has a very high accuracy, even when I withheld half the data for testing! (Accuracy dropped one percent only!). So at the end, it may seem easy for the company to predict who is leaving, but the real work starts on figuring out how to make them stay.
