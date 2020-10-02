#!/usr/bin/env python
# coding: utf-8

# # Introduction
# As a marathon runner, this dataset naturally got my attention. Let's how what's the power of data science for my beloved sport. 
# 
# One thing I notice while I am watching marathons on TV is that, there is usually projected finished time at the bottom. Of course, this is only for the elites. Nowadays, you can also track almost all the runner through live tracking, with the chip attached to each runner. When the runner pass one of the timing mats along the course, their times will be sent to the tracking server, and naturally, one can predict their projected finish time. The power we have here is that, we have more information about the individual runner, such as nationality, and I think this is the goal of this exercise - using the intermediate time plus these extra information, to build a better predictor.
# 
# # Data inspection and cleaning
# ## - Runner information
# Firstly, I will load the training data, and take a peek of the structure.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)import re
import fnmatch    # for later text matching
import matplotlib.pyplot as plt    # for plotting
plt.style.use('ggplot')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv('../input/challenge.csv')
print(train_df.shape)
train_df.head()


# There are 5616 rows and 11 columns, where the last 3 columns are the 10 km, half way (21km) and 30 km time.  However, there is no apparent gender, and age information. They can be included in the "Category" code, let's take a closer look at this Category.

# In[ ]:


train_df["Category"].unique()


# So there are 8 different category codes, and it consists of three letters plus one optional number.  All the category codes start with letter 'M', therefore it bears no information in this context (my best guess is that 'M' stands for Marathon). I will then move on to 2nd letter, where the options are 'M' and 'F'. An educational guess is that it distinguishes 'M'ale and 'F'emale, but let's check anyway.

# In[ ]:


# true table for male and female
Male_list = [fnmatch.fnmatch(item, '?M*') for item in train_df["Category"].tolist()]
Female_list = [fnmatch.fnmatch(item, '?F*') for item in train_df["Category"].tolist()]
# create the male and female dataframes
train_Male_df = train_df[Male_list]
train_Female_df = train_df[Female_list]


# If the guess is correct (and there is no missing data), then the row number of the gender-specific dataframe should equal to the Gender Position of the respective last row. Let's see....

# In[ ]:


print('The total entries in Male dataset is ' + str(train_Male_df.shape[0]))
print('The Gender Position of the last entry in Male dataset is '+ str(train_Male_df["Gender Position"].iloc[-1]))
print('The total entries in Female dataset is ' + str(train_Female_df.shape[0]))
print('The Gender Position of the last entry in Female dataset is '+ str(train_Female_df["Gender Position"].iloc[-1]))


# First observation is that the male runners are more than 5 times more than female runners. Well, this is close but not quite right... since the sum of the entry numbers (4713 + 903 = 5613) equals to the entry number of the original dataset, therefore I am not missing anything. Also the discrepancies are (relatively) small, which is enough for me to validate my guess (the meaning of the 2nd letter).
# 
# Then the explanation for the difference can only be that there are discontinuities (such as due to ties) in the Gender Position. Although such discontinuities probably won't be a problem if my goal is to predict the finishing time, but I am just can't let it go... at the very least, I want to find out where they happen...
# 
# (note added: I will not print this outcome since it takes too much space...)

# In[ ]:


# temporay dataframe
temp_df = pd.DataFrame(columns=train_df.columns)
for i in range(1, train_df.shape[0]):
    if (abs(train_df["Overall Position"].iloc[i] - train_df["Overall Position"].iloc[i-1]) > 1
        and train_df["Official Time"].iloc[i] != train_df["Official Time"].iloc[i-1]):
        temp_df = temp_df.append(train_df.iloc[i-1])
        temp_df = temp_df.append(train_df.iloc[i])


# Ok, many observations:
# 
# First, the complexity is (partially) caused by the tie in the Official Time, since the Position is ranked by the Official Time, rather than the Net Time. When there is a tie, it is broken neither by "Net Time", nor by "Race No". See examples of lines (89 and 90), which are seemingly broken by "Net Time" not by "Race No", and examples of lines (204 and 205), which are seemingly broken by "Race No" but not "Net Time".
# 
# Second, the format of column "Race No" becomes float, rather than the original integer. It is because the dataframe "temp_df" is empty before I insert the first row, therefore its default format is float. It shouldn't affect later operations, but just keeping in mind. 
# 
# Given this seemingly annoying mismatch, I decide to rank the entries by myself. During the process, let me also convert the time to seconds, in order to facilitate later regressions.

# In[ ]:


ranked_df = train_df.copy(deep = True)
## check data types
# print(ranked_df.dtypes)
official_time_temp = pd.DatetimeIndex(ranked_df['Official Time'])
net_time_temp = pd.DatetimeIndex(ranked_df['Net Time'])
ranked_df['Official Time (s)'] = official_time_temp.hour*3600 + official_time_temp.minute*60 + official_time_temp.second
ranked_df['Net Time (s)'] = net_time_temp.hour*3600 + net_time_temp.minute*60 + net_time_temp.second
ranked_df.sort_values('Net Time (s)');
ranked_df.tail(5)
    


# ## - Runner demographic
# Now I have the data nicely sorted, I will proceed to look at the runner demographic. Recall that there are 8 different categories, 4 for each gender, let me break them down:
# 

# In[ ]:


# prepare the data
groups = ranked_df['Category'].unique();
group_labels = [x[1:] for x in groups];  

# initilize the runner demographic dictionary
runner_demo = {label: 0 for label in group_labels}; 

# update the dictionary
for i in range(0, ranked_df.shape[0]):
    runner_demo[ranked_df.loc[i]['Category'][1:]] += 1

# plot!
gender_less_group = [x[1:] for x in group_labels if x[0] == 'M']
female_demo = [runner_demo['F' + x] for x in gender_less_group]
male_demo = [runner_demo['M' + x] for x in gender_less_group]
index = np.arange(len(group_labels)/2)
bar_width = 0.35
opacity = 0.4
fig, ax = plt.subplots()
rects1 = ax.bar(index, female_demo, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Female')
rects2 = ax.bar(index + bar_width, male_demo, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Male')
plt.xlabel('Category')
plt.ylabel('Number of runners')
plt.title('Runners by category and gender')
plt.xticks(index + bar_width, (gender_less_group))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.tight_layout()
def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
plt.show()


# That is fun... But an afterthought: I didn't take any advantage of pandas, i.e. using its built-in methods. Why shouldn't I?

# In[ ]:


ranked_df['Gender'] = ranked_df['Category'].apply(lambda x: x[1])    # create a gender column
ranked_df['Gender less category'] = ranked_df['Category'].apply(lambda x: x[2:])    # create a gender less category column
gender_cat = ranked_df['Gender'].groupby(ranked_df['Gender less category']) 
gender_cat.value_counts().unstack().plot(kind = 'bar')    # unstack the multi-index series, and then plot!
plt.ylabel('Number of runners');


# Aside from the absence of the number on top of each bar, I am quite happy with the slick codes enabled by pandas! Given that the efforts behind pandas, I should use its functionality as much as possible. With regard to the showing number on the bar, or even better, with interactive hovering feature, I shall try the Bokeh libraries. Let's see... (One problem, though, is that the figure will not be rendered until the notebook is published. Hence, debugging will be a bit tedious...)
# 

# In[ ]:


from bokeh.plotting import figure, output_file, show, output_notebook
from bokeh.charts import Bar
from bokeh.models import ColumnDataSource, HoverTool
output_notebook()  # this is to enable the plot the figure in notebook

hover = HoverTool(
        tooltips=[
            ("Num", "@height{int}"),  
        ]
    )

p = Bar(ranked_df, values = 'Gender', label = 'Gender less category', agg = 'count', stack = 'Gender' tools=[hover])
show(p)


# Next, let's look at how the runners break down to countries, and genders from each country. Let's get a big picture first...

# In[ ]:


gender_country = ranked_df['Gender'].groupby(ranked_df['Country ']).value_counts().unstack().fillna(0).astype(int)
gender_country['Total'] = gender_country.apply(np.sum, axis = 1)
gender_country = gender_country.sort_values('Total', ascending = False)
gender_country['Percentage'] = gender_country['Total'] / gender_country['Total'].sum() * 100
gender_country.head(5)


# It is quite clear that the first 3 countries and regions have made up the majority of the runners.  Therefore, it is better to group the rest of the countries into one category (such as "Others"). Let me do that.

# In[ ]:


gender_country_2 = gender_country.iloc[0:3]
gender_country_3 = gender_country.iloc[3:]
gender_country_2
others = gender_country_3.apply(np.sum)
others.name = 'Others'
gender_country_2 = gender_country_2.append(others)
gender_country_2['Total'].plot(kind = 'pie')


# In[ ]:


countries = ranked_df['Country '].unique()
f_country = {x: 0 for x in countries}
m_country = {x: 0 for x in countries}
for i in range(0, ranked_df.shape[0]):
    if ranked_df.loc[i]['Category'][1] == 'F':
        f_country[ranked_df.loc[i]['Country ']] += 1
    elif ranked_df.loc[i]['Category'][1] == 'M':
        m_country[ranked_df.loc[i]['Country ']] += 1

# plot!
index = np.arange(len(countries))
country_list = [[x, f_country[x], m_country[x], f_country[x] + m_country[x]] for x in countries]
country_list.sort(key = lambda x: x[-1])

bar_height = 2
index = np.arange(len(country_list))
opacity = 0.4
fig = plt.figure(figsize=(5,9), frameon=False)
ax = fig.add_subplot(111)

bar1 = ax.barh(3*bar_height*index, [x[1] for x in country_list], 
              height = bar_height, align='center', alpha=opacity, 
              color = 'b', label = 'Female')
bar2 = ax.barh(3*bar_height*index + bar_height, [x[2] for x in country_list], 
              height = bar_height,align='center', alpha=opacity, 
              color = 'r', label = 'Male')
plt.yticks(3*bar_height*index, ([x[0] for x in country_list]))
plt.tick_params(
    axis='y',          # changes apply to the y-axis
    which='both',      # both major and minor ticks are affected
    left='off',      # ticks along the bottom edge are off
    right='off')         # ticks along the top edge are off  
plt.legend(('Female', 'Male'), loc = 4)
plt.show()


# Well, the plot didn't look quite nice, but that's mainly due to the fact that the distribution is largely skewed towards the first 3 countries (and regions): HK, CHN, and JPN. In this case, maybe a table-like metric is more appropriate. Let's see...

# In[ ]:


ranked_df['Country '].describe()


# ## TODO
# - prettify the above plot
# - look for "cheaters"
