#!/usr/bin/env python
# coding: utf-8

# ## **2018 Kaggle Survey Exploration**

# This year, as in last year, Kaggle conducted an industry-wide survey that presents a truly comprehensive view of the state of data science and machine learning. The survey was conducted live for a week in October.

# ## **Import Python Libraries**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
from bokeh.plotting import figure, show, gridplot
from bokeh.io import output_notebook
from bokeh.palettes import d3, brewer
from bokeh.models import LabelSet, ColumnDataSource

warnings.filterwarnings('ignore')
print(os.listdir("../input"))


# In[ ]:


os.chdir('../input/')


# In[ ]:


output_notebook()


# ## **Read multipleChoiceResponses Data**

# This kernel takes in to account the data present only in the **multipleChoiceResponses.csv**

# In[ ]:


multiplechoice = pd.read_csv('multipleChoiceResponses.csv', low_memory= False)


# In[ ]:


multiplechoice.info()


# From the above cell, it can be found that the multiplechoice dataframe contains 395 columns and 23860 respondents.  There are a total of **50** **questions** in the multipleChoiceResponses.csv file and it is represented in these 395 columns.

# ## **Data Processing**

# Before Analysing the data, the first row of the dataframe, which contains the questions asked in survey, has to be seperated from the rest of the data (numerical data).

# For doing so, here I extracted the question number from the column header and appended the data in the first row to the question number extracted

# In[ ]:


multiplechoice.columns = [x.split('_')[0] for x in list(multiplechoice.columns)]
multiplechoice.head(3)


# In[ ]:


multiplechoice.columns = multiplechoice.columns + '_' + multiplechoice.iloc[0]
multiplechoice = multiplechoice.drop([0])
multiplechoice.head(3)


# The new dataframe with the updated column headers as mentioned is shown above. The column headers now has question numbers along with the question. With the processed data, we can now begin the analysis. 

# First is the Survey Duration ->
# 

# ## **Survey Duration**

# Survey Duration is the time that each respondent took to complete the survey. Since the survey had over 50 questions, it sure takes some time to get it done.     
#                                                                                                                                                                                                                         
# Now lets check how much time people spend on surveys.

# * The data in column1 of the dataframe is of 'str' datatype, eventhough they look like integers. So we must first change the datatype of the column

# In[ ]:


multiplechoice['Time from Start to Finish (seconds)_Duration (in seconds)'] = multiplechoice['Time from Start to Finish (seconds)_Duration (in seconds)'].astype('float')


# In[ ]:


multiplechoice['Time from Start to Finish (seconds)_Duration (in seconds)'] = multiplechoice['Time from Start to Finish (seconds)_Duration (in seconds)'].apply(lambda x:x/3600)


# * The range of values in Time Duration column is pretty big to be represented in seconds. So it is converted into hours so as to bring down the data to a smaller range of values.

# In[ ]:


print(str(round(multiplechoice['Time from Start to Finish (seconds)_Duration (in seconds)'].min()*60, 2)) + ' min')
print(str(round(multiplechoice['Time from Start to Finish (seconds)_Duration (in seconds)'].max(), 2)) + ' hrs')
print(str(round(multiplechoice['Time from Start to Finish (seconds)_Duration (in seconds)'].median(), 2)) + ' hrs')


# The above cell gives the following results:
# 
# 1.  The quickest survey was done in *0.27 min*
# 2. The longest survey took* 246.01 hours* which is over 10 days!!!
# 3. The median time taken to complete the survey is *0.28 hours*

# * As we expected, due to over 50 questions in the survey, the average survey time is high.

# * Larger average survey time can also indicate that most respondents spent time reading and understanding each of the survey questions before answering. Thus we can consider the survey data to be genuine.

# In[ ]:


sns.set_style('darkgrid')
plt.figure(figsize= (16,12))

### Histogram plot
plt.subplot(221)
plt.hist(multiplechoice['Time from Start to Finish (seconds)_Duration (in seconds)'], bins = 25)
plt.yscale('log')
plt.xlabel('Duration (in hrs)', fontsize = 'large')
plt.ylabel('Number of Respondents', fontsize = 'large')
plt.title('Histogram', fontsize = 'x-large', fontweight = 'roman')

### Density plot
plt.subplot(222)
ax = sns.kdeplot(multiplechoice['Time from Start to Finish (seconds)_Duration (in seconds)'])
ax.legend_.remove()
plt.xlabel('Duration (in hrs)', fontsize = 'large')
plt.ylabel('Density', fontsize = 'large')
plt.title('KdePlot', fontsize = 'x-large', fontweight = 'roman')


# * The y-axis of histogram is in log scale. So don't be surprised by the height of the bars!!   
#                                                                                                                                                                                                                                 
# * By looking at the figure, it appears that the bar heigth only goes down slowly, but there is a considerable difference in the number of respondents, as the Duration (in hrs) increases. This can be clearly observed from the Kdeplot.

# * From the above plots, it is easily understood that majority of the people completed the survey within 10hours. Only a very few people took more than 50 hours to complete the survey.

# Now let us analyse the Gender, Age and Country of the Kagglers who did the survey ->

# ## **Gender of Survey Participants**

# In[ ]:


gender = multiplechoice['Q1_What is your gender? - Selected Choice'].value_counts().to_frame()


# In[ ]:


TOOLS="pan,wheel_zoom,zoom_in,zoom_out,undo,redo,reset,tap,save"


# In[ ]:


p = figure(x_range = gender.index.values,plot_height = 450, tools = TOOLS, )
source = ColumnDataSource(dict(x=gender.index.values, y=gender.values.reshape(len(gender))))
labels = LabelSet(x='x', y='y', text='y', level='glyph', x_offset=-15, y_offset=0, source=source, render_mode='canvas')

p.vbar(gender.index.values, width = 0.5, top = gender.values.reshape(len(gender)), color=d3['Category20b'][len(gender)])
p.xaxis.axis_label = 'Gender'
p.yaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
p.xgrid.grid_line_color = None
p.add_layout(labels)
show(p)


# * The number of Male Kagglers is much higher than(almost 5 times) the Female Kagglers  

# ## **Country of Residence of Participants**

# In[ ]:


top_countries = multiplechoice['Q3_In which country do you currently reside?'].value_counts().to_frame().iloc[:10].sort_values('Q3_In which country do you currently reside?')


# In[ ]:


p = figure(y_range = top_countries.index.values, plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(top_countries.index.values, height = 0.5, right = top_countries.values.reshape(len(top_countries)), color=d3['Category20b'][len(top_countries)])
p.yaxis.axis_label = 'Country'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)


# * Since there a lot of participants in the survey from different countries, only the top 10 countries with respect to the number of participants are        considered here.
# 
# * There a lot of Kagglers in USA, India and China compared to the rest of the world.

# 

# Now let us analyze the effect of Gender on the country of residence ->

# ### **Effect of Gender on the country of residence**

# In[ ]:


df = multiplechoice.iloc[:,[1,3,4]].set_index('Q3_In which country do you currently reside?', drop = True)
df = df.loc[top_countries.index.values, :]


# In[ ]:


male, female =[],[]
for val in top_countries.index.values[::-1]:
    male.append(len(df[df['Q1_What is your gender? - Selected Choice'] == 'Male'].loc[val]))
    female.append(len(df[df['Q1_What is your gender? - Selected Choice'] == 'Female'].loc[val]))


# In[ ]:


p1 = figure(x_range = top_countries.index.values[::-1], plot_width = 400, plot_height = 500, tools = TOOLS, title = 'Male')
p1.vbar(top_countries.index.values[::-1], width = 0.5, top = male, color=d3['Category20b'][len(male)])
p1.xaxis.axis_label = 'Country'
p1.yaxis.axis_label = 'Number of Respondents'
p1.yaxis.axis_label_text_font = 'times'
p1.yaxis.axis_label_text_font_size = '12pt'
p1.xaxis.axis_label_text_font = 'times'
p1.xaxis.axis_label_text_font_size = '12pt'
p1.xaxis.major_label_orientation = math.pi/2
p1.xgrid.grid_line_color = None

p2 = figure(x_range = top_countries.index.values[::-1], plot_width = 400, plot_height = 500, tools = TOOLS, title = 'Female')
p2.vbar(top_countries.index.values[::-1], width = 0.5, top = female, color=d3['Category20b'][len(female)])
p2.xaxis.axis_label = 'Country'
p2.yaxis.axis_label = 'Number of Respondents'
p2.yaxis.axis_label_text_font = 'times'
p2.yaxis.axis_label_text_font_size = '12pt'
p2.xaxis.axis_label_text_font = 'times'
p2.xaxis.axis_label_text_font_size = '12pt'
p2.xaxis.major_label_orientation = math.pi/2
p2.xgrid.grid_line_color = None

p = gridplot([[p1, p2], [None, None]])
show(p)


# * When considering the number of Male participants it is India and USA that produces the highest number. 
# * In case of Female participants, it is USA that does produce good number of them followed by India and China.

# 

# Now let us check on the age groups of the Kagglers  ->

# ## **Age group of Kagglers**

# In[ ]:


ylab = multiplechoice['Q2_What is your age (# years)?'].sort_values().unique()
age_df = multiplechoice['Q2_What is your age (# years)?'].value_counts().to_frame().loc[ylab]


# In[ ]:


p = figure(x_range = age_df.index.values, plot_width = 600, plot_height = 400, tools = TOOLS)
p.vbar(age_df.index.values, width = 0.5, top = age_df.values.reshape(len(age_df)), color=d3['Category20b'][len(age_df)])
p.xaxis.axis_label = 'Age Groups'
p.yaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.xaxis.major_label_orientation = math.pi/4
p.xgrid.grid_line_color = None
show(p)


# * The highest number of Kagglers are in the age group 25-29. There are also a lot of Kagglers in the age group 18-21 and in 22-24. 

# * The good number of young Kagglers is definetely an indication that a lot of people have started to consider Data Science as a career option. 
# * This is clearly an indication of the growth of Data Science.

# The countries USA, India and China have over 1000 respondents. So let us step in to these 3 countries to find out the age group of participants from these countries. 

# *United States of America*

# In[ ]:


usa = df.loc['United States of America'].groupby('Q2_What is your age (# years)?').count()


# In[ ]:


p = figure(x_range = usa.index.values, plot_width = 600, plot_height = 400, tools = TOOLS)
p.vbar(usa.index.values, width = 0.5, top = usa.values.reshape(len(usa)), color=brewer['Paired'][len(usa)])
p.xaxis.axis_label = 'Age Groups'
p.yaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.xaxis.major_label_orientation = math.pi/4
p.xgrid.grid_line_color = None
show(p)


# *India*

# In[ ]:


india = df.loc['India'].groupby('Q2_What is your age (# years)?').count()


# In[ ]:


p = figure(x_range = india.index.values, plot_width = 600, plot_height = 400, tools = TOOLS)
p.vbar(india.index.values, width = 0.5, top = india.values.reshape(len(india)), color=brewer['Paired'][len(india)])
p.xaxis.axis_label = 'Age Groups'
p.yaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.xaxis.major_label_orientation = math.pi/4
p.xgrid.grid_line_color = None
show(p)


# *China*

# In[ ]:


china = df.loc['China'].groupby('Q2_What is your age (# years)?').count()


# In[ ]:


p = figure(x_range = china.index.values, plot_width = 600, plot_height = 400, tools = TOOLS)
p.vbar(china.index.values, width = 0.5, top = china.values.reshape(len(china)), color=brewer['Paired'][len(china)])
p.xaxis.axis_label = 'Age Groups'
p.yaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.xaxis.major_label_orientation = math.pi/4
p.xgrid.grid_line_color = None
show(p)


# * In USA, the number of Kagglers is maximum in 25-29 age group. This age group mostly represent working people

# * A lot of young Kagglers are emerging form India. It is easily visible from the large number of people in 18-21 and 22-24 people.
# * This is an indication that most of the Indians are opting Data Science as their career option.

# * China is similiar to USA with most people in age groups 22-24 and 25-29 which mostly represent working class.

# * A lot of Kagglers being under 30 is definetly a proof that Data Science is indeed growing at a good pace. So people can confidently consider Data Science as the career option

# ## **Highest level of Education**

# In[ ]:


educ_lvl = multiplechoice['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts().to_frame()


# In[ ]:


p = figure(y_range = educ_lvl.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(educ_lvl.index.values[::-1], height = 0.5, right = educ_lvl.values.reshape(len(educ_lvl))[::-1], color=brewer['PuBuGn'][len(educ_lvl)])
p.yaxis.axis_label = 'Highest Education Level'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)


# ## **Undergraduate major**

# In[ ]:


und_grad_major = multiplechoice['Q5_Which best describes your undergraduate major? - Selected Choice'].value_counts().to_frame()


# In[ ]:


p = figure(y_range = und_grad_major.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(und_grad_major.index.values[::-1], height = 0.5, right = und_grad_major.values.reshape(len(und_grad_major))[::-1], color=d3['Category20'][len(und_grad_major)])
p.yaxis.axis_label = 'Undergraduate Major'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)


# * Majority of Kagglers have a Masters degree followed by the people with Bachelors and Doctoral degree.                                                                                             
# * There are even guys that don't have a Bachelors degree. 

# * As expected, most of Kagglers have a Computer Science background.
# * There are also a lot of respondents from Engineering background in non-computer focused areas and Mathematics.

# * There are even Kagglers from areas such as fine arts, humanities and so on, even if their number is not high as compared to others

# 

# Let us now analyze the level of education of the Kagglers who have done undergraduate major in Computer Science, Engineering and Mathematics

# *Computer Science*

# In[ ]:


comp_sc = multiplechoice[multiplechoice['Q5_Which best describes your undergraduate major? - Selected Choice'] == 'Computer science (software engineering, etc.)'].iloc[:,5].value_counts().to_frame()


# In[ ]:


p = figure(y_range = comp_sc.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(comp_sc.index.values[::-1], height = 0.5, right = comp_sc.values.reshape(len(comp_sc))[::-1], color=brewer['PuBuGn'][len(comp_sc)])
p.yaxis.axis_label = 'Highest Education Level'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)


# *Engineering (non-computer focused)*

# In[ ]:


engg = multiplechoice[multiplechoice['Q5_Which best describes your undergraduate major? - Selected Choice'] == 'Engineering (non-computer focused)'].iloc[:,5].value_counts().to_frame()


# In[ ]:


p = figure(y_range = engg.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(engg.index.values[::-1], height = 0.5, right = engg.values.reshape(len(engg))[::-1], color=brewer['PuBuGn'][len(engg)])
p.yaxis.axis_label = 'Highest Education Level'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)


# *Mathematics or Statistics*

# In[ ]:


math = multiplechoice[multiplechoice['Q5_Which best describes your undergraduate major? - Selected Choice'] == 'Mathematics or statistics'].iloc[:,5].value_counts().to_frame()


# In[ ]:


p = figure(y_range = math.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(math.index.values[::-1], height = 0.5, right = math.values.reshape(len(math))[::-1], color=brewer['PuBuGn'][len(math)])
p.yaxis.axis_label = 'Highest Education Level'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)


# * The level of education is similiar in all the three cases considered.
# * Most of the respondents have a Masters degree followed by Bachelors and Doctoral degree.

# The above data can help people interested in doing Data Science, in choosing what undergraduate major they should focus on and also the level of education they should go for, to achieve what they want.

# Anyway as seen, there are a lot of Kagglers from different undergraduate areas and with different level of education who loves doing Data Science.                                                                                                                                                                                                                                     
# So don't let your education status affect your decision to join Data Science. 

# 

# Now that we know the education status of respondents, lets go check the employment status of them ->

# The employment details that I wish to consider now are their Job title, type of industry where they are employed in and their experience in the respective areas

# ## **Job Title of Survey Participants**

# In[ ]:


job_title = multiplechoice['Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].value_counts().to_frame()


# In[ ]:


p = figure(y_range = job_title.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(job_title.index.values[::-1], height = 0.5, right = job_title.values.reshape(len(job_title))[::-1], color=d3['Category20b'][len(job_title)-1]+['#636363'])
p.yaxis.axis_label = 'Job Title'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)


# ## **Type of Industry Kagglers work at**

# In[ ]:


industry = multiplechoice['Q7_In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice'].value_counts().to_frame()


# In[ ]:


p = figure(y_range = industry.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(industry.index.values[::-1], height = 0.5, right = industry.values.reshape(len(industry))[::-1], color=d3['Category20b'][len(industry)])
p.yaxis.axis_label = 'Industry Type'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)


# ## **Work Experience**

# In[ ]:


work_exp = multiplechoice['Q8_How many years of experience do you have in your current role?'].value_counts().to_frame()


# In[ ]:


p = figure(y_range = work_exp.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(work_exp.index.values[::-1], height = 0.5, right = work_exp.values.reshape(len(work_exp))[::-1], color=brewer['PuOr'][len(work_exp)])
p.yaxis.axis_label = 'Work Experience'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)


# Let us now discuss the inferences from the above 3 plots ->

# * As indicated by the young age-group, majority of the survey participants are either Students.
# * Since Kaggle is one of the perfect place for one to practice Data Science problems, more students might be turning to Kaggle which explains the large number of students.
# * Apart from students, there are a good number of Data Scientists, Software Engineers and Data Analysts who use Kaggle.
# 

# * As expected Computers/technology is the industry that employs the highest number of the Data Science people.
# * Also even if the number is low, the respondents are also employed in industries such as military, finance, hospitality etc.
# * A lot of industry now depend on Data Science and the plot of Industry type vs Number of Respondents is the proof for that.

# * Finally from the experience plot, it is evident that most people are only with less than 2 years of experience.
# * This clearly is an indication that this is the perfect time to join Data Science domain where the demand for people is high.

# 

# #### For people new to Data Science, there is always a confusion on where to start, and this final section is about that.

# * The first question is always which programming language one should learn
#                                                                                                                                                                                                                                           
# * This section tells you the programming language most used and recommended by the fellow Kagglers so that you can start learning them.

# ## **Most Used Programming Language**

# In[ ]:


pgm_lang = multiplechoice['Q17_What specific programming language do you use most often? - Selected Choice'].value_counts().to_frame()


# In[ ]:


p = figure(y_range = pgm_lang.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(pgm_lang.index.values[::-1], height = 0.5, right = pgm_lang.values.reshape(len(pgm_lang))[::-1], color=d3['Category20b'][len(pgm_lang)])
p.yaxis.axis_label = 'Programming Language'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)


# ## **Most Recommended Programming Language**

# In[ ]:


recom_lang = multiplechoice['Q18_What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'].value_counts().to_frame()


# In[ ]:


p = figure(y_range = recom_lang.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(recom_lang.index.values[::-1], height = 0.5, right = recom_lang.values.reshape(len(recom_lang))[::-1], color=d3['Category20b'][len(recom_lang)])
p.yaxis.axis_label = 'Programming Language'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)


# * **Python** is most used and recommended programming language by the Kaggle Survey Respondents.
# * The usage of python is much higher when compared to other programming languages.

# ## **Most Used Visualization Libraries**

# In[ ]:


visual_lib = multiplechoice['Q22_Of the choices that you selected in the previous question, which specific data visualization library or tool have you used the most? - Selected Choice'].value_counts().to_frame()


# In[ ]:


p = figure(y_range = visual_lib.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(visual_lib.index.values[::-1], height = 0.5, right = visual_lib.values.reshape(len(visual_lib))[::-1], color=d3['Category20'][len(visual_lib)])
p.yaxis.axis_label = 'Visualization Libraries'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)


# * Once you start using a programming language, the next important thing is the visualization of data.
# * It is clear from the kernel that visualizations can easily give you insight about the data. Proper insight can help you decide how you approach the data better

# The above plot contains the most used visualization libraries.

# ## **Common Data types People Interact with at Work or School**

# In[ ]:


datatypes = multiplechoice['Q32_What is the type of data that you currently interact with most often at work or school? - Selected Choice'].value_counts().to_frame()


# In[ ]:


p = figure(y_range = datatypes.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(datatypes.index.values[::-1], height = 0.5, right = datatypes.values.reshape(len(datatypes))[::-1], color=d3['Category20b'][len(datatypes)])
p.yaxis.axis_label = 'Common DataTypes People Interact With'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)


# * Now that you have seen the most used programming language and visualization libraries, it is also important to know the type of data people interact with.
# * Numerical data is most used data followed by Tabular and Text data.
# * People new to this field can start with these datatypes, which is commonly used and much easier to handle than some of the other datatypes like audio and video data.

# ## **Most Used Online Platforms for Data Science**

# In[ ]:


online_plat = multiplechoice['Q37_On which online platform have you spent the most amount of time? - Selected Choice'].value_counts().to_frame()


# In[ ]:


p = figure(y_range = online_plat.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(online_plat.index.values[::-1], height = 0.5, right = online_plat.values.reshape(len(online_plat))[::-1], color=d3['Category20b'][len(online_plat)])
p.yaxis.axis_label = 'Online Platforms Kagglers Use'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)


# * Now that you have figured on what programming language and visualization library you should learn, and the type of data who might be handling, the final question that still remains is where you can learn these

# There a lot of Online Platforms that provides good courses, that you can learn all these on your own.

# * Coursera is the most preferred online platform used by Kagglers to learn Data Science. 
# * People new to Data Science can thus use Coursera or other platforms which they find suitable, to learn all that is needed to fulfill their dreams :)

# 

# One other thing that people new to any field wonders about, is the yearly compensation they get.                                                                                                           
# Let us explore the yearly compensation of Kagglers ->

# ## **Yearly Compensation**

# * The following plot ignores the respondents who do not wish to disclose their approximate yearly compensation.

# In[ ]:


yearly_comp = multiplechoice['Q9_What is your current yearly compensation (approximate $USD)?'].value_counts().to_frame()[1:]


# In[ ]:


p = figure(y_range = yearly_comp.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(yearly_comp.index.values[::-1], height = 0.5, right = yearly_comp.values.reshape(len(yearly_comp))[::-1], color=d3['Category20b'][len(yearly_comp)])
p.yaxis.axis_label = 'Yearly Compensation (approx $USD)'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)


# * Most of the respondents are only having yearly compensation in 0 - 30000 range. By looking at the experience graph, it is clear that the reason for the low yearly compensation is the fewer years of experience.
# * As experience increases, so does the yearly compensation.
# * There are also good number of Kagglers with yearly compensation of 100k+ 

# #### Happy exploring! Let me know any noteworthy findings in the comments.
