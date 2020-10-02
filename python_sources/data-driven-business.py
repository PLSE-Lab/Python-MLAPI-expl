#!/usr/bin/env python
# coding: utf-8

# In this report we will do an Exploratory Data Analysis (EDA) to identify the most significant variables that could effect attrition. 
# 
# #### Problem: 
# 
# The company is worried about the departure of the most valuable talent to other firms. The HR department is designing a new HR program to reduce the employee attrition by identifying and retaining the most valuable talent subject to leave. 
# 
# 
# Initially we are going to analyze the whole dataset and understand what is in it and how the variables are distributed.
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Checking data
data = pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
data.head()


# In[ ]:


data.describe()


# In[ ]:


#Changing attrition into boolean
data["Attrition"] = np.where(data['Attrition']=='Yes', 1, 0)


# In[ ]:


data.head()


# In[ ]:


#Checking for null values
data.isnull().sum()


# In[ ]:


sns.countplot('Attrition', data=data)


# In[ ]:


count_attrition = pd.DataFrame(data['Attrition'].value_counts())
count_attrition['percentage'] = round((count_attrition["Attrition"]/data.Attrition.count())*100,2)
count_attrition


# In[ ]:


# we can see that only 16% seem to fall into attrition, it's very unbalananced to create a model based on this data


# In[ ]:


count_attrition


# In[ ]:


#Checking distribution
plt.hist(data["Age"], bins=20)


# In[ ]:


#We can see that the Age is sort of normally distributed and that the mean is around 37 years old
data["Age"].mean()


# In[ ]:


#We proceed to make a pitvot table to explore the different variables based on attrition (Yes and No)
t=data.pivot_table(index='Attrition', aggfunc='mean')
t


# In[ ]:


t.Age


# In[ ]:


t.MonthlyIncome


# In[ ]:


t.JobSatisfaction


# In[ ]:


#Then we decided to explore the variables we considered important for a predictive model such as Age, Monthly Income and Job Satisfaction
#The results show that the age average for attrition is lower which shows that older employees tend to keep their jobs
#Then the montly income shows the obvious, the employees with lower salaries tend to leave the company.
#There is a slightly difference in Job Satisfaction where employees with lower satisfaction tend to leave.


# In[ ]:


x=data[data.Attrition==0]
y = data[data.Attrition==1]


# In[ ]:


plt.figure(figsize=(10,5))

plt.subplot(111)

sns.distplot(x.Age, color='red')
sns.distplot(y.Age,  color='blue')
plt.legend(title="Attrition", loc='upper right', labels = ["No", "Yes"])
plt.title('Age')


# In[ ]:


#We also wanted to review the age variable more into depth and the distributions shows that the younger people tend to 
#quit their job more often. The mean age for attrition is 33 years old. It just proved what we observed in the previous pivot table


# In[ ]:


#Checking the distributions
plt.figure(figsize=(10,5))

plt.subplot(221)

sns.kdeplot(x.MonthlyIncome, shade= True, color='red')
sns.kdeplot(y.MonthlyIncome, shade= True, color='blue')
plt.legend(title="Attrition", loc='upper right', labels = ["No", "Yes"])
plt.title('Monthly Income')

plt.subplot(222)
sns.kdeplot(x.DailyRate, shade= True, color='red')
sns.kdeplot(y.DailyRate, shade= True, color='blue')
plt.legend(title="Attrition", loc='upper right', labels = ["No", "Yes"])
plt.title('Daily')

plt.subplot(223)
sns.kdeplot(x.MonthlyRate, shade= True, color='red')
sns.kdeplot(y.MonthlyRate, shade= True, color='blue')
plt.legend(title="Attrition", loc='upper right', labels = ["No", "Yes"])
plt.title('Monthly Rate')

plt.subplot(224)
sns.kdeplot(x.HourlyRate, shade= True, color='red')
sns.kdeplot(y.HourlyRate, shade= True, color='blue')
plt.legend(title="Attrition", loc='upper right', labels = ["No", "Yes"])
plt.title('Hourly Rate')
plt.tight_layout()

plt.show()


# In[ ]:


#The monthly income distribution seem to be supporting our theory that the people that are earning the lest are more likely to leave
# Also the people who are earning less on a daily basis seem to be more likely to leave
#It can be also noticeable in the Hourly rate in a lower proportion but still follows the logic
#However, it is curious to review the Montly Rate distribution because it does not follow the logic


# ### Tableau

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1575998527239' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Da&#47;Dashboard2_15759984165290&#47;Dashboard2&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Dashboard2_15759984165290&#47;Dashboard2' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Da&#47;Dashboard2_15759984165290&#47;Dashboard2&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1575998527239');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1227px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# First graph shows Years at company in relation to Years since last promotion, grouped by both attrition and overtime. The graph may mean that people are not really growing within the company so they tend to leave. We can see more percentage wise that there are more people working overtime and also leaving. It seems that there may be a pattern of people leaving because they are not promoted although they work hard. 
# 
# Second graph shows that those who rated their work-life balance relatively low were travelling longer to work in comparison with those who rated their work-life balance as really good. This difference is more pronounced in the group of those who are not in the company anymore, suggesting a possibly influential attrition factor.
# 
# 
# Third graph shows a trend that most of the people staying are not doing overtime which makes sense. On the other side, the people that are leaving the company are very close. There is a small difference in favor of overtime which makes sense.
# 
# 
# Forth graph shows that people who are staying in the company are satisfied with their job. Those employees are doing more overtime which coincides also with them stating there is a good work environment. Opposite to logic, people that are leaving are working less overtime. However, it can also mean that these people are less willing to work in the company.
# 

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1575998582864' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Da&#47;Dashboard1_15759969896590&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Dashboard1_15759969896590&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Da&#47;Dashboard1_15759969896590&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1575998582864');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1227px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# First Graph shows that the people that are leaving are traveling freqently a bit more than those that are staying. Other than that it seems like it there's not much of a difference between the leaving and staying people in terms of traveling.
# 
# Second graph shows that in general, people that are leaving are also earning less, no matter what level of education they have. It also shows that people that are earning a lot and have a high education, are staying. It just points out the importance of the salary for the employees.
# 
# Third graph shows that most of the peolpe that are leaving are single no matter the sex. While people that are staying, it is mostly married. It shows that an unstable marital status like single are more prone to change their jobs.
# 
# Forth graph, it shows that income is the most sensitive variable to predict attrition. Job satisfaction is simnilar across the board. It does not vary so it is not sensitive for the attrition model.

# In[ ]:


pd.set_option('display.max_columns', 30)


# In[ ]:


thresh = 0.4


# In[ ]:


correlation = data.corr()


# In[ ]:


correlation


# We proceed to review the high correlations among the variables to decide whihc ones are more appropiate for the model.

# ### Conclusions
# 
# Attrition can be explained mostly based on the imbalance of the income. The salary is the most sensitive variable to describe it. It specially applies to employees working overtime with low incomes. The income is reflected in different variables such as Education, Job Level, Overtime, Year at company and more. 
# 
# We have also found that different variations of work-life balance might represent an issue for our employees. 
# There seems to be a link between attrition and age as well as the number of companies worked for.
# Also, there seems to be a clear pattern that the people who left travelled more frequently compared to the others. This might have also been a big reason behind them leaving. 
# 
# In order to fight attrition, we believe that we should focus on the following variables: 
# We considered Age, Monthly Income, OverTime, Marital Status, JobLevel, YearsAtCompany, Number of Companies worked for, Business Travel.
#  
# 
