#!/usr/bin/env python
# coding: utf-8

# ## Human Resource Analytics

#  This is my first **kernel** on Python and in this notebook I intend to do data analysis and visualisation on the HR dataset.I intend to use pandas,seaborn for my visualisation.As always, **if you like my kernel,pls upvote**

# In[ ]:


#loading the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(context="notebook",style="white",palette="dark")
plt.style.use('fivethirtyeight')


# In[ ]:


# read the datafile
HR=pd.read_csv("../input/HR_comma_sep.csv")


# In[ ]:


# glimpse of the data
HR.head()


# In[ ]:


#Getting to know about the data types
HR.info()


# In[ ]:


#Check for anymissing values
HR.isnull().sum()


# The dataset is superclean with no variables having null values.Therefore we directly venture into some data munging.

# ## Average Monthly Hours Trend:

# Let us check the trend of the average monthly hours.

# In[ ]:


plt.figure(figsize=(12,7))
sns.distplot(HR.average_montly_hours,bins=30,kde=False)
plt.title("Distribution of Average Monthly Hours")
plt.xlabel("Average Monthly Hours",fontsize=12)
plt.ylabel("Count",fontsize=12)
plt.show()


# From the plot,it is understood that the curve is bimodal with peaks at around 150 and 280.This means that the average  monthly working hours of most people is 150 hrs and 280 hrs.

# Let us check this trend over turnover rates.

# In[ ]:


plt.figure(figsize=(12,7))
ax=sns.kdeplot(HR.loc[(HR.left==0),'average_montly_hours'],color="g",shade=True,label="Stays in company")
ax=sns.kdeplot(HR.loc[(HR.left==1),'average_montly_hours'],color="r",shade=True,label="Left the company")
ax.set(xlabel='Average monthly hours',ylabel="Frequency")
plt.title("Employee Turnover with Average Monthly Hours",fontsize=16)


# * From the plot,it is understood that there is an increasing trend of turnover for employees who work for 150 hrs on an monthly average and this trend is replicated for 250 average monthly working hours.
# * The same trend is observed for those who do not leave the company but the frequency difference is higher for those who work for 150 avg monthly hours than who work for 250 avg monthly hours.

# ## Satisfaction Level:

# Let us check the satisfaction level of the employees with the help of histogram.

# In[ ]:


plt.figure(figsize=(12,5))
sns.distplot(HR.satisfaction_level,kde=False)
plt.title("Distribution of Satisfaction Level",fontsize=16)
plt.xlabel("Satisfaction Level",fontsize=12)
plt.ylabel("Count",fontsize=12)


# The satisfaction level is around 0.8 for most of the employees.Let us analyse it separately with turnover.

# In[ ]:


fig,ax=plt.subplots(ncols=2,figsize=(12,5))
left=HR[HR.left==0]
stay=HR[HR.left==1]
sns.kdeplot(left.satisfaction_level,shade=True,color="r",ax=ax[0],legend=False)
ax[0].set_xlabel("Satisfaction Level")
ax[0].set_ylabel("Density")
ax[0].set_title("Those who Leave")
sns.kdeplot(stay.satisfaction_level,shade=True,color="g",ax=ax[1],legend=False)
ax[1].set_xlabel("Satisfaction Level")
ax[1].set_ylabel('Density')
ax[1].set_title('Those who stay')
plt.suptitle("Satisfaction level Vs Turnover",fontsize=16)


# It is strange that those who leave have higher satisfaction level compared significantly to those who stay.Is it true or am i making any mistake here ????

# ## Number of Projects 

# In[ ]:


fig=plt.figure(figsize=(12,8))
g=sns.factorplot(x="number_project",hue="left",data=HR,kind="count",legend_out=True,size=8,aspect=0.7)
g._legend.set_title("Turnover")
plt.xlabel("Number of Projects",fontsize=12)
plt.ylabel("Count",fontsize=12)
plt.title("Number of Projects Vs Turnover",fontsize=16)


# It is seen that there are people with 3,4 number of projects and tend to stay within the company.But there are  people with 2 projects,7 projects leave the company.With this data,we come to know that people with less or the most projects leave the company whereas people who get balanced projects tend to stay.This is indicated by huge difference between turnover and no turnover for 3,4,5 projects.

# Let us analyse the satisfaction level for various project levels with the help of boxplot.

# In[ ]:


fig=plt.figure(figsize=(10,10))
sns.boxplot(x="number_project",y="satisfaction_level",hue="left",data=HR,palette='viridis',linewidth=2.5)
plt.xlabel("Number of Projects",fontsize=12)
plt.ylabel("Satisfaction Level",fontsize=12)
plt.title("Boxplot of Satisfaction Level with Number of Projects",fontsize=16)


# From the boxplot it is understood that people with 2 and 3 projects have  shown higher level of satisfaction and stay with the company.There is a huge and significant difference between the satisfaction levels of those who leave and stay with number of projects as 6.

# ## Last Evaluation 

# Let us check the trend of last evaluation and the turnover rates.

# In[ ]:


HR.last_evaluation.describe()


# The mean evaluation score is 0.716 whereas the median score is 0.72.Let us check who all have scored above and below the score.

# In[ ]:


print("There are {} people having evaluation score greater than 0.7".format(len(HR[HR.last_evaluation>0.7])))
print("There are {} people having evaluation score lesser than 0.7".format(len(HR[HR.last_evaluation<0.7])))


# From the summary and the above statement,it seems that more people have evaluation score <0.7.

# In[ ]:


plt.figure(figsize=(7,5))
sns.distplot(HR.last_evaluation,bins=30,color="r")
plt.xlabel("Evaluation",fontsize=12)
plt.ylabel("Frequency",fontsize=12)
plt.title("Distribution of Evaluation",fontsize=16)


# Let us see this evaluation score with turnover and number of projects.

# In[ ]:


ax=plt.figure(figsize=(7,8))
ax=sns.factorplot(x="left",y="last_evaluation",col="number_project",data=HR,kind="box",size=4,aspect=0.6)
#ax.set(xlabel="Turnover",ylabel="EvaluationScore",title="Trend of Turnover with Number of Projects and Evaluation")
ax.set_xlabels("Turnover")
ax.set_ylabels('Evaluation Score')
ax.fig.suptitle("Trend of Turnover with Number of Projects and Evaluation",x=0.5,y=1.2)


# * From the boxplot data,we find that there is a significant difference in the evaluation scores between tunover rates as stated earlier.
# * There is a higher evaluation scores for people handling more projects (projects > 3) but despite this they tend to leave the company.
# * This trend is reversed for number of projects =1 where we find that people  having evaluation score of 0.65 tend to stay with the company. 

# ## Time spent with the company 

# Let us know about the average number of years people tend to spend with the company

# In[ ]:


pd.crosstab(HR["time_spend_company"],HR["left"],margins=False).apply(lambda x: (x/x.sum())*100).round()


# From the table,it is seen that 44 %  have  left the company who had 3 years of experience. whereas almost a closer percentage 42 % stay in the company.Let us visualise this in the form of barplot.

# In[1]:


ax=plt.figure(figsize=(7,5))
ax=sns.barplot(x="time_spend_company",y="time_spend_company",data=HR,hue="left",estimator=lambda x: len(x) / len(HR) * 100)
ax.set(ylabel="Percentage")
ax.set(title="Years In Company Vs Turnover")
ax.set_ylim(0,50)


# ** Work in progress ** ..
