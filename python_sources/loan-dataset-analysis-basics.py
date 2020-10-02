#!/usr/bin/env python
# coding: utf-8

# <font size="5">Welcome to Loan Dataset Analysis</font>
# 

# <img src="https://images.unsplash.com/photo-1518458028785-8fbcd101ebb9?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=750&q=80" width="1000px">
# 
# 
# 
# Photo by Sharon McCutcheon on Unsplash
# 

# <font size="5">Objective</font>

# Welcome to this kernel where we will be exploring a loan dataset adn try to find the insights on it 
# 
# First of all we will import required python liabraries. Dataset used in this kernel is freely available on Kaggle ( free for public usage) 
# 

# In[ ]:



import numpy as np # Numpy library
import pandas as pd # Pandas library
import matplotlib.pyplot as plt # Matplotlib library for visualisation 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns # Matplotlib library for visualisation 

import warnings # Import warning liabraries to ignore standard warnings 
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

import os # os liabrary to find the directory where dataset is placed
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


mydf=pd.read_csv("../input/Loan payments data.csv") # Read the dataset CSV
mydf.head(5) # Let's find top 5 records of the dataset 


# In[ ]:


# Let's generate descriptive statistics of 
#dataframe (mydf) using describe function 
mydf.describe()


# <font size="5">Dataset Details</font>
# > 
# 
# 
# > Please find below details of the dataset which can help to understand the dataset
# 1. Loan_id : A unique loan (ID) assigned to each loan customers- system generated
# 2. Loan_status : Tell us if a loan is paid off, in collection process - customer is yet to payoff, or paid off after the collection efforts
# 3. Principal : Pincipal loan amount at the case origination OR Amount of Loan Applied
# 4. terms : Schedule
# 5. Effective_date : When the loan got originated (started)
# 6. Due_date : Due date by which loan should be paid off
# 7. Paidoff_time : Actual time when loan was paid off , null means yet to be paid 
# 8. Pastdue_days : How many days a loan has past due date 
# 9. Age : Age of customer 
# 10. Education : Education level of customer applied for loan
# 11. Gender : Customer Gender (Male/Female)[](http://)

# In[ ]:


#Let's concise summary of our dataset using pandas info function
mydf.info()


# Check how many missing values do we have in our dataset ?

# In[ ]:


#From below query we can see we have 100 null (NAN) values in paid_off_time and 300 null values in 
#past_due_days which is fine , reason - if someone pays earlier before due date these columns will not
#have values specified

mydf.isnull().sum()


# In[ ]:


# Check dataset shape [rows, columns],below query shows we have a dataset of 500 rows , 11 columns
mydf.shape


# 
# 
# **Let's do some Exploratory Data Analysis (EDA) **
# 
# Lets create some visualisation to see what dataset tells us when we interrogate it by visaulising it
# 
# 
# 

# In[ ]:


sns.set(style="whitegrid") # Lets set background of charts as white 


# In[ ]:


# First of all lets find out how many loan cases are Paid Off, Collection or Collection_PaidOff status
x = sns.countplot(x="loan_status", data=mydf )


# Above Graph shows that nearly 60% of loan are in PAID OFF State while , 20 % are in Collection and 20% are in CollectionPaidOff status

# 
# Now Let's see loan status based on Gender .... 

# In[ ]:


y = sns.countplot(x="loan_status", data=mydf , hue='Gender')


# Above graph shows that Girls have lesser ratio of loans 

# 
# 
# 
# 
# 
# 
# Let's see how many people loan applications have been applied for weekly (7 days), Fortnightly (15 Days) , Monthly (30days) payment mode and whats the loan status , how well are weekly, fortnightly and monthly loans are in terms of paid status 

# In[ ]:


x = sns.countplot(x="terms", data=mydf , hue='loan_status', palette='pastel', linewidth=5)


# 
# 
# From above graph we can see that very few people go for weekly pay off , however fortnighty and monthly payment modes are quiet famous 
# Most of the applications are having monthly mode as people do get monthly wages mostly from where they would like to pay off for the loan amount

# .

# Lets move further and see how education affects the loan amount and payment status **?**

# In[ ]:


g = sns.catplot("loan_status", col="education", col_wrap=4,
                 data=mydf[mydf.loan_status.notnull()],
                 kind="count", height=12.5, aspect=.6)


# 
# Above visualisation shows that most of the loans applied in below series
# 1. College 
# 2. High School or below 
# 3. Bachelor
# 4. Master and Above 
# 
# What this series tells us **?????**
# 
# Well from above graph we can deduce that most of the college or high school students apply for the loan and pay back well in time so they are better candidates for loans by banks 
# 
# However for Bachelors degree students - Loan applications are less and return is also not so rewarding means a bit less preferable to return the money by themselves 
# 
# 
# Lastly - Very few loan applications for students going for Masters and above degree which is a valid insight as very few people (from crowd) opt for masters  degree or higher (bit costly than others) so less applicants for loan. Such candidates pay off well and very few people take time to pay back to bank 
# 
# 
# 

# 
# 
# 
# 
# 
# .

# Lets go further to dig down to see how Gender variable impacts the loan amount 
# 
# If we see below graph it clearly tells us the amount of loan amount applied by Male/Female candidates ( yeah yeah i know its very less amount in really none applies for 300 , 500 figures... we working with masked dataset :) ) 
# 
# 

# .
# .
# .
# .
# ..
# 

# 1. Below visualisation shows us Age vs Principal amount based on Gender 

# In[ ]:


ax = sns.barplot(x="Principal", y="age",hue="Gender" ,  data=mydf)
ax.legend(loc="upper right")


# Well below insight is almost same telling about how many applications are Paid Off , in Collection , Collection_PaidOff Status
# Nothing different i am just trying to showcase different color combination ( paster color) soothing to eyes when client wants nice presentations , we should use paster color combinations as they have nice bright color combination. For further information we can google it up 

# In[ ]:


sns.set(style="whitegrid")
ax = sns.countplot(x="loan_status", hue="Gender", data=mydf ,palette='pastel' ,edgecolor=sns.color_palette("dark", 3))


# Now if we want to showcase different graphs for different Gender to understand how many Loan applications made by Male/Females
# 
# Below visuations is best example as it clearly shows that Male candidates do have more Loan applications compared to Female candidates 
# and most of the canidates who apply for the loan are in higher side of Principal Amount means they wish to go for Better studies 

# In[ ]:


fig = plt.figure(figsize=(25,5))
g = sns.catplot(x="Principal", hue="loan_status", col="Gender",palette='pastel',
                data=mydf, kind="count",
                 height=4, aspect=.7);


# In[ ]:


# Lets draw a pairplot to see data visualisation from different variables impact factor 
sns.pairplot(mydf, hue='Gender')


# >

# >

# >

# When we really want to see the spread of the data , Violin plots are one of the good example 
# For example - lets say that we wish to see Principal Amount for which Loan has been applied by Male vs Female candidates based on their age range
# 
# If we talk about very first plot of violin in below graph, only Male candidates applied for Principal 400- 500 range
# 
# Female candidates apply for Principal amount starting with 800 and apply for most of application having principal amount = 1000

# In[ ]:


sns.set(style="whitegrid", palette="pastel", color_codes=True)

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x="Principal", y="terms", hue="Gender",
               split=True, inner="quart",
               
               data=mydf)
sns.despine(left=False)


# Lmplot visualisation - Just to showcases data distribution of Loan applications based on age
# 
# From below visualisation we can clearly say that most of the applications are of higher amount 800, 1000 with a very less applications of other amount 

# In[ ]:


g = sns.lmplot(x="age", y="Principal", hue="Gender",
               truncate=True, height=5, data=mydf)

# Use more informative axis labels than are provided by default
g.set_axis_labels("Age", "Principal")


# .

# .

# Seaborn Replot will tell us the data spread ... i just wanted to see the applications Principal Amount based on Education , age 
# 
# From below visualisation again we can confirm that most of the loan applicants do apply for loan amount as 800 , 1000 ( Seems like most of the people applicants go for higher/better studies for a bright future 

# In[ ]:


# Plot miles per gallon against horsepower with other semantics
sns.relplot(x="Principal", y="age", hue="education",size="Gender",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=mydf)


# How many people applied and paid off before due date ..... Sad 

# In[ ]:


#(mydf.shape[0])
mydf['past_due_days'].isnull().sum()


# In[ ]:



defaultPerc=((mydf.shape[0]-mydf['past_due_days'].isnull().sum())/mydf.shape[0])*100
print(defaultPerc,"% of people paid after time")


# .

# **Show me - in a high Level how many applications are paying before time or after due date **
# 
# From below visualisation we can see that 40% people paid loan after due date which is not a good figure we need to work on our loan collection process and streamlining the loan guidelines so that people pay before time rather than being defaulter 
# 
# I dont like **Defaulters :)** 
# 
# 
# <img src=" https://images.unsplash.com/photo-1456534231849-7d5fcd82d77b?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=888&q=80" width="1000px">
# 
# *
# Photo by meredith hunter on Unsplash*

# In[ ]:


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'After Due Date', 'Before Due Date'
sizes = [defaultPerc,100-defaultPerc]
explode = (0, 0.1)  # only "explode" the 2nd slice 

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

fig1.suptitle('People who paid Before Due Date or After Due Date', fontsize=16)


plt.show()


# **Box Plot - ** 
# 
# Another great visualisation to see data spread, median and outliers in the dataspread 
# 
# From below visualisation we can clearly see that people going for better education ( College , Masters ) apply for high Principal amount of loan , which is true , better/Higher educations are costly and come with price
# 
# <img src=" https://images.unsplash.com/photo-1523050854058-8df90110c9f1?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1350&q=80" width="1000px">
# 
# Photo by Vasily Koloda on Unsplash

# In[ ]:


sns.boxplot(x='education', y='Principal', data=mydf)
plt.show()


# .
# 

# .

# Below is just another visualisation showing both male and female applicants who are going for higher education apply for Principal amount of 800 or 1000

# In[ ]:


sns.lmplot(x='Principal', y='age', hue = 'Gender', data=mydf, aspect=1.5, fit_reg = False)

plt.show()


# In[ ]:


sns.lmplot(x='Principal', y='age', hue = 'education', data=mydf, aspect=1.5, fit_reg = False)
plt.show()


# Lets check whats the loan applications applied on various dates and see the pattern

# 
# 
# 
# 
# 
# <img src=" https://images.unsplash.com/photo-1556740758-90de374c12ad?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1350&q=80" width="1000px">
# 
# 
# Photo by Blake Wisz on Unsplash

# **Insight** 
# 
# Below figure shows some unsual activity of loan applicants on 11 September ( need to check with Business Sales Expert) on reason why so many people applied/got loan on 11th September? What was unusual as most of the people who applied on 11th September are in defaulters' list ( Did we gave loans to Negative Credit Rating people as well without doing enough background checks ? )
# 
# 
# 
# We should definitely check business reason for high number of loan applications on 11th September which could further reveal the facts like lesser interest rates etc or loan without background check or any suspicious activities so we need to check with Bank Sales team expert what and why this happened and what we doing to prevent such occurances in future

# In[ ]:


fig = plt.figure(figsize=(15,5))
ax = sns.countplot(x="effective_date", hue="loan_status", data=mydf ,palette='pastel' ,edgecolor=sns.color_palette("dark", 3))
ax.set_title('Loan date')
ax.legend(loc='upper right')
for t in ax.patches:
    if (np.isnan(float(t.get_height()))):
        ax.annotate(0, (t.get_x(), 0))
    else:
        ax.annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))
plt.show();


# 

# **Overall High level insights in a nutshell**
# **
# 
# 
# <img src=" https://images.unsplash.com/photo-1444653614773-995cb1ef9efa?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1355&q=80" width="1000px">
# 
# 
# Photo by Adeolu Eletu on Unsplash

# 1. People going for higher studies apply for loan with Principal Amount of 800 , 1000 
# 2. Male candidates apply wide variety of loans 
# 3. 40% of people applying for loans to this (xyz) bank are defaulters whcih means bank need to work on their policies and recovering rules
# 4. An unsual activity seen in data on 11th September showing very high sales with a higher defaulters - 
#     4.1 any change in bank loan policy on 11th Sep
#     4.2. Was our loan agent well aware about loan policy before giving loan to customer
#     4.3  Did our loan agent do proper credit rating checks before giving loans?
#     4.4  Similarly seems like even on 9th , 10th Sep some policy check failure , which dindt check credit worthiness of applicants 

# **Please upvote my kernel if you like it...Feeback also welcome ** 
# 
# <img src=" https://images.unsplash.com/photo-1534293230397-c067fc201ab8?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=334&q=80
# " width="1000px">
# 
# Photo by Parker Johnson on Unsplash

# In[ ]:




