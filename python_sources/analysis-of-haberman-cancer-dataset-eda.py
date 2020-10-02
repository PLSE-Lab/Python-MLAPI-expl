#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <h1>Description of the data</h1>
# <hr>
# The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.

# In[ ]:


haberman = pd.read_csv("/kaggle/input/habermans-survival-data-set/haberman.csv")
#haberman.columns
haberman.columns = ['age','year','nodes','status']
haberman.columns


# These are the features we have in the dataset where 'status' being the dependent variable. Lets go through each of them. 

# <h1>Attribute information</h1>
# <hr>
# 1. Age : Age of patient at time of operation (numerical)
# 2. Year : Patient's year of operation (year - 1900, numerical)
# 3. Nodes :Number of positive axillary nodes detected (numerical)
# 4. Status : Survival status (class attribute) 1 = the patient survived 5 years or longer 2 = the patient died within 5 year
#  

# In[ ]:


haberman.status.value_counts()


# We can see that the number of people who have survived is comparatively more than the number of people who didn't. Its a good thing. But lets dive a little deep into it. 

# In[ ]:


# mean age 
print("Mean of age = {0:.1f}".format(haberman.age.mean()))
# median age
print("Median of age = {0:.1f}".format(haberman.age.median()))
print("Max age = {0:.1f}".format(haberman.age.max()))
print("-"*50)
# mean age 
print("Mean of nodes = {0:.1f}".format(haberman.nodes.mean()))
# median age
print("Median of nodes= {0:.1f}".format(haberman.nodes.median()))
print("Max nodes = {0:.1f}".format(haberman.nodes.max()))
print("-"*50)
print("Mean of years = {0:.1f}".format(haberman.year.mean()))
# median age
print("Median of years= {0:.1f}".format(haberman.year.median()))
print("Max years = {0:.1f}".format(haberman.year.max()))
print("-"*50)


# Mean and Median are two measures of central tendency of the data. However, mean is highy affected by the presence of outliers or when the data is skewed. If the data is symetric then the value of mean is close to the value of median. 
# 
# Here from the above analysis, we can observe that age and year data points seem to be symetric as thier mean and meidan are close to each other. However the mean of nodes is quite greater than the median of nodes. It indicitates that our data is "<a href="https://www.mathsisfun.com/data/skewness.html">Right skewed</a>" i.e. it has some outliers on the larger side.  
# 
# Now lets plot some graphs to better understand the data.

# <h1>Bi-variate analysis using Scatter Plots</h1>
# Bi-variate analysis refers to the analysis of data using any two features/variables. Scatter plots are used to plot data points on a horizontal and a vertical axis in the attempt to show how much one variable is affected by another. We can plot scatter plots between different variables here and see how it affects our surrvival status. 
# 
# Lets look at one example.

# <h1>Age vs Year</h1>

# In[ ]:


haberman.plot(kind='scatter',x='age',y='year')
plt.show()


# This is a scatter plot between year vs age but this is really confusing , lets color the dots based on the survival status and see what we get. 
# 

# In[ ]:


sns.FacetGrid(haberman,hue='status',height=7)    .map(plt.scatter,'age','year')    .add_legend()
plt.show()


# So this gives us a much clearer picture than the previous one. Lets analyze it. 
# <br>At first glace we can deduce,
# 
# 1. People yougner than the age 40 most likely survived. 
# 1. There are very less number of cases above the age of 70. 
# 
# But this is a very overlapping graph. I am sure we can find a good pair of variables where things are clearer. 
# We have 3 independendent variables. And with the theory of combinations, there will be 3C2 number of possible unique pairs. Well, there is a good a way to plot all these pairs with a few lines of code. 

# In[ ]:


sns.set_style('whitegrid')
sns.pairplot(haberman,hue='status',height=5)
plt.show()


# The above figure has 9 plots in total. The digonal plots are called Probability Densifity Function plots. We will explore them later. For now lets stick to the scatter plots. 
# 
# Among the rest of 6 scatter plots, 3 are just mirror images of each other. (As we disscussed, there can only be 3C2 = 3 number of unique plots possible). So we will not consider the plot-4 , plot-7 and plot-8. 
# <br>
# Lets analyze the rest. 

# <h1>Age vs Nodes (Plot - 3)</h1>

# In[ ]:


sns.FacetGrid(haberman,hue='status',height=7)    .map(plt.scatter,'age','nodes')    .add_legend()
plt.show()


# This graph looks quite interesting. We can clearly see how the number of dots are heavily clustered at the lower side of the nodes and gradually decrease as the number of nodes increase. So lets write our observation down. 
# 
# 1. It is very rare to have more than 20 number of nodes. 
# 2. Peopele with nodes less than equal to 1 and age between 50 to 60 or less than equal to 40 , most likely survived. 
# 3. People with nodes greater than 10 and age between 50 to 70 , most likely did not survive. 
# 4. Less number of nodes seem to be a good sign.
# 
# 

# <h1>Node vs Year (Plot - 8)</h1>

# In[ ]:


sns.FacetGrid(haberman,hue='status',height=7)    .map(plt.scatter,'nodes','year')    .add_legend()
plt.show()


# This graph doesn't seem to help much as a lot of cases are overlapped onto each other. Lets just move on. 

# All the scatter plot analysis that we did can be generalised as bi variate analysis. Which means analysing the data based on two variables. Now for the next phase of analysis, we will dive into univariate analysis i.e. see how a single variable affects the survivle of a patient, using the concepts of Proababilty desnity function (PDF) and Cummulative density function (CDF).

# <h1>Univariate analysis using PDF and CDF</h1>
# <hr>

# We will not go deep into what Probability Density Function is but for now we can assume, in a pdf graph the x axis reprensents the variable we have taken and y axis represent the count of cases corresponding to a particualr variable.
# Lets draw the PDF (Probability Density Function) of age and see.  

# <h3>PDF of Age</h3>

# In[ ]:


sns.FacetGrid(haberman,hue='status',height=7)    .map(sns.distplot,"age")    .add_legend()
plt.show()


# As I had explained before the x-axis here represents different values of ages and the y-axis represent the count. Well  you can see its not actually count because they are all in decimal , they are actually the probability density but lets not go deeper and assume they are some form of count. 
# 
# The above PDF of age shows the following things ,
# 
# 1. People younger than 40 most likely survied. Infact all the people youger than 34 survived. 
# 2. People between the age gap 40 - 70 almost have the same survival rate. 
# 3. People above the age 75 tend not to survive.
# 
# Lets draw PDF for other variables and explore. 

# <h3>PDF of Year</h3>

# In[ ]:


sns.FacetGrid(haberman,hue='status',height=7)    .map(sns.distplot,"year")    .add_legend()
plt.show()


# As the previouse analysis, the year variable doesn't tell us much. Lets move on to the Nodes. 

# <h3>PDF of Nodes</h3>

# In[ ]:


sns.FacetGrid(haberman,hue='status',height=7)    .map(sns.distplot,"nodes")    .add_legend()
plt.show()


# As always the presence of nodes seems to be the key factor in deciding the survival status. From the above graph we can deduce 
# 
# 1. People with number of nodes close to 0 (0 to 2) have much higher chances of survival. 
# 2. Survival chances of people with number of nodes greater than 2 is low. 

# Lets draw the CDF (Commulative Density Function) plot to explore more. You can assume CDF as a "commulative version of PDF", in simple words lets assume, for point age = 20 , PDF told us how many cases were there for age = 20. But CDF will tell us , for age = 20 , how many cases are there for age <= 20 (Well, it will simply not be a count but the probability). 

# <h3>CDF of Nodes</h3>

# In[ ]:



counts , bin_ages = np.histogram(haberman[haberman.status ==1].nodes,bins=10,density=True)
pdf = counts/sum(counts)

cdf = np.cumsum(pdf)
plt.plot(bin_ages[1:],pdf)
plt.plot(bin_ages[1:],cdf)



plt.show()


# In the above graph the orange line is our CDF and the blue one is the PDF. Analysing both of them , we can see , nearly 80% of the people who survied had auxilary nodes less than equal to 5. 80-85% of people who survived had number of nodes less than or equal to 7.  But then as the node number increases the survival rate also decreases drastically. 

# In[ ]:


counts , bin_ages = np.histogram(haberman[haberman.status ==2].nodes,bins=10,density=True)
pdf = counts/sum(counts)

cdf = np.cumsum(pdf)
plt.plot(bin_ages[1:],pdf)
plt.plot(bin_ages[1:],cdf)



plt.show()


# From this CDF we can see, nearly  55-60% of people who didn't survive had nodes less than 5.

# In[ ]:



counts , bin_ages = np.histogram(haberman[haberman.status == 1].nodes,bins=10,density=True)
pdf = counts/sum(counts)

cdf = np.cumsum(pdf)
plt.plot(bin_ages[1:],pdf)
plt.plot(bin_ages[1:],cdf)

counts , bin_ages = np.histogram(haberman[haberman.status == 2].nodes,bins=10,density=True)
pdf = counts/sum(counts)

cdf = np.cumsum(pdf)
plt.plot(bin_ages[1:],pdf)
plt.plot(bin_ages[1:],cdf)


plt.show()


# Observations from above graph : 
# 
# 1. Combining the above two graphs we can confirm this that people with auxilary nodes <= 5 , more likely survived. However, following this simple rule will still give us an error of 55% in the case of people who didn't survive
# 2. After the number of nodes increase beyond 20 , the chances of survial and non survial is same as the red and orange line overlap with each other. 
# 3. Looking at the difference between green and blue line (PDFs) we can say that there are less number of people who survived with higher number of nodes and otherwise.  

# <h3>CDF of Age</h3>

# In[ ]:



counts , bin_ages = np.histogram(haberman[haberman.status == 1].age,bins=10,density=True)
pdf = counts/sum(counts)

cdf = np.cumsum(pdf)
plt.plot(bin_ages[1:],pdf)
plt.plot(bin_ages[1:],cdf)

counts , bin_ages = np.histogram(haberman[haberman.status == 2].age,bins=10,density=True)
pdf = counts/sum(counts)

cdf = np.cumsum(pdf)
plt.plot(bin_ages[1:],pdf)
plt.plot(bin_ages[1:],cdf)


plt.show()


# Observations from the above graph are as follows : 
# 1. People yougner than the age of 40 most likely to survive. 
# 2. After the age of 50 the survial and non survial chances are similar to each other as both orange and red lines are overlapping. 
# 3. Before the age of 50 the survival chances seem higher. 

# Lets check out the spread of data using concpets like standard deviation , median absolute deviation and Quantiles.

# <h1>Standard Deviation , Median Absolute Deviation and Quantiles</h1>
# <hr>
# 

# In[ ]:


print("Standard deviation of nodes for people who survived  ")
print(np.std(haberman[haberman.status==1].nodes))

print("Standard deviation of nodes for people who didn't survive ")
print(np.std(haberman[haberman.status==2].nodes))
print("-"*50)

print("Standard deviation of ages for people who survived")
print(np.std(haberman[haberman.status==1].age))

print("Standard deviation of ages for people who didn't survive")
print(np.std(haberman[haberman.status==2].age))
print("-"*50)


# From the above analysis you can observe that the spread of number of nodes in people who couldn't survive is comparatively higher than the people who survived. However, the age seem to be spread equally in both the scenarios.

# In[ ]:


from statsmodels import robust
print("Median absolute deviation of nodes for people who survived")
print(robust.mad(haberman[haberman.status==1].nodes))

print("Median absolute deviation of nodes for people who didin't survived")
print(robust.mad(haberman[haberman.status==2].nodes))
print("-"*50)

print("Median absolute deviation of age for people who survived ")
print(robust.mad(haberman[haberman.status==1].age))

print("Median absolute deviation of age for people who didn't survived ")
print(robust.mad(haberman[haberman.status==2].age))
print("-"*50)


# Median absolute deviation is another measure of spread , used when our data is skewed. In this case it doesn't provide any extra information than the previous analysis. So lets move on. 

# As we had seen earlier that there seem to be some oulier in nodes datapoints since mean and median of nodes had reasonable difference. So we will use median to check the central tendency of both the classes. 

# In[ ]:


print("Median of nodes for people who survived ")
print(np.median(haberman[haberman.status==1].nodes))

print("Median of nodes of people who didn't survive ")
print(np.median(haberman[haberman.status==2].nodes))
print("-"*50)

print("Median of age of people who survived")
print(np.median(haberman[haberman.status==1].age))

print("Median of age of people who didn't survive ")
print(np.median(haberman[haberman.status==2].age))
print("-"*50)


# From the above analysis we can conclude that people who have survived , more likely had less number of auxilary nodes , while it seems to  be the opposite case for people who didn't survive. 
# The median of age doesn't carry much information as it is close to each other in both the cases. 
# 

# Lets print the percentiles and quantiles of the datapoints and see what the ytell us. 

# In[ ]:


print("Quantiles of nodes for people who survived")
print(np.percentile(haberman[haberman.status==1].nodes,np.arange(0,100,25)))
print("90th percentile of nodes for people who survived")
print(np.percentile(haberman[haberman.status==1].nodes,90))
print("\n")
print("Quantiles of nodes for people who didn't survive")
print(np.percentile(haberman[haberman.status==2].nodes,np.arange(0,100,25)))
print("90th percentile of nodes for people who didn't survive")
print(np.percentile(haberman[haberman.status==2].nodes,90))
print("-"*50)
print("Quantiles of age for people who survived")
print(np.percentile(haberman[haberman.status==1].age,np.arange(0,100,25)))
print("90th percentile of age for people who survived")
print(np.percentile(haberman[haberman.status==1].age,90))
print("\n")
print("Quantiles of age for people who didn't survive")
print(np.percentile(haberman[haberman.status==2].age,np.arange(0,100,25)))
print("90th percentile of age for people who didn't survive")
print(np.percentile(haberman[haberman.status==2].age,90))


# Looking at the analysis we observe : 
# 1. Among the people who survived had , 75% of them had 0 number of auxilary nodes and 90% of them had less than 8 number of auxilary nodes.
# 2. Among the people who didn't survive, only less than 25% of them had 0 auxilary nodes. 90% of them had less than 20 auxilary nodes. 
# 3. So we can wirte if auxilary nodes <= 3 then survial = 1 (survived) else suvival = 2 (didn't survive). This will give us a 75% accuracy for people who survived but 50% of error for people who didn't.

# Lets use these concepts of percentiles and quanitles and plot some graphs. 

# <h1>Boxplot with whiskers</h1>
# <hr>

# In[ ]:


sns.boxplot(x='status',y='nodes',data=haberman)
plt.show()


# First a little introduction of what we are looking at. In the orange box of the figure, the lower edge of the box represents the 25% Quantile (Q1), the line inside the box represents the 50% Quantile (Q2) and the upper edge of the box represents the 75% Quantile (Q3). The width of the box represents the *mid 50%* of the datapoints (from 25% to 75%). The same goes for the blue box but the Q1 and Q2 are so close they are overlapping with each other. 

# Now that we know a littel bit about box plots , lets analyse what they are telling us.  You can see that the middle 50% of the nodes of people who survived fall between 0 to 3. While as middle 50% of the nodes of people who didn't survive for long, fall between 1 to 11. 
# 
# The straight lines above and below the boxes are called whiskers , which represent the max and min value of the data points. (For the blue box the min point is equal to Q1 so it is overlapped).

# In[ ]:


sns.boxplot(x='status',y='age',data=haberman)
plt.show()


# This is a box plot with respect to age variable. It looks a lot clearer than before. Analyzing this gives us the following observations. 
# 1. 50% of the cases in both classes aproximately fall under  42 to 60 years old. 
# 2. All the people below the age 34 survived. 
# 3. Anyone above the age 77 didn't survive. 

# <h1>Violin Plots</h1>
# <hr>

# In[ ]:


sns.violinplot(x='status',y='nodes',data=haberman)
plt.show()


# Violine plots are the combination of both Boxplots and PDF. The little gaps you can see in between each plot represent the box plots, while the area sorrounding it is the PDF. The same PDF is mirrored on both sides of the box plot vertically for asthetics purposes. 

# In[ ]:


sns.violinplot(x='status',y='age',data=haberman)
plt.show()


# This is the violin plot for age. 

# Analyzing both of them gives no new information but confirms our previous analysis. 

# <h1>Contour Plots</h1>
# <hr>

# Contour plots (sometimes called Level Plots) are a way to show a three-dimensional surface on a two-dimensional plane. The higher the density of the poitns the darker the color of the region in the plot. 
# 
# Below is a contour plot between age and nodes for the people who survived.

# In[ ]:


sns.jointplot(x='age',y='nodes',data=haberman[haberman.status==1],kind='kde')
plt.show()


# Observations from the above plot are : 
# 1. Most people who survived are between the age 45-65 and with nodes between 0 to 1.

# In[ ]:


sns.jointplot(x='age',y='nodes',data=haberman[haberman.status==2],kind='kde')
plt.show()


# Observations from the above plot are : 
# 1. Most people who didn't survive are between the age 47 to 55 with nodes between 3 to 5

# <h1>Conclusion</h1>
# <hr>

# After analysing the dataset closely with various concepts of statistics and graphs , we reach the following conclusions

# 1. Number of nodes is the most important feature of the dataset that affect the survival 
# 2. Higher number nodes increases the chances of death within 5 years. 
# 3. Age also somewhat affects the chances of survival. People with higher nubmer of nodes and age above 50 years old most likely didn't survive.
# 4. People with number of nodes close to 0 and age between 30-35 or 50-60 years old tend to survive.
# 
# 

# <h3></h3>
