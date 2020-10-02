#!/usr/bin/env python
# coding: utf-8

# > # Exploratory data analysis

# # The data

# Haberman data contains information from a study conducted between 1958 to 1969 of patients who has done surgery for breast cancer.
# 
# * The column, nodes, represents the number of auxillary nodes affected. Auxillary nodes are located in a person's armpits. 
# More info on auxillay nodes and breast cancer: https://www.medicalnewstoday.com/articles/319713.php
# 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


haberman = pd.read_csv("../input/haberman.csv",header=None, names=['age', 'year', 'nodes', 'status'])
print(haberman.columns)
print (haberman.head())                       
survived = haberman[haberman.status == 1]
died = haberman[haberman.status == 2]


# In[ ]:


print ('Number of rows: {}'.format(haberman.shape[0]))
print ('Number of columns: {}'.format(haberman.shape[1]))


# * There are 306 observations and 4 variables
# 
# Let's look at few of the observations

# In[ ]:


print ('*********** Top five observations ***********')
print (haberman.head())
print ('*********** Bottom five observations ***********')
print (haberman.tail())


# * All the columns are numbers. However, we know that the status and years are categorical variables

# In[ ]:


haberman.describe()


# * We have data of people who has done operation between 1958 and 1969 with age ranging from 30 to 83
# 

# In[ ]:


haberman['status'].value_counts()


# * Status 1 represents the patients survived 5 years of longer after the surgery and 2 represents the patients who died within 5 years of surgery
# 
# * In our data, there are 225 observations of survived patients and 81 of people who did not survive, implying an imbalanced dataset 

# # Objective

# There are 81 patients died within 5 years of the surgery, but 225 did not.
# 
# __Can we find the variables influencing the death of the patients using simple plotting techniques?__ 
# 
# Let's find out..! 

# In[ ]:


features = list(haberman.columns)
target = 'status'
features.remove(target)
print (features)


# # Univariate Analysis

# ### Probability density function and cumulative density function 

# In[ ]:


def plot_pdf_and_cdf(data, variable):
    counts, bin_edges = np.histogram(haberman[variable], bins=10, 
                                 density = False)
    pdf = counts/(sum(counts))
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:], pdf)
    plt.plot(bin_edges[1:], cdf)
    plt.title('{}: PDF and CDF '.format(variable.capitalize()))
    plt.xlabel(variable)
    
    
plt.figure(figsize=(15,5))
for index, feature in enumerate(features):
    plt.subplot(1,3, index+1)
    plot_pdf_and_cdf(haberman, feature)


# ### Findings:
# * Most of the patients are aged below 70
# * The year variable seems to be faily distributed across the years
# * Around 81% of patients have less than 10 auxilary nodes detected 

# In[ ]:


print ('**** Age **** \n 95th percentile:{} \n 99th percentile:{}'.format(np.percentile(haberman['age'],95),round(np.percentile(haberman['age'],99),2)))
print ('**** Nodes **** \n 95th percentile:{} \n 99th percentile:{}'.format(np.percentile(haberman['nodes'],95),round(np.percentile(haberman['nodes'],99),2)))
print ('**** Year **** \n 95th percentile:{} \n 99th percentile:{}'.format(np.percentile(haberman['year'],95),round(np.percentile(haberman['year'],99),2)))


# ### Findings

# * 95 % of the patients are aged below 70 and 99% are aged below 76
# * 99% of the patients have less than 30, implying that it is rare for for the breast cancer to spread to over 30 nodes

# ### Histogram and density plots

# In[ ]:


def hist_density_plot(data, feature, target):
    sns.FacetGrid(data, hue=target, size=5).map(sns.distplot, feature)     .add_legend().fig.suptitle('Distribution of {}'.format(feature))

    


# In[ ]:


hist_density_plot(haberman, 'age', 'status')


# * From the distribution, age does not have much influence on the survival. 
# * Also, ages range from 30 to 83 - no indication of outliers 
# 
# 

# In[ ]:


print ('Mean age of the survived patients: {}'.format(round(np.mean(survived['age']), 2)))
print ('Mean age of the died patients: {}'.format(round(np.mean(died['age']), 2)))


# * Mean of both died and survived patients are close to each other, strengthening our conclusion that age does not influence on the survival

# In[ ]:


plt.figure(figsize=(1,1))
hist_density_plot(haberman, 'nodes', 'status')


# * The nodes' spread of died patients seems to be wide. Let's look at the mean values

# In[ ]:


print ('Mean nodes of the survived patients: {}'.format(round(np.mean(survived['nodes']), 2)))
print ('Mean nodes of the died patients: {}'.format(round(np.mean(died['nodes']), 2)))


# * There is a noticable difference between the averages nodes of the survived and of the dead. However, from the distribution, it is clear that there are outliers in the data

# In[ ]:





# In[ ]:


hist_density_plot(haberman, 'year', 'status')


# * As we already know, the dataset contains observations from 1958 and 1969
# * The data seems to be fairly distributed with peaks at certain periods. However, the influence on the target variable is unclear from the plot 

# ### Box plots

# In[ ]:


def box_plot(data, x, y, ax):
    sns.boxplot(x=x, y=y, data=data, ax = ax).set_title('Boxplot: {} vs {}'.format(x.capitalize(),y.capitalize()))
    #plt.show()

    


# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for index, feature in enumerate(features):
    box_plot(data = haberman, x = 'status', y = feature, ax = axes[index])
plt.show()


# * Age is neatly distributed with no outliers. Both survived and died patients have a similar average age
# * The year variable does not have any outliers and does not seem to influence the survival chance
# * As seen earlier, there are few outliers in the nodes variable

# In[ ]:


print ('Median nodes of the survived patients: {}'.format(round(np.median(survived['nodes']), 4)))
print ('Median nodes of the died patients: {}'.format(round(np.median(died['nodes']), 4)))


# * The median of number of nodes is significantly different between the survived and the dead, indicating influence of number of nodes on the survival of patients

# ### Violin plots

# In[ ]:


def violin_plot(data,target, y, palette, ax):
    sns.violinplot(x= target, y= y, data=haberman, size=8, hue = target, palette= palette, ax = ax)     .set_title('Violin plot: {} vs {}'.format(target.capitalize(),y.capitalize()))
    plt.legend(title = 'status',  loc = 'lower center')


# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(17, 4))
for index, feature in enumerate(features):
    violin_plot(data = haberman, target = 'status', y = feature, palette = 'Set2',ax = axes[index])
plt.show()


# * Age and Year does not have any influence on our target variable
# * Number of auxilary nodes affected seems to the influence on our target variable

# # Bivariate analysis

# 

# ### Age and Nodes 

# In[ ]:


sns.FacetGrid(haberman, hue="status", size=4)    .map(plt.scatter, "age", "nodes")    .add_legend();
plt.show();


# * We cannot linearly seperate our class using nodes and ages

# ### Age and Year

# In[ ]:


sns.FacetGrid(haberman, hue="status", size=4)    .map(plt.scatter, "age", "year")    .add_legend();
plt.show();


# * We cannot linearly seperate our class using nodes and year

# ### Nodes and Year

# In[ ]:


sns.FacetGrid(haberman, hue="status", size=4)    .map(plt.scatter, "nodes", "year")    .add_legend();
plt.show();


# * We cannot linearly seperate our class using nodes and year
# * It is difficult to linearly seperate the target class from the above plots

# ### Pair plot

# In[ ]:


sns.pairplot(haberman, hue="status", size=3, vars = features);
plt.show()
#plt.close()


# * None of the plots helps us in seperating the target class as there is plenty of overlap 

# # Conclusions

# * The number of auxillary nodes detected affects the survival chance 
# * Age and year of surgery do not affect the survival chance
# 
