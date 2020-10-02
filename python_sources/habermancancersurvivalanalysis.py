#!/usr/bin/env python
# coding: utf-8

# # Haberman Cancer Survival Study

# ## Description from Kaggle:
# The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.

# ## Objective : 
#     1. To analyse survival of patients who have undergone surgery for brest cancer
#     

# In[620]:


import pandas as pd
columns = ['age_of_patient', 'year_of_operation', 'positive_axilary_nodes', 'survival_status']
csdf = pd.read_csv('../input/haberman.csv', names = columns, header = None)
csdf.head()


# In[622]:


# Changed survival status into english for better understanding
csdf.survival_status.replace([1, 2], ['survived', 'dead'], inplace=True)
csdf.head()


# In[629]:


# number of datapoints and features
csdf.shape


# In[631]:


# 306 datapoints and 4 features to classify between survived and dead patients


# In[636]:


csdf['survival_status'].value_counts()


# In[638]:


# Pairplotting all the data to visually check what features can easily classify between the classes
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.pairplot(csdf, hue='survival_status', size=5, markers="+").add_legend()


# ### Observations
# 1. Datapoints are not identical they are overlapping each other only few conculsions can be made
# 2. 225 out of 306 patients survived in total which is 73.5% survival.
# 3. Year 1961 had best cases of survival of cancer patients whereas 1965 had worst cases of suvival of cancer patients
# 
# In[645]:


# plotting one dimentionally
sns.FacetGrid(csdf, hue = 'survival_status', size = 5).map(sns.distplot, 'age_of_patient').add_legend()


# In[653]:


sns.FacetGrid(csdf, hue = 'survival_status', size = 5).map(sns.distplot, 'year_of_operation').add_legend()


# In[655]:


sns.FacetGrid(csdf, hue = 'survival_status', size = 5).map(sns.distplot, 'positive_axilary_nodes').add_legend()


# ### Univariate Analysis

# In[661]:


# Survival percentage on yearly basis:
import numpy as np
yof = set(csdf['year_of_operation'])
for year in yof:
    csdf_temp = csdf.loc[csdf['year_of_operation'] == year]
    x, y = csdf_temp['survival_status'].value_counts()
    per_x, per_y = x*100/(x+y), y*100/(x+y) 
    print('in year 19{} survived: {}, dead: {}, percentage_survived: {}%, percentage_dead: {}%'          .format(year, x, y, round(per_x, 1), round(per_y, 1)))


# In[666]:


# Analysing more on yearly basis using histogram, PDF and CDF
survivors = csdf.loc[csdf['survival_status'] == 'survived']
counts, bin_edges = np.histogram(survivors['year_of_operation'], bins = 15, density = True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)


# In[668]:


dead = csdf.loc[csdf['survival_status'] == 'dead']
counts, bin_edges = np.histogram(dead['year_of_operation'], bins = 20, density = True)
pdfd = counts/sum(counts)
cdfd = np.cumsum(pdfd)
plt.plot(bin_edges[1:], pdfd)
plt.plot(bin_edges[1:], cdfd)


# ### Violen and Box Plots 

# In[677]:


sns.boxplot(x='survival_status', y='year_of_operation', data=csdf)
sns.violinplot(x = 'survival_status', y= 'year_of_operation', data=csdf)
# changing grid
#plt.yticks(np.linspace(1, 100, 100))
plt.yticks(csdf['year_of_operation'])
min = csdf['year_of_operation'].min();
max = csdf['year_of_operation'].max();
# increasing size of plot for better understanding
plt.rcParams['figure.figsize'] = (20,20)
# increasing font size of labels
plt.tick_params(axis='both', which='major', labelsize=20)# no need to see full scale limiting plot in range of  and 
plt.ylim([min-1, max+1])
plt.show()


# In[679]:


#Yearly observation says:
#1. 1961 survivals to the dead ratio is greatest 
#2. 1965 has worst survival to dead ratio
#3. According to PDF the trend of survials is decreasing every third alternate year
#4. percentile of survivals is almost linear and stablises every 3rd year 
#5. death rate has increase significantly between 1963 and 1965
#6. 25th percentile of survivors till 1960, 50 percentile of survivors till 1963, 75 percentile of survivors till 1966
#   This shows survival rate is nearly same in 2nd quartile and 3rd quartile 
#7. 25th percentile of dead till 1959, 50 percentile of dead till 1963, 75 percentile of dead till 1965
#   This shows that the death rate is very high in 3rd quartile than 2rd quartile


# In[681]:


# Survival percentage on age basis:
import numpy as np
aop = set(csdf['age_of_patient'])
for age in aop:
    csdf_temp = csdf.loc[csdf['age_of_patient'] == age]
    counts = csdf_temp['survival_status'].value_counts()
    if counts.shape[0] == 1:
        first_index = (counts.index[0])
        if first_index == 'survived':
            x = csdf_temp['survival_status'].value_counts()[0]
            y = 0
        elif first_index == 'dead':
            y = csdf_temp['survival_status'].value_counts()[0]
            x = 0
    else:
        x, y = csdf_temp['survival_status'].value_counts()
    per_x, per_y = x*100/(x+y), y*100/(x+y) 
    print('patients of age {} survived: {}, dead: {}, percentage_survived: {}%, percentage_dead: {}%'          .format(age, x, y, round(per_x, 1), round(per_y, 1)))


# In[683]:


sns.boxplot(x = 'survival_status', y = 'age_of_patient', data = csdf)
sns.violinplot(x = 'survival_status', y = 'age_of_patient', data = csdf)

#changing grid
#plt.yticks(np.linspace(1,100,50))
plt.yticks(csdf['age_of_patient'])
# limiting y axis upto our needs
min = csdf['age_of_patient'].min();
max = csdf['age_of_patient'].max();
# increasing size of plot for better understanding
plt.rcParams['figure.figsize'] = (20,20)
# increasing font size of labels
plt.tick_params(axis='both', which='major', labelsize=20)
plt.ylim(min-1, max+1)
plt.show()
#sns.violinplot(x = 'survival_status', y = 'age_of_patient', data = csdf, size = 5)


# In[685]:


# Age wise Observation as per available dataset:
#1. Patients on or above age 78 has nearly 0% chances of survival 
#2. Patients below age 33 has 100% chances of survival
#3. 25th percentile of survivors are under age 43, 50th percentile of survivors are below age 52,
#   75th percentile of survivors are under age 60
#4. 25% of dead are under age 46, 50% of dead are under 53, 75% of dead are under 61
#   25% death rate between age 53-61 leaves this age group with very less survivors


# In[687]:


# Survival analysis on feature positive_axilary_nodes:
plt.close()
sns.violinplot(x='survival_status', y='positive_axilary_nodes', data=csdf)
sns.boxplot(x='survival_status', y='positive_axilary_nodes', data=csdf)
min = csdf['positive_axilary_nodes'].min()
max = csdf['positive_axilary_nodes'].max()
plt.yticks(csdf['positive_axilary_nodes'])
plt.ylim(min-1, max+1)
plt.rcParams['figure.figsize'] = (20,20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()


# In[689]:


#Observation
#1. 50th percentile of survivors have 0 positive nodes, 75th percentie of survivors have less than 3 positive axilary nodes
#2. 25th percentile of dead have 1 positive axilary node, 50th percentile of dead have positive axilary nodes below 4,
#   75th percentile of dead have positive nodes below 11


# In[ ]:




References:
http://people.duke.edu/~ccc14/cfar-data-2016/Customizing_Plots_Solutions.html
https://seaborn.pydata.org/generated/seaborn.pairplot.html
https://www.kaggle.com/gilsousa/habermans-survival-data-set
https://www.youtube.com/watch?v=hJI0wZV7VnA   