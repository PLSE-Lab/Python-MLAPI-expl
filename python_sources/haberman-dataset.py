#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
haberman = pd.read_csv("../input/habermans-survival-data-set/haberman.csv", header=None, names= ['Patient_Age', 'Operation_year', '# of tumors', 'Survival_status'])
haberman.shape


# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


haberman.head()


# In[ ]:


haberman.columns


# **There are four different columns, i.e. 'Patient_Age', 'Operation_year', '# of tumors', 'Survival_status'. Let's understand what can we infered from them.**
# 
# 1. First column, i.e. 'Patient_Age', corresponds to Age of the patient at the time of operation.
# 2. Second column, i.e. 'Operation_year', corresponds to the age of patient's year of operation.
# 3. Third column, i.e. '# of tumors', corresponds to the number of tumors found in the body of the patient.
# 4. Fourth column, i.e. 'Survival_status', corresponds to the survival of the patient.
# 
# **If status is equal to 1 then it means that the patient survive 5 years or more after operation. Whereas status is equal to 2 then it means that the patient died within 5 years of the operation.**

# In[ ]:


haberman.describe()


# **Observations:**
# 
# * There are total 306 counts for each columns, i.e. 'age', 'year' and 'nodes'.
# * The average number of tumors tested in the body of the patient are 4 whereas minimum number of tumors are 0 and maximum are 52.
# * The year of the operation lies in between 1958 and 1969.
# * The patient age lies in the interval of 30 and 83 with mean of 52.
# * From the above table we can conclude that the column 'status' is skewed in favour of survivors i.e 1.

# In[ ]:


haberman.info()


# Observations:
# * There is no missing value in this dataset. There is no need to handle the missing values.
# * The values in the status column are 1 and 2 which refers to patient survive after the operation and do not survive after the operation respectively. The values of the status are not meaningful. Hence they are mapped to 'Not_died'(survive after 5 years of operation) or 'died'(died within 5 years of operation).

# In[ ]:


haberman['Survival_status'] = haberman['Survival_status'].map({1: 'Not_died', 2: 'died'})
haberman['Survival_status'] = haberman['Survival_status'].astype('category')
haberman.head()


# In[ ]:


haberman.info()


# **Observations:**
# * The status feature is changed to a catorgorical feature to compute the result of the dataset easily.

# In[ ]:


print('Target variable distribution\n', haberman.Survival_status.value_counts())


# **Observations:**
# * There are 225 patients who survived 5 or more than 5 years after the operation. This is a good point.

# In[ ]:


haberman.iloc[:, -1].value_counts(normalize=True)


# In[ ]:


percent_yes = haberman.loc[haberman.Survival_status == 'Not_died', 'Survival_status'].count()/haberman.Survival_status.count()
print("% of survival patient in haberman's survival dataset: ", percent_yes)


# **Observations:**
# 
# * The percentage of the patient who survived 5 or more than 5 years after operation is 73.5%.

# # 2D scatter plot
# * Now, the variables of the dataset are being analysed to check their effects on the survival of the patients after the operation. There are different types of plot are available in the matplotlib library and seaborn.

# In[ ]:


haberman.plot(kind='scatter', y='Patient_Age', x='# of tumors')
plt.title("Age of patients vs # of tumors")
plt.show()


# In[ ]:


sns.set(style="whitegrid")
sns.FacetGrid(haberman, hue='Survival_status', height=6)  .map(plt.scatter, 'Patient_Age', 'Operation_year')  .add_legend()
plt.title('Year of operation vs Age of patients using seaborn')
plt.show()


# In[ ]:


sns.set(style="whitegrid")
sns.FacetGrid(haberman, hue='Survival_status', height=6)  .map(plt.scatter, '# of tumors', 'Operation_year')  .add_legend()
plt.title('Year of operation vs # of tumors using seaborn')
plt.show()


# # Pair Plots

# In[ ]:


plt.close()
sns.set_style('white')
sns.pairplot(haberman, hue = 'Survival_status', height = 3)
plt.show()


# **Observations:**
# * The pair plot is drawn to see the relation between all variable to make any classification decision.From the above plot, it can be observed that the data points pertaining to both survivers and non-survivers (irespective of combination of the plot) are mixed together, therefore it is not possible to classify with regular observation and requires further analysis using univariable. Therefore, We can not find "lines" and "if-else" conditions to build a simple model to classify the patients who survive or not.

# # Histogram, PDF & CDF

# In[ ]:


age_patient= haberman['Patient_Age']
year_operation = haberman['Operation_year']
plt.plot(age_patient, np.zeros_like(age_patient), 'o')
plt.plot(year_operation, np.zeros_like(year_operation), 'o')
plt.legend(labels=['patient age', 'year of operation'])
plt.show()


# In[ ]:


# histogram
sns.set_style('white')
sns.FacetGrid(haberman, hue='Survival_status', height=5).map(sns.distplot, 'Patient_Age').add_legend()
plt.title("Age of patients distribution plot")
plt.show()


# **Observations:**
# * There is a lot of overlapping which signifies that the age of patient is not a good variable for classifying the dataset.

# In[ ]:


sns.FacetGrid(haberman, hue='Survival_status', height=5).map(sns.distplot, 'Operation_year').add_legend()
plt.title("Year of operation distribution plot")
plt.show()


# In[ ]:


sns.FacetGrid(haberman, hue='Survival_status', height=5).map(sns.distplot, '# of tumors').add_legend()
plt.title("Number of tumors distribution plot")
plt.show()


# **Observations:**
# * There is overlapping but one conclusion can be drawn from the above that the patients who has very less number of tumors have more chance of survival. But it is not enough to make the decision for classifying the dataset.

# In[ ]:


counts, bins = np.histogram(age_patient, bins=5, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bins)
sns.set_style('whitegrid')
cdf = np.cumsum(pdf)
plt.plot(bins[1:], pdf)
plt.plot(bins[1:], cdf)
plt.legend(labels=['pdf', 'cdf'])
plt.title('Age of patient')
plt.show()


# In[ ]:


counts, bins = np.histogram(year_operation, bins=5, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bins)
sns.set_style('whitegrid')
cdf = np.cumsum(pdf)
plt.plot(bins[1:], pdf)
plt.plot(bins[1:], cdf)
plt.legend(labels=['pdf', 'cdf'])
plt.title('Year of operation')
plt.show()


# In[ ]:


counts, bins = np.histogram(age_patient, bins=5, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bins)
sns.set_style('whitegrid')
cdf = np.cumsum(pdf)
plt.plot(bins[1:], pdf)
plt.plot(bins[1:], cdf)



counts, bins = np.histogram(year_operation, bins=5, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bins)
sns.set_style('whitegrid')
cdf = np.cumsum(pdf)
plt.plot(bins[1:], pdf)
plt.plot(bins[1:], cdf)
plt.show()


# # Box plots

# In[ ]:


sns.boxplot(x='Survival_status', y='Patient_Age', data=haberman)
plt.title('Box plot for Age of patients')
plt.show()


# **Observations:**
# * In Not_died class: where age of patient is equal to 60, it found out that 75% of the patients (i.e. the 75th percentile) are of age less than 60 who survived for more than 5 years. The 50th percentile is around 52 or 53 and 25th percentile is around the value 43. Most of the datapoints lie within 30 and around 78 (i.e. within the whiskers).
# 
# * In Died class: The 75th percentile value is near about 60, 50th percentile is almost the same value as the 50th percentile of class Not_died. 25th percentile is near about 48. Most of the datapoints lie within 83 and 35 (within the whiskers).
# 
# * Not_died box plot has more Inter-quartile Range than died class

# # Violin plots

# In[ ]:


sns.violinplot(x='Survival_status', y='# of tumors', inner=None, data = haberman)
plt.title('Nodes VS Status Violinplot')
plt.show()


# In[ ]:


sns.violinplot(x='Survival_status', y='Operation_year', inner=None, data = haberman)
plt.title('Year VS Status Violinplot')
plt.show()


# **Observation:**
# * Comparing with the box plots, we get the same observation from the violin plots dark part going vertically across the center. PDFs are also similar to the ones we obtained while plotting the histograms. We could have gone to directly plotting violin plots instead of plotting histograms (PDFs) and box plots separately.

# # Conclusion
# **After a lot of analysis, there are some conclusion drawn about the haberman's survival dataset.**
# 
# * Out of all the independent variables, the number of tumors(i.e. Nodes) has most impact in the decision feature for the classification of the dataset about the survival status of the patients after the operation.
# * Those patients who have been diagnosed with very less number of tumors cases (i.e. nodes) are less prone to the death after operation. As the nodes increases, the chance of survival drop drastically.
