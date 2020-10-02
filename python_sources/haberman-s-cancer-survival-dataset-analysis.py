#!/usr/bin/env python
# coding: utf-8

# # Haberman's Cancer Survival DataSet Analysis

# ## 0. Data Description

# * The Dataset contains cases from a study.
# * Conducted between 1958 and 1970 at the University of Chicago's Billings Hospital.
# * On the survival of patients who had undergone surgery for breast cancer.

# ### Attribute Information:
# 
# 1. Age of patient at the time of operation (numerical)
# 2. Patient's year of operation (from 1958 to 1970, numerical)
# 3. Number of positive auxilliary nodes detected (numerical)
# 4. Survival status(class attribute)
#     * 1 = the patient survived 5 years or longer
#     * 2 = the patient died within 5 years

# ## Objective
# 
# Our objective is to predict whether the patient will survive after treatment or not based upon the patient's age,
#     year of treatment and the number of lymph nodes detected during dygnosis.
#     
# Or simply how effective the treatement on different age group having differnet number of positive lymph nodes detected.

# ## 1. Environment Configuration  

# In[ ]:


# Importing Libraries

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

'''Download haberman.csv from https://www.kaggle.com/
            gilsousa/habermans-survival-data-set'''

# Load haberman.csv into a pandas dataframe.

cancer_df = pd.read_csv("../input/haberman.csv", header=None, names = ['Age', 'Year_of_Treatment', 'Positive_Lymph_nodes', 'Surv_status'])
print(cancer_df.head())


# ## 2. Data Preparation

# In[ ]:


print(cancer_df.info())


# ### Observations:
# 
# * There are no missing values in this dataset. So there is no need to do data imputation.
# * The datatype of 'Surv_status' column is integer. We can convert it into catogorical type for convenience.
# * We can convert the values of 'Surv_status' column into categorical type as 'yes' (survived after treatement) and 'no' (not survived).
# 
#     Check the unique values of the target column. Modify the       target column values to be meaningful as well as         catogorical.

# In[ ]:


print(list(cancer_df['Surv_status'].unique()))


# In[ ]:


cancer_df['Surv_status'] = cancer_df['Surv_status'].map({1:"yes", 2:"no"})

cancer_df['Surv_status'] = cancer_df['Surv_status'].astype('category', copy=False)

print(cancer_df.head())
#cancer_df


# In[ ]:


print(cancer_df.info())


# ## 3. High Level Statistics
# 
# Describe the centratility and the dispersion are used to understand the essence of the features.

# In[ ]:


print(cancer_df.describe())


# In[ ]:


# Data points and features.
cancer_df.shape # Output as (row, column)


# In[ ]:


# column-names in dataset.
print(cancer_df.columns)


# In[ ]:


# Data-points for each attribute-class.
# (or) Target Variable Distribution.

cancer_df['Surv_status'].value_counts()

#Output indicates dataset contains two target-classes
# 1. person who is survived after the surgery.
# 2. person who is not survived.


# ### Observations:
#     * The age of the patients vary from 30 to 83 with the median of 52.
#     * Although the maximum number of positive lymph nodes observed is 52, but nearly 75% of the patients have less than 5 
#       positive lymph nodes and nearly 25% of the patients have no positve lymph nodes.
#     * The dataset contains small number of records (306).
#     * The dataset is imbalanced with 73% of values are 'yes'.

# ## 4. Uni-variate Analysis
# 
# The purpose is to describe, summarize and find patterns in the single features of the dataset.

# ### 4.1 Distribution plots
# 
# Distribution plots are used to visually assess how the data points are distributed with respect to its frequency.
# 
# * Usually the data points are grouped into bins and the height of the bars representing each group increases with increase in the number of data points lie within that group. (histogram)
# * Probability Density Function (PDF) is the probability that the variable takes a value x. (smoothed version of the histogram)
# * Kernel Density Estimate (KDE) is the way to estimate the PDF. The area under the KDE curve is 1.
# * Here the height of the bar denotes the percentage of the data points under the corresponding group.

# In[ ]:


# 1-D scatter-plot of age
sns.set_style('whitegrid')
survived = cancer_df.loc[cancer_df['Surv_status'] == 'yes'];
dead = cancer_df.loc[cancer_df['Surv_status'] == 'no'];

plt.plot(survived['Age'],
        np.zeros_like(survived['Age']), 'o')

plt.plot(dead['Age'],
        np.zeros_like(dead['Age']), 'o')

plt.show()


# In[ ]:


# 1-D scatter-plot of Positive_Lymph_nodes
survived = cancer_df.loc[cancer_df['Surv_status'] == 'yes'];
dead = cancer_df.loc[cancer_df['Surv_status'] == 'no'];

plt.plot(survived['Positive_Lymph_nodes'],
        np.zeros_like(survived['Positive_Lymph_nodes']), 'o')

plt.plot(dead['Positive_Lymph_nodes'],
        np.zeros_like(dead['Positive_Lymph_nodes']), 'o')

plt.show()


# In[ ]:


# Histogram for Age column
sns.FacetGrid(cancer_df, hue='Surv_status', height=4)     .map(sns.distplot, 'Age')     .add_legend();
plt.show();


# * From the above figure, there is much overlapping between the patients who survived and the patients who couldn't based on the patient's age.
# * So, we can't determine the survival of patient based on his age.

# In[ ]:


# Histogram for 'Year_of_Treatment' column
sns.FacetGrid(cancer_df, hue='Surv_status', height=4)     .map(sns.distplot, 'Year_of_Treatment')     .add_legend();
plt.show();


# * From the above figure also, there is much overlapping between the patients who survived and the patients who couldn't based on the patient's year of treatment.
# * So, we can't determine the survival of patient based on his year of treatment.

# In[ ]:


# Histogram for 'Positive_Lymph_nodes' column
sns.FacetGrid(cancer_df, hue='Surv_status', height=4)     .map(sns.distplot, 'Positive_Lymph_nodes')     .add_legend();
plt.show();


# * Now from this figure, if no of positive_lymph_nodes <= 3 then there is more chances that the patient would survive.
# * Here the probability distribution function (PDF) for the survived patient is high as compared to the patient who died.

# ### 4.2 Cumulative Distribution Function (CDF)
# 
# The cumulative distribution function (cdf) is the probability that the variable takes a value less than or equal to x.

# In[ ]:


#CDF and PDF plot of  Age column for survived patients.

counts, bin_edges = np.histogram(survived['Age'], bins=10,
                                density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);

cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf);
plt.plot(bin_edges[1:], cdf)
plt.legend(['survived_PDF', 'survived_CDF'])
plt.xlabel('Age')
plt.ylabel('Probability')
plt.title('CDF plot of Age for the Survived Patients')

''''counts, bin_edges = np.histogram(survived['Age'], bins=20,
                                density = True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:], pdf)'''

plt.show()


# In[ ]:


#CDF and PDF plot of  Age column for survived patients.

counts, bin_edges = np.histogram(dead['Age'], bins=10,
                                density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);

cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf);
plt.plot(bin_edges[1:], cdf)
plt.legend(['dead_PDF', 'dead_CDF'])
plt.xlabel('Age')
plt.ylabel('Probability')
plt.title('CDF plot of Age for the Dead Patients')

''''counts, bin_edges = np.histogram(survived['Age'], bins=20,
                                density = True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:], pdf)'''

plt.show()


# In[ ]:


#CDF and PDF plot of  Positive_Lymph_nodes column for survived patients.

counts, bin_edges = np.histogram(survived['Positive_Lymph_nodes'], bins=10,
                                density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);

cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf);
plt.plot(bin_edges[1:], cdf)
plt.legend(['survived_PDF', 'survived_CDF'])
plt.xlabel('Positive_Lymph_nodes')
plt.ylabel('Probability')
plt.title('CDF plot of Positive_Lymph_nodes for the Survived Patients')

''''counts, bin_edges = np.histogram(survived['Positive_Lymph_nodes'], bins=20,
                                density = True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:], pdf)'''

plt.show()


# * 100% of the survived patient had less than 40 positive_lymph_nodes.
# * and 90% had less than 10 positive_lymph_nodes.

# In[ ]:


#CDF and PDF plot of  Positive_Lymph_nodes column for dead patients.

counts, bin_edges = np.histogram(dead['Positive_Lymph_nodes'], bins=10,
                                density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);

cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf);
plt.plot(bin_edges[1:], cdf)
plt.legend(['dead_PDF', 'dead_CDF'])
plt.xlabel('Positive_Lymph_nodes')
plt.ylabel('Probability')
plt.title('CDF plot of Positive_Lymph_nodes for the dead Patients')

''''counts, bin_edges = np.histogram(dead['Positive_Lymph_nodes'], bins=20,
                                density = True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:], pdf)'''

plt.show()


# * 70% of the dead patients had less than 10 positive_lymph_nodes.

# In[ ]:


#CDF and PDF plot of  Positive_Lymph_nodes column for survived patients.

counts, bin_edges = np.histogram(survived['Positive_Lymph_nodes'], bins=10,
                                density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);

cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf);
plt.plot(bin_edges[1:], cdf)


#CDF and PDF plot of  Positive_Lymph_nodes column for dead patients.

counts, bin_edges = np.histogram(dead['Positive_Lymph_nodes'], bins=10,
                                density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);

cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf);
plt.plot(bin_edges[1:], cdf)
plt.legend(['survived_PDF', 'survived_CDF', 'dead_PDF', 'dead_CDF'])
plt.xlabel('Positive_Lymph_nodes')
plt.ylabel('Probability')
plt.title('CDF plot of Positive_Lymph_nodes for the both classes')

''''counts, bin_edges = np.histogram(dead['Positive_Lymph_nodes'], bins=20,
                                density = True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:], pdf)'''

plt.show()


# ### Observations:
# 
# 1. PDF of both classes first intersect at 8, if we take this point then with 40% of probablity we can say survival rates are high for the patients having less than 8 positive_lymph_nodes.
# 
# 2. Hence positive_lymph_nodes is the most import feature to predict the survival status after 5 years.
# 
# 3. The survival rates is extremely high for patients having less than 3 positive_lymph_nodes.

# ### 4.3 Mean, Variance and Std-dev

# In[ ]:


print("Means:")
print(np.mean(survived['Positive_Lymph_nodes']))
print(np.mean(dead['Positive_Lymph_nodes']))

print("\nStd-Dev:")
print(np.std(survived['Positive_Lymph_nodes']))
print(np.std(dead['Positive_Lymph_nodes']))


# ### 4.4 Box Plots
# 
# Box plot takes a less space and visually represents the five number summary of the data points in a box.
# 
# The outliers are displayed as points outside the box.
# 1. Q1 -- 1.5*IQR
# 2. Q1 (25th percentile)
# 3. Q2 (50th percentile or median)
# 4. Q3 (75th percentile)
# 5. Q3 + 1.5*IQR
# 
# Inter Quartile Range = Q3-Q1

# In[ ]:


sns.boxplot(x='Surv_status', y='Positive_Lymph_nodes', data=cancer_df).set_title('Box Plot of Positive_Lymph_nodes and Survival Status ')  
plt.show()


# * 75% of survived Patients had positive_lymph_nodes less than 2.
# * 25% of dead patients had positive_lymph_nodes less than 1, while 50% of dead patient had positive_lymph_nodes less than 3.

# In[ ]:


sns.boxplot(x='Surv_status', y='Year_of_Treatment', data=cancer_df).set_title('Box Plot of Year_of_Treatment and Survival Status ')  
plt.show()


# ### 4.4 Voilin Plots
# 
# Voilin plot is the combination of box plot and probability density function.

# In[ ]:


sns.violinplot(x='Surv_status', y='Positive_Lymph_nodes', data=cancer_df, size=8)
plt.show()


# In[ ]:


sns.violinplot(x='Surv_status', y='Year_of_Treatment', data=cancer_df, size=8)
plt.show()


# ## 5. Multivariate Analysis
# The analysis of the relationship between multiple variables and its effect on the target class is known as multi variate analysis.
# 
# Pair plot is commonly used to visualize the relationship between Quantitative columns. As all the independent columns in our data set are quantitative, use the pair plot.
# 
# Pair plot in seaborn plots the scatter plot between every two data columns in a given data frame.it is used to visualize the relationship between two variables.

# ### 5.1 2-D scatter plot

# In[ ]:


sns.set_style('whitegrid')
sns.FacetGrid(cancer_df, hue='Surv_status', height=5)     .map(plt.scatter, 'Age', 'Year_of_Treatment')     .add_legend()
plt.show()


# ### Observations:
# By scattering the data points between <i>Year_of_Treatment</i> and <i>Age</i>, we can see the much overlapping between the two classes

# In[ ]:


sns.FacetGrid(cancer_df, hue='Surv_status', height=5)     .map(plt.scatter, 'Age', 'Positive_Lymph_nodes')     .add_legend()
plt.show()


# In[ ]:


sns.FacetGrid(cancer_df, hue='Surv_status', height=5)     .map(plt.scatter, 'Year_of_Treatment', 'Positive_Lymph_nodes')     .add_legend()
plt.show()


# ### Observations:
# 
# By scattering the data points between Year_of_Treatment and Positive_lymph_nodes, we can see the better seperation between the two classes than the others.

# ### 5.2 Pair Plots

# In[ ]:


sns.set_style('whitegrid')
sns.pairplot(cancer_df, hue='Surv_status', height=3);
plt.show()


# ### 5.3 Contour Plot

# In[ ]:


sns.jointplot(x='Year_of_Treatment', y='Positive_Lymph_nodes', data=cancer_df, kind='kde');
plt.show()


# ## Conclusion
# 
# * By plotting all the necessary tools (pdf, cdf, box-plot, pair-plot etc.)
# * we conclude that if number of positive_lymph_nodes is less, than there is more probability of the survival of patients.
# * On the basis of ages of the patients we can say that if the age of the patient is less than 40, then his survival rate is probabily high.
# * for further detailed analysis more feature columns of data set is required.

# In[ ]:




