#!/usr/bin/env python
# coding: utf-8

# **Dataset Description:**
# The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# 
# **Number of Instances:** 306
# 
# **Number of Attributes:** 4 (including the class attribute)
# 
# **Attribute Information**
# 1. Age of patient at time of operation (numerical)
# 2. Patient's year of operation (year - 1900, numerical)
# 3. Number of positive axillary nodes detected (numerical)
# 4. Survival status (class attribute) 1 = the patient survived 5 years or longer 2 = the patient died within 5 year

# **Objective:**
# To predict whether a patient will survive or not after 5 years based on patient's - Age, Year of Operation, and Number of Positive Axiliary Nodes detected.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# **Load and Prepare Dataset:**

# In[ ]:


# Load dataset and add headers to the columns as described in the dataset.
haberman_df = pd.read_csv('../input/haberman.csv', names=['Age', 'Operation Year', 'Positive Axiliary Nodes', 'Survival Status After 5 Years'])
haberman_df.head()


# In[ ]:


# Map Numeric value of 'Survival Status After 5 Years' to values - "Survived", "Died"
survival_status_dict = {1:"Survived",2:"Died"}
haberman_df['Survival Status After 5 Years'] = haberman_df['Survival Status After 5 Years'].map(survival_status_dict)
haberman_df.head()


# **Statistics of Dataset:**

# In[ ]:


haberman_df.info()


# In[ ]:


haberman_df.describe()


# In[ ]:


haberman_df.shape


# In[ ]:


haberman_df['Survival Status After 5 Years'].value_counts()


# **Observations:**
# 
# 1. There are 306 entries found. There are 4 columns, each having 306 not null values.
# 2. 3 columns are input data-points, and 1 data-point ('Survival Status After 5 Years') as output.
# 3. 25% patients have NO positive axiliary nodes, and 75% patients have 4 or less positive axiliary nodes.
# 4. "Survival Status After 5 Years" column takes only 2 values (categorical) - "Survived", "Died"
# 5. It's an imbalanced dataset because there is significant gap between number of data-points for each category.

# 

# **1. Univariate Data Analyis (PDF, CDF, Box, Violin Plots)**
# 
# **Histogram, PDF**

# In[ ]:


haberman_survived_df = haberman_df.loc[haberman_df['Survival Status After 5 Years'] == 'Survived']
haberman_died_df = haberman_df.loc[haberman_df['Survival Status After 5 Years'] == 'Died']

plt.plot(haberman_survived_df['Age'], np.zeros_like(haberman_survived_df['Age']), 'o')
plt.plot(haberman_died_df['Age'], np.zeros_like(haberman_died_df['Age']), 'o')
plt.show()


# In[ ]:


# Distribution Plot
# 1.1
sns.FacetGrid(haberman_df, hue="Survival Status After 5 Years", size=5)     .map(sns.distplot, "Age")     .add_legend();
plt.show();


# In[ ]:


plt.close()
plt.plot(haberman_survived_df['Operation Year'], np.zeros_like(haberman_survived_df['Operation Year']), 'o')
plt.plot(haberman_died_df['Operation Year'], np.zeros_like(haberman_died_df['Operation Year']), 'o')
plt.show()


# In[ ]:


# 1.2
sns.FacetGrid(haberman_df, hue="Survival Status After 5 Years", size=5)     .map(sns.distplot, "Operation Year")     .add_legend();
plt.show();


# In[ ]:


plt.close()
plt.plot(haberman_survived_df['Positive Axiliary Nodes'], np.zeros_like(haberman_survived_df['Positive Axiliary Nodes']), 'o')
plt.plot(haberman_died_df['Positive Axiliary Nodes'], np.zeros_like(haberman_died_df['Positive Axiliary Nodes']), 'o')
plt.show()


# In[ ]:


# 1.3
sns.FacetGrid(haberman_df, hue="Survival Status After 5 Years", size=5)     .map(sns.distplot, "Positive Axiliary Nodes")     .add_legend();
plt.show();


# **Observations:**
# 1. Features are overlapping with each other for each category.
# 2. Number of survivors are densed for "Positive Axiliary Nodes" with values between 0 to 4. (# 1.3)

# **CDF**

# In[ ]:


#1.4
# PDF, CDF as per Patient's Age

# Survived Patients
counts, bin_edges = np.histogram(haberman_survived_df['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
# Compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

# Died Patients
counts, bin_edges = np.histogram(haberman_died_df['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
# Compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

plt.title("PDF and CDF for Age of Patients")
plt.xlabel("Age")
plt.ylabel("% of Patient's")

label = ["PDF (Survived)", "CDF (Survived)", "PDF (Died)", "CDF (Died)"]
plt.legend(label)

plt.show();


# In[ ]:


# 1.5
# PDF, CDF as per Patient's Age

# Survived Patients
counts, bin_edges = np.histogram(haberman_survived_df['Operation Year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
# Compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

# Died Patients
counts, bin_edges = np.histogram(haberman_died_df['Operation Year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
# Compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

plt.title("PDF and CDF for Operation Years of Patients")
plt.xlabel("Operation Year")
plt.ylabel("% of Patient's")

label = ["PDF (Survived)", "CDF (Survived)", "PDF (Died)", "CDF (Died)"]
plt.legend(label)

plt.show();


# In[ ]:


# 1.6
# PDF, CDF as per Patient's Age

# Survived Patients
counts, bin_edges = np.histogram(haberman_survived_df['Positive Axiliary Nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
# Compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

# Died Patients
counts, bin_edges = np.histogram(haberman_died_df['Positive Axiliary Nodes'], bins=10,
                                 density = True)
pdf = counts/(sum(counts))
# Compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

plt.title("PDF and CDF for Positive Axiliary Nodes of Patients")
plt.xlabel("Positive Axiliary Nodes")
plt.ylabel("% of Patient's")

label = ["PDF (Survived)", "CDF (Survived)", "PDF (Died)", "CDF (Died)"]
plt.legend(label)

plt.show();


# **Observations:**
# 1. Patient's with "Positive Axiliary Nodes" > 46 have NOT Survived. (# 1.6)
# 2. Among the "Survived patient's", 82% have "Positive Axiliary Nodes" < 4 (# 1.6)
# 3. Among the "Died patient's", 58% have "Positive Axiliary Nodes" < 5 (# 1.6)

# **Box Plot**

# In[ ]:


# 1.7
sns.boxplot(x='Survival Status After 5 Years',y='Age', data=haberman_df)
plt.show()


# In[ ]:


# 1.8
sns.boxplot(x='Survival Status After 5 Years',y='Operation Year', data=haberman_df)
plt.show()


# In[ ]:


# 1.9
sns.boxplot(x='Survival Status After 5 Years',y='Positive Axiliary Nodes', data=haberman_df)
plt.show()


# **Observations:**
# 1. Chances or Survival was higher for Patient's operated after 1966. (# 1.8)
# 2. Chances or Death was higher for Patient's operated before 1959. (# 1.8)

# **Violin Plot**

# In[ ]:


# 1.10
sns.violinplot(x='Survival Status After 5 Years',y='Age', data=haberman_df, size=8)
plt.show()


# In[ ]:


# 1.11
sns.violinplot(x='Survival Status After 5 Years',y='Operation Year', data=haberman_df, size=8)
plt.show()


# In[ ]:


# 1.12
sns.violinplot(x='Survival Status After 5 Years',y='Positive Axiliary Nodes', data=haberman_df, size=8)
plt.show()


# **2. Bi-variate Analysis (Scatter, Pair Plots)**
# 
# **2D Scatter Plot**

# In[ ]:


# 2.1
sns.set_style('whitegrid');
sns.FacetGrid(haberman_df, hue='Survival Status After 5 Years', size=5)    .map(plt.scatter, 'Age', 'Operation Year')    .add_legend();
plt.show();


# In[ ]:


# 2.2
sns.set_style('whitegrid');
sns.FacetGrid(haberman_df, hue='Survival Status After 5 Years', size=5)    .map(plt.scatter, 'Age', 'Positive Axiliary Nodes')    .add_legend();
plt.show();


# In[ ]:


# 2.3
sns.set_style('whitegrid');
sns.FacetGrid(haberman_df, hue='Survival Status After 5 Years', size=5)    .map(plt.scatter, 'Operation Year', 'Positive Axiliary Nodes')    .add_legend();
plt.show();


# **Pair Plot**

# In[ ]:


# 2.4
plt.close();
sns.set_style("whitegrid");
sns.pairplot(haberman_df, hue="Survival Status After 5 Years", size=5);
plt.show()


# **Observations:**
# 1. Features are not linearly separable to build any model and conclude the outcome.

# **Contour Plot**

# In[ ]:


# 2.5
sns.jointplot(x="Age", y="Positive Axiliary Nodes", data=haberman_survived_df, kind="kde");
plt.show();


# In[ ]:


# 2.6
sns.jointplot(x="Operation Year", y="Positive Axiliary Nodes", data=haberman_survived_df, kind="kde");
plt.show();


# In[ ]:


# 2.7
sns.jointplot(x="Age", y="Operation Year", data=haberman_survived_df, kind="kde");
plt.show();


# **Observations:**
# 1. Most of the patient's who have gone through the operation are Aged between 45-55 and Operation Year between 1962-63 (# 2.7)

# In[ ]:




