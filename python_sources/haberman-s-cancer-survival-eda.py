#!/usr/bin/env python
# coding: utf-8

# # Haberman's Survival Dataset

# 1. Sources: (a) Donor: Tjen-Sien Lim (limt@stat.wisc.edu) (b) Date: March 4, 1999
# 2. The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings      Hospital on the survival of patients who had undergone surgery for breast cancer.
# 3. Number of Datapoints / Observation - 306
# 4. Numbre of Features / Independent Variable - 3 (Age, Patients Operation Year, Number of Positive auxillary nodes)
# 5. Number of Class label / Dependent Variable - 1 (Survival Status)
# 6. Survival Status - 1 = the patient survived 5 years or longer 2 = the patient died within 5 years

# # Objective
# 
# To Classify the survival status of patients into 2 categories : -
#     a. The patient survived 5 years or longer
#     b. The patient died within 5 years

# In[ ]:


# Importing Required modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Load Habermans survival dataset into pandas dataframe
haberman = pd.read_csv("../input/haberman.csv", names = ['Age','Operation Year','Auxillary Node','Survival Status'])


# In[ ]:


# How many data points and features are there in the Haberman's Dataset?
print(haberman.shape)


# In[ ]:


# What are the column names of our dataset?
print(haberman.columns)


# In[ ]:


# How many classes are present and how many data points per class are present?
haberman["Survival Status"].value_counts()

# Haberman is an imbalanced dataset as number of points are not almost equal


# # Univariate Analysis
# 
# ## Probability Density Function (PDF) and Histogram

# In[ ]:


# Plotting distribution of Age

sns.FacetGrid(haberman, hue = "Survival Status", size = 5)   .map(sns.distplot, "Age")   .add_legend()


# In[ ]:


# Plotting distribution of Patient's Year of Operation

sns.FacetGrid(haberman, hue = "Survival Status", size = 5)   .map(sns.distplot, "Operation Year")   .add_legend()


# In[ ]:


# Plotting distribution of Number of auxillary nodes

sns.FacetGrid(haberman, hue = "Survival Status", size = 5)   .map(sns.distplot, "Auxillary Node")   .add_legend()


# #### Observations
# 
# 1. None of the above variables (Age, Operation Year or Auxillary Node) are useful in distinguishing or classifying the Survival Status of the patients.
# 
# 2. As none of the plots have survival status pdf cruves or histogram well seperated, we cannot classify the survival status of the patients.
# 
# 3. We can just observe that highest number of survivers are in the range 0 to 4(approximately) and number of deaths increases as auxillary node increases.

# ## Cumulative Distribution Function (CDF)

# In[ ]:


haberman_survived = haberman.loc[haberman["Survival Status"] == 1]
haberman_not_survived = haberman.loc[haberman["Survival Status"] == 2]


# In[ ]:


# Plots of CDF of Age for both Categories

# Survived
counts, bin_edges = np.histogram(haberman_survived['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'PDF of Age (Survived)')
plt.plot(bin_edges[1:], cdf, label = 'CDF of Age (Survived)')

# Not_Survived
counts, bin_edges = np.histogram(haberman_not_survived['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'PDF of Age (Not Survived)')
plt.plot(bin_edges[1:], cdf, label = 'CDF of Age (Not Survived)')
plt.title("PDF/CDF of Age for Survival Status")
plt.xlabel("Survival Status")
plt.legend()

plt.show()


# In[ ]:


# Plots of CDF of Operation Year for both Categories

# Survived
counts, bin_edges = np.histogram(haberman_survived['Operation Year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'PDF of Operation Year (Survived)')
plt.plot(bin_edges[1:], cdf, label = 'CDF of Operation Year (Survived)')

# Not_Survived
counts, bin_edges = np.histogram(haberman_not_survived['Operation Year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'PDF of Operation Year (Not Survived)')
plt.plot(bin_edges[1:], cdf, label = 'CDF of Operation Year (Not Survived)')
plt.title("PDF/CDF of Operation Year for Survival Status")
plt.xlabel("Survival Status")
plt.legend()

plt.show()


# In[ ]:


# Plots of CDF of Number of Auxillary Nodes for both Categories

# Survived
counts, bin_edges = np.histogram(haberman_survived['Auxillary Node'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'PDF of Auxillary Node (Survived)')
plt.plot(bin_edges[1:], cdf, label = 'CDF of Auxillary Node (Survived)')

# Not_Survived
counts, bin_edges = np.histogram(haberman_not_survived['Auxillary Node'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = 'PDF of Auxillary Node (Not Survived)')
plt.plot(bin_edges[1:], cdf, label = 'CDF of Auxillary Node (Not Survived)')
plt.title("PDF/CDF of Auxillary Node for Survival Status")
plt.xlabel("Survival Status")
plt.legend()

plt.show()


# #### Observations
# 
# 1. The above CDF plots are also overlapping and hence, not useful for our objective which is to classify survival status of patients.
# 
# 2. From Auxillary plot, we can see that less number of auxillary nodes have better chances of survival. As the auillary node increases, the number of deaths also increases.

# ## Box Plot with Whiskers

# In[ ]:


# Age

sns.boxplot(x='Survival Status',y='Age', data=haberman)
plt.show()


# In[ ]:


# Year of Operation

sns.boxplot(x='Survival Status',y='Operation Year', data=haberman)
plt.show()


# In[ ]:


# Number of Auxillary Nodes

sns.boxplot(x='Survival Status',y='Auxillary Node', data=haberman)
plt.show()


# #### Observation
# 
# 1. From the above box plots also, we cannot clearly classify the Survival Status of Patients.
# 2. Those who did not survive mostly have auxillary nodes more than 3 (approximately).

# ## Violin Plots

# In[ ]:


# Age

sns.violinplot(x='Survival Status',y='Age', data=haberman, size = 8)
plt.show()


# In[ ]:


# Year of Operation

sns.violinplot(x='Survival Status',y='Operation Year', data=haberman, size = 8)
plt.show()


# In[ ]:


# Number of Auxillary Nodes

sns.violinplot(x='Survival Status',y='Auxillary Node', data=haberman, size = 8)
plt.show()


# #### Observation
# 
# 1. From the Violin Plots also, we cannot classify survival status of patients.
# 
# 2. From the auxillary node vs survival status violin plot, we can however observe that the number of deaths are more as auxillary node increases.
# 
# 3. Univariate Analysis is not useful for the clear classification of Survival Status of patients for Haberman's Cancer Survival Dataset.
# 
# 4. However, we could clearly observe that number of auxillary nodes were affecting the survival status. As the number of auxillary nodes were increasing, the deaths were more.
# 
# 5. Also, the patients operated before 1959 have lower chances of survival and patients operated after 1965 have more chances of survival. This can be observed from the density of the boxes in the violin plot.

# # Bi-variate Analysis

# ## Pair Plots

# In[ ]:


# Pairwise Scatter Plot - Pair Plot

plt.close();
sns.set_style("whitegrid");
sns.pairplot(haberman, hue="Survival Status", size=3, vars = ['Age', 'Operation Year', 'Auxillary Node']);
plt.show()


# #### Observations
# 
# 1. We cannot classify Survival Status on the basis of the above pair plots as none of them are linearly seperable.
# 
# 2. However, the best pair plot for classification seems to be Operation Year vs Auxillary Node.
