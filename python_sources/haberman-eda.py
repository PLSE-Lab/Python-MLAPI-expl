#!/usr/bin/env python
# coding: utf-8

# This notebook can be viewed at [this link](https://www.kaggle.com/shashank49/haberman-eda)

# # Haberman's Survival Data
# **About data: The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.**
# 
# **Attribute information:<br>
# 1.Age of patient at time of operation (numerical)<br>
# 2.Patient's year of operation (year - 1900, numerical)<br>
# 3.Number of positive axillary nodes detected (numerical)<br>
# 4.Survival status (class attribute) 1 = the patient survived 5 years or longer 2 = the patient died within 5 year**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import os
print(os.listdir("../input"))


# In[ ]:


warnings.filterwarnings("ignore")


# In[ ]:


#importing data
data = pd.read_csv("../input/haberman.csv")


# In[ ]:


column = ['Patient_age', 'Year_of_operation', 'positive_axillary_nodes', 'Survival_status']
data = pd.read_csv("../input/haberman.csv", names = column)
data.head()


# In[ ]:


data.shape


# In[ ]:


print("Data points per class:") 
data['Survival_status'].value_counts()


# The total number of rows are 305 and 4 columns

# In[ ]:


#Survival status
# 1 = patient survived 5 years or longer
# 2 = patient died within 5 years
data['Survival_status'].value_counts()


# ## Summary:
# **a.Data has 305 rows and 4 columns.<br>
# b. Coumns are age of patient, operation year, axiliary nodes detected and survival status of the patient<br>
# c. Survival status has two classes. 1 = patient survived 5 years or longer, 2 =  patient died within 5 years <br>
# d. Data is skewed as survival status of patients column has imbalanced numbers of patient.<br>
# **

# In[ ]:


data.describe()


# **Observation:<br>
# 1.Patients' age vary from 30 to 83<br>
# 2.Axil nodes detected vary from 0 to 52 with average of 4<br>
# 3.Although the maximum number of positive lymph nodes observed is 52, nearly 75% of the patients have less than 5 positive lymph nodes and nearly 25% of the patients have no positive lymph nodes**
# 

# In[ ]:


data_survived = data.loc[data['Survival_status'] == 1]
data_died = data.loc[data['Survival_status'] == 2]


# In[ ]:


print(data_survived.max())
print()
print(data_survived.min())


# In[ ]:


data_survived.describe()


# In[ ]:


print(data_died.max())
print()
print(data_died.min())
data_survived.describe()


# **Auxiliary node comes out to be most important feature.<br>
# 75% of survived patients had positive axillary nodes less than or equal to 3 and 75% of patients who died had positive axillary nodes less than or equal to 11.<br>
# However, there are cases where patient with low or no auxialiary nodes has not survived, therefore it cannot be taken as the sole feature to rely upon**

# # Univariate Analysis

# ## 1. Patient Age

# In[ ]:


#graphical representation
sns.set_style('whitegrid')
sns.FacetGrid(data=data, hue='Survival_status', height=5)    .map(sns.distplot, 'Patient_age')    .add_legend()
plt.title("Patient_age histogram")
plt.show()
print("Histogram with PDF")


# ### CDF and PDF plots

# In[ ]:


counts, bin_edges = np.histogram(data_survived['Patient_age'], bins=20, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)


counts, bin_edges = np.histogram(data_died['Patient_age'], bins=20, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:] ,cdf)
plt.legend(['sur_pdf', 'sur_cdf','died_pdf', 'died_cdf'])
plt.title("pdf and cdf plots")
plt.show();


# ### Box plot

# In[ ]:


sns.boxplot(x='Survival_status',y='Patient_age', data= data)
plt.title("Patient_age Box plot")
plt.show()


# ## Observation <br>
# 1.Patient age has similar effect for either classes<br>
# 2.Patient age can not be sole factor to determine survival status of the patient

# ## 2.Year of operation

# **Histogram with pdf**

# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(data , hue = "Survival_status" , size = 6).map(sns.distplot , "Year_of_operation").add_legend();
plt.title("Year_of_operation histogram")
plt.show()


# ### PDF and CDF plot

# In[ ]:


counts, bin_edges = np.histogram(data_survived['Year_of_operation'], bins= 30, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)


counts, bin_edges = np.histogram(data_died['Year_of_operation'], bins=30, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:] ,cdf)
plt.legend(['sur_pdf', 'sur_cdf','died_pdf', 'died_cdf'])
plt.title("Year_of_operation pdf and cdf plots")
plt.show();


# ### Box plot

# In[ ]:


sns.boxplot(x='Survival_status',y='Year_of_operation', data= data)
plt.title("Year_of_operation box plot")
plt.show()


# # Observation <br>
# **As the data is overlapping, we cannot take year of opeartion alone as the factor for patient survival in our dataset**

# ## 3.Positive axillary nodes

# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(data , hue = "Survival_status" , size = 8).map(sns.distplot , "positive_axillary_nodes").add_legend();
plt.title("Positive axilliary node histogram")
plt.show()


# ### PDF and CDF

# In[ ]:


counts, bin_edges = np.histogram(data_survived['positive_axillary_nodes'], bins=20, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)


counts, bin_edges = np.histogram(data_died['positive_axillary_nodes'], bins=20, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:] ,cdf)
plt.legend(['sur_pdf', 'sur_cdf','died_pdf', 'died_cdf'])
plt.title("Positive axilliary node pdf and cdf plots")
plt.show()


# ### Box plot

# In[ ]:


sns.boxplot(x='Survival_status',y='positive_axillary_nodes', data= data)
plt.title("Positive axilliary node box plot")
plt.show()


# ## Observation <br>
# **As the data is still overlapping, it is still difficult to demarcate a value which can clearly separate surviving and non surviving patients.<br>
# As the number of nodes increases, there are lesser chances thatpatient might survive<br>
# 50% of patients who survived had no axil nodes<br>**

# # Pair plots

# In[ ]:


sns.set_style('whitegrid')
sns.pairplot(data,hue='Survival_status',vars=['Patient_age','Year_of_operation','positive_axillary_nodes'])
plt.title("All pair plots")
plt.plot()


# ** Axillary node doesn't depend on patients age<br>**

# In[ ]:


sns.set_style('whitegrid');
sns.FacetGrid(data, hue = 'Survival_status', size = 6)   .map(plt.scatter, 'Patient_age', 'positive_axillary_nodes')   .add_legend();
plt.title("Patient age vs Positive axilliary node")
plt.show();


# **More survivors have zero to less than 10 axil nodes**

# In[ ]:


sns.set_style('whitegrid');
sns.FacetGrid(data, hue = 'Survival_status', size = 6)   .map(plt.scatter, 'Patient_age', 'Year_of_operation')   .add_legend();
plt.title("Patient age vs Year of Operation plot")
plt.show();


# **No significant threshold can be confirmed from the plot**

# In[ ]:


sns.set_style('whitegrid');
sns.FacetGrid(data, hue = 'Survival_status', size = 6)   .map(plt.scatter, 'Year_of_operation', 'positive_axillary_nodes')   .add_legend();
plt.title("Year of operation vs Positive axilliary nodes")
plt.show();


# **No significant threshold can be confirmed from the plot as the data is overlapping**

# # Final Conclusions

# 1. Patients with 0 axil nodes have more chances of survival<br>
# 2. Patients with age > 50 and axil nodes > 10 are more likely to die<br>
# 3. Most of the patients operated in '60 - '65 did not survive<br>
# 4. Number of deaths are more between age 40-65<br>
# 5. Patient's age and Year of operation alone are not deciding factors for his/her survival.<br>
# 6. It will be difficult to classify a new patient's survival status based on present data
# 
# **Axil nodes detected and age are much better factors to predict survival of patients than year of operation**
