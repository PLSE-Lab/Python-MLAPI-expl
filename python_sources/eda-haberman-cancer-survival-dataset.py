#!/usr/bin/env python
# coding: utf-8

# # Haberman's Survival Data : Exploratory Data Analysis
# 
# 

# # 1. Domain Knowledge about Dataset
# Dataset is Haberman's Survival Data
# 
# The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# 
# Number of Instances: 306
# 
# Number of Attributes: 4 (including the class attribute)
# 
# Attribute Information:
# 
# 1st column : Age of patient at time of operation (numerical)
# 
# 2nd column : Patient's year of operation (year - 1900, numerical)
# 
# 3rd column : Number of positive axillary nodes detected (numerical)
# 
# 4th column : Survival status : 2 class labels (1,2)
# 
# (class attribute) 1 = the patient survived 5 years or longer
# 
# (class attribute) 2 = the patient died within 5 year
# 

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category= RuntimeWarning)
def warn(*args, **kwargs):
    pass
warnings.warn = warn


# In[ ]:


haberman_database = pd.read_csv("../input/haberman.csv") #haberman is breast cancer


# In[ ]:


# print(haberman_database.describe)
print(haberman_database.columns)
haberman_database.describe()


# In[ ]:


print(haberman_database.head()) #database columns name are not meaningfull


# In[ ]:


breast_cancer = pd.read_csv("../input/haberman.csv",header=None, names=['age', 'year_of_operation', 'aux_nodes_detected', 'survival_status'])

# print(breast_cancer.head())
# breast_cancer.describe()


# # 2. High Level Statistics on Dataset

# 1. Number of Points in Dataset is : 306
# 2. Number of Features in Dataset is : 4
# 3. Number of Classes : 2 (Class 1, class 2)
# 4. Number of Datapoints per class :
#     
#     #people survived more than 5 years after operation = 225 
#     
#     #people survived less than 5 years after operation = 81
# 

# In[ ]:


# modify the "survival_status"column values to be meaningful as well as categorical
breast_cancer['survival_status'] = breast_cancer['survival_status'].map({1:"survived", 2:"not_survived"})
breast_cancer['survival_status'] = breast_cancer['survival_status'].astype('category')
print(breast_cancer.head())


# In[ ]:


print(breast_cancer.shape)
print(breast_cancer.columns)
print(breast_cancer['survival_status'].unique())
print(breast_cancer['survival_status'].value_counts())


# # 3. Objective :
# The objective is to predict wether the patient will survive for more than 5 years or not given patient's Age, year of of operation,and Number of positive axillary nodes detected. This given Problem is classification problem, where we have to classify the data in any one of the two class label.

# # 4. Univariate Analysis :
# 
# # (4.1) Histogram, PDF, CDF

# In[ ]:


import numpy as np
survived_patients = breast_cancer.loc[breast_cancer['survival_status']=='survived']
not_survived_patients = breast_cancer.loc[breast_cancer['survival_status']=='not_survived']

plt.figure(1,figsize=(14,4))


plt.subplot(121)
plt.plot(survived_patients["age"], np.zeros_like(survived_patients['age']), 'o',label='survived')
plt.plot(not_survived_patients["age"], np.zeros_like(not_survived_patients['age']), 'o',label='not-survived')
plt.legend()
plt.xlabel('age')
plt.title('surviavl_status based on age')

plt.subplot(122)
plt.plot(survived_patients["aux_nodes_detected"], np.zeros_like(survived_patients['aux_nodes_detected']), 'o',label='survived')
plt.plot(not_survived_patients["aux_nodes_detected"], np.zeros_like(not_survived_patients['aux_nodes_detected']), 'o',label='not-survived')
plt.legend()
plt.xlabel('aux_nodes_detected')
plt.title('surviavl_status based on auxillary nodes')


plt.show()


# # OBSERVATION Based on above Plots
# 1. We can not say much about the data as there are huge numbers of overlapping. 

# In[ ]:


sns.FacetGrid(breast_cancer, hue="survival_status", size=5).map(sns.distplot, "age").add_legend()
plt.title('Histogram for survival_status based on age')
plt.show()


# In[ ]:


sns.FacetGrid(breast_cancer, hue="survival_status", size=5).map(sns.distplot, "year_of_operation").add_legend()
plt.title('Histogram for survival_status based on year_of_operation')
plt.show()


# In[ ]:


sns.FacetGrid(breast_cancer, hue="survival_status", size=5).map(sns.distplot, "aux_nodes_detected").add_legend()
plt.title('Histogram for survival_status based on auxillary_nodes_detected')
plt.show()


# # OBSERVATIONS Based on Histogram Plots
# 1. Patient aged less than 40 are more likely to survived more than 5 years
# 2. Patient aged between 40-60 might not survived more than 5 years
# 3. Patient who got operated in between 1958-1963 or 1966-1968 are more likely to survive more than 5 years.
# 4. Patient who got operated in between 1963-1966 might not survive more than 5 years.
# 5. Patient having less than 5 auxillary nodes are more likely to survive more than 5 years.
# 6. Patient having more than 5 auxillary nodes might not survive more than 5 years.

# In[ ]:


plt.figure(3,figsize=(20,5))
for idx, feature in enumerate(list(survived_patients.columns)[:-1]):
    plt.subplot(1, 3, idx+1)
    
    print("="*30+"SURVIVED_PATIENT"+"="*30)
    print("********* "+feature+" *********")
    counts, bin_edges = np.histogram(survived_patients[feature], bins=10, density=True)
    print("Bin Edges: {}".format(bin_edges))
    pdf = counts/sum(counts)
    print("PDF: {}".format(pdf))
    cdf = np.cumsum(pdf)
    print("CDF: {}".format(cdf))
    plt.plot(bin_edges[1:], pdf, label = 'pdf_survived')
    plt.plot(bin_edges[1:], cdf, label= 'cdf_survived')
    
    print("="*30+"NOT_SURVIVED_PATIENT"+"="*30)
    counts, bin_edges = np.histogram(not_survived_patients[feature], bins=10, density=True)
    print("Bin Edges: {}".format(bin_edges))
    pdf = counts/sum(counts)
    print("PDF: {}".format(pdf))
    cdf = np.cumsum(pdf)
    print("CDF: {}".format(cdf))
    plt.plot(bin_edges[1:], pdf, label = 'pdf_not_survived')
    plt.plot(bin_edges[1:], cdf, label= 'cdf_not_survived')
    
    plt.title('pdf & cdf for patients based on '+feature)
    plt.legend()
    plt.xlabel(feature)


# # OBSERVATIONS based on PDF/CDF plots 
#     1. Patient aged between 45-55 are less likely to survive more than 5 years.
#     2. Patient aged 67 or more might or might not survive more than 5 years.
#     3. Patient aged less than 45 are more likely to survive more than 5 years.
#     4. Patient who got operated in between 1960-1962 or 1967-1968 are more likely to survive more than 5 years.
#     5. Patient who got operated in between 1965-1967 might not survive more than 5 years.
#     6. Patient having less than 5 auxillary nodes are more likely to survive more than 5 years.
#     7. Patient having more than 5 auxillary nodes might not survive more than 5 years.

# # (4.2) Box Plot

# In[ ]:


sns.boxplot(x='survival_status',y='age', data=breast_cancer)
plt.title('box_plot based on age')
plt.show()


# In[ ]:


sns.boxplot(x='survival_status',y='year_of_operation', data=breast_cancer)
plt.title('box_plot based on year_of_operation')


# In[ ]:


sns.boxplot(x='survival_status',y='aux_nodes_detected', data=breast_cancer)
plt.title('box_plot based on auxillary_nodes_detected')


# # OBSERVATION based on BOX plots 
#     1. More than 75% of the patients have auxillary nodes less than 10.

# # (4.3) Violin Plot

# In[ ]:


sns.violinplot(x="survival_status", y="age", data=breast_cancer, size=8)
plt.title('violin_plot based on age')


# In[ ]:


sns.violinplot(x="survival_status", y="year_of_operation", data=breast_cancer, size=8)
plt.title('violin_plot based on year_of_operation')


# In[ ]:


sns.violinplot(x="survival_status", y="aux_nodes_detected", data=breast_cancer, size=8)
plt.title('violin_plot based on auxillary_nodes_detected')


# # OBSERVATION based on VIOLEN Plots
# 
# 1. Patients having more than 10 auxillary nodes are less probable to live more than five years.

# # 5. Multivariate (Bivariate) Analysis
# # (5.1) 2D Scatter Plot

# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(breast_cancer, hue="survival_status",size=5).map(plt.scatter,"age","aux_nodes_detected").add_legend()
plt.title('2D scatter plot for age vs auxillary_nodes_detected')
plt.show()


# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(breast_cancer, hue="survival_status",size=5).map(plt.scatter,"year_of_operation","aux_nodes_detected").add_legend();
plt.title('2D scatter plot for year_of_operation vs auxillary_nodes_detected')
plt.show()


# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(breast_cancer, hue="survival_status",size=5).map(plt.scatter,"year_of_operation","age").add_legend()
plt.title('2D scatter plot for year_of_operation vs age')
plt.show()


# # (5.2) Pair Plot

# In[ ]:


plt.close()
sns.set_style("whitegrid")
sns.pairplot(breast_cancer, hue="survival_status",vars=['age','year_of_operation','aux_nodes_detected'], size=4)
plt.show()


# # OBSERVATION based on Pair Plots and 2D scatter plots
# 
# 1. Patients having age less than 40 years are more probable to live more than five years. (from year_of_operation vs age graph)

# # (5.3) Probability Density, Contour Plot

# In[ ]:


sns.jointplot(x="age", y="year_of_operation", data=breast_cancer, kind="kde")
plt.title('Contour Plot age vs year_of_operation')
plt.show()


# In[ ]:


sns.jointplot(y="aux_nodes_detected", x="age", data=breast_cancer, kind="kde")
plt.title('Contour Plot age vs auxillary_nodes_detected')
plt.show()


# In[ ]:


sns.jointplot(x="year_of_operation", y="aux_nodes_detected", data=breast_cancer, kind="kde");
plt.title('Contour Plot year_of_operation vs auxillary_nodes_detected')
plt.show();


# # OBSERVATION based on CONTOUR plots
# 
# 1. Patients aged between 40-60 are mostly operated in between 1960-1964.
# 2. Patients with more than 5 auxillary nodes are rare.

# # CONCLUSION based upon all the OBSERVATIONS
# 1. Patients having age less than 40 years are more probable to Survive
# 2. Patients with less number of auxillary nodes detected are more probable to survive
# 2. More than 75% of the patients have auxillary nodes less than 10.
