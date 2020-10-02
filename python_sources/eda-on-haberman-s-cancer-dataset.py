#!/usr/bin/env python
# coding: utf-8

# > # EDA on Haberman's Cancer Dataset

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#Load haberman.csv into a pandas dataFrame. This dataset is on the survival of patients who had undergone surgery for breast cancer.
#Breast Cancer = bc
bc_df = pd.read_csv("../input/haberman.csv")


# ## Data Information : Getting a sense of the data provided

# In[ ]:


bc_df.shape   #306 rows, 4 columns


# In[ ]:


bc_df.columns 


# In[ ]:


bc_df.head()  #the information stored in data


# ##### This dataset is on the survival of patients who had undergone surgery for breast cancer. 
# ###### It consists of 306 entries.
# ##### Age column represents age of patient at time of operation (numerical)
# ##### Year column represents patient's year of operation (year - 1900, numerical)
# ##### The nodes column represents the number of positive axillary nodes detected (numerical).
# ##### The status column indicates the survival status of patients after surgery, where  1 = the patient survived 5 years or longer 2 = the patient died within 5 year

# In[ ]:


bc_df.info()  # Checking the number of entries and their data type 


# ##### All values are of int type and there is no null value in the data 

# ## High Level Statistics 

# In[ ]:


bc_df.describe()  


# ###### The count of all samples is 306
# ###### A high standard deviation in columns 'age' and 'nodes' indicates a high variation in their data
# ###### Age of patients lie in the range of 30-83 and the year of surgery ranges from 1958-1969
# ###### The median and mean for 'nodes' column vary significantly.  Also, 75% of 'nodes' data has a value of 4 but the maximum value is 52, indicating this column has outliers. 

# In[ ]:


print('Survival Status : ')
x = ((bc_df['status'].value_counts(normalize = True))*100).tolist();
print('Patients surviving for 5 years or longer :'+str(round(x[0],2))+'%');
print('Patients died within 5 years :'+str(round(x[1],2))+'%');


# ###### The status of patients is divided into 2 categories as mentioned above. The ones who survived for 5 years or longer are 73.53% and the ones who died within 5 years are 26.47% ; indicating that the data is imbalanced 

# ## Objective : To perform exploratory data analysis on Haberman's survival dataset in order to check for the pattern of data, any anomalies and to understand relation between features using statistical and graphical representations.

# ## Univariate Analysis 

# ### Distribution plot for age, year, nodes , differentiating on the basis of status of patient after cancer surgery

# #### Distribution plots give us an idea about the distribution of data with respect to a particular feature. 
# ##### The datapoints are subdivided into small bins and height of each bin gives the frequency of data in that bin. These plots tell us about the pdf distribution of data

# In[ ]:


sns.FacetGrid(bc_df, hue="status", height=5).map(sns.distplot, "nodes").add_legend();
plt.suptitle('Distribution plot for nodes', size=15);
plt.show();  #due to outliers

sns.FacetGrid(bc_df, hue="status", height=5).map(sns.distplot, "age").add_legend();
plt.suptitle('Distribution plot for age', size=15);
plt.show();

sns.FacetGrid(bc_df, hue="status", height=5).map(sns.distplot, "year").add_legend();
plt.suptitle('Distribution plot for year', size=15);
plt.show();


# ### PDF and CDF Plots

# In[ ]:


#Segregating data on the basis of survival status of patient
bc_status_1 = bc_df.loc[bc_df["status"] == 1]
bc_status_2 = bc_df.loc[bc_df["status"] == 2]


# In[ ]:


#Creating 3 plots in one cell
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))

#Calculating pdf/cdf for nodes 
counts, bin_edges1 = np.histogram(bc_status_1['nodes'], bins=10, density = True)
pdf1 = counts/(sum(counts))
cdf1 = np.cumsum(pdf1)
ax1.plot(bin_edges1[1:],pdf1, label="pdf_status_1");
ax1.plot(bin_edges1[1:], cdf1, label="cdf_status_1");
ax1.legend()

counts, bin_edges2 = np.histogram(bc_status_2['nodes'], bins=10, density = True)
pdf2 = counts/(sum(counts))
cdf2 = np.cumsum(pdf2)
ax1.plot(bin_edges2[1:],pdf2, label="pdf_status_2");
ax1.plot(bin_edges2[1:], cdf2, label="cdf_status_2")
ax1.legend()

ax1.set(xlabel="pdf/cdf for nodes");
ax1.set_title("PDF/CDF for nodes")

print("*******Feature : Node*******");
print("Patients surviving for 5 years or longer :\nBin Edges : {0}\nPDF : {1}\nCDF : {2}".format(bin_edges1[1:],pdf1,cdf1))
print("\nPatients surviving for not more than 5 years :\nBin Edges : {0}\nPDF : {1}\nCDF : {2}".format(bin_edges2[1:],pdf2,cdf2))


###############################################

#Calculating pdf/cdf for age 
counts, bin_edges1 = np.histogram(bc_status_1['age'], bins=10, density = True)
pdf1 = counts/(sum(counts))
cdf1 = np.cumsum(pdf1)
ax2.plot(bin_edges1[1:],pdf1, label="pdf_status_1");
ax2.plot(bin_edges1[1:], cdf1, label="cdf_status_1");
ax2.legend()

counts, bin_edges2 = np.histogram(bc_status_2['age'], bins=10, density = True)
pdf2 = counts/(sum(counts))
cdf2 = np.cumsum(pdf2)
ax2.plot(bin_edges2[1:],pdf2, label="pdf_status_2");
ax2.plot(bin_edges2[1:], cdf2, label="cdf_status_2")
ax2.legend()

ax2.set(xlabel="pdf/cdf for age");
ax2.set_title("PDF/CDF for age")

print("\n*******Feature : Age*******");
print("Patients surviving for 5 years or longer :\nBin Edges : {0}\nPDF : {1}\nCDF : {2}".format(bin_edges1[1:],pdf1,cdf1))
print("\n\nPatients surviving for not more than 5 years :\nBin Edges : {0}\nPDF : {1}\nCDF : {2}".format(bin_edges2[1:],pdf2,cdf2))


###############################################

#Calculating pdf/cdf for year 
counts, bin_edges1 = np.histogram(bc_status_1['year'], bins=10, density = True)
pdf1 = counts/(sum(counts))
cdf1 = np.cumsum(pdf1)
ax3.plot(bin_edges1[1:],pdf1, label="pdf_status_1");
ax3.plot(bin_edges1[1:], cdf1, label="cdf_status_1");
ax3.legend()

counts, bin_edges2 = np.histogram(bc_status_2['year'], bins=10, density = True)
pdf2 = counts/(sum(counts))
cdf2 = np.cumsum(pdf2)
ax3.plot(bin_edges2[1:],pdf2, label="pdf_status_2");
ax3.plot(bin_edges2[1:], cdf2, label="cdf_status_2")
ax3.legend()

ax3.set(xlabel="pdf/cdf for Year");
ax3.set_title("PDF/CDF for Year")

print("\n*******Feature : Year*******");
print("Patients surviving for 5 years or longer :\nBin Edges : {0}\nPDF : {1}\nCDF : {2}".format(bin_edges1[1:],pdf1,cdf1))
print("\n\nPatients surviving for not more than 5 years :\nBin Edges : {0}\nPDF : {1}\nCDF : {2}".format(bin_edges2[1:],pdf2,cdf2))


# ###### PDF indicates the likelihood of a particular point/bin and CDF indicates the percentage of population under a certain pdf
# ###### For nodes : In status 1 - PDF decreses and reaches 0 for nodes higher than 15 indicating that the likelihhod of having higher nodes is less. The CDF increases, indicating almost entire population(except a few) have nodes < 30. In status 2 - a similar trend is followed. However the frequency varies.
# ###### For age : More patients ranging from 48-53 years, died under 5 years of surgery in comparison to those who survived.
# ###### For Year : 1965-66 experienced maximum cases which lead to death of patients

# ### Box Plots

# #### These plots give us an idea of data distribution where the middle line represents median(50%) of data.
# #### IQR = Q3-Q1

# In[ ]:


i=0;
fig, ax = plt.subplots(1, 3, figsize=(15,5)) #Creating 3 plots in one cell

for feature in bc_df:
    sns.boxplot(x='status',y=feature, data=bc_df, ax=ax[i])
    i= i+1;
    if i == 3:
        break;
ax[0].set_title('Age vs Survival status')
ax[1].set_title('Year vs Survival status')
ax[2].set_title('Nodes vs Survival status')

plt.show()


# ###### People with survival status 2 have higher age than the ones with survival status 1.
# ###### The median of nodes with survival status 1 is 0, indicating 50% of the people which survived didn't have positive nodes and the maximum is below 10 with outliers. Maximum outliers can be seen in case of nodes.

# ### Violin Plots 

# #### These density plots are a combination of box plot and pdf.  A fatter area indicates higher pdf 

# In[ ]:


i=0;
fig, ax = plt.subplots(1, 3, figsize=(15,5)) #Creating 3 plots in one cell

for feature in bc_df:
    sns.violinplot(x='status',y=feature, data=bc_df, ax=ax[i])
    i= i+1;
    if i == 3:
        break;
ax[0].set_title('Age vs Survival status')
ax[1].set_title('Year vs Survival status')
ax[2].set_title('Nodes vs Survival status')

plt.show()


# ## Bivariate Analysis

# In[ ]:


# pairwise scatter plot: Pair-Plot
#This plot establishes relation between 2 features(all possible combinations) in a data frame

sns.set_style("whitegrid");
sns.pairplot(bc_df, hue="status", vars=['year','age','nodes'], height=4) ;
plt.show();
# The diagnol elements are PDFs for each feature. 


# ###### None of the pair of features can be used to classify patients as status 1/2. However, if only the nodes are considered patients with lower nodes have higher probability of belonging to status 1 and patients with higher nodes have higher probability of belonging to status 2 

# In[ ]:


## MultiVariate
#2D Density plot, contors-plot


sns.jointplot(x="status", y="nodes", data=bc_df, kind="kde");
plt.show();


# ###### This plot indicates that the probability of lower nodes is higher in patients under status 1 

# ## Conclusion

# ###### An exploratory analysis of data was performed to check the data structure for entries, relation between features and anomalies, if present.
# ###### None of the entries of data are empty.
# ###### The data can be divided categorically as patients under status 1 i.e. those who survived for 5 years and longer after surgery and patients under status 2 i.e. those who died within 5 years.
# ###### The data present in "nodes" feature is highly variable. However, a rough estimate can be made that people with lower nodes fall under "status 1" category.
# ###### In order to establish a strong correlation, more features are required
