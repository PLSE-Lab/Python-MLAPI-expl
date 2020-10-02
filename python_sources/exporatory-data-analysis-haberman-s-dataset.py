#!/usr/bin/env python
# coding: utf-8

# # Exporatory Data Analysis : Haberman's Dataset

# By<br>
# ***Kranthi Kumar Valaboju***

# Haberman's Dataset contains the data of the Breast Cancer Patients who had undergone Surgery.

# In[ ]:


# Importing Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# Loading the Haberman Data into a pandas DataFrame and adding the column names
# 'Age_of_Patient','Operation_Year','Axil_Nodes','Survival_Status'

haberman_data=pd.read_csv("../input/haberman.csv",header=None, names=['Age_of_Patient','Operation_Year','Axil_Nodes','Survival_Status'])


# In[ ]:


# Printing the number of data points

print(haberman_data.shape)


# #### Observation : 

# This dataset contains 306 data points (rows) with 3 variables and 1 Cross Label

# In[ ]:


# Printing the column names in Haberman Dataset

print(haberman_data.columns)


# #### Observation :

# The 4 Columns including Cross Label are
# 
# - Age_of_Patient : Age of the Patient
# - Operation_Year
# - Axil_Nodes : Number of Axil Nodes that are detected
# - Survival_Status : Survival Status of the patient after 5 years

# In[ ]:


# Replacing the Survival_Status Values

haberman_data['Survival_Status']=haberman_data['Survival_Status'].replace([1,2],['yes','no'])
print("Survival Status Count of Patients:")
print(haberman_data['Survival_Status'].value_counts())


# #### Observation : 
# 
#  - Haberman's Dataset is an Imbalanced dataset as the number of datapoints for each class are different.
#  - 73.5 % of the patients have survived after the surgery.

# In[ ]:


haberman_data.describe()


# #### Observation : 
# 
# - The Age of the patients is in the range of 30 to 83.
# - The data collected is from the period 58 - 69.
# - The minimum number of Axil Nodes is 0 and maximum is 52 with a mean of 4 axil nodes and 75% patients have less than 5 Axil Nodes.

# ## Univariate Analysis :
# 
# ### Histogram

# In[ ]:


# Plotting the Histogram for Age of Patient directly and also  based on Cross Label i.e. Survival Status

plt.hist(haberman_data["Age_of_Patient"])
plt.xlabel("Age of Patient")

sb.FacetGrid(haberman_data,hue='Survival_Status',height=5.5).map(sb.distplot,"Age_of_Patient").add_legend()
plt.show()


# #### Observation :
# 
# - There are more number of patients aged between 45 to 55 years.
# - Exact inferences can't be drawn from the above histogram's. But, Patients aged between 30-40 had more survival chances. And this is not the, only parameter that has to be considered. The combination of Age and Number of Axil Nodes can infer better.

# In[ ]:


# Plotting the Histogram for Operation Year directly and also  based on Cross Label i.e. Survival Status

plt.hist(haberman_data["Operation_Year"])
plt.xlabel("Year  of Operation")

sb.FacetGrid(haberman_data,hue="Survival_Status",height=5.5).map(sb.distplot,"Operation_Year").add_legend()
plt.show()


# #### Observation :
# 
# - Exact inferences can' be obtained as the Histogram's are overlapped. 
# - More Number of surgeries were done in period 58-59.
# - Success rate was high during the period 60-61. Even the number of Axil Nodes detected and the age of the patient also has influence on survival rate.

# In[ ]:


# Plotting the Histogram for Axil_Nodes directly and also  based on Cross Label i.e. Survival Status

plt.hist(haberman_data["Axil_Nodes"])
plt.xlabel("Axil Nodes")

sb.FacetGrid(haberman_data,hue="Survival_Status",height=5.5).map(sb.distplot,"Axil_Nodes")
plt.show()


# #### Observation : 
# 
# - Most of the patients do have less than 5 Positively Detected Axil Nodes.
# - Patients having less than 5 Axil Nodes have the highest Survival Rate. 

# ### PDF, CDF

# In[ ]:


# Segregating the data based on the Class Label i.e. Survival Status

haberman_data_Survived=haberman_data.loc[haberman_data["Survival_Status"]=="yes"]
haberman_data_Not_Survived=haberman_data.loc[haberman_data["Survival_Status"]=="no"]


# In[ ]:


# Plotting the PDF,CDF for Age of Patient for the above Segregated Data

density_age_survived,bin_edges_age_survived=np.histogram(haberman_data_Survived['Age_of_Patient'],bins=10,density=True)
pdf_age_survived=(density_age_survived)/(sum(density_age_survived))


density_Age_Not_Survived,bin_Edges_Age_Not_Survived=np.histogram(haberman_data_Not_Survived['Age_of_Patient'],bins=10,density=True)
pdf_Age_Not_Survived=(density_Age_Not_Survived)/(sum(density_Age_Not_Survived))

print("Bin Edges Survived : {}".format(bin_edges_age_survived))
print("PDF Survived : {}".format(pdf_age_survived))
print("Bin Edges Not Survived :{}".format(bin_Edges_Age_Not_Survived))
print("PDF Not Survived : {}".format(pdf_Age_Not_Survived))


cdf_Age_Not_Survived=np.cumsum(pdf_Age_Not_Survived)
cdf_age_survived=np.cumsum(pdf_age_survived)

plt.plot(bin_edges_age_survived[1:],pdf_age_survived)
plt.plot(bin_edges_age_survived[1:],cdf_age_survived)
plt.plot(bin_Edges_Age_Not_Survived[1:],pdf_Age_Not_Survived)
plt.plot(bin_Edges_Age_Not_Survived[1:],cdf_Age_Not_Survived)
plt.xlabel('Age of Patient')
plt.legend(['Survived_PDF','Survived_CDF','Not Survived PDF','Not Survived CDF'])
plt.show()


# #### Observation : 
#  - The Survival chances are high for the patients having age less than 40 years.

# In[ ]:


# Plotting the PDF,CDF for Year of Operation for the above Segregated Data

density_op_year_survived,bin_edges_op_year_survived=np.histogram(haberman_data_Survived['Operation_Year'],bins=10,density=True)
pdf_op_year_survived=(density_op_year_survived)/(sum(density_op_year_survived))

density_Op_Year_Not_Survived,bin_Edges_Op_Year_Not_Survived=np.histogram(haberman_data_Not_Survived['Operation_Year'],bins=10,density=True)
pdf_Op_Year_Not_Survived=(density_Op_Year_Not_Survived)/(sum(density_Op_Year_Not_Survived))

print("Bin Edges Survived : {}".format(bin_edges_op_year_survived))
print("PDF Survived : {}".format(pdf_op_year_survived))
print("Bin Edges Not Survived :{}".format(bin_Edges_Op_Year_Not_Survived))
print("PDF Not Survived : {}".format(pdf_Op_Year_Not_Survived))

cdf_op_year_survived=np.cumsum(pdf_op_year_survived)
cdf_Op_Year_Not_Survived=np.cumsum(pdf_Op_Year_Not_Survived)


plt.plot(bin_edges_op_year_survived[1:],pdf_op_year_survived)
plt.plot(bin_edges_op_year_survived[1:],cdf_op_year_survived)
plt.plot(bin_Edges_Op_Year_Not_Survived[1:],pdf_Op_Year_Not_Survived)
plt.plot(bin_Edges_Op_Year_Not_Survived[1:],cdf_Op_Year_Not_Survived)
plt.xlabel('Operation Year')
plt.legend(['Survived_PDF','Survived_CDF','Not Survived PDF','Not Survived CDF'])
plt.show()


# #### Observation : 
#  - As the plots are overlapped, exact inferences can't be drawn out. And this parameter alone can't be used for obtaining the inference. 

# In[ ]:


# Plotting the PDF,CDF for Axil Nodes for the above Segregated Data

density_axil_nodes_survived,bin_edges_axil_nodes_survived=np.histogram(haberman_data_Survived['Axil_Nodes'],bins=10,density=True)
pdf_axil_nodes_survived=(density_axil_nodes_survived)/(sum(density_axil_nodes_survived))

density_Axil_Nodes_Not_Survived,bin_Edges_Axil_Nodes_Not_Survived=np.histogram(haberman_data_Not_Survived['Axil_Nodes'],bins=10,density=True)
pdf_Axil_Nodes_Not_Survived=(density_Axil_Nodes_Not_Survived)/(sum(density_Axil_Nodes_Not_Survived))

print("Bin Edges Survived : {}".format(bin_edges_axil_nodes_survived))
print("PDF Survived : {}".format(pdf_axil_nodes_survived))
print("Bin Edges Not Survived :{}".format(bin_Edges_Axil_Nodes_Not_Survived))
print("PDF Not Survived : {}".format(pdf_Axil_Nodes_Not_Survived))



cdf_axil_nodes_survived=np.cumsum(pdf_axil_nodes_survived)
cdf_Axil_Nodes_Not_Survived=np.cumsum(pdf_Axil_Nodes_Not_Survived)


plt.plot(bin_edges_axil_nodes_survived[1:],pdf_axil_nodes_survived)
plt.plot(bin_edges_axil_nodes_survived[1:],cdf_axil_nodes_survived)
plt.plot(bin_Edges_Axil_Nodes_Not_Survived[1:],pdf_Axil_Nodes_Not_Survived)
plt.plot(bin_Edges_Axil_Nodes_Not_Survived[1:],cdf_Axil_Nodes_Not_Survived)
plt.xlabel('Axil Nodes')
plt.legend(['Survived_PDF','Survived_CDF','Not Survived PDF','Not Survived CDF'])
plt.show()


# #### Observation : 
#  - The Survival Rate for the patients having less than 5 auxilary nodes is high i.e. ~82%.
#  - If the number of axil nodes are less, then the survival chances are high.

# ### Box Plots

# In[ ]:


sb.boxplot(x="Survival_Status",y="Operation_Year",data=haberman_data)
plt.show()


# In[ ]:


sb.boxplot(x="Survival_Status",y="Age_of_Patient",data=haberman_data)
plt.show()


# #### Observation :
# - This parameter alone can't be used for obtaining the inference but from Box Plot we can infer that,
# - Patients aged between 30 and 35 survived.
# - Patients whose age is less than 45 have more chances of survival.

# In[ ]:


sb.boxplot(x="Survival_Status",y="Axil_Nodes",data=haberman_data)
plt.show()


# #### Observation :
#  - The Lesser the number of axil nodes, the more the chance of survival.
#  - More than 60% of the patients survived are having less than 5 axil nodes.

# ### Violin Plots 

# In[ ]:


sb.violinplot(x="Survival_Status",y="Age_of_Patient",data=haberman_data)
plt.show()


# In[ ]:


sb.violinplot(x="Survival_Status",y="Axil_Nodes",data=haberman_data)
plt.show()


# In[ ]:


sb.violinplot(x="Survival_Status",y="Operation_Year",data=haberman_data)
plt.show()


# #### Observation :
# - Violin plots are the combination of Box Plots and Distribution Functions. So, the same inferences can be obtained.

# ### Bivariate  Analysis

# In[ ]:


sb.pairplot(haberman_data,hue="Survival_Status",height=4)
plt.show()


# In[ ]:


sb.set_style('whitegrid')
sb.FacetGrid(haberman_data,hue='Survival_Status',height=5).map(plt.scatter,'Axil_Nodes','Age_of_Patient')
plt.show()


# #### Observation :
#  - All the plots are overlapped.
#  - From the plot between Age_of_Patient and Axil_Nodes, We can infer that there are more number of patients with number of axil nodes less than 5 as the scattered plot is most densely concentrated in that region.

# In[ ]:



sb.jointplot(x="Axil_Nodes",y="Age_of_Patient",data=haberman_data_Survived,kind="kde",color='g')
sb.jointplot(x="Axil_Nodes",y="Age_of_Patient",data=haberman_data_Not_Survived,kind="kde",color='r')
plt.show()


# #### Observation :
#  - From the above plots we can conclude that, the lesser the number of axil nodes, the chances of survival are high.
#  - Survived patients have less number of axil nodes i.e. less than 5.
#  - So, the number of Axil Nodes is the important parameter in determining the Survival Status of the Patient.

#   

# ### Final Conclusions : 
# 
#  - Haberman's Dataset is Imbalanced Dataset and 73.5% of the patients have survived after the surgery.
#  - Age of patients is in the range of 30 to 83.
#  - Operation's are performed during the period 1958-1969.
#  - The minimum number of Axil Nodes is 0 and maximum is 52 with a mean of 4 axil nodes and 75% patients have less than 5 Axil Nodes.
#  - Patients aged between 40 and 55 are more in number.
#  - Patients with age less than 40 years have more chances of survival.
#  - Major concentration of the patients are having less than 5 Positively detected Axil Nodes.
#  - Patients having less than 5 Positively Detected Axil Nodes have the highest survival rate (~82%).
#  - Exact inferences can't be drawn from the above plots.As this is Imbalanced dataset and all the parameters are overlapped, by considering only one parameter we can't infer correctly. The combination of parameters under conditions can infer better.
#  - Number of Positively Detected Axil Nodes and Age of Patient are the two important parameters, which when combinedly used under conditions can infer better in building a model.
