#!/usr/bin/env python
# coding: utf-8

# # Haberman's Survival Data Set  - EDA (Exploratory Data Analysis)

# ## 1. Data Understanding

# * The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# * Number of instances or data points: 306
# * Number of variables or attributes: 4 (including the class attribute)
# 
# ----------------------------------------------------------------------------------------------
#                         DATA DICTIONARY
# ----------------------------------------------------------------------------------------
# | Attribute | Description |  
# |:---|:---|  
# | patient_age: | Age of patient at time of operation (numerical) |
# | year_of_operation: | Patient's year of operation (year - 1900, numerical) |    
# | positive_axillary_nodes: | Number of positive axillary nodes detected (numerical) |  
# | survival_status: | Survival status (class attribute) where 1 = the patient survived 5 years or longer, and 2 = the patient died within 5 years | 
# 
# [//]:# (Note: :---, :---:, and ---: tokens used to describe justification. [This is a hidden information line])
#  
# #### References:
# * Haberman's Survival Dataset : [https://www.kaggle.com/gilsousa/habermans-survival-data-set]
# * Dataset Information: [https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival]
# * What is meant by axillary node in Haberman dataset? [https://en.wikipedia.org/wiki/Axillary_lymph_nodes]
# * What is meant by positive axillary nodes? [https://www.breastcancer.org/symptoms/diagnosis/lymph_nodes]

# ## 2. Data Preparation
# 

# In[ ]:


# To import all the required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go 
import plotly
import warnings

# TO ignore or hide all the warnings
warnings.filterwarnings('ignore');


# In[ ]:


# As haberman.csv doesn't contain column names, so providing the names for columns in the dataset
col_Names=["patient_age", "year_of_operation", "positive_axillary_nodes", "survival_status"];

# To load the Haberman's Survival dataset into a pandas dataFrame
patient_df = pd.read_csv('../input/haberman.csv', names=col_Names);


# In[ ]:


# High level statistics of the dataset 

# To check the number of datapoints and number of attributes or features available in the dataset
print(patient_df.shape)


# **Observation(s):**
# * Number of datapoints: 306
# * Number of features or attributes: 4

# In[ ]:


# To see column names in the dataset
print(patient_df.columns)

# To see first few data points in the dataset
patient_df.head()


# In[ ]:


# Let's have look at the different column values
patient_df.describe().transpose()


# **Observation(s):**
# * All the column values are of integer types
# * patient_age range between 30 to 83 years
# * The year of operation range between 1958 to 1969
# * The number of positive axillary nodes ranges from 0 to 52
# * There are no missing values in the dataset   

# In[ ]:


# To check distinct survival_status values
patient_df["survival_status"].unique()


# **Observation(s):**
# * There are only two classes of survival status 1 and 2, where the value
# *  1 indicates the patient survived 5 years or longer 
# *  2 indicates the patient died within 5 years 

# In[ ]:


# To check number of classes in survival_status
patient_df["survival_status"].value_counts()


# **Observation(s):**
# * There are 225 points under class 1, and 81 points under class 2
# * i.e. There are 225 patients survived for 5 years or longer and 81 patients died within 5 years of operation.

# In[ ]:


# Let's classify the data based on survival status
patient_survived = patient_df[patient_df["survival_status"]==1];
patient_dead = patient_df[patient_df["survival_status"]==2];


# In[ ]:


print("Patients survived for 5yrs or above:")
patient_survived.head()


# In[ ]:


print("Patients couldn't survived for 5yrs:")
patient_dead.head()


# ## 3. Problem Statement/ Objective: 
# 1. To analyse each feature in Haberman's cancer survival dataset, and find the important features among available features (patient_age, year_of_operation, positive_axillary_nodes) that can be used to predict survival_status of a patient who had undergone surgery for breast cancer.
# 
# 2. To use the important features to decide whether the patient will be able to survide for 5 years or longer after operation.

# ## 4. Exploratory Data Analysis

# ### 4.1. Univaraite Analysis
# 
# ### 4.1.1. Plotting using PDF and CDF values

# In[ ]:


#Probability Density Functions (PDF)
#Cumulative Distribution Function (CDF)

##Let's Plots for PDF and CDF of patient_age for both survival_status.
#For patient_survived
counts, bin_edges = np.histogram(patient_survived["patient_age"], bins=10, density=True) #bin='auto'
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of patient_survived")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of patient_survived") 

#For patient_dead
counts, bin_edges = np.histogram(patient_dead["patient_age"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of patient_dead")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of patient_dead")
plt.legend(loc = "upper left")
plt.xlabel("Patient Age") 

plt.show()

#Ref: https://matplotlib.org/api/legend_api.html


# ** Observation(s)**
# * There is max 40% chances that patients with age <=49 yrs would be able to survive for 5 years.
# * There is 8% to 60% chances that patients with age between 44-56 yrs would not be able to survive for 5 years.

# In[ ]:


##Plots for PDF and CDF of year_of_operation for both survival_status.
#For patient_survived
counts, bin_edges = np.histogram(patient_survived["year_of_operation"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="year_of_operation - PDF of patient_survived")
plt.plot(bin_edges[1:],cdf, label="year_of_operation - CDF of patient_survived") 

#For patient_dead
counts, bin_edges = np.histogram(patient_dead["year_of_operation"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="year_of_operation - PDF of patient_dead")
plt.plot(bin_edges[1:],cdf, label="year_of_operation - CDF of patient_dead")
plt.legend(loc = "upper left")
plt.xlabel("Patient Year of Operation") 

plt.show()


# ** Observation(s)**
# * 36% to 70% of the time patients undergone operation between year 1961-65 were able to survive for 5 years or longer.
# * 71% to 90% of the time patients undergone operation between year 1965-67 were not able to survive for 5 years.

# In[ ]:


##Plots for PDF and CDF of positive_axillary_nodes for both survival_status.
#For patient_survived
counts, bin_edges = np.histogram(patient_survived["positive_axillary_nodes"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="+ve axillary nodes - PDF of patient_survived")
plt.plot(bin_edges[1:],cdf, label="+ve axillary nodes - CDF of patient_survived") 

#For patient_dead
counts, bin_edges = np.histogram(patient_dead["positive_axillary_nodes"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="+ve axillary nodes - PDF of patient_dead")
plt.plot(bin_edges[1:],cdf, label="+ve axillary nodes - CDF of patient_dead")
plt.legend(loc = "center")
plt.xlabel("Patient positive axillary nodes") 

plt.show()


# ** Observation(s)**
# * More than 81% of the time patients with 8 or less positive axillary nodes survived for 5 years or longer.

# In[ ]:


# Let's have look at mean, median, variance, and standard deviation of different features

print("Mean of Age for Patient survived: {}".format(np.mean(patient_survived["patient_age"])))
print("Mean of Age for Patient not survived: {}".format(np.mean(patient_dead["patient_age"])))
print("Median of Age for Patient survived: {}".format(np.median(patient_survived["patient_age"])))
print("Median of Age for Patient not survived: {}".format(np.median(patient_dead["patient_age"])))
print("Variance of Age for Patient survived: {}".format(np.var(patient_survived["patient_age"])))
print("Variance of Age for Patient not survived: {}".format(np.var(patient_dead["patient_age"])))
print("Std-dev of Age for Patient survived: {}".format(np.std(patient_survived["patient_age"])))
print("Std-dev of Age for Patient not survived: {}".format(np.std(patient_dead["patient_age"])))

print("\nMean of Operation Year for Patient survived: {}".format(np.mean(patient_survived["year_of_operation"])))
print("Mean of Operation Year for Patient not survived: {}".format(np.mean(patient_dead["year_of_operation"])))
print("Median of Operation Year for Patient survived: {}".format(np.median(patient_survived["year_of_operation"])))
print("Median of Operation Year for Patient not survived: {}".format(np.median(patient_dead["year_of_operation"])))
print("Variance of Operation Year for Patient survived: {}".format(np.var(patient_survived["year_of_operation"])))
print("Variance of Operation Year for Patient not survived: {}".format(np.var(patient_dead["year_of_operation"])))
print("Std-dev of Operation Year for Patient survived: {}".format(np.std(patient_survived["year_of_operation"])))
print("Std-dev of Operation Year for Patient not survived: {}".format(np.std(patient_dead["year_of_operation"])))

print("\nMean of +ve axillary nodes for Patient survived: {}".format(np.mean(patient_survived["positive_axillary_nodes"])))
print("Mean of +ve axillary nodes for Patient not survived: {}".format(np.mean(patient_dead["positive_axillary_nodes"])))
print("Median of +ve axillary nodes for Patient survived: {}".format(np.median(patient_survived["positive_axillary_nodes"])))
print("Median of +ve axillary nodes for Patient not survived: {}".format(np.median(patient_dead["positive_axillary_nodes"])))
print("Variance of +ve axillary nodes for Patient survived: {}".format(np.var(patient_survived["positive_axillary_nodes"])))
print("Variance of +ve axillary nodes for Patient not survived: {}".format(np.var(patient_dead["positive_axillary_nodes"])))
print("Std-dev of +ve axillary nodes for Patient survived: {}".format(np.std(patient_survived["positive_axillary_nodes"])))
print("Std-dev of +ve axillary nodes for Patient not survived: {}".format(np.std(patient_dead["positive_axillary_nodes"])))


# ** Observation(s)**
# * Most of the Patients with more than 8 positive axillary nodes were not able to survive for 5 years.

# ### 4.1.2. Box plot and Whiskers

# In[ ]:


## Box-plot for patient_age
#Box-plot can be visualized as a PDF on the side-ways.
#Whiskers in the plot below donot correposnd to the min and max values.
ax = sbn.boxplot(x="survival_status", y="patient_age", hue = "survival_status", data=patient_df)  
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ["Yes", "No"], loc = "upper center")
plt.xlabel("Survival Status (1=Yes, 2=No)") 
plt.ylabel("Patient Age") 
plt.show()

# Here Survival status as 1 means Yes i.e. patient are able to survive 5 yrs or longer
# And Survival status as 2 means No i.e. patient are not able to survive for 5 yrs or longer


# ** Observation(s)**
# * Most of the patients with age < 45 yrs are able to survive for 5 yrs or longer, however patients with < 35 yrs age are surely able to survive for 5 yrs or longer.
# * Almost 2% of Patients having age > 60 yrs were not able to survive for 5 yrs, however patients having age > 78 yrs are certainly not able to survive for 5 yrs.

# In[ ]:


## Box-plot for year_of_operation
ax = sbn.boxplot(x="survival_status", y="year_of_operation", hue = "survival_status", data=patient_df)  
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ["Yes", "No"], loc = "upper center")
plt.xlabel("Survival Status (1=Yes, 2=No)") 
plt.ylabel("Year of Operation") 
plt.show()


# ** Observation(s)**
# * Out of 50% of Patients who were able to survive for 5 years or longer: 25% patients operation year 1960-63 and other 25% of patients having operation year 1963-66.
# * Out of 50% of Patients who were not able to survive for 5 years: 25% patients operation year 1959-63 and other 25% of patients having operation year 1963-65.

# In[ ]:


## Box-plot for positive_axillary_nodes
ax = sbn.boxplot(x="survival_status", y="positive_axillary_nodes", hue = "survival_status", data=patient_df)  
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ["Yes", "No"], loc = "upper center")
plt.xlabel("Survival Status (1=Yes, 2=No)") 
plt.ylabel("Positive axillary nodes") 
plt.show()


# ** Observation(s)**
# * 25% of Patients with 3 positive axillary nodes were able to survive for 5 years.
# * More than 25% of Patients having >= 4 positive axillary nodes were not able to survive for 5 years.

# ### 4.1.3. Violin Plots

# In[ ]:


### A violin plot combines the benefits of Box-plot and PDF
#Let's have a look at patient age wise Violin Plot
sbn.violinplot(x="survival_status", y="patient_age", data=patient_df, width=0.9)
plt.xlabel("Patient Survival Status \n[1 = able to survive for 5yrs or longer, 2 = dead before 5 yrs]") 
plt.ylabel("Patient Age") 
plt.show()


# ** Observation(s)**
# * Few patients above 85 age certainly dead before 5 yrs of treatment.

# In[ ]:


#Let's have a look at year of treatment wise Violin Plot
sbn.violinplot(x="survival_status", y="year_of_operation", data=patient_df, width=0.9)
plt.xlabel("Patient Survival Status \n[1 = able to survive for 5yrs or longer, 2 = dead before 5 yrs]") 
plt.ylabel("Year of Operation") 
plt.show()


# ** Observation(s)**
# * More number of patients dead between 1959-64.

# In[ ]:


#Let's have a look at +ve axillary nodes wise Violin Plot
sbn.violinplot(x="survival_status", y="positive_axillary_nodes", data=patient_df, width=0.9)
plt.xlabel("Patient Survival Status \n[1 = able to survive for 5yrs or longer, 2 = dead before 5 yrs]") 
plt.ylabel("Positive axillary nodes") 
plt.show()


# ** Observation(s)**
# * Most of the case Patients with <=8 positive axillary nodes were able to survive for 5 years or longer.
# * Patients with > 50 positive axillary nodes certainly not able to survive for 5 years.

# ### 4.2. Bi-varaite Analysis

# ### 4.2.1.  2-D Scatter Plot

# In[ ]:


#To see 2-D scatter plot without any classification
patient_df.plot(kind='scatter', x='patient_age', y='positive_axillary_nodes');
plt.show()


# ** Observation(s)**
# * We can see here most of the patients whose age lies between 33-76 approx, have positive axillary nodes between 0-22 approx.

# In[ ]:


# Let's check with (patient_age, year_of_operation) combination
patient_df.plot(kind='scatter', x='patient_age', y='year_of_operation');
plt.show() 


# ** Observation(s)**
# * All the points are mixed up, No significant information can be derived

# In[ ]:


# Let's check with (year_of_operation, positive_axillary_nodes) combination
patient_df.plot(kind='scatter', x='year_of_operation', y='positive_axillary_nodes');
plt.show() 


# ** Observation(s)**
# * Due to considerable overlaps, No significant information can be derived

# In[ ]:


# Let's see survival_status wise
sbn.set_style("whitegrid");
sbn.FacetGrid(patient_df, hue="survival_status", size=5)   .map(plt.scatter, "patient_age", "positive_axillary_nodes")   .add_legend();
plt.show(); 


# ** Observation(s)**
# * Points are not lineraly seperable as well as showing many overlaps, so it's hard to derive any significant information.

# In[ ]:


# Let's see survival_status wise
sbn.set_style("whitegrid");
sbn.FacetGrid(patient_df, hue="survival_status", size=5)   .map(plt.scatter, "year_of_operation", "positive_axillary_nodes")   .add_legend();
plt.show();


# ** Observation(s)**
# * Due to considerable overlaps, No significant information can be derived

# ### 4.2.2. 3D Scatter plot

# In[ ]:


# Learn about API authentication here: https://plot.ly/pandas/getting-started
# Find your api_key here: https://plot.ly/settings/api

plotly.tools.set_credentials_file(username='rsnayak', api_key='CCK19Gs5LhdbLKsearIW')

#Below code is working fine outside Kaggle i.e. in my Jupyter Notebook
#but in kaggle error out so can be commented

df = patient_df
data = []
clusters = []
colors = ['rgb(228,26,28)','rgb(55,126,184)','rgb(77,175,74)']

for i in range(len(df['survival_status'].unique())):
    name = df['survival_status'].unique()[i]
    color = colors[i]
    x = df[ df['survival_status'] == name ]['patient_age']
    y = df[ df['survival_status'] == name ]['year_of_operation']
    z = df[ df['survival_status'] == name ]['positive_axillary_nodes']
    
    trace = dict(
        name = name,
        x = x, y = y, z = z,
        type = "scatter3d",    
        mode = 'markers',
        marker = dict( size=3, color=color, line=dict(width=0) ) )
    data.append( trace )

layout = dict(
    width=800,
    height=550,
    autosize=False,
    title='Haberman Survival Dataset',
    scene=dict(
        xaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        yaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        zaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        aspectratio = dict( x=1, y=1, z=0.7 ),
        aspectmode = 'manual'        
    ),
)

fig = dict(data=data, layout=layout)

# IPython notebook
py.iplot(fig, filename='pandas-3d-haberman-dataset', validate=False)

url = py.plot(fig, filename='pandas-3d-haberman-dataset', validate=False)


# ** Observation(s)**
# * Points are not linearly separable, and so No significant information can be derived

# ### 4.2.3. Pair-plot

# In[ ]:


plt.close();
sbn.set_style("whitegrid");
sbn.pairplot(patient_df, hue="survival_status", vars=['patient_age', 'year_of_operation', 'positive_axillary_nodes'], height=4);
plt.show();


# ** Observation(s)**
# * The pair plot for combination (year_of_operation, poitive_axillary_nodes) looks significant with points distribution in comparision to others.
# * In other pair plots the points are overlapped, and not lineraly seperable, so no more significant information can be derived from pair-plot.

# ### 4.2.4. 1-D scatter plot

# In[ ]:


# 1-D scatter plot using just one feature
# #1-D scatter plot of patient_age

plt.plot(patient_survived["patient_age"], np.zeros_like(patient_survived["patient_age"]), 'o');
plt.plot(patient_dead["patient_age"], np.zeros_like(patient_dead["patient_age"]), 'o');
plt.show();


# ** Observation(s)**
# * Very hard to make sense as points are overlapping a lot.

# In[ ]:


# #1-D scatter plot of year_of_operation

plt.plot(patient_survived["year_of_operation"], np.zeros_like(patient_survived["year_of_operation"]), 'o');
plt.plot(patient_dead["year_of_operation"], np.zeros_like(patient_dead["year_of_operation"]), 'o');
plt.show();


# ** Observation(s)**
# * Very hard to make sense as points are overlapping a lot due to same year of operation for survival_status type 1 and 2.

# In[ ]:


# #1-D scatter plot of positive_axillary_nodes

plt.plot(patient_survived["positive_axillary_nodes"], np.zeros_like(patient_survived["positive_axillary_nodes"]), 'o');
plt.plot(patient_dead["positive_axillary_nodes"], np.zeros_like(patient_dead["positive_axillary_nodes"]), 'o');
plt.show();


# ** Observation(s)**
# * Very hard to make sense as points are overlapping a lot.

# ### Histogram (with PDF)

# In[ ]:


#Patient Age wise survival status
sbn.FacetGrid(patient_df, hue="survival_status", height=5)   .map(sbn.distplot, "patient_age")   .add_legend();

plt.show();


# ** Observation(s)**
# * Patients chances to survive for 5yrs or more is greater when their age <= 40 yrs.
# * Patients chances to survive is less when their age lies in intervals 41-51 yrs, 64-70 yrs, or > 78 yrs.

# In[ ]:


#year_of_operation wise survival status
sbn.FacetGrid(patient_df, hue="survival_status", height=5)   .map(sbn.distplot, "year_of_operation")   .add_legend();

plt.show(); 


# ** Observation(s)**
# * Patients chances to survive for 5yrs or more is greater between 1958-62.
# * Patients chances to survive is less in the year 1963-66.

# In[ ]:


#positive_axillary_nodes wise survival status
sbn.FacetGrid(patient_df, hue="survival_status", height=5)   .map(sbn.distplot, "positive_axillary_nodes")   .add_legend();

plt.show();


# ** Observation(s)**
# * Patients chances to survive for 5 yrs or longer is greater if positive lymph nodes count is <=3.
# * Patients chances to survive is less if positive lymph nodes count is >3.

# ### 4.3. Multi-varaite Analysis

# ### 4.3.1. Multivariate probability density, contour plot.

# In[ ]:


## Let's do multivariate analyis
sbn.jointplot(x="patient_age", y="positive_axillary_nodes", data=patient_survived, kind="kde");
plt.show();


# ** Observation(s)**
# * Patients having age 40-66 yrs and 0-5 positive axillary nodes have survived for 5 yrs or longer.

# In[ ]:


sbn.jointplot(x="patient_age", y="positive_axillary_nodes", data=patient_dead, kind="kde");
plt.show();


# ** Observation(s)**
# * No significant conclusion can be derived.

# ## 5. Final Observasion:
# 
# ##### 1.  From the pair plot we could able to understand the patient age and positive lymp nodes are important features to derive the significant insights to determine patient's survival status.
# 
# ##### 2. Below are the list of main conclusions:
# * If patient has more than 8 axillary nodes and treatment year 1965-67, then certainly they will not be able to survive for 5 yrs.
# * If patient has less than or 8 axillary nodes and treatment year 1961-65, then more chances they will be able to survive for 5 yrs.
# * Patients with age < 35 yrs and positive axillary nodes less than or 8 will be able to survive for 5 yrs or longer.
# * Patients with > 50 positive axillary nodes certainly not able to survive for 5 years.
# * Patients chances to survive is less when their age lies in intervals 41-51 yrs, 64-70 yrs, or > 78 yrs.
# * Patients chances to survive for 5yrs or more is greater when their age <= 40 yrs.
# * Patients chances to survive for 5yrs or more is greater between 1958-62.
# * Patients chances to survive is less in the year 1963-66.
# * Patients having age 40-66 yrs and 0-5 positive axillary nodes have survived for 5 yrs or longer.

# In[ ]:


## Below Logic can be built on the basis of above final observation:

if (positive_axillary_nodes between 0-5 and patient_age between 40-66 yrs): 
    print("patient will certainly be able to survive for 5 yrs or longer);
else if (positive_axillary_nodes <=8 and (treatment_year between 1961-65 OR patient_age <= 40 yrs) ): 
    print("patient is be able to survive for 5 yrs or longer);
else if (positive_axillary_nodes >50): 
    print("patient will certainly NOT be able to survive for 5 yrs);
else if ((positive_axillary_nodes >8 and positive_axillary_nodes <= 50) and treatment_year between 1965 and 1967): 
    print("patient will NOT be able to survive for 5 yrs); 


# In[ ]:




