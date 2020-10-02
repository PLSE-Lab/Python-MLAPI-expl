#!/usr/bin/env python
# coding: utf-8

# The Haberman's Survival dataset contains information from a study conducted between 1958 and 1970 at the University of Chicago's Billings hospital
# on the survival of patients who had undergone surgery for breast cancer.

# In[2]:


#Loading packages
import pandas as po
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings


# In[3]:


#Reading Dataset
hdat=po.read_csv("../input/haberman.csv")


# In[4]:


#Number of rows and columns
hdat.shape


# In[5]:


#Column names
hdat.columns


# In[6]:


hdat.count()


# In[7]:


#Updating the column names
hdat.columns=['Age','Op_Year','axil_nodes','Surv_status']
hdat.head()


# In[8]:


#Converting 'Survival Status to categorical type
hdat.Surv_status=hdat.Surv_status.replace({1:"Survived",2:"Did Not Survive"})


# In[9]:


hdat['Age'].min()


# In[10]:


hdat['Age'].max()


# In[11]:


hdat['Age'].mean()


# In[12]:


hdat['Surv_status'].value_counts()


# In[13]:


#Bar plot to show Survival Status
plot1=hdat['Surv_status'].value_counts().plot(kind='bar',title='Survival Staus of patients')
plot1.set_xlabel('Survival Status')
plot1.set_ylabel('Count')


# ## Univariate Analysis

# In[14]:


#Probability Density Function

warnings.filterwarnings("ignore")
sb.FacetGrid(hdat, hue = "Surv_status", size = 5).map(sb.distplot, "Age").add_legend()
plt.title("Age of Patient vs Survival Status")
plt.ylabel("Density")
plt.show()


# Observation:
# From the above histogram, we observe that patients who were less than 40 years of age were likely to survive surgery.

# In[15]:


#Cummulative Distribution Function

counts, bin_edges = np.histogram(hdat['axil_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
legend=['PDF','CDF']
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.title('PDF & CDF for Number of Axillary Nodes')
plt.xlabel('Axillary Nodes')
plt.ylabel('Percentage')
plt.legend(legend)
plt.show()


# Observation:
# (1): From PDF we can say that roughly 10% of patients have axillary nodes between 0 and 10.
# (2): From CDF we can infer that around 83% of patients have less than 10 axillary nodes.

# ## Box Plot

# In[16]:


#Boxplot showing Survival 
sb.boxplot(x='Surv_status',y='axil_nodes', data=hdat)
plt.title("Survival Status vs Number of Axillary Nodes")
plt.show()


# Observation:
# From the above boxplot, we observe that patients with axillary nodes between 1 and 23 did not survive surgery.

# ## Violin Plot

# In[17]:


sb.violinplot(x='Surv_status', y='axil_nodes', data = hdat, size = 8.5)
plt.xlabel('Survival Status')
plt.ylabel('Number of Axillary Nodes')
plt.title('Survival Status vs Number of Axillary Nodes')
plt.show()


# Observation:
# From the violin plot we observe that patients with less than 10 axillary nodes survived surgery.

# ## Multivariate Analysis

# In[18]:


#Scatter plot showing Survival Status based on Age and Axillary Nodes
sb.set_style("whitegrid");
sb.FacetGrid(hdat, hue="Surv_status", size=6.5)    .map(plt.scatter, "Age", "axil_nodes")    .add_legend();
plt.title('Surgery survival based on Age and Number of Axillary Nodes')
plt.show();


# Observation:
# From the above scatter plot, we observe that patients with axillary nodes between 0 and 10 were more likely to survive surgery(more number of blue dots indicating survival).

# ## Pair Plot

# In[19]:


sb.set_style("whitegrid")
sb.pairplot(hdat, hue = "Surv_status", size = 3)
plt.suptitle("Pair plot of Age,Year of operation and Axillary Nodes")
plt.show()


# Observation:
# The plot does not give much information about which feature can be used to classify whether a prospective patient will survive surgery or not as there is data overlapping.

# ## Final Conclusions

# Conclusions:
# (1): There are 306 rows and 4 columns in the dataset.
# (2): All the four columns have count of 306 which indicates there are no missing values.
# (3): The column 'Surv_status' is converted to categorical variable.
# (4): Minimum and Maximum age of the respondents are 30 and 83 years respectively. Average age is 52 years.
# (5): Out of 306 respondents, 225 have survived surgery and 81 did not survive.
# (6): Patients who were less than 40 years of age were likely to survive surgery.
# (7): 83% patients have less than 10 axillary nodes which further suggests that lesser the number of axillary nodes more likely the patient is likely to survive surgery.
