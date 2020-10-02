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


# **Importing Necessary Libraries**

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# **Loading the Haberman dataset**

# In[ ]:


df = pd.read_csv(r'/kaggle/input/habermans-survival-data-set/haberman.csv')


# In[ ]:


print(df)


# **Checking Important info about the dataset**

# In[ ]:


df.info()


# In[ ]:


print(df.columns)


# **From the above we can see that there are no meaningful column names on the dataset. So lets create them**

# In[ ]:


df.columns=['Patient Age','TreatmentYear','PositiveLymphNodes','Survivalstatus5years']
print(df.columns)


# **The 'SurvivalStatus5years' has to be changed to categorical datatype as this is the main objective of our problem and we want to carry out our EDA based upon that.**

# In[ ]:


status = {1:'yes',2:'no'}
df.Survivalstatus5years = [status[newval] for newval in df.Survivalstatus5years]


# **Checking the ratio of Survival status verses Non Survival status in terms of yes and no**

# In[ ]:


df["Survivalstatus5years"].value_counts()
print(df.iloc[:,-1].value_counts(normalize=True))


# In[ ]:


df.describe()


# **Observations:
# 1) The minimum to maximum range of the Patients ranges from 30 years to 83 years. 2) The average age of the patients is about 52 years. 3) The maximum number of positive lymph nodes is 52, out of which 75% of the age group have less than 4 positive lymph nodes while 50% and 25% have less than 5 or no positive lymph nodes respectively. 4) It is an imbalanced dataset as 73% of the values are in the Survival side while only 26% are not in the survival side. So a big gap is observed between survival and non survival side, hence imbalanced dataset**

# **Plotting Graphs**

# **2-D Scatter plot with color-coding for survival status class between Patient's Age and PositiveLymphNodes**

# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(df, hue='Survivalstatus5years',height=4)    .map(plt.scatter,"Patient Age","PositiveLymphNodes")    .add_legend();
plt.title("Relationship between Patient's Age and PositiveLymphNodes")
plt.show();


# **Observation:From the above plot we can see that there are too many overlapping values between a Patient's age and Postive Nodes. So its very difficult to choose the relevant variable**

# **2-D Scatter plot with color-coding for survival status class between Patient's Age and Year of Operation**

# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(df, hue='Survivalstatus5years',height=4)    .map(plt.scatter,"Patient Age","TreatmentYear")    .add_legend();
plt.title("Relationship between Patient's Age and Operation Year")
plt.show();


# **Observation:From the above plot we can see that there are too many overlapping points between Patient's age and Year of Operation Hence its very difficult to choose the relevant variable in this case**

# **2-D Scatter plot with color-coding for survival status class between Year of Operation  and Positive Lymph Node**

# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(df, hue='Survivalstatus5years',height=4)    .map(plt.scatter,"TreatmentYear","PositiveLymphNodes")    .add_legend();
plt.title("Relationship between Operation Year and PositiveLymphNodes")
plt.show();


# **Observation:This plot looks a bit better but still there are a lot of overlapping points so cant consider any variables for classification**

# **Plotting Pair Plot**

# In[ ]:


plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="Survivalstatus5years",height=3);
plt.show()


# **Observation: 1)From the above pair plot variables "Patient Age" and "Treatment Year" cannot be used for classifcation due to lots of overlapping points between them 2) PositiveLymphNodes looks like an appropriate variable for classification out of the rest as we can see that the survival rate for patients with more than 10 nodes can be separated from patients having less than 10 nodes**

# **Plotting Density Plots**

# **Density Plot for Patient's Age and Survival Status**

# In[ ]:


sns.FacetGrid(df,hue = "Survivalstatus5years", height=5)     .map(sns.distplot, "Patient Age")     .add_legend();
plt.title("Density Plot for Patient's Age and Survival Status")
plt.show();


# **Density Plot for Operation Year and Survival Status**

# In[ ]:


sns.FacetGrid(df,hue = "Survivalstatus5years", height=5)     .map(sns.distplot, "TreatmentYear")     .add_legend();
plt.title("Density Plot for Operation Year and Survival Status")
plt.show();


# **Density Plot for PositiveLymphNodes and SurvivalStatus**

# In[ ]:


sns.FacetGrid(df,hue = "Survivalstatus5years", height=5)     .map(sns.distplot, "PositiveLymphNodes")     .add_legend();
plt.title("Density Plot for PositiveLymphNodes and SurvivalStatus")
plt.show();


# **Observations:From the above PDF Plots the PositiveLymphNodes is the most clear one as it gives the below insight. 1) The survival rate is higher than the non survival rate 2) The survival rate decreases if the PositiveLymphNodes exceeds more than 3**

# **PDF & CDF Analysis**

# In[ ]:


df_survival = df.loc[df["Survivalstatus5years"] == 'yes']
counts, bin_edges = np.histogram(df_survival['PositiveLymphNodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)

#Compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend("Survivalstatus5years")
plt.legend(['Survived PDF','Survived CDF'])
plt.xlabel("PositiveLymphNodes")
plt.ylabel("Proability Rate")
plt.title("PDF & CDF Plot")

plt.show();


# In[ ]:


df_survival = df.loc[df["Survivalstatus5years"] == 'no']
counts, bin_edges = np.histogram(df_survival['PositiveLymphNodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)

#Compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend("Survivalstatus5years")
plt.legend(['Died PDF','Died CDF'])
plt.xlabel("PositiveLymphNodes")
plt.ylabel("Proability Rate")
plt.title("PDF & CDF Plot")
plt.show();


# **Observations:From the above graph we can say that patients having less than 10 Positive lymph nodes have 80% chances of survival rate than patients having more than 40 Positivelymphnodes**

# **Plotting Box Plots now**

# **Box Plot for SurvivalStatus verses PositiveLymphNodes**

# In[ ]:


sns.boxplot(x='Survivalstatus5years',y='PositiveLymphNodes',data=df)
plt.title("Box Plot for SurvivalStatus verses PositiveLymphNodes")
plt.show()


# **Observations:1) From the first box plot we can say that 75th percentile value for the PositiveLymphNodes is like 2 and less than 5, for which the survival rate is quite high. 2) From the second box plot we can say that 75th percentile value for the PositiveLymphNodes is 3 and higher for the Patient having less chances of survival**

# **Plotting Violin Plot**

# **Violin Plot for SurvivalStatus verses PositiveLymphNodes**

# In[ ]:


sns.violinplot(x='Survivalstatus5years',y="PositiveLymphNodes",data=df,height=8)
plt.title("Violin Plot for SurvivalStatus verses PositiveLymphNodes")
plt.show()


# **Observations:1)From the first violin plot we can say that 50th percentile value for the PositiveLymphNodes is like 0 for which the survival rate is quite high than 75th percentile of the Patients having less than 3 PositiveLymphNodes 2)From the second violin plot we can say that 25th percentile value for the PositiveLymphNodes is like 1 for a Patient having less chances of survival,50th percentile value for patients likely to die with PositiveLymphNodes being below 4 and 75th percentile value for patients likely to die with PositiveLymphNodes being above 11**

# ****Conclusion**
# 1) In order to survive the patients shouldn't have positive lymph nodes greater than 4
# 2) The less the age of the patient, more chances of survival
# 3) Overall fewer the number of nodes the more chances of survival
# 4) Patien't year of Operation has nothing to do with her Survival status**
# 
