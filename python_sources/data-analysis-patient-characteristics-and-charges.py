#!/usr/bin/env python
# coding: utf-8

# **<h3>INTRODUCTION</h3>**
# * In this tutorial,I will describe it from my data analysis as a beginner on the 2015 de-identified NY inpatient discharge dataset.
# 
# Content:
# * [Summarize the Dataset](#1)
# * [Missing Data Capture](#2)
# * [Tidy Data](#3)
# * [Concatenating Data](#4)
# * [Building Data Frames From Scratch](#5)
# * [Visual Exploratory Data Analysis](#6)
# * [Indexing Pandas Time Series](#7)
# * [Resampling Pandas Time Series](#8)
# * [Indexing and Silicing Data Frames](#9)
# * [Filtering Data Frames](#10)
# * [Transforming Data](#11)
# * [Grouping Data](#12)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visualization

import warnings            
warnings.filterwarnings("ignore") 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **<h3> Summarize the Dataset </h3>**
# Now it is time to take a look at the data.
# <br>In this step we are going to take a look at the data a few different ways:</br>
#   * Dimensions of the dataset.
#         We can get a quick idea of how many instances (rows) and how many attributes (columns) the data contains with the shape property.
#   * Features name in dataset.
#   * Peek at the data itself.
#   * Statistical summary of all attributes.
#         This includes the count, mean, the min and max values as well as some percentiles.

# In[ ]:


#Load dataset
data=pd.read_csv('../input/Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015.csv')
#data includes how many rows and columns
data.shape
print("Our data has {} rows and {} columns".format(data.shape[0],data.shape[1]))
#Features name in data
data.columns


# In[ ]:


#diplay first 5 rows
data.head()


# In[ ]:


#display last 5 rows
data.tail()


# In[ ]:


print("Data Type:")
data.dtypes


# <p>When we look at the data types of our data set, we have the Total Charges and Total Costs property type object and the dollar sign in their content.  We can not convert it to a float type when there is a  dollar sign in those features. we need to remove this sign. But first we need to organize the space between the names of our data columns.</p>
# <p>I want to see the Length of Stay feature in the statistical summary, and I need to convert the object type to int first. But I get an error because of the '120 +' record in the conversion. I am clearing the '+' sign in this record.</p>
# <br>Then we can change the types to float so we can see these features statistically.</br>

# In[ ]:


#column name change
data.columns=[each.replace(" ","_") for each in data.columns]

#remove dollar sign
data.Total_Charges=[each.replace("$","") for each in data.Total_Charges]
data.Total_Costs=[each.replace("$","") for each in data.Total_Costs]

#lets convert object to float
data["Total_Charges"]=data["Total_Charges"].astype('float')
data["Total_Costs"]=data["Total_Costs"].astype('float')

#Delete the + sign
data.Length_of_Stay=[each.replace("+","") if(each=="120 +") else each for each in data.Length_of_Stay]
#lets convert object to int
data["Length_of_Stay"]=data["Length_of_Stay"].astype('int')


# In[ ]:


#Let's look again
data.dtypes


# In[ ]:


data.loc[:,["Total_Costs","Total_Charges","Birth_Weight","Length_of_Stay"]].describe()


# <a id=2></a>
# **<h3>Missing Data Capture</h3>**
# * Let's check if there is missing data in the features of our data.

# In[ ]:


#checking for missing values
print("Are there missing values? {}".format(data.isnull().any().any()))
#missing value control in features
data.isnull().sum()


# * Missing data control with Assert method.
# <br>We use the Assert method to check for missing data in the Hospital_Country property. We are planning to get an error because we have missing data.</br>

# In[ ]:


assert data["Hospital_County"].notnull().all()


# In[ ]:


#we found out how many Type of Admission
print("Type of Admission in Dataset:\n")
print(data.Type_of_Admission.unique())
#we found out how many Age group
print("\n\nAge Group in Dataset:\n")
print(data.Age_Group.unique())
#we found out how many ARP Risk of Mortality
print("\n\nARP Risk of Mortality:\n")
print(data.APR_Risk_of_Mortality.unique())
#we found out how many hospital country in our data
print("\n\nHospital Country in Dataset:\n")
print("There are {} different values\n".format(len(data.Hospital_County.unique())))
print(data.Hospital_County.unique())
#we found out how many ARP MDC Description
print("\n\nARP MDC Description(disease diagnosis) in Dataset:\n")
print("There are {} different values\n".format(len(data.APR_MDC_Description.unique())))
print(data.APR_MDC_Description.unique())


# In[ ]:


#We group features by data numbers
#show it if missing value(dropna=False)
data["Type_of_Admission"].value_counts(dropna=False)


#  We group the Type of Admission property into values and we see that there are many patients who are accepted emergency.Also we again understand that there is no missing data in the Type of Admission feature.

# In[ ]:


#number of patients by age groups
#show it if missing value(dropna=False)
data["Age_Group"].value_counts(dropna=False)


# In[ ]:


#show it if missing value(dropna=False)
print("Patients with or without abortion:\n")
print(data["Abortion_Edit_Indicator"].value_counts(dropna=False))


# * When we group the Abortion_Edit_Indicator feature, Type_of_Admission does not contain only newborns. So we found newborns with filtering.

# In[ ]:


#filtering
data_newborn=data['Type_of_Admission']=='Newborn'
print("Total Newborns:",data_newborn.count())
data[data_newborn].head()


# In[ ]:


#grouping of mortality risk values
#show it if missing value(dropna=False)
data["APR_Severity_of_Illness_Description"].value_counts(dropna=False)


# <a id=3></a>
# **<h3>Tidy Data(Melting)</h3>**
# * We have transformed into a different structure with the melt () method to find out the features of the first five elements in our dataset ['Age_Group', 'Length_of_Stay', 'Type_of_Admission'].

# In[ ]:



data_new = data.head()
melted = pd.melt(frame = data_new, id_vars = 'APR_MDC_Description', value_vars = ['Age_Group','Type_of_Admission'])
melted


# <a id=4></a>
# **<h3>Concatenating Data</h3>**
# * age group of the diagnosis and the patient

# In[ ]:


#firstly lets create 2 data frame
data1=data['APR_MDC_Description'].tail()
data2=data['Age_Group'].tail()

conc_data_col=pd.concat([data1,data2],axis=1)
conc_data_col


# <a id=5></a>
# **<h3>Building Data Frames From Scratch</h3>**

# In[ ]:


#data frames from dictionary
Hospital=list(data["Hospital_County"].head())
Facility=list(data["Facility_Name"].head())
Year=list(data["Discharge_Year"].head())
Costs=list(data["Total_Costs"].head())

list_label=["hospital_country","facility_name","discharge_year","total_costs"]
list_col=[Hospital,Facility,Year,Costs]
zipped=list(zip(list_label,list_col))
data_dict=dict(zipped)

df=pd.DataFrame(data_dict)
df


# **<h3>Broadcasting</h3>**
# * Create new column and assign a value to entire column
#     <br>There is no special feature about patients' entry into the hospital, so let's add it and get zero by default.</br>

# In[ ]:


#add new column
data["Entry_Year"]=0
data.head()


# <a id=6></a>
# **<h3>Visual Exploratory Data Analysis</h3>**
# * plot
# * subplot
# * histogram

# In[ ]:


#ploting
data1=data.loc[:,["Total_Costs","Total_Charges","Birth_Weight","Length_of_Stay"]]
data1.plot()
plt.show()
#this is complete


# In[ ]:


#To solve the above complexity
#subplot
data1.plot(subplots=True)
plt.show()


# * Let's look at the frequency of total costs with histogram

# In[ ]:


#histogram
data1.plot(kind="hist",y="Total_Costs",bins=50,range=(0,250),normed=True)
plt.show()


# In[ ]:


#histogram subplot with non cumulative an cumulative
fig,axes=plt.subplots(nrows=2,ncols=1)

data1.plot(kind="hist",y="Total_Costs",bins=50,range=(0,250),normed=True,ax=axes[0])
data1.plot(kind="hist",y="Total_Costs",bins=50,range=(0,250),normed=True,ax=axes[1],cumulative=True)

plt.savefig("Graph.png")
plt.show()


# <a id=7></a>
# **<h3>Indexing Pandas Time Series</h3>**
# * First we print the Discharge_Year property in our data queue. We convert the type of our feature to datetime with the to_datetime() method.

# In[ ]:


print(df["discharge_year"])
df.discharge_year=pd.to_datetime(df["discharge_year"])
#lets make discharge_year as index
df=df.set_index("discharge_year")
df


# <a id=8></a>
# **<h3>Resampling Pandas Time Series</h3>**
# * Let's take the average of all years in our data. Since our data is a single year, a single-line result is output.

# In[ ]:


df.resample("A").mean()
#lets resample with month
#df.resample("M").mean()


# <a id=9></a>
# **<h3>Indexing and Silicing Data Frames</h3>**
# 

# In[ ]:


#indexing data frame
#using loc accessor
print(data.loc[85,['APR_DRG_Description']])
#selecting only some columns
data[["APR_DRG_Description","Age_Group","Length_of_Stay"]].head(20)


# In[ ]:


#silincing and indexing data series
print(data.loc[1:10,"Race":"Length_of_Stay"])
#from something to end
data.loc[1:10,"Gender":]


# <a id=10></a>
# **<h3>Filtering Data Frames</h3>**

# In[ ]:


first_filter=data.Gender=="F"
second_filter=data.Abortion_Edit_Indicator=="Y"
data[first_filter & second_filter].head()
#filtering columns based others
#data.Gender[data.Race=="Black/African American"]


# <a id=11></a>
# **<h3>Transforming Data</h3>**

# In[ ]:


#Defining column using other columns
data["Average_Costs"]=data.Total_Costs.mean()
data.head()

#print(data.Total_Costs.apply(lambda n:n/2))


# <a id=12></a>
# **<h3>Grouping Data</h3>**

# In[ ]:


print("Total hospitalization times for patients admitted to the hospital as Urgent:",
      data['Length_of_Stay'][data['Type_of_Admission']=='Urgent'].sum())

#The first value of unique races of patients coming to the hospital
data.groupby("Race").first()

