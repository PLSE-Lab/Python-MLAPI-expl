#!/usr/bin/env python
# coding: utf-8

# # Hepatitis Dataset

# What is Hepatitis?
# Hepatitis is a term used to describe inflammation (swelling) of the liver. It can be caused due to viral infection or when liver is exposed to harmful substances such as alcohol. Hepatitis may occur with limited or no symptoms, but often leads to jaundice, anorexia (poor appetite) and malaise. Hepatitis is of 2 types: acute and chronic.
# 
# Acute hepatitis occurs when it lasts for less than six months and chronic if it persists for longer duration.
# 
# A group of viruses known as the hepatitis viruses most commonly   cause the disease, but hepatitis can also be caused by toxic substances (notably alcohol, certain medications, some industrial organic solvents and plants), other infections and autoimmune diseases.

# ![Movies](data:image/jpeg;base64,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)

# ##### DATASET Abstract:
# 

# 
# Data Set Characteristics:  Multivariate
# 
# Number of Instances:155
# 
# Area: Life
# 
# Attribute Characteristics: Categorical, Integer, Real
# 
# Number of Attributes: 19
# 
# Date Donated: 1988-11-01
# 
# Associated Tasks: Classification
# 
# Missing Values? : Yes
# 
# Number of Web Hits: 237264
# 
# 

# ##### Attribute information: 

#                 1     2
#      1. Class: DIE, LIVE
#      2. AGE: 10, 20, 30, 40, 50, 60, 70, 80
#      3. SEX: male, female
#      4. STEROID: no, yes
#      5. ANTIVIRALS: no, yes
#      6. FATIGUE: no, yes
#      7. MALAISE: no, yes
#      8. ANOREXIA: no, yes
#      9. LIVER BIG: no, yes
#     10. LIVER FIRM: no, yes
#     11. SPLEEN PALPABLE: no, yes
#     12. SPIDERS: no, yes
#     13. ASCITES: no, yes
#     14. VARICES: no, yes
#     15. BILIRUBIN: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00
#     16. ALK PHOSPHATE: 33, 80, 120, 160, 200, 250
#     17. SGOT: 13, 100, 200, 300, 400, 500, 
#     18. ALBUMIN: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0
#     19. PROTIME: 10, 20, 30, 40, 50, 60, 70, 80, 90
#     20. HISTOLOGY: no, yes
#     
# About the hepatitis database and BILIRUBIN problem I would like to say the following: BILIRUBIN is continuous attribute (= the number of it's "values" in the ASDOHEPA.DAT file is negative!!!); "values" are quoted because when speaking about the continuous attribute there is no such thing as all possible values.

# ##### MORE ABOUT ATTRIBUTES:

# 
#     FATIGUE --- Fatigue is a term used to describe an overall feeling of tiredness or lack of energy.
#     MALAISE --- A general sense of being unwell, often accompanied by fatigue, diffuse pain or lack of interest in activities.
#     ANOREXIA -- An eating disorder causing people to obsess about weight and what they eat.
#     LIVER BIG - An enlarged liver is one that's bigger than normal.
#     LIVER FIRM - The edge of the liver is normally thin and firm.
#     SPLEEN PALPABLE - The spleen is the largest organ in the lymphatic system. It is an important organ for keeping bodily 
#                       fluids balanced, but it is possible to live without it.
#     SPIDERS --- Spider nevus (also known as spider angioma or vascular spider) is a common benign vascular anomaly that may  
#                 appear as solitary or multiple lesions.
#     ASCITES --- Ascites is extra fluid in the space between the tissues lining the abdomen and the organs in the abdominal 
#                 cavity (such as the liver, spleen, stomach).
#     VARICES --- The liver becomes scarred, and the pressure from obstructed blood flow causes veins to expand.
#     BILIRUBIN -- Levels of bilirubin in the blood go up and down in patients with hepatitis C. ... High levels of bilirubin can 
#                  cause jaundice (yellowing of the skin and eyes, darker urine, and lighter-colored bowel movements). 
#     ALK PHOSPHATE -- Alkaline phosphatase (often shortened to alk phos) is an enzyme made in liver cells and bile ducts. The 
#                      alk phos level is a common test that is usually included when liver tests are performed as a group.
#     SGOT --- AST, or aspartate aminotransferase, is one of the two liver enzymes. It is also known as serum glutamic-
#              oxaloacetic transaminase, or SGOT. When liver cells are damaged, AST leaks out into the bloodstream and the level 
#              of AST in the blood becomes elevated.
#     ALBUMIN -- A low albumin level in patients with hepatitis C can be a sign of cirrhosis (advanced liver disease). Albumin 
#                levels can go up and down slightly. Very low albumin levels can cause symptoms of edema, or fluid accumulation, 
#                in the abdomen (called ascites) or in the leg (called edema).
#     PROTIME -- The "prothrombin time" (PT) is one way of measuring how long it takes blood to form a clot, and it is measured 
#                in seconds (such as 13.2 seconds). A normal PT indicates that a normal amount of blood-clotting protein is 
#                available.

# # Importing the Libraries
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Importing the dataset
# 
# 

# In[ ]:


data = pd.read_csv('../input/hepatitis-data/hepatitisdata.csv')


# In[ ]:


data.head()


# ##### Check Describe:

# In[ ]:


data.describe()


# As we can, we unable to describe all the column of our data set. lets check info of dataset.

# ##### Check Information :

# In[ ]:


data.info()


# Now, by comparing the info() and describe() of data, only the integer data type are visible to us. 

# # Data Cleaning

# ##### Change DataType

# In[ ]:


data['steroid'] = pd.to_numeric(data['steroid'],errors='coerce')
data['fatigue'] = pd.to_numeric(data['fatigue'],errors='coerce')
data['malaise'] = pd.to_numeric(data['malaise'],errors='coerce')
data['anorexia'] = pd.to_numeric(data['anorexia'],errors='coerce')
data['liver_big'] = pd.to_numeric(data['liver_big'],errors='coerce')
data['liver_firm'] = pd.to_numeric(data['liver_firm'],errors='coerce')
data['spleen_palable'] = pd.to_numeric(data['spleen_palable'],errors='coerce')
data['spiders'] = pd.to_numeric(data['spiders'],errors='coerce')
data['ascites'] = pd.to_numeric(data['ascites'],errors='coerce')
data['varices'] = pd.to_numeric(data['varices'],errors='coerce')
data['bilirubin'] = pd.to_numeric(data['bilirubin'],errors='coerce')
data['alk_phosphate'] = pd.to_numeric(data['alk_phosphate'],errors='coerce')
data['sgot'] = pd.to_numeric(data['sgot'],errors='coerce')
data['albumin'] = pd.to_numeric(data['albumin'],errors='coerce')
data['protime'] = pd.to_numeric(data['protime'],errors='coerce')


# ##### Check Info()

# In[ ]:


data.info()


# ##### Replace (1,2) values with (0,1)

# In[ ]:


data["class"].replace((1,2),(0,1),inplace=True)
data["sex"].replace((1,2),(0,1),inplace=True)
data["age"].replace((1,2),(0,1),inplace=True)
data["steroid"].replace((1,2),(0,1),inplace=True)
data["antivirals"].replace((1,2),(0,1),inplace=True)
data["fatigue"].replace((1,2),(0,1),inplace=True)
data["malaise"].replace((1,2),(0,1),inplace=True)
data["anorexia"].replace((1,2),(0,1),inplace=True)
data["liver_big"].replace((1,2),(0,1),inplace=True)
data["liver_firm"].replace((1,2),(0,1),inplace=True)
data["spleen_palable"].replace((1,2),(0,1),inplace=True)
data["spiders"].replace((1,2),(0,1),inplace=True)
data["ascites"].replace((1,2),(0,1),inplace=True)
data["varices"].replace((1,2),(0,1),inplace=True)
data["histology"].replace((1,2),(0,1),inplace=True)


# In[ ]:


data.head()


# ##### Check Null Values

# In[ ]:


data.isna().sum()


# Here we can see many null values.

# ##### Fill Null Values

# In this data set there are two type of variables, i.e. 
#     1. Catagorical.
#     2. Numerical
# so, to fill null values of catagorical variable we used mode and for numerical variable we used mean or median after checking there skewness. 

# # Catagorical Columns

# In[ ]:


data['steroid'].mode()


# In[ ]:


data['steroid'].replace(to_replace=np.nan,value=1,inplace=True)
data['steroid'].head()


# In[ ]:


data['fatigue'].mode()


# In[ ]:


data['fatigue'].replace(to_replace=np.nan,value=0,inplace=True)


# In[ ]:


data['malaise'].mode()


# In[ ]:


data['malaise'].replace(to_replace=np.nan,value=1,inplace=True)


# In[ ]:


data['anorexia'].mode()


# In[ ]:





# In[ ]:


data['anorexia'].replace(to_replace=np.nan,value=1,inplace=True)


# In[ ]:


data['liver_big'].mode()


# In[ ]:


data['liver_big'].replace(to_replace=np.nan,value=1,inplace=True)


# In[ ]:


data['liver_firm'].mode()


# In[ ]:


data['liver_firm'].replace(to_replace=np.nan,value=1,inplace=True)


# In[ ]:


data['spleen_palable'].mode()


# In[ ]:


data['spleen_palable'].replace(to_replace=np.nan,value=1,inplace=True)


# In[ ]:


data['spiders'].mode()


# In[ ]:


data['spiders'].replace(to_replace=np.nan,value=1,inplace=True)


# In[ ]:


data['ascites'].mode()


# In[ ]:


data['ascites'].replace(to_replace=np.nan,value=1,inplace=True)


# In[ ]:


data['varices'].mode()


# In[ ]:


data['varices'].replace(to_replace=np.nan,value=1,inplace=True)


# # Numerical Columns

# we check skewness by skew() of pandas.

# In[ ]:


data['bilirubin'].skew(axis=0,skipna = True)


# In[ ]:


data['bilirubin'].median()


# In[ ]:


data['bilirubin'].replace(to_replace=np.nan,value=1,inplace=True)


# In[ ]:


data['alk_phosphate'].skew(axis=0,skipna = True)


# In[ ]:


data['alk_phosphate'].median()


# In[ ]:


data['alk_phosphate'].replace(to_replace=np.nan,value=85,inplace=True)


# In[ ]:


data['sgot'].skew(axis=0,skipna = True)


# In[ ]:


data['sgot'].median()


# In[ ]:


data['sgot'].replace(to_replace=np.nan,value=58,inplace=True)


# In[ ]:


data['albumin'].skew(axis=0,skipna = True)


# In[ ]:


data['albumin'].median()


# In[ ]:


data['albumin'].mean()


# In[ ]:


data['albumin'].replace(to_replace=np.nan,value=4,inplace=True)


# In[ ]:


data['protime'].skew(axis=0,skipna = True)


# Here skewness is near to symmetri, so we can check both mean and median.

# In[ ]:


data['protime'].median()


# In[ ]:


data['protime'].mean()


# In[ ]:


data['protime'].replace(to_replace=np.nan,value=61,inplace=True)


# now we filled all the null values.

# ##### Check for Null Value Count.

# In[ ]:


data.isnull().sum()


# Now, there is no null values present in our data set.

# ##### Check Describe

# In[ ]:


data.describe()


# Here we can see various parameters such as Mean,Standard Deviation,Minimum value,Maximum value,Median and quartiles of the dataset.

# In[ ]:


data.head(10)


# # Data Visualization

# ##### Plot Pie Chart of Class Column.

# In[ ]:


die =len(data[data['class'] == 0])
live = len(data[data['class']== 1])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'DIE','LIVE'
sizes = [die,live]
colors = ['orange', 'lightgreen']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# Here we can see the Ratio of alive and died.

# ##### Plot Pie Chart of Sex Column

# In[ ]:


male =len(data[data['sex'] == 0])
female = len(data[data['sex']==1])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'Male','Female'
sizes = [male,female]
colors = ['skyblue', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# Here we can see the ratio of male and female

# ##### Plot Pie Chart of Steroid Column

# In[ ]:


no =len(data[data['steroid'] == 0])
yes = len(data[data['steroid']== 1])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'Avoid','Consume'
sizes = [no,yes]
colors = ['skyblue', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# ##### Draw count plot to visualize consumption of steriod in relation with Age

# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x='age',data = data, hue = 'steroid')
plt.show()


# ##### Plot Pie Chart of antivirals Column

# In[ ]:


no =len(data[data['antivirals'] == 0])
yes = len(data[data['antivirals']== 1])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'Avoid','Consume'
sizes = [no,yes]
colors = ['skyblue', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# ##### Draw count plot to visualize consumption of antivirals in relation with Age

# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x='age',data = data, hue = 'antivirals',palette='GnBu')
plt.show()


# ##### Plot Pie Chart of fatigue Column

# In[ ]:


no =len(data[data['fatigue'] == 0])
yes = len(data[data['fatigue']== 1])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'Never Exausted','Was Excausted'
sizes = [no,yes]
colors = ['skyblue', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# ##### Draw Count plot to visualize count of people sufer from fatigue in relation with Age.

# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x='age',data = data, hue = 'fatigue',palette='BrBG')
plt.show()


# ##### Plot Pie Chart of malaise Column

# In[ ]:


no =len(data[data['malaise'] == 0])
yes = len(data[data['malaise']== 1])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'Never in Discomfort','Was in Discomfort'
sizes = [no,yes]
colors = ['skyblue', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# ##### Draw Count plot to visualize count of people sufer from malaise in relation with Age.

# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x='age',data = data, hue = 'malaise',palette='RdPu')
plt.show()


# ##### Plot Pie Chart of anorexia Column

# In[ ]:


no =len(data[data['anorexia'] == 0])
yes = len(data[data['anorexia']== 1])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'NO','YES'
sizes = [no,yes]
colors = ['skyblue', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# ##### Draw Count plot to visualize count of people sufer from anorexia in relation with Age.

# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x='age',data = data, hue = 'anorexia',palette='RdPu')
plt.show()


# ##### Plot Pie Chart of liver_big Column

# In[ ]:


no =len(data[data['liver_big'] == 0])
yes = len(data[data['liver_big']== 1])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'NO','YES'
sizes = [no,yes]
colors = ['skyblue', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# ##### Draw Count plot to visualize count of people sufer from liver_big in relation with Age.

# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x='age',data = data, hue = 'liver_big',palette='RdPu')
plt.show()


# ##### Plot Pie Chart of liver_firm Column

# In[ ]:


no =len(data[data['liver_firm'] == 0])
yes = len(data[data['liver_firm']== 1])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'NO','YES'
sizes = [no,yes]
colors = ['skyblue', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# ##### Draw Count plot to visualize count of people sufer from liver_firm in relation with Age.

# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x='age',data = data, hue = 'liver_firm',palette='RdPu')
plt.show()


# ##### Plot Pie Chart of spleen_palable Column

# In[ ]:


no =len(data[data['spleen_palable'] == 0])
yes = len(data[data['spleen_palable']== 1])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'NO','YES'
sizes = [no,yes]
colors = ['skyblue', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# ##### Draw Count plot to visualize count of people sufer from spleen_palable in relation with Age.

# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x='age',data = data, hue = 'spleen_palable',palette='RdPu')
plt.show()


# ##### Plot Pie Chart of spiders Column

# In[ ]:


no =len(data[data['spiders'] == 0])
yes = len(data[data['spiders']== 1])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'NO','YES'
sizes = [no,yes]
colors = ['skyblue', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# ##### Draw Count plot to visualize count of people sufer from spiders in relation with Age.

# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x='age',data = data, hue = 'spiders',palette='RdPu')
plt.show()


# ##### Plot Pie Chart of ascites Column

# In[ ]:


no =len(data[data['ascites'] == 0])
yes = len(data[data['ascites']== 1])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'NO','YES'
sizes = [no,yes]
colors = ['skyblue', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# ##### Draw Count plot to visualize count of people sufer from ascites in relation with Age.

# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x='age',data = data, hue = 'ascites',palette='RdPu')
plt.show()


# ##### Plot Pie Chart of varices Column

# In[ ]:


no =len(data[data['varices'] == 0])
yes = len(data[data['varices']== 1])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'NO','YES'
sizes = [no,yes]
colors = ['skyblue', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# ##### Draw Count plot to visualize count of people sufer from varices in relation with Age.

# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x='age',data = data, hue = 'varices',palette='RdPu')
plt.show()


# ##### Draw a Scatter Plot to visualize bilirubin test values with Age and Hue Class

# In[ ]:


plt.figure(figsize=(10,8))
sns.scatterplot(x='age',y='bilirubin',data = data,hue = 'class')
plt.title('Bilirubin test values according to AGE')
plt.show()


# ##### Draw a Scatter Plot to visualize alk_phosphate test values with Age and Hue Class

# In[ ]:


plt.figure(figsize=(10,8))
sns.scatterplot(x='age',y='alk_phosphate',data = data,hue = 'class')
plt.title('alk_phosphate test values according to AGE')
plt.show()


# ##### Draw a Scatter Plot to visualize sgot test values with Age and Hue Class

# In[ ]:


plt.figure(figsize=(10,8))
sns.scatterplot(x='age',y='sgot',data = data,hue = 'class')
plt.title('sgot test values according to AGE')
plt.show()


# ##### Draw a Scatter Plot to visualize albumin test values with Age and Hue Class

# In[ ]:


plt.figure(figsize=(10,8))
sns.scatterplot(x='age',y='albumin',data = data,hue = 'class')
plt.title('albumin test values according to AGE')
plt.show()


# ##### Draw a Scatter Plot to visualize protime test values with Age and Hue Class

# In[ ]:


plt.figure(figsize=(10,8))
sns.scatterplot(x='age',y='protime',data = data,hue = 'class')
plt.title('protime test values according to AGE')
plt.show()


# ##### Plot Pie Chart of histology Column

# In[ ]:


no =len(data[data['histology'] == 0])
yes = len(data[data['histology']== 1])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'NO','YES'
sizes = [no,yes]
colors = ['skyblue', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# ##### Draw a countplot to show count of people having positive history with Age.

# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x='age',data = data, hue = 'histology',palette='GnBu')
plt.show()


# ##### Draw a Heat Map of Data

# In[ ]:


plt.figure(figsize=(15,12))
sns.heatmap(data.corr(), cmap='coolwarm',linewidths=.1,annot = True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




