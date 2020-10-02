#!/usr/bin/env python
# coding: utf-8

# # Hepatitis Dataset

# What is Hepatitis?
# Hepatitis is a term used to describe inflammation (swelling) of the liver. It can be caused due to viral infection or when liver is exposed to harmful substances such as alcohol. Hepatitis may occur with limited or no symptoms, but often leads to jaundice, anorexia (poor appetite) and malaise. Hepatitis is of 2 types: acute and chronic.
# 
# Acute hepatitis occurs when it lasts for less than six months and chronic if it persists for longer duration.
# 
# A group of viruses known as the hepatitis viruses most commonly   cause the disease, but hepatitis can also be caused by toxic substances (notably alcohol, certain medications, some industrial organic solvents and plants), other infections and autoimmune diseases.

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

# ### Importing the Libraries
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
from time import time


# ### Importing the dataset
# 
# 

# In[ ]:


data = pd.read_csv('../input/hepatitis-data/hepatitisdata.csv')


# In[ ]:


data.head()


# # EDA

# ##### Drop Column Unnamed: 0

# In[ ]:


data.drop('Unnamed: 0',axis=1,inplace=True)
data.head(2)


# ##### Display Dataset

# In[ ]:


data.head()


# Now, we can see the column names.

# ##### Check Describe:

# In[ ]:


data.describe()


# As we can, we unable to describe all the column of our data set. lets check info of dataset.

# ##### Check Information :

# In[ ]:


data.info()


# Now, by comparing the info() and describe() of data, only the integer data type are visible to us. 

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

# ##### Catagorical Columns

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


# ##### Numerical Columns

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


# ## Visualization

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


# # Model Training

# In[ ]:


data['age'].unique()


# In[ ]:


data['age']=np.where((data['age'] <18) ,'Teenager/Child',
                               np.where((data['age'] >=18) & (data['age'] <=25),'Young',
                                np.where((data['age'] >=25) & (data['age'] <=40),'Adult',
                               'Old')))


# In[ ]:


data['age'].value_counts()


# In[ ]:


data =pd.get_dummies(data)
data.head()


# #### Split Data

# In[ ]:


x=data.iloc[:,1:]
y=data['class']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[ ]:


scale=['bilirubin', 'alk_phosphate', 'sgot', 'albumin']
x_train[scale]=sc.fit_transform(x_train[scale])
x_test[scale]=sc.transform(x_test[scale])


# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


params_reg= {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,'max_iter':[100,150,200,250,300]}


# In[ ]:


reg_cv=RandomizedSearchCV(reg,params_reg,cv=10,random_state=42)
reg_cv.fit(x_train,y_train)


# In[ ]:


print(reg_cv.best_score_)
print(reg_cv.best_params_)


# In[ ]:


log_reg=LogisticRegression(max_iter=250,C=0.1)
start_reg=time()
log_reg.fit(x_train,y_train)
end_reg=time()
time_reg=end_reg-start_reg


# In[ ]:


log_train_time = log_reg.score(x_train,y_train)
log_test_time = log_reg.score(x_test,y_test)
print('Training score: ',log_reg.score(x_train,y_train))
print('Testing score: ',log_reg.score(x_test,y_test))
print('Training time: ',time_reg)


# In[ ]:


y_predict_reg=log_reg.predict(x_test)
y_predict_reg


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[ ]:


accuracy_score(y_test,y_predict_reg)


# In[ ]:


cm_reg=confusion_matrix(y_test,y_predict_reg)
cm_reg


# In[ ]:


print(classification_report(y_test,y_predict_reg))


# ### Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[ ]:


params_dt={'criterion':['gini','entropy'],'min_samples_split':np.arange(2,10),'splitter':['best','random'],'max_depth':[2,3,4,5,6],'max_features':['auto','sqrt','log2',None]}


# In[ ]:


dt_cv=RandomizedSearchCV(dt,params_dt,cv=10,random_state=15)
dt_cv.fit(x_train,y_train)


# In[ ]:


print(dt_cv.best_score_)
print(dt_cv.best_params_)


# In[ ]:


decision_tree=DecisionTreeClassifier(splitter='random',min_samples_split=2,max_features='log2',max_depth=6,criterion='entropy',random_state=10)
start_dt=time()
decision_tree.fit(x_train,y_train)
end_dt=time()
time_dt=end_dt-start_dt


# In[ ]:


dt_train_time = decision_tree.score(x_train,y_train)
dt_test_time = decision_tree.score(x_test,y_test)
print('Training score: ',decision_tree.score(x_train,y_train))
print('Testing score: ',decision_tree.score(x_test,y_test))
print('Training time: ',time_dt)


# In[ ]:


y_predict_dt=decision_tree.predict(x_test)
y_predict_dt


# In[ ]:


cm_dt=confusion_matrix(y_test,y_predict_dt)
print(cm_dt)


# In[ ]:


print(classification_report(y_test,y_predict_dt))


# ### Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()


# In[ ]:


params_rf={'n_estimators':[5,10,15,20,50,100,200,300,400,500],'criterion':['entropy','gini'],'max_depth':[2,3,4,5,6],'max_features':['auto','sqrt','log2',None],'bootstrap':[True,False]}


# In[ ]:


rf_cv=RandomizedSearchCV(rf,params_rf,cv=10,random_state=15)
rf_cv.fit(x_train,y_train)


# In[ ]:


print(rf_cv.best_score_)
print(rf_cv.best_params_)


# In[ ]:


random_forest=RandomForestClassifier(n_estimators=300,max_features='log2',max_depth=5,criterion='gini',bootstrap=False,random_state=0)
start_rf=time()
random_forest.fit(x_train,y_train)
end_rf=time()
time_rf=end_rf-start_rf


# In[ ]:


rf_train_time = random_forest.score(x_train,y_train)
rf_test_time = random_forest.score(x_test,y_test)
print('Training score: ',random_forest.score(x_train,y_train))
print('Testing score: ',random_forest.score(x_test,y_test))
print('Training time: ',time_rf)


# In[ ]:


y_predict_rf=random_forest.predict(x_test)
y_predict_rf


# In[ ]:


cm_rf=confusion_matrix(y_test,y_predict_rf)
print(cm_rf)


# In[ ]:


print(classification_report(y_test,y_predict_rf))


# ### KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[ ]:


params_knn={'n_neighbors':[5,6,7,8,9,10]}


# In[ ]:


knn_cv=RandomizedSearchCV(knn,params_knn,cv=10,random_state=42)
knn_cv.fit(x_train,y_train)
print(knn_cv.best_score_)
print(knn_cv.best_params_)


# In[ ]:


KNN=KNeighborsClassifier(n_neighbors=10)
start_knn=time()
KNN.fit(x_train,y_train)
end_knn=time()
time_knn=end_knn-start_knn


# In[ ]:


knn_train_time = KNN.score(x_train,y_train)
knn_test_time = KNN.score(x_test,y_test)
print('Training score: ',KNN.score(x_train,y_train))
print('Testing score: ',KNN.score(x_test,y_test))
print('Training time: ',time_knn)


# In[ ]:


y_predict_knn=KNN.predict(x_test)
y_predict_knn


# In[ ]:


cm_knn=confusion_matrix(y_test,y_predict_knn)
print(cm_knn)


# In[ ]:


print(classification_report(y_test,y_predict_knn))


# ### Support Vector Classifier

# In[ ]:


from sklearn.svm import SVC
svc=SVC()


# In[ ]:


params_svm={'kernel':['linear','poly','rbf','sigmoid'],'C':list(np.arange(0.1,0.6)),'gamma':[0.0001,0.001,0.01,0.1,1,10,100,0.02,0.03,0.04,0.05],'degree':[1,2,3,4,5,6]}


# In[ ]:


svm_cv=RandomizedSearchCV(svc,params_svm,cv=10,random_state=7)
svm_cv.fit(x_train,y_train)


# In[ ]:


print(svm_cv.best_score_)
print(svm_cv.best_params_)


# In[ ]:


SVM=SVC(kernel='linear',gamma=1,degree=2,C=0.1)
start_svm=time()
SVM.fit(x_train,y_train)
end_svm=time()
time_svm=end_svm-start_svm


# In[ ]:


SVM_train_time = SVM.score(x_train,y_train)
SVM_test_time = SVM.score(x_test,y_test)
print('Training score: ',SVM.score(x_train,y_train))
print('Testing score: ',SVM.score(x_test,y_test))
print('Training time: ',time_svm)


# In[ ]:


y_predict_svm=SVM.predict(x_test)
y_predict_svm


# In[ ]:


cm_svm=confusion_matrix(y_test,y_predict_svm)
print(cm_svm)


# In[ ]:


print(classification_report(y_test,y_predict_svm))


# ### Gaussian Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()


# In[ ]:


start_gnb=time()
gnb.fit(x_train,y_train)
end_gnb=time()
time_gnb=end_gnb-start_gnb


# In[ ]:


gnb_train_time = gnb.score(x_train,y_train)
gnb_test_time = gnb.score(x_test,y_test)
print('Training score: ',gnb.score(x_train,y_train))
print('Testing score: ',gnb.score(x_test,y_test))
print('Training time: ',time_gnb)


# In[ ]:


y_predict_gnb=gnb.predict(x_test)
y_predict_gnb


# In[ ]:


cm_gnb=confusion_matrix(y_test,y_predict_gnb)
print(cm_gnb)


# In[ ]:


print(classification_report(y_test,y_predict_gnb))


# ## Compairing Training Accuracy of Different Models

# In[ ]:


model_training_time = pd.Series(data=[knn_train_time,log_train_time,dt_train_time,rf_train_time,SVM_train_time,gnb_train_time],
                          index=['KNN','Logistic','DecisionTreeClassifier','RandomForestClassifier','Support Vector','Naive Bayes'])
fig= plt.figure(figsize=(10,7))
model_training_time.sort_values().plot.barh()
plt.title('Model Training Accuracy')


# ## Compairing Testing Accuracy of Different Models
# 

# In[ ]:


model_testing_time = pd.Series(data=[knn_test_time,log_test_time,dt_test_time,rf_test_time,SVM_test_time,gnb_test_time],
                          index=['KNN','Logistic','DecisionTreeClassifier','RandomForestClassifier','Support Vector','Naive Bayes'])
fig= plt.figure(figsize=(10,7))
model_testing_time.sort_values().plot.barh()
plt.title('Model Testing Accuracy')


# ## Comparing Confusion Matrix of different Classifiers

# In[ ]:


knn_con = confusion_matrix(y_test, y_predict_knn)
log_con = confusion_matrix(y_test, y_predict_reg)
nb_con = confusion_matrix(y_test, y_predict_gnb)
dtc_con = confusion_matrix(y_test, y_predict_dt)
rf_con = confusion_matrix(y_test, y_predict_rf)
svm_con = confusion_matrix(y_test, y_predict_svm)


plt.figure(figsize=(20,10))
plt.subplot(2,4,1)
plt.title("KNeighborsClassifier")
sns.heatmap(knn_con,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.subplot(2,4,2)
plt.title("LogisticRegression")
sns.heatmap(log_con,annot=True,cmap="Oranges",fmt="d",cbar=False)
plt.subplot(2,4,3)
plt.title("GaussianNB")
sns.heatmap(nb_con,annot=True,cmap="Greens",fmt="d",cbar=False)
plt.subplot(2,4,4)
plt.title("DecisionTreeClassifier")
sns.heatmap(dtc_con,annot=True,cmap="Purples",fmt="d",cbar=False)
plt.subplot(2,4,5)
plt.title("RandomForestClassifier")
sns.heatmap(rf_con,annot=True,cmap="Purples",fmt="d",cbar=False)
plt.subplot(2,4,6)
plt.title("Support Vector Classifier")
sns.heatmap(svm_con,annot=True,cmap="Greens",fmt="d",cbar=False)
plt.show()


# ### Comparing Training times of different Classifiers

# In[ ]:


training_times=[time_dt,time_rf,time_knn,time_gnb,time_svm,time_reg]
algo=['Decision Tree Classifier','Random Forest Classifier','KNN','Gaussian Naive Bayes','Support Vector Classifier','Logistic Regression']


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(y=algo,x=training_times,palette='ocean')
plt.xlabel('Training Time')
plt.grid()
plt.show()


# ### Bagging Classifier

# In[ ]:


from sklearn.ensemble import BaggingClassifier


# In[ ]:


df_bagging=DataFrame(columns=['Base Estimator','Training Score','Testing Score','Time Taken','Case Of','Diff(Training-Testing)'])
to_bag=[decision_tree,random_forest,log_reg,KNN,SVM,gnb]
algo_name=['Decision Tree','Random Forest','Logistic Regression','KNN','Support Vector Classifier','Gaussian Naive Bayes Classifier']
for i in range(len(to_bag)):
    case_of=''
    difference=0
    bag=BaggingClassifier(to_bag[i],bootstrap=True,random_state=0)
    start=time()
    bag.fit(x_train,y_train)
    end=time()
    time_taken=end-start
    if bag.score(x_train,y_train)>bag.score(x_test,y_test):
        case_of='Overfitting'
    else:
        case_of='Underfitting'
    difference=bag.score(x_train,y_train)-bag.score(x_test,y_test)
    df_bagging.loc[i]=[algo_name[i],bag.score(x_train,y_train),bag.score(x_test,y_test),time_taken,case_of,difference]
df_bagging


# ### Adaboost Classifier

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


df_adaboost=DataFrame(columns=['Base Estimator','Training Score','Testing Score','Time Taken','Case Of','Diff(Training-Testing)'])
to_bag=[decision_tree,random_forest,log_reg,KNN,SVM,gnb]
to_boost=[decision_tree,random_forest,log_reg,SVM,gnb]
algo_name=['Decision Tree','Random Forest','Logistic Regression','Support Vector Classifier','Gaussian Naive Bayes Classifier']
for i in range(len(to_boost)):
    boost=AdaBoostClassifier(to_boost[i],n_estimators=100,algorithm='SAMME',random_state=7)
    start=time()
    boost.fit(x_train,y_train)
    end=time()
    time_taken=end-start
    if boost.score(x_train,y_train)>boost.score(x_test,y_test):
        case_of='Overfitting'
    else:
        case_of='Underfitting'
    difference=boost.score(x_train,y_train)-boost.score(x_test,y_test)
    df_adaboost.loc[i]=[algo_name[i],boost.score(x_train,y_train),boost.score(x_test,y_test),time_taken,case_of,difference]
df_adaboost


# ### Gradient Boosting Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
params_gb={'learning_rate':[0.1,0.2,0.3,0.4,0.5,1,2,0.01,0.02,0.05],'n_estimators':[100,150,200,300],
           'max_depth':[2,3,4,5,6],'min_samples_split':list(np.arange(1,10)),'criterion':['friedman_mse','mse','mae']}
GB_cv=RandomizedSearchCV(gb,params_gb,cv=10,random_state=7)
GB_cv.fit(x_train,y_train)


# In[ ]:


print('Best score: ',GB_cv.best_score_)
print('Best parameters: ',GB_cv.best_params_)


# In[ ]:


GB=GradientBoostingClassifier(n_estimators=150,max_depth=2,learning_rate=0.5,min_samples_split=8,criterion='mse')


# In[ ]:


GB.fit(x_train,y_train)
GB.score(x_test,y_test)


# In[ ]:


df_gboost=DataFrame(columns=['Base Estimator','Training Score','Testing Score','Time Taken','Case Of','Diff(Training-Testing)'])
to_boost=[decision_tree,random_forest,log_reg,SVM,gnb]
algo_name=['Decision Tree','Random Forest','Logistic Regression','Support Vector Classifier','Gaussian Naive Bayes Classifier']
for i in range(len(to_boost)):
    gboost=GradientBoostingClassifier(n_estimators=200,max_depth=2,learning_rate=0.2,min_samples_split=8,criterion='mse')
    start=time()
    gboost.fit(x_train,y_train)
    end=time()
    time_taken=end-start
    if gboost.score(x_train,y_train)>boost.score(x_test,y_test):
        case_of='Overfitting'
    else:
        case_of='Underfitting'
    difference=boost.score(x_train,y_train)-boost.score(x_test,y_test)
    df_gboost.loc[i]=[algo_name[i],gboost.score(x_train,y_train),gboost.score(x_test,y_test),time_taken,case_of,difference]
df_gboost


# ### USING SMOTE

# In[ ]:


from imblearn.over_sampling import SMOTE
sm=SMOTE(sampling_strategy=1,k_neighbors=5,random_state=0)
x_train_res,y_train_res=sm.fit_sample(x_train,y_train)


# In[ ]:


print(x_train_res.shape)
print(y_train_res.shape)


# ### Logistic Regression

# In[ ]:


params_reg= {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,'max_iter':[100,150,200,250,300]}


# In[ ]:


reg_cv1=RandomizedSearchCV(reg,params_reg,cv=10,random_state=0)
reg_cv1.fit(x_train_res,y_train_res)
print(reg_cv1.best_score_)
print(reg_cv1.best_params_)


# In[ ]:


LR=LogisticRegression(max_iter=100,C=10,random_state=0)
LR.fit(x_train_res,y_train_res)
print('Training score: ',LR.score(x_train_res,y_train_res))
print('Testing score: ',LR.score(x_test,y_test))


# In[ ]:


y_predict_LR=LR.predict(x_test)
print('CONFUSION MATRIX:  ')
print(confusion_matrix(y_test,y_predict_LR))
print()
print('CLASSIFICATION REPORT: ')
print()      
print(classification_report(y_test,y_predict_LR))


# ### Decision Tree Classifier

# In[ ]:


params_dt={'criterion':['gini','entropy'],'min_samples_split':np.arange(2,10),'splitter':['best','random'],'max_depth':[2,3,4,5,6],'max_features':['auto','sqrt','log2',None]}


# In[ ]:


dt_cv1=RandomizedSearchCV(dt,params_dt,cv=15,random_state=42)
dt_cv1.fit(x_train_res,y_train_res)
print(dt_cv1.best_score_)
print(dt_cv1.best_params_)


# In[ ]:


DT=DecisionTreeClassifier(splitter='best',min_samples_split=7,max_features=None,max_depth=5,criterion='entropy')
DT.fit(x_train_res,y_train_res)
print('Training score: ',DT.score(x_train_res,y_train_res))
print('Testing score: ',DT.score(x_test,y_test))


# In[ ]:


y_predict_DT=DT.predict(x_test)
print('CONFUSION MATRIX:  ')
print(confusion_matrix(y_test,y_predict_DT))
print()
print('CLASSIFICATION REPORT: ')
print()      
print(classification_report(y_test,y_predict_DT))


# ### Random Forest Classifier

# In[ ]:


params_rf={'n_estimators':[5,10,15,20,50,100,200,300,400,500],'criterion':['entropy','gini'],'max_depth':[2,3,4,5,6],'max_features':['auto','sqrt','log2',None],'bootstrap':[True,False]}


# In[ ]:


rf_cv1=RandomizedSearchCV(rf,params_rf,cv=10,random_state=0)
rf_cv1.fit(x_train_res,y_train_res)
print(rf_cv1.best_score_)
print(rf_cv1.best_params_)


# In[ ]:


RF=RandomForestClassifier(n_estimators=500,max_features='log2',max_depth=6,criterion='gini',bootstrap=True)
RF.fit(x_train_res,y_train_res)
print('Training score: ',RF.score(x_train_res,y_train_res))
print('Testing score: ',RF.score(x_test,y_test))


# In[ ]:


y_predict_RF=RF.predict(x_test)
print('CONFUSION MATRIX:  ')
print(confusion_matrix(y_test,y_predict_RF))
print()
print('CLASSIFICATION REPORT: ')
print()      
print(classification_report(y_test,y_predict_RF))


# ### KNN

# In[ ]:


params_knn={'n_neighbors':[5,6,7,8,9,10]}


# In[ ]:


knn_cv1=RandomizedSearchCV(knn,params_knn,cv=10,random_state=0)
knn_cv1.fit(x_train_res,y_train_res)
print(knn_cv1.best_score_)
print(knn_cv1.best_params_)


# In[ ]:


Knn=KNeighborsClassifier(n_neighbors=5)
Knn.fit(x_train_res,y_train_res)
print('Training score: ',RF.score(x_train_res,y_train_res))
print('Testing score: ',RF.score(x_test,y_test))


# In[ ]:


y_predict_KNN=Knn.predict(x_test)
print('CONFUSION MATRIX:  ')
print(confusion_matrix(y_test,y_predict_KNN))
print()
print('CLASSIFICATION REPORT: ')
print()      
print(classification_report(y_test,y_predict_KNN))


# ### Support Vector Classifier

# In[ ]:


params_svm={'kernel':['linear','poly','rbf','sigmoid'],'C':list(np.arange(0.1,0.6)),'gamma':[0.0001,0.001,0.01,0.1,1,10,100,0.02,0.03,0.04,0.05],'degree':[1,2,3,4,5,6]}


# In[ ]:


svm_cv1=RandomizedSearchCV(svc,params_svm,cv=10,random_state=7)
svm_cv1.fit(x_train_res,y_train_res)
print(svm_cv1.best_score_)
print(svm_cv1.best_params_)


# In[ ]:


SVM1=SVC(kernel='poly',gamma=10,degree=3,C=0.1)
SVM1.fit(x_train_res,y_train_res)
print('Training score: ',SVM1.score(x_train_res,y_train_res))
print('Testing score: ',SVM1.score(x_test,y_test))


# In[ ]:


y_predict_SVM1=SVM1.predict(x_test)
print('CONFUSION MATRIX:  ')
print(confusion_matrix(y_test,y_predict_SVM1))
print()
print('CLASSIFICATION REPORT: ')
print()      
print(classification_report(y_test,y_predict_SVM1))


# ### Gaussian Naive Bayes Classifier

# In[ ]:


from sklearn.naive_bayes import GaussianNB
GNB=GaussianNB()
GNB.fit(x_train_res,y_train_res)
print('Training score: ',GNB.score(x_train_res,y_train_res))
print('Testing score: ',GNB.score(x_test,y_test))


# In[ ]:


y_predict_GNB=GNB.predict(x_test)
print('CONFUSION MATRIX:  ')
print(confusion_matrix(y_test,y_predict_GNB))
print()
print('CLASSIFICATION REPORT: ')
print()      
print(classification_report(y_test,y_predict_GNB))


# ### Bagging Classifier 

# In[ ]:


df_bagging_sm=DataFrame(columns=['Base Estimator','Training Score','Testing Score','Time Taken','Case Of','Diff(Training-Testing)'])
to_bag=[DT,RF,LR,Knn,SVM1,GNB]
algo_name=['Decision Tree','Random Forest','Logistic Regression','KNN','Support Vector Classifier','Gaussian Naive Bayes Classifier']
for i in range(len(to_bag)):
    case_of=''
    difference=0
    BAG=BaggingClassifier(to_bag[i],bootstrap=True,random_state=0)
    start=time()
    BAG.fit(x_train_res,y_train_res)
    end=time()
    time_taken=end-start
    if BAG.score(x_train_res,y_train_res)>BAG.score(x_test,y_test):
        case_of='Overfitting'
    else:
        case_of='Underfitting'
    difference=BAG.score(x_train_res,y_train_res)-BAG.score(x_test,y_test)
    df_bagging_sm.loc[i]=[algo_name[i],BAG.score(x_train_res,y_train_res),BAG.score(x_test,y_test),time_taken,case_of,difference]
df_bagging_sm


# ### Adaboost Classifier

# In[ ]:


df_adaboost_sm=DataFrame(columns=['Base Estimator','Training Score','Testing Score','Time Taken','Case Of','Diff(Training-Testing)'])
to_boost=[DT,RF,LR,SVM1,GNB]
algo_name=['Decision Tree','Random Forest','Logistic Regression','Support Vector Classifier','Gaussian Naive Bayes Classifier']
for i in range(len(to_boost)):
    boost=AdaBoostClassifier(to_boost[i],n_estimators=100,algorithm='SAMME',random_state=0)
    start=time()
    boost.fit(x_train_res,y_train_res)
    end=time()
    time_taken=end-start
    if boost.score(x_train_res,y_train_res)>boost.score(x_test,y_test):
        case_of='Overfitting'
    else:
        case_of='Underfitting'
    difference=boost.score(x_train_res,y_train_res)-boost.score(x_test,y_test)
    df_adaboost_sm.loc[i]=[algo_name[i],boost.score(x_train_res,y_train_res),boost.score(x_test,y_test),time_taken,case_of,difference]
df_adaboost_sm


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
params_gb={'learning_rate':[0.1,0.2,0.3,0.4,0.5,1,2,0.01,0.02,0.05],'n_estimators':[100,150,200,300],
           'max_depth':[2,3,4,5,6],'min_samples_split':list(np.arange(1,10)),'criterion':['friedman_mse','mse','mae']}
GB_cv1=RandomizedSearchCV(gb,params_gb,cv=10,random_state=7)
GB_cv1.fit(x_train_res,y_train_res)


# In[ ]:


print('Best score: ',GB_cv.best_score_)
print('Best parameters: ',GB_cv.best_params_)


# In[ ]:


GB_sm=GradientBoostingClassifier(n_estimators=300,min_samples_split=7,max_depth=3,learning_rate=0.5,criterion='mse')
GB_sm.fit(x_train_res,y_train_res)
print('Training score: ',GB_sm.score(x_train_res,y_train_res))
print('Testing score: ',GB_sm.score(x_test,y_test))


# In[ ]:


df_gboost_sm=DataFrame(columns=['Base Estimator','Training Score','Testing Score','Time Taken','Case Of','Diff(Training-Testing)'])
to_boost=[DT,RF,LR,SVM1,GNB]
algo_name=['Decision Tree','Random Forest','Logistic Regression','Support Vector Classifier','Gaussian Naive Bayes Classifier']
for i in range(len(to_boost)):
    gboost=GradientBoostingClassifier()
    start=time()
    gboost.fit(x_train_res,y_train_res)
    end=time()
    time_taken=end-start
    if gboost.score(x_train_res,y_train_res)>gboost.score(x_test,y_test):
        case_of='Overfitting'
    else:
        case_of='Underfitting'
    difference=gboost.score(x_train_res,y_train_res)-gboost.score(x_test,y_test)
    df_gboost_sm.loc[i]=[algo_name[i],gboost.score(x_train_res,y_train_res),gboost.score(x_test,y_test),time_taken,case_of,difference]
df_gboost_sm


# ## Conclusion

# #### In all the Cases Logistic Regression performed well and got least difference between Training and Testing Score.

# **Hope you like this notebook consider Upvoting**

# In[ ]:




