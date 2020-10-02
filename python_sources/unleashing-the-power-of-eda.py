#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing the libraries

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.offline import iplot


# In[ ]:


data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')


# In[ ]:


print("Number of rows in data :", data.shape[0])
print("Number of columns in data :", data.shape[1])


# In[ ]:


data.head()


# In[ ]:


data.info()


# **Cool, this data doesn't contain any categorical variable and null values which is a really good thing as we will not have to invest a lot of time on cleaning the data and we can focus more on predictive modelling.**
# 
# 
# **Also, the target here is our target vector and all others are predictors? Well, we can't say this early.**
# 
# **Let's dive straight in and find out!**

# In[ ]:


# Statistical properties of data
data.describe().round(3)


# **As we can see that we got a lot of information using a single function. These are called the descriptive properties of data which are very useful for a data like this which is full of numerical attributes. So never leave this step.**

# In[ ]:


# Columns names

data.columns


# **These names looks gribberish! We don't want to move back and forth seeing what our columns mean so let's change the column names to something meaningful.**

# In[ ]:


data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate',
       'exercise_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']


# In[ ]:


data.head()


# **That's more like it!**
# 
# **Also, these columns are already encoded so for data analysis it is better to change them so that it is easily to look what we are exactly plotting. Don't worry we will do this on a copy of data.**

# In[ ]:


new_data = data.copy() # for future use


# In[ ]:


# Let's see which columns has less or equal to 5 classes

for column in data.columns:
    if len(data[column].unique())<=5:
        print(f"{column} has {len(data[column].unique())} classes.")


# In[ ]:


# Let's map these class values to something meaning full information given in dataset description
# Note: As rest_ecg  and num_major_vessels doesn't have any description so I left that column

data.sex = data.sex.map({0:'female', 1:'male'})

data.chest_pain_type = data.chest_pain_type.map({1:'angina pectoris', 2:'atypical angina', 3:'non-anginal pain', 4:'SMI', 0:'absent'})

data.fasting_blood_sugar = data.fasting_blood_sugar.map({0:'lower than 120mg/ml', 1:'greater than 120mg/ml'})

data.exercise_angina = data.exercise_angina.map({0:'no', 1:'yes'})

data.st_slope = data.st_slope.map({1:'upsloping', 2:'horizontal', 3:'downsloping', 0:'absent'})

data.thalassemia = data.thalassemia.map({1:'normal', 2:'fixed defect', 3:'reversable defect', 0:'absent'})

data.target = data.target.map({0:'No Heart Disease', 1:'Heart Disease'})


# In[ ]:


data.head()


# **That was a lot of typing but the end result looks so much more tempting to be analysed. Let's move to our favourite part, GRAPHS!**

# # Analysis of associations of different columns

# ## Correlation Plot

# In[ ]:


# Drawing a correlation Plot

fig=plt.figure(figsize=(12,8))
sns.heatmap(new_data.corr(), annot= True, cmap='Blues')


# ## Age and Sex Column

# In[ ]:


print(f"Minimum Age : {min(data.age)} years")
print(f"Maximum Age : {max(data.age)} years")


# In[ ]:


sex_counts = data['sex'].value_counts().tolist()

print(f"Number of Male patients: {sex_counts[0]}")
print(f"Number of Female patients: {sex_counts[1]}")


# In[ ]:


# Count of male and female patients

sns.countplot('sex', hue="sex", data=data, palette="bwr")


# In[ ]:


# Let's look at the distribution of age

hist_data = [data['age']]
group_labels = ['age'] 

colors = ['#835AF1']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data, group_labels=group_labels, colors=colors,
                         bin_size=10, show_rug=False)

# Add title
fig.update_layout(width=700, title_text='Age Distribution')
fig.show()


# **OBSERVATION**
# 
# **Distribution of age looks close to a Normal Distribution. This data has age of patients ranging from 29-77 which is good as the data is not biased towards certain kind of patients.**
# 
# **Let's look at some of the ranges.**

# In[ ]:


young_patients = data[(data['age']>=29)&(data['age']<40)]
middle_aged_patients = data[(data['age']>40)&(data['age']<55)]
old_aged_patients = data[(data['age']>55)]

print(f"Number of Young Patients : {len(young_patients)}")
print(f"Number of Middle Aged Patients: {len(middle_aged_patients)}")
print(f"Number of Old Aged Patients : {len(old_aged_patients)}")


# **OBSERVATION**
# 
# **There are 16 young patients which is quite obvious beacuse heart diseases is not very common in younger population but we have almost large number of patients in middle age and Old age which is also obvious.**

# In[ ]:


# Plotting a pie chart for age ranges of patients

labels = ['Young Age','Middle Aged','Old Aged']
values = [
      len(young_patients), 
      len(middle_aged_patients),
      len(old_aged_patients)
]
colors = ['gold', 'mediumturquoise', 'darkorange']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='value+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, line=dict(color='#000000', width=2)))

plotly.offline.iplot([trace], filename='styled_pie_chart')


# In[ ]:


# Age and Target based on sex

fig = px.bar(data, x=data['target'], y=data['age'], color='sex', height=500, width=800)
fig.update_layout(title_text='BarChart for Age Vs Target on Basis of Sex')
fig.show()


# **OBSERVATION**
# 
# **We can see that the people suffering from heart disease and people not suffering from heart disease almost similiar ages and there are almost equal number of male and female patients having heart disease.**

# In[ ]:


# Box plot for target and age based on sex

fig = px.box(data, x="target", y="age", points="all", color='sex')
fig.update_layout(title_text='BoxPlot of Age Vs Target')
fig.show()


# **OBSERVATION**
# 
# **The median of ages of patients not having heart disease is slightly higher than patients having heart diseases. We can see that there is one point in male patients without heart disease which is exceeding the BoxPlot whiskers thus it can be termed as an Outlier.**

# In[ ]:


# Plot of age and Maximum Heart Rate

df = px.data
fig = px.scatter(data, x="age", y="max_heart_rate", color="sex", hover_data=['age','max_heart_rate'])
fig.update_layout(title = "Scatter Plot of Age Vs Max Heart Rate")
fig.show()


# **OBSERVATION**
# 
# **Too bad! This data isn't correlated at all so it is of no use to us.**

# In[ ]:


# Plot of Age vs Resting Blood Pressure

df = px.data
fig = px.scatter(data, x="age", y="resting_blood_pressure", color="sex", hover_data=['age','resting_blood_pressure'])
fig.update_layout(title = "Scatter Plot of Age Vs Resting Blood Pressure")
fig.show()


# In[ ]:


# Plot of age vs Serum Cholesterol

df = px.data
fig = px.scatter(data, x="age", y="serum_cholesterol", color="sex", hover_data=['age','serum_cholesterol'])
fig.update_layout(title = "Scatter Plot of Age Vs Serum Cholesterol")
fig.show()


# **OBSERVATION**
# 
# **Both Resting Blood Pressure and Serum Cholesterol shows a bit positive correlation but not that much. Also, Resting Blood Pressure and Serum Cholesterol have few Outliers which we will remove in preprocessing steps.**

# ## Chest Pain Type

# In[ ]:


# Counts of Chest pain type among Heart Patients and Non-Heart Patients

sns.countplot(x="chest_pain_type", hue="target", data=data, palette="bwr")
plt.title("Chest Pain Types grouped by Targets")


# **OBSERVATION**
# 
# **The most common type of chest pain among heart disease patients is "atypical agina" followed by "agina pectoris". Also, we can see around 40 of the patients don't have chest pain but still have heart disease so absence of chest pain does not guarantee that the patient being diagonsed has no Heart Disease.**

# In[ ]:


# Counts of Chest pain type among male and female patients

sns.countplot(x="chest_pain_type", hue="sex", data=data, palette="bwr")


# **OBSERVATION**
# 
# **There are more male patients with no chest pain as compared to female.**

# ## Resting Blood Pressure

# In[ ]:


# Finding the maximum and minimum Resting Blood Pressure

print(f"Maximum Resting Blood Pressure : {data['resting_blood_pressure'].max()}")
print(f"Minimum Resting Blood Pressure : {data['resting_blood_pressure'].min()}")


# In[ ]:


# Let's look at the distribution of Resting Blood Pressure

hist_data = [data['resting_blood_pressure']]
group_labels = ['resting_blood_pressure'] 

colors = ['#008080']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data, group_labels=group_labels, colors=colors,
                         bin_size=10, show_rug=False)

# Add title
fig.update_layout(width=700, title_text='Resting Blood Pressure Distribution')
fig.show()


# In[ ]:


# Plot of Resting Blood Pressure with Target

sns.barplot(x="target", y='resting_blood_pressure',data = data, palette="bwr")
plt.title('Resting Blood Pressure vs Target')
plt.show()


# In[ ]:


# Boxplot of Resting Blood Pressure with Target

fig = px.box(data, x="target", y="resting_blood_pressure", points="all", color='sex')
fig.update_layout(title_text='BoxPlot of Resting Blood Pressure Vs Target')
fig.show()


# **OBSERVATIONS**
# 
# **(I) The distribution of resting blood pressure is close to normal distribution.**<br><br>
# **(II) The resting blood pressure of both Heart patients and Non-heart patients is almost same.**<br><br>
# **(III) Median Resting Blood Pressure with Heart Disease - Male (130) and Female (130)<br><br>
#         Median Resting Blood Pressure without Heart Disease - Male (130) and Female (140)**

# ## Serum Cholesterol	

# In[ ]:


# Finding the maximum and minimum Serum Cholesterol

print(f"Maximum Serum Cholesterol : {data['serum_cholesterol'].max()}")
print(f"Minimum Serum Cholesterol : {data['serum_cholesterol'].min()}")


# In[ ]:


# Let's look at the distribution of Serum Cholesterol

hist_data = [data['serum_cholesterol']]
group_labels = ['serum_cholesterol'] 

colors = ['#DA70D6']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data, group_labels=group_labels, colors=colors,
                         bin_size=10, show_rug=False)

# Add title
fig.update_layout(width=700, title_text='Serum Cholesterol Distribution')
fig.show()


# In[ ]:


# Plot of Serum Cholesterol and Target

sns.barplot(x="target", y='serum_cholesterol',data = data, palette="bwr")
plt.title('Serum Cholesterol vs Target')
plt.show()


# In[ ]:


# Box Plot on basis of Sex

fig = px.box(data, x="target", y="serum_cholesterol", points="all", color='sex')
fig.update_layout(title_text='BoxPlot of Serum Cholesterol Vs Target')
fig.show()


# In[ ]:


# Box plot on basis of st_slope

fig = px.box(data, x="target", y="serum_cholesterol", points="all", color='st_slope')
fig.update_layout(title_text='Serum Cholesterol Vs Target ')
fig.show()


# **OBSERVATIONS**
# 
# **(I) The distribution of serum cholesterol is close to normal distribution and has a long tail.**<br><br>
# **(II) The resting blood pressure of both Heart Disease patients and Non-heart Disease patients is almost same.**<br><br>
# **(III) Median Serum Cholesterol with Heart Disease - Male (228) and Female (249)<br><br>
#         Median Serum Cholesterol without Heart Disease - Male (247.5) and Female (265.5)**<br><br>
# **(IV) st_slope doesn't seem to vary much**

# ## Fasting Blood Sugar

# In[ ]:


# Counts of Heart Disease and No Heart Disease Patients with fasting blood sugar above 120 mg/dl
# and lower than 120 mg/dl

sns.set(rc={'figure.figsize':(8.7,5.27)})

sns.countplot(hue='fasting_blood_sugar',x ='target',data = data, palette="bwr")
plt.title('Fasting Blood Sugar > 120 mg/dl')
plt.show()


# In[ ]:


# Counts of Male and Female Patients with fasting blood sugar above 120 mg/dl
# and lower than 120 mg/dl

sns.countplot(hue='fasting_blood_sugar',x ='sex',data = data, palette="bwr")
plt.title('Fasting Blood Sugar > 120 mg/dl')
plt.show()


# **OBSERVATIONS**
# 
# **(I) Patients having fasting blood sugar lower than 120 mg/dl have a high chance of suffering from Heart Disease but this is not a clear indicator as many of the patients without Heart Disease also have the same scenario.**
# 
# **(II) There are more Male Patients with lower and greater than 120 mg/dl fasting sugar as compared to Female Patients.**

# ## Rest ECG (rest_ecg)

# In[ ]:


# Resting electrocardiographic results (values 0,1,2))

sns.countplot(x='rest_ecg', hue ='target', data = data, palette="bwr")
plt.title('Resting electrocardiographic Results')
plt.show()


# In[ ]:


# Resting electrocardiographic results (values 0,1,2))

sns.countplot(x='rest_ecg', hue ='sex', data = data, palette="bwr")
plt.title('Resting electrocardiographic Results')
plt.show()


# **OBSERVATIONS**
# 
# **(I) There are many patients with having rest_ecg = 1 and suffering from heart disease. Also, rest_ecg = 2 seems to be very rare.**
# 
# **(II) rest_ecg = 0 and rest_ecg = 1 is found in most of the male as well as female patients.**

# ## Maximum Heart Rate

# In[ ]:


print(f"Maximum Max Heart Rate : {data['max_heart_rate'].max()}")
print(f"Minimum Max Heart Rate: {data['max_heart_rate'].min()}")


# In[ ]:


# Let's look at the distribution of Maximum Heart Rate

hist_data = [data['max_heart_rate']]
group_labels = ['max_heart_rate'] 

colors = ['#808000']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data, group_labels=group_labels, colors=colors,
                         bin_size=10, show_rug=False)

# Add title
fig.update_layout(width=700, title_text='Maximum Blood Pressure Distribution')
fig.show()


# In[ ]:


# Maximum heart rate and target 

sns.barplot(x="target", y='max_heart_rate',data = data, palette="bwr")
plt.title('Maximum Heart Rate vs Target')
plt.show()


# In[ ]:


# Box plot on basis of sex

fig = px.box(data, x="target", y="max_heart_rate", points="all", color='sex')
fig.update_layout(title_text='Maximum Heart Rate Vs Target')
fig.show()


# In[ ]:


# Box plot on basis of st_slope

fig = px.box(data, x="target", y="max_heart_rate", points="all", color='st_slope')
fig.update_layout(title_text='Maximum Heart Rate Vs Target ')
fig.show()


# **OBSERVATIONS**
# 
# **(I) The distribution of Maximum Heart Rate is close to normal distribution with little bit skewing to the left.**<br>
# **(II) The Maximum Heart Rate in Heart Disease Patients is  a bit more and than Non-Heart Disease Patients.**<br>
# **(III) Median Maximum Heart Rate with Heart Disease - Male (163) and Female (159)<br>
#         Median Maximum Heart Rate Cholesterol without Heart Disease - Male (141) and Female (145.5)**<br>

# ## Exercise Induced Angina (exercise_angina)

# In[ ]:


# Plot on basis of Target

sns.countplot(x='exercise_angina', hue ='target', data = data, palette="bwr")
plt.title('Exercise Induced Angina')
plt.show()


# In[ ]:


# Count of Male and Female patients with and withour Exercise Induced Angina

sns.countplot(x = 'exercise_angina', hue ='sex', data = data, palette="bwr")
plt.title('Exercise Induced Angina')
plt.show()


# **OBSERVATIONS**
# 
# **(I) Excercise Induced Angina is not very common in patients with Heart Diseases.**
# 
# **(II) There are more patients without Exercise Induced Angina out of which male are more in number and this is even true for patients with Exercise Induced Angina.**

# ## ST depression

# In[ ]:


print(f"Maximum Depression : {data['st_depression'].max()}")
print(f"Minimum Depression : {data['st_depression'].min()}")


# In[ ]:


# Let's look at the distribution of ST Depression

hist_data = [data['st_depression']]
group_labels = ['st_depression'] 

colors = ['#808000']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data, group_labels=group_labels, colors=colors,
                         bin_size=0.2, show_rug=False)

# Add title
fig.update_layout(width=700, title_text='ST Depression Distribution')
fig.show()


# In[ ]:


# Box plot on basis of sex

fig = px.box(data, x="target", y="st_depression", points="all", color='sex')
fig.update_layout(title_text='ST Depression Vs Target')
fig.show()


# **OBSERVATIONS**
# 
# **(I) The distribution of st_depression does not follow normal distribution.**<br><br>
# **(II) Non-Heart Disease Patients seem to have high Depression.**<br><br>
# **(III) Female Patients doesn't matter whether there is disease or not, seems to have high st_depression as compared to Male.** 

# ## Peak exercise ST segment (st_slope)

# In[ ]:


# The slope of the peak exercise ST segment

sns.countplot(hue='st_slope',x ='target',data = data, palette="winter_r")
plt.title('Slope of the peak exercise ST segment')
plt.show()


# In[ ]:


# the slope of the peak exercise ST segment

sns.countplot(hue='st_slope',x ='sex',data = data, palette="winter_r")
plt.title('Slope of the peak exercise ST segment')
plt.show()


# **OBSERVATION**
# 
# **Most of the patients either male or female mostly have a heart disease have a slope = 2 followed by slope = 1**

# ## Number of major vessels

# In[ ]:


# Number of major vessels (0-4)

sns.countplot(hue='num_major_vessels',x ='target',data = data, palette="rainbow_r")
plt.title('Number of major vessels (0-4) colored by flourosopy')
plt.show()


# **OBSERVATIONS**
# 
# **There are large number of Heart Patients with vessel=0 as compared to others whereas there are almost same number of Non-Heart Patients for vessels 0 and 1. There are very less number of patients with vessel = (2,3 or 4)**

# ## 	Thalassemia

# In[ ]:


# Thalassemia types based on target

sns.countplot(hue='thalassemia',x ='target',data = data, palette="gist_ncar")
plt.title('Thalassmia')
plt.show()


# In[ ]:


# Thalassemia types based on Chest Pain Type

plt.figure(figsize=(10,5))
sns.countplot(x="chest_pain_type", hue="thalassemia", data=data, palette="YlOrRd_r")


# In[ ]:


# Thalassemia types based on st_slope

plt.figure(figsize=(10,5))
sns.countplot(x="st_slope", hue="thalassemia", data=data, palette="YlOrRd_r")


# In[ ]:


# Thalassemia types based on sex

sns.countplot(hue='thalassemia',x ='sex',data = data, palette="gist_ncar")
plt.title('Thalassmia')
plt.show()


# **OBSERVATIONS**
# 
# **(I) Almost every patient, having heart disease or not, has Thalassemia.**<br><br>
# **(II) Fixed Defect Thalassemia is more common in Patients with Heart Disease whereas Reversable Defect Thalassemia is more common in Patients without Heart Disease.**<br><br>
# **(III) There are more number of patients with Fixed Defect Thalassemia in any type of chest pains whereas patients without chest pain and Reversable Defect Thalassemia are very common.**<br><br>
# **(IV) Fixed Defect Thalassemia is very common among Male and Female Patients but there are very large number of Reversable Defect Thalassemia Male Patients as compared to Female Patients.**<br><br>
# **(V) Among Horizontal Slope Patients - Fixed Defect Thalassemia is more common as compared to others whereas Upsloping patients have more number of Reversable Defect Thalassemia Patients.**

# ### Check out the Preprocessing and Modelling part [here](https://www.kaggle.com/nishkarshtripathi/predicting-like-a-boss-91-8-accuracy).
# 
# 
# # THANKS FOR READING!
