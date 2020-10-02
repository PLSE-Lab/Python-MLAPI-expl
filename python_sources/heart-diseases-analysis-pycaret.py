#!/usr/bin/env python
# coding: utf-8

# **Heart Diseases Analysis**

# **Importing Libraries**

# In[ ]:


get_ipython().system('pip3 install bubbly')
#!pip3 install pandas-profiling
#!pip3 install shap
get_ipython().system('pip3 install pycaret')


# In[ ]:


# for basic operations
import numpy as np
import pandas as pd
#import dtale
# for data visualizations
import matplotlib.pyplot as plt
import seaborn as sns
# for advanced visualizations 
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from bubbly.bubbly import bubbleplot
# for providing path
import os
# for model 
import pycaret

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# reading the data
dataset = pd.read_csv('../input/heart-disease-uci/heart.csv')
data = dataset.copy()
# getting the shape
data.shape


# In[ ]:


# getting info about data
data.info()


# **Data Description**

# age: The person's age in years
# 
# 
# sex: The person's sex (1 = male, 0 = female)
# 
# 
# cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
# 
# 
# trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
# 
# 
# chol: The person's cholesterol measurement in mg/dl
# 
# 
# fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
# 
# 
# restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
# 
# 
# thalach: The person's maximum heart rate achieved
# 
# 
# exang: Exercise induced angina (1 = yes; 0 = no)
# 
# 
# oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
# 
# 
# slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
# 
# 
# ca: The number of major vessels (0-3)
# 
# 
# thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
# 
# 
# target: Heart disease (0 = no, 1 = yes)
# 

# In[ ]:


# reading the head of the data
data.head()


# In[ ]:


# describing the data
data.describe()


# ### EDA and Data Visualization

# In[ ]:


# making a heat map
plt.rcParams['figure.figsize'] = (20, 15)
plt.style.use('ggplot')
sns.heatmap(data.corr(), annot = True)
plt.title('Heatmap for the Dataset', fontsize = 20)
plt.show()


# > The above heat map is to show the correlations amongst the different attributes of the given dataset. The above Heat Map shows that almost all of the features/attributes given in the dataset are very less correlated with each other. This implies we must include all of the features, as we can only eliminate those features where the correlation of two or more features are very high.

# In[ ]:


# checking the distribution of age among the patients
from scipy.stats import norm
sns.distplot(data['age'], fit=norm, kde=False)
plt.title('Distribution of Age', fontsize = 10)
plt.show()


# > The above Distribution plot shows the distribution of Age amongst all of the entries in the dataset about the heart patients. The Graph suggests that the highest number of people suffering from heart diseases are in the age group of 55-65 years. The patients in the age group 20-30 are very less likely to suffer from heart diseases.
# >> As we know that the number of people in the age group 65-80 has a very low population, hence distribution is also less. we might have to opt for other plots to investigate further and get some more intuitive results.

# In[ ]:


# Checking Target
sns.countplot(data['target'])
plt.xlabel(" Target")
plt.ylabel("Count")
plt.show()


# The dataset is quite balanced with almost equal number of Positive and Negative Classes. The Positive Class says that the patient is suffering from the disease and the Negative class says that the patient is not suffering from the disease.

# In[ ]:


# plotting a donut chart for visualizing each of the recruitment channel's share
size = data['sex'].value_counts()
colors = ['lightblue', 'lightgreen']
labels = "Male", "Female"
explode = [0, 0.01]
my_circle = plt.Circle((0, 0), 0.7, color = 'white')
plt.pie(size, colors = colors, labels = labels, shadow = True, explode = explode, autopct = '%.2f%%')
plt.title('Distribution of Gender', fontsize = 15)
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.legend()
plt.show()


# > The above Pie chart, whhich shows us the distribution of Gender in the Heart diseases. By looking at the plot, we can **assume** that Males are two times more likely to suffer from heart diseases in comparison to females.
# >> According to our study, From all the Entries in our dataset 68% of the patients are men whereas only 32% are women. More number of men took participation in heart disease check ups.

# In[ ]:


# Checking chest type
sns.countplot(data['cp'])
plt.xlabel(" Chest type")
plt.ylabel("Count")
plt.show()


# In[ ]:


# tresbps vs target
sns.boxplot(data['target'], data['trestbps'], palette = 'viridis')
plt.title('Relation of tresbps with target', fontsize = 20)
plt.show()


# > tresbps: Resting Blood Pressure, The above Bivariate plot between tresbps(the resting blood pressure of a patient), and the target which says that whether the patient is suffering from the heart disease or not. The plot clearly suggests that the patients who are most likely to not suffer from the disease have a slighly greater blood pressure than the patients who have heart diseases.

# In[ ]:


# cholestrol vs target
sns.violinplot(data['target'], data['chol'], palette = 'colorblind')
plt.title('Relation of Cholestrol with Target', fontsize = 20, fontweight = 30)
plt.show()


# > The above Bivariate plot between cholestrol levels and target suggests that the Patients likely to suffer from heart diseases are having higher cholestrol levels in comparison to the patients with target 0(likely to not suffer from the heart diseases.
# >> Hence, we can infer from the above plot that the cholestrol levels plays an important role in determining heart diseases. We all must keep our cholestrol levels in control as possible.

# In[ ]:


# Resting electrocardiographic measurement vs target
  
dat = pd.crosstab(data['target'], data['restecg']) 
dat.div(dat.sum(1).astype(float), axis = 0).plot(kind = 'bar', 
                                                 stacked = False, 
                                                 color = plt.cm.rainbow(np.linspace(0, 1, 4)))
plt.title('Relation of ECG measurement with Target', fontsize = 20, fontweight = 30)
plt.show()


# > The above plot is column bar chart representing target vs ECG Measurements(Electro Cardio Gram), The above plot shows that the more number of patients not likely to suffer from heart diseases are having restscg value 0 whereas more number of people have restecg value 1 in case of more likelihood of suffering from a heart disease.

# > This Heat Map, between Target and Maximum Heart Rate shows that the patients who are likely to suffer from heart diseases are having higher maximum heart rates whereas the patients who are not likely to suffer from any heart diseases are having lower maximum heart rates.
# >> This implies it is very important to keep our heart rates low, to keep ourselves healthy and safe from any dangerous heart diseases.

# In[ ]:


# slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
# checking the relation between slope and target

plt.rcParams['figure.figsize'] = (15, 9)
sns.boxenplot(data['target'], data['slope'], palette = 'copper')
plt.title('Relation between Peak Exercise and Target', fontsize = 20, fontweight = 30)
plt.show()


# > Slope : 0 refers to upsloping, 1 refers to flat Exercises pattern.
# >>This plot clearly shows that the patients who are not likely to suffer from any heart diseases are mostly having value 1 means upsloping, whereas very few people suffering from heart diseases have upsloping pattern in exercises.
# >> Also, Flat Exercises are mostly seen in the cases of Patients who are more likely to suffer from heart diseases.

# In[ ]:


#ca: The number of major vessels (0-3)

sns.boxenplot(data['target'], data['ca'], palette = 'Reds')
plt.title('Relation between no. of major vessels and target', fontsize = 20, fontweight = 30)
plt.show()


# > The above Bivariate plot between Target and Number of Major Vessels, shows that the patients who are more likely to suffer from Heart diseases are having high values of Major Vessels wheras the patiets who are very less likely to suffer from any kind of heart diseases have very low values of Major Vessels.
# >> Hence, It is also helpful in determining the heart diseases, the more the number of vessels, the more is the chance of suffering from heart diseases.

# In[ ]:


# relation between age and target

plt.rcParams['figure.figsize'] = (15, 9)
sns.swarmplot(data['target'], data['age'], palette = 'winter', size = 10)
plt.title('Relation of Age and target', fontsize = 20, fontweight = 30)
plt.show()


# > From the above Swarm plot between the target and the age of the patients, we are not able to find any clue or pattern, so age is not a very good attribute to determine the heart disease of a patient as a patient of heart diseases range from 30-70, whereas it is not important that all of the people lying in that same age group are bound to suffer from the heart diseases.

# In[ ]:


# relation between sex and target
sns.boxenplot(data['target'], data['sex'], palette = 'Set3')
plt.title('Relation of Sex and target', fontsize = 20, fontweight = 30)
plt.show()


# In[ ]:


# checking the relation between 
#thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)

sns.boxenplot(data['target'], data['thal'], palette = 'magma')
plt.title('Relation between Target and Blood disorder-Thalessemia', fontsize = 20, fontweight = 30)
plt.show()


# >In the above Boxen plot between Target and a Blood disorder called Thalessemia, It can be easily inferred that the patients suffering from heart diseases have low chances of also suffering from thalessemia in comparison to the patients who are less likely to suffer from the heart diseases. Hence, It is also a good feature to classify heart diseases.

# In[ ]:


# target vs chol and hue = thalach

plt.scatter(x = data['target'], y = data['chol'], s = data['thalach']*100, color = 'yellow')
plt.title('Relation of target with cholestrol and thalessemia', fontsize = 20, fontweight = 30)
plt.show()


# In[ ]:


# multi-variate analysis
sns.boxplot(x = data['target'], y = data['trestbps'], hue = data['sex'], palette = 'rainbow')
plt.title('Checking relation of tresbps with genders to target', fontsize = 20, fontweight = 30)
plt.show()


# > In the above Box plot between Target and tresbps wrt Gender, shows that Women have higher tresbps than men in case of not suffering from any heart diseases, whereas men and women have almost equal tresbps in case of suffering from a heart diseases. Also, In case of suffering from heart diseases, patients have a slightly lower tresbps in comparison to the patients who are not suffering from heart diseases.

# ## Preparing data to Model

# In[ ]:


#check the shape of data
dataset.shape


# In[ ]:


dataset.head()


# In[ ]:


data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index).reset_index(drop=True)
data.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


# ### Setting up Environment in PyCaret

# In[ ]:


from pycaret.classification import *

exp_clf101 = setup(data = data, target = 'target', session_id=123,
                   normalize = True, 
                   transformation = True, 
                   ignore_low_variance = True,
                   remove_multicollinearity = True, 
                   multicollinearity_threshold = 0.95)


# ### Comparing All Models

# In[ ]:


compare_models()


# ### Creating a Model
# 
# Here, we will choose Logistic Regression model, which presented the best AUC score

# In[ ]:


lr = create_model('lr', fold = 10, round = 3)


# In[ ]:


#trained model object is stored in the variable 'lr'. 
print(lr)


# ### Tuning Model

# In[ ]:


tuned_lr = tune_model('lr', fold = 10, round = 3, optimize = 'AUC')


# In[ ]:


# Checking hyperparameters
plot_model(tuned_lr, plot = 'parameter')


# ### Model visualization

# In[ ]:


#AUC plot
plot_model(tuned_lr, plot = 'auc')


# In[ ]:


# Precision-Recall Curve
plot_model(tuned_lr, plot = 'pr')


# In[ ]:


# Feature Importance 
plot_model(tuned_lr, plot='feature')


# In[ ]:


# Confusion Matrix
plot_model(tuned_lr, plot = 'confusion_matrix')


# In[ ]:


evaluate_model(tuned_lr)


# ### Prediction

# In[ ]:


predict_model(tuned_lr);


# ### Finalize Model for Deployment

# In[ ]:


final_lr = finalize_model(tuned_lr)
print(final_lr)


# ### Predicting on Unseen Data

# In[ ]:


unseen_predictions = predict_model(final_lr, data=data_unseen)
unseen_predictions.head()


# ### Saving the model

# In[ ]:


save_model(final_lr,'Final_LR_Model_Jun2020')


# ### Loading the saved model

# In[ ]:


saved_final_lr = load_model('Final_LR_Model_Jun2020')


# In[ ]:


new_prediction = predict_model(saved_final_lr, data=data_unseen)
new_prediction.head()


# Next steps: 
# - to build ensemble models
