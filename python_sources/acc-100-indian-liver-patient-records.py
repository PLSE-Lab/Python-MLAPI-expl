#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.svm import SVC


# In[ ]:


raw_data = pd.read_csv('../input/indian-liver-patient-records/indian_liver_patient.csv')
raw_data.head()


# In[ ]:


raw_data.isnull().sum()


# We have 4 Null values to deal with.

# In[ ]:


raw_data['Dataset'].value_counts()


# There are 416 liver patients and 167 non-liver patients. Since it is more important for us to identify the liver patients, we need an algorithm with a high recall score.

# # Feature Selection

# Two features that stand out are Age and Gender. Let's establish the importance of these.

# In[ ]:


sns.lmplot(data=raw_data, x='Age', y='Albumin');
sns.lmplot(data=raw_data, x='Age', y='Total_Protiens');
sns.lmplot(data=raw_data, x='Age', y='Albumin_and_Globulin_Ratio');


# We see that Albumin, Total_Protiens and Albumin_and_Globulin_Ratio are linearly dependent on Age.

# In[ ]:


sns.countplot(data=raw_data, x='Gender');


# The data has more Male records than female. We will have to observe them separately.

# In[ ]:


g = sns.FacetGrid(raw_data, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Age")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');


# Among both genders, most liver patients are aged between 30-60 years.

# In[ ]:


corr = raw_data.drop('Dataset',axis=1).corr()
plt.figure(figsize=(30, 30))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           cmap= 'coolwarm');


# There seems to be a high correlation between  
# Total_Bilirubin and Direct_Bilirubin  
# Aspartate_Aminotransferase and Alamine_Aminotransferase  
# Albumin and Total_Protiens  
#   
# We can remove some of these. We will remove the ones with low variation.

# In[ ]:


total_direct_bil = raw_data[["Total_Bilirubin", "Direct_Bilirubin"]]
sns.violinplot(data=total_direct_bil);


# Total_Bilirubin has a higher variation.

# In[ ]:


aspartate_alamine = raw_data[["Aspartate_Aminotransferase", "Alamine_Aminotransferase"]]
sns.violinplot(data=aspartate_alamine);


# Aspartate_Aminotransferase has a higher variation.

# In[ ]:


Total_Protiens_alb = raw_data[["Albumin", "Total_Protiens"]]
sns.violinplot(data=Total_Protiens_alb);


# Both have a substantially high variation.  
#   
# The features I will keep in the dataset are:  
# Age, Gender, Total_Bilirubin, Aspartate_Aminotransferase, Albumin, Total_Protiens, Albumin_and_Globulin_Ratio, Dataset

# In[ ]:


reduced_data = raw_data[["Age","Gender","Total_Bilirubin","Aspartate_Aminotransferase","Albumin", "Total_Protiens", "Albumin_and_Globulin_Ratio","Dataset"]]
reduced_data.head()


# # Normalization and Nulls

# In[ ]:


reduced_data[reduced_data['Albumin_and_Globulin_Ratio'].isnull()]


# In[ ]:


grouped = reduced_data.groupby(["Gender","Dataset"])
reduced_data['Albumin_and_Globulin_Ratio'] = grouped['Albumin_and_Globulin_Ratio'].transform(lambda x: x.fillna(x.mean()))


# We replace the Null values with group means to maintain the effects of Gender.

# In[ ]:


le = LabelEncoder()
reduced_data.Gender = le.fit_transform(reduced_data.Gender)
reduced_data.head()


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(reduced_data, reduced_data.Dataset, test_size=0.2)


# In[ ]:


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# # Machine Learning

# In[ ]:


clf = SVC(gamma='auto')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# using Support Vector Classifier we get a 100% accuracy.
