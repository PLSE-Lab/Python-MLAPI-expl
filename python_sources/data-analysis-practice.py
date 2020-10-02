#!/usr/bin/env python
# coding: utf-8

# <h2> Description of Pima Diabetes dataset </h2>
# 
# <p> Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage. ADAP is an adaptive learning routine that generates and executes digital analogy of perceptron-like devices. It is a unique algorithm; see the paper for details.</p>
# 
# <h3> Attribute Information:</h3>
# <ul>
#     <li> Number of times pregnant </li>
#     <li> Plasma glucose concentration a 2 hours in an oral glucose tolerance test </li>
#     <li> Diastolic blood pressure (mm Hg) </li>
#     <li> Triceps skin fold thickness (mm) </li>
#     <li> 2-Hour serum insulin (mu U/ml) </li>
#     <li> Body mass index (weight in kg/(height in m)^2) </li> 
#     <li> Diabetes pedigree function </li>
#     <li> Age (years) </li>
#     <li> Class variable (0 or 1) ** </li>
# </ul>
# 
# <div class="alert alert-info">
#   <h2><strong>Important Note: </strong></h2>
#   <p> There are zeros in places where they are biologically impossible, such as the blood pressure attribute. It seems very likely that zero values encode missing data. However, since the dataset donors made no such statement, every individual are allowed to make there best judgement and state assumptions. </p>
# </div>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# <h1>Data Description </h1>
# <p>Load the data and take a first look at the values</p>

# In[ ]:


raw_data = pd.read_csv('../input/diabetes.csv')
raw_data.head()


# In[ ]:


raw_data.describe()


# In[ ]:


raw_data.info()


# <h2> Five point Summary for the variables</h2> 

# In[ ]:


fig, axes = plt.subplots(2, 4, figsize=(20,10))

sns.boxplot(  y='Pregnancies', data=raw_data, orient='v', ax=axes[0, 0])
sns.boxplot(  y='Glucose', data=raw_data,  orient='v', ax=axes[0, 1])
sns.boxplot(  y='BloodPressure', data=raw_data, orient='v', ax=axes[0,2])
sns.boxplot(  y='SkinThickness', data=raw_data, orient='v', ax=axes[0,3])
sns.boxplot(  y='Insulin', data=raw_data, orient='v', ax=axes[1,0])
sns.boxplot(  y='BMI', data=raw_data, orient='v', ax=axes[1,1])
sns.boxplot(  y='DiabetesPedigreeFunction', data=raw_data, orient='v', ax=axes[1,2])
sns.boxplot(  y='Age', data=raw_data, orient='v', ax=axes[1,3])


# In[ ]:


raw_data.hist(figsize=(18, 9))


# <p> From the above visualisation it can be clearly seen that there are outliers present on both the upper and lower range. </p>
# <p>For example, its highly unlikely to have been pregnant 17 times but the mean is slightly above 3 which shows the normal behaviour of the sample</p>

# In[ ]:


# Finding the number of Zeros per columns
for col in raw_data:
    print('{:>25}:{:>5}'.format(col, raw_data[col].loc[raw_data[col] == 0].count()))


# In[ ]:


# dataset with non zero values of critical attributes
# ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'Age']

non_zero_data = raw_data.loc[(raw_data['Glucose'] != 0) & (raw_data['BloodPressure'] != 0) 
                             & (raw_data['SkinThickness'] != 0) & (raw_data['BMI'] != 0)]
corr = non_zero_data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(12,6))
sns.heatmap(corr, mask=mask, annot=True, cmap='plasma',vmin=-1,vmax=1)


# The count of Zeros in the columns like `Glucose` and `BloodPressure` is low we can ignore these but for `SkinThickness` its quite high(227 out of 768). we might have to find a way to fill these. 
# 
# 
# `SkinThickness` have a fairly high correlation with the `BMI`, one possible way of filling this value could be based on the `BMI` column
# 
# other correlations are normal, 
# `Glucose` and `Insulin` correlation is fine
# `Age` and `Pregnancies` is obvious correlations
# 
# Overall if we see `SkinThickness`, `Insulin`, `Glucose`, `BMI` and `Age` have a fairly high linear correlation 

# <h1>Data Exploration</h1>

# In[ ]:


raw_data.groupby('Outcome')['Outcome'].count()


# In[ ]:


plt.figure(figsize=(8,4))
ax = sns.countplot(raw_data['Outcome'])
plt.title('Distribution of OutCome')
plt.xlabel('Outcomes')
plt.ylabel('Frequency')


# One by One will try to explore which attributes are affecting the outcome

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(18, 6))
sns.boxplot(x='Outcome', y='Pregnancies', data=raw_data, ax=axes[0])
sns.countplot(raw_data['Pregnancies'], hue = raw_data['Outcome'], ax=axes[1])


# From the boxplot, it is clear that the mean of the number of pregnancy is higher in diabetic patients. a similar observation can be seen in the histogram comparison, the number of women with more than 6 pregnancy are diabetic. 
# 
# Since the number of pregnancy correlation with age is more than 0.6, so it can't be said with confidence that pregnancy is the cause. 

# In[ ]:


raw_data['age_group'] = pd.cut(raw_data['Age'], range(0, 100, 10))
g = sns.catplot(x="age_group", y="Pregnancies", hue="Outcome",
               data=raw_data, kind="box"
              )
g.fig.set_figheight(4)
g.fig.set_figwidth(20)


# It seems diabetes is distributed normally across age group, the highest diabetic patients can be seen in the 30, 60 age group. 

# In[ ]:


raw_data['age_group'] = pd.cut(raw_data['Age'], range(0, 100, 10))
g = sns.catplot(x="age_group", y="BMI", hue="Outcome",
               data=raw_data, kind="box"
              )
g.fig.set_figheight(4)
g.fig.set_figwidth(20)


# The BMI doesn't seem to have any pattern across age groups, there are a few outliers because of missing values and some outliers are in upper range but not something major

# In[ ]:


raw_data['age_group'] = pd.cut(raw_data['Age'], range(0, 100, 10))
g = sns.catplot(x="age_group", y="SkinThickness", hue="Outcome",
               data=raw_data, kind="box"
              )
g.fig.set_figheight(4)
g.fig.set_figwidth(20)


# In[ ]:


raw_data['age_group'] = pd.cut(raw_data['Age'], range(0, 100, 10))
g = sns.catplot(x="age_group", y="Glucose", hue="Outcome",
               data=raw_data, kind="box"
              )
g.fig.set_figheight(4)
g.fig.set_figwidth(20)


# In[ ]:


raw_data.groupby('Outcome')['Glucose'].plot(kind='density', legend=True)


# Higher Glucose seems to have been influencing diabetes.

# In[ ]:


raw_data.groupby('Outcome')['SkinThickness'].plot(kind='density', legend=True)


# SkinThickness can be a major indicator of diabetes

# In[ ]:


raw_data.groupby('Outcome')['Pregnancies'].plot(kind='density', legend=True)


# In[ ]:


raw_data.groupby('Outcome')['BloodPressure'].plot(kind='density', legend=True)


# In[ ]:


raw_data.groupby('Outcome')['Insulin'].plot(kind='density', legend=True)


# In[ ]:


raw_data.groupby('Outcome')['DiabetesPedigreeFunction'].plot(kind='density', legend=True)


# In[ ]:


raw_data.groupby('Outcome')['Age'].plot(kind='density', legend=True)


# Based on these density visualisations of all the features against the Outcome, it is quite clear that the result is highly influenced with `DiabetesPedigreeFunction`, `Insulin`, `SkinThickness` and `Glucose`.

# <h2> Trying Logistic Model for predecting the Outcome based on all the features</h2>
# <div class="alert alert-info">
#   <h2><strong>Note: </strong></h2>
#   <p> we can use the age_group in place of age after doing one-hot-encoding but for now I am jsut using the default data i have got. Re-populating the zeros in the critical calumn with some kind of mean value based on age group, might provide better results</p>
# </div>
# 

# In[ ]:


#import the model and datsplit and crossvalidation utilities
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = raw_data[feature_names]
Y = raw_data.Outcome

lr = LogisticRegression(solver='liblinear')


# <h4>Spliting the data into test and train</h4>

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify = raw_data.Outcome, random_state=0)


# <h4>Train and predict with the model</h4> 

# In[ ]:


lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


# <h4>Checking the Accuracy</h4>

# In[ ]:


accuracy_score(y_test, y_pred)


# <h4>Cross-Validation</h4>
# 
# For cross-validation I am using K-Fold Algorithm, K-Flod, as the name suggests, splits the data into k equal batches and use one batch for test and rest of the batches for training, it runs the same K times each time selecting a different batch for testing.
# 
# One important thing to take care is its a CPU and Memory intensive task.

# In[ ]:


from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, random_state=10)
recall = cross_val_score(lr, X, Y, cv=kfold, scoring='recall').mean()
accuracy = cross_val_score(lr, X, Y, cv=kfold, scoring='accuracy').mean()
print('With {:0.2f}% Accuracy and {:0.2f}% true positive rate, the model is able to predict that a given patent have diabetes or not'.format(accuracy, recall))


# In[ ]:


cross_val_score(lr, X, Y, cv=kfold, scoring='precision').mean()

