#!/usr/bin/env python
# coding: utf-8

# ## COVID-19 Analysis

# The dataset used in this notebook (Covid-19_dataset.csv) is same as the COVID19_line_list_data.csv dataset taken from https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset, but the only difference is that in our dataset death and recovered features are encoded as (0 or 1) and not in form of dates as in the later dataset.

# ## Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics
from sklearn.metrics import classification_report


# ## Obtaining Data

# In[ ]:


data = pd.DataFrame(pd.read_csv("../input/covid19-refined-dataset/Covid-19_dataset.csv"))
data.head(2)


# In[ ]:


data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
data.head(2)


# In[ ]:


data.shape


# ## Cleaning Data

# In[ ]:


data.drop(data.columns[[0,1,3,8,9,10,11,12,17,18,19]], axis = 1, inplace = True)
data['reporting_date'] = pd.to_datetime(data.reporting_date)

data.head()


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


print('Number of Null values in Columns')
data.isnull().sum()


# ### Deleting Rows with Null Values in Specific columns

# In[ ]:


refined_data = data.dropna(subset=['gender', 'age', 'from_wuhan'])


# In[ ]:


print('Number of Null values in Columns')
refined_data.isnull().sum()


# In[ ]:


refined_data.head(5)


# In[ ]:


refined_data.shape


# In[ ]:


refined_data.columns


# In[ ]:


refined_data.describe()


# In[ ]:


refined_data.info()


# ## Encoding Data 

# #### SPECIFIC LOCATION IN A COUNTRY

# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
refined_data[refined_data.columns[1]] = labelencoder.fit_transform(refined_data[refined_data.columns[1]])


# #### COUNTRY

# In[ ]:


labelencoder = LabelEncoder()
refined_data[refined_data.columns[2]] = labelencoder.fit_transform(refined_data[refined_data.columns[2]])


# #### GENDER

# In[ ]:


labelencoder = LabelEncoder()
refined_data[refined_data.columns[3]] = labelencoder.fit_transform(refined_data[refined_data.columns[3]])


# In[ ]:


refined_data.head(5)


# In[ ]:


refined_data.info()


# ## Feature Selection

# ### Univariate Selection

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = refined_data.iloc[:,1:7]  #independent columns
y = refined_data.iloc[:,7]    #target column i.e Death


# In[ ]:


#apply SelectKBest class to extract top 5 best features
bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X,y)


# In[ ]:


dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)


# In[ ]:


#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
featureScores


# In[ ]:


print(featureScores.nlargest(6,'Score'))


# In[ ]:


labels = 'age', 'location', 'country', 'from_wuhan', 'visiting_wuhan', 'gender'
sizes = [444.595027, 407.528544, 81.670406, 53.350788, 9.143794, 3.567871]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=190)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# Therefore we can conclude that Deaths are mostly effected by the 'Location' and 'Age' of the Patient as those features have the highest Feature Score.

# .
# 

# ### Feature Importance

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)


# In[ ]:


print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers


# In[ ]:


#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# We can conclude that 'Gender' of the patient or Whether they have 'visited Wuhan' has very little impact on their chance of Death.

# ## Visualization

# In[ ]:


refined_data.sample(5)


# In[ ]:


refined_data.shape


# In[ ]:


y= refined_data["death"]
y


# In[ ]:


x= refined_data["age"]
y= refined_data["death"]
plt.bar(x,y)
plt.title("Number of Patients Died based on their age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# We can conclude that Pateints with age more than 55 have high chance of death because of their weaker immune system.

# In[ ]:


x= refined_data["age"]
y= refined_data["recovered"]
plt.bar(x,y)
plt.title("Number of Patients Recovered based on their age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# Patients with age below 55 have high chance of recovery.

# In[ ]:


print("Current count of patients:", refined_data.shape[0])
print("Recovered:",len(refined_data[refined_data.recovered == 1]))
print("Dead:",len(refined_data[refined_data.death == 1]))
print("Number of Patients receiving treatment:", refined_data.shape[0] - len(refined_data[refined_data.recovered == 1]) 
      - len(refined_data[refined_data.death == 1]))

y = np.array([len(refined_data[refined_data.recovered == 1]),len(refined_data[refined_data.death == 1])])
x = ["Recovered","Dead"]
plt.bar(x,y)
plt.title("Patient Status")
plt.xlabel("Patients")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


print("Male:",len(refined_data[refined_data.gender == 1][refined_data.death == 1]))
print("Female:",len(refined_data[refined_data.gender == 0][refined_data.death == 1]))

y = np.array([len(refined_data[refined_data.gender == 1][refined_data.death == 1]),
              len(refined_data[refined_data.gender == 0][refined_data.death == 1])])
x = ["Male","Female"]
plt.bar(x,y)
plt.title("Number of Patients Died")
plt.xlabel("Patients")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


print("Male:",len(refined_data[refined_data.gender == 1][refined_data.recovered == 1]))
print("Female:",len(refined_data[refined_data.gender == 0][refined_data.recovered == 1]))

y = np.array([len(refined_data[refined_data.gender == 1][refined_data.recovered == 1]),
              len(refined_data[refined_data.gender == 0][refined_data.recovered == 1])])
x = ["Male","Female"]
plt.bar(x,y)
plt.title("Number of Patients Recovered")
plt.xlabel("Patients")
plt.ylabel("Frequency")
plt.show()


# As more Males die as well as recover with comparison to Females, therefore this data is redundant and we can not predict pateint's health based on thier gender.

# In[ ]:


print("From Wuhan :",len(refined_data[refined_data.from_wuhan == 1][refined_data.death == 1]))
print("Not From Wuhan:",len(refined_data[refined_data.from_wuhan == 0][refined_data.death == 1]))

y = np.array([len(refined_data[refined_data.from_wuhan == 1][refined_data.death == 1]),
              len(refined_data[refined_data.from_wuhan == 0][refined_data.death == 1])])
x = ["From Wuhan","Not From Wuhan"]
plt.bar(x,y)
plt.title("Number of Patients Died")
plt.xlabel("Patients")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


print("From Wuhan :",len(refined_data[refined_data.from_wuhan == 1][refined_data.recovered == 1]))
print("Not From Wuhan:",len(refined_data[refined_data.from_wuhan == 0][refined_data.recovered == 1]))

y = np.array([len(refined_data[refined_data.from_wuhan == 1][refined_data.recovered == 1]),
              len(refined_data[refined_data.from_wuhan == 0][refined_data.recovered == 1])])
x = ["From Wuhan","Not From Wuhan"]
plt.bar(x,y)
plt.title("Number of Patients Recovered")
plt.xlabel("Patients")
plt.ylabel("Frequency")
plt.show()


# Therefore patients from Wuhan have high chance of death then recovery as compared to patients not from Wuhan.

# In[ ]:


group = data.groupby('country').size()
group.head()


# In[ ]:


x= ['Afghanistan','Algeria','Australia','Austria','Bahrain','Belgium',
'Cambodia','Canada','China','Croatia','Egypt','Finland','France',
'Germany','Hong Kong','India','Iran','Israel','Italy','Japan',
'Kuwait','Lebanon','Malaysia','Nepal','Philippines','Russia',
'Singapore','South Korea','Spain','Sri Lanka','Sweden','Switzerland',
'Taiwan','Thailand','UAE','UK','USA', 'Vietnam']
y= group

plt.title("Patients identified at different locations")
plt.xlabel("Location")
plt.ylabel("Number of Covid Patients")
plt.xticks(rotation=90)
plt.bar(x,y)


# Certain locations have more patients as compared to others.

# In[ ]:


# WORLD MAP SHOWING LOCATIONS WITH COVID-19 PATIENTS
import plotly.express as px
fig = px.choropleth(data, locations="country", locationmode='country names', 
                    hover_name="country", title='PATIENTS IDENTIFIED AT DIFFERENT LOCATIONS', 
                    color_continuous_scale=px.colors.sequential.Magenta)
fig.update(layout_coloraxis_showscale=False)
fig.show()


# In[ ]:


# Over the time

fig = px.choropleth(data, locations="country", locationmode='country names', 
                    hover_name="country", animation_frame=data["reporting_date"].dt.strftime('%Y-%m-%d'),
                    title='OVER THE TIME PATIENTS IDENTIFIED BASED ON THEIR LOCATION', 
                    color_continuous_scale=px.colors.sequential.Magenta)
fig.update(layout_coloraxis_showscale=False)
fig.show()


# ## Prediction

# #### Predicting DEATH of a Patient

# In[ ]:


X = refined_data[refined_data.columns[1:7]] #(location, country, gender, age, visiting wuhan, from wuhan)
y = refined_data[refined_data.columns[[7]]] #death


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

reg=LogisticRegression()
reg.fit(X_train,y_train)


# In[ ]:


reg.score(X_train,y_train)


# In[ ]:


pdt = reg.predict(X_test)
pdt


# In[ ]:


#CONFUSION MATRIX
cm = metrics.confusion_matrix(y_test, pdt)
print(cm)


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test,pdt))
rms


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, pdt))


# #### Predicting RECOVERY of a Patient

# In[ ]:


X = refined_data[refined_data.columns[1:7]] #(location, country, gender, age, visiting wuhan, from wuhan)
y = refined_data[refined_data.columns[[8]]] #recovered


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

reg=LogisticRegression()
reg.fit(X_train,y_train)


# In[ ]:


reg.score(X_train,y_train)


# In[ ]:


pdt = reg.predict(X_test)
pdt


# In[ ]:


#CONFUSION MATRIX
cm = metrics.confusion_matrix(y_test, pdt)
print(cm)


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test,pdt))
rms


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, pdt))


# ## STAY SAFE! STAY AT HOME!
