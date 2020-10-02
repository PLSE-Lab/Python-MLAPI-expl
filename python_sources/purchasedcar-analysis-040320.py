#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Dataset Load
url = '../input/prediction-of-purchased-car/Social_Network_Ads.csv'
data = pd.read_csv(url, header='infer')


# **Exploration**

# In[ ]:


data.shape


# In[ ]:


#checking for null or missing values
data.isna().sum()


# **Pre-Processing**

# In[ ]:


#dropping the User Id column
data = data.drop(columns='User ID',axis=1)


# In[ ]:


#Function to define category name for Purchased column

def pur_cat(pur_code):
    if pur_code == 0:
        return 'Not Purchased'
    else:
        return 'Purchased'
    
#Applying the function to Purchased Column
data['Purchased'] = data['Purchased'].apply(pur_cat)


# In[ ]:


data.head()


# > **Data Visualisation**

# In[ ]:


#Estimated Salary vs Gender -- Purchase Stats
sns.set(style="darkgrid")
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.swarmplot (x='Gender', y='EstimatedSalary', data=data, hue = 'Purchased',palette="Set2")
plt.title('Estimated Salary vs Gender -- Purchase Stats')
plt.ylabel('Estimated Salary')
plt.xlabel('Gender')
ax.legend(fancybox=True, shadow=True )


# In[ ]:


#Estimated Age vs Gender -- Purchase Stats
sns.set(style="darkgrid")
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.swarmplot (x='Gender', y='Age', data=data, hue = 'Purchased',palette="Set2")
plt.title('Estimated Salary vs Gender -- Purchase Stats')
plt.ylabel('Estimated Salary')
plt.xlabel('Gender')
ax.legend(fancybox=True, shadow=True )


# In[ ]:


#Gender vs Purchase -- Average Age Heatmap
AvgAge_PTable = data.pivot_table(values='Age', index='Gender', columns='Purchased', aggfunc=np.mean)

# Using seaborn heatmap
plt.figure(figsize=(6,6))
plt.title('Gender vs Purchase -- Average Age Heatmap', fontsize=14)
plt.tick_params(labelsize=10)
sns.heatmap(AvgAge_PTable.round(), cmap='icefire', linecolor='grey',linewidths=0.1, cbar=False, annot=True, fmt=".0f")


# In[ ]:


#Gender vs Purchase -- Average Estimated Salary Heatmap
AvgSal_PTable = data.pivot_table(values='EstimatedSalary', index='Gender', columns='Purchased', aggfunc=np.mean)

# Using seaborn heatmap
plt.figure(figsize=(6,6))
plt.title('Gender vs Purchase -- Average Estimated Salary Heatmap', fontsize=14)
plt.tick_params(labelsize=10)
sns.heatmap(AvgSal_PTable.round(), cmap='icefire', linecolor='grey',linewidths=0.1, cbar=False, annot=True, fmt=".0f")


# **Modeling & Prediction**

# Under modeling & prediction, I shall compare multiple Classifier Models from SKLEARN library and try to find the model with highest accuracy. The model that gives the higest accuracy will then be used for making predictions on unseen data

# In[ ]:


# --- Importing ML libraries ---
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 

#Metrics Libraries
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

#ML Classifier Algorithm Libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# **Converting the Gender category to numerical code using Label Encoding**

# In[ ]:


lae = LabelEncoder()

data['Gender'] = lae.fit_transform(data['Gender'])


# In[ ]:


#Backup of the original dataset
data_backup = data.copy()


# Since there is no seperate test dataset, I will slice the top 10 rows of the existing dataset and use them as "unseen" dataset for validating the model at the end.

# In[ ]:


#creating unseen dataframe
unseen_df = data.iloc[:10,:]


# In[ ]:


unseen_df.head(10)


# In[ ]:


#Dropping the top 10 rows from the original dataset
data.drop(data.index[:10],inplace=True)

# Re-indexing the original dataset
data.reset_index(inplace=True, col_level=1, drop=True)


# In[ ]:


data.head(10)


# In[ ]:


#Feature & Target Selection
features = ['Gender', 'Age', 'EstimatedSalary']
target = ['Purchased']

# Feature& Target  Dataset
X = data[features]
y = data[target]


# In[ ]:


#Dataset Split  [train = 90%, test = 10%]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0) 

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# **Model Creation & Evaluation**

# In[ ]:


# -- Building Model List --
models = []
models.append(('CART', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RFC', RandomForestClassifier()))


# In[ ]:


# -- Model Evaluation --
model_results = []
model_names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=None, shuffle=False)
    cross_val_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    model_results.append(cross_val_results)
    model_names.append(name)
    print(name, ":--", "Mean Accuracy =", '{:.2%}'.format(cross_val_results.mean()), 
                       "Standard Deviation Accuracy =", '{:.2%}'.format(cross_val_results.std())
         )


# In[ ]:


# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(model_results)
ax.set_xticklabels(model_names)
plt.show()


# **As you can observe, KNN Model gives us the highest accuracy of 91.11%**

# **KNN Model Predictions & Metrics****

# In[ ]:


#Instantiating KNN Model
knc = KNeighborsClassifier(n_neighbors=3, metric='minkowski')


# In[ ]:


#Training the KNN Model
knc.fit(X_train, y_train)


# In[ ]:


#Making a prediction
y_pred = knc.predict(X_test)

# -- Calculating Metrics 
print("Trained Model Accuracy Score - KNN Model: ",'{:.2%}'.format(accuracy_score(y_test,y_pred)) )


# **Making Prediction on Unseen Data**

# In[ ]:


unseen_df.head(10)


# In[ ]:


#Making prediction on unseen data
unseenData_pred = knc.predict(unseen_df.iloc[:,0:3])

#Appending the prediction to the unseen dataset
unseen_df['Pred_Purchased'] = unseenData_pred


# In[ ]:


unseen_df.head(10)


# In[ ]:


# -- Calculating Metrics 
print("Trained KNN Model Accuracy Score on unseen data: ",'{:.2%}'.format(accuracy_score(unseen_df['Purchased'],unseenData_pred)) )


# I believe that the low accuracy on the unseen data is due to the fact that the model has been trained on less number of data records i.e. around 360. 
