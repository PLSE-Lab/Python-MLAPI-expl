#!/usr/bin/env python
# coding: utf-8

# In this kernel I have performed Exploratory Data Analysis on the Heart Diseases UCI and tried to identify relationship between heart disease  and various other features. After EDA data pre-processing is done I have applied k-NN(k-Nearest Neighbors) method and Logistic Regression Algorithm to make the predictions.
# I will use various other algorithms for predictions in future and add them in this kernel.

# I hope you find this kernel helpful and some **<font color='red'>UPVOTES</font>** would be very much appreciated

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# ### **Importing required libraries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# ### **Loading the data**

# In[ ]:


df = pd.read_csv('../input/heart.csv')
df.head(3)


# The dataset contains the following features:<br>
# **1. age(in years)**<br>
# **2. sex:** (1 = male; 0 = female)<br>
# **3. cp:** chest pain type<br>
# **4. trestbps:** resting blood pressure (in mm Hg on admission to the hospital)<br>
# **5. chol:** serum cholestoral in mg/dl<br>
# **6. fbs:** (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)<br>
# **7. restecg:** resting electrocardiographic results<br>
# **8. thalach:** maximum heart rate achieved<br>
# **9. exang:** exercise induced angina (1 = yes; 0 = no)<br>
# **10. oldpeak**: ST depression induced by exercise relative to rest<br>
# **11. slope:** the slope of the peak exercise ST segment<br>
# **12. ca:** number of major vessels (0-3) colored by flourosopy<br>
# **13. thal:** 3 = normal; 6 = fixed defect; 7 = reversable defect<br>
# **14. target:** 1 or 0 <br>

# ### **Features of the data set**

# In[ ]:


df.info()


# **Dimensions of the dataset**

# In[ ]:


print('Number of rows in the dataset: ',df.shape[0])
print('Number of columns in the dataset: ',df.shape[1])


# **Checking for null values in the dataset**

# In[ ]:


df.isnull().sum()


# There are no null values in the dataset

# In[ ]:


df.describe()


# **The features described in the above data set are:**
# 
# **1. Count** tells us the number of NoN-empty rows in a feature.<br>
# 
# **2. Mean** tells us the mean value of that feature.<br>
# 
# **3. Std** tells us the Standard Deviation Value of that feature.<br>
# 
# **4. Min** tells us the minimum value of that feature.<br>
# 
# **5. 25%**, **50%**, and **75%** are the percentile/quartile of each features.<br>
# 
# **6. Max** tells us the maximum value of that feature.<br>
# 

# ### **Checking features of various attributes**

# #### **1. Sex**

# In[ ]:


male =len(df[df['sex'] == 1])
female = len(df[df['sex']== 0])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'Male','Female'
sizes = [male,female]
colors = ['skyblue', 'yellowgreen']
explode = (0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# #### **2. Chest Pain Type**

# In[ ]:


plt.figure(figsize=(8,6))

# Data to plot
labels = 'Chest Pain Type:0','Chest Pain Type:1','Chest Pain Type:2','Chest Pain Type:3'
sizes = [len(df[df['cp'] == 0]),len(df[df['cp'] == 1]),
         len(df[df['cp'] == 2]),
         len(df[df['cp'] == 3])]
colors = ['skyblue', 'yellowgreen','orange','gold']
explode = (0, 0,0,0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=180)
 
plt.axis('equal')
plt.show()


# #### **3. fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)**

# In[ ]:


plt.figure(figsize=(8,6))

# Data to plot
labels = 'fasting blood sugar < 120 mg/dl','fasting blood sugar > 120 mg/dl'
sizes = [len(df[df['fbs'] == 0]),len(df[df['cp'] == 1])]
colors = ['skyblue', 'yellowgreen','orange','gold']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=180)
 
plt.axis('equal')
plt.show()


# #### **4.exang: exercise induced angina (1 = yes; 0 = no)**

# In[ ]:


plt.figure(figsize=(8,6))

# Data to plot
labels = 'No','Yes'
sizes = [len(df[df['exang'] == 0]),len(df[df['exang'] == 1])]
colors = ['skyblue', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()


# ### **Exploratory Data Analysis**

# In[ ]:


sns.set_style('whitegrid')


# #### **1. Heatmap**

# In[ ]:


plt.figure(figsize=(14,8))
sns.heatmap(df.corr(), annot = True, cmap='coolwarm',linewidths=.1)
plt.show()


# #### **Plotting the distribution of various attribures**

# #### **1. thalach: maximum heart rate achieved**

# In[ ]:


sns.distplot(df['thalach'],kde=False,bins=30,color='violet')


# #### **2.chol: serum cholestoral in mg/dl **

# In[ ]:


sns.distplot(df['chol'],kde=False,bins=30,color='red')
plt.show()


# #### **3. trestbps: resting blood pressure (in mm Hg on admission to the hospital)**

# In[ ]:


sns.distplot(df['trestbps'],kde=False,bins=30,color='blue')
plt.show()


# #### **4. Number of people who have heart disease according to age **

# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x='age',data = df, hue = 'target',palette='GnBu')
plt.show()


# #### **5.Scatterplot for thalach vs. chol **

# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='chol',y='thalach',data=df,hue='target')
plt.show()


# #### **6.Scatterplot for thalach vs. trestbps **

# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='trestbps',y='thalach',data=df,hue='target')
plt.show()


# ### **Making Predictions**

# **Splitting the dataset into training and test set**

# In[ ]:


X= df.drop('target',axis=1)
y=df['target']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42)


# **Preprocessing - Scaling the features**

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.transform(X_test)
X_test = pd.DataFrame(X_test_scaled)


# ## **1. k-Nearest Neighor Algorithm**

# **Implementing GridSearchCv to select best parameters and applying k-NN Algorithm**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
knn =KNeighborsClassifier()
params = {'n_neighbors':list(range(1,20)),
    'p':[1, 2, 3, 4,5,6,7,8,9,10],
    'leaf_size':list(range(1,20)),
    'weights':['uniform', 'distance']
         }


# In[ ]:


model = GridSearchCV(knn,params,cv=3, n_jobs=-1)


# In[ ]:


model.fit(X_train,y_train)
model.best_params_           #print's parameters best values


# **Making predictions**

# In[ ]:


predict = model.predict(X_test)


# **Checking accuracy**

# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix
print('Accuracy Score: ',accuracy_score(y_test,predict))
print('Using k-NN we get an accuracy score of: ',
      round(accuracy_score(y_test,predict),5)*100,'%')


# **Confusion Matrix**

# In[ ]:



cnf_matrix = confusion_matrix(y_test,predict)
cnf_matrix


# In[ ]:


class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)

#create a heat map
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'YlGnBu',
           fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for k-Nearest Neighbors Model', y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


# **Classification report**

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predict))


# **Receiver Operating Characterstic(ROC) Curve**

# In[ ]:


from sklearn.metrics import roc_auc_score,roc_curve


# In[ ]:


#Get predicted probabilites from the model
y_probabilities = model.predict_proba(X_test)[:,1]


# In[ ]:


#Create true and false positive rates
false_positive_rate_knn,true_positive_rate_knn,threshold_knn = roc_curve(y_test,y_probabilities)


# In[ ]:


#Plot ROC Curve
plt.figure(figsize=(10,6))
plt.title('Revceiver Operating Characterstic')
plt.plot(false_positive_rate_knn,true_positive_rate_knn)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[ ]:


#Calculate area under the curve
roc_auc_score(y_test,y_probabilities)


# ## **2. Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()


# In[ ]:


# Setting parameters for GridSearchCV
params = {'penalty':['l1','l2'],
         'C':[0.01,0.1,1,10,100],
         'class_weight':['balanced',None]}
log_model = GridSearchCV(log,param_grid=params,cv=10)


# In[ ]:


log_model.fit(X_train,y_train)

# Printing best parameters choosen through GridSearchCV
log_model.best_params_


# **Making predictions**

# In[ ]:


predict = log_model.predict(X_test)


# **Accuracy Metrics**

# In[ ]:


from sklearn.metrics import accuracy_score
print('Accuracy Score: ',accuracy_score(y_test,predict))
print('Using Logistic Regression we get an accuracy score of: ',
      round(accuracy_score(y_test,predict),5)*100,'%')


# In[ ]:


from sklearn.metrics import recall_score,precision_score,classification_report,roc_auc_score,roc_curve
print(classification_report(y_test,predict))


# **Confusion Matrix**

# In[ ]:


cnf_matrix = confusion_matrix(y_test,predict)
cnf_matrix


# In[ ]:


class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)

#create a heat map
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'YlGnBu',
           fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for Logisitic Regression Model', y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


# **Receiver Operating Characterstic(ROC) Curve**

# In[ ]:


#Get predicted probabilites
target_probailities_log = log_model.predict_proba(X_test)[:,1]


# In[ ]:


#Create true and false positive rates
log_false_positive_rate,log_true_positive_rate,log_threshold = roc_curve(y_test,
                                                             target_probailities_log)


# In[ ]:


#Plot ROC Curve
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.title('Reciver Operating Characterstic Curve')
plt.plot(log_false_positive_rate,log_true_positive_rate)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()


# In[ ]:


#Calculate area under the curve
roc_auc_score(y_test,target_probailities_log)


# ## **3. Decision Tree**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtree= DecisionTreeClassifier(random_state=7)


# In[ ]:


#Setting parameters for GridSearchCV
params = {'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11]}
tree_model = GridSearchCV(dtree, param_grid=params, n_jobs=-1)


# In[ ]:


tree_model.fit(X_train,y_train)
#Printing best parameters selected through GridSearchCV
tree_model.best_params_


# **Making predictions**

# In[ ]:


predict = tree_model.predict(X_test)


# **Accuracy Metrics**

# In[ ]:


from sklearn.metrics import accuracy_score
print('Accuracy Score: ',accuracy_score(y_test,predict))
print('Using Decision Tree we get an accuracy score of: ',
      round(accuracy_score(y_test,predict),5)*100,'%')


# In[ ]:


from sklearn.metrics import classification_report,roc_auc_score,roc_curve


# **Classification Report**

# In[ ]:


print(classification_report(y_test,predict))


# **Confusion Matrix**

# In[ ]:


cnf_matrix = confusion_matrix(y_test,predict)
cnf_matrix


# In[ ]:


class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)

#create a heat map
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'YlGnBu',
           fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for Decision Tree Model', y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


# **Receiver Operating Characterstic(ROC) Curve**

# In[ ]:


#Get predicted probabilites
target_probailities_tree = tree_model.predict_proba(X_test)[:,1]


# In[ ]:


#Create true and false positive rates
tree_false_positive_rate,tree_true_positive_rate,tree_threshold = roc_curve(y_test,
                                                             target_probailities_tree)


# In[ ]:


#Plot ROC Curve
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.title('Reciver Operating Characterstic Curve')
plt.plot(tree_false_positive_rate,tree_true_positive_rate)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()


# In[ ]:


#Calculate area under the curve
roc_auc_score(y_test,target_probailities_tree)


# ** Comparing ROC Curve of k-Nearest Neighbors, Logistic Regression and Decision Tree**

# In[ ]:


#Plot ROC Curve
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.title('Reciver Operating Characterstic Curve')
plt.plot(false_positive_rate_knn,true_positive_rate_knn,label='k-Nearest Neighbor')
plt.plot(log_false_positive_rate,log_true_positive_rate,label='Logistic Regression')
plt.plot(tree_false_positive_rate,tree_true_positive_rate,label='Decision Tree')
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.legend()
plt.show()


# **Suggestions are welcome**
