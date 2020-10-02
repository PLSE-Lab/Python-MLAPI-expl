#!/usr/bin/env python
# coding: utf-8

# ## **Churn Prediction of Telco Customers using Classification Algorithms**
# 
# In this kernel, I want to apply some classification algorithm on Telco customers churn data set.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head()


# In[ ]:


data.info()


# TotalCharges column is seen as object type, but includes numeric type values. Convert this column to numeric. 

# In[ ]:


data.TotalCharges = pd.to_numeric(data.TotalCharges, errors='coerce')
data.info()


# There are 11 missing values in TotalCharges column. We can fill the missing values with median data, set it to 0 or delete these rows, it is up to you. I prefer deleting these columns because it is a small part compared to all data.

# In[ ]:


#delete rows including null values
data.dropna(inplace = True)


# We don't need customerID column for analyzing, so we can drop this column. 

# In[ ]:


data.drop(["customerID"],axis=1,inplace = True)


# Replace text columns to integers. The columns below includes similar text values so I changed them once.

# In[ ]:


data.gender = [1 if each == "Male" else 0 for each in data.gender]

columns_to_convert = ['Partner', 
                      'Dependents', 
                      'PhoneService', 
                      'MultipleLines',
                      'OnlineSecurity',
                      'OnlineBackup',
                      'DeviceProtection',
                      'TechSupport',
                      'StreamingTV',
                      'StreamingMovies',
                      'PaperlessBilling', 
                      'Churn']

for item in columns_to_convert:
    data[item] = [1 if each == "Yes" else 0 if each == "No" else -1 for each in data[item]]
    
data.head()


# Let's look at the distribution of Churn values. As you can see below, the data set is imbalanced. But for now, I will ignore this.

# In[ ]:


sns.countplot(x="Churn",data=data);


# In[ ]:


sns.pairplot(data,vars = ['tenure','MonthlyCharges','TotalCharges'], hue="Churn")


# People having lower tenure and higher monthly charges are tend to churn more.
# Also as you can see below; having month-to-month contract and fiber obtic internet have a really huge effect on churn probability.

# In[ ]:


sns.set(style="whitegrid")
g1=sns.catplot(x="Contract", y="Churn", data=data,kind="bar")
g1.set_ylabels("Churn Probability")

g2=sns.catplot(x="InternetService", y="Churn", data=data,kind="bar")
g2.set_ylabels("Churn Probability")


# Convert remaining text based columns to dummy columns using pandas get_dummies function. This function creates new columns named as values of the related columns.
# 
# Now our data set only have integer and numerical columns so that we can apply statistical models.

# In[ ]:


data = pd.get_dummies(data=data)
data.head()


# Let's see the correlation between churn and the remaining columns. Customers having month-to-month contract, having fiber optic internet service and using electronic payment are tend to churn more whereas people having two-year contract and having internet service are tend to not churn.

# In[ ]:


data.corr()['Churn'].sort_values()


# **Prepare x and y**
# 
# First, seperate x and y values. y would be our class which is Churn column in this dataset. x would be the remaing columns.
# Also, apply normalization to x in order to scale all values between 0 and 1.

# In[ ]:


#assign Class_att column as y attribute
y = data.Churn.values

#drop Class_att column, remain only numerical columns
new_data = data.drop(["Churn"],axis=1)

#Normalize values to fit between 0 and 1. 
x = (new_data-np.min(new_data))/(np.max(new_data)-np.min(new_data)).values


# **Splitting Data**
# 
# Split the data set as train and test with %20-%80 ratio.

# In[ ]:


#Split data into Train and Test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state =1)


# ## **Apply Machine Learning Algorithms**
# 
# Let's start to apply some machine learning algorithms and find the accuracy of each.
# 
# 
# 
# **KNN Classification**

# In[ ]:


# %%KNN Classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) #set K neighbor as 3
knn.fit(x_train,y_train)
predicted_y = knn.predict(x_test)
print("KNN accuracy according to K=3 is :",knn.score(x_test,y_test))


# We assume K = 3 for first iteration, but actually we don't know what is the optimal K value that gives maximum accuracy. So we can write a for loop that iterates for example 25 times and gives the accuracy at each iteartion. So that we can find the optimal K value.

# In[ ]:


score_array = []
for each in range(1,25):
    knn_loop = KNeighborsClassifier(n_neighbors = each) #set K neighbor as 3
    knn_loop.fit(x_train,y_train)
    score_array.append(knn_loop.score(x_test,y_test))
    
plt.plot(range(1,25),score_array)
plt.xlabel("Range")
plt.ylabel("Score")
plt.show()


# As you can see above, if we use K = 11, then we get maximum score of %78.7

# In[ ]:


knn_model = KNeighborsClassifier(n_neighbors = 11) #set K neighbor as 11
knn_model.fit(x_train,y_train)
predicted_y = knn_model.predict(x_test)
accuracy_knn = knn_model.score(x_test,y_test)
print("KNN accuracy according to K=11 is :",accuracy_knn)


# **Logistic Regression Classification**

# In[ ]:


# %%Logistic regression classification
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)
accuracy_lr = lr_model.score(x_test,y_test)
print("Logistic Regression accuracy is :",accuracy_lr)


# **SVM(Support Vector Machine) Classification**

# In[ ]:


# %%SVM Classification
from sklearn.svm import SVC
svc_model = SVC(random_state = 1)
svc_model.fit(x_train,y_train)
accuracy_svc = svc_model.score(x_test,y_test)
print("SVM accuracy is :",accuracy_svc)


# **Naive Bayes Classification**

# In[ ]:


# %%Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(x_train,y_train)
accuracy_nb = nb_model.score(x_test,y_test)
print("Naive Bayes accuracy is :",accuracy_nb)


# **Decision Tree Classification**

# In[ ]:


# %%Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train,y_train)
accuracy_dt = dt_model.score(x_test,y_test)
print("Decision Tree accuracy is :",accuracy_dt)


# **Random Forest Classification**

# In[ ]:


# %%Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rf_model_initial = RandomForestClassifier(n_estimators = 5, random_state = 1)
rf_model_initial.fit(x_train,y_train)
print("Random Forest accuracy for 5 trees is :",rf_model_initial.score(x_test,y_test))


# I set tree number as 5 initially. But I want to find the appropriate tree number. Let's try to find the best number with trying 1 to 50.

# In[ ]:


score_array = []
for each in range(1,50):
    rf_loop = RandomForestClassifier(n_estimators = each, random_state = 1) #set K neighbor as 3
    rf_loop.fit(x_train,y_train)
    score_array.append(rf_loop.score(x_test,y_test))
    
plt.plot(range(1,50),score_array)
plt.xlabel("Range")
plt.ylabel("Score")
plt.show()


# As you can see, the highest accuracy is at n_estimators = 33.

# In[ ]:


rf_model = RandomForestClassifier(n_estimators = 33, random_state = 1) #set tree number as 33
rf_model.fit(x_train,y_train)
accuracy_rf = rf_model.score(x_test,y_test)
print("Random Forest accuracy for 33 trees is :",accuracy_rf)


# Logistic regression and SVC classificagtion algorithms have the highest accuracy. But as I mentioned before, our data is imbalanced. So it is important to look at the confusion matrix according to these two algorithms. With imbalanced datasets, the highest accuracy does not give the best model. Assume we have 1000 total rows, 10 rows are churn and 990 rows are non-churn. If we find all these 10 churn rows as non-churn, then the accuracy will be still %99. Althogh it is a wrong model, if we do not look at the confusion matrix, then we can not see the mistake.
# 
# Confusion matrix gives us FN(false negative), FP(false positive), TN(true negative) and TP(true positive) values.
# <figure>
#     <img src='https://drive.google.com/uc?export=view&id=1Ks14UX9YOnXvSVZx4ucfU1bIgNpiYMdz' 
#          style="width: 400px; max-width: 100%; height: auto"
#          alt='missing'/>
# </figure>
# 
# 

# In[ ]:


# %%Confusion Matrix libraries
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report

#for Logistic Regression
cm_lr = confusion_matrix(y_test,lr_model.predict(x_test))

# %% confusion matrix visualization
import seaborn as sns
f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm_lr, annot = True, linewidths = 0.5, color = "red", fmt = ".0f", ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_true")
plt.title("Confusion Matrix of Logistic Regression")
plt.show()


# For logistic regression confusion matrix; 
# 
#     TN = 927
#     FP = 114
#     FN = 169
#     TP = 197
#  
#  This means; there are total 927+114 = 1041 actual non-churn values and the algorithm predict 927 of them as non-churn and 114 of them churn. Also there are total 169 + 197 = 366 actual churn values and the algorithm predict 169 of them as non-churn and 197 of them as churn. 
#  
#  Acuuracy should not be used as solely metric for imbalance datasets. There are some other metrics named as recall and precision.
# 
# <figure>
#     <img src='https://drive.google.com/uc?export=view&id=1_4ZejjBAy3WUo-7OkS0hJpzBEaRr3otQ' 
#          style="width: 400px; max-width: 100%; height: auto"
#          alt='missing'/>
# </figure>
# 
# Sometimes we get high recall and low precision or vice versa. There is another metric that combines both precision and recall like below. We will use F1 score to identify the best algorithm score.
# 
# <figure>
#     <img src='https://drive.google.com/uc?export=view&id=1sbGivo8TRDQr-iEBrM4HdHTuhVJD8kms' 
#          style="width: 300px; max-width: 100%; height: auto"
#          alt='missing'/>
# </figure>

# Let's write a function that calculates and print both accuracy, recall, precision and weighted F1 score.

# In[ ]:


# the function that prints all scores
def print_scores(headline, y_true, y_pred):
    print(headline)
    acc_score = accuracy_score(y_true, y_pred)
    print("accuracy: ",acc_score)
    pre_score = precision_score(y_true, y_pred)
    print("precision: ",pre_score)
    rec_score = recall_score(y_true, y_pred)                            
    print("recall: ",rec_score)
    f_score = f1_score(y_true, y_pred, average='weighted')
    print("f1_score: ",f_score)


# We can also use classification_report function from skleran library to show all these metrics.

# In[ ]:


report = classification_report(y_test, lr_model.predict(x_test))
print(report)


# Print all results of each algorithm.

# In[ ]:


print_scores("Logistic Regression;",y_test, lr_model.predict(x_test))
print_scores("SVC;",y_test, svc_model.predict(x_test))
print_scores("KNN;",y_test, knn_model.predict(x_test))
print_scores("Naive Bayes;",y_test, nb_model.predict(x_test))
print_scores("Decision Tree;",y_test, dt_model.predict(x_test))
print_scores("Random Forest;",y_test, rf_model.predict(x_test))


# **CONCLUSION**
# * Since data set is imbalanced, we prefered to use F1 score rather than accuracy.
# * Logistic Regression gives the highest F1 Score, so it is the best model. 
# * Naive Bayes is the worst model because it gives the lowest F1 score.
# * Sex has no impact on churn.
# * People having month-to-month contract tend to churn more than people having long term conracts.
# * As the tenure increases, the probability of churn decreases.
# * As tmonthly charges increases, the probability of churn increases.
# 
# I hope you liked my Kernel! If you have any comments or  suggestions, please write below.
# 
# Thank you!
# 
