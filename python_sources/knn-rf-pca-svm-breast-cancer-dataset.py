#!/usr/bin/env python
# coding: utf-8

# ** Objective - predict the type of diagnosis of the breast cancer basis the observations. Classification between - Benign & Malignant
# - import the necessary libraries for EDA
# 
# ** Note - Results vary on each run due to random selection of data and hence the discrepancy between the comments & the result (This has been corrected - Figure out HOW?)
# 
# ** Idea of this notebook is to use multiple ML  models on a well know dataset and see how they perform and learn how to improve results
# 
# ** Models used - KNN, Random forests, Decision trees with boosting, PCA with SVM

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## The objective of this notebook is to introduce beginners to a black box implementation of KNN, Decision trees (With boosting), Random forest and working from there.
# - I have deliberately left certain analysis like Feature visualization (To know distribution of variables) and Feature engineering

# In[ ]:


cancer = pd.read_csv('../input/breast_cancer.csv')


# In[ ]:


cancer.head()


# What we need to check in the above data
# What are the type (categorical/quant) & number (check for missing values)
# - As seen below, there's no missing value and there are 31 variables.
# - Need to select features for a model
# - Classifications can be predicted by - Decision trees, KNN, Logit (Multiple classification), SVM - WHICH one to choose?
# 
# Since the incidence of breast cancer is anyways low in the population, accuracy of the model has to consider the same
# - **not the case in this dataset**

# In[ ]:


cancer.info() #missing values can be checked here & also the datatype of the variables


# Lets study this data further
# - how do we select variables (Check correlation between result and variables, which are the top such variables)
# - Which model to choose 
# - What data trends need to be checked out

# In[ ]:


# how is the cancer spread in the data
t= cancer.groupby('diagnosis')
t.count()


# As Andrew Ng suggests, lets run our data in some model and see the prelim results
# ## Randomnly select feature set and predict result using KNN

# In[ ]:


from sklearn.preprocessing import StandardScaler # need to scale feature set to fit KNN
scaler = StandardScaler() # initialise a scaler object to run on a dataframe
# create a df randomnly of 9 features
random_df = cancer[['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean']]


# In[ ]:


scaler.fit(random_df.drop('diagnosis',axis=1)) # run the above scaler method on the selected dataframe
scaled_features = scaler.transform(random_df.drop('diagnosis',axis=1)) #scaled features is the new transformed df with nomralized values
#on which the KNN algo can be run


# In[ ]:


from sklearn.model_selection import train_test_split #Basic practice of train/test splitting the data
X_train, X_test, y_train, y_test = train_test_split(scaled_features,random_df['diagnosis'],
                                                    test_size=0.30, random_state=101)


# **Now that we have 1. Selected the features, 2. Scaled the features, 3. Done the train/test split, lets initialise the KNN classifier and apply it to our data**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20) # initialise the KNN classifier with neighbours=20 in this case
knn.fit(X_train,y_train)


# In[ ]:


pred = knn.predict(X_test) #run the KNN model on the test data


# In[ ]:


unique, counts = np.unique(pred, return_counts=True) # checking the variable spread in the prediction dataset
print (unique, counts)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,pred))


# Recall & Precision measures enable us to focus on the right metrics to check performance of the model
# * In a breast cancer dataset, it is more important to identify the cases of Malignancy (M) correctly and hence lower False Negatives and hence have a better Recall, than a good precision. As we can see in the below classification report, Recall - M of .86 is a poor performance of the model.  (Numbers in the below report have further changed due to the random selection of train/test data - Has been addressed using "Random State "command)
# * Which is the case when FP hurts more than FN. This will be the one where Precision will be a more important metric than Recall. Selection of new employees or Identifying whom to give a loan or credit card? Rejecting good employees (False Negatives) hurts lesser than a False Positive.

# In[ ]:


print(classification_report(y_test,pred))


# So what exactly do the above terms mean in the context of knowing the Precision and Recall of our model

# - False positive (FP)= (model predicts 1, actual 0)
# - True positive (TP) = (model predicts 1, actual 1) 
# - False negative (FN)= (model predicts 0, actual 1)
# - True negative (TN)=  (model predicts 0, actual 0)
# - True positive rate (TPR) = TP/(TP+FN)  (RECALL M)
# - True negative rate (TNR) = TN/(TN+FP)  (RECALL B)
# - Overall accuracy = (TP+TN)/(TP+FP+FN+TN)  

# ## randomnly selecting 9 features & running KNN outputs 89-95% accuracy!! Is that good or bad or in-between (Ponder and work-out in the context of precision & recall)
# - I fixed the uncertain output coming from randomly selecting test/train datasets - **Figure out How?**

# ## How about running random forests in the above case of randomnly choosing 9 features & we get 90-95% accuracy
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train) # This has been run on scaled features


# In[ ]:


rfc_pred = rfc.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,rfc_pred))


# In[ ]:


print(classification_report(y_test,rfc_pred))


# ## Question - We see variance in results whenever this dataset is run repeatedly. Why? More importantly, how is this addressed?
# - Cross validation
# - Different algorithm (Boosting?)
# - Better feature selection and maybe some feature engg 

# ## Lets see the impact of Adaboost on the above dataset

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dt = DecisionTreeClassifier() 
clf = AdaBoostClassifier(n_estimators=100, base_estimator=dt,learning_rate=1)
#Above I have used decision tree as a base estimator, you can use any ML learner as base estimator if it ac# cepts sample weight 
clf.fit(X_train,y_train)


# In[ ]:


clf_pred = clf.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,clf_pred))


# In[ ]:


print(classification_report(y_test,clf_pred)) #M (Malignant) is 1 here, assumed by the model


# ## Observe correlated features and remove them

# In[ ]:


plt.figure(figsize = (18,18))
sns.heatmap(cancer.corr(), cmap='coolwarm', annot = True)


# - Radium mean, perimter mean, area mean, radius worst, perimeter worst, area worst..These 6 can be replaced with one 
# - texture mean is correlated with texture worst
# - area, radius, perimeter are all correlated
# - SE metrics for radius are not correlated with the other radius measure
# # ACTION
# - Drop - perimeter_mean, area_mean, radius_worst, perimeter_worst, area_worst and KEEP radius_mean
# - Drop - all "worst" features
# # Features on which to re-run KNN
# - diagnosis , radius_mean, texture_mean ,smoothness_mean, compactness_mean, concavity_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, compactness_se, concavity_se, concave points_se, symmetry_se, fractal_dimension_se  

# In[ ]:


# creating a DF with the above selected features only
chosen = cancer[['diagnosis' , 'radius_mean', 'texture_mean' ,'smoothness_mean', 'compactness_mean', 'concavity_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se']]


# # Run KNN on the selected features only

# In[ ]:


scaler.fit(chosen.drop('diagnosis',axis=1))
scaled_features = scaler.transform(chosen.drop('diagnosis',axis=1))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,chosen['diagnosis'],
                                                    test_size=0.30, random_state=101)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


pred = knn.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))


# In[ ]:


print(classification_report(y_test,pred))


# ## While the model's accuracy improved, the critical metric of Recall - M which penalizes for False Negatives has only marginally improved. So what can be done now? (#Why do we need to improve recall - M??)
# - Should we compromise on Precision in this case?
# - Can we perform better feature selection (Try SelectKBest)

# ## Run random forest on the selected features only
# 

# In[ ]:


# creating a DF with the above selected features only
chosen = cancer[['diagnosis' , 'radius_mean', 'texture_mean' ,'smoothness_mean', 'compactness_mean', 'concavity_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se']]


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = chosen.drop('diagnosis',axis=1)
y = chosen['diagnosis']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[ ]:


rfc_pred = rfc.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,rfc_pred))


# In[ ]:


print(classification_report(y_test,rfc_pred))


# ## Through our "Blackbox" approach in this notebook, we've been able to learn couple of valuable things
# - Random forests seems to perform better than KNN, ceteris paribus
# - Depending on the classification problem at hand, precision or recall needs to play a more important role
# - Even basic feature selection led to significantly better results
#     - Always better to select a certain set of features, though in this case an intuitive understanding on how to choose is missing
# - Depending on the data on which the algorithm runs, the output varies significantly, can be sorted by 
#     - cross validation
#     - boosting (How about other boosting methods)
# 

# ## Implementing PCA and then SVM on this dataset
# - Since this is a 30 variable dataset, a shortcut we can use to select the right parameters to model the target on, is using PCA

# In[ ]:


df = cancer.drop('diagnosis',axis=1)


# In[ ]:


df.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)


# # We need to scale the feature before we run a Principal Component Analysis

# In[ ]:


type(scaled_data)


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA(n_components=2)


# In[ ]:


pca.fit(scaled_data)


# In[ ]:


x_pca = pca.transform(scaled_data)


# In[ ]:


scaled_data.shape


# In[ ]:


x_pca.shape


# # What we have done in Principal components Analysis till now is 
# - Idea is to reduce the data into the principal components (Video for the Math behind PCA - https://www.youtube.com/watch?v=N9MRzIHyA_Q)
# - The principal components are the axis which explain the max variation of the target variable. Once we know these principal components, we can input them into whichever suitable model (KNN, SVM etc)to predict the "Target"variable 
# - Data needs to be scaled before we run PCA
# - We then review the change that has happened by running the PCA

# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['diagnosis'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# # We see above that PCA has separated the underlying observations of the "Target"variable very neatly. And now it becomes easy to apply an ML algorithm on these PCA-1, PCA-2 to predict outcomes

# In[ ]:


reformed = pd.DataFrame(x_pca)


# In[ ]:


reformed.info()


# In[ ]:


df1 = pd.DataFrame(cancer['diagnosis'])


# In[ ]:


df1.info()


# In[ ]:


final_df = reformed.join(df1)
# We are creating a df which has only the principal components and the "Target"variable so that we can a ML algo on this dataframe


# In[ ]:


final_df.info()


# In[ ]:


final_df.columns = ['PCA-1', 'PCA-2', 'Target']


# In[ ]:


from sklearn.model_selection import train_test_split
X = final_df[['PCA-1', 'PCA-2']]
y = final_df['Target']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.30, random_state=101)


# In[ ]:


X_train.info()


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


model = SVC()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


predictions = model.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))


# In[ ]:


print(classification_report(y_test,predictions))


# # We observe that we have the highest recall-M (91%) using the above approach of PCA and then SVM

# In[ ]:




