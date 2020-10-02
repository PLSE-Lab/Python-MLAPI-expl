#!/usr/bin/env python
# coding: utf-8

# # Pima Dataset with Neural Network, Random Forest & Gradient Boosted Classifier
# 
# This notebook explores the effectivness of a Neuaral Network, Random Forest and a Greadient Boosted Classifier on the Pima dataset. Most algorithms I've experimented with have trouble reaching an accuracy greater than ~80%, however all three algorithms presented here were able to bypass these limitations.
# 
# Please upvote if you found this helpfull. Comments and critisms welcome.
# 
# Thank you.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, recall_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
import warnings

warnings.filterwarnings('ignore')


# ## Inspecting & Preparing the Data

# In[2]:


data = pd.read_csv('../input/diabetes.csv')

display(data.head())
display(data.describe())
sns.countplot(data['Outcome'],label="Count")


# The first table tells us there are 5 features describing the dataset. Looking at the __min__ row of the second, we note that the features of glucose, blood pressure, skin thickness, insulin and BMI have a minimum value of 0. This indatctes that these features contain missing values. Also note that most of the features are not the same order of magnitude. It will be helpul to rescale the data before training the models.
# 
# From the count plot we see that 2/3 of the data points are about healthy individuals while 1/3 is of diabetics. Due to the imbalance, accuracy may not be the best metric by which to measure model performance.
# 
# Next, we check for any NaN values:

# In[3]:


display(data.info())


# There are no NaN values in any columns of the dataset.
# 
# Before proeceeding we replace the zero values in the columns by the average value in the column depending on the outcome of that point (healthy or diabetic). This is better practice than replacing the 0 values by the average of the whole column irrespective of the point's outcome.
# 
# Then extract data, split into training and test sets and rescale. Note to stratify the splitting process to maintian the ratio of positive and negative outcomes in the split sets.

# In[4]:


#Feature Preprocessing
for col in ['BloodPressure', 'Glucose','SkinThickness','Insulin','BMI','Age']:
    for target in data.Outcome.unique():
        mask = (data[col] != 0) & (data['Outcome'] == target)
        data[col][(data[col] == 0) & (data['Outcome'] == target)] = data[col][(data[col] == 0) & (data['Outcome'] == target)].replace(0,data[mask][col].mean())
        
#Extract data
X = data.iloc[:,0:8]
y = data.iloc[:,-1]

#X2 = X.copy()
#X2['Glucose<125'] = X2['Glucose']<125
#X2['Glucose<125'] = X2['Glucose<125'].astype(int)

#X_train,X_test,y_train,y_test = train_test_split(X2, y, random_state=0,stratify=y)

X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0,stratify=y)

scaler = StandardScaler()
scaler.fit(X_train,y_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## Feature Correlations
# 
# Assuming a non-gaussian distribution of feature values, we choose the Spearman mathod to calculate correlations between the features.

# In[5]:


corr_mat=sns.heatmap(data.corr(method='spearman'),annot=True,cbar=True,
            cmap='viridis', vmax=1,vmin=-1,
            xticklabels=X.columns,yticklabels=X.columns)
corr_mat.set_xticklabels(corr_mat.get_xticklabels(),rotation=90)


# Note the following features are well correlated:
# -  Age and Pregnancies
# -  Insulin and Glucose
# -  Skin thickness and BMI
# 
# More importantly, note that Glucose and Insulin are the two features with the strongest correlation to the outcome. 
# 
# The piar plot below will help us better understand how each feature is distributed throughout the dataset.

# In[6]:


sns.pairplot(data, hue = "Outcome", diag_kind='kde')


# We see that it is difficult to properly separate any two feature distributions by eye, however we hope that the model will be able to learn their differences. The strong peaks shown in the skin thickness and insulin KDE's are due to the replacement of zeros by the outcome average we did previously.
# 
# ## Neural Network
# 
# For our first model, we consider a neural network and use cross validaton to determine the optimum regularization parameter. The number of nodes and levels was determined via trial and error so it is likely that there is a better way to implement this model to gain higher accuracy.

# In[7]:


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
param_grid = {'alpha': np.arange(1,40)}
mlp = MLPClassifier(solver='lbfgs', activation='relu', random_state=9, learning_rate='adaptive',
                        hidden_layer_sizes=[10,10,10])
grid_mlp = GridSearchCV(mlp, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)
grid_mlp.fit(X_train_scaled, y_train)
print("Best cross-validation accuracy: {:.3f}".format(grid_mlp.best_score_))
print("Test set score: {:.3f}".format(grid_mlp.score(X_test_scaled,y_test)))
print("Best parameters: {}".format(grid_mlp.best_params_))


# In[8]:


mlp = MLPClassifier(solver='lbfgs', activation='relu', random_state=9, learning_rate='adaptive',
                        hidden_layer_sizes=[10,10,10], alpha=grid_mlp.best_params_['alpha'])
mlp.fit(X_train_scaled,y_train)
conf_mat_mlp = confusion_matrix(y_test, mlp.predict(X_test_scaled))
sns.heatmap(conf_mat_mlp, annot=True, cbar=False, cmap="viridis_r",
            yticklabels=['Healthy', 'Diabetes'], xticklabels=['Healthy', 'Diabetes'])


# The above confusion matrix indicates that the neural network does well at classifying healthy patients correctly (~83%) but struggles with classifying diabetics correctly (~69%). This is captured by the recall score for the fit.

# In[9]:


print("Neural Net Recall Score: {:.2f}".format(recall_score(y_test, mlp.predict(X_test_scaled))))


# 
# 
# 
# ## Random Forest

# In[10]:


param_grid = {'n_estimators': [100,200,300,400,500]}
forest = RandomForestClassifier(random_state=79)
grid_forest = GridSearchCV(forest, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)
grid_forest.fit(X_train_scaled, y_train)
print("Best cross-validation accuracy: {:.3f}".format(grid_forest.best_score_))
print("Test set score: {:.3f}".format(grid_forest.score(X_test_scaled,y_test)))
print("Best parameters: {}".format(grid_forest.best_params_))
#1


# In[11]:


forest = RandomForestClassifier(n_estimators=grid_forest.best_params_['n_estimators'])
forest.fit(X_train_scaled,y_train)
conf_mat_forest = confusion_matrix(y_test, forest.predict(X_test_scaled))
sns.heatmap(conf_mat_forest, annot=True, cbar=False, cmap="viridis_r",
            yticklabels=['Healthy', 'Diabetes'], xticklabels=['Healthy', 'Diabetes'])


# We see that the higher accuracy attributed to the Random Forest classifier is due to better classification of healthy samples. The number of correctly classified diabetics is still the same as that of the neural network.

# ## Gradient Boosted Classifier

# In[12]:


param_grid = {'n_estimators': [100,200,300,400,500], 'max_depth': [2,3,4,5]}
gbrt = GradientBoostingClassifier(random_state=0)
grid_gbrt = GridSearchCV(gbrt, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)
grid_gbrt.fit(X_train_scaled, y_train)
print("Best cross-validation accuracy: {:.3f}".format(grid_gbrt.best_score_))
print("Test set score: {:.3f}".format(grid_gbrt.score(X_test_scaled,y_test)))
print("Best parameters: {}".format(grid_gbrt.best_params_))


# In[13]:


gbrt = GradientBoostingClassifier(random_state=0, max_depth=grid_gbrt.best_params_['max_depth'],
                                 n_estimators=grid_gbrt.best_params_['n_estimators'])
gbrt.fit(X_train_scaled,y_train)
conf_mat_gbrt = confusion_matrix(y_test, gbrt.predict(X_test_scaled))
sns.heatmap(conf_mat_gbrt, annot=True, cbar=False, cmap="viridis_r",
            yticklabels=['Healthy', 'Diabetes'], xticklabels=['Healthy', 'Diabetes'])


# We see that the gradient boosted model performs the best. Its classification of healthy patients is similiar to that of that of the random forest, however it is better at classifying diabetics correctly. This results is shown by a greater recall score:

# In[14]:


print("GBRT Recall Score: {:.2f}".format(recall_score(y_test, gbrt.predict(X_test_scaled))))


# 
# 
# ## ROC Curve
# 
# Another good metric of performance is the area under an ROC curve (AUC). 
# 

# In[15]:


#gbrt
fpr,tpr,thresholds = roc_curve(y_test, gbrt.decision_function(X_test_scaled))
plt.plot(fpr,tpr, label='GBRT')
print("AUC GBRT: {:.3f}".format(roc_auc_score(y_test, gbrt.decision_function(X_test_scaled))))

#Forest
fpr,tpr,thresholds = roc_curve(y_test, forest.predict_proba(X_test_scaled)[:,1])
plt.plot(fpr,tpr, label='Forest')
print("AUC Forest: {:.3f}".format(roc_auc_score(y_test, forest.predict_proba(X_test_scaled)[:,1])))


#nn
fpr,tpr,thresholds = roc_curve(y_test, mlp.predict_proba(X_test_scaled)[:,1])
plt.plot(fpr,tpr, label='NN')
print("AUC NN: {:.3f}".format(roc_auc_score(y_test, mlp.predict_proba(X_test_scaled)[:,1])))
plt.legend(loc='best')
plt.xlim(0,1)
plt.ylim(0,1)


# As expected, the classifiers fall in the same order of performance as above.

# ## Summary

# In[16]:


summary = {'Metrics': ["Accuracy", "Recall", "AUC"],
           'Neural Network': [mlp.score(X_test_scaled,y_test), recall_score(y_test, mlp.predict(X_test_scaled)), roc_auc_score(y_test, mlp.predict_proba(X_test_scaled)[:,1])], 
           'Random Forest': [forest.score(X_test_scaled,y_test), recall_score(y_test, forest.predict(X_test_scaled)), roc_auc_score(y_test, forest.predict_proba(X_test_scaled)[:,1])],
           'Gradient Boosted Classifier': [gbrt.score(X_test_scaled,y_test), recall_score(y_test, gbrt.predict(X_test_scaled)), roc_auc_score(y_test, gbrt.predict_proba(X_test_scaled)[:,1])]}
summary_df = pd.DataFrame(data=summary)
display(summary_df)

