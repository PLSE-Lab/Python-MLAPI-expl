#!/usr/bin/env python
# coding: utf-8

# # Predicting heart disease using Machine learning
# 
# `Python-based machine learning model capable of predicting whether or not someone has heart disease based on their medical history.`

# ***My Approach :-***
#     
#     Problem definition
#     Data
#     Evaluation
#     Features
#     Modelling
#     Experimentation
# 

# # 1. Problem Definition
# 
# In a statement,
# 
# > ***Given clinical parameters about a patient, can we predict whether or not they have heart disease?***
# 

# #  2. Data
# 
# > The original data came from the Cleavland data the UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/heart+Disease
# 
# > Other version of it available on Kaggle. https://www.kaggle.com/ronitf/heart-disease-uci

# # 3. Evaluation
# 
#   >  If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept.We will get success result.

# # 4. Features
# 
# > This is where you'll get different information about each of the features in your data. You can do this via doing your own research (such as looking at the links above) or by talking to a subject matter expert (someone who knows about the dataset).
# 
# > Create data dictionary
# 
#  `age - age in years
#     sex - (1 = male; 0 = female)
#     cp - chest pain type
#         0: Typical angina: chest pain related decrease blood supply to the heart
#         1: Atypical angina: chest pain not related to heart
#         2: Non-anginal pain: typically esophageal spasms (non heart related)
#         3: Asymptomatic: chest pain not showing signs of disease
#     trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern
#     chol - serum cholestoral in mg/dl
#         serum = LDL + HDL + .2 * triglycerides
#         above 200 is cause for concern
#     fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
#         '>126' mg/dL signals diabetes
#     restecg - resting electrocardiographic results
#         0: Nothing to note
#         1: ST-T Wave abnormality
#             can range from mild symptoms to severe problems
#             signals non-normal heart beat
#         2: Possible or definite left ventricular hypertrophy
#             Enlarged heart's main pumping chamber
#     thalach - maximum heart rate achieved
#     exang - exercise induced angina (1 = yes; 0 = no)
#     oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more
#     slope - the slope of the peak exercise ST segment
#         0: Upsloping: better heart rate with excercise (uncommon)
#         1: Flatsloping: minimal change (typical healthy heart)
#         2: Downslopins: signs of unhealthy heart
#     ca - number of major vessels (0-3) colored by flourosopy
#         colored vessel means the doctor can see the blood passing through
#         the more blood movement the better (no clots)
#     thal - thalium stress result
#         1,3: normal
#         6: fixed defect: used to be defect but ok now
#         7: reversable defect: no proper blood movement when excercising
#     target - have disease or not (1=yes, 0=no) (= the predicted attribute)`
# 

# # Tools Used
#  > *** Pandas, Matplotlib and NumPy for data analysis and manipulation.***
#  
#    * pandas for data analysis.
#    * NumPy for numerical operations.
#    * Matplotlib/seaborn for plotting or data visualization.
#    * Scikit-Learn for machine learning modelling and evaluation.
# 

# In[ ]:


#Import all the tools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn


#plots to appear inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

#Models from scikit-Learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve


# # Load data

# In[ ]:


heart_disease_data  =  pd.read_csv("../input/heart-disease.csv")
#Size of data (Rows,Column)
heart_disease_data.shape


# In[ ]:


#Top 5 results
heart_disease_data.head()


# 
# # Data Exploration OR Exploratory data analysis(EDA)
# 
#  >  Goal here is to find out more about the data and become a subject matter export on the dataset.
# 
#    1. What question(s) are you trying to solve?
#    2. What kind of data do we have and how do we treat different types?
#    3. What's missing from the data and how do you deal with it?
#    4. Where are the outliers and why should you care about them?
#    5. How can you add, change or remove features to get more out of your data?
# 
# 

# In[ ]:


# 1 - > Having heart disease
heart_disease_data["target"].value_counts()


# In[ ]:


heart_disease_data["target"].value_counts().plot(kind="bar",color=["brown","green"]);


# In[ ]:


#heart_disease_data.info()
#heart_disease_data.describe()
#Check missing values
heart_disease_data.isna().sum()


# In[ ]:


heart_disease_data.sex.value_counts()


# In[ ]:


# Compare target column with sex column
pd.crosstab(heart_disease_data.target,heart_disease_data.sex)


# In[ ]:


# Create a plot of crosstab

pd.crosstab(heart_disease_data.target,heart_disease_data.sex).plot(kind="bar",figsize=(10,6),
                    color=["brown","green"])
plt.title("Heart disease frequency  for sex")
plt.xlabel("0 = No disease 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female","Male"])
plt.xticks(rotation=0)


# In[ ]:


#Graph b/w Age and Max Heart Rate for Heart Disease

plt.figure(figsize=(10,6))

#sbn.set_style("darkgrid")

#plt.style.use("dark_background")

#Scatter with positive
plt.scatter(heart_disease_data.age[heart_disease_data.target == 1],
        heart_disease_data.thalach[heart_disease_data.target==1],
           color="orange")

#Scatter with negative
plt.scatter(heart_disease_data.age[heart_disease_data.target == 0],
        heart_disease_data.thalach[heart_disease_data.target== 0],
           color="blue");

plt.title("Hear disease scatter plot for Age vs Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max heart rate")
plt.legend(["Disease","No-Disease"])


# In[ ]:


#Check the distribution of age with histogram 

heart_disease_data.age.plot.hist()


#  ***Hear disease frequency per Chest Pain Type***
#  * cp - chest pain type
# * 0:  Typical angina: chest pain related decrease blood supply to the heart 
# *  1: Atypical angina: chest pain not related to heart
# * 2: Non-anginal pain: typically esophageal spasms (non heart related)
# * 3: Asymptomatic: chest pain not showing signs of disease

# In[ ]:


pd.crosstab(heart_disease_data.cp,heart_disease_data.target).plot(kind="bar",color=["yellow","blue"],figsize=(10,6))
plt.title("Hear disease frequency per Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Frequency")
plt.legend(["Disease","No-Disease"])
plt.xticks(rotation=0)


# In[ ]:


pd.crosstab(heart_disease_data.cp,heart_disease_data.target)


# In[ ]:


# Find the correlation between our independent variables
corr_heart_matrix = heart_disease_data.corr()
corr_heart_matrix

#Negative Correlation = a relationship b/w two varibles in which one variable increases  as the other decreases
'''It also means that with the decrease of variable X, variable Y should increase instead. '''
corr_heart_matrix = heart_disease_data.corr()
plt.figure(figsize=(15, 10))

fig, ax = plt.subplots(figsize=(15,10))
ax = sbn.heatmap(corr_heart_matrix, 
            annot=True, 
            linewidths=0.5, 
            fmt= ".2f", 
            cmap="YlGnBu");
bottom,top = ax.get_ylim()
ax.set_ylim(bottom+0.5,top-0.5)


# # 5. Modelling

# In[ ]:


#Split data into X and y
X = heart_disease_data.drop("target",axis=1)
y = heart_disease_data["target"]


# In[ ]:


# Split data into train and test sets
np.random.seed(42)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# 
# ***Model choices***
# 
# Now we've got our data prepared, we can start to fit models. We'll be using the following and comparing their results.
# 
#    1. Logistic Regression - LogisticRegression()
#    2. K-Nearest Neighbors - KNeighboursClassifier()
#    3.  RandomForest - RandomForestClassifier()
# 
# * Scikit-Learn algorithm cheat sheet https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

# In[ ]:


# Create a new models in a dictionary
models = {"Logistic Regrssion": LogisticRegression(),
         "KNN":KNeighborsClassifier(),
         "Random Forest":RandomForestClassifier()}

#Function for Fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of different Scikit-Learn machine learning models
    X_train : training data
    X_test : testing data
    y_train : labels assosciated with training data
    y_test : labels assosciated with test data
    """
    # Random seed for reproducible results
    np.random.seed(42)
    # Make a list to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores


# In[ ]:


model_scores = fit_and_score(models=models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test)
model_scores


# ### Model Comparison

# In[ ]:


model_compare_heart = pd.DataFrame(model_scores,index=["accuracy"])
#model_compare_heart.plot.bar()
model_compare_heart.T.plot.bar()


# ### Hyperparameter Tuning

# 
# * Hyperparameter tuning - Each model you use has a series of dials you can turn to dictate how they perform. Changing these values may increase or decrease model performance.
# * Feature importance - If there are a large amount of features we're using to make predictions, do some have more importance than others? For example, for predicting heart disease, which is more important, sex or age?
# * [Confusion Matrix](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/) - Compares the predicted values with the true values in a tabular way, if 100% correct, all values in the matrix will be top left to bottom right (diagnol line).
# * [Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html) - Splits your dataset into multiple parts and train and tests your model on each part and evaluates performance as an average.
# *    [Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score) - Proportion of true positives over total number of samples. Higher precision leads to less false positives.
# *    [Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score) - Proportion of true positives over total number of true positives and false negatives. Higher recall leads to less false negatives.
# *    [F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) - Combines precision and recall into one metric. 1 is best, 0 is worst.
# *    [Classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) - Sklearn has a built-in function called classification_report() which returns some of the main classification metrics such as precision, recall and f1-score.
# *    [ROC Curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_score.html) - Receiver Operating Characterisitc is a plot of true positive rate versus false positive rate.
# *    [Area Under Curve (AUC)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) - The area underneath the ROC curve. A perfect model achieves a score of 1.0.
# 

# ### Tune KNeighborsClassifier (K-Nearest Neighbors or KNN) by hand
# * There's one main hyperparameter we can tune for the K-Nearest Neighbors (KNN) algorithm, and that is number of neighbours. The default is 5 (n_neigbors=5).
# * Imagine all our different samples on one graph like the scatter graph we have above. KNN works by assuming dots which are closer together belong to the same class. If n_neighbors=5 then it assume a dot with the 5 closest dots around it are in the same class.
# 

# In[ ]:


# Create a list of train and test scores
train_scores = []
test_scores = []

# create a list of different values for N_neighbors
neighbors= range(1,25)

knn = KNeighborsClassifier()

for i in neighbors:
    knn.set_params(n_neighbors = i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))
    


# In[ ]:


plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(1, 25, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")


# ### Tuning models with with RandomizedSearchCV
# [Hyperparameter](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV)
# 

# In[ ]:


# Different LogisticRegression hyperparameters
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Different RandomForestClassifier hyperparameters
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}


# In[ ]:


#Tune LogisticsRegression
np.random.seed(42)

# cv - >cross validation 5+5+5+5+5+5=20
#Setup random hyperparameter
hyper_log_lgres = RandomizedSearchCV(LogisticRegression(),
                                    param_distributions=log_reg_grid,
                                    cv= 5,n_iter=20,verbose=True)
hyper_log_lgres.fit(X_train,y_train)


# In[ ]:


#Best Hyperparameter

hyper_log_lgres.best_params_


# In[ ]:


#Evalauate the LogisticRegression model 
hyper_log_lgres.score(X_test,y_test)


# In[ ]:


#Tune RandomForestClassifier
np.random.seed(42)

# cv - >cross validation 5+5+5+5+5+5=20
#Setup RandomForestClassifier
hyper_log_rdmFostCls = RandomizedSearchCV(RandomForestClassifier(),
                                    param_distributions=rf_grid,
                                    cv= 5,n_iter=20,verbose=True)
hyper_log_rdmFostCls.fit(X_train,y_train)


# In[ ]:


#Best Hyperparameter
hyper_log_rdmFostCls.best_params_


# In[ ]:


#Evalauate the RandomForestClassifier model 
hyper_log_rdmFostCls.score(X_test,y_test)


# ### Tuning model with GridSearchCv
# * The difference between RandomizedSearchCV and GridSearchCV is where RandomizedSearchCV searches over a grid of hyperparameters performing n_iter combinations, GridSearchCV will test every single possible combination.
# 

# In[ ]:


#setup for LogisticRegression
log_reg_grid = {"C":np.logspace(-4,4,30),
               "solver":["liblinear"]}
gs_log_reg = GridSearchCV(LogisticRegression(),
                         param_grid=log_reg_grid,
                         cv= 5,
                         verbose=True)
#Fit grid hyperparameter search model
gs_log_reg.fit(X_train,y_train)


# In[ ]:


#Check best hyperparameter
gs_log_reg.best_params_


# In[ ]:


#Evaluate the grid search LogisticRegression model
gs_log_reg.score(X_test,y_test)


# ### Evaluating a classification model, beyond accuracy
# 
#    * ROC curve and AUC score - [plot_roc_curve()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_roc_curve.html#sklearn.metrics.plot_roc_curve)
#    * Confusion matrix - [confusion_matrix()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
#    * Classification report - [classification_report()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
#    * Precision - [precision_score()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
#    * Recall - [recall_score()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
#    * F1-score - [f1_score()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
# 

# In[ ]:


# Make preidctions on test data
y_preds = gs_log_reg.predict(X_test)
y_preds


# In[ ]:


# Plot ROC curve and calculate AUC metric
from sklearn.metrics import plot_roc_curve


#roc_curve(gs_log_reg, X_test)


# In[ ]:


plot_roc_curve(gs_log_reg,X_test,y_test)


# In[ ]:


#Confusion matrix
print(confusion_matrix(y_test,y_preds))


# In[ ]:


# Import Seaborn
import seaborn as sbn
sbn.set(font_scale=1.5) # Increase font size

def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sbn.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True, 
                     cbar=True)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    
plot_conf_mat(y_test, y_preds)


# In[ ]:


#Classification report
print(classification_report(y_test,y_preds))


# ### Calculate evaluation metrics using cross-validation
# * calculate accuracy,precision,recall and f1 score using cross-validation by `cross_val_score()` 

# In[ ]:


#Check Best hyperparameter
gs_log_reg.best_params_


# In[ ]:


# New classifier with best parameter
clf = LogisticRegression(C=0.20433597178569418,solver='liblinear')
clf


# In[ ]:


#Crosss-validated accuracy
crss_acc = cross_val_score(clf,X,y,cv=5,scoring="accuracy")
#crss_acc
cv_acc_mean = np.mean(crss_acc)
cv_acc_mean


# In[ ]:


#Crosss-validated precision
crss_prec = cross_val_score(clf,X,y,cv=5,scoring="precision")
cv_prec_mean = np.mean(crss_prec)
cv_prec_mean


# In[ ]:


#Crosss-validated recall
crss_rec = cross_val_score(clf,X,y,cv=5,scoring="recall")
cv_rec_mean = np.mean(crss_rec)
cv_rec_mean


# In[ ]:


#Crosss-validated F1
crss_f1 = cross_val_score(clf,X,y,cv=5,scoring="f1")
crss_f1_mean = np.mean(crss_f1)
crss_f1_mean


# In[ ]:


# Visualizing cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc_mean,
                            "Precision": cv_prec_mean,
                            "Recall": cv_rec_mean,
                            "F1": crss_f1_mean},
                          index=[0])
cv_metrics.T.plot.bar(title="Cross-Validated Metrics", legend=False);


# ### Feature Importance
# * It means which feature contribute most to the outcomes of the model and How did they contribute?
# * Using Logistics Regression
# 

# In[ ]:


#gs_log_reg.best_params_
# Fit an instance of Logistic Regression
clf  = LogisticRegression(C=0.20433597178569418,
                         solver="liblinear")
clf.fit(X_train,y_train)


# In[ ]:


#Check coeff
clf.coef_


# In[ ]:


# Match features to columns
features_dict = dict(zip(heart_disease_data.columns, list(clf.coef_[0])))
features_dict


# In[ ]:


# Visualize feature importance
features_heart_disease = pd.DataFrame(features_dict, index=[0])
features_heart_disease.T.plot.bar(title="Feature Importance", legend=False, color="blue");


# #### Looking at these figures and this specific dataset, it seems if the patient is female, they're more likely to have heart disease.
# 

# In[ ]:


pd.crosstab(heart_disease_data["sex"], heart_disease_data["target"])


# In[ ]:


### Slope (positive coefficient) with target
pd.crosstab(heart_disease_data["slope"], heart_disease_data["target"])


# 
#     0: Upsloping: better heart rate with excercise (uncommon)
#     1: Flatsloping: minimal change (typical healthy heart)
#     2: Downslopins: signs of unhealthy heart
# 

# ## 6. Experimentation
# 

# >   If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursure this project.
# 
# ### Could you collect more data?
# 
# * Could you try a better model? If you're working with structured data, you might want to look into [CatBoost](https://catboost.ai/) or [XGBoost](https://xgboost.ai/).
# 
# * Could you improve the current models (beyond what we've done so far)?
#     If your model is good enough, how would you export it and share it with others? (Hint: check out [Scikit-Learn's documentation on model persistance](https://scikit-learn.org/stable/modules/model_persistence.html))
# 
#     #### The more you try, the more you figure out what doesn't work, the more you'll start to get a hang of what does.
#                                                                                      
#                     

# In[ ]:




