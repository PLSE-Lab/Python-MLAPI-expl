#!/usr/bin/env python
# coding: utf-8

# # Predicting Heart Disease using Machine Learning
# 
# 

# 

# # Problem Definition
# 
# 
# > Given clinical parameters about a patient, can we predict whether or not they have heart disease?
# 
# 
# 
# The following is the data dictionary for the dataset.
# 
# 1. age - age in years 
# 2. sex - (1 = male; 0 = female) 
# 3. cp - chest pain type 
#     * 0: Typical angina: chest pain related decrease blood supply to the heart
#     * 1: Atypical angina: chest pain not related to heart
#     * 2: Non-anginal pain: typically esophageal spasms (non heart related)
#     * 3: Asymptomatic: chest pain not showing signs of disease
# 4. trestbps - resting blood pressure (in mm Hg on admission to the hospital)
#     * anything above 130-140 is typically cause for concern
# 5. chol - serum cholestoral in mg/dl 
#     * serum = LDL + HDL + .2 * triglycerides
#     * above 200 is cause for concern
# 6. fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) 
#     * '>126' mg/dL signals diabetes
# 7. restecg - resting electrocardiographic results
#     * 0: Nothing to note
#     * 1: ST-T Wave abnormality
#         - can range from mild symptoms to severe problems
#         - signals non-normal heart beat
#     * 2: Possible or definite left ventricular hypertrophy
#         - Enlarged heart's main pumping chamber
# 8. thalach - maximum heart rate achieved 
# 9. exang - exercise induced angina (1 = yes; 0 = no) 
# 10. oldpeak - ST depression induced by exercise relative to rest 
#     * looks at stress of heart during excercise
#     * unhealthy heart will stress more
# 11. slope - the slope of the peak exercise ST segment
#     * 0: Upsloping: better heart rate with excercise (uncommon)
#     * 1: Flatsloping: minimal change (typical healthy heart)
#     * 2: Downslopins: signs of unhealthy heart
# 12. ca - number of major vessels (0-3) colored by flourosopy 
#     * colored vessel means the doctor can see the blood passing through
#     * the more blood movement the better (no clots)
# 13. thal - thalium stress result
#     * 1,3: normal
#     * 6: fixed defect: used to be defect but ok now
#     * 7: reversable defect: no proper blood movement when excercising 
# 14. target - have disease or not (1=no heart disease, 0=heart disease) (= the predicted attribute)
# 
# 

# 

# In[ ]:



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

## Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier


## Model evaluators
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve;


# 

# In[ ]:


df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv") # 'DataFrame' shortened to 'df'
df.shape # (rows, columns)


# # Data Exploration (exploratory data analysis or EDA)

# In[ ]:


df.head()


# In[ ]:



df.head(10)


# 

# In[ ]:


# No. of positive and negative patients in our samples
df.target.value_counts()


# Since these two values are nearly equal, our `target` column is **balanced**. An **unbalanced** target column, having different number of counts in each label, can be harder to model than a balanced set.
# 

# In[ ]:


# Normalized value counts
df.target.value_counts(normalize=True)


# 

# In[ ]:


# Plot the value counts with a bar graph
df.target.value_counts().plot(kind="bar", color=["purple", "magenta"]);


# 

# In[ ]:


df.info()


# In[ ]:


df.describe()


# # Age Vs Sex for heart disease and other crosstabs

# In[ ]:


df.sex.value_counts()


# In[ ]:


pd.crosstab(df.target, df.age)


# There are 207 males and 96 females in our study.

# In[ ]:


# Compare target column with sex column
pd.crosstab(df.target, df.sex)


# 
# 
# Since there are about 100 women and 72 of them have a postive value of heart disease being present, we might infer, based on this one variable if the participant is a woman, there's a 75% chance she does not have heart disease.
# 
# As for males, there's about 200 total with around half indicating a presence of heart disease. So we might predict, if the participant is male, 50% of the time he will have heart disease.
# 

# In[ ]:



pd.crosstab(df.target, df.sex).plot(kind="bar", figsize=(10,6), color=["salmon", "lightblue"])

plt.title("Heart Disease Frequency for Sex")
plt.xlabel("0 = Disease, 1 = No Disease")
plt.ylabel("Amount")
plt.legend(["Female", "Male"])
plt.xticks(rotation=0);


# From the above visualization, it is clear that males are at a higher risk of having heart disease with more than 50 percent of the included patients having the disease.

# ### Age vs Max Heart rate for Heart Disease
# 
# 

# In[ ]:



plt.figure(figsize=(10,6))

# For positve examples
plt.scatter(df.age[df.target==0], 
            df.thalach[df.target==0], 
            c="salmon") # define it as a scatter figure

# Now for negative examples, 
plt.scatter(df.age[df.target==1], 
            df.thalach[df.target==1], 
            c="lightblue") 


plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.legend(["Disease", "No Disease"])
plt.ylabel("Max Heart Rate");


# 
# 
# For patients without heart disease, it seems the younger someone is, the higher their max heart rate (dots are higher on the left of the graph and decreases somewhat linearly), and it decreases with age. There is no such fixed pattern in patients with heart disease.
# 

# In[ ]:


# Histograms to check age distribution 
df.age.plot.hist();


# It is not a perfect normal distribution but sways to the right.

# ### Heart Disease Frequency per Chest Pain Type
# 

# In[ ]:


pd.crosstab(df.cp, df.target)


# In[ ]:



pd.crosstab(df.cp, df.target).plot(kind="bar", 
                                   figsize=(10,6), 
                                   color=["lightblue", "salmon"])


plt.title("Heart Disease Frequency Per Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Frequency")
plt.legend(["Disease", "No disease"])
plt.xticks(rotation = 0);


# 
# 
# Remember from our data dictionary what the different levels of chest pain are.
# 
# 3. cp - chest pain type 
#     * 0: Typical angina: chest pain related decrease blood supply to the heart
#     * 1: Atypical angina: chest pain not related to heart
#     * 2: Non-anginal pain: typically esophageal spasms (non heart related)
#     * 3: Asymptomatic: chest pain not showing signs of disease
# 
# The bargraph validates the data dictionary.

# In[ ]:


# Find the correlation between our independent variables
corr_matrix = df.corr()
corr_matrix 


# In[ ]:



corr_matrix = df.corr()
fig, ax=plt.subplots(figsize=(15, 15))
ax=sns.heatmap(corr_matrix, 
            annot=True, 
            linewidths=0.5, 
            fmt= ".2f", 
            cmap="YlGnBu");
bottom, top=ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)


# Maximum heart rate (thalach) and chest pain type have high positive correlation with having no heart disease. (remember 1= no heart disease). 

# # Modeling
# 
# 

# In[ ]:


df.head()


# In[ ]:


# Everything except target variable
X = df.drop("target", axis=1)

# Target variable
y = df.target.values


# In[ ]:


# Independent variables (no target column)
X.head()


# In[ ]:


# Targets
y


# In[ ]:



np.random.seed(42)


X_train, X_test, y_train, y_test = train_test_split(X,  
                                                    y, 
                                                    test_size = 0.2) 


# In[ ]:


X_train.head()


# In[ ]:


y_train, len(y_train)


# Beautiful, we can see we're using 242 samples to train on. Let's look at our test data.

# In[ ]:


X_test.head()


# In[ ]:


y_test, len(y_test)


# And we've got 61 examples we'll test our model(s) on. Let's build some.

# ### Model choices
# 
# Now we've got our data prepared, we can start to fit models. We'll be using the following and comparing their results.
# 
# 1. Logistic Regression - [`LogisticRegression()`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
# 2. K-Nearest Neighbors - [`KNeighboursClassifier()`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
# 3. RandomForest - [`RandomForestClassifier()`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

# In[ ]:


# Put models in a dictionary
models = {"KNN": KNeighborsClassifier(),
          "Logistic Regression": LogisticRegression(), 
          "Random Forest": RandomForestClassifier(), "Decision Tree":DecisionTreeClassifier()}

# Create function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):

    
    np.random.seed(42)
    
    model_scores = {}
    
    for name, model in models.items():
        
        model.fit(X_train, y_train)
        
        model_scores[name] = model.score(X_test, y_test)*100
    return model_scores


# In[ ]:


model_scores = fit_and_score(models=models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test)
model_scores


# # Model Comparison
# 
# 

# In[ ]:


model_compare = pd.DataFrame(model_scores, index=['accuracy'])
model_compare.plot.bar();


# Since KNN, Decision Tree, AdaBoost gives relatively quite low accuracy value I decide to ignore them.
# 
# ### Tuning models with RandomizedSearchCV
# 
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


# We first use `RandomizedSearchCV` to try and tune our `LogisticRegression` model.
# 
# 

# In[ ]:



np.random.seed(42)


rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)


rs_log_reg.fit(X_train, y_train);


# In[ ]:


rs_log_reg.best_params_


# In[ ]:


rs_log_reg.score(X_test, y_test)


# I'll do the same for `RandomForestClassifier`.

# In[ ]:


# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

# Fit random hyperparameter search model
rs_rf.fit(X_train, y_train);


# In[ ]:



rs_rf.best_params_


# In[ ]:



rs_rf.score(X_test, y_test)


# Because Logistic Regression is returning a better accuracy score, I use GridSearchCV
# on it.

# In[ ]:



log_reg_grid = {"penalty" :['l2'],
"C":np.logspace(-4,4,30),
"class_weight":[{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}],
"solver": ['liblinear', 'saga','sag','newton-cg','lbfgs'],"max_iter":[10] }



gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)


gs_log_reg.fit(X_train, y_train);


# In[ ]:


# Check the best parameters
gs_log_reg.best_params_


# In[ ]:


# Evaluate the model
gs_log_reg.score(X_test, y_test)


# ## Using AdaBoost Classifier

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
adaboost=AdaBoostClassifier(base_estimator=LogisticRegression(C= 2.592943797404667,
 class_weight= {1: 0.5, 0: 0.5},
 max_iter= 10,
 penalty= 'l2',
 solver= 'liblinear'),n_estimators=100)
adaboost.fit(X_train,y_train)
adaboost.score(X_test,y_test)


# # Evaluating other metrics

# In[ ]:


from sklearn.metrics import roc_curve

y_preds = gs_log_reg.predict(X_test)


# In[ ]:


y_preds


# In[ ]:


y_test


# In[ ]:


y_probs=gs_log_reg.predict_proba(X_test)
y_probs_positive=y_probs[:,1]
y_probs_positive


# In[ ]:



from sklearn.metrics import auc

fpr, tpr, thresholds= roc_curve(y_test, y_probs_positive)
def plot_roc_curve(fpr,tpr):
 plt.plot(fpr, tpr, color="orange",label="ROC")
 plt.plot([0,1],[0,1],color="darkblue",linestyle="--",label="Guessing")
 plt.xlabel("False positive rate")
 plt.ylabel("True positive rate")
 plt.title("Receiver Operating Characterisitics curve")
 plt.legend()
 plt.show()
plot_roc_curve(fpr,tpr)


# In[ ]:


roc_auc=auc(fpr,tpr)
roc_auc


# In[ ]:


# Display confusion matrix
print(confusion_matrix(y_test, y_preds))


# In[ ]:



import seaborn as sns
sns.set(font_scale=1.5) 
def plot_conf_mat(y_test, y_preds):
    
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True, # Annotate the boxes
                     cbar=True)
    plt.xlabel("true label")
    plt.ylabel("predicted label")
    bottom,top=ax.get_ylim()
    ax.set_ylim(bottom+0.5, top-0.5)
    
plot_conf_mat(y_test, y_preds)


# In[ ]:


prec=precision_score(y_test,y_preds)
prec


# In[ ]:


rec=recall_score(y_test,y_preds)
rec


# In[ ]:


# Show classification report
print(classification_report(y_test, y_preds))


# **It is important to note that 0 implies heart disease while 1 implies no heart disease. Hence from the confusion matrix, TPR is not 0.87 but 0.89. Similarly FPR is not 0.11 but 0.12. Similarly, 

# In[ ]:


rs_rf.best_params_


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
rf=RandomForestClassifier(n_estimators=560,
 min_samples_split=12,
 min_samples_leaf=15,
 max_depth=3)
cv_acc=np.mean(cross_val_score(rf,X,y,cv=5,scoring="accuracy"))
cv_prec=np.mean(cross_val_score(rf,X,y,cv=5,scoring="precision"))
cv_recall=np.mean(cross_val_score(rf,X,y,cv=5,scoring="recall"))
cv_f1=np.mean(cross_val_score(rf,X,y,cv=5,scoring="f1"))
cv_acc,cv_prec,cv_recall,cv_f1


# In[ ]:


cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                            "Precision": cv_prec,
                            "Recall": cv_recall,
                            "F1": cv_f1},
                          index=[0])
cv_metrics.T.plot.bar(title="Random Forest Cross-Validated Metrics", legend=False);


# In[ ]:



gs_log_reg.best_params_


# In[ ]:



from sklearn.model_selection import cross_val_score

clf = LogisticRegression(C=0.23357214690901212,
                         solver="liblinear")


# In[ ]:



cv_acc = cross_val_score(clf,
                         X,
                         y,
                         cv=5, 
                         scoring="accuracy")


# In[ ]:


cv_acc = np.mean(cv_acc)
cv_acc


# In[ ]:


# Cross-validated precision score
cv_precision = np.mean(cross_val_score(clf,
                                       X,
                                       y,
                                       cv=5,
                                       scoring="precision")) 
cv_precision


# In[ ]:



cv_recall = np.mean(cross_val_score(clf,
                                    X,
                                    y,
                                    cv=5, 
                                    scoring="recall")) 
cv_recall


# In[ ]:



cv_f1 = np.mean(cross_val_score(clf,
                                X,
                                y,
                                cv=5, 
                                scoring="f1")) 
cv_f1


# In[ ]:


# Visualizing cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                            "Precision": cv_precision,
                            "Recall": cv_recall,
                            "F1": cv_f1},
                          index=[0])
cv_metrics.T.plot.bar(title="Logistic Regression Cross-Validated Metrics", legend=False);


# In[ ]:



clf.fit(X_train, y_train);


# In[ ]:


# Check feature importance
clf.coef_


# In[ ]:


# Match features to columns
features_dict = dict(zip(df.columns, list(clf.coef_[0])))
features_dict


# In[ ]:


# Visualize feature importance
features_df = pd.DataFrame(features_dict, index=[0])
features_df.T.plot.bar(title="Feature Importance", legend=False);


# All our predictions is picked up by the model. As sex variable decreases towards 0 (female), target variable points towards 1 ( no disease). As max heart rate increases, probability of having heart disease decreases.
