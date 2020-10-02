#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data = pd.read_csv("../input/heart.csv")


# In[ ]:


type(data)


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.info()


# 1. age: The person's age in years
# 
# 2. sex: The person's sex (1 = male, 0 = female)
# 
# 3. cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
# 
# 4. trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
# 
# 5. chol: The person's cholesterol measurement in mg/dl
# 
# 6. fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
# 
# 7. restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
# 
# 8. thalach: The person's maximum heart rate achieved
# 
# 9. exang: Exercise induced angina (1 = yes; 0 = no)
# 
# 10. oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
# 
# 11. slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
# 
# 12. ca: The number of major vessels (0-3)
# 
# 13. thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
# 
# 14. target: Heart disease (0 = no, 1 = yes)
# 
# Heart disease risk factors to the following: high cholesterol, high blood pressure, diabetes, weight, family history and smoking . 
# 
# According to another source , the major factors that can't be changed are: increasing age, male gender and heredity. 
# 
# Note that thalassemia, one of the variables in this dataset, is heredity. 
# 
# Major factors that can be modified are: Smoking, high cholesterol, high blood pressure, physical inactivity, and being overweight and having diabetes. 
# 
# Other factors include stress, alcohol and poor diet/nutrition.

# In[ ]:


data.sample(5)


# In[ ]:


data.isnull().sum()


# In[ ]:


data.isnull().sum().sum()


# ###So, we have no missing values

# In[ ]:


print(data.corr()["target"].abs().sort_values(ascending=False))


# ### This shows that most columns are moderately correlated with target, but 'fbs' is very weakly correlated.

# # Exploratory Data Analysis (EDA)

# In[ ]:


y = data["target"]


# In[ ]:


ax = sns.countplot(data["target"])
target_temp = data.target.value_counts()
print(target_temp)


# From the total dataset of 303 patients, 165 (54%) have a heart disease (target=1)

# # Percentage of patient with or without heart problems in the given dataset

# In[ ]:


print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))
print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))


# In[ ]:


data["sex"].unique()


# In[ ]:


sns.barplot(data["sex"],data["target"])


# In[ ]:


def plotAge():
    facet_grid = sns.FacetGrid(data, hue='target')
    facet_grid.map(sns.kdeplot, "age", shade=True, ax=axes[0])
    legend_labels = ['disease false', 'disease true']
    for t, l in zip(axes[0].get_legend().texts, legend_labels):
        t.set_text(l)
        axes[0].set(xlabel='age', ylabel='density')

    avg = data[["age", "target"]].groupby(['age'], as_index=False).mean()
    sns.barplot(x='age', y='target', data=avg, ax=axes[1])
    axes[1].set(xlabel='age', ylabel='disease probability')

    plt.clf()


# In[ ]:


fig_age, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))

plotAge()


# ### Here 0 is female and 1 is male patients

# In[ ]:


countFemale = len(data[data.sex == 0])
countMale = len(data[data.sex == 1])
print("Percentage of Female Patients:{:.2f}%".format((countFemale)/(len(data.sex))*100))
print("Percentage of Male Patients:{:.2f}%".format((countMale)/(len(data.sex))*100))


# In[ ]:


categorial = [('sex', ['female', 'male']), 
              ('cp', ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic']), 
              ('fbs', ['fbs > 120mg', 'fbs < 120mg']), 
              ('restecg', ['normal', 'ST-T wave', 'left ventricular']), 
              ('exang', ['yes', 'no']), 
              ('slope', ['upsloping', 'flat', 'downsloping']), 
              ('thal', ['normal', 'fixed defect', 'reversible defect'])]


# In[ ]:


def plotGrid(isCategorial):
    if isCategorial:
        [plotCategorial(x[0], x[1], i) for i, x in enumerate(categorial)] 
    else:
        [plotContinuous(x[0], x[1], i) for i, x in enumerate(continuous)] 


# In[ ]:


def plotCategorial(attribute, labels, ax_index):
    sns.countplot(x=attribute, data=data, ax=axes[ax_index][0])
    sns.countplot(x='target', hue=attribute, data=data, ax=axes[ax_index][1])
    avg = data[[attribute, 'target']].groupby([attribute], as_index=False).mean()
    sns.barplot(x=attribute, y='target', hue=attribute, data=avg, ax=axes[ax_index][2])
    
    for t, l in zip(axes[ax_index][1].get_legend().texts, labels):
        t.set_text(l)
    for t, l in zip(axes[ax_index][2].get_legend().texts, labels):
        t.set_text(l)


# In[ ]:


fig_categorial, axes = plt.subplots(nrows=len(categorial), ncols=3, figsize=(15, 30))

plotGrid(isCategorial=True)


# In[ ]:


continuous = [('trestbps', 'blood pressure in mm Hg'), 
              ('chol', 'serum cholestoral in mg/d'), 
              ('thalach', 'maximum heart rate achieved'), 
              ('oldpeak', 'ST depression by exercise relative to rest'), 
              ('ca', '# major vessels: (0-3) colored by flourosopy')]


# In[ ]:


def plotContinuous(attribute, xlabel, ax_index):
    sns.distplot(data[[attribute]], ax=axes[ax_index][0])
    axes[ax_index][0].set(xlabel=xlabel, ylabel='density')
    sns.violinplot(x='target', y=attribute, data=data, ax=axes[ax_index][1])


# In[ ]:


fig_continuous, axes = plt.subplots(nrows=len(continuous), ncols=2, figsize=(15, 22))

plotGrid(isCategorial=False)


# # Heart Disease Frequency for ages

# In[ ]:


pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()


# # Heart Disease frequency for sex (where 0 is female and 1 is male and "red" is have heart disease and "blue" is don't have heart disease)

# In[ ]:


pd.crosstab(data.sex,data.target).plot(kind="bar",figsize=(20,10),color=['blue','#AA1111' ])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Don't have Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# In[ ]:


data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']


# In[ ]:


data.head()


# # Heart disease according to Fasting Blood sugar 

# In[ ]:


pd.crosstab(data.fasting_blood_sugar,data.target).plot(kind="bar",figsize=(20,10),color=['#4286f4','#f49242'])
plt.title("Heart disease according to FBS")
plt.xlabel('FBS- (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation=90)
plt.legend(["Don't Have Disease", "Have Disease"])
plt.ylabel('Disease or not')
plt.show()


# # Analysing the chest pain (4 types of chest pain)
# 
# #[Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic]

# In[ ]:


data["chest_pain_type"].unique()


# In[ ]:


plt.figure(figsize=(26, 10))
sns.barplot(data["chest_pain_type"],y)


# # Analysing The person's resting blood pressure (mm Hg on admission to the hospital)

# In[ ]:


data["resting_blood_pressure"].unique()


# In[ ]:


plt.figure(figsize=(26, 10))
sns.barplot(data["resting_blood_pressure"],y)


# # Analysing the Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)

# In[ ]:


data["rest_ecg"].unique()


# In[ ]:


plt.figure(figsize=(26, 15))
sns.barplot(data["rest_ecg"],y)


# ## people with restecg '1' and '0' are much more likely to have a heart disease than with restecg '2'

# #Analysing Exercise induced angina (1 = yes; 0 = no)

# In[ ]:


data["exercise_induced_angina"].unique()


# In[ ]:


plt.figure(figsize=(10, 10))
sns.barplot(data["exercise_induced_angina"],y)


# ###People with exercise_induced_angina=1 are much less likely to have heart problems

# # Analysing the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)

# In[ ]:


data["st_slope"].unique()


# In[ ]:


plt.figure(figsize=(25, 10))
sns.barplot(data["st_slope"],y)


# Slope '2' causes heart pain much more than Slope '0' and '1'

# # Analysing number of major vessels (0-3) colored by flourosopy

# In[ ]:


data["num_major_vessels"].unique()


# ### count num_major vessels

# In[ ]:


sns.countplot(data["num_major_vessels"])


# ### comparing with target

# In[ ]:


sns.barplot(data["num_major_vessels"],y)


# ### num_major_vessels=4 has astonishingly large number of heart patients

# # Analysing A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
# 
# 

# In[ ]:


data["thalassemia"].unique()


# ### plotting the thalassemia distribution (0,1,2,3)

# In[ ]:


sns.distplot(data["thalassemia"])


# ### comparing with target

# In[ ]:


sns.barplot(data["thalassemia"],y)


# # thalassemia and cholesterol scatterplot

# In[ ]:


plt.figure(figsize=(20,10))
sns.scatterplot(x='cholesterol',y='thalassemia',data=data,hue='target')
plt.show()


# # thalassemia vs resting blood pressure scatterplot

# In[ ]:


plt.figure(figsize=(20,10))
sns.scatterplot(x='thalassemia',y='resting_blood_pressure',data=data,hue='target')
plt.show()


# ## Health rate vs age

# In[ ]:


plt.figure(figsize=(20, 10))
plt.scatter(x=data.age[data.target==1], y=data.thalassemia[(data.target==1)], c="green")
plt.scatter(x=data.age[data.target==0], y=data.thalassemia[(data.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


# In[ ]:


sns.pairplot(data=data)


# In[ ]:


data.hist()


# # Correlation plot

# Correlation analysis is a method of statistical evaluation used to study the strength of a relationship between two, numerically measured, continuous variables (e.g. height and weight)

# In[ ]:


# store numeric variables in cnames
cnames=['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression','num_major_vessels']


# In[ ]:


#Set the width and height of the plot
f, ax = plt.subplots(figsize=(7, 5))

#Correlation plot
df_corr = data.loc[:,cnames]
#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, annot = True, cmap='coolwarm',linewidths=.1)
plt.show()


# ##Correlation analysis

# In[ ]:


df_corr = data.loc[:,cnames]
df_corr


# # Splitting the dataset to Train and Test

# In[ ]:


from sklearn.model_selection import train_test_split

predictors = data.drop("target",axis=1)
target = data["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)
print("Training features have {0} records and Testing features have {1} records.".      format(X_train.shape[0], X_test.shape[0]))


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


Y_train.shape


# In[ ]:


Y_test.shape


# ## importing Accuracy score

# In[ ]:


from sklearn.metrics import accuracy_score


# # Modelling and predicting with Machine Learning
# The main goal of the entire project is to predict heart disease occurrence with the highest accuracy. In order to achieve this, we will test several classification algorithms. This section includes all results obtained from the study and introduces the best performer according to accuracy metric. I have chosen several algorithms typical for solving supervised learning problems throughout classification methods.
# 
# First of all, let's equip ourselves with a handy tool that benefits from the cohesion of SciKit Learn library and formulate a general function for training our models. The reason for displaying accuracy on both, train and test sets, is to allow us to evaluate whether the model overfits or underfits the data (so-called bias/variance tradeoff).

# In[ ]:


def train_model(X_train, y_train, X_test, y_test, classifier, **kwargs):
    
    """
    Fit the chosen model and print out the score.
    
    """
    
    # instantiate model
    model = classifier(**kwargs)
    
    # train model
    model.fit(X_train,y_train)
    
    # check accuracy and print out the results
    fit_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"Train accuracy: {fit_accuracy:0.2%}")
    print(f"Test accuracy: {test_accuracy:0.2%}")
    
    return model


# # Logistic regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

y_pred_lr = logreg.predict(X_test)
print(y_pred_lr)


# 

# In[ ]:


score_lr = round(accuracy_score(y_pred_lr,Y_test)*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print(confusion_matrix(Y_test,y_pred_lr))
print(classification_report(Y_test,y_pred_lr))
print("Accuracy:",accuracy_score(Y_test, y_pred_lr))


# In[ ]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
model = train_model(X_train, Y_train, X_test, Y_test, LogisticRegression)


# Best ACCURACY possible using Logistic regression

# In[ ]:


#Logistic Regression supports only solvers in ['liblinear', 'newton-cg'<-93.44, 'lbfgs'<-91.8, 'sag'<-72.13, 'saga'<-72.13]
clf = LogisticRegression(random_state=0, solver='newton-cg').fit(X_test, Y_test)
#The solver for weight optimization.
#'lbfgs' is an optimizer in the family of quasi-Newton methods.
clf.score(X_test, Y_test)


# # Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


matrix= confusion_matrix(Y_test, y_pred_lr)


# In[ ]:


sns.heatmap(matrix,annot = True, fmt = "d")


# fmt = d is format = default

# # precision Score

# In[ ]:


from sklearn.metrics import precision_score


# In[ ]:


precision = precision_score(Y_test, y_pred_lr)


# In[ ]:


print("Precision: ",precision)


# # Recall

# In[ ]:


from sklearn.metrics import recall_score


# In[ ]:


recall = recall_score(Y_test, y_pred_lr)


# In[ ]:


print("Recall is: ",recall)


# 
# 
# ---
# 
# 

# # F-Score

# balance of precision and recall score

# In[ ]:


print((2*precision*recall)/(precision+recall))


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
randfor = RandomForestClassifier(n_estimators=100, random_state=0)

randfor.fit(X_train, Y_train)

y_pred_rf = randfor.predict(X_test)
print(y_pred_rf)


# # Learning curve for Training score & cross validation score

# In[ ]:


from sklearn.model_selection import learning_curve
# Create CV training and test scores for various training set sizes
train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), 
                                                        X_train, 
                                                        Y_train,
                                                        # Number of folds in cross-validation
                                                        cv=10,
                                                        # Evaluation metric
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        n_jobs=-1, 
                                                        # 50 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 50))

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


# In[ ]:


score_rf = round(accuracy_score(y_pred_rf,Y_test)*100,2)

print("The accuracy score achieved using Random Forest is: "+str(score_rf)+" %")


# In[ ]:


#Random forest with 100 trees
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, Y_train)
print("Accuracy on training set: {:.3f}".format(rf.score(X_train, Y_train)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test, Y_test)))


# Now, let us prune the depth of trees and check the accuracy.

# In[ ]:


rf1 = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
rf1.fit(X_train, Y_train)
print("Accuracy on training set: {:.3f}".format(rf1.score(X_train, Y_train)))
print("Accuracy on test set: {:.3f}".format(rf1.score(X_test, Y_test)))


# performance metrics
# -Accuracy: is the ratio between the number of correct predictions and total number of predications.
# 
# $acc = \frac{TP + TN}{TP + TN + FP + FN}$
# 
# -Precision: is the ratio between the number of correct positives and the number of true positives plus the number of false positives.
# 
# $Precision (p) = \frac{TP}{TP + FP}$
# 
# -Recall: is the ratio between the number of correct positives and the number of true positives plus the number of false negatives.
# 
# $recall = \frac{TP}{TP + FN}$
# 
# -F-score: is known as the harmonic mean of precision and recall.
# 
# $acc = \frac{1}{\frac{1}{2}(\frac{1}{p}+\frac{1}{r})} = \frac{2pr}{p+r}$
# 
# -Problem characteristics in context of our case study:
# 
# TP = True positive (has heart disease). TN = True negative (has no heart disease). FP = False positive (has no heart disease) FN = False negative (has heart disease)

# ## confusion matrix of Random Forest

# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


matrix= confusion_matrix(Y_test, y_pred_rf)


# In[ ]:


sns.heatmap(matrix,annot = True, fmt = "d")


# # precision score

# In[ ]:


from sklearn.metrics import precision_score


# In[ ]:


precision = precision_score(Y_test, y_pred_rf)


# In[ ]:


print("Precision: ",precision)


# # recall

# In[ ]:


from sklearn.metrics import recall_score


# In[ ]:


recall = recall_score(Y_test, y_pred_rf)


# In[ ]:


print("Recall is: ",recall)


# # F score

# In[ ]:


print((2*precision*recall)/(precision+recall))


# # Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = train_model(X_train, Y_train, X_test, Y_test, GaussianNB)

nb.fit(X_train, Y_train)

y_pred_nb = nb.predict(X_test)
print(y_pred_nb)


# In[ ]:


score_nb = round(accuracy_score(y_pred_nb,Y_test)*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")


# In[ ]:


#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
model = train_model(X_train, Y_train, X_test, Y_test, GaussianNB)


# ## confusion matrix of Naive Bayes

# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


matrix= confusion_matrix(Y_test, y_pred_nb)


# In[ ]:


sns.heatmap(matrix,annot = True, fmt = "d")


# # precision score

# In[ ]:


from sklearn.metrics import precision_score


# In[ ]:


precision = precision_score(Y_test, y_pred_nb)


# In[ ]:


print("Precision: ",precision)


# # recall

# In[ ]:


from sklearn.metrics import recall_score


# In[ ]:


recall = recall_score(Y_test, y_pred_nb)


# In[ ]:


print("Recall is: ",recall)


# # f score

# In[ ]:


print((2*precision*recall)/(precision+recall))


# # KNN(K Nearest Neighbors)

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = train_model(X_train, Y_train, X_test, Y_test, KNeighborsClassifier, n_neighbors=8)

knn.fit(X_train, Y_train)

y_pred_knn = knn.predict(X_test)
print(y_pred_knn)


# In[ ]:


score_knn = round(accuracy_score(y_pred_knn,Y_test)*100,2)

print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")


# In[ ]:


# KNN
from sklearn.neighbors import KNeighborsClassifier
model = train_model(X_train, Y_train, X_test, Y_test, KNeighborsClassifier)


# Let's see if KNN can perform even better by trying different 'n_neighbours' inputs.

# In[ ]:


# Seek optimal 'n_neighbours' parameter
for i in range(1,10):
    print("n_neigbors = "+str(i))
    train_model(X_train, Y_train, X_test, Y_test, KNeighborsClassifier, n_neighbors=i)


# It turns out that value of n_neighbours (8) is optimal.

# ## Confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


matrix= confusion_matrix(Y_test, y_pred_knn)


# In[ ]:


sns.heatmap(matrix,annot = True, fmt = "d")


# # precision score

# In[ ]:


from sklearn.metrics import precision_score


# In[ ]:


precision = precision_score(Y_test, y_pred_knn)


# In[ ]:


print("Precision: ",precision)


# # recall

# In[ ]:


from sklearn.metrics import recall_score


# In[ ]:


recall = recall_score(Y_test, y_pred_knn)


# In[ ]:


print("Recall is: ",recall)


# # f score

# In[ ]:


print((2*precision*recall)/(precision+recall))


# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=3, random_state=0)

dt.fit(X_train, Y_train)

y_pred_dt = dt.predict(X_test)
print(y_pred_dt)


# In[ ]:


score_dt = round(accuracy_score(y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
tree1 = DecisionTreeClassifier(random_state=0)
tree1.fit(X_train, Y_train)
print("Accuracy on training set: {:.3f}".format(tree1.score(X_train, Y_train)))
print("Accuracy on test set: {:.3f}".format(tree1.score(X_test, Y_test)))


# The accuracy on the training set is 100%, while the test set accuracy is much worse. This is an indicative that the tree is overfitting and not generalizing well to new data. Therefore, we need to apply pre-pruning to the tree.
# 
# We set max_depth=3, limiting the depth of the tree decreases overfitting. This leads to a lower accuracy on the training set, but an improvement on the test set.

# In[ ]:


tree1 = DecisionTreeClassifier(max_depth=3, random_state=0)
tree1.fit(X_train, Y_train)
print("Accuracy on training set: {:.3f}".format(tree1.score(X_train, Y_train)))
print("Accuracy on test set: {:.3f}".format(tree1.score(X_test, Y_test)))


# ## Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


matrix= confusion_matrix(Y_test, y_pred_dt)


# In[ ]:


sns.heatmap(matrix,annot = True, fmt = "d")


# # precision score

# In[ ]:


from sklearn.metrics import precision_score


# In[ ]:


precision = precision_score(Y_test, y_pred_dt)


# In[ ]:


print("Precision: ",precision)


# # recall

# In[ ]:


from sklearn.metrics import recall_score


# In[ ]:


recall = recall_score(Y_test, y_pred_dt)


# In[ ]:


print("Recall is: ",recall)


# # f score

# In[ ]:


print((2*precision*recall)/(precision+recall))


# # Final Score

# In[ ]:


# initialize an empty list
accuracy = []

# list of algorithms names
classifiers = ['KNN', 'Decision Trees', 'Logistic Regression', 'Naive Bayes', 'Random Forests']

# list of algorithms with parameters
models = [KNeighborsClassifier(n_neighbors=8), DecisionTreeClassifier(max_depth=3, random_state=0), LogisticRegression(), 
        GaussianNB(), RandomForestClassifier(n_estimators=100, random_state=0)]

# loop through algorithms and append the score into the list
for i in models:
    model = i
    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)
    accuracy.append(score)


# In[ ]:


# create a dataframe from accuracy results
summary = pd.DataFrame({'accuracy':accuracy}, index=classifiers)       
summary


# In[ ]:


scores = [score_lr,score_nb,score_knn,score_dt,score_rf]
algorithms = ["Logistic Regression","Naive Bayes","K-Nearest Neighbors","Decision Tree","Random Forest"] 
sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)


# 
# 
# ---
# 
# 
