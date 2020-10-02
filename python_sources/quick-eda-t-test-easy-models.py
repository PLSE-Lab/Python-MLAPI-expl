#!/usr/bin/env python
# coding: utf-8

# # Students Performance in Exams: Quick EDA & Easy Models
# 
# In this kernel I would like to demonstrate some of the techniques for EDA and consolidate my knowledge in this area as an MBA. 
# 
# The starting point is a data set with a sample size of 1000 children, whose results have been recorded in some tests. In addition, their ethnic group, gender, education of parents and participation in the preparation course were recorded. An interesting variable is "Lunch", which indicates whether the child participates in the reduced lunch program. This is, if I have understood it correctly, a program that supports socially disadvantaged families by providing lunch for the children at school.

# First of all we do the boring stuff like importing plugins, reading dataframes, checking for missing values,, and so on. You know the game.

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import warnings  
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/StudentsPerformance.csv')
df.head()


# In[ ]:


#Check for missing values
if df.isnull().any().unique() == False:
    print("No missing values")
else: 
    print(df.isnull().any())


# ## Distribution
# 
# Now let's draw the pair plot and have a look at the distribution of the data. We immediately recognize that we have a skewness in the data and in the scatterplot we also see some outliers. Let's take a closer look at this
# 
# ### Skewness & Kurtosis
# First we look at the skewness. On the plot we can already see that we have to do with left-skewed data. A look at the values confirms this. If we look at the kurtosis we also see that it deviates strongly from the normal distribution, which is also confirmed by the values. However, it should be noted that real data is never completely normally distributed and the values are within a normal tolerance range. However, we can try to normalize them a bit. Especially since we have some outliers, the distance of the outliers can bring us closer to a normal distribution.
# 
# ### Outliers
# In addition to the scatterplot, we also look at a boxplot for the three variables in order to be able to recognize outliers. The box plot also confirms our assumption. We remove the outliers using the interquantile distance between the 1st and 3rd quantiles. This is a frequently used method.
# 
# ### Results
# As you can see, there are no more outliers on the box plots. Also, the values for skewness and kurtosis have improved. On the pairplot the distribution doesn't look as left-skewed anymore.

# In[ ]:


sns.pairplot(df)


# In[ ]:


#Skewness & Kurtosis
#The values for asymmetry and kurtosis between -2 and +2 are considered acceptable in order to prove normal univariate distribution (George & Mallery, 2010)
print("Skewness \n", df.skew(), "\n\nKurtosis\n", df.kurtosis())


# In[ ]:


#Outliers
f, axes = plt.subplots(1, 3, figsize=(15,5))
sns.boxplot(df["math score"], orient='v' ,ax = axes[0])
sns.boxplot(df["reading score"],orient='v' , ax = axes[1])
sns.boxplot(df["writing score"],orient='v' , ax = axes[2])


# In[ ]:


#Remove Outliers
def remove_outlier(col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3-q1
    #Cut-Off
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    #Remove Outliers
    df2 = df.loc[(df[col] > fence_low) & (df[col] < fence_high)]
    return df2
num_rows_org = df.shape[0]
columns = ["math score", "reading score", "writing score"]
for i in columns:
    df = remove_outlier(i) 
num_rows_new = df.shape[0]
print(num_rows_org-num_rows_new,"Outliers Removed")


# In[ ]:


f, axes = plt.subplots(1, 3, figsize=(15,5))
sns.boxplot(df["math score"], orient='v' ,ax = axes[0])
sns.boxplot(df["reading score"],orient='v' , ax = axes[1])
sns.boxplot(df["writing score"],orient='v' , ax = axes[2])


# In[ ]:


#Skewness & Kurtosis
#The values for asymmetry and kurtosis between -2 and +2 are considered acceptable in order to prove normal univariate distribution (George & Mallery, 2010)
print("Skewness \n", df.skew(), "\n\nKurtosis\n", df.kurtosis())


# In[ ]:


sns.pairplot(df)


# ## EDA
# In the next step we would like to have a look at the descriptive statistics and do a quick EDA.
# 
# ### Encoding
# In order to be able to use the nominally and ordinally scaled data, we will first encode them. Since I would like to plot them in a bar chart, I will not encode them as dummy variables. 
# Furthermore, I have another variable calculated, which indicates the average score of all three subjects. I do this on the assumption that students who are good in one subject may also be successful in other subjects. 
# 
# ### Characteristics Distribution
# As you can see in the count plots, the classes are imbalanced. We have some more boys attending school. Also, different ethnic groups are represented differently, which can be due to different reasons. In addition, the parents have very different levels of education. It is clear that most children receive a lunch allowance and most also attend the preparation course.

# In[ ]:


#Encode Variables
from sklearn.preprocessing import LabelEncoder
df['gender'] = LabelEncoder().fit_transform(df["gender"]) #'male' or 'female'
df['race/ethnicity'] = LabelEncoder().fit_transform(df["race/ethnicity"]) #Group A to E
df['parental level of education'] = LabelEncoder().fit_transform(df["parental level of education"])#'bachelor's degree', 'some college', "master's degree","associate's degree", 'high school' or 'some high school'
df['lunch'] = LabelEncoder().fit_transform(df["lunch"]) #'standard' or 'free/reduced'
df['test preparation course'] = LabelEncoder().fit_transform(df["test preparation course"])#'none' or 'completed'

df.head(33)


# In[ ]:


df["overall_score"] = ((df["math score"] + df["reading score"] + df["writing score"]) / 3).astype(int)
median = df["overall_score"].median()
df.head()


# In[ ]:


f, axes = plt.subplots(1, 5, figsize=(25,5))
sns.countplot(df["gender"], ax = axes[0])
sns.countplot(df["race/ethnicity"], ax = axes[1])
sns.countplot(df["parental level of education"], ax = axes[2])
sns.countplot(df["lunch"], ax = axes[3])
sns.countplot(df["test preparation course"], ax = axes[4])


# ### Correlation
# Now let's take a look at the correlation. As already suspected, the grades of the individual subjects correlate strongly with each other, which is why we can assume that good students perform well in all subjects. Unfortunately, the other variables hardly correlate with each other, making it difficult to predict what will affect the notes later on.
# 
# ### Let's Combine Some Attributes
# If we now combine different attributes, we see that slightly more boys get lunch allowance as girls. Gender also has no great influence on participation in the preparation course. Whereby we could check this with a t-test.
# 

# In[ ]:


corr = df.corr()
sns.heatmap(corr)


# In[ ]:


print("Boys with Lunch:", len(df[(df["gender"] == 0)  & (df["lunch"] == 0)]) / len(df[df["gender"] == 0]))
print("Girls with Lunch:", len(df[(df["gender"] == 1)  & (df["lunch"] == 0)]) / len(df[df["gender"] == 1]))

print("Boys with Preparation:", len(df[(df["gender"] == 0)  & (df["test preparation course"] == 0)]) / len(df[df["gender"] == 0]))
print("Girls with Preparation:", len(df[(df["gender"] == 1)  & (df["test preparation course"] == 0)]) / len(df[df["gender"] == 1]))

print("Preparation with Lunch:", len(df[(df["lunch"] == 0)  & (df["test preparation course"] == 0)]) / len(df[df["test preparation course"] == 0]))
print("Preparation without Lunch:", len(df[(df["lunch"] == 1)  & (df["test preparation course"] == 0)]) / len(df[df["test preparation course"] == 1]))


# ### t-test
# If we do an independent t-test, we see that the children who do not receive a lunch allowance achieve better results. This could be explained, for example, by the fact that children from socially disadvantaged families tend to do not receive as much help with homework or learning at home, for example by private tutoring.
# 
# On the other hand, it can be seen that children who attend the course do better than children who do not.
# 
# If we now compare the children who do not receive a subsidy and do not take part in the course because they receive help at home with the preparation for the exam, for example, with the performance of the children who come from socially disadvantaged families and receive help in school, we see that the results are the same. Thus, at first glance, the offers of help seem to work.

# In[ ]:


from scipy.stats import ttest_ind
w_lunch_mean = df[(df["lunch"] == 0)]
wo_lunch_mean = df[(df["lunch"] == 1)]

print(ttest_ind(w_lunch_mean['overall_score'], wo_lunch_mean['overall_score'], nan_policy='omit'))

print('With Lunch:', df[(df["lunch"] == 0)].overall_score.mean())
print('Without Lunch:', df[(df["lunch"] == 1)].overall_score.mean())


# In[ ]:


from scipy.stats import ttest_ind
w_course_mean = df[(df["test preparation course"] == 0)]
wo_course_mean = df[(df["test preparation course"] == 1)]

print(ttest_ind(w_course_mean['overall_score'], wo_course_mean['overall_score'], nan_policy='omit'))

print('With Course:', df[(df["test preparation course"] == 0)].overall_score.mean())
print('Without Course:', df[(df["test preparation course"] == 1)].overall_score.mean())


# In[ ]:


from scipy.stats import ttest_ind
n_lunch_n_course = df[(df["lunch"] == 1)  & (df["test preparation course"] == 1)]
lunch_course = df[(df["lunch"] == 0)  & (df["test preparation course"] == 0)]

print(ttest_ind(n_lunch_n_course['overall_score'], lunch_course['overall_score'], nan_policy='omit'))

print('With Course:', df[(df["lunch"] == 0)  & (df["test preparation course"] == 0)].overall_score.mean())
print('Without Course:', df[(df["lunch"] == 0)  & (df["test preparation course"] == 0)].overall_score.mean())


# ## Prediction
# 
# Now let's try to find a simple model that helps us predict how the student will perform. 
# As you can see, if you keep the other scores in the model, you can predict the math score. This is logical because we have a high correlation between these variables. If we take out the scores, however, the accuracy of the model deteriorates.
# 
# ### Dummy Variable
# Now we simplify the problem and let only the model predict whether the student is good or bad. To do this, we create a dummy variable that matches the division using the median. I like to use the median because it is more robust than the arithmetic mean.As you can see now, the Random Forest model is a bit better now. I only left the Linear Regressor for comparison, but it rarely works with binary variables.
# If we take a quick look at the median again, we see that the student's performance could also depend on the ethnicity, which has a variety of reasons.
# Without going into these reasons, since I am not a sociologist, I try to form groups to improve the model. As we can see, however, it improves only slightly
# 

# In[ ]:


from sklearn.model_selection import cross_val_score, RepeatedKFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression

x = df.drop(["math score","overall_score"], axis = 1)
y = df["math score"]

score_lr = cross_val_score(LinearRegression(), x, y, cv = RepeatedKFold(n_repeats = 10))
score_rf = cross_val_score(RandomForestRegressor(n_estimators = 100), x, y, cv = RepeatedKFold(n_repeats = 10))
print("Math Score:\nLinear Regression:",np.mean(score_lr),"\nRandom Forest:",np.mean(score_rf))


# In[ ]:


x = df.drop(["math score", "writing score", "reading score", "overall_score"], axis = 1)
y = df["overall_score"]

score_lr = cross_val_score(LinearRegression(), x, y, cv = RepeatedKFold(n_repeats = 10))
score_rf = cross_val_score(RandomForestRegressor(n_estimators = 100), x, y, cv = RepeatedKFold(n_repeats = 10))
print("Overall_Score:\nLinear Regression:",np.mean(score_lr),"\nRandom Forest:",np.mean(score_rf))


# In[ ]:


df["overall_median"] = df["overall_score"]
df.overall_median[(df["overall_score"] >= median)] = 1
df.overall_median[(df["overall_score"] < median)] = 0
df.head()


# In[ ]:


x = df.drop(["math score", "writing score", "reading score", "overall_score", "overall_median"], axis = 1)
y = df["overall_median"]

score_lr = cross_val_score(LinearRegression(), x, y, cv = RepeatedKFold(n_repeats = 10))
score_rf = cross_val_score(RandomForestClassifier(n_estimators = 100), x, y, cv = RepeatedKFold(n_repeats = 10))
print("Good/Bad Student:\nLinear Regression:",np.mean(score_lr),"\nRandom Forest:",np.mean(score_rf))


# In[ ]:


x = df.drop(["math score", "writing score", "reading score", "overall_score", "overall_median", "parental level of education"], axis = 1)
y = df["overall_median"]

score_lr = cross_val_score(LinearRegression(), x, y, cv = RepeatedKFold(n_repeats = 10))
score_rf = cross_val_score(RandomForestClassifier(n_estimators = 100), x, y, cv = RepeatedKFold(n_repeats = 10))
print("Good/Bad Student:\nLinear Regression:",np.mean(score_lr),"\nRandom Forest:",np.mean(score_rf))


# In[ ]:


sns.countplot(x = "overall_median", hue = "race/ethnicity", data = df)


# In[ ]:


df["ethnicity"] = df["race/ethnicity"]
df.ethnicity[(df["race/ethnicity"] <= 2)] = 0
df.ethnicity[(df["race/ethnicity"] > 2)] = 1
df.head()


# In[ ]:


x = df.drop(["math score", "writing score", "reading score", "overall_score", "overall_median", "parental level of education", "race/ethnicity"], axis = 1)
y = df["overall_median"]

score_lr = cross_val_score(LinearRegression(), x, y, cv = RepeatedKFold(n_repeats = 10))
score_rf = cross_val_score(RandomForestClassifier(n_estimators = 100), x, y, cv = RepeatedKFold(n_repeats = 10))
print("Good/Bad Student:\nLinear Regression:",np.mean(score_lr),"\nRandom Forest:",np.mean(score_rf))


# ## Compare Different Models
# 
# ### Parameter Tuning
# Now I played around with the models and tried to improve them by parameter tuning. However, you can clearly see that with an accuracy of 66% it's over. Nevertheless, you can see that some estimators could improve, but didn't cross the threshold. 
# 
# ### Feature Importance
# If we look at the feature importance we can clearly see that lunch and the income of the parents expressed through it as well as the participation in the preparation course have an important influence on the predictability of the grade.
# 
# ### Model Fit
# The ROC curve rises a bit before the error rate rises and looks relatively good. We can therefore assume that the models perform relatively well.
# On the other hand, there are clear differences between the models in the recall curve. If we were to deal with a real model, we could now prevent false-positives and select the best model.
# Finally, we consider the learning curve for the RFC. This clearly shows that with increasing sample size the model accuracy decreases again, which probably causes an overfitting in our case. 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

x = df.drop(["math score", "writing score", "reading score", "overall_score", "overall_median", "parental level of education", "race/ethnicity"], axis = 1)
y = df["overall_median"]

from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(x)
x = scaler.transform(x)
x = np.clip(x, a_min=-1, a_max=1)

models = [LogisticRegression(), GradientBoostingClassifier(), RandomForestClassifier(),AdaBoostClassifier()]

scores = []
names = []

for alg in models: 
    score = cross_val_score(alg, x, y, cv = RepeatedKFold(n_repeats = 3))
    scores.append(np.mean(score))    
    names.append(alg.__class__.__name__)
    print(alg.__class__.__name__, "trained")
    
sns.set_color_codes("muted")
sns.barplot(x=scores, y=names, color="g")

plt.xlabel('Accuracy')
plt.title('Classifier Scores')
plt.show()


# In[ ]:


lr_params = dict(     
    C = [n for n in range(1, 10)],     
    tol = [0.0001, 0.001, 0.001, 0.01, 0.1, 1],  
)

lr = LogisticRegression(solver='liblinear')
lr_cv = GridSearchCV(estimator=lr, param_grid=lr_params, cv=5) 
lr_cv.fit(x, y)
lr_est = lr_cv.best_estimator_
print(lr_cv.best_score_)


# In[ ]:


gb_params = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(2, 5)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(10, 40, 5)],
)

gb = GradientBoostingClassifier()
gb_cv = GridSearchCV(estimator=gb, param_grid=gb_params, cv=5) 
gb_cv.fit(x, y)
gb_est = gb_cv.best_estimator_
print(gb_cv.best_score_)


# In[ ]:


forest_params = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(4, 11)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(10, 40, 5)],
)

forest = RandomForestClassifier()
forest_cv = GridSearchCV(estimator=forest, param_grid=forest_params, cv=5) 
forest_cv.fit(x, y)
forest_est = forest_cv.best_estimator_
print(forest_cv.best_score_)


# In[ ]:


ada_params = dict(          
    learning_rate = [0.05, 0.1, 0.15, 0.2],  
    n_estimators = [n for n in range(10, 40, 5)],
)

ada = AdaBoostClassifier()
ada_cv = GridSearchCV(estimator=ada, param_grid=ada_params, cv=5) 
ada_cv.fit(x, y)
ada_est = ada_cv.best_estimator_
print(ada_cv.best_score_)


# In[ ]:


cv_models = [lr_est, gb_est, forest_est, ada_est]

scores_cv = []
names_cv = []

for alg in cv_models: 
    score = cross_val_score(alg, x, y, cv = RepeatedKFold(n_repeats = 3))
    scores_cv.append(np.mean(score))    
    names_cv.append(alg.__class__.__name__)
    print(alg.__class__.__name__, "trained")
    
sns.set_color_codes("muted")
sns.barplot(x=scores_cv, y=names_cv, color="b")
sns.barplot(x=scores, y=names, color="g")

plt.xlabel('Accuracy')
plt.title('Classifier Scores')
plt.show()


# In[ ]:


col = df.drop(["math score", "writing score", "reading score", "overall_score", "overall_median", "parental level of education", "race/ethnicity"], axis = 1)
f = plt.figure(figsize=(5,10))

plt.subplot(3, 1, 1)
(pd.Series(forest_est.feature_importances_, index=col.columns).nlargest(10).plot(kind='barh') )

plt.subplot(3, 1, 2)
(pd.Series(gb_est.feature_importances_, index=col.columns).nlargest(10).plot(kind='barh') )

plt.subplot(3, 1, 3)
(pd.Series(ada_est.feature_importances_, index=col.columns).nlargest(10).plot(kind='barh') )


# In[ ]:


#ROC Curve
from sklearn.metrics import roc_curve, precision_recall_curve
fpr_model, tpr_model, thresholds_model = roc_curve(y, lr_cv.predict_proba(x)[:,1])
plt.plot(fpr_model, tpr_model, label = "LogisticRegression")

fpr_knn, tpr_knn, thresholds_knn = roc_curve(y, gb_cv.predict_proba(x)[:,1])
plt.plot(fpr_knn, tpr_knn, label = "GB")

fpr_knn, tpr_knn, thresholds_knn = roc_curve(y, forest_cv.predict_proba(x)[:,1])
plt.plot(fpr_knn, tpr_knn, label = "RFC")

fpr_rfc, tpr_rfc, thresholds_rfc = roc_curve(y, ada_cv.predict_proba(x)[:,1])
plt.plot(fpr_rfc, tpr_rfc, label = "ADA")
plt.xlabel("P(FP)")
plt.ylabel("P(TP)")
plt.legend(loc = "best")


# In[ ]:


#Recall Curve
precision_model, recall_model, thresholds_model = precision_recall_curve(y, lr_cv.predict_proba(x)[:,1])
plt.plot(precision_model, recall_model, label = "LogisticRegression")

precision_gb, recall_gb, thresholds_gb = precision_recall_curve(y, gb_cv.predict_proba(x)[:,1])
plt.plot(precision_gb, recall_gb, label = "GB")

precision_rfc, recall_rfc, thresholds_rfc = precision_recall_curve(y, forest_cv.predict_proba(x)[:,1])
plt.plot(precision_rfc, recall_rfc, label = "RFC")

precision_ada, recall_ada, thresholds_ada = precision_recall_curve(y, ada_cv.predict_proba(x)[:,1])
plt.plot(precision_ada, recall_ada, label = "Ada")


# In[ ]:


#Learning Curve
from sklearn.model_selection import learning_curve
from sklearn.utils import shuffle
x, y = shuffle(x, y)

train_sizes_abs, train_scores, test_scores = learning_curve(RandomForestClassifier(), x, y)
plt.plot(train_sizes_abs, np.mean(train_scores, axis = 1))
plt.plot(train_sizes_abs, np.mean(test_scores, axis = 1))

