#!/usr/bin/env python
# coding: utf-8

# # Exploratory data analysis

# In[ ]:


# importing libraries

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc


# In[ ]:


data = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")

data.drop("sl_no", axis=1, inplace=True) # Removing Serial Number


# In[ ]:


print("Number of rows in data :",data.shape[0])
print("Number of columns in data :", data.shape[1])


# In[ ]:


data.head()


# In[ ]:


data.info()


# **We have 7 columns with real values and 8 with object datatype**
# 
# **It is clear that only salary has null columns. Let's see how much?**

# In[ ]:


# Percentage of null values present in salary column

p = data['salary'].isnull().sum()/(len(data))*100

print(f"Salary column has {p.round(2)}% null values.")


# **This means that around 31% candidates were not placed which is sad but let's see what were the reasons :)**

# In[ ]:


data.describe()


# In[ ]:


# Let's peek at the object data types seperately

data.select_dtypes(include=['object']).head()


# <b>We have a total of 8 columns with non-integer or float data. Let's now look at the amount of classes (unique values) these columns.<b>

# In[ ]:


# getting the object columns
object_columns = data.select_dtypes(include=['object']).columns

# iterating over each object type column
for col in object_columns:
    print('-' * 40 + col + '-' * 40 , end='-')
    display(data[col].value_counts())


# <h3>OBSERVATION:</h3>
# 
# Looks like except for hsc_s and degree_t with 3 classes, all other have 2 classes each and also we can see that this data is slightly imbalanced as we have 148 placed students and 67 not placed students.

# ## Exploring the columns

# ### Gender (Male, Female)

# In[ ]:


sns.countplot("gender", data = data)
plt.show()


# In[ ]:


# Let's look at more important plot i.e gender vs status (target)

sns.countplot("gender", hue="status", data=data)
plt.show()


# <h3>OBSERVATIONS:</h3>
# 
# **(I)** The number of **male students are almost double as compared to female**.<br>
# **(II)** As the fraction of placed vs not placed for female candidates is significantly low as compared to male candidates thus we can conclude **male candidates are accepted more often than female.**

# ### ssc_b (Board of Education - Secondary)

# In[ ]:


sns.countplot("ssc_b", data = data)
plt.show()


# In[ ]:


# Let's see the impact of taking a spcific board in 10th grade on placements

sns.set(rc={'figure.figsize':(8.7,5.27)})

sns.countplot("ssc_b", hue="status", data=data)
plt.show()


# <h3>OBSERVATIONS:</h3>
# 
# **(I)** There is **count of central board students is very high as compared to all other boards**.It might be because **central board is easy**.
# 
# **(II)** The count of placed students from central board is little more than others category which doesn't say much.
# 

# ### ssc_p (Secondary Education - 10th grade)

# In[ ]:


# Let's plot percentage vs status to see how much effect they make

sns.barplot(x="status", y="ssc_p", data=data)


# In[ ]:


# Let's see the how much percentage was scored by students in different boards

sns.barplot(x="ssc_b", y="ssc_p", data=data)


# <h3>OBSERVATIONS:</h3>
# 
# 
# **(I)** Percentage in secondary education has a clear impact on placements.**Higher percentage candidates have a very good chance of getting placed**.
# 
# **(II)** Looks like there is not impace of boards on percentages of students.

# ### hsc_b (Board of Education - Higher)

# In[ ]:


# Let's look at how many students opted for central this time?

sns.countplot("hsc_b", data = data)
plt.show()


# In[ ]:


# Let's see the impact of a spcific board on placements

sns.set(rc={'figure.figsize':(8.7,5.27)})

sns.countplot("hsc_b", hue="status", data=data)
plt.show()


# <h3>OBSERVATIONS:</h3>
# 
# 
# **(I)** Looks like **more number of candidates opted for others for 12th grade as compared to 10th grade.**
# 
# **(II)** This time also not much difference between in the fraction of placed candidates in respective boards. Thus, **board doesn't matter in placements**.

# ### hsc_p (Percentage- 12th Grade)

# In[ ]:


# Let's plot percentage vs status to see how much effect they make

sns.barplot(x="status", y="hsc_p", data=data)


# In[ ]:


# Let's see the how much percentage was scored by students in 12th grade in different boards

sns.barplot(x="hsc_b", y="hsc_p", data=data)


# <h3>OBSERVATIONS:</h3>
# 
# 
# **(I)** Percentage in higher secondary education also has a clear impact on placements. **Higher percentage candidates have a very good chance of getting placed**.<br>
# 
# **(II) Board isn't a determinant in defining how much precentage students score.**<br>
# 
# Thus, it turns out that a piece of paper can definately decide your future atleast for placements, so study hard!

# ### hsc_s (Specialization in Higher Secondary Education) 

# In[ ]:


# Let's see what count of students opted for in 12th grade

sns.countplot("hsc_s", data=data)


# In[ ]:


# Let's look at how well each specialisation students performed

ax = sns.barplot(x="hsc_s", y="hsc_p", data=data)


# In[ ]:


# Let's see the impact of taking a spcific branch on placements

sns.countplot("hsc_s", hue="status", data=data)


# <h3>OBSERVATIONS:</h3>
# 
#     
# **(I)** The **most popular branch turns out to be commerce** or maybe as most of students get average marks so they were admitted to got commerce on based of their marks. **Science is the second most popular and the least popular is arts.**
# 
# **(II)** **Almost every branch students performed equally but commerce students have slightly better score than other two.**
# 
# **(III)** Looking at the fraction of placed and not placed we can say that **science branch students have more chance of getting placed than commerce students and most around 45% of the students in arts are not placed**

# ### degree_t & degree_p (Degree Type and Degree percentage)

# In[ ]:


# Let's see what count of students opted for what after 12th grade

sns.countplot("degree_t", data=data)


# In[ ]:


# Let's look at how well each field students performed

sns.barplot(x="degree_t", y="degree_p", data=data)


# In[ ]:


# Let's see the impact of taking a field on placements

sns.countplot("degree_t", hue="status", data=data)


# <h3>OBSERVATION:</h3>
# 
# 
# **(I)** The students opted for following fields:
# 
# <ol>
# <li>Science and Technology (must be science students)</li>
# <li>Commerce and management (might be a mixture of commerce and Arts)</li>
# <li>Others</li>
# </ol>
# 
# **(II)** There is **not much difference in performace of students from Science and Commerce** but there but **students who opted for "Others" have low performance**
# 
# **(III)** Looks like **Commerce and Science degree students are preffered by companies** which is obvious. **Students who opted for Others have very low placement chance.**

# ### workex (Work Experience)

# In[ ]:


# Let's see if the work experience impacts on placements or not

data['status'] = data['status'].map( {'Placed':1, 'Not Placed':0})

sns.barplot(x="workex", y="status", data=data)


# <h3>OBSERVATION:</h3>
# 
# Companies prefer candidates with work experience so the **students with internships and past job experience have better chances of being placed.**

# ### etest_p (Employability test percentage)

# In[ ]:


sns.barplot(x="status", y="etest_p", data=data)


# <h3>OBSERVATION:</h3>
# 
# We can see that getting good percentages in **employability test does not guarantee placement** of the candidate.

# ### specialisation (Post Grad - MBA)

# In[ ]:


# Let's see how specialisation effects the placement of candidates

sns.countplot("specialisation", hue="status", data=data)


# <h3>OBSERVATION:</h3>
# 
# Specialisation is a clear indicator in placements. Compared to MktandFin, **Mkt&HR students have low placements**. This might be because there is low requirements for HR in a company. 

# ### mba_p (MBA percentage)

# In[ ]:


sns.barplot(x="status", y="mba_p", data=data)
plt.title("Salary vs MBA Percentage")


# We can see that getting good percentages in **MBA does not guarantee placement** of the candidate.

# ### Salary 

# In[ ]:


# Let's look at the distribution of salary

plt.figure(figsize=(10,5))
sns.distplot(data['salary'], bins=50, hist=False)
plt.title("Salary Distribution")
plt.show()


# **Looking at the distribution we can say that the most of the students get a package between 200k-400k and most salaries above 400,000 are outliers.**

# In[ ]:


sns.barplot(x="gender", y="salary", data=data)
plt.title("Salary vs gender")


# <h3>OBSERVATION:</h3>
# 
# 
# **Male candidates are making more money as compared to female candidates.**

# In[ ]:


sns.violinplot(x=data["gender"], y=data["salary"], hue=data["specialisation"])
plt.title("Salary vs Gender based on specialisation")


# <h3>OBSERVATIONS:</h3>
# 
# **(I)** Salary column for male candidates seems to have more outliers than females which means that a lot **more male candidates got more than the average CTC.**
# 
# **(II) Mean salary is somewhere around 220k**.
# 
# **(III) Mkt&Fin students are given higher salaries as compared to Mkt&HR.**

# In[ ]:


sns.violinplot(x=data["gender"], y=data["salary"], hue=data["workex"])
plt.title("Gender vs Salary based on work experience")


# <h3>OBSERVATIONS:</h3>
# 
# **(I)** Work Experience is a clear indicator as **more work experience results in higher CTC jobs.**
# 
# **(II) The maximum salary in male candidates with experience is >1M and for female it is ~700k.
# The maximum salary in male candidates without experience is ~550k and for female it is ~430k.**

# In[ ]:


sns.violinplot(x=data["gender"], y=data["salary"], hue=data["ssc_b"])
plt.title("Salary vs Gender based on Board in 10th grade")


# <h3>OBSERVATION:</h3>
# 
# 
# Both Male and Female candidates from Central board got higher CTC as compared to other boards thus we can that central board in 10th grade might fetch you higher CTCs.

# In[ ]:


sns.violinplot(x=data["gender"], y=data["salary"], hue=data["hsc_b"])
plt.title("Salary vs Gender based on Board in 12th grade")


# <h3>OBSERVATION:</h3>
# 
# Male candidates from Central board got higher CTC as compared to other boards whereas this was totally opposite in case of female candidates thus there is not much guarantee that either of the board will fetch higher CTCs.

# In[ ]:


sns.violinplot(x=data["gender"], y=data["salary"], hue=data["degree_t"])
plt.title("Salary vs Gender based on Degree Type")


# <h3>OBSERVATIONS:</h3>
# 
# **(I)** Both male and female candidate got high CTCs choosing Comm&Mgmt as their degree.
# 
# **(II)** Male candidates from Sci&Tech got high CTCs as compared to Female candidates.
# 
# **(III)** None of the male candidates got placed from "Others" category whereas for female candidates the package is close to what female Sci&Tech candidates got.

# # Conclusions Drawn

# <ol>
# <li>More male candidates got placed as compared to female candidates.</li>
# <li>Male Candidates got higher CTCs as compared to female candidates.</li>
# <li>Type of Board choosen does not have any effect on placements thus we can drop in preprocessing steps.</li>
# <li>Most of the students preferred Central board in 10th grade whereas other boards in 12th grade.</li>
# <li>Candidates with higher percentages have better chance of placements.</li>
# <li>Choosing Science and Commerce as Specialisation seems to have perk when it comes to placments.</li>
# <li>Maximum package was bagged by male candidate from Mkt&Fin branch which is around 940k.</li>
# <li>Commerce is the most popular branch among candidates.</li>
# <li>Mean CTC is around 220k for male and female candidates individually.</li>
# <li>Choosing Sci&Tech and Comm&Mngmt as degree will fetch you higher CTCs.</li>
# <li>Mkt&Fin major have higher salaries and more placement chance as compared to Mkt&HR.</li>
# <li>Employability test percentage and MBA percentage does not effect the placements</li>

# # Preprocessing

# **The data doesn't have any missing values except for salary (not useful - we will see why) so there is not much data cleaning but we will have to do encoding of categorical variables, note that - target(status) is already been encoded during EDA.**
# 
# **Before that, we will drop both secondary and higher secondary boards as discussed in conclusions of EDA.**
# 
# **We will also remove salary column as it is clearly depends on whether the candidate got placed or not so it is not at all useful and can fool us by giving 100%. (BIG MISTAKE!)**

# In[ ]:


# Dropping useless columns

data.drop(['ssc_b','hsc_b', 'salary'], axis=1, inplace=True)


# ### Encoding
# 
# We have **gender, hsc_s, degree_t, workex and specialisation** as categorical so let's encode them.

# In[ ]:


# Using simple binary mapping on two class categorical variables (gender, workerx, specialisation)

data["gender"] = data.gender.map({"M":0,"F":1})
data["workex"] = data.workex.map({"No":0, "Yes":1})
data["specialisation"] = data.specialisation.map({"Mkt&HR":0, "Mkt&Fin":1})


# In[ ]:


# Using get dummies for 3 class categorical variables (hsc_s and degree_t)

for column in ['hsc_s', 'degree_t']:
    dummies = pd.get_dummies(data[column])
    data[dummies.columns] = dummies


# In[ ]:


# Now let's clean up the left overs (already encoded so no use now)

data.drop(['degree_t','hsc_s'], axis=1, inplace=True)


# In[ ]:


# Now let us look at the data

data.head()


# **Our data is now completely encoded. GOOD JOB!**

# In[ ]:


# Let's do a sanity check by peeking at the data

data.head()


# In[ ]:


# Let's plot correlation matrix to find out less correlated variable to drop them

cor=data.corr()
plt.figure(figsize=(14,6))
sns.heatmap(cor,annot=True)


# In[ ]:


# From the correlation matrix we can see that some of the features are not much useful like "Others" and "Arts" which are negatively 
# correlated as well as have low value.

# Another reason to remove these variables is the so called Dummy variable trap which occurs when we do encoding of multiclass features

data.drop(['Others', 'Arts'], axis=1, inplace=True)


# ### Splitting the data 

# In[ ]:


# target vector
y = data['status']

# dropping as it is not a predictor
data.drop('status', axis = 1, inplace = True)

# scaling the data so as to get rid of any dramatic results during modelling
sc = StandardScaler()

# predictors
X = sc.fit_transform(data)

# Let us now split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)


print("X-Train:",X_train.shape)
print("X-Test:",X_test.shape)
print("Y-Train:",y_train.shape)
print("Y-Test:",y_test.shape)


# # Model Prediction
# 
# **Problem:** Predict whether the candidate will be placed or not based on some predictors.
# 
# **Nature of Problem:** As the target is a binary data thus it is a **binary classification problem.**

# ### Logistic Regression
# 
# **Let's apply logistic regression as it is a classification algorithm that works well with binary data.**

# In[ ]:


# creating our model instance
log_reg = LogisticRegression()

# fitting the model
log_reg.fit(X_train, y_train)


# In[ ]:


# predicting the target vectors

y_pred=log_reg.predict(X_test)


# In[ ]:


# creating confusion matrix heatmap

conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred))
fig = plt.figure(figsize=(10,7))
sns.heatmap(conf_mat, annot=True, annot_kws={"size": 16}, fmt='g')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# ### Some Insights:
# 
# Our confusion Matrix looks decent. We have correctly predicted 42 (placed) + 14 (not-placed) correct predictions and 7 (not placed as placed) + 2(placed as not-placed) incorrect predictions.
# 
# We need to decrease these incorrect predictions because a good candidate can be rejected (false positive) [Type I error] and a unfit candidate can be selected (false negatives) [Type II Error]. 

# In[ ]:


# getting precision, recall and f1-score via classification report

print(classification_report(y_test, y_pred))


# **Looking at the precision, recall and f1_score we can saw that our Logistic Regression model did fairly well!**

# In[ ]:


# let's look at our accuracy

accuracy = accuracy_score(y_pred, y_test)

print(f"The accuracy on test set using Logistic Regression is: {np.round(accuracy, 3)*100.0}%")


# **We achieved 86% without doing any tuning so it means we did preprocessing steps really well. Great!**

# In[ ]:


# plotting the ROC curve

auc_roc = roc_auc_score(y_test, log_reg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, log_reg.predict_proba(X_test)[:,1])

plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label='Average ROC curve (area = {0:0.3f})'.format(auc_roc))
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', 
         label= 'Average ROC curve (area = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


# calculate auc 
auc_score = auc(fpr, tpr)
print(f"Our auc_score came out to be {round(auc_score, 3)}.")


# ###  Decision Tree Classifier
# 
# **Let's try some decision trees now and see how well they perform but as Decision trees are easy to overfit so I will use K-FOLD CV first to find the best depth.**

# In[ ]:


# creating a list of depths for performing Decision Tree
depth = list(range(1,10))

# list to hold the cv scores
cv_scores = []

# perform 10-fold cross validation with default weights
for d in depth:
  dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
  scores = cross_val_score(dt, X_train, y_train, cv=10, scoring='accuracy', n_jobs = -1)
  cv_scores.append(scores.mean())

# finding the optimal depth
optimal_depth = depth[cv_scores.index(max(cv_scores))]
print("The optimal depth value is: ", optimal_depth)


# **Looks like max_depth = 8 is good for our model**
# 
# **Lets plot some graph to see what was the trend of our accuracies.**

# In[ ]:


# plotting accuracy vs depth
plt.plot(depth, cv_scores)
plt.xlabel("Depth of Tree")
plt.ylabel("Accuracy")
plt.title("Accuracy vs depth Plot")
plt.grid()
plt.show()

print("Accuracy scores for each depth value is : ", np.round(cv_scores, 3))


# **This doesn't look awesome but isn't bad either so let's trust our CV and use the optimal max_depth = 8 to train our model.**

# In[ ]:


# create object of classifier
dt_optimal = DecisionTreeClassifier(criterion="gini", max_depth=optimal_depth, random_state=42)

# fit the model
dt_optimal.fit(X_train,y_train)

# predict on test vector
y_pred = dt_optimal.predict(X_test)

# evaluate accuracy score
accuracy = accuracy_score(y_test, y_pred)*100
print(f"The accuracy on test set using optimal depth = {optimal_depth} is {np.round(accuracy, 3)}%")


# **We achieved 86% accuracy which is similiar to what we achieved using logistic regression so they seem to work equally well.**
# 
# **What if we could combine the power of our two heroes to get a superhero? Sounds weird? It is!**

# ### Ensemble Modelling
# 
# **We will train a voting classifier using our previously trained logistic regeression and Decision tree model**

# In[ ]:


# creating a list of our models
ensembles = [log_reg, dt_optimal]

# Train each of the model
for estimator in ensembles:
    print("Training the", estimator)
    estimator.fit(X_train,y_train)


# In[ ]:


# Find the scores of each estimator

scores = [estimator.score(X_test, y_test) for estimator in ensembles]

scores


# **We didn't do anything awesome yet. We just made a list of both of our previously trained classifiers. Let's add some awesomeness!**

# In[ ]:


# Training a voting classifier with hard voting and using logistic regression and decision trees as estimators

from sklearn.ensemble import VotingClassifier

named_estimators = [
    ("log_reg",log_reg),
    ("dt_tree", dt_optimal),

]


# In[ ]:


# getting an instance for our Voting classifier

voting_clf = VotingClassifier(named_estimators)


# In[ ]:


# fit the classifier

voting_clf.fit(X_train,y_train)


# In[ ]:


# Let's look at our accuracy
acc = voting_clf.score(X_test,y_test)

print(f"The accuracy on test set using voting classifier is {np.round(acc, 4)*100}%")


# **We went from 86.4% to 92.3% accuracy score!**
# 
# **If that isn't amazing I don't know what is.**

# ## Thanks for reading!
