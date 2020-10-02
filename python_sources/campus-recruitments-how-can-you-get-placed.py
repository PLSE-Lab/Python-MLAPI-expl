#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
sns.set(style="darkgrid")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# helper function for printing unique values in columns 
def print_unique(df, col):
    print(f"The {col} column contains:")
    u_vals = df[col].unique()
    for i, val in enumerate(u_vals):
        print(f"{i+1}. {val}")


# ## About the Dataset
# I found this dataset [here](https://www.kaggle.com/benroshan/factors-affecting-campus-placement) on Kaggle.
# 
# Campus Recruitment is an obstacle that almost all Engineering students face at some point in their lives. (Except of course for those special snowflakes that decide to pursue higher studies instead). As a final year Computer Science Student, I was instantly drawn to this dataset in hopes of not only understanding the general trend in the industry but also of reassuring myself that I was not a lost cause. Although this dataset is from an MBA college, I think it can still be used to extract valuable information about how one's academic choices can impact their placements.
# 
# A quick glance at the dataset description on Kaggle tells us that it has the following columns:
# 1. `sl_no` : Serial Number
# 2. `gender` : Gender- Male='M',Female='F'
# 3. `ssc_p` : Secondary Education percentage- 10th Grade
# 4. `ssc_b` : Board of Education- Central/ Others
# 5. `hsc_p` : Higher Secondary Education percentage- 12th Grade
# 6. `hsc_b` : Board of Education- Central/ Others
# 7. `hsc_s` : Specialization in Higher Secondary Education
# 8. `degree_p` : Degree Percentage
# 9. `degree_t` : Under Graduation(Degree type)- Field of degree education
# 10. `workex` : Work Experience
# 11. `etest_p` : Employability test percentage ( conducted by college)
# 12. `specialisation` : Post Graduation(MBA)- Specialization
# 13. `mba_p` : MBA percentage
# 14. `status` : Status of placement- Placed/Not placed
# 15. `salary` : Salary offered by corporate to candidates
# 
# Let us quickly view the first few rows and extract some prelimindary information about the dataset.

# In[ ]:


# Loading in the dataset and viewing first few rows 
df = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
df.head()


# In[ ]:


df.describe()


# ### Observations: 
# It seems that we have some big outliers in the salary. The mean is about 2.9 lakhs with a standard deviation of about 90,000 but we have some salary that is over 9 Lakhs, which is more than 6 standard deviations away from the mean!
# 
# Looks like we have some missing values in the "salary column". This is probably for the students who were not placed but let us just check to make sure that the students who are placed all have their salary listed.

# In[ ]:


# Check for missing salaries
df.query("status == 'placed'")["salary"].isnull().any()


# Looks like all the placed students all have their salaries listed so we could try and predict that down the line. However, we will only have 148 samples to work with in that case. Let's go ahead with some analysis for now and we'll cross that bridge when we get to it.

# # Business Understanding
# The first step of the CRISP-DM process is to develop a **Business Understanding**. This means asking relevant questions that we want the data to answer. After examining the data, I am particularly interested in the `Status` (whether or not the student was placed) and `Salary` (How much the student is paid by the corporate). The data available on hand makes it possible to ask questions such as:  
# 
# * Does the board of education influence placements? 
# 
# * Is there a gender bias when it comes to salaries offered to new recruits? 
# 
# * Do companies prefer candidates with prior work experience? 
# 
# * Which combination of degree/specialization has the highest salary?

# # Data Understanding 
# 
# Let us  break down these questions and others and try to answer them by analysing and visualizing our data.

# In[ ]:


# Checking for correlations among columns
plt.subplots(figsize=(10,8))
sns.heatmap(df.corr(), annot=True);


# From the above heatmap, there doesnt seem to be any significant correlations present in the data.
# 
# We do however, see some weak correlation between the `hsc_p`, `ssc_p` and `degree_p` columns. But this is to be expected as a student who performed well in their 10th is likely to continue to work hard and perform well in their subsequent academic endeavours. The inverse may also be true.
# 
# We do see some slight correlation between salary and `mba_p` and `etest_p`. While it makes intuitive sense to me that since the recruitment is for MBA graduates, their MBA scores and E-Test scores (Employability test) might have a correlation with their chances of getting placed or salary, the data on hand does not provide us with strong enough evidence to confirm or deny the same.

# ## Question 1 
# ### Does the board of education affect placements? 
# Reasoning: I find this question particularly interesting because the general trend that I have observed in new parents these days is their increasing preference to enrol their kids into the central board than state. This dataset allows us to investigate the following two questions: 
# 
# a. Are kids from CBSE or higher more likely to get placed than those that are not?  
# b. Are these kids more likely to get a higher salary?

# In[ ]:


# Check the number of unique values 
print_unique(df, "ssc_b")
print_unique(df, "hsc_b")


# In[ ]:


# Create filter for SSC Central board 
ssc_central = df.ssc_b == "Central"

# Create filter for HSC Central board 
hsc_central = df.hsc_b == "Central"

both = ssc_central & hsc_central
neither = ~ssc_central & ~hsc_central

df.groupby(["ssc_b", "hsc_b"]).status.value_counts().reset_index(name="count")


# In[ ]:


# Vizualization of board vs placement rate 
grouped_data = df.groupby(["ssc_b", "hsc_b"]).status.value_counts(normalize=True).unstack(2)
grouped_data.plot.bar(title = "Placements by Board of Education", rot = 0, figsize = (8,6)).set_xlabel("Board of Education for 10th and 12th");


# From the above plots, it might seem that those who took a combination of Central and Others between their 10th and 12th grades were less likely to get placed than those that took Central or Others consistently. However, looking at the number of samples in each of these groupings, we see that (Others, Central) and (Central, Others) have significantly fewer samples than the other two. For our analysis, we will focus on (Central, Central) and (Others, Others) and see if one has any noticeable advantages over the other.

# ### Exploring Board vs Salary 
# 
# From the above graph, it seems that the choice of board doesnt seem to particularly increase one's chances of getting placed but maybe it has a role to play in determining ones salary? Let's check!

# In[ ]:


# Plotting Board vs Salary
salary_by_board = df[both | neither].groupby(["ssc_b", "hsc_b"]).salary.mean().sort_values()
salary_by_board.plot.bar(rot=0).set_xlabel("Board of Education for 10th and 12th");


# These two groups seem to have about the same average salary. If the board did indeed matter, we would expect one group to have significantly higher salary than the other. In case of the number of placed themselves, the (others, others) group does seem to have slightly higher chances of placements but this could be attributed to the difference in sample counts. 
# 
# Now that we have checked potential reasons for parents to choose Central over others, let us now analyse from a student's perspective. The general opinion is that Central board can be much harder than others (i.e state board). Does this mean that students have to work harder to get placed in one group than the other? Let us check.

# In[ ]:


# Create a filter for placed students 
placed = df["status"] == "Placed"

# Plotting distribution of scores for placed students in Central and Others
fig, axes = plt.subplots(nrows = 1, ncols =2, figsize = (15,5))

axes[0].hist(df[both & placed].ssc_p, alpha = 0.5, label = "Central", color = "red");
axes[0].hist(df[neither & placed].ssc_p, alpha = 0.5, label = "Others", color = "green");
axes[0].set_xlabel("10th Grade")
axes[0].set_ylabel("Number of Students")

axes[1].hist(df[both & placed].hsc_p, alpha = 0.5, label = "Central", color="red");
axes[1].hist(df[neither & placed].hsc_p, alpha = 0.5, label = "Others", color="green");
axes[1].set_xlabel("12th Grade")
axes[1].set_ylabel("Number of Students")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.suptitle("Scores Obtained by Placed Students", fontsize=16)
plt.show()


# It seems that among the placed students, those from Central Board scored less in the 10th grade than their `Others` counterparts. This could be an indication of how the examinations in the central board tend to be more difficult resulting in a lower average score. However, the opposite seems to be true in the case of the 12th grade. Maybe the more rigorous process the students faced in their secondary schooling better prepared them for their 12th? 
# 
# You know what they say : "*Some questions are better left ~~unanswered~~ for a different dataset.*"

# ## Question 2
# ### Does it really matter how much you score in your school days? 
# 
# Growing up with Indian parents, I was always told throughout my school days that I should work hard and do well in my exams so that one day I would be able to *reap its benefits*. This led me to always trying to score as highly as I could on my examinations. This dataset makes me want to ask "Does it really matter how much you score?" or to ask a more appropiate question : "Is there evidence to prove that scoring higher in your school days helps you get placed?"

# In[ ]:


# Plots to see 
f, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")
sns.kdeplot(df[placed].hsc_p, df[placed].ssc_p, cmap="Reds",shade_lowest=False)
sns.kdeplot(df[~placed].hsc_p, df[~placed].ssc_p,  cmap ="Blues", shade_lowest=False)
plt.xlabel("12th Percentage")
plt.ylabel("10th Percentage")
plt.show();


# Interesting! There does seem to be a visible trend here. Students who got Placed (orange) seem to have on an average, scored better in both their 10th and 12th than their Non Placed counterparts (blue). It could be the case that the scores were indeed helpful in impressing potential emloyers but on the other hand it is also possible that those students that worked hard in their school days developed the right mindset to keep working and built other skills that increased their employability.

# ## Question 3
# 
# ### Is one stream inherently better than the other? 
# 
# When we complete our 10th grade education, we are met with arguably the most important crossroads our our lives. What next : Science, Commerce or Arts? 
# Unfortunately during my 10th grade, I was faced with a number of health issues and I had to rely on my parents and relatives to choose my path for me and they insisted that I take up science, citing reasons such as "there's a lot of scope for science" and "you'll get job easily". 
# 
# Looking back on my journey so far, I do not regret having taken up Science because eventually Computer Science is where I found my calling. But I'm curious to see if students in one stream are more likely to get a job than some other. Is my relatives' line of reasoning correct?

# In[ ]:


# Plotting placement rate by Stream
df.groupby("hsc_s").status.value_counts(normalize=True).unstack(1).plot.bar( title = "Placement rate by HSC Stream", rot=0, figsize = (8,6)).set_xlabel("Selected Stream");


# Arts does seem to be lagging behind in placements but then again it has significantly fewer samples than commerce and science. For our anaylsis, we will focus on only Science and Commerce, which from the looks of it, have near-identical placment rates. 
# 
# Maybe one stream is more likely to be offered a higher salary?

# In[ ]:


df.groupby("hsc_s").salary.mean().plot.bar(rot=0).set_xlabel("Selected Stream");

Hmm. Even though the mean salary does look quite similar, maybe the difference lies in the distribution of jobs? Maybe the reason my relatives tried to push Science as the "superior stream" was that there are more high paying jobs than Commerce. (i.e, maybe commerce has more  average-pay jobs while science has a mix of high and low pay?)
# In[ ]:


# Create filter for science
science = df["hsc_s"] == "Science"

# Create a filter for commerce
commerce = df["hsc_s"] == "Commerce"

# Plot Salary distribution by Stream
f , ax = plt.subplots(figsize=(8,8))
ax= sns.boxplot(x="salary", y="hsc_s", data =df[science | commerce]);
ax.set_yticklabels(ax.get_yticklabels(),rotation=90)
plt.ylabel("Selected Stream")
plt.xlabel("Salary")
plt.show()


# Looks like both streams have about the same spread of salaries. There are some outliers in both cases and Science does seem to have an ever-so-slightly higher third quartile value. But not enough to make it significant. 
# 
# ***Conclusion*** : *My relatives are lunatics.*

# ## Question 4 
# ### Which degree and MBA specialization has the highest Salary?
# 
# We finally come to what I think might be the most relevant information to an employer. A student's MBA specialization is closely tied to the kind of skills that a company would be looking for and consequently, the salary being offered. The next closest qualification of importance would be the student's undergraduate degree. Let us look at how these two features affect the salaries being offered.

# In[ ]:


# Plotting Salary vs specialization and degree
plt.subplots(figsize=(8,8))
sns.violinplot(x="degree_t", y="salary", hue="specialisation", data=df, scale="count", palette="Set3");
plt.xlabel("Degree Type")
plt.ylabel("Salary")
plt.show()


# It seems as though the highest salaries that were offered were to students who pursued the Marketing & Finance specialisation after obtaining a UG degree in Commerce and Management. However, these are clearly major outliers. It seems than in general, Students with a Science and Technology UG degree pursuring Mkt&Fin specialization were more likely to get higher paying jobs.

# ## Question 5
# ### Does gender bias exist in campus recruitment? 
# 
# The last question that I would like to answer is an important one. Do companies seem to discrimate between men and women when it comes to placements and salaries?

# In[ ]:


# Plotting salaries vs gender by specialization
plt.subplots(figsize=(8,8))
sns.violinplot(y = "salary", x="specialisation", hue = "gender", palette="Set2", data=df, scale="count");
plt.xlabel("Specialization")
plt.ylabel("Salary")
plt.show();


# There does seem to be a general trend that men in this dataset ghave been paid higher than women. But we can also see that the dataset contains fewer samples of females than it does of males. Regardless, I think this is slightly concerning and that the dataset creator might want to take this up with his college!

# # Data Preparation 
# 
# This dataset has very few samples and even fewer if we eliminate the ones without a specified salary. So instead of predicting salary I would like to predict whether a student would be placed or not. I start by creating a feature matrix `X` containing all the features from the dataset except for `Salary` and `Status`. I also drop the serial number as it could potentially lead to overfitting and adds no useful information to the dataset. 
# 
# Then I proceeded to One-hot encode for the categorical variables and dropped the original ones. Since most of the categorical columns contained only 2 or 3 unique values, the final dataset was not bloated too much and had 14 columns for the model to work with. 
# 
# However, when I tried to train the model, I ran into an error that prevented the Linear SVM from converging. [This](https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati) post on stackoverflow explained that one way to solve it was to make use of a Standard Scaler to normalize the data in order to speed up convergence and doing so hepled me correctly fit the model.

# In[ ]:


from sklearn.preprocessing import StandardScaler

# Data Preparation
X = df.drop(["salary", "status", "sl_no"], axis = 1)
X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
cols = X.columns
X = pd.DataFrame(scaler.fit_transform(X))
X.columns = cols
y = df["status"]


# In[ ]:


# Checking the final dataset
X.head(5)


# # Modelling & Evaluation
# Based on the cheat sheet provided [here](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html). I will try to use Linear SVC. 
# 
# Since the number of samples in the dataset is pretty small, using a traditional train-test split means we would be losing out on data, which is already a precious commodity. We would not be able to trust the final score as much, since we would be introducing some bias towards the training set. 
# 
# I decided to make use of KFold cross validation so that the final score would be a better representation of real-world performance.

# In[ ]:


from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold

scores = []
kfold = KFold(n_splits=5, random_state = 42, shuffle=True)
for train_index, test_index in kfold.split(X):
    model = LinearSVC()
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train);
    scores.append(model.score(X_test, y_test))
    
print(f"CV score is: {sum(scores)/len(scores)}")


# Not bad! We were able to achieve a cross-validation score of 0.865 
# 
# Considering that we only had about 200 samples to work with, this score seems pretty decent.This means that although we only had such a small dataset to work with, the Linear SVM model was able to find some patterns within in and is able to predict whether a student will be placed or not, based on the data provided!

# # Conclusion
# 
# ## Quick Recap
# In this notebook, we explored dataset containing campus recruitment data for students from an MBA college. Though our analysis, we found that:
# 1. The board of school education doesnt matter when it comes to placements or even salaries. 
# 2. Placed students seem to have performed better in their 10th and 12th exams than Non Placed Students 
# 3. Among the various specializations, students with a Sci&Tech UG degree and Mkt&Fin MBA specialization appear to have slightly higher salaries.
# 4. The salaries for female students seems to be generally slightly lower than men.
# 
# We then proceeded to fit a linear SVM to predict whether or not a student would be placed and  achieved reasonably good results, with a test accuracy of **86.5**% after some data preprocessing.
# 
# ## What does this mean for you?  
# 
# We must first understand that the results and observations made in this notebook need to be taken with a grain of salt. After all, this dataset is not representative of the full spectrum of students that one comes across during a campus recruitment program. The dataset itself is quite limited in nature and as such, varies in the number of samples between different groups making it diffult to report with confidence the trends that we come across.
# 
# Instead, I urge you to simply follow your heart and make your own choices. Regardless of what a dataset might indicate as being the *best stream* or *best specialization*, ultimately, the *best job* is the one that makes you happy! 
# 
# Good luck!
