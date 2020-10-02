#!/usr/bin/env python
# coding: utf-8

# In this kernel we want to see if there are any correlations between the final grade of students and different characteristics of their live.
# Does their performance in school really just depend on how much they learn or are there maybe some other features that affect their results? Maybe students who's mothers are teachers are just more predestined to get good grades than students with mothers who stay at home? Or the other way around?
# 
# For this purpose we first want to compare the students of the two courses (math and portugese) to see if there are already some characteristic differences that could be lead back to such characteristics. And we will also see how the different features affect their grades.
# 
# After that we want to start our analysis with the DecisionTreeRegressor.

# # Import data:

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold 
from sklearn.model_selection import GridSearchCV 

sns.set()

import os
print(os.listdir("../input"))


# In[ ]:


math_data = pd.read_csv('../input/student-mat.csv')
port_data = pd.read_csv('../input/student-por.csv')


# In[ ]:


math_data.head()


# In[ ]:


port_data.head()


# # What kind of students choose which courses?

# First we want to see what kind of students choose which course. Probably there are already some differences we can detect and use for our grade-analysis later!
# 
# It is possible that some students do attend both courses, but we will neglect than for now because we only have anonymous data.

# ### Number of students in each course:

# We can see that there are almost twice as much Portugese-students as Math-Students!

# In[ ]:


print("Number of Math-Students:", math_data.shape[0])
print("Number of Portugese-Students:", port_data.shape[0])


# ### Age of students:

# In the plot below we can see that the age-distribution of the two courses. The blue bars are the age of Math-Students and the red bars the age of Portugese-Students. As we can see the age-distribution is really similar in both courses, but it seems like the Math-Students are a bit younger than the Portugese-students.

# In[ ]:


sns.distplot(math_data.age, color = "midnightblue", bins = 10)
sns.distplot(port_data.age, color = "red", bins = 10)


# ### Gender of the students:

# Interestingly, in both courses are more girls than boys! But the portugese course still seems to be a bit more popular for girls than math.

# In[ ]:


print("Percentage of females in math:", (round(math_data[math_data.sex == "F"].shape[0] / math_data.shape[0], 2)))
print("Percentage of females in portugese:", (round(port_data[port_data.sex == "F"].shape[0] / port_data.shape[0], 2)))


# ### Family situation

# Now we want to look at all family- and home-related situations the students of both courses live in. Again blue stands for math and red for portugese:

# In[ ]:


def get_percentage_count(feature, data):
    counts = data.groupby(feature).school.count()
    counts = counts / data.shape[0] * 100
    return counts


# In[ ]:


fig, ax = plt.subplots(3,3, figsize = (15,15))
sns.barplot(x = get_percentage_count("address", math_data).index.values, y = get_percentage_count("address", math_data).values, ax = ax[0,0], color = "mediumblue", alpha = 0.5)
sns.barplot(x = get_percentage_count("address", port_data).index.values, y = get_percentage_count("address", port_data).values, ax = ax[0,0], color = "red", alpha = 0.4)
ax[0,0].set_title("Address")
sns.barplot(x = get_percentage_count("famsize", math_data).index.values, y = get_percentage_count("famsize", math_data).values, ax = ax[0,1], color = "mediumblue", alpha = 0.5)
sns.barplot(x = get_percentage_count("famsize", port_data).index.values, y = get_percentage_count("famsize", port_data).values, ax = ax[0,1], color = "red", alpha = 0.4)
ax[0,1].set_title("Family Size (GT3-more than 3, ET3-up to 3)")
sns.barplot(x = get_percentage_count("guardian", math_data).index.values, y = get_percentage_count("guardian", math_data).values, ax = ax[0,2], color = "mediumblue", alpha = 0.5)
sns.barplot(x = get_percentage_count("guardian", port_data).index.values, y = get_percentage_count("guardian", port_data).values, ax = ax[0,2], color = "red", alpha = 0.4)
ax[0,2].set_title("Guardian")
sns.barplot(x = get_percentage_count("Pstatus", math_data).index.values, y = get_percentage_count("Pstatus", math_data).values, ax = ax[1,0], color = "mediumblue", alpha = 0.5)
sns.barplot(x = get_percentage_count("Pstatus", port_data).index.values, y = get_percentage_count("Pstatus", port_data).values, ax = ax[1,0], color = "red", alpha = 0.4)
ax[1,0].set_title("Are parents living together? (A- Apart, T-Together)")
sns.barplot(x = get_percentage_count("Medu", math_data).index.values, y = get_percentage_count("Medu", math_data).values, ax = ax[1,1], color = "mediumblue", alpha = 0.5)
sns.barplot(x = get_percentage_count("Medu", port_data).index.values, y = get_percentage_count("Medu", port_data).values, ax = ax[1,1], color = "red", alpha = 0.4)
ax[1,1].set_title("Mothers Education (0-none, 4-high)")
sns.barplot(x = get_percentage_count("Fedu", math_data).index.values, y = get_percentage_count("Fedu", math_data).values, ax = ax[1,2], color = "mediumblue", alpha = 0.5)
sns.barplot(x = get_percentage_count("Fedu", port_data).index.values, y = get_percentage_count("Fedu", port_data).values, ax = ax[1,2], color = "red", alpha = 0.4)
ax[1,2].set_title("Fathers Education (0-none, 4-high)")
sns.barplot(x = get_percentage_count("famrel", math_data).index.values, y = get_percentage_count("famrel", math_data).values, ax = ax[2,0], color = "mediumblue", alpha = 0.5)
sns.barplot(x = get_percentage_count("famrel", port_data).index.values, y = get_percentage_count("famrel", port_data).values, ax = ax[2,0], color = "red", alpha = 0.4)
ax[2,0].set_title("Quality of family relations")
sns.barplot(x = get_percentage_count("internet", math_data).index.values, y = get_percentage_count("internet", math_data).values, ax = ax[2,1], color = "mediumblue", alpha = 0.5)
sns.barplot(x = get_percentage_count("internet", port_data).index.values, y = get_percentage_count("internet", port_data).values, ax = ax[2,1], color = "red", alpha = 0.4)
ax[2,1].set_title("Internet access")
sns.barplot(x = get_percentage_count("famsup", math_data).index.values, y = get_percentage_count("famsup", math_data).values, ax = ax[2,2], color = "mediumblue", alpha = 0.5)
sns.barplot(x = get_percentage_count("famsup", port_data).index.values, y = get_percentage_count("famsup", port_data).values, ax = ax[2,2], color = "red", alpha = 0.4)
ax[2,2].set_title("Educational support in family")


# None of the features differ a lot, but there are still some with differences that could tell us what courses they would rather take:
# The education of their parents seem to make the biggest impact out of the features above: Students with at least one parent who got at least secondary education are more often taking math than students with parents who only visited school to maximum 9th grade. These students on the other hand are more often taking portugese than the students with higher parental education.
# That's really interesting! Maybe higher educated parents can encourage their kids more to take math because they are able to help them better than lower educated parents? Or maybe do they tell their kids how important math is and maybe even force them to take math-courses?
# 
# Moreover the address of the students and their internet-access seem to influence the students. More students with internet access are taking math than portugese, while students with no internet access are more often taking portugese instead of math. Maybe because the students have to research more for math than for portugese? Also more student from the urban-adresses take math and more students from rural-addresses take portugese! But maybe this also linked to the education of their parents and so their income? We will look at this later!
# 
# Many features don't seem to differ that much: so the size of their family, their gurdian, the living-situation of their parents, the quality of their family relationships and especially their educational support in the family don't seem to have any influence on the courses they pick.

# ### Freetime-activites

# After we looked at the situation of the students at home we now want to see what they do in their freetime. Do they take extra classes or are they more into drinking and meeting with friends? Let's see!:
# 
# (cyan - math, pink - portugese)

# In[ ]:


fig, ax = plt.subplots(3,3, figsize = (15,15))
sns.barplot(x = get_percentage_count("studytime", math_data).index.values, y = get_percentage_count("studytime", math_data).values, ax = ax[0,0], color = "aquamarine", alpha = 0.5)
sns.barplot(x = get_percentage_count("studytime", port_data).index.values, y = get_percentage_count("studytime", port_data).values, ax = ax[0,0], color = "deeppink", alpha = 0.4)
ax[0,0].set_title("Weekly study-time (1: <2, 2: 2-5, 3: 5-10, 4: >10)")
sns.barplot(x = get_percentage_count("paid", math_data).index.values, y = get_percentage_count("paid", math_data).values, ax = ax[0,1], color = "aquamarine", alpha = 0.5)
sns.barplot(x = get_percentage_count("paid", port_data).index.values, y = get_percentage_count("paid", port_data).values, ax = ax[0,1], color = "deeppink", alpha = 0.4)
ax[0,1].set_title("Do they take extra classes in their course?")
sns.barplot(x = get_percentage_count("activities", math_data).index.values, y = get_percentage_count("activities", math_data).values, ax = ax[0,2], color = "aquamarine", alpha = 0.5)
sns.barplot(x = get_percentage_count("activities", port_data).index.values, y = get_percentage_count("activities", port_data).values, ax = ax[0,2], color = "deeppink", alpha = 0.4)
ax[0,2].set_title("Do they join extracurricular activites?")
sns.barplot(x = get_percentage_count("romantic", math_data).index.values, y = get_percentage_count("romantic", math_data).values, ax = ax[1,0], color = "aquamarine", alpha = 0.5)
sns.barplot(x = get_percentage_count("romantic", port_data).index.values, y = get_percentage_count("romantic", port_data).values, ax = ax[1,0], color = "deeppink", alpha = 0.4)
ax[1,0].set_title("Are they in a romantic relationship?")
sns.barplot(x = get_percentage_count("freetime", math_data).index.values, y = get_percentage_count("freetime", math_data).values, ax = ax[1,1], color = "aquamarine", alpha = 0.5)
sns.barplot(x = get_percentage_count("freetime", port_data).index.values, y = get_percentage_count("freetime", port_data).values, ax = ax[1,1], color = "deeppink", alpha = 0.4)
ax[1,1].set_title("Freetime (1- low, 5- high)")
sns.barplot(x = get_percentage_count("goout", math_data).index.values, y = get_percentage_count("goout", math_data).values, ax = ax[1,2], color = "aquamarine", alpha = 0.5)
sns.barplot(x = get_percentage_count("goout", port_data).index.values, y = get_percentage_count("goout", port_data).values, ax = ax[1,2], color = "deeppink", alpha = 0.4)
ax[1,2].set_title("Going out with friends (1-low, 5-high)")
sns.barplot(x = get_percentage_count("Dalc", math_data).index.values, y = get_percentage_count("Dalc", math_data).values, ax = ax[2,0], color = "aquamarine", alpha = 0.5)
sns.barplot(x = get_percentage_count("Dalc", port_data).index.values, y = get_percentage_count("Dalc", port_data).values, ax = ax[2,0], color = "deeppink", alpha = 0.4)
ax[2,0].set_title("Alcohol consumption on weekdays")
sns.barplot(x = get_percentage_count("Walc", math_data).index.values, y = get_percentage_count("Walc", math_data).values, ax = ax[2,1], color = "aquamarine", alpha = 0.5)
sns.barplot(x = get_percentage_count("Walc", port_data).index.values, y = get_percentage_count("Walc", port_data).values, ax = ax[2,1], color = "deeppink", alpha = 0.4)
ax[2,1].set_title("Alcohol consumption on weekdends")


# Okay, we can see some stronger differences here! Students who took math are more tend to take extra paid courses than portugese students - almost 50% of Math-Students take such classes - while almost no Portugese-Students do. This probably comes from the fact that more students have problems with math than with portugese. That also relates to the fact that mainly Portugese-Students tend to study less than 2h and Math-Students tend to study a bit more than portugese students (but the distribution is still similar - so most students tend to study 2-5h).
# 
# All other features don't differ that much - like you would expect from the steretypes of Math-Students they don't have as much relationships as Portugese-students, but all other features don't differ really much. Both courses are even drinking the same amount of alcohol on weekdays!

# ### Other influences

# Now we already looked at the basic characteristics of the students, their family situation and how they spend their freetime. But there are still some interesting features left we didn't looked at. We want to do that now:
# 
# (orange - portugese, green - math)

# In[ ]:


fig, ax = plt.subplots(2,3, figsize = (15,10))
sns.barplot(x = get_percentage_count("school", math_data).index.values, y = get_percentage_count("school", math_data).values, ax = ax[0,0], color = "lightseagreen", alpha = 0.5)
sns.barplot(x = get_percentage_count("school", port_data).index.values, y = get_percentage_count("school", port_data).values, ax = ax[0,0], color = "orangered", alpha = 0.4)
ax[0,0].set_title("Which school do they go to?")
sns.barplot(x = get_percentage_count("reason", math_data).index.values, y = get_percentage_count("reason", math_data).values, ax = ax[0,1], color = "lightseagreen", alpha = 0.5)
sns.barplot(x = get_percentage_count("reason", port_data).index.values, y = get_percentage_count("reason", port_data).values, ax = ax[0,1], color = "orangered", alpha = 0.4)
ax[0,1].set_title("Why did they choose their school?")
sns.barplot(x = get_percentage_count("nursery", math_data).index.values, y = get_percentage_count("nursery", math_data).values, ax = ax[0,2], color = "lightseagreen", alpha = 0.5)
sns.barplot(x = get_percentage_count("nursery", port_data).index.values, y = get_percentage_count("nursery", port_data).values, ax = ax[0,2], color = "orangered", alpha = 0.4)
ax[0,2].set_title("Did they went to Kindergarden?")
sns.barplot(x = get_percentage_count("higher", math_data).index.values, y = get_percentage_count("higher", math_data).values, ax = ax[1,0], color = "lightseagreen", alpha = 0.5)
sns.barplot(x = get_percentage_count("higher", port_data).index.values, y = get_percentage_count("higher", port_data).values, ax = ax[1,0], color = "orangered", alpha = 0.4)
ax[1,0].set_title("Do they want to take higher education?")
sns.barplot(x = get_percentage_count("health", math_data).index.values, y = get_percentage_count("health", math_data).values, ax = ax[1,1], color = "lightseagreen", alpha = 0.5)
sns.barplot(x = get_percentage_count("health", port_data).index.values, y = get_percentage_count("health", port_data).values, ax = ax[1,1], color = "orangered", alpha = 0.4)
ax[1,1].set_title("Health (1- bad, 5- good)")
sns.distplot(math_data.absences, color = "lightseagreen", ax = ax[1,2])
sns.distplot(port_data.absences, color = "orangered", ax = ax[1,2])
ax[1,2].set_xlim(0,40)


# In[ ]:


print("Max. number of absent days of Math-students:", max(math_data.absences))
print("Max. number of absent days of Portugese-students:", max(port_data.absences))


# As we can see the Gabriel Pereira - school seems to offer more math-courses than portugese courses, while the Mousinho da Silveira - school consists of about 2/3 more Portugese-students compared to the Math-students.
# While more Portugese-students chose their school because of the course itself, the Math-students are more often choosing their school because of reputation or because its near their home.
# Also a bit more Math-students want to take higher education than the Portugese-students, but there doesn't seem to be such a bit difference at all, just like the other features don't differ than much.
# 
# Only the absence is still a bit different: It seems like the Portugese-students are missing school more often to about 10 days, while the Math students are more often missing at 10 days and up. We can also see that the maximum number of absent days is 75 for Math-students and only 32 for Portugese-students.
# So we can say: Portugese-students are more often missing for short periods of time, while the Math-students tend to miss school a lot more. (We can also see that in the much smaller bar of the Math-students at 0 absences).

# # What are the grades in both courses?

# Now that we know a bit more about the students in both courses we want to look at the important thing (at least for the life as a student): their grades!
# 
# Let's first have a look at how the grades of our students are in general.
# In the plot below you can see the distribution of the First Period Grades, Second Period Grades and Final Grades. The blue distribution shows the Math-Students, the red distribution the Portugese-Students:

# In[ ]:


fig, ax = plt.subplots(3,1, figsize = (20,15))
sns.distplot(math_data.G1, kde = True, ax = ax[0], color = "Blue", bins = 20)
sns.distplot(port_data.G1, kde = True, ax = ax[0], color = "r", bins = 20)
sns.distplot(math_data.G2, kde = True, ax = ax[1], color = "Blue", bins = 20)
sns.distplot(port_data.G2, kde = True, ax = ax[1], color = "r", bins = 20)
sns.distplot(math_data.G3, kde = True, ax = ax[2], color = "Blue", bins = 20)
sns.distplot(port_data.G3, kde = True, ax = ax[2], color = "r", bins = 20)
ax[0].set_xlim(0,20)
ax[1].set_xlim(0,20)
ax[2].set_xlim(0,20)
ax[0].set_ylim(0,0.22)
ax[1].set_ylim(0,0.22)
ax[2].set_ylim(0,0.22)
ax[0].set_title("First Period Grades")
ax[1].set_title("Second Period Grades")
ax[2].set_title("Final Grades")
fig.suptitle('Blue: Math, Red: Portugese', fontsize = 20)


# We can see that the Portugese-Students seem to have quite constant grades in first and second period, while the Math-Students seem to differ way more: While they had a quite different grade-distribution first, they got a way more alike one in the second period. Alltogether the grades of the Portugese-Students resemble a normal-distribution around 10 points (what should be the goal by creating exams) way better than the Math-Students do.
# It's also really showy that in all three plots our Math-Students have the (most) highest achieved grade(s), but also more students with 0 points tha the Portugese-Students. So in the differences of the students math-skills are way higher than in Portugese.

# # What does really affect the students grades?
# ### Math-students

# In[ ]:


fig, ax = plt.subplots(5,6, figsize = (20,20))
sns.barplot(x = math_data.school, y = math_data.G3, ax = ax[0,0])
sns.barplot(x = math_data.sex, y = math_data.G3, ax = ax[0,1])
sns.barplot(x = math_data.age, y = math_data.G3, ax = ax[0,2])
sns.barplot(x = math_data.address, y = math_data.G3, ax = ax[0,3])
sns.barplot(x = math_data.famsize, y = math_data.G3, ax = ax[0,4])
sns.barplot(x = math_data.Pstatus, y = math_data.G3, ax = ax[0,5])
sns.barplot(x = math_data.Medu, y = math_data.G3, ax = ax[1,0])
sns.barplot(x = math_data.Fedu, y = math_data.G3, ax = ax[1,1])
sns.barplot(x = math_data.Mjob, y = math_data.G3, ax = ax[1,2])
sns.barplot(x = math_data.Fjob, y = math_data.G3, ax = ax[1,3])
sns.barplot(x = math_data.reason, y = math_data.G3, ax = ax[1,4])
sns.barplot(x = math_data.guardian, y = math_data.G3, ax = ax[1,5])
sns.barplot(x = math_data.traveltime, y = math_data.G3, ax = ax[2,0])
sns.barplot(x = math_data.studytime, y = math_data.G3, ax = ax[2,1])
sns.barplot(x = math_data.failures, y = math_data.G3, ax = ax[2,2])
sns.barplot(x = math_data.schoolsup, y = math_data.G3, ax = ax[2,3])
sns.barplot(x = math_data.famsup, y = math_data.G3, ax = ax[2,4])
sns.barplot(x = math_data.paid, y = math_data.G3, ax = ax[2,5])
sns.barplot(x = math_data.activities, y = math_data.G3, ax = ax[3,0])
sns.barplot(x = math_data.nursery, y = math_data.G3, ax = ax[3,1])
sns.barplot(x = math_data.higher, y = math_data.G3, ax = ax[3,2])
sns.barplot(x = math_data.internet, y = math_data.G3, ax = ax[3,3])
sns.barplot(x = math_data.romantic, y = math_data.G3, ax = ax[3,4])
sns.barplot(x = math_data.famrel, y = math_data.G3, ax = ax[3,5])
sns.barplot(x = math_data.freetime, y = math_data.G3, ax = ax[4,0])
sns.barplot(x = math_data.goout, y = math_data.G3, ax = ax[4,1])
sns.barplot(x = math_data.Dalc, y = math_data.G3, ax = ax[4,2])
sns.barplot(x = math_data.Walc, y = math_data.G3, ax = ax[4,3])
sns.barplot(x = math_data.health, y = math_data.G3, ax = ax[4,4])
sns.barplot(x = math_data.absences, y = math_data.G3, ax = ax[4,5])


# In[ ]:


math_data[math_data.Medu == 0].count()


# # Missing values?
# 
# Before we can start our analysis we have to check if there are any missing values:

# In[ ]:


math_data.isnull().sum().sum()


# In[ ]:


port_data.isnull().sum().sum()


# As we can see there are no missing vales. That's great! We don't have to fill up or delete any columns or rows and can directly start with our analysis.

# # Feature Transformation
# 
# Now we still have to transform our features so our model can work with it - we only want to have binary features after this transformation.

# In[ ]:


math_data["Dalc"] = math_data.Dalc.astype("object")
math_data["age"] = math_data.age.astype("object") 
math_data["Medu"] = math_data.Medu.astype("object") 
math_data["Fedu"] = math_data.Fedu.astype("object") 
math_data["traveltime"] = math_data.traveltime.astype("object")
math_data["studytime"] = math_data.studytime.astype("object") 
math_data["failures"] = math_data.failures.astype("object") 
math_data["famrel"] = math_data.famrel.astype("object") 
math_data["freetime"] = math_data.freetime.astype("object") 
math_data["goout"] = math_data.goout.astype("object") 
math_data["Walc"] = math_data.Walc.astype("object") 
math_data["health"] = math_data.health.astype("object")


# In[ ]:


pd.get_dummies(math_data) 

math_data = pd.get_dummies(math_data) 
math_data.head()


# In[ ]:


pd.get_dummies(port_data) 

port_data = pd.get_dummies(port_data) 
port_data.head()


# # How accurate are our predictions with a decision tree? 

# To do our analysis we first want to use the DesicionTreeClassifier and to get a meaningful result we are going to do a cross-validation (that means we are seperating our data in n parts and are training and testing them in different combinations).
# 
# Then we want to see how good our model can predict the final grades of the students! *(Not good right now because our transformation is not done yet!)*

# In[ ]:


def get_cvscores(data, model):
    y = data["G3"].values
    X = data.drop(["G1", "G2", "G3"], axis=1).values
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    scores = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    return scores, model


# In[ ]:


model = DecisionTreeClassifier(random_state = 0) 
get_cvscores(math_data, model) 


# In[ ]:


model = DecisionTreeClassifier(random_state = 0) 
get_cvscores(port_data, model)


# # GridSearchCV

# In[ ]:


parameters = {'max_depth':[1, 20], 'min_samples_leaf':[1, 20], "min_weight_fraction_leaf": [0, 0.5], "min_samples_split": [2, 20], "max_features": [1, 20]} 

model = DecisionTreeClassifier()
clf = GridSearchCV(model, parameters, cv=5)

scores, model = get_cvscores(math_data, model) 
print(scores) 

y = math_data["G3"].values
X = math_data.drop(["G1", "G2", "G3"], axis=1).values 

clf.fit(X,y)
best_params = clf.best_params_
print(best_params)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score'] 


# In[ ]:


model = DecisionTreeClassifier(max_depth = 20, max_features = 20, min_samples_leaf= 20, min_samples_split = 20, min_weight_fraction_leaf = 0, random_state = 0) 
scores, model = get_cvscores(math_data, model)


# In[ ]:


from sklearn import tree 
import graphviz
dot_data = tree.export_graphviz(model, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("students") 

