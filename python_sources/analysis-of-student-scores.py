#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# This notebook will attempt to analyze high school student's performance in math, reading, and writing. The dataset comes from http://roycekimmons.com/tools/generated_data/exams, and contains the performance data of 1000 students from a public school. In addition, it also contains information regarding their background such as gender, race, lunch plan, whether or not they completed a test preparation course, and their parent's level of education.
# 
# The notebook will take a look at each factor to see if they play a significant role in a student's academic success in those subjects. In addition, please note that the data used is fictional, so the findings have no significant meaning.
# 
# Advice and suggestions for improvement would also be greatly appreciated.

# ## Factors to look at
# 
# Factors and questions that the notebook will be looking at are:
# 
# 1. Do students on average score significantly better at a certain subject?
# 2. Do students with a certain gender typically score better on certain subjects than their counterpart?
# 3. Do students of a certain race score better than others?
# 4. On average, does completing a test preparation course increase a student's score?
# 5. If test prep helps, is the group that has the best score also the group that has the highest test preparation course?
# 6. What about whether or not the parents have a degree?
# 7. Do low income students (defined by whether or not they are on a lunch plan) score worse?
# 8. How many students on a lunch plan complete a test preparation course (assuming that the test preparation course costs money)?
# 9. Will it be possible to predict the range of a student's score using Machine Learning?

# In[ ]:


# Import the libraries that will be used
import numpy as np
import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[ ]:


# Read the file and put it into a dataframe
data = pd.read_csv("../input/StudentsPerformance.csv")


# In[ ]:


# Get a general overview of the data
data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


# Quick box plot of the data
boxplot = data.boxplot(patch_artist=True)
plt.title("Box plot of scores")
plt.ylabel("Points")


# In[ ]:


# Histogram of the scores for the three subjects
plt.hist("math score", data=data, alpha=0.5, bins=25)
plt.hist("reading score", data=data, alpha=0.5, bins=25)
plt.hist("writing score", data=data, alpha=0.5, bins=25)
plt.xlabel("Score")
plt.ylabel("Count")
plt.title("Score Distributions Among the Three Subjects")
plt.legend(["Math", "Reading", "Writing"])


# In[ ]:


# Add a column for the cumulative score
data["Cumulative Score"] = data["math score"] + data["reading score"] + data["writing score"]
data.head()


# In[ ]:


# Boxplot of the cumulative score
data.boxplot("Cumulative Score", patch_artist=True)
plt.title("Box plot of Cumulative Score")
plt.ylabel("Points")


# In[ ]:


# Find the mean and standard deviation of the cumulative score
print("Mean: %.2f \nStd: %.2f" %(data["Cumulative Score"].mean(), data["Cumulative Score"].std()))


# ## Quick Glance
# 
# From a quick glance at the data, there are no null values that will need to be accounted for, and students generally score the highest in the reading section with the math section being the lowest. However, all of the subjects score roughly the same at around the 60-80 range, and the score distribution forms a bell curve like shape. When looking at cumulative scores, the standard deviation is much higher and a student's score can range from around 150-250 on average. The mean for the cumulative scores are around 203.31 with a standard deviation of around 42.77.
# 
# Some points that would be interesting to see are if this holds true for all genders, races, etc.

# In[ ]:


# Create a function to quickly group the data by a certain field and returns the mean scores
# In addition, it plots a histogram of the mean math, reading, writing, and cumulative scores
# Inputs are the dataframe, the groupby field, and a colorset
def group_scores(df, groupby_field, **colors):
    df = df[[groupby_field, "math score", "reading score", "writing score", "Cumulative Score"]].groupby(groupby_field)
    df = DataFrame(np.transpose(df.mean()))
    df.index = ["Math", "Reading", "Writing", "Cumulative"]

    if colors.get("color"):
        df.plot(kind="bar", rot=0, color=colors.get("color"))
    else:
        df.plot(kind="bar", rot=0, cmap=colors.get("color"))

    plt.title("Mean scores based on " + groupby_field)
    plt.ylabel("Mean Scores")
    plt.legend(loc="upper center")
    return df


# ## Analysis Based on Gender

# In[ ]:


# Making sure that there is enough data for both genders
data["gender"].value_counts()


# In[ ]:


# Create the plot using the function
gender_scores = group_scores(data, "gender", color=["magenta", "blue"])


# In[ ]:


# Print the mean scores for each section
gender_scores


# From the results, males scored better on average than their female counterparts in the math section. However, in the reading and writing section, females scored better than males. Males on average being stronger in math may be one of the reasons why there are typically males in STEM fields.
# 
# Overall, females score higher than males by around ~11 points.

# ## Analysis Based on Race

# In[ ]:


# Check the counts for the race ot make sure there is a good distribution in the data
data["race/ethnicity"].value_counts()


# In[ ]:


# Plot
race_scores = group_scores(data, "race/ethnicity", cmap="set1")


# In[ ]:


# Print the mean scores in table format
race_scores


# On average, it seems that group E scores the highest on all three subjects and group A scores the lowest. Perhaps this is because group E has a larger percentage of people that took a test perparation course. 
# 
# It will be interesting to see later on if the students that completed the test preparation course have higher scores, and whether or not group E has the highest percentage of students that completed the course.

# ## Analysis Based on Completion of Test Preparation Course

# In[ ]:


# Check the counts
data["test preparation course"].value_counts()


# In[ ]:


# Plot
test_prep_scores = group_scores(data, "test preparation course", cmap="set1")


# In[ ]:


# Print
test_prep_scores


# It seems that students that completed the test preparation course score higher than those that did not. This makes sense since those students are expected to learn more and have more experience in solving related problems. It may also be the case that students that completed the course studied more than those that did not. The data also shows that the test preparation course is working as intended, and may not necessarily be a waste of money.

# ## Percentage of Students per Group That Completed a Test Preparation Course

# In[ ]:


# Group by race and test prep course
group_race_test_counts = data[["race/ethnicity", "test preparation course"]].groupby(["race/ethnicity", "test preparation course"])


# In[ ]:


# Get the counts using size() and make it into a dataframe
group_race_test_counts = DataFrame(group_race_test_counts.size().reset_index(name="Counts")).set_index("race/ethnicity")


# In[ ]:


# Get total number of students in each race that completed test prep
group_race_test_sums = group_race_test_counts.groupby(group_race_test_counts.index).sum()


# In[ ]:


# Rename column
group_race_test_sums.columns = ["Total Num"]


# In[ ]:


# Join the data
group_race_test = pd.concat([group_race_test_counts, group_race_test_sums], axis=1, join="inner")


# In[ ]:


# Create a new column for the percentage of students
group_race_test["Percentage"] = np.round(group_race_test["Counts"]/group_race_test["Total Num"], 2)


# In[ ]:


# Print out the percentages for each group that completed the test prep score
group_race_test[group_race_test["test preparation course"] == "completed"]


# When analyzing the races, group E scored the highest in all subjects. In addition, when comparing the students that completed a test preparation course with those that did not, it was shown that test preparation courses helped improve scores on average.
# 
# Looking at the percentage of students from each group that completed the test preparation score, group E had the highest percentage of students that completed the course. This may be the one reason why group E scored the highest out of all the groups.
# 
# Surprisingly, group D had the lowest percentage of completion yet they scored the second highest out of all the groups in all subjects. This may be due to group D studying a significant amount more than group A, B, and C despite having a lower percentage of students that completed the test preparation course.

# ## Analysis on Parent's Educational Background and Student Scores

# In[ ]:


# Counts
data["parental level of education"].value_counts()


# In[ ]:


# Plot
parent_ed_scores = group_scores(data, "parental level of education", cmap="set1")


# In[ ]:


# Print
parent_ed_scores


# It looks like if the parents have some sort of degree, the student will score higher on average than those whose parents do not have a degree. Surprisingly, children of parents that did not finish high school scored better than the children whose parents actually completed high school. Perhaps this is due to those parents realizing the drawbacks of not having a high school diploma and emphasizing the importance of an education to their children.
# 
# To make the divide between scores from children whose parents have a degree and not, a new yes/no column will be added to show whether or not the parent has a degree.

# In[ ]:


# If the parental level of education column contains "degree", set the value to yes
data["Degree"] = data["parental level of education"].apply(
    lambda x: "yes" if x.find("degree") >= 0 else "no")


# In[ ]:


# View result
data.head()


# In[ ]:


# Plot again
parent_degree_scores = group_scores(data, "Degree", color=["red", "green"])


# In[ ]:


# View the numbers
parent_degree_scores


# In[ ]:


# A quick plot to see if children whose parents have a degree complete test preparation courses 
sns.countplot("Degree", data=data, hue="test preparation course")


# From the data, children whose parents have a degree score better across all three subjects resulting in a mean cumulative difference of around 15 points. This may be due to the parents with degrees valuing education higher than those without and are more willing to help their children out. On the other hand, parents without a degree may have some sort of work that do not require an advanced education. From the data, children whose parents have a degree score better across all three subjects resulting in a mean cumulative difference of around 15 points.
# 
# The countplot of children whose parents have a degree attempts to see if children have a higher percentage to complete the test preparation course if their parents have a degree. This information can potentially show that parents with a degree have a higher likelihood to make their children complete the test preparation course if the discrepancy is large enough. The following section will be taking a look at the exact percentages and seeing if this is the case.

# ## Percentage of Students with Parents Having a Degree That Completed a Test Preparation Course
# 

# In[ ]:


# Group relevant data
group_degree_test_counts = data[["Degree", "test preparation course"]].groupby(["Degree", "test preparation course"])


# In[ ]:


# Get the counts using size() and make it into a dataframe
group_degree_test_counts = DataFrame(group_degree_test_counts.size().reset_index(name="Counts")).set_index("Degree")


# In[ ]:


# Get total number of students in each degree group
group_degree_test_sums = group_degree_test_counts.groupby(group_degree_test_counts.index).sum()


# In[ ]:


# Rename column
group_degree_test_sums.columns = ["Total Num"]


# In[ ]:


# Join the data
group_degree_test = pd.concat([group_degree_test_counts, group_degree_test_sums], axis=1, join="inner")


# In[ ]:


# View the data
group_degree_test


# In[ ]:


# Create a new column for the percentage of students
group_degree_test["Percentage"] = np.round(group_degree_test["Counts"]/group_degree_test["Total Num"], 2)


# In[ ]:


# Print out the percentages
group_degree_test[group_degree_test["test preparation course"] == "completed"]


# A more granular look at the percentages of students that completed the test preparation course shows that roughly the same percentage of students complete the test preparation course regardless of whether or not the parents have a degree or not.
# 
# Therefore, it is not the case that those students are scoring better because the parents are making them complete the test preparation course. However, it can still be a case that those parents are helping the students more outside of the classroom and/or trying to teach their children about the benefits of education.

# ## Analysis on Lunch Plan (Income) and Student Scores

# In[ ]:


# Checking to see students with free/reduced lunch complete the test preparation course
sns.countplot("test preparation course", data=data, hue="lunch")


# In[ ]:


# Plot
lunch_scores = group_scores(data, "lunch", cmap="set1")


# In[ ]:


lunch_scores


# It seems that students on a standard lunch plan score higher than students on a reduced lunch plan on average. Surprisingly, there are still students on a reduced lunch that complete the test preparation course, so it may be the case that the test preparation course does not cost money.
# 
# As to why the students on a standard lunch plan score higher, it may be due to those students having more resources to exceed. For example, they may not have worries such as making sure they have enough to eat, taking tests while hungry, etc. In addition, some may even have private tutors or some sort of afterschool class to help them improve their scores. In contrast, students on a free/reduced lunch plan may not have the funds to access those resources, and may even be in a worse physical condition than those on a standard lunch plan when taking the tests.

# ## Machine Learning to Predict Scores

# In[ ]:


# Function to convert the binary categorical fields to 1s and 0s
def convert_binary_categorical(df, key_val_dict):
    df = df.copy()
    for key in key_val_dict:
        df[key] = df[key].apply(lambda x: 1 if x == key_val_dict[key] else 0)
    return df


# In[ ]:


# Mapping the gender, test prep, race, degree as 1
mapping_dict = {"gender": "male", "Degree": "yes", "test preparation course": "completed"}


# In[ ]:


# Create the dataframe for the binary categorical fields and convert
data_binary_x = data[["gender", "Degree", "test preparation course"]]
data_binary_x = convert_binary_categorical(data_binary_x, mapping_dict)


# In[ ]:


# Check results
data_binary_x.head()


# In[ ]:


# Create dummies for the non binary categorical field
data_non_binary_x = pd.get_dummies(data["race/ethnicity"])


# In[ ]:


# Drop a field to prevent over fitting and view
data_non_binary_x.drop("group E", axis=1, inplace=True)
data_non_binary_x.head()


# In[ ]:


# Combine the binary and nonbinary dataframes to get the X data
data_x = pd.concat([data_binary_x, data_non_binary_x], axis=1)


# In[ ]:


# View
data_x.head()


# In[ ]:


# Generate the Y target
data_y_math = data["math score"]
data_y_reading = data["reading score"]
data_y_writing = data["writing score"]


# In[ ]:


# Split the x and y into training and testing datasets
x_train_math, x_test_math, y_train_math, y_test_math = train_test_split(data_x, data_y_math)
x_train_reading, x_test_reading, y_train_reading, y_test_reading = train_test_split(data_x, data_y_reading)
x_train_writing, x_test_writing, y_train_writing, y_test_writing = train_test_split(data_x, data_y_writing)


# In[ ]:


# Predict the probability of each score occuring
clf_math = LogisticRegression()
clf_math.fit(x_train_math, y_train_math)
y_pred_math = clf_math.predict_proba(x_test_math)


# In[ ]:


# Probability of each score occuring for each row
y_pred_math


# In[ ]:


# Group the math, reading, and writing score into bins to view easier
# 0, 1, 2, 3, 4, 5 corresponds to [0-20), [20-40), [40-60)...
data_y_math = np.floor(data_y_math/20)
data_y_reading = np.floor(data_y_reading/20)
data_y_writing = np.floor(data_y_writing/20)


# In[ ]:


# Checking
[data_y_math.unique(), data_y_reading.unique(), data_y_writing.unique()]


# In[ ]:


# Split into training and testing datasets using the new y data
x_train_math, x_test_math, y_train_math, y_test_math = train_test_split(data_x, data_y_math, random_state=42)
x_train_reading, x_test_reading, y_train_reading, y_test_reading = train_test_split(data_x, data_y_reading)
x_train_writing, x_test_writing, y_train_writing, y_test_writing = train_test_split(data_x, data_y_writing)


# In[ ]:


# Predict the bin given the gender, degree, test prep, and group
clf_math = LogisticRegression()
clf_math.fit(x_train_math, y_train_math)
y_pred_math = clf_math.predict(x_test_math)

clf_reading = LogisticRegression()
clf_reading.fit(x_train_reading, y_train_reading)
y_pred_reading = clf_reading.predict(x_test_reading)

clf_writing = LogisticRegression()
clf_writing.fit(x_train_writing, y_train_writing)
y_pred_writing = clf_writing.predict(x_test_writing)


# In[ ]:


# Check the accuracy
print("Math: %.5f \nReading: %.5f \nWriting: %.5f " 
      %(accuracy_score(y_test_math, y_pred_math), accuracy_score(y_test_reading, y_pred_reading), accuracy_score(y_test_writing, y_pred_writing)))


# If you just randomly guess a group, you have a 1/5 chance of getting it correct, so using Logistic Regression may be beneficial. However, since the majority of students fall in bin 3 (60-80), it is important to take into account what the accuracy of only guessing bin 3 is for each subject.

# ## Probabily of Guessing Correct Bin if Only Choosing Bin 3

# In[ ]:


# Function to calculate probability when only choosing 3
def calc_percentage(data_array, bin_num):
    # Count of bin num in the array/count of all the bins in the array
    return (data_array == bin_num).sum()/len(data_array) # data_array == bin_num returns True/False and sum will sum the trues


# In[ ]:


# Print accuracy of only guessing 3
print("Math: %.5f \nReading: %.5f \nWriting: %.5f " 
      %(calc_percentage(y_test_math, 3), calc_percentage(y_test_reading, 3), calc_percentage(y_test_writing, 3)))


# Using a logistic regression will perform slightly better than if you were to just guess bin 3 for each student. However, it is not too noticeable a difference, and it is likely that regardless of a student's background, they will score roughly within the same 60-80 range.

# ## Summary
# There are many factors that seem to affect a students performance. For example, on average, a student is expected to do better if they take a test preparation course, if they are on a standard lunch plan, and if their parents have a degree. In addition, the average male performed better than the average female in the math section while the females outperformed the males in the reading and writing sections.
# 
# As for using machine learning to predict scores, the Logistic Regression model was used, and it performed slightly better than just picking the most frequent bin (bin 3). However, bin 3 encompasses a signficant range of scores from D- to almost B- (60-80). Therefore, since the accuracy score only slightly increased, and the score range that the model predicts is significantly wide, the results of the logistic regression model used in this study may not be too useful.
