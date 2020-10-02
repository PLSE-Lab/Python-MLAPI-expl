#!/usr/bin/env python
# coding: utf-8

# # Student Alcohol Consumption
# 
# ### Exploratory Data Analysis

# In this notebook we are going to see how to perform an exploratory data analysis, a step that should be performed over every new dataset before to proceed with the development of predictive models. In this case, our goal is to understand what are the factors that affect the performance of students at high school. In particular, we are interested to know the impact of the consumption of alcohol.

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


math = pd.read_csv("../input/student-mat.csv")


# Let's see what the dataset looks like.

# In[ ]:


math.head()


# In[ ]:


math.columns


# In[ ]:


math.describe()


# We have some variables that could be interesting. Some of the variables are categorical, and other are continuous. They require a different type of analysis.

# It is highly convenient to add the average grade to the dataframe.

# In[ ]:


math["Average"] = (math['G1'] + math['G2'] + math['G3']) / 3


# Also, it would be very nice to have a variable showing if the student has or has not passed the course, but we are not sure if that it is computed as the average of the three grades. So, let's call it if the student has been approved (1 - approved, 0 - not approved).

# In[ ]:


math["Approved"] = [ (1 if x > 10 else 0) for x in math["Average"] ]


# ### School

# Since we have data from two different schools, let's see if there are any differences in grades by school

# In[ ]:


math["school"].unique()


# In[ ]:


len(math[math["school"] == 'GP'])


# In[ ]:


len(math[math["school"] == 'MS'])


# 46 students is barely statistically significant. Moreover if we take into account that for model evaluation we have to split the data in train/test subsets, and that the sampling should be stratified. 

# Anyway, check if school is a good predictior.

# In[ ]:


data = [list( (math[math["school"] == 'GP']['G1'] + math[math["school"] == 'GP']['G2'] + math[math["school"] == 'GP']['G3']) / 3),
        list( (math[math["school"] == 'MS']['G1'] + math[math["school"] == 'MS']['G2'] + math[math["school"] == 'MS']['G3']) / 3)]


# In[ ]:


plt.boxplot(data)
plt.xticks(np.arange(3), ['', 'Gabriel Pereira', 'Mousinho da Silveira'])
plt.title("Grades per School")
plt.ylabel("Average Grades")
plt.xlabel("School")
plt.show()
plt.show()


# It seems there are no relevant differences.

# Let's see why people have chosen one school or the ohter.

# In[ ]:


math[math["school"] == 'GP']['reason'].value_counts().plot(kind='bar')


# In[ ]:


math[math["school"] == 'MS']['reason'].value_counts().plot(kind='bar')


# Fortunately is not due to school "reputation", since there are no differences in grades.

# ### Internet

# Let's see, for example, if having Internet at home makes any difference in grades.

# In[ ]:


math["internet"].unique()


# In[ ]:


len(math[math["internet"] == 'yes']), len(math[math["internet"] == 'no'])


# In[ ]:


data = [list(math[math["internet"] == "yes"]["Average"]), list(math[math["internet"] == "no"]["Average"])]


# In[ ]:


plt.boxplot(data)
plt.xticks(np.arange(3), ['', 'Yes', 'No'])
plt.title("Internet at Home")
plt.ylabel("Average Grades")
plt.xlabel("Internet")
plt.show()


# No, it seems it does not.

# ### Identify Predictive Features

# The same work we have done with the "internet" variable can be done with the rest of the categorical variables

# In[ ]:


data = [list(math[math["sex"]        == "M"]["Average"]),   list(math[math["sex"]        == "F"]["Average"]),
        list(math[math["address"]    == "U"]["Average"]),   list(math[math["address"]    == "R"]["Average"]),
        list(math[math["famsize"]    == "LE3"]["Average"]), list(math[math["famsize"]    == "GT3"]["Average"]),
        list(math[math["Pstatus"]    == "T"]["Average"]),   list(math[math["Pstatus"]    == "A"]["Average"]),
        list(math[math["schoolsup"]  == "yes"]["Average"]), list(math[math["schoolsup"]  == "no"]["Average"]),        
        list(math[math["higher"]     == "yes"]["Average"]), list(math[math["higher"]     == "no"]["Average"]),
        list(math[math["nursery"]    == "yes"]["Average"]), list(math[math["nursery"]    == "no"]["Average"]), 
        list(math[math["activities"] == "yes"]["Average"]), list(math[math["activities"] == "no"]["Average"]),
        list(math[math["paid"]       == "yes"]["Average"]), list(math[math["paid"]       == "no"]["Average"]),        
        list(math[math["famsup"]     == "yes"]["Average"]), list(math[math["famsup"]     == "no"]["Average"]),        
        list(math[math["romantic"]   == "yes"]["Average"]), list(math[math["romantic"]   == "no"]["Average"]),        
       ]


# Let's make the size of the figures a little bit bigger.

# In[ ]:


from matplotlib.pylab import rcParams


# In[ ]:


rcParams['figure.figsize'] = 20, 10


# In[ ]:


plt.boxplot(data)
plt.xticks(np.arange(23), ["", "Male", "Female", "Urban", "Rural", "LE3", "GE3", "Together", "Apart", "Support-Y", "Support-N", 
                          "High-Y", "High-N", "Nursery-Y", "Nursery-N", "Activities-Y", "Activities-N", "Paid-Y", "Paid-N",
                          "Famsup-Y", "Famsup-N", "Romance-Y", "Romance-N"])
plt.title("Predictive Feautres")
plt.ylabel("Average Grades")
plt.xlabel("Feature")
plt.show()


# Two good candidates as predictive features are if the student plans to study higher education and if the student has extra educational support. In the former case, probably, because students that want to continue studies make an extra effor to get better grades; and in the latter case because it seems that bad students require additional study support.

# ### Multicategory variables

# For the analysis of non-binary categorical variables we will use a simmilar technique.

# In[ ]:


math[["Mjob", "Average"]].boxplot(by="Mjob")
plt.title("Mother Jobs")
plt.ylabel("Average Grades")
plt.xlabel("Job")
plt.show()


# In[ ]:


math[["Fjob", "Average"]].boxplot(by="Fjob")
plt.title("Father Jobs")
plt.ylabel("Average Grades")
plt.xlabel("Job")
plt.show()


# In[ ]:


math[["Fjob", "Average"]].boxplot(by="Fjob")
plt.title("Student's Guardian")
plt.ylabel("Average Grades")
plt.xlabel("Guardian")
plt.show()


# All the three variables have low predictive power to the final grade, and so, we will not use them in the final model.

# ### Alcohol Consumption

# The same technique applied in the above section can be used to check if alcohol consumption is an indicator of poor school performance.

# In[ ]:


math[["Dalc", "Average"]].boxplot(by="Dalc")
plt.title("Daily Alcohol Consumption")
plt.ylabel("Average Grades")
plt.xlabel("Level of Consumption")
plt.show()


# In[ ]:


math[["Walc", "Average"]].boxplot(by="Walc")
plt.title("Weelend Alcohol Consumption")
plt.ylabel("Average Grades")
plt.xlabel("Level of Consumption")
plt.show()


# It seems that alcohol consumption is not strongly related to school performance.

# ### Predictive Models

# Given the characteristics of the dataset (number of samples and type of variables) I would recommend to use decision trees as the family of candidate models to consider. An advantage of decision tress is the interpretability of results, so we can understand why do students fail to pass exams.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# Let's use the numberical variables

# In[ ]:


attributes = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]


# In[ ]:


X = math[attributes]
y = math["Approved"]


# And the two identified categorical attributes.

# In[ ]:


X = X.assign(schoolsup = [ (1 if x == "yes" else 0) for x in math["schoolsup"] ])
X = X.assign(higher = [ (1 if x == "yes" else 0) for x in math["higher"] ])


# In[ ]:


attributes.append("schoolsup")
attributes.append("higher")


# In[ ]:


X.head()


# Fit a single model

# In[ ]:


model = DecisionTreeClassifier(min_samples_leaf=20)
model.fit(X, y)


# In[ ]:


model.score(X, y)


# Let's see what would happen in case of a random guessing.

# In[ ]:


np.sum(math["Approved"]) / len(math)


# It seems that the model has "true" prediction capabilities.

# Let's see how the tree looks like

# In[ ]:


import graphviz
from sklearn import tree


# In[ ]:


dot_data = tree.export_graphviz(model, out_file=None)
graph = graphviz.Source(dot_data)
graph


# In[ ]:


attributes[5]


# It seems that if a student has already failed, he will fail again.

# In[ ]:


attributes[13]


# And that if the student is getting external support is because it is likely that he will fail again

# The amount of alcohol ingested during week days nor weekends is not a predictive variable according to the identified model (it does not appear until the sixth level of the tree).

# ### Advanced Test

# The previous model was tested on the same training sample. A more realistic testing requires to use a separate training/testing subsets.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[ ]:


model = DecisionTreeClassifier(min_samples_leaf=20)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# We still have some predictive capabilities.

# And a more advanced testing uses a cross validation approach.

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit


# In[ ]:


model = DecisionTreeClassifier(min_samples_leaf=20)
cv_ss = ShuffleSplit(n_splits=100, test_size=0.3, random_state=1)
scores = cross_val_score(model, X, y, cv=cv_ss, n_jobs=-1)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# Now we can say we are confident about the reported accuracy. Unfortunately, the model has low predictive power.

# ### Conclusions

# The original question was if alcohol consumption affects the performance of students at school. Of course, it is impossible to answer that question given the fact that we have data from only two schools. The only question that could be answered with the dataset provided is if alcohol consumption is "somehow related" to school performance for the students of that school and for that particular year, and it turned out that it does not.
# 
# A serious study would require a large sample of randomly selected students and schools.
# 
# The causal implication of alcohol consumption to school performance can only be proved by means of running a controlled experiment. Unfortunately, that cannot be done in practice, sice we can not ask students to consume large amounts of alcohol during an accademic year.
# 
# A final note about "research ethics" is worth mentioning. It is highly surprising that the authors of the study have disclosed the real names of the schools from which the data was gathered. Morevoer, if we take into account that the dataset is, by no means, statistically significant. That could lead to all sorts of mistakes and misunderstandings that should have been avoided. 

# ### Future work

# Since we have a second dataset, the "language" dataset, it would be very nice to check if the grades of both datasets are correlated, and if not, if the conclusions are different. A difficult problem is how to relate the students from both datasets, what it is called "identity matching" problem. Fortunately, the authors of the dataset have provided a solution to this problem (in the form of an R script).
