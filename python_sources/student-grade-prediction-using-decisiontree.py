#!/usr/bin/env python
# coding: utf-8

# # Step 1. Acquiring the data

# Let's import the necessary libraries:

# In[ ]:


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus


# Import the CSV file:

# In[ ]:


data = pd.read_csv('../input/student-mat.csv')
data.shape


# # Step 2. Data preparation

# Check for NULL values:

# In[ ]:


data.isnull().values.any()


# OK great so there are no NULL values, let's see the data preview

# In[ ]:


data.columns


# **There are 33 columns:**
# 
# **school:**
# student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
# 
# **sex:**
# student's sex (binary: 'F' - female or 'M' - male)
# 
# **age:**
# student's age (numeric: from 15 to 22)
# 
# **address:**
# student's home address type (binary: 'U' - urban or 'R' - rural)
# 
# **famsize:**
# family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
# 
# **Pstatus:**
# parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
# 
# **Medu:**
# mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
# 
# **Fedu:**
# father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
# 
# **Mjob:**
# mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
# 
# **Fjob:**
# father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
# 
# **reason:**
# reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
# 
# **guardian:**
# student's guardian (nominal: 'mother', 'father' or 'other')
# 
# **traveltime:**
# home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
# 
# **studytime:**
# weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
# 
# **failures:**
# number of past class failures (numeric: n if 1<=n<3, else 4)
# 
# **schoolsup:**
# extra educational support (binary: yes or no)
# 
# **famsup:**
# family educational support (binary: yes or no)
# 
# **paid:**
# extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
# 
# **activities:**
# extra-curricular activities (binary: yes or no)
# 
# **nursery:**
# attended nursery school (binary: yes or no)
# 
# **higher:**
# wants to take higher education (binary: yes or no)
# 
# **internet:**
# Internet access at home (binary: yes or no)
# 
# **romantic:**
# with a romantic relationship (binary: yes or no)
# 
# **famrel:**
# quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
# 
# **freetime:**
# free time after school (numeric: from 1 - very low to 5 - very high)
# 
# **goout:**
# going out with friends (numeric: from 1 - very low to 5 - very high)
# 
# **Dalc:**
# workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 
# **Walc:**
# weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 
# **health:**
# current health status (numeric: from 1 - very bad to 5 - very good)
# 
# **absences:**
# number of school absences (numeric: from 0 to 93)
# 
# **G1:**
# first period grade (numeric: from 0 to 20)
# 
# **G2:**
# second period grade (numeric: from 0 to 20)
# 
# **G3:**
# final grade (numeric: from 0 to 20)

# In[ ]:


data.head()


# How many unique data?

# In[ ]:


data.nunique()


# From the data above, let's create one more column to get the average grade from G1 to G3 (3 years average):

# In[ ]:


data['GAvg'] = (data['G1'] + data['G2'] + data['G3']) / 3


# Now lets create a grading based on its G Average:
# 
# Above 90% = Grade A
# 
# Between 70% & 90% = Grade B
# 
# Below 70% = Grade C

# In[ ]:


def define_grade(df):
    # Create a list to store the data
    grades = []

    # For each row in the column,
    for row in df['GAvg']:
        # if more than a value,
        if row >= (0.9 * df['GAvg'].max()):
            # Append a letter grade
            grades.append('A')
        # else, if more than a value,
        elif row >= (0.7 * df['GAvg'].max()):
            # Append a letter grade
            grades.append('B')
        # else, if more than a value,
        elif row < (0.7 * df['GAvg'].max()):
            # Append a letter grade
            grades.append('C')   
    # Create a column from the list
    df['grades'] = grades
    return df


# In[ ]:


data = define_grade(data)
data.head()


# Next, we can drop school name and age feature because it is not a computational value
# 

# In[ ]:


data.drop(["school","age"], axis=1, inplace=True)


# # Step 3. Data analyzing
# 

# OK, to have a better picture of the data, let's describe the values

# In[ ]:


data.describe()


# **Some insights of the stats above:**
# 
# **Age:** Average Age of the respondent is 16 Years Old
# 
# **traveltime** Some kids travel 4 Hours a day just to reach school
# 
# **famrel** Average kids have a good relationship with their family
# 
# **absences** Average kids only have 6 days absences, and we spot outliers with 75 days absences
# 
# **Dalc** Daily alcohol consumption among kids is very low (which is good)

# # Step 4. Build Machine Learning Model

# In this project, we will use DecisionTree Algorithm to predict the result.
# To be able to work with DecisionTree, The sklearn library requires all computational values to be **numerical**, so first, 
# we should convert those categorical values to numerical values

# In[ ]:


# for yes / no values:
d = {'yes': 1, 'no': 0}
data['schoolsup'] = data['schoolsup'].map(d)
data['famsup'] = data['famsup'].map(d)
data['paid'] = data['paid'].map(d)
data['activities'] = data['activities'].map(d)
data['nursery'] = data['nursery'].map(d)
data['higher'] = data['higher'].map(d)
data['internet'] = data['internet'].map(d)
data['romantic'] = data['romantic'].map(d)


# Then for the rest categorical values:

# In[ ]:


# map the sex data
d = {'F': 1, 'M': 0}
data['sex'] = data['sex'].map(d)

# map the address data
d = {'U': 1, 'R': 0}
data['address'] = data['address'].map(d)

# map the famili size data
d = {'LE3': 1, 'GT3': 0}
data['famsize'] = data['famsize'].map(d)

# map the parent's status
d = {'T': 1, 'A': 0}
data['Pstatus'] = data['Pstatus'].map(d)

# map the parent's job
d = {'teacher': 0, 'health': 1, 'services': 2,'at_home': 3,'other': 4}
data['Mjob'] = data['Mjob'].map(d)
data['Fjob'] = data['Fjob'].map(d)

# map the reason data
d = {'home': 0, 'reputation': 1, 'course': 2,'other': 3}
data['reason'] = data['reason'].map(d)

# map the guardian data
d = {'mother': 0, 'father': 1, 'other': 2}
data['guardian'] = data['guardian'].map(d)

# map the grades data
d = {'C': 0, 'B': 1, 'A': 2}
data['grades'] = data['grades'].map(d)


# Let's see the unique data again, just to make sure that we have done the mapping successfully.

# In[ ]:


data.nunique()


# Now we can collect all predictive feature columns, and then remove **grades** from it because **grades** is our target

# In[ ]:


student_features = data.columns.tolist()
student_features.remove('grades') 
student_features.remove('GAvg') 
student_features.remove('G1') 
student_features.remove('G2') 
student_features.remove('G3') 
student_features


# Then we can copy those features to **X**

# In[ ]:


X = data[student_features].copy()
X.columns


# also for the target to **y**

# In[ ]:


y=data[['grades']].copy()


# Next we can split the train data and test data using **train_test_split** function

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)


# Then we can create the classifier 

# In[ ]:


grade_classifier = tree.DecisionTreeClassifier(max_leaf_nodes=len(X.columns), random_state=0)
grade_classifier.fit(X_train, y_train)


# We can also view how the grade_classifier divide the logic using **pydotplus** library

# In[ ]:


dot_data = StringIO()  
tree.export_graphviz(grade_classifier, out_file=dot_data,  
                         feature_names=student_features)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  


# let's create our prediction :

# In[ ]:


predictions = grade_classifier.predict(X_test)


# Finally, we can measure the accuracy of our classifier:

# In[ ]:


accuracy_score(y_true = y_test, y_pred = predictions)


# Well, accuracy score of **0.775** is not bad, we can also tune the hyperparameters to increase the accuracy score.

# In[ ]:




