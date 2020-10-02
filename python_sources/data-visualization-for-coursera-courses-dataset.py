#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


data = pd.read_csv('../input/coursera-course-dataset/coursea_data.csv')
print(data.shape)
data.head()


# In[ ]:


# replacing Unnamed: 0 column with Id
data['Id'] = data['Unnamed: 0']
data.drop(['Unnamed: 0'], axis=1, inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.info()


# ## Data Pre-processing
# As we can see that the column course_students_enrolled is of the type object. That is due to the "k" or "m" after the number. We need to deal with that before starting any analysis.

# In[ ]:


std_enroll = []
for i in range(0, len(data)):
    try:
        num = data['course_students_enrolled'].str.split('k')[i][0]
        num = float(num) * 1000 
        std_enroll.append(num)
    except:
        num = data['course_students_enrolled'].str.split('m')[i][0]
        num = float(num) * 1000000
        std_enroll.append(num)
data['course_students_enrolled'] = std_enroll
data['course_students_enrolled'] = data['course_students_enrolled'].astype(float)


# In[ ]:


# check using describe method
data.describe()


# Now, its fine. We can start with data analysis!

# ## Which organization has the most number of courses?
# Here, we want to check two things. First, which organization has the most number of courses on Coursera. Next, which all organizations have more than 10 courses on coursera. This is just a basic analysis which could show the organizations which dominate on Coursera.

# In[ ]:


def find_org_greater_than_10(data):
    """Returns a dataframe with course_organization and number of courses > 10"""
    dict = {}
    course_org = data['course_organization'].to_list()
    for org in course_org:
        if org in dict:
            dict[org] += 1
        else:
            dict[org] = 1
    orgs = []
    counts = []
    for key, value in dict.items():
        if value > 10:
            orgs.append(key)
            counts.append(value)
        else:
            continue
    course_org_greater_than_1 = pd.DataFrame({'course_organization':orgs, 'count':counts})
    course_org_greater_than_1.sort_values(by='count', ascending=False, inplace=True)
    return course_org_greater_than_1


# In[ ]:


course_org_greater_than_1 = find_org_greater_than_10(data)

# plot a barh chart
course_org_greater_than_1.plot(kind='barh', x='course_organization', y='count')
plt.title('Organizations with more than 10 courses')
plt.xlabel('count')
plt.show()


# University of Pennsylvania dominates here, with just about 60 courses, followed University of Michigan who has just over 40 courses on Coursera!

# ## Average course rating of the organizations with more than 10 courses
# Here, we want to see whether these organizations produce quality courses or is it just quantity over quality?

# In[ ]:


# dictionary containing organization as key and avg rating as value
dom_dict = round(data.groupby('course_organization')['course_rating'].mean(), 1).to_dict()


# In[ ]:


# Filter out organizations, as we only want those organizations with more than 10 courses
orgs = course_org_greater_than_1['course_organization'].to_list()
avg_rating = []
for org in orgs:
    for key, value in dom_dict.items():
        if key == org:
            avg_rating.append(value)
        else:
            continue
course_org_greater_than_1['avg_rating'] = avg_rating


# In[ ]:


course_org_greater_than_1


# In[ ]:


# plot a barh chart
course_org_greater_than_1.plot(kind='barh', x='course_organization', y='avg_rating')
plt.title('Average course rating of organizations with more than 10 courses')
plt.xlabel('Average course rating')
plt.show()


# As you can see, there is a very little difference between the average course rating of all these organizations. Hence, they don't focus on just quantity.

# ## Which organization has the highest course rating?
# What do you think? Will the organization with highest course rating have more than 10 courses?

# In[ ]:


dom_dict = round(data.groupby('course_organization')['course_rating'].mean(), 1).to_dict()
dom_dict = {k: v for k, v in sorted(dom_dict.items(), key=lambda item: item[1], reverse=True)}
dom_dict


# In[ ]:


for key, value in dom_dict.items():
    if value == 4.9:
        print(key)


# As you can see, none of these organizations belong to that group who have more than 10 courses.

# ## Top 5 courses on Coursera
# Let's look at the top 5 courses on Coursera by course rating

# In[ ]:


course_dict = data.groupby('course_title')['course_rating'].mean().to_dict()
course_dict = {k: v for k, v in sorted(course_dict.items(), key=lambda item: item[1], reverse=True)}
course_dict


# In[ ]:


cour_df = pd.DataFrame({'course_title':list(course_dict.keys()), 'course_rating':list(course_dict.values())})
cour_df[:10]


# ## Does course difficulty affect number of students enrolled?
# 
# For this we will check the average number of students enrolled grouping by difficulty.

# In[ ]:


stud_dict = round(data.groupby('course_difficulty')['course_students_enrolled'].mean(), 0).to_dict()
stud_dict = {k: v for k, v in sorted(stud_dict.items(), key=lambda item: item[1], reverse=True)}
stud_dict


# In[ ]:


diff_stud = pd.DataFrame({'difficulty':list(stud_dict.keys()), 'avg_students':list(stud_dict.values())})
diff_stud


# In[ ]:


diff_stud.plot(kind='bar', x='difficulty', y='avg_students', title='Distribution by difficulty')
plt.ylabel('Number of students')


# Well, there are more number of students enrolled per Mixed difficulty level course rather than Beginner.

# ## Does course difficulty affect rating?
# Do courses with advanced level of difficulty get lower rating? Is there any relation between the two?

# In[ ]:


rate_dict = round(data.groupby('course_difficulty')['course_rating'].mean(), 1).to_dict()
rate_dict = {k: v for k, v in sorted(rate_dict.items(), key=lambda item: item[1], reverse=True)}
rate_dict


# In[ ]:


diff_stud['avg_rating'] = list(rate_dict.values())
diff_stud.plot(kind='bar', x='difficulty', y='avg_rating', title='Average rating by difficulty')


# Nope, very less difference in the ratings.

# ## Does certificate type impact number of students enrolled?
# This could be a very important question for Coursera and for the organizations who are making courses. What type of certificate should be there for maximum student enrollment and how should Coursera optimize the course recommendations based on the type of certificates?

# In[ ]:


cert_dict = round(data.groupby('course_Certificate_type')['course_students_enrolled'].mean(), 0).to_dict()
cert_dict = {k: v for k, v in sorted(cert_dict.items(), key=lambda item: item[1], reverse=True)}
cert_dict


# In[ ]:


cert_df = pd.DataFrame({'course_Certificate_type':list(cert_dict.keys()), 'avg_students':list(cert_dict.values())})
cert_df.plot(kind='bar', x='course_Certificate_type', y='avg_students', title='Distribution by course_Certificate_type')
plt.ylabel('Number of students')
plt.show()


# Yes! More number of students on an average enroll in Professional Certificate courses followed by a specialization.

# ## Does certificate type impact course rating?
# Do professional certificates have a higher rating because they are accepted everywhere or is there still no impact?

# In[ ]:


rate_cert_dict = round(data.groupby('course_Certificate_type')['course_rating'].mean(), 1).to_dict()
rate_cert_dict = {k: v for k, v in sorted(rate_cert_dict.items(), key=lambda item: item[1], reverse=True)}
rate_cert_dict


# In[ ]:


cert_df['avg_rating'] = list(rate_cert_dict.values())
cert_df.plot(kind='bar', x='course_Certificate_type', y='avg_rating', title='Average rating by certificate type')
plt.ylabel('Number of students')
plt.show()


# Still, no difference.

# ## Number of students enrolled vs. course rating
# Last but not the least, if there a large number of students enrolled in your course, does that impact the course rating?

# In[ ]:


data.plot(kind='scatter', x='course_students_enrolled', y='course_rating', title='Number of students enrolled vs. course rating')


# There is a small datapoint at the bottom which says that less number of students meaning low rating, but this does not prove our assumption as there is no strong relation. We can check how strong is the relation by finding correlation between the two.

# In[ ]:


data.corr()


# You see, there is a very small positive correlation, hence its not enough to prove our assumption that more number of students enrolled in your course does lead to higher rating.

# ## Conclusion
# 
# - University of Pennsylvania has most number of courses on Coursera followed by University of Michigan.
# - Organizations with more than 10 courses on Coursera are not present in the top 5 organizations list with highest rating.
# - There is no Computer Science course in the top 10 highest rated courses on Coursera. (Please prove me wrong!)
# - Course difficulty does affect the number of students enrolled. There are more students enrolled in Beginner and Mixed level of difficulty courses than Intermediate and Advanced.
# - Course difficulty does not affect course rating.
# - Course certificate type does impact the number of students enrolled in that course. More number of students tend to enroll for Professional Certificate courses than just "COURSE" certificate.
# - Certificate type does not impact course rating.
# - There is no strong relation to prove that if more number of students are enrolled in a course, the course rating is a higher.

# In[ ]:




