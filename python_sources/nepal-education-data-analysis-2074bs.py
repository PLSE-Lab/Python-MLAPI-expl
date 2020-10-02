#!/usr/bin/env python
# coding: utf-8

# # Nepal Education Data Analysis(2074BS)
# 
# <img src="data:image/jpeg;base64,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" width="1024" height="1024">
# 
# # Table of Content
# 
# 1. Sex-Wise Student Enrollment of Higher Education by Province in 2074 BS
# 1. Grade-wise Student Enrollment of School Education in 2074 BS
# 1. Faculty-Wise Student Enrollment of Higher Education in 2074 BS
# 1. Level-Wise Student Enrollment of Higher Education in 2074 BS
# 1. Literacy Rate of 5 Years and Above by Province in 2011AD
# 1. Total Number of Teacher at Primary to Higher Secondary Level in all Types of Schools in 2074BS
# 1. University-Wise Student Enrollment of Higher Education by Sex in 2074 BS
# 1. Province-Wise Student Enrollment of Higher Education by Level in 2074 BS
# 1. Total Number of Universities and Campuses/Colleges by Province in 2074 BS
# 1. University-Wise Student Enrollment of Higher Education by Levels in 2074 BS
# 1. Faculty Wise Total Student Enrollment in Higher Education by Province in 2074 BS
# 1. Analysis of Teachers
# 1. Summary

# In[ ]:


import numpy as np 
import pandas as pd 
import plotly.graph_objects as go

from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objects as go
import plotly.express as px

import cufflinks as cf
cf.go_offline()

from plotly.subplots import make_subplots


# * You can get data from : https://www.kaggle.com/milan400/education-in-figures-2017-at-a-glance
# * Data is extracted from : https://moe.gov.np/assets/uploads/files/Educational_Brochure_2017.pdf
# 
# * This notebook contains Visualization gained from some of the data. The data maynot be in proper form. So, there is need to do preprocessing.
# 
# * A lot of insight of **Nepal Education System** can be gained from Visualization

# # Sex-Wise Student Enrollment of Higher Education by Province in 2074 BS

# In[ ]:


student_enrollment_sex = pd.read_csv('/kaggle/input/education-in-figures-2017-at-a-glance/data_csv_student/nepal_education_csv/Male_Female_CSV/Sex-Wise Student Enrollment of Higher Education by Province in 2074 BS.csv')
#student_enrollment_sex = student_enrollment_sex.dropna(axis=1)

#Extracting only Province, Male, Female Column
student_enrollment_sex = student_enrollment_sex.drop(columns = ['Total'])
student_enrollment_sex = student_enrollment_sex[student_enrollment_sex['Province'] !='Total']

student_enrollment_sex.iplot(x='Province', y=['Female','Male'], kind='bar', xTitle='Sex-Wise Student Enrollment of Higher Education by Province in 2074 BS', yTitle='No of Students')


# **PieChart: Sex-Wise Student Enrollment of Higher Education by Province in 2074 BS**

# In[ ]:


labels = student_enrollment_sex['Province'].tolist()

values1 = student_enrollment_sex['Male'].tolist()
values2 = student_enrollment_sex['Female'].tolist()


# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=labels, values=values1, name="Male"),
              1, 1)
fig.add_trace(go.Pie(labels=labels, values=values2, name="Female"),
              1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
    autosize=True,
    width=800,
    title_text="Sex-Wise Student Enrollment of Higher Education by Province in 2074 BS",
    # Add annotations in the center of the donut pies.
    annotations=[
                 dict(text='Boys', x=0.2, y=0.5, font_size=13, showarrow=False),
                dict(text='Girls', x=0.8, y=0.5, font_size=13, showarrow=False)])
fig.show()


# **Bubble Chart: Sex-Wise Student Enrollment of Higher Education by Province in 2074 BS**

# In[ ]:


fig = px.scatter(student_enrollment_sex, x="Male", y="Female", size="Male", color="Province",
           hover_name="Province", log_x=True, size_max=60)
fig.show()


# * Province 3 has the highest number of students enrollment
# * Karnali has the lowest number of students enrollment

# # Grade-wise Student Enrollment of School Education in 2074 BS

# In[ ]:


grade_wise_student = pd.read_csv('/kaggle/input/education-in-figures-2017-at-a-glance/data_csv_student/nepal_education_csv/Male_Female_CSV/Grade-wise Student Enrollment of School Education in 2074 BS.csv')

grade_wise_student = grade_wise_student[['Grade','Gis','Boys','Total']]
grade_wise_student = grade_wise_student.rename(columns={'Gis':'Girls'})

#Removing the values of range
noise = ['Total Grade (1-8)',' 	Grades(6-8)','Grade (9-10)', 'Grade (11-12)','Grand Total (1-12)','Grades(1-5)','Grades(6-8)']
for n in noise:
    grade_wise_student = grade_wise_student[grade_wise_student.Grade != n]
grade_wise_student.iplot(x='Grade', y=['Girls','Boys'], kind='bar', xTitle='Grade-wise Student Enrollment of School Education in 2074 BS', yTitle='No of Students')


# In[ ]:


grade_wise_student.iplot(x='Grade', y=['Girls','Boys'], kind='scatter', xTitle='Grade-wise Student Enrollment of School Education in 2074 BS',yTitle='No of Students')


# **Pie Chart for Enrollment of Student**

# In[ ]:


labels = grade_wise_student['Grade'].tolist()

values1 = grade_wise_student['Total'].tolist()
values2 = grade_wise_student['Boys'].tolist()
values3 = grade_wise_student['Girls'].tolist()


# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])

fig.add_trace(go.Pie(labels=labels, values=values1, name="Totals Students"),
              1, 1)
fig.add_trace(go.Pie(labels=labels, values=values2, name="Boys"),
              1, 2)
fig.add_trace(go.Pie(labels=labels, values=values3, name="Girls"),
              1, 3)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
    autosize=True,
    width=1500,
    title_text="Student Enrollment per Grade",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Total Students', x=0.11, y=0.5, font_size=13, showarrow=False),
                 dict(text='Boys', x=0.5, y=0.5, font_size=13, showarrow=False),
                dict(text='Girls', x=0.87, y=0.5, font_size=13, showarrow=False)])
fig.show()


# **Bubble Chart for Enrollment of Student**

# In[ ]:


fig = px.scatter(grade_wise_student, x="Boys", y="Girls", size="Boys", color="Grade",
           hover_name="Grade", log_x=True, size_max=60)
fig.show()


# * As the Grade increase, number of student goes on decreasing
# * Number of students in Grade 11 and 12  seems quite same
# * Grade 11 and 12 enrollment for both gender is quite low
# * Number of students increases from Grade 7 to Grade 8 but keeps on decreasing after that Grade

# # Faculty-Wise Student Enrollment of Higher Education in 2074 BS

# In[ ]:


Faculty_wise = pd.read_csv('/kaggle/input/education-in-figures-2017-at-a-glance/data_csv_student/nepal_education_csv/Male_Female_CSV/Faculty-Wise Student Enrollment of Higher Education in 2074 BS.csv')
Faculty_wise = Faculty_wise.dropna()
Faculty_wise = Faculty_wise.drop(columns=['SN'])
Faculty_wise['Faculties'] = Faculty_wise['Faculties'].replace({'Sanskr t':'Sanskrit'})

Faculty_wise.iplot(x='Faculties', y=['Female','Male'], kind='bar', xTitle='Faculty-Wise Student Enrollment of Higher Education in 2074 BS')


# **PieChart: Faculty-Wise Student Enrollment of Higher Education in 2074 BS**

# In[ ]:


labels = Faculty_wise['Faculties'].tolist()

values1 = Faculty_wise['Total'].tolist()
values2 = Faculty_wise['Male'].tolist()
values3 = Faculty_wise['Female'].tolist()


# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])

fig.add_trace(go.Pie(labels=labels, values=values1, name="Totals Students"),
              1, 1)
fig.add_trace(go.Pie(labels=labels, values=values2, name="Boys"),
              1, 2)
fig.add_trace(go.Pie(labels=labels, values=values3, name="Girls"),
              1, 3)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
    autosize=True,
    width=1500,
    title_text="Faculty wise student Enrollment",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Total Students', x=0.11, y=0.5, font_size=13, showarrow=False),
                 dict(text='Male', x=0.501, y=0.5, font_size=13, showarrow=False),
                dict(text='Female', x=0.88, y=0.5, font_size=13, showarrow=False)])
fig.show()


# **Bubble Chart: Faculty-Wise Student Enrollment of Higher Education in 2074 BS**

# In[ ]:


fig = px.scatter(Faculty_wise, x="Male", y="Female", size="Male", color="Faculties",
           hover_name="Faculties", log_x=True, size_max=60)
fig.show()


# * Management has the highest student enrollment for both gender
# * In Management, Education,Humanities and Medical female has more enrollment
# * in Engineering and Science&Technology, male has more enrollment
# * Buddism has lowest enrollment

# # Level-Wise Student Enrollment of Higher Education in 2074 BS

# In[ ]:


level_wise = pd.read_csv('/kaggle/input/education-in-figures-2017-at-a-glance/data_csv_student/nepal_education_csv/Male_Female_CSV/Level-Wise Student Enrollment of Higher Education in 2074 BS.csv')
level_wise = level_wise.rename(columns={'male':'Female'})
level_wise = level_wise[level_wise['Level'] != 'Total']

level_wise.iplot(x='Level', y=['Female','Male'], kind='bar', xTitle='Level-Wise Student Enrollment of Higher Education in 2074 BS')


# **Pie Chart: Level-Wise Student Enrollment of Higher Education in 2074 BS**

# In[ ]:


labels = level_wise['Level'].tolist()

values1 = level_wise['Total'].tolist()
values2 = level_wise['Male'].tolist()
values3 = level_wise['Female'].tolist()


# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])

fig.add_trace(go.Pie(labels=labels, values=values1, name="Totals Students"),
              1, 1)
fig.add_trace(go.Pie(labels=labels, values=values2, name="Boys"),
              1, 2)
fig.add_trace(go.Pie(labels=labels, values=values3, name="Girls"),
              1, 3)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
    autosize=True,
    width=1500,
    title_text="Level-Wise Student Enrollment of Higher Education in 2074 BS",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Total Students', x=0.11, y=0.5, font_size=13, showarrow=False),
                 dict(text='Male', x=0.501, y=0.5, font_size=13, showarrow=False),
                dict(text='Female', x=0.88, y=0.5, font_size=13, showarrow=False)])
fig.show()


# **Bubble Chart: Level-Wise Student Enrollment of Higher Education in 2074 BS**

# In[ ]:


fig = px.scatter(level_wise, x="Male", y="Female", size="Male", color="Level",
           hover_name="Level", log_x=True, size_max=60)
fig.show()


# * Compared to Bachelor, very few student enroll in further studies
# * More number of female enrollment in Bachelor 
# * Slightly more number of male enrollment in Master 
# * After their Bachelor most people donot continue Academics(at least in nepal)

# # Literacy Rate of 5 Years and Above by Province in 2011AD

# In[ ]:


campus_wise = pd.read_csv('/kaggle/input/education-in-figures-2017-at-a-glance/data_csv_student/nepal_education_csv/Male_Female_CSV/Literacy Rate of 5 Years and Above by Province in 2011AD.csv')
campus_wise = campus_wise.dropna(axis=1)

campus_wise_transpose = campus_wise.T
campus_wise_transpose.set_axis(['Both Sex','Male','Female'], axis='columns', inplace=True)

campus_wise_transpose = campus_wise_transpose[campus_wise_transpose['Both Sex'] != 'Both Sex']

campus_wise_transpose = campus_wise_transpose.reset_index()
campus_wise_transpose = campus_wise_transpose.rename(columns={'index':'Province'})

campus_wise_transpose = campus_wise_transpose[campus_wise_transpose['Province'] != 'Total']
campus_wise_transpose = campus_wise_transpose.astype({'Province':str,'Male':float,'Female':float,'Both Sex':float})

campus_wise_transpose.iplot(x='Province', y=['Female','Male'], kind='bar', xTitle='Literacy Rate of 5 Years and Above by Province in 2011AD')


# **Pie Chart: Literacy Rate of 5 Years and Above by Province in 2011AD**

# In[ ]:



labels = campus_wise_transpose['Province'].tolist()

values1 = campus_wise_transpose['Both Sex'].tolist()
values2 = campus_wise_transpose['Male'].tolist()
values3 = campus_wise_transpose['Female'].tolist()


# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])

fig.add_trace(go.Pie(labels=labels, values=values1, name="Totals Students"),
              1, 1)
fig.add_trace(go.Pie(labels=labels, values=values2, name="Boys"),
              1, 2)
fig.add_trace(go.Pie(labels=labels, values=values3, name="Girls"),
              1, 3)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
    autosize=True,
    width=1500,
    title_text="Literacy Rate per Province",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Total Students', x=0.11, y=0.5, font_size=13, showarrow=False),
                 dict(text='Male', x=0.501, y=0.5, font_size=13, showarrow=False),
                dict(text='Female', x=0.875, y=0.5, font_size=13, showarrow=False)])
fig.show()


# **Bubble Chart: Literacy Rate of 5 Years and Above by Province in 2011AD**

# In[ ]:


fig = px.scatter(campus_wise_transpose, x="Male", y="Female", size="Male", color="Province",
           hover_name="Province", log_x=True, size_max=60)
fig.show()


# * Province 2 has the lowest literacy of 5 years
# * Province 3 and Gandaki has the highest number of literacy of 5 years

# # University-Wise Student Enrollment of Higher Education by Sex in 2074 BS

# In[ ]:


university_student = pd.read_csv('/kaggle/input/education-in-figures-2017-at-a-glance/data_csv_student/nepal_education_csv/University-Wise Student Enrollment of Higher Education by Sex in 2074 BS.csv')
university_student = university_student.dropna()
university_student.iplot(x='University', y=['Female','Male'], kind='bar', xTitle='University-Wise Student Enrollment of Higher Education by Sex in 2074 BS')


# **Pie Chart: University-Wise Student Enrollment of Higher Education by Sex in 2074 BS**

# In[ ]:


labels = university_student['University'].tolist()

values1 = university_student['Total'].tolist()
values2 = university_student['Male'].tolist()
values3 = university_student['Female'].tolist()


# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])

fig.add_trace(go.Pie(labels=labels, values=values1, name="Totals Students"),
              1, 1)
fig.add_trace(go.Pie(labels=labels, values=values2, name="Boys"),
              1, 2)
fig.add_trace(go.Pie(labels=labels, values=values3, name="Girls"),
              1, 3)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
    autosize=True,
    height = 600,
    width=1500,
    title_text="University-Wise Student Enrollment of Higher Education by Sex in 2074 BS",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Total Students', x=0.11, y=0.5, font_size=13, showarrow=False),
                 dict(text='Male', x=0.501, y=0.5, font_size=13, showarrow=False),
                dict(text='Female', x=0.88, y=0.5, font_size=13, showarrow=False)])
fig.show()


# **Bubble Chart: University-Wise Student Enrollment of Higher Education by Sex in 2074 BS**

# In[ ]:


fig = px.scatter(university_student, x="Male", y="Female", size="Male", color="University",
           hover_name="University", log_x=True, size_max=60)
fig.show()


# * Tribhuvan University has the highest number of student enrollment
# * Rajarshi Janaki Univeristy, Nepal open University, Karnali Academy Health Science has no student enrollment
# * Except Tribuwan University , Female enrollement is more in other Univeristy compared to Male

# # Province-Wise Student Enrollment of Higher Education by Level in 2074 BS

# In[ ]:


province_wise_higher = pd.read_csv('/kaggle/input/education-in-figures-2017-at-a-glance/data_csv_student/nepal_education_csv/Province-Wise Student Enrollment of Higher Education by Level in 2074 BS.csv')
province_wise_higher = province_wise_higher.dropna(axis=1)
province_wise_higher = province_wise_higher[province_wise_higher['Province'] != 'Total']

province_wise_higher.iplot(x='Province', y=['Bachelor','PGD', 'Master' ,'M Phil', 'Ph.D'], kind='bar', xTitle='Province-Wise Student Enrollment of Higher Education by Level in 2074 BS')


# **PieChart: Province-Wise Student Enrollment of Higher Education by Level in 2074 BS**

# In[ ]:


province_wise_higher.iplot(labels='Province', values='Total', kind='pie', title='Province-Wise Student Enrollment of Higher Education by Level in 2074 BS')


# * Province 3 has the highest enrollment in all higher education
# * karnali has the lowest enrollment in all higher education
# * Only Province 3 has enrollment after Master

# # Total Number of Universities and Campuses/Colleges by Province in 2074 BS

# In[ ]:


uni_colg_province = pd.read_csv('/kaggle/input/education-in-figures-2017-at-a-glance/data_csv_student/nepal_education_csv/Total Number of Universities and Campuses I Colleges by Province in 2074 BS.csv')
uni_colg_province = uni_colg_province.drop(columns=['Unnamed: 6', 'SN'])
uni_colg_province = uni_colg_province.dropna(axis=0)

uni_colg_province_transpose = uni_colg_province.T
uni_colg_province_transpose = uni_colg_province_transpose.reset_index()

uni_colg_province_transpose.set_axis(['Province','Tribhuvan University', 'NepalSanskrit University', 'Kathmandu University', 'Purbanchal University', 'Pokhara University', 'LumbiniBauddha University','Agriculture and Forestry University', 'Mid-Western University', 'Far Western University', 'BP Koirala Institute for Health Sciences', 'National Academy of Medical Sciences', 'Patan Academy of Health Science', 'KarnaliAcademy HealthSciences', 'Total'], axis='columns', inplace=True)

noise_not_needed = ['University', 'Total']
for col in noise_not_needed:
    uni_colg_province_transpose = uni_colg_province_transpose[uni_colg_province_transpose['Province'] != col]

uni_colg_province_transpose.iplot(x='Province', y=['Tribhuvan University', 'NepalSanskrit University', 'Kathmandu University', 'Purbanchal University', 'Pokhara University', 'LumbiniBauddha University','Agriculture and Forestry University', 'Mid-Western University', 'Far Western University', 'BP Koirala Institute for Health Sciences', 'National Academy of Medical Sciences', 'Patan Academy of Health Science', 'KarnaliAcademy HealthSciences'], kind='bar', xTitle='Total Number of Universities and Campuses/Colleges by Province in 2074 BS')


# **Pie Chart: Total Number of Universities and Campuses/Colleges by Province in 2074 BS**

# In[ ]:


uni_colg_province_transpose.iplot(labels='Province', values='Total', kind='pie', title='Total Number of Universities and Campuses/Colleges by Province in 2074 BS')


# # University-Wise Student Enrollment of Higher Education by Levels in 2074 BS

# In[ ]:


uni_student_enroll = pd.read_csv('/kaggle/input/education-in-figures-2017-at-a-glance/data_csv_student/nepal_education_csv/University-Wise Student Enrollment of Higher Education by Levels in 2074 BS.csv')
uni_student_enroll = uni_student_enroll.dropna()
uni_student_enroll = uni_student_enroll.drop(columns=['SN'])

uni_student_enroll = uni_student_enroll.rename(columns={'Ph. D':'PhD', 'M.Phil':'MPhil'})

uni_student_enroll.iplot(x='University', y=['Bachelor','PGD', 'Master' ,'MPhil', 'PhD'], kind='bar', xTitle='University-Wise Student Enrollment of Higher Education by Levels in 2074 BS')


# **PieChart: University-Wise Student Enrollment of Higher Education by Levels in 2074 BS**

# In[ ]:


uni_student_enroll.iplot(labels='University', values='Total', kind='pie', title={'text':'University-Wise Student Enrollment of Higher Education by Levels in 2074 BS','y':'0.99'})


# * Province 3 has highest number of University and its afflicated colleges
# * Tribuwan University and its afflicated colleges are more in number
# * Province 5  has the highest number of Sanskrit University and its afflicated colleges

# # Analysis of Teachers
# **Total Number of Teacher at Primary to Higher Secondary Level in all Types of Schools in 2074BS**

# In[ ]:


teacher_highersecondary = pd.read_csv('/kaggle/input/education-in-figures-2017-at-a-glance/data_csv_student/nepal_education_csv/Male_Female_CSV/Total Number of Teacher at Primary to Higher Secondary Level in all Types of Schools in 2074BS_secondone.csv')
teacher_highersecondary.set_axis(['Province','Female','Male','Total'], axis='columns', inplace=True)
teacher_highersecondary = teacher_highersecondary.dropna(axis=0)
teacher_highersecondary = teacher_highersecondary.astype({'Province':str,'Male':int,'Female':int,'Total':int})
teacher_highersecondary.iplot(x='Province', y=['Female','Male'],kind='bar', xTitle='Total Number of Teacher at Primary to Higher Secondary Level in all Types of Schools in 2074BS')


# **PieChart: Total Number of Teacher at Primary to Higher Secondary Level in all Types of Schools in 2074BS**

# In[ ]:


labels = teacher_highersecondary['Province'].tolist()

values1 = teacher_highersecondary['Total'].tolist()
values2 = teacher_highersecondary['Male'].tolist()
values3 = teacher_highersecondary['Female'].tolist()


# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])

fig.add_trace(go.Pie(labels=labels, values=values1, name="Totals Students"),
              1, 1)
fig.add_trace(go.Pie(labels=labels, values=values2, name="Boys"),
              1, 2)
fig.add_trace(go.Pie(labels=labels, values=values3, name="Girls"),
              1, 3)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
    autosize=True,
    height = 600,
    width=1500,
    title_text="Total Number of Teacher at Primary to Higher Secondary Level in all Types of Schools in 2074BS",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Total Students', x=0.11, y=0.5, font_size=13, showarrow=False),
                 dict(text='Male', x=0.501, y=0.5, font_size=13, showarrow=False),
                dict(text='Female', x=0.88, y=0.5, font_size=13, showarrow=False)])
fig.show()


# **Bubble Chart: Total Number of Teacher at Primary to Higher Secondary Level in all Types of Schools in 2074BS**

# In[ ]:


fig = px.scatter(teacher_highersecondary, x="Male", y="Female", size="Male", color="Province",
           hover_name="Province", log_x=True, size_max=60)
fig.show()


# * Province 3 has highest number of teachers
# * Karnali has lowest number of teachers
# * Difference in number of male and female teacher is lowest in province 3

# # Summary:
# 
# * Province 3 has the highest number of students and teachers
# * Karnali has the lowest number of students and teachers
# * As the Grade increase, number of student goes on decreasing
# * Management has the highest student enrollment for both gender
# * In Management, Education,Humanities and Medical female has more enrollment
# * In Engineering and Science&Technology, male has more enrollment
# * Compared to Bachelor, very few student enroll in further studies
# * Province 2 has the lowest literacy of 5 years
# * Province 3 and Gandaki has the highest number of literacy of 5 years
# * Tribhuvan University has the highest number of student enrollment
# * Except Tribuwan University , Female enrollement is more in other Univeristy compared to Male
# * Only Province 3 has enrollment after Master
# * Province 3 has highest number of University and its afflicated colleges
# * Province 5 has the highest number of Sanskrit University and its afflicated colleges
# * Difference in number of male and female teacher is lowest in province 3
