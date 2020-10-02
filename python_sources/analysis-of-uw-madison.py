#!/usr/bin/env python
# coding: utf-8

# # Analysis of UW Madison Courses and Grades 2006-2017
# ### Courses, grades, instructors, and subjects at UW Madison since 2006.
# 

# In[ ]:


import pandas as pd
import numpy as np
import plotly.offline as py
from plotly import graph_objs as go
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import plotly.tools as tls


# ## Teaching 

# In[ ]:


teach=pd.read_csv('../input/teachings.csv')
print(teach.shape)
teach.head()


# In[ ]:


print(len(teach.instructor_id.unique()))
print(len(teach.section_uuid.unique()))


#  - There are 315211 sections (all are unique)
#  - But there are a total of 18737 teachers responsible for teaching all of these sections

# ## Subjects

# In[ ]:


sub=pd.read_csv('../input/subjects.csv')
print(sub.shape)
sub.head()


# In[ ]:


len(sub.code.unique())


# - There are overall 200 subjects available for the students to choose from

# ## Courses

# In[ ]:


course=pd.read_csv('../input/courses.csv')
print(course.shape)
course.head()


# - The number column perhaps signifies that the courses belong in the same category
# 

# ## Course Offerings Every Semester

# In[ ]:


offer=pd.read_csv('../input/course_offerings.csv')
print(offer.shape)
offer.head()


# In[ ]:


print(len(offer.term_code.unique()))
print(len(offer.course_uuid.unique()))
print(len(offer.uuid.unique()))


# - the term_code refers the the unique code to identify the [semester] - [year] currently ongoing in the college. We will have to figure out what each code stands for
# - all the courses seem to be listed in the course offerings at one point or another
# 

# ## Subject Membership

# In[ ]:


sub_mem=pd.read_csv('../input/subject_memberships.csv')
print(sub_mem.shape)
sub_mem.head()


# - this dataset will be helpful later on when merging datasets to gain insights

# ## Instructors

# In[ ]:


inst=pd.read_csv('../input/instructors.csv')
print(inst.shape)
inst.head()


# - This dataset gives us the names of the teachers corresponding to their ids

# ## Sections

# In[ ]:


section=pd.read_csv('../input/sections.csv')
print(section.shape)
section.head()


# In[ ]:


print(len(section.uuid.unique()))
print(len(section.course_offering_uuid.unique()))
print(section.section_type.unique())


# ## Schedules

# In[ ]:


schedule=pd.read_csv('../input/schedules.csv')
print(schedule.shape)
schedule.head()


# - Let's add a column which will tell us how many days of classes there are in a week in a particular schedule
# - Let's also add another column which will calculate the duration of a class by subtracting the 'end_time' and 'start_time' columns for every row

# In[ ]:


schedule['num_days']=schedule.select_dtypes(include=['bool']).sum(axis=1)
schedule['duration']=schedule['end_time']-schedule['start_time']


# In[ ]:


schedule.head()


# ## Rooms

# In[ ]:


room=pd.read_csv('../input/rooms.csv')
print(room.shape)
room.head()


#  - We can get the samntic meanings of the facility code by the accessing the link provided on [kaggle] 
#  [kaggle]: http://www.map.wisc.edu/buildings/

# In[ ]:


print(len(room.facility_code.unique()))


# ## Grades

# In[ ]:


grade=pd.read_csv('../input/grade_distributions.csv')
print(grade.shape)
grade.head()


# In[ ]:


print(len(grade.course_offering_uuid.unique()))


# In[ ]:


grade['avg_gpa']=(((grade['a_count']*4)+(grade['ab_count']*3.5)+(grade['b_count']*3)+(grade['bc_count']*2.5)+
(grade['c_count']*2)+(grade['d_count']*1))/(grade['a_count']+grade['ab_count']+grade['b_count']+grade['bc_count']+
grade['c_count']+grade['d_count']))


# In[ ]:


grade.head()


# ## Making a SuperSet
# ### And also gaining a few insights on the way

# In[ ]:


teach_v1=teach.merge(inst,left_on='instructor_id',right_on='id')
print(teach_v1.head())
teach_v1.shape


# In[ ]:


temp_series=teach_v1.name.value_counts()[0:10]
teachers=np.array(temp_series.index)
num_class=np.array(temp_series)
num_class


# In[ ]:


trace = go.Bar(
    y=num_class,
    x=teachers,
    orientation = 'v'
)

layout = dict(
    title='Number of Classes Taught by Instructors 2006-2017',
    yaxis= dict(title='Number of Classes'),
    xaxis= dict(title='Name of Instructor')
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Superheroes")


# In[ ]:


teach_v1.groupby('instructor_id').name.count().mean()


# - The average number of sections taught by a professor is 16 whereas the the top 10 professors we saw above have significantly higher number of sections under their belt. This indicates that there have been a number of new hires in the teaching staff

# In[ ]:


grade_v1=grade.merge(offer,left_on='course_offering_uuid',right_on='uuid')[['uuid','course_uuid','term_code','name','avg_gpa']]
grade_v1.head()


# In[ ]:


print((sorted(list(grade_v1.term_code.unique()))))
sem={1072:'Fall 2006',1074:'Spring 2007',1082:'Fall 2007',1084:'Spring 2008',1092:'Fall 2008',
    1094:'Spring 2009',1102:'Fall 2009',1104:'Spring 2010',1112:'Fall 2010',1114:'Spring 2011',
    1122:'Fall 2011',1132:'Fall 2012',1134:'Spring 2013',1142:'Fall 2013',
    1144:'Spring 2014',1152:'Fall 2014',1154:'Spring 2015',1162:'Fall 2015',1164:'Spring 2016',
    1172:'Fall 2016',1174:'Spring 2017',1182:'Fall 2017'}
s=[sem[i] for i  in grade_v1['term_code']]
grade_v1['sem']=pd.Series(s)


# In[ ]:


grade_v1.head()


# - We used https://registrar.wisc.edu/grade-reports/ to decode the term_code columns. 'Spring 2012' Seems to be missing from our dataset

# In[ ]:


section_v1=section.merge(schedule,left_on='schedule_uuid',right_on='uuid')
section_v1=section_v1[['uuid_x','course_offering_uuid','section_type','number','room_uuid','schedule_uuid','num_days','duration']]
print(section_v1.shape)
section_v1.head()


# In[ ]:


section_v1.num_days.value_counts()


# - It's surprising to find out that there are 37 courses which have classes daily
# - Now Let's try to look at the schedule base on the type of section the course requires (Lecture, Discussion, Field, etc)

# In[ ]:


temp_series=section_v1.section_type.value_counts()


# In[ ]:


labels = (np.array(temp_series.index))
sizes = np.array(temp_series)

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Section-Type Distribution',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Comic") 


# In[ ]:


temp=section_v1.groupby('section_type').num_days.mean().sort_values(ascending=False)
temp


# In[ ]:


trace = go.Bar(
    y=temp,
    x=temp.index,
    orientation = 'v',
    marker=dict(color=['purple','blue','crimson','green','brown','orange'])
)

layout = dict(
    title='Number of Classes Taught by Instructors 2006-2017',
    yaxis= dict(title='Number of Classes'),
    xaxis= dict(title='Name of Instructor')
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Superheroes")


# - Even though 26.1% (82,396) of all classes are under IND class, they don't require a single day of formal classes. They might just be digital courses

# In[ ]:


section_v1[section_v1['num_days']==7].head()


# - Almost all courses which have classes daily are of Field type and are a total of 720 minutes

# In[ ]:


temp=section_v1.duration.value_counts()
temp.index.sort_values()
section_v1['hours']=section_v1['duration']//60
section_v1.head()


# In[ ]:


section_v2=section_v1.merge(teach_v1,left_on='uuid_x',right_on='section_uuid')
section_v2=section_v2[['course_offering_uuid', 'section_type', 'number', 'room_uuid',
       'schedule_uuid', 'num_days', 'duration', 'hours', 'instructor_id',
       'section_uuid', 'name']]
section_v2.head()


# In[ ]:


grade_v2=grade_v1.merge(sub_mem,right_on='course_offering_uuid',left_on='uuid')
grade_v2=grade_v2[['course_offering_uuid','course_uuid','name','avg_gpa','sem','subject_code']]
grade_v2.head()


# In[ ]:



sub=sub[sub['code'].apply(lambda x: str(x).isdigit())]
sub['code']=sub['code'].astype(int)


# In[ ]:


grade_v3=grade_v2.merge(sub,left_on='subject_code',right_on='code')
grade_v3.head()


# In[ ]:


print(grade_v3.shape)
print(section_v2.shape)


# In[ ]:


print(len(section_v2.section_uuid.unique()))
print(len(grade_v3.course_offering_uuid.unique()))


# In[ ]:


grade_v4=grade_v3.merge(section_v2,on='course_offering_uuid')
grade_v4.drop_duplicates(inplace=True)
grade_v4.shape


# In[ ]:


grade_v4.columns


# In[ ]:


x=grade_v4.groupby(['abbreviation','name','sem'],as_index=False).avg_gpa.mean()
x.groupby(['abbreviation','sem']).avg_gpa.mean()


# In[ ]:


grade_v4['year']=[int(i[-4:]) for i in grade_v4['sem']]


# In[ ]:


y=grade_v4.groupby('instructor_id')
years=dict((y.year.max()-y.year.min()+1).sort_values())

print(len(years))
print(len(grade_v4.instructor_id.unique()))
print(len(grade_v4.name.unique()))

grade_v4['years_teacher']=[years[i] for i in grade_v4['instructor_id']]


# In[ ]:


grade_v4['taught']=pd.cut(grade_v4['years_teacher'], 4,labels=['0-3 years','3-6 years','6-9 years','9-12 years'],retbins=False)


# In[ ]:


temp=grade_v4.groupby('taught').avg_gpa.mean()
temp.index


# In[ ]:


trace = go.Bar(
    x=temp,
    y=temp.index,
    orientation = 'h',
    
)

layout = dict(
    title='Experience of Teacher vs Class Performance',
    yaxis= dict(title='Years Taught',tickangle=45,tickfont=dict(color='crimson')),
    xaxis= dict(title='Average GPA'),
    width=1000,
    height=400
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
#fig.update_yaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))
py.iplot(fig, filename="Superheroes")


# - We can see how Teacher's Performance improves steadily as his/her teaching experience increases

# In[ ]:


temp=grade_v4.groupby('hours').avg_gpa.mean()


# In[ ]:


trace = go.Bar(
    y=temp,
    x=temp.index,
    orientation = 'v',

    
)

layout = dict(
    title='Class Hours vs Class Performance',
    xaxis= dict(title='Class Hours',tickangle=45,tickfont=dict(color='crimson')),
    yaxis= dict(title='Average GPA'),
    width=1000,
    height=400
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
#fig.update_yaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))
py.iplot(fig, filename="Superheroes")


# In[ ]:


grade_v4.groupby('section_type').avg_gpa.mean().sort_values()


# In[ ]:


grade_v5=grade_v4.merge(room,left_on='room_uuid',right_on='uuid')


# In[ ]:


grade_v5[grade_v5['facility_code'] == 'ONLINE'].avg_gpa.mean()


# In[ ]:


grade_v5[grade_v5['facility_code'] != 'ONLINE'].avg_gpa.mean()


# In[ ]:


trace = go.Bar(
    y=[3.641070033853206,3.3717104926401062],
    x=['Online Courses','In Class Courses'],
    orientation = 'v',
    marker=dict(color=['Green','Blue'])
    
)

layout = dict(
    title='Online Courses vs In-Class Courses',
    yaxis= dict(title='Average GPA',tickangle=45,tickfont=dict(color='crimson')),
    xaxis= dict(title='Course-Type'),
    width=1000,
    height=400
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
#fig.update_yaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))
py.iplot(fig, filename="Superheroes")


# In[ ]:


grade_v5[grade_v5['facility_code'] != 'ONLINE'].years_teacher.value_counts().sort_values()


# In[ ]:


lowest=grade_v5.groupby('abbreviation').avg_gpa.mean().sort_values()[0:11]


# In[ ]:


trace = go.Bar(
    y=lowest,
    x=lowest.index,
    orientation = 'v'
    
)

layout = dict(
    title='Top 10 Worst Performing Courses',
    yaxis= dict(title='Average GPA',tickangle=45,tickfont=dict(color='crimson')),
    xaxis= dict(title='Course-Type'),
    width=1000,
    height=400
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
#fig.update_yaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))
py.iplot(fig, filename="Superheroes")


# In[ ]:


highest=grade_v5.groupby('abbreviation').avg_gpa.mean().sort_values(ascending=False)[0:11]


# In[ ]:


trace = go.Bar(
    y=highest,
    x=highest.index,
    orientation = 'v'
    
)

layout = dict(
    title='Top 10 Highest Scoring Courses',
    yaxis= dict(title='Average GPA',tickangle=45,tickfont=dict(color='crimson')),
    xaxis= dict(title='Course-Type'),
    width=1000,
    height=400
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
#fig.update_yaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))
py.iplot(fig, filename="Superheroes")


# In[ ]:


h=grade_v5[grade_v5['abbreviation'].isin(highest.index)]
l=grade_v5[grade_v5['abbreviation'].isin(lowest.index)]


# In[ ]:


h.section_type.value_counts().sort_values() 


# In[ ]:


l.section_type.value_counts().sort_values()


#  - Courses involving Lab or Discussion seem to be the most least scoring whereas as Courses involving Field seem to be the most scoring courses

# In[ ]:


t_top=grade_v5.groupby('instructor_id').avg_gpa.mean().sort_values(ascending=False)[0:16]
t_low=grade_v5.groupby('instructor_id').avg_gpa.mean().sort_values()[0:16]


# In[ ]:


tt=grade_v5[grade_v5['instructor_id'].isin(t_top.index)]
tl=grade_v5[grade_v5['instructor_id'].isin(t_low.index)]


# In[ ]:


tt1=tt.groupby(['name','year'],as_index=False).num_days.count()
print(tt1.num_days.mean())
tt1.groupby('name').num_days.mean()


# In[ ]:


tl1=tl.groupby(['name','year'],as_index=False).num_days.count()
print(tl1.num_days.mean())
tl1.groupby('name').num_days.mean()


# - Lowest Performing Teachers were usually burdened with more classes, almost twice in comparison to the Highest Performing Teachers

# In[ ]:


trace = go.Bar(
    y=[2.0833333333333335,4.764705882352941],
    x=['Best Performing Teachers','Worst Performing Teachers'],
    orientation = 'v',
    marker=dict(color=['Green','Red'])
    
)

layout = dict(
    title='Impact of Tight Scedule on Teachers',
    yaxis= dict(title='Number of Classes Per Sem',tickangle=45,tickfont=dict(color='crimson')),
    xaxis= dict(title='Teacher'),
    width=1000,
    height=400
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
#fig.update_yaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))
py.iplot(fig, filename="Superheroes")


# In[ ]:


tl.name_y.value_counts().sort_values()


# In[ ]:


tt.name_y.value_counts().sort_values()


# - Surprisingly medical courses are among the courses taught bt the best performing teachers in addition to extra-curicculars like Music 
# - Where as the Botany and Zoology are the courses taught by some of the worst performing teachers

# # Conclusion
# 
# I'd like to conclude my analysis by condensing my insights into a few bullet points
# 
# - The average number of sections taught by a professor is 16 whereas the the top professors have taught exponentially more number of classes(over 500). This means **there has been an increase in the number of new hires in the Faculty department which explains why there is such a difference in the maximum and mean number of classes taught.**
# 
# - **[Tidbit]** Around 37 courses which were held daily (Mon-Sun). They were mostly Field Types
# 
# - *Discussion(21.2%)* , *Lecture(28.3%)* and *IND(26.1%)* made up the majority of the classes. Even though *IND* made up a 1/4 of the courses, they weren't scheduled regularly
# 
# - **Teacher's performance improved steadily as they became more experienced in the teaching field**. Average GPA increased from *3.3* to *3.5*
# 
# - **Classes which involved lower number of hours per week resulted in lower performance** but as the number of hours increased, the Average GPA rose from *3.37* to *3.96*
# 
# - **[Tidbit]** The Performance peaked at 10 Hours but then took a sudden hit again, dropping to 3.4 before coming back up again 
# 
# - **Online Courses(3.6) were easier to score in** when compared to In - Class Courses(3.3).
# 
# - Most of the **Worst Performing Courses (Maths,Accounting,Physics,Economics) involved higher degree of number crunching** where as the **Top Performing Courses (Radiology,Molecule Biology,Neurology) were more medical centric courses in addition to extra-curriculars such as Arts,Music etc.**
# 
# - Courses involving Lab or Discussion seem to be the most least scoring whereas as Courses involving Field seem to be the most scoring courses
# 
# - **Lowest Performing Teachers were burdened with more classes per sem, almost twice** in comparison to the Highest Performing Teachers
# 

# In[ ]:




