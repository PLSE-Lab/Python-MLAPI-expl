#!/usr/bin/env python
# coding: utf-8

# In this study, student alcohol consumption data is evaluated according to their student class, student school, sex, age and address. In the next parts, how the family features affect alcohol consumption, relationship between alcohol consumption and school success and the other features will be evaluated.  
# 
# > 1. Data Evaluation
# > 2. Age
# > 3. Sex
# > 4. School
# > 5. Address
# > 6. Conclusion

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))


# In[ ]:


mat_data= pd.read_csv("../input/student-mat.csv")
por_data= pd.read_csv("../input/student-por.csv")


# there are several (370)students that belong to both datasets . These students have only failure, paid, absences, G1,G2,G3 grades in mathematics and portuguese classes differences. Firstly we merge them without counting the intersections.

# In[ ]:


student = pd.merge(por_data, mat_data, how='outer', on=["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet","guardian","traveltime","studytime","famsup","activities","higher","romantic","famrel","freetime","goout","Dalc","Walc","health","schoolsup"])
student


# Some nan values occurs. These values have not been taken into account since they will not be examined in this part. It will be examined in later parties.

# In[ ]:


student.info()


# There are 33 features in this data. What they represent is given below.
# 
# **school**   : Student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
# 
# **sex**    : Student's sex (binary: 'F' - female or 'M' - male)
# 
# **age**    : Student's age (numeric: from 15 to 22)
# 
# **address**  : Student's home address type (binary: 'U' - urban or 'R' - rural)
# 
# **famsize**  : Family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
# 
# **Pstatus**  : Parent's cohabitation status (binary: 'T' - living together or 'A' - living apart)
# 
# **Medu** : Mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education, or 4 - higher education)
# 
# **Fedu**  : Father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education, or 4 - higher education)
# 
# **Mjob**  : Mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
# 
# **Fjob**   : Father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
# 
# **reason**   : Reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
# 
# **guardian**    : Student's guardian (nominal: 'mother', 'father' or 'other')
# 
# **traveltime** : Home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
# 
# **studytime** : Weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
# 
# **failures**     : Number of past class failures (numeric: n if 1<=n<3, else 4)
# 
# **schoolsup**  : Extra educational support (binary: yes or no)
# 
# **famsup** : Family educational support (binary: yes or no)
# 
# **paid**   : Extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
# 
# **activities** : Extra-curricular activities (binary: yes or no)
# 
# **nursery**   : Attended nursery school (binary: yes or no)
# 
# **higher**    : Wants to take higher education (binary: yes or no)
# 
# **internet** : Internet access at home (binary: yes or no)
# 
# **romantic** : With a romantic relationship (binary: yes or no)
# 
# **famrel**  : Quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
# 
# **freetime**  : Free time after school (numeric: from 1 - very low to 5 - very high)
# 
# **goout**  : Going out with friends (numeric: from 1 - very low to 5 - very high)
# 
# **Dalc**  : Workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 
# **Walc** : Weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 
# **health**  : Current health status (numeric: from 1 - very bad to 5 - very good)
# 
# **absences** :  Number of school absences (numeric: from 0 to 93)
# 
# **G1**  : First period grade (numeric: from 0 to 20)
# 
# **G2** : Second period grade (numeric: from 0 to 20)
# 
# **G3** : Final grade (numeric: from 0 to 20, output target)
# 

# In[ ]:


student.head(20)


# In[ ]:


student.describe()


# When we look at the descriptive statistic table, there are 674 students aged 15 to 22. Average age is 16.8.
# 
# The average level of education of mother's is 2.5, in other words  5th to 9th grade between secondary education and father's average education level is 2.3. It is slightly lower than the average of mothers and When we look at the median of the educational level of the parents, we see that mothers education level is seconday education and fathers education level is 5th to 9 grade.
# 
# Students' travel time home to school is changing between 1 hour and 4 hours. Average travel time is 1.56 hours.
# 
# Students are studying weekly minimum 1 hour, maximum 4 hour and on the average, 1.93 hours.
# 
# Average quality of family relationships is 3.93, that means students have strong relations with their families on the average.
# 
# Students' average free time after school is 3.18, it is between medium and high.
# 
# Students are going outside 3.17 averagely, it is close to students average free time.
# 
# As for the most important part, students consume alcohol  1.5, in other words very low to low level on work days, 2.28 ,is that low to medim level at weekends on the average.
# 
# Average current health status level is 3.5, they are moderate to good healthy.
# 

# In[ ]:


student.corr()

f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(student.corr(),annot=True,linewidth=0.5,fmt='.3f',ax=ax)
plt.show()


# In the correlation map, we can see that there is a high positive association fathers education level and mothers education level 
# Also, there is a very high positive correlation between the grades of the students and there is moderate positive relation between workdays and weekend alcohol consumption. And we can say that going out  and weekend alcohol consumption have moderate positive correlation. 

# In[ ]:


l=[1,2,3,4,5] #Alcohol consumption level
labels="1-Very Low","2-Low","3-Medium","4-High","5-Very High"


# > **AGE**
# 
# In the first descriptive statistical table, we found that ages ranged from 15 to 22 years. Now let's see what these ages are. 

# In[ ]:


student.age.unique()


# Students ages are 15, 16, 17, 18, 19, 20, 21, 22. Also you can also see age percentage in the chart below.

# In[ ]:


#Age
student.age.unique()
plt.figure(figsize=(10,5))
plt.hist(student.age,bins=7,color="mediumpurple",width=0.8,density=True)
plt.xlabel("Age")
plt.ylabel("Percentage")
plt.show()


# Now, lets examine 15 years old students.

# In[ ]:


age15=student[(student.age==15)]
age15.describe()


# There are 113 fifteen years old students. Their alcohol consumption level is  1.38 level averagely on  workdays and 2.02 level on weekends. 
# 
# 

# Line plot is shown in below according to number of students and alcohol consumption levels.

# In[ ]:


age15_workday=list(map(lambda l: list(age15.Dalc).count(l),l))
age15_weekend=list(map(lambda l: list(age15.Walc).count(l),l))
plt.style.use("bmh")
plt.figure(figsize=(10,5))
plt.plot(labels,age15_workday,color="dodgerblue",linestyle="--",marker="X", markersize=10,label="Workday")
plt.plot(labels,age15_weekend,color="darkmagenta",linestyle="--",marker="X", markersize=10,label="Weekend")
plt.title("Age 15 Student Alcohol Consumption")
plt.ylabel("Number of Students")
plt.legend()


plt.show()


# In the figure, about 80 students are drinking alcohol on working days in the very low level and this number falls to about 55 students on weekends.
# In the low level, the difference between them does not change so much. 
# In the medium level, about 20 students are drinking on weekends and  about 8 students on workdays.
# In the high level, about 11 students consume alcohol on weekends and almost never consume on  workdays.
# And the very high level, there are a few students consume on weekends and almost never consume on workdays.

# **16 year old students **

# In[ ]:


age16=student[(student.age==16)]
age16.describe()


# There are 179 sixteen years old student in this dataset. Thir alcohol consumption level is 1.4 averagely on the workdays and 2.23 on weekends. 
# 
# In the figure below, their alcohol consumption level is shown by their numbers.

# In[ ]:


age16_workday=list(map(lambda l: list(age16.Dalc).count(l),l))
age16_weekend=list(map(lambda l: list(age16.Walc).count(l),l))
plt.style.use("bmh")
plt.figure(figsize=(10,5))
plt.plot(labels,age16_workday,color="sandybrown",linestyle=":",marker="s", markersize=10,label="Workday")
plt.plot(labels,age16_weekend,color="seagreen",linestyle=":",marker="s", markersize=10,label="Weekend")
plt.title("Age 16 Student Alcohol Consumption")
plt.ylabel("Number of Students")
plt.legend()
plt.show()


#  In the figure, about 130 students are drinking alcohol on working days in the very low level and this number falls to about 70 students on weekends.
# In the low level, about 35 students are consuming alcohol on working days  and this number rises to about 40 on weekends . 
# In the medium level, about 23 students are drinking on weekends and  about 10 students on working days.
# In the high level, about 23 students consume alcohol on weekends and almost never consume on working days.
# And the very high level, there are about 10 students consume on weekends and a few students consume on working days.

# **17 years old students **

# In[ ]:


age17=student[(student.age==17)]
age17.describe()


# There are 180 seventeen years old student. They  drink alcohol  1.55 level averagely on the workdays and 2.43 level at the weekends. 
# 
# 
# 
# Line plot is shown in below according to number of students and alcohol consumption levels.

# In[ ]:


age17_workday=list(map(lambda l: list(age17.Dalc).count(l),l))
age17_weekend=list(map(lambda l: list(age17.Walc).count(l),l))
plt.style.use("bmh")
plt.figure(figsize=(10,5))
plt.plot(labels,age17_workday,color="lawngreen",linestyle="dotted",marker="*", markersize=12,label="Workday")
plt.plot(labels,age17_weekend,color="rebeccapurple",linestyle="dotted",marker="*", markersize=12,label="Weekend")
plt.legend()
plt.ylabel("Number of Students")
plt.title("Age 17 Student Alcohol Consumption")
plt.show()


#  In the figure, about 120 students are drinking alcohol on working days in the very low level and this number falls to about 55 students on weekends.(over 50%)
# In the low level, about 38 students are consuming alcohol on working days  and about 50 students consume on weekends.
# In the medium level, about 15 students are drinking on workdays and  this number rises to about 40 on weekends . 
# In the high level, about 20 students consume alcohol on weekends and almost no student consume on workdays.
# And the very high level, there are about 18 students consume on weekends and a few students consume workdays.

# **18 years old students**

# In[ ]:


age18=student[(student.age==18)]
age18.describe()


# There are 145 eighteen years old students. Their alcohol consumption level is 1.55 on the average on the workdays and 2.38 on  weekends. 
# 
# 
# 
# Line plot is shown in below according to number of students and alcohol consumption levels.

# In[ ]:


age18_workday=list(map(lambda l: list(age18.Dalc).count(l),l))
age18_weekend=list(map(lambda l: list(age18.Walc).count(l),l))
plt.style.use("bmh")
plt.figure(figsize=(10,5))
plt.plot(labels,age18_workday,color="darkgoldenrod",linestyle="dotted",marker="P", markersize=10,label="Workday")
plt.plot(labels,age18_weekend,color="mediumblue",linestyle="dotted",marker="P", markersize=10,label="Weekend")
plt.legend()
plt.ylabel("Number of Students")
plt.title("Age 18 Student Alcohol Consumption")
plt.show()


# In the very low level about 100 students are drinking alcohol on working days and this number decreases to about 50 students on weekends.
# In the low level, about 35 students are consuming alcohol on weekends  and about 25 students on workdays.
# In the medium level, about 10 students are drinking on workdays and  this number rises to about 25 on weekends . 
# In the high level, about 25 students consume alcohol on weekends and a few students consume on workdays.
# And the very high level, there are about 10 students consume on weekends and a few students consume workdays.

# **19 years old students**

# In[ ]:


age19=student[(student.age==19)]
age19.describe()


# There are 45 nineteen years old students. Their alcohol consumption level is 1.64 on the average on the workdays and 2.04 on  weekends. 
# 
# 
# 
# Line plot is shown in below according to number of students and alcohol consumption levels.

# In[ ]:


age19_workday=list(map(lambda l: list(age19.Dalc).count(l),l))
age19_weekend=list(map(lambda l: list(age19.Walc).count(l),l))
plt.style.use("bmh")
plt.figure(figsize=(10,5))
plt.plot(labels,age19_workday,color="dodgerblue",marker="H", markersize=10,label="Workday")
plt.plot(labels,age19_weekend,color="darkmagenta",marker="H", markersize=10,label="Weekend")
plt.legend()
plt.ylabel("Number of Students")
plt.title("Age 19 Student Alcohol Consumption")
plt.show()


# In the very low level about 30 students are drinking alcohol on working days and about 20 students on weekends.
# In the low level, about 8 students are consuming alcohol on weekends  and about 5 students on workdays.
# In the medium level, about 6 students are drinking on workdays and  this number rises to about 13 on weekends . 
# In the high level and very high level, difference between them does not change much. 

# **20 years old students**

# In[ ]:


age20=student[(student.age==20)]
age20.describe()


# There are 8 twenty years old students in this dataset and their consumption level is 1.62 on workdays and 2.62 on weekends. 

# In[ ]:


age20_workday=list(map(lambda l: list(age20.Dalc).count(l),l))
age20_weekend=list(map(lambda l: list(age20.Walc).count(l),l))
plt.style.use("bmh")
plt.figure(figsize=(15,5))
plt.plot(labels,age20_workday,color="darkorange",marker="d", markersize=10,label="Workday")
plt.plot(labels,age20_weekend,color="deepskyblue",marker="d", markersize=10,label="Weekend")
plt.legend()
plt.ylabel("Number of Students")
plt.title("Age 20 Student Alcohol Consumption")
plt.show()


# 5 students are consuming alcohol on workdays and 3 students on weekends in very low level. 
# In the low level, 2 students on workdays, 1 student on weekends, 
# In the medium level 1 student on weekends, In the high level 1 student on workdays, 2 students on weekends
# And in the very high level there is a 1 student consume alcohol on weekends. 
# 
# There isn't any student consume alcohol on workdays in the medium and very high level.

# **21 years old student**

# In[ ]:


age21=student[(student.age==21)]
age21.describe()


# There are 3 students at the age of 21. 

# In[ ]:


age21_workday=list(map(lambda l: list(age21.Dalc).count(l),l))
age21_weekend=list(map(lambda l: list(age21.Walc).count(l),l))
plt.figure(figsize=(10,5))
plt.plot(labels,age21_workday,color="chocolate",linestyle=":",linewidth=2,marker="X", markersize=10,label="Workday")
plt.plot(labels,age21_weekend,color="indigo",linestyle=":",linewidth=2,marker="X", markersize=10,label="Weekend")
plt.legend()
plt.style.use("bmh")
plt.ylabel("Number of Students")
plt.title("Age 21 Student Alcohol Consumption")
plt.show()


# There is 1 student consumes alcohol in the very low level and medium level on weekends and on workdays,.
# In the low level there is 1 student on weekends, In the high level there is not any student consume alcohol, And very high level there is 1 student drinks alcohol.  

# ***Looking at the figures, we can say that students of all age groups consume more alcohol on weekends than on working days.***
# 
# ***Alcohol consumption of 15-year-olds students is generally very low, but few students consume alcohol moderate and high  levels on weekends.***
# 
# ***Alcohol consumption of 16-year-old students is usually low, but on weekends, few students consume more alcohol in the medium, high and very high levels.***
# 
# ***Although 17-year-old students have a low level of alcohol consumption on working days, we see that alcohol consumption level is getting higher than other age groups when compared to their alcohol consumption.***
# 
# ***The alcohol consumption of 18-year-old students can is similar to 17-year-old students.***
# 
# ***When we look at the 19-year-old students, we can say that the alcohol level consumed on working days and weekends decreased compared to other age groups.***

# > **SEX**

# In[ ]:


#sex
female=student[student.sex=="F"]
male= student[student.sex=="M"]


# In[ ]:


female.describe()


# There are 396 female students in this dataset and their average alcohol consumption level is 1.27 on working days and 1.93 on weekends.

# In[ ]:


female_workday= list(map(lambda l: list(female.Dalc).count(l),l))
female_weekend= list(map(lambda l: list(female.Walc).count(l),l))

n = 5
fig, ax = plt.subplots(figsize=(10,5))
i = np.arange(n)    
w = 0.4   

plot1= plt.bar(i, female_workday, w, color="g")
plot2= plt.bar(i+w, female_weekend, w, color="r" )

plt.ylabel('Number of Student')
plt.title('Female Student Alcohol Consumption')
plt.xticks(i+w/2, labels)
plt.legend((plot1[0],plot2[0]),("Workday","Weekend"))
plt.tight_layout()
plt.style.use("bmh")
plt.show()


# On the working days, female students alcohol consumtion level is generally very low, however on the weekends they consumes more alcohol medium-high level.

# In[ ]:


male.describe()


# There are 278 male students in this dataset and their average alcohol consumption level is 1.81 on working days and 2.77 on weekends. 

# In[ ]:


male_workday= list(map(lambda l: list(male.Dalc).count(l),l))
male_weekend= list(map(lambda l: list(male.Walc).count(l),l))

n = 5
fig, ax = plt.subplots(figsize=(10,5))
i = np.arange(n)    
w = 0.4   

plot1= plt.bar(i, male_workday, w, color="cadetblue")
plot2= plt.bar(i+w,male_weekend, w, color="b" )

plt.ylabel("Number of Student")
plt.title("Male Student Alcohol Consumption")
plt.xticks(i+w/2, labels)
plt.legend((plot1[0],plot2[0]),("Workday","Weekend"))
plt.tight_layout()
plt.style.use("bmh")
plt.show()


# ***Looking at the figures, when we compare the alcohol consumption level of male and female students, we see that male students consume much more alcohol. And also level of alcohol consumption of male students is much higher than the compared to other specifications on other working days and especially on weekends.***
# 

# **> SCHOOL**

# In[ ]:


#school
student.school.unique()


# There are 2 types of school in this dataset.  GP represents Gabriel Pereira and MS is Mousinho da Silveira

# In[ ]:


GP=student[student.school=="GP"]
MS= student[student.school=="MS"]


# In[ ]:


GP.describe()


# There are 441 students in the Gabriel Pereira School. Their alcohol consumption level is 1.45 on working days and 2.25 on weekends.

# In[ ]:


GP_workday= list(map(lambda l: list(GP.Dalc).count(l),l))

colors="mediumspringgreen","orchid","orangered","darkgoldenrod","aqua"
plt.figure(figsize=(8,8))
plt.pie(GP_workday,colors=colors,autopct='%1.1f%%', startangle=90)
plt.title("GP Students Workday Alcohol Consumptions")
plt.legend(labels)
plt.show()


# 71.4% of the GP students consume alcohol in the very low level, 18.6% low level, 5% medium level and 2.7% high and 2.3% very high levels on the working days.

# In[ ]:


GP_weekend= list(map(lambda l: list(GP.Walc).count(l),l))

plt.figure(figsize=(8,8))
plt.pie(GP_weekend,colors=colors,autopct='%1.1f%%', startangle=90)
plt.title("GP Students Weekend Alcohol Consumptions")
plt.legend(labels)
plt.show()


# However, weekends alcohol consumption level percentage is changing according to workdays. 40% percentage of GP students consume alcohol in the very low level, 21.8% low level, 17.9% medium level, 13.6% high level and 6.8 very high level.

# In[ ]:


MS.describe()


# There are 233 students in the Mousinho da Silveira School. Their average alcohol consumption level is 1.57 on working days and 2.33 on weekends.

# In[ ]:


MS_workday= list(map(lambda l: list(MS.Dalc).count(l),l))
colors2="sandybrown","springgreen","tomato","grey","pink"
plt.figure(figsize=(8,8))
plt.pie(MS_workday,colors=colors2,autopct='%1.1f%%', startangle=90)
plt.title("MS Students Workday Alcohol Consumptions")
plt.legend(labels)
plt.show()


# 66.1% of the MS students consume alcohol in the very low level, 18.5% low level, 10% medium level and 2.6% high and 3% very high levels on the working days. It seems MS students are consuming more alcohol than GP students on the working days.

# In[ ]:


MS_weekend= list(map(lambda l: list(MS.Walc).count(l),l))
plt.figure(figsize=(8,8))
plt.pie(MS_weekend,colors=colors2,autopct='%1.1f%%', startangle=90)
plt.title("MS Students Weekend Alcohol Consumptions")
plt.legend(labels)
plt.show()


# 34.8% percentage of MS students consume alcohol in the very low level, 25.3% low level, 19.7% medium level, 13.3% high level and 6.9 very high level on weekends. Although MS students consume more alcohol than GP students on working days, this rate slightly descrease during the weekend.

# > **ADDRESS**

# In[ ]:


student.address.unique()


# There are two different student types by address. U represents urban, R is rural. 

# In[ ]:


urban=student[student.address=="U"]
rural= student[student.address=="R"]


# In[ ]:


urban.describe()


# There are 470 students living in urban area. Their average alcohol consumption level is 1.47 on working days and 2.26 on weekends.

# In[ ]:


urban_workday=list(map(lambda l: list(urban.Dalc).count(l),l))
urban_weekend=list(map(lambda l: list(urban.Walc).count(l),l))

n = 5
fig, ax = plt.subplots(figsize=(10,5))
i = np.arange(n)   
w = 0.4   

plot1= plt.bar(i, urban_workday, w, color="peachpuff")
plot2= plt.bar(i+w, urban_weekend, w, color="skyblue" )

plt.ylabel('Number of Student')
plt.title('Urban Student Alcohol Consumption')
plt.xticks(i+w/2, labels)
plt.legend((plot1[0],plot2[0]),("Workday","Weekend"))
plt.tight_layout()
plt.grid()
plt.show()


# In[ ]:


rural.describe()


# There are 204 students living in rural area. Their average alcohol consumption level is 1.56 on working days and 2.31 on weekends.

# In[ ]:


rural_workday=list(map(lambda l: list(rural.Dalc).count(l),l))
rural_weekend=list(map(lambda l: list(rural.Walc).count(l),l))

n = 5
fig, ax = plt.subplots(figsize=(10,5))
i = np.arange(n)    
w = 0.4 
p1= plt.bar(i, rural_workday, w, color="lightsalmon")
p2= plt.bar(i+w,rural_weekend, w, color="cornflowerblue" )

plt.ylabel('Number of Student')
plt.title('Rural Student Alcohol Consumption')
plt.xticks(i+w/2, labels)
plt.legend((p1[0],p2[0]),("Workday","Weekend"))
plt.tight_layout()
plt.grid()
plt.show()


# According to the graphs, students living in urban areas consume more alcohol than students living in rural areas.

# > **CONCLUSION**
# 
# When we consider the average alcohol consumption levels of all students, we see that all of the features we examined are very close to the average levels of alcohol consumption. In the df table below, features are shown according to their number and average working day and weekenend alcohol consumption levels(from 1 - very low to 5 - very high).

# In[ ]:


d= {"Feature": ["All Students","Age 15","Age 16","Age 17","Age 18", "Age 19", "Age 20", "Age 21", "Female","Male","GP","MS","Urban","Rural"],
    "Count": [student.shape[0],age15.shape[0],age16.shape[0],age17.shape[0],age18.shape[0],age19.shape[0],age20.shape[0],age21.shape[0],female.shape[0],male.shape[0],GP.shape[0],MS.shape[0],urban.shape[0],rural.shape[0]],
    "Average Working Day Alcohol Consumption": [student.Dalc.mean(),age15.Dalc.mean(),age16.Dalc.mean(),age17.Dalc.mean(),age18.Dalc.mean(),age19.Dalc.mean(),age20.Dalc.mean(),age21.Dalc.mean(),female.Dalc.mean(),male.Dalc.mean(),GP.Dalc.mean(),MS.Dalc.mean(),urban.Dalc.mean(),rural.Dalc.mean()],
    "Average Weekend Alcohol Consumption": [student.Walc.mean(),age15.Walc.mean(),age16.Walc.mean(),age17.Walc.mean(),age18.Walc.mean(),age19.Walc.mean(),age20.Walc.mean(),age21.Walc.mean(),female.Walc.mean(),male.Walc.mean(),GP.Walc.mean(),MS.Walc.mean(),urban.Walc.mean(),rural.Walc.mean()]}

df=pd.DataFrame(d)


# In[ ]:


df

