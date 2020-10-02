#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In this study, student alcohol consumption data is evaluated according to their extra-curricular activities, internet connection, romantic relationship, freetime, go out, health. In the previous part, *(http://www.kaggle.com/gulsahdemiryurek/student-alcohol-consumption-part-1)* , age, sex, school, address was investigated.
# 

# In[ ]:


mat_data= pd.read_csv("../input/student-mat.csv")
por_data= pd.read_csv("../input/student-por.csv")


# In[ ]:


student = pd.merge(por_data, mat_data, how='outer', on=["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet","guardian","traveltime","studytime","famsup","activities","higher","romantic","famrel","freetime","goout","Dalc","Walc","health","schoolsup"])


# In this study, following features will be used. 
# 
# **activities** : Extra-curricular activities (binary: yes or no)
# 
# **internet** : Internet access at home (binary: yes or no)
# 
# **romantic** : With a romantic relationship (binary: yes or no)
# 
# **freetime** : Free time after school (numeric: from 1 - very low to 5 - very high)
# 
# **goout** : Going out with friends (numeric: from 1 - very low to 5 - very high)
# 
# **Dalc** : Workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 
# **Walc** : Weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 
# **health** : Current health status (numeric: from 1 - very bad to 5 - very good)
# 
# 

# In[ ]:


l=[1,2,3,4,5] #Alcohol consumption level
labels="1-Very Low","2-Low","3-Medium","4-High","5-Very High"
colorset="darkorange","chartreuse","seagreen","slateblue","firebrick"


# In[ ]:


def barplot(value1,value2,yLabel,Title,Legend1,Legend2):  #gives 2 barchart
    """
    parameter: value1,value2,ylabel,Title,Legend1,Legend2
    return 2 barchart 
    """
    
    n = 5
    fig, ax = plt.subplots(figsize=(10,5))

    i = np.arange(n)    # the x locations for the groups
    w = 0.4   # the width of the bars: can also be len(x) sequence
    
    plot1= plt.bar(i,value1, w, color="teal")
    plot2= plt.bar(i+w,value2, w, color="darkmagenta" )

    plt.ylabel(yLabel)
    plt.title(Title)
    plt.xticks(i, labels)
    plt.legend((plot1[0],plot2[0]),(Legend1,Legend2))
    plt.tight_layout()
    plt.show()  


# In[ ]:


def piechart(value,colorset,Title,labels):    
    """
    parameter: value, colorset, Title
    return piechart
    """
    plt.figure(figsize=(8,8))
    plt.pie(value,colors=colorset,autopct='%1.1f%%', startangle=90)
    plt.legend(labels)
    plt.title(Title)


# In[ ]:


def linechart(value1,label1,value2,label2,Ylabel,Title):
    """
    parameter:value1,label1,value2,label2,Ylabel,Title
    label 1:label of value1
    label 2: label of value2
    return 2 linechart
    """
    plt.figure(figsize=(10,5))
    plt.plot(labels,value1,color="blue",marker="o", linestyle="dashed", markersize=10,label=label1)
    plt.plot(labels,value2,color="red",marker="o", linestyle="dashed", markersize=10,label=label2)
    plt.ylabel(Ylabel)
    plt.title(Title)
    plt.grid()
    plt.legend()
   


# > **ACTIVITIES**

# In[ ]:


student.activities.unique()


# This feature is divided into extra activity participants ("yes") and non-attendees ("no")

# The following function gives the percentage of level of alcohol consumption consumed on working days or on weekends, depending on whether or not they participate in activities. In the first parameter, the working day (Dalc) or weekend (Walc) and the second parameter are typed in the activities (yes) or not (no).

# In[ ]:


def a(alc,answer):  #alc="Walc" or "Dalc", answer= "yes" or "no"
    if alc=="Dalc":
        y= list(map(lambda l :list(student[(student.activities==answer)].Dalc).count(l),l)) 
        percent=[i/sum(y)*100 for i in y]
        #print("1-Vey Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        #print("1-Very Low:",percent[0],"2-Low:",percent[1],"3-Medium",percent[2] ,"4-High",percent[3], "5-Very High:",percent[4])
        return percent
    elif alc=="Walc":
        y= list(map(lambda l :list(student[(student.activities==answer)].Walc).count(l),l)) 
        percent=[i/sum(y)*100 for i in y]
        #print("1-Very Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        #print("1-Very Low:",percent[0],"2-Low:",percent[1],"3-Medium",percent[2] ,"4-High",percent[3], "5-Very High:",percent[4])
        return percent


# In[ ]:


a("Dalc","yes")


# That means , 70 percent of students participating in activities consume a very low level of alcohol on workdays. 17.8 percent of the low level, 5.84 percent of the middle level, 2.46 percent of the high level 3.7 percent of the very high level.
# If we want to show this in the pie chart:

# In[ ]:


piechart(a("Dalc","yes"),colorset,"Percentage of alcohol consumed by students participating in activities on workdays",labels)


# In[ ]:


piechart(a("Dalc","no"),colorset,"Percentage of alcohol consumed by students not participating in activities on workdays",labels)


# 69 percent of students not participating in activities consume a very low level of alcohol on workdays. 19.2 percent of the low level, 7.4 percent of the middle level, 2.9 percent of the high level 1.4 percent of the very high level.

# If we show the percentage of alcohol consumption on the weekends according to the activities of the students:

# In[ ]:


linechart(a("Walc","yes"),"yes",a("Walc","no"),"no","Percentage","the percentage of alcohol consumption on the weekends according to the activities of the students")


# On the weekends, 
# At the very low level of alcohol consumption, students who participate in activities of 40 percent, 36 percent of students who do not consume alcohol.
# At the low level, about 20% students who answer yes, about 25% of students who give no answer.
# At the medium and high level, rates are close and students  who answer yes by a small differences consumes a high level of alcohol.

# > **INTERNET CONNECTION**

# In[ ]:


student.internet.unique()


# This feature is divided into students with and without internet connection.

# The following function gives the number of level of alcohol consumption consumed on working days or on weekends, depending on their internet connection. In the first parameter, the working day (Dalc) or weekend (Walc) and the second parameter is have internet connection (yes) or not (no).

# In[ ]:


def i(alc,answer): #alc="Dalc" or Walc , answer "yes" or "no"
    if alc=="Dalc":
        y= list(map(lambda l :list(student[(student.internet==answer)].Dalc).count(l),l)) 
        #print("1-Vey Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        return y
    elif alc=="Walc":
        y= list(map(lambda l :list(student[(student.internet==answer)].Walc).count(l),l)) 
        #print("1-Very Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        return y


# In[ ]:


print(i("Dalc","yes"))
print(sum(i("Dalc","yes")))


# That means there are 356 students who has internet connection consumes alcohol at the very low level on workdays, 96 students at the low level, 33 students at the medium level, 14 students at the high level and 15 students at the very high level. 
# Also there are 514 students  with internet connection. We determine to the number of students in the previous study is 674, according to this, students with internet connection is approximately 76%.

# In[ ]:


print(i("Dalc","no"))
print(sum(i("Dalc","no")))


# There are 113 students who has not internet connection consumes alcohol at the very low level on workdays, 29 students at the low level, 12 students at the medium level, 4 students at the high level and 2 students at the very high level. 
# Also there are 160 students  without internet connection. 

# Lets show them with bar chart

# In[ ]:


barplot(i("Dalc","yes"),i("Dalc","no"),"Number of Students","Alcohol Consumption Levels on Working days According to Internet Connection","yes","no")


# Since the number of students is not close to each other, let's examine each one according to the amount of alcohol consumed on working days or weekend.

# In[ ]:


barplot(i("Dalc","yes"),i("Walc","yes"),"Number of Students","Students who have internet connection Alcohol consumption levels on workdays or on weekends ","working day","weekend")


# In[ ]:


barplot(i("Dalc","no"),i("Walc","no"),"Number of Students","Students who have not internet connection Alcohol consumption levels on workdays or on weekends ","working day","weekend")


# Looking at this graphs we can say that there is not much of a relationship with the consumption of alcohol with internet connection

# > **ROMANTIC RELATIONSHIP**

# The following function gives the percentage of students of level of alcohol consumption consumed on working days or on weekends, depending on their romantic relationships. In the first parameter, the working day (Dalc) or weekend (Walc) and the second parameter is have a relationship (yes) or not (no).

# In[ ]:


def r(alc,answer):   #alc="Dalc" or Walc , answer "yes" or "no"
    if alc=="Dalc":
        y= list(map(lambda l :list(student[(student.romantic==answer)].Dalc).count(l),l)) 
        print("1-Very Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        print("sum of students:", sum(y))
        percent=[i/sum(y) for i in y]
        return percent
    elif alc=="Walc":
        y= list(map(lambda l :list(student[(student.romantic==answer)].Walc).count(l),l))
        print("1-Very Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        print("sum of students:",sum(y))
        percent=[i/sum(y) for i in y]
        return percent


# In[ ]:


print(r("Dalc","yes"))
print(r("Dalc","no"))


# 252 students have a romantic relationship, 422 students have not. 

# In[ ]:


barplot(r("Dalc","yes"),r("Dalc","no"),"Percentage","Percentage of alcohol consumed by students on working days according to have relationship" ,"yes","no")


# As shown in the graph, the alcohol intake rate on working days at the very low and low level  so close to each other. But at the medium level, the ratio of those who give "yes" answer is more than "no"  answer. At the high and very high level "yes" respondents have a little more alcohol consumption rate.

# In[ ]:


barplot(r("Walc","yes"),r("Walc","no"),"Percentage","Percentage of alcohol consumed by students on weekends according to have relationship" ,"yes","no")


# On weekends there is no so much difference between students who have relationship or students who have no relationship. Just in the high level students who have no romantic relationship consume slightly more alcohol.

# **> FREE TIME**

# In[ ]:


student.freetime.describe()


# When we examine students' free time after school, we can see that it is  at the  3.18 level on the average. That means students have free time at the medium level. Also we can show that by histogram

# In[ ]:


plt.hist(student.freetime,bins=5)


# The following function gives the percentage of students of level of alcohol consumption consumed on working days or on weekends, depending on their freetime. In the first parameter, the working day (Dalc) or weekend (Walc) and the second parameter is freetime level (1-very low, 5-very high).

# In[ ]:


def f(alc,l1): #alc ="yes" or "no" , l1= 1,2,3,4,5 (level of freetime)(1- very low-5 very high)
    if alc=="Dalc":
        y= list(map(lambda l :list(student[(student.freetime==l1)].Dalc).count(l),l)) 
        percent=[i/sum(y)*100 for i in y]
        #print("1-Vey Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        #print("1-Very Low:",percent[0],"2-Low:",percent[1],"3-Medium",percent[2] ,"4-High",percent[3], "5-Very High:",percent[4])
        return percent
    elif alc=="Walc":
        y= list(map(lambda l :list(student[(student.freetime==l1)].Walc).count(l),l)) 
        percent=[i/sum(y)*100 for i in y]
        #print("1-Very Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        #print("1-Very Low:",percent[0],"2-Low:",percent[1],"3-Medium",percent[2] ,"4-High",percent[3], "5-Very High:",percent[4])
        return percent


# The following table shows the percentage of alcohol consumption on working days according to the students' free time levels.

# In[ ]:


DalcFreetime= {"Workday\Free Time": ["1-Very Low","2-Low","3-Medium","4-High","5-Very High"],
    "1-Very Low": f("Dalc",1),"2-Low": f("Dalc",2), "3-Medium": f("Dalc",3),"4-High":f("Dalc",4),"5-Very High": f("Dalc",5)}
dfDalcFreetime=pd.DataFrame(DalcFreetime)
dfDalcFreetime


# In[ ]:



dfDalcFreetime.plot(kind='bar',x="Workday\Free Time" ,grid=True, title="percentage of alcohol consumption on working days according to the students' free time levels",figsize=(15,5),
        sharex=True, sharey=False, legend=True)
plt.ylabel("Percentage")
plt.xlabel("Alcohol Consumption Level")
plt.show()


# The chart shows the percentage of alcohol consumption on working days according to the students' free time levels. According 
# to that, 
# 
# If we examine the students with **the least(1)** of their free time after school (blue one on the chart) :
# 
#  Very Low Alcohol Consumption: 72.34%, Low: 12.76%, Medium: 6.38%,  High: 4.25%,  Very High: 4.25%
#  
# ** Low level(2) free time(yellow):**
#   
#  Very Low Alcohol Consumption: 	75.67%, Low: 14.41%, Medium:7.2%, High:2.7%, Very High: 0%
# 
# **Medium level(3) free time(green):**
# 
#  Very Low Alcohol Consumption: 75%, Low: 18.36%, Medium:2.73%, High: 1.95%, Very High: 1.95%
#  
#  **High level(4) free time(red):**
# 
#  Very Low Alcohol Consumption: 61.17%, Low: 22.87%, Medium:11.17%, High: 2.66%, Very High: 2.12%
#  
#   **Very High level(5) free time(purple):**
# 
#  Very Low Alcohol Consumption: 61.11%, Low: 18.05%, Medium:8.33%, High: 4.16%, Very High: 8.33%
#  
#  With all this, we can say that the group with the highest amount of alcohol consumption is the group with the most free time (8.33%) on working days. At the high level alcohol consumption, students who have the least and highest level of free time are close to each other(4.25%-4.17%) but highest free time level is higher. At the medium level alcohol consumption students who have the high and very level of free time are higher than the others (11.17%-8.33%). Although percentages in other alcohol consumption levels may change, students consume alchohol same way at the very low and low level on the working days
# 
# Now lets check weekends.
# 
# 

# The following table shows the percentage of alcohol consumption on weekends according to the students' free time levels.

# In[ ]:


WalcFreetime= {"Weekend\Free Time": ["1-Very Low","2-Low","3-Medium","4-High","5-Very High"],
    "1-Very Low": f("Walc",1),"2-Low": f("Walc",2), "3-Medium": f("Walc",3),"4-High":f("Walc",4),"5-Very High": f("Walc",5)}
dfWalcFreetime=pd.DataFrame(WalcFreetime)
dfWalcFreetime


# In[ ]:


dfWalcFreetime.plot(kind='bar',x="Weekend\Free Time" ,grid=True, title="percentage of alcohol consumption on weekends according to the students' free time levels",figsize=(15,5), legend=True)
plt.xlabel("Alcohol Consumption Level")
plt.ylabel("Percentage")
plt.show()


# The chart shows the percentage of alcohol consumption on weekends according to the students' free time levels. According 
# to that, 
# 
#  **Very Low free time(1)(blue one on the chart)** :
# 
#  Very Low Alcohol Consumption: 53.19%, Low: 17.02%, Medium: 14.89%,  High: 2.12%,  Very High: 12.76%
#  
# ** Low level(2) free time(yellow):**
#   
#  Very Low Alcohol Consumption: 34.23%, Low: 36.03%, Medium:12.61%, High: 12.61%, Very High: 4.5%
# 
# **Medium level(3) free time(green):**
# 
#  Very Low Alcohol Consumption: 41.4%, Low: 23.82%, Medium:18.36%, High: 10.93%, Very High:5.47%
#  
#  **High level(4) free time(red):**
# 
#  Very Low Alcohol Consumption: 34.04%, Low: 18.08%, Medium:21.8%, High: 21.27%, Very High: 4.78%
#  
#   **Very High level(5) free time(purple):**
# 
#  Very Low Alcohol Consumption: 33.33%, Low: 16.67%, Medium:22.2%, High: 11.11%, Very High: 16.67%
#  
#  With all this, we can say that the group with the highest amount of alcohol consumption is the group with the most free time (16.67%) on weekends, however 12.76% students who have the least free time are consuming alcohol at the very high level, surprisingly . 21.27% students who have high freetime are drinking alcohol at the high level alcohol consumption.
#  
#  

# **As a result, we can say that alcohol consumption on working days increases in parallel to the free time after school. However 12 percent of students who say that after school is the least of their free time, consume  very high level of alcohol on weekends.**

# 

# > **GO OUT** 

# In[ ]:


student.goout.describe()


#  Going out friends level is  3.17 on the average. That means students are going out with their friends  at the medium level. Also we can show that by histogram

# In[ ]:


plt.hist(student.goout,bins=5)


# The following function gives the percentage of students of level of alcohol consumption consumed on working days or on weekends, depending on going out with their levels. In the first parameter, the working day (Dalc) or weekend (Walc) and the second parameter is go out level (1-very low, 5-very high).

# In[ ]:


def g(alc,l1):   #alc="Dalc" or "Walc", l1= 1,2,3,4,5 (level of going out)(1- very low-5 very high)
    if alc=="Dalc":
        y= list(map(lambda l :list(student[(student.goout==l1)].Dalc).count(l),l)) 
        percent=[i/sum(y)*100 for i in y]
        #print("1-Vey Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        #print("1-Very Low:",percent[0],"2-Low:",percent[1],"3-Medium",percent[2] ,"4-High",percent[3], "5-Very High:",percent[4])
        return percent
    elif alc=="Walc":
        y= list(map(lambda l :list(student[(student.goout==l1)].Walc).count(l),l)) 
        percent=[i/sum(y)*100 for i in y]
        #print("1-Very Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        #print("1-Very Low:",percent[0],"2-Low:",percent[1],"3-Medium",percent[2] ,"4-High",percent[3], "5-Very High:",percent[4])
        return percent


# The following table shows the percentage of alcohol consumption on working days according to the students' go out levels.

# In[ ]:


DalcGoout= {"Workday\Go out": ["1-Very Low","2-Low","3-Medium","4-High","5-Very High"],
    "1-Very Low": g("Dalc",1),"2-Low": g("Dalc",2), "3-Medium": g("Dalc",3),"4-High":g("Dalc",4),"5-Very High": g("Dalc",5)}
DalcGoout=pd.DataFrame(DalcGoout)
DalcGoout


# In[ ]:


DalcGoout.plot(kind='bar',x="Workday\Go out" ,grid=True, title="percentage of alcohol consumption on working days according to the students' go out levels",figsize=(15,5), legend=True)
plt.xlabel("Alcohol Consumption Level")
plt.ylabel("Percentage")
plt.show()


#  **Very Low going out (1)(blue one on the chart)** :
# 
#  Very Low Alcohol Consumption: 87.75%, Low: 8.16%, Medium: 4.08%,  High: 0%,  Very High: 0%
#  
# ** Low level(2) free time(yellow):**
#   
#  Very Low Alcohol Consumption: 79.47%, Low: 14.56%, Medium:3.31%, High: 1.98%, Very High: 0.66%
# 
# **Medium level(3) free time(green):**
# 
#  Very Low Alcohol Consumption: 72.55%, Low:18.6%, Medium:5.58%, High: 1.39%, Very High:1.86%
#  
#  **High level(4) free time(red):**
# 
#  Very Low Alcohol Consumption: 59.73%, Low: 25.5%, Medium:9.39%, High: 4.02%, Very High: 1.34%
#  
#   **Very High level(5) free time(purple):**
# 
#  Very Low Alcohol Consumption: 55.45%, Low: 19%, Medium:11%, High: 5.45%, Very High: 9%
#  
# As seen in the chart, we see that the group that consumes the most alcohol during the working days is the group of mostly  with their friends. At the same time, we see that the group who are going out very low, don't consume alcohol at the high and very high levels.
# 
# If we examine the weekends:
# 
# The following table shows the percentage of alcohol consumption on weekends  according to the students' go out levels.

# In[ ]:


WalcGoout= {"Weekend\Go out": ["1-Very Low","2-Low","3-Medium","4-High","5-Very High"],
    "1-Very Low": g("Walc",1),"2-Low": g("Walc",2), "3-Medium": g("Walc",3),"4-High":g("Walc",4),"5-Very High": g("Walc",5)}
WalcGoout=pd.DataFrame(WalcGoout)
WalcGoout


# In[ ]:


WalcGoout.plot(kind='bar',x="Weekend\Go out" ,grid=True, title="percentage of alcohol consumption on weekends according to the students' go out levels",figsize=(15,5), legend=True)
plt.xlabel("Alcohol Consumption Level")
plt.ylabel("Percentage")
plt.show()


# That is very interesting graph and perhaps the very important one when the compared the other features both this part and the part 1.  
# 
# **Very Low going out (1)(blue one on the chart)** :
# 
#  Very Low Alcohol Consumption: 69.38%, Low: 14.28%, Medium: 8.16%,  High: 6.12%,  Very High: 2.04%
#  
# ** Low level(2) free time(yellow):**
#   
#  Very Low Alcohol Consumption: 52.31%, Low: 27.15%, Medium:13.9%, High: 4.63%, Very High: 1.98%
# 
# **Medium level(3) free time(green):**
# 
#  Very Low Alcohol Consumption: 36.74%, Low:29.3%, Medium:22.8%, High: 8.83%, Very High:2.32%
#  
#  **High level(4) free time(red):**
# 
#  Very Low Alcohol Consumption: 30.20%, Low:17.44 %, Medium:20.8%, **High: 24.16%,** Very High: 7.37%
#  
#   **Very High level(5) free time(purple):**
# 
#  Very Low Alcohol Consumption: **18.18%**, Low: 16.36%, Medium:18.18%, High: 23.63%, **Very High: 23.63%**
#  
# 
# 
# We see that the alcohol consumption levels of the group who are going out with their friends very low on weekdays and weekends are very similar. And always students' alcohol consumption levels have always been very low or low on both weekends and working days.  however, we see that most of the group that has been going out with their friends at the very high level consume very high level of alcohol. **Friends really affects your alcohol consumption !!**
# 

# **> HEALTH**

# In[ ]:


student.health.describe()


# In[ ]:


plt.hist(student.health,bins=5)


# Students' health levels seem to be on average, but when we look at the graph we see that they are very healthy.

# The following function gives the percentage of students of level of alcohol consumption consumed on working days or on weekends, depending on their health level. In the first parameter, the working day (Dalc) or weekend (Walc) and the second parameter is health level (1-very low, 5-very high).
# 

# In[ ]:


def h(alc,l1):   #alc="Dalc" or "Walc", l1= 1,2,3,4,5 (level of health)(1- very low-5 very high)
    if alc=="Dalc":
        y= list(map(lambda l :list(student[(student.health==l1)].Dalc).count(l),l)) 
        percent=[i/sum(y)*100 for i in y]
        #print("1-Vey Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        #print("1-Very Low:",percent[0],"2-Low:",percent[1],"3-Medium",percent[2] ,"4-High",percent[3], "5-Very High:",percent[4])
        return percent
    elif alc=="Walc":
        y= list(map(lambda l :list(student[(student.health==l1)].Walc).count(l),l)) 
        percent=[i/sum(y)*100 for i in y]
        #print("1-Very Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        #print("1-Very Low:",percent[0],"2-Low:",percent[1],"3-Medium",percent[2] ,"4-High",percent[3], "5-Very High:",percent[4])
        return percent


# The following table shows the percentage of alcohol consumption on working days according to the students' health levels.

# In[ ]:


DalcHealth= {"Workday\Health": ["1-Very Low","2-Low","3-Medium","4-High","5-Very High"],
    "1-Very Low": h("Dalc",1),"2-Low": h("Dalc",2), "3-Medium": h("Dalc",3),"4-High":h("Dalc",4),"5-Very High": h("Dalc",5)}
DalcHealth=pd.DataFrame(DalcHealth)
DalcHealth


# The following  chart shows the percentage of alcohol consumption on working days according to the students' health levels.

# In[ ]:


DalcHealth.plot(kind='bar',x="Workday\Health" ,grid=True, title="percentage of alcohol consumption on working days according to the students' health levels",figsize=(15,5), legend=True)
plt.xlabel("Alcohol Consumption Level")
plt.ylabel("Percentage")
plt.show()


# The following table shows the percentage of alcohol consumption on working days according to the students' health levels.

# In[ ]:


WalcHealth= {"Weekend\Health": ["1-Very Low","2-Low","3-Medium","4-High","5-Very High"],
    "1-Very Low": h("Walc",1),"2-Low": h("Walc",2), "3-Medium": h("Walc",3),"4-High":h("Walc",4),"5-Very High": h("Walc",5)}
WalcHealth=pd.DataFrame(WalcHealth)
WalcHealth


# The following chart shows the percentage of alcohol consumption on working days according to the students' health levels.

# In[ ]:


WalcHealth.plot(kind='bar',x="Weekend\Health" ,grid=True, title="percentage of alcohol consumption on working days according to the students' health levels",figsize=(15,5), legend=True)
plt.xlabel("Alcohol Consumption Level")
plt.ylabel("Percentage")
plt.show()


# As seen in the graphs, we see that the health status is not very much related to the level of alcohol consumption. We can say that it is similar to other features alcohol consumption levels.
