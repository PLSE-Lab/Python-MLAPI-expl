#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sts


# What tools Kaggler use?. It is considered that Better the tools better the result. In this analysis i will take you through analysis on that, what tools Kagglers like most.

# In[ ]:


path2019 = "/kaggle/input/kaggle-survey-2019/"
mulcho19 = path2019 + "multiple_choice_responses.csv"
survey2019 = pd.read_csv(mulcho19,
                         skiprows=0,
                         header=1
                        )


# # General analysis 

# 
# In 2019 survey by kaggle, attracted **19717** participants. 

# In[ ]:


age19 = survey2019.iloc[:,1]
gender19 = survey2019.iloc[:,2]
country19 = survey2019.iloc[:,4]
education19 = survey2019.iloc[:,5]
jobrole19 = survey2019.iloc[:,6]
df19 = pd.DataFrame({
    "Age" : age19,
    "Gender" : gender19,
    "Country" : country19,
    "Education" : education19,
    "Jobrole" : jobrole19
})


# In[ ]:


g = sns.catplot(y="Age",
                 data=df19.sort_values("Age"), kind="count",
                height=4, aspect=2,
                edgecolor=sns.color_palette("dark", 3));
plt.title(" Age group wise participants frequency in Survey");
plt.figtext(0.5, 1.1, "Figure 1 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From Figure 1, It is observed that, participants are in majority from age group of **25-29**. But best part is that, Kaggle is popular to yongest and people started to join Kaggle at the age of 18. In order to amazed further Kaggle have some participants whose age is more than 70 years. Let us analyze, what is genderwise participants of kagglers in Survey. **Figure 2** display frequency of participants from different gender.

# In[ ]:


g = sns.catplot(y="Gender",
                 data=df19.sort_values("Age"), kind="count",
                height=4, aspect=2,
                edgecolor=sns.color_palette("dark", 3));
plt.title("Figure : Genderwise participants frequency");
plt.figtext(0.5, 1.1, "Figure 2 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# No need to say that the maximum participants are male. But many question arise from **Figure 2**, what is the reason of very less female participants. Are Kaggle having very less frequency of Female on Kaggle or they have not participated in survey due to one or other reason. In **Figure 3**, I am going to discuss the Country wise participants. I am just depicting participants from Countries where participation is high.

# In[ ]:


temp = df19.Country.value_counts().sort_values(ascending=False)[0:10]
top10Country = temp.reset_index()
top10Country.columns = ["Country","NumberOfParticipants"]
top10Country


# In[ ]:


g = sns.catplot(y="NumberOfParticipants", x= "Country",
                 data=top10Country, kind="bar",
                height=5, aspect=2,
                edgecolor=sns.color_palette("dark", 3));
plt.title("Participants gender count");
g.set_xticklabels(rotation=90);
plt.figtext(0.5, 1.1, "Figure 3 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# According to **Figure 3**, highest number of participants are from **India**, which is followed by **United States of America**

# In[ ]:


g = sns.catplot(y="Education",
                 data=df19, kind="count",
                height=5, aspect=2,
                edgecolor=sns.color_palette("dark", 3));
plt.title("Education wise participants frequency");
g.set_xticklabels(rotation=90);
plt.figtext(0.5, 1.1, "Figure 4 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# Maximum Kaggle survey participants are Master degree holder as **Figure 4** indicates. The frequency of **Bachelor's Degree Holder** are following **Master Degree Holder**. In following analysis i am going to show that, what tools kaggles use for their analysis. And how age group, gender, country, and education is going to affect the tool choise for them. Let me start with IDE (Which of the following integrated development environments Kaggels use)

# # IDE 2019

# In[ ]:


# Which of the following integrated development environments (Done)
jupyter19 = survey2019.iloc[:,56].notnull().astype('int')
rstudio19 = survey2019.iloc[:,57].notnull().astype('int')
pycharm19 = survey2019.iloc[:,58].notnull().astype('int')
atom19 = survey2019.iloc[:,59].notnull().astype('int')
matlab19 = survey2019.iloc[:,60].notnull().astype('int')
visualstudio19 = survey2019.iloc[:,61].notnull().astype('int')
spyder19 = survey2019.iloc[:,62].notnull().astype('int')
vimemacs19 = survey2019.iloc[:,63].notnull().astype('int')
notpadplusplus19 = survey2019.iloc[:,64].notnull().astype('int')
sublime19 = survey2019.iloc[:,65].notnull().astype('int')

idedf19 = pd.DataFrame({
    "Jupyter" : jupyter19,
    "R_Studio": rstudio19,
    "Pycharm": pycharm19,
    "Atom": atom19,
    "Matlab": matlab19,
    "Visual_Studio": visualstudio19,
    "Spyder" : spyder19,
    "Vim_Emac": vimemacs19,
    "Notpad++": notpadplusplus19,
    "Sublime" : sublime19
})
ideused19 = idedf19.sum(axis=0).sort_values(ascending=False)
ideused19 = ideused19.reset_index() 
ideused19.columns = ["IDE","Number of Users"]
sns.catplot(y="IDE", x="Number of Users",data=ideused19, 
            kind='bar', height=6, aspect=1.5);
plt.figtext(0.5, 1.1, "Figure 5 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# It is very easy to conclude that, the majority of participants use **Jupter Notebook** or **JupyterLab**.  Frequency of  users of **Visual Studio**, **R Studio** and **PyCharm** is following the frequency of **Jupter Notebook** or **JupyterLab** users. But many questions will start attacikng on our brain. Important questions are as follows :
# 
# - Is frequency of different IDE users depend on their Age group ?
# - Is frequency of different IDE users depends on their gender ?
# - Is frequency of different IDE users depends on their Country?
# - Is frequency of different IDE users depends on their Education?

# # Relationship between age group and frequency of different IDE users

# In[ ]:


combineIDE19 = pd.concat([df19, idedf19], axis = 1)
agewiseIDE2019 = combineIDE19.iloc[:,[0,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Age",as_index=False).sum()
meltedIDE19 = agewiseIDE2019.melt(id_vars="Age",var_name="IDE",value_name="Users")
sns.catplot(y="Age", x="Users",data=meltedIDE19, 
            kind='bar', height=9, aspect=1.5,hue="IDE");
plt.title("Frequency of different IDE users by their age group");
plt.figtext(0.5, 1.1, "Figure 6 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 6** it is vivid that, in every age group **Jupter Notebook** or **JupyterLab** is more popular than any other IDE. But we are left with a question that, **Is frequency of different IDE users depend on their Age group ?**. If it is independent then **Jupter Notebook** or **JupyterLab** popularity level is same in each group.

# In[ ]:


agewiseIDE2019


# # Is frequency of different IDE users depend on their Age group ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# - $H_0$ : Frequency of different IDE users is independent of their age group
# - $H_1$ : Frequency of different IDE users is depending on their age group
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(agewiseIDE2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different age group is having different likedness for different IDE. We can say that **Jupter Notebook** or **JupyterLab**  is not having same popularity in each age group.

# # Relationship between Gender and frequency of different IDE users

# In[ ]:


genderwiseIDE2019 = combineIDE19.iloc[:,[1,5,6,7,8,9,10,11,12,13,14]].groupby("Gender",as_index=False).sum()
genderwiseIDE2019 = genderwiseIDE2019[(genderwiseIDE2019.Gender =="Male")| (genderwiseIDE2019.Gender =="Female")]
genmeltIDE19 = genderwiseIDE2019.melt(id_vars="Gender",var_name="IDE",value_name="Users")

sns.catplot(y="Gender", x="Users",data=genmeltIDE19,
            kind='bar', height=9, aspect=1.5,hue="IDE");
plt.title("Frequency of users of different IDE by their Gender")
plt.figtext(0.5, 1.1, "Figure 7 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 7** it is known that, in every gender group **Jupter Notebook** or **JupyterLab** is more popular than any other IDE. But we are left with a question that, **Is frequency of different IDE users depend on their Gender ?**. If it is independent, then **Jupter Notebook** or **JupyterLab** popularity level is same in each gender group.

# In[ ]:


genderwiseIDE2019


# # Is frequency of different IDE users depend on their Gender?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# - $H_0$ : Frequency of different IDE users is independent of their gender
# - $H_1$ : Frequency of different IDE users is depending on their gender
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(genderwiseIDE2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different gender is having different likedness for different IDE. We can say that **Jupter Notebook** or **JupyterLab**  is not having same popularity in each age group. **But from Figure 7 it can be told easily that Jupyter Notebook and JupyterLab is very much popular in Male than Female.**

# # Relationship between Country of residance and frequency of different IDE users

# In[ ]:


CountrywiseJupyter2019 = combineIDE19.iloc[:,[2,5,6,7,8,9,10,11,12,13,14]].groupby("Country",as_index=False).sum()
CountrywiseJupyter2019 = CountrywiseJupyter2019[CountrywiseJupyter2019.Country.isin(top10Country.Country.to_list())]
CountrymeltedIDE19 = CountrywiseJupyter2019.melt(id_vars="Country",var_name="IDE",value_name="Users")
CountrymeltedIDE19.head()
sns.catplot(y="Country", x="Users",data=CountrymeltedIDE19, 
            kind='bar', height=9, aspect=1.5,hue="IDE");
plt.title("Frequency of users of different IDE by their Country")
plt.figtext(0.5, 1.1, "Figure 8 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 8** it is known that, in every Country  **Jupter Notebook** or **JupyterLab** is more popular than any other IDE. But we are left with a question that, **Is frequency of different IDE users depend on their Country ?**. If it is independent, then **Jupter Notebook** or **JupyterLab** popularity level is same in each gender group.
# But Figure 8 is concluding that **Jupter Notebook** or **JupyterLab**  is most popular in Indian Kagglers the any other IDE 

# In[ ]:


CountrywiseJupyter2019


# # Is frequency of different IDE users depend on their Country of Residance?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# - $H_0$ : Frequency of different IDE users is independent of their Country
# - $H_1$ : Frequency of different IDE users is depending on their Country
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(CountrywiseJupyter2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that resident from different Country is having different likedness for different IDE. We can say that **Jupter Notebook** or **JupyterLab**  is not having same popularity in each age group. **But from Figure 8 it can be told easily that Jupyter Notebook and JupyterLab is very much popular in India than Any other country.**

# # Relationship between Education of Participants and frequency of different IDE users

# In[ ]:


EducationwiseIDE2019 = combineIDE19.iloc[:,[3,5,6,7,8,9,10,11,12,13,14]].groupby("Education",as_index=False).sum()

EducationmeltedIDE19 = EducationwiseIDE2019.melt(id_vars="Education",var_name="IDE",value_name="Users")
EducationmeltedIDE19.head()
sns.catplot(y="Education", x="Users",data=EducationmeltedIDE19, 
            kind='bar', height=9, aspect=1.5,hue="IDE");
plt.title("Frequency of users of different IDE by their Education")
plt.figtext(0.5, 1.1, "Figure 9 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 9** it is vivid that, in every education group **Jupter Notebook** or **JupyterLab** is more popular than any other IDE. But we are left with a question that, **Is frequency of different IDE users depend on their Education?**. If it is independent then **Jupter Notebook** or **JupyterLab** popularity level is same in each group.
# 

# In[ ]:


EducationwiseIDE2019


# # Is frequency of different IDE users depend on their Education group ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# - $H_0$ : Frequency of different IDE users is independent of their Education
# - $H_1$ : Frequency of different IDE users is depending on their Education
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(EducationwiseIDE2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different education level is having different likedness for different IDE. We can say that **Jupter Notebook** or **JupyterLab**  is not having same level of popularity in each age group.

# # Relationship between Job Role of Participants and frequency of different IDE users

# In[ ]:


JobrolewiseIDE2019 = combineIDE19.iloc[:,[4,5,6,7,8,9,10,11,12,13,14]].groupby("Jobrole",as_index=False).sum()

JobrolemeltedIDE19 = JobrolewiseIDE2019.melt(id_vars="Jobrole",var_name="IDE",value_name="Users")
JobrolemeltedIDE19.head()
sns.catplot(y="Jobrole", x="Users",data=JobrolemeltedIDE19, 
            kind='bar', height=9, aspect=1.5,hue="IDE");
plt.title("Frequency of users of different IDE by their JobRole")
plt.figtext(0.5, 1.1, "Figure 10 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 10** it is vivid that, apart from **statisticians**, in every job role  group **Jupter Notebook** or **JupyterLab** is more popular than any other IDE. But we are left with a question that, **Is frequency of different IDE users depend on their Job Role group ?**. **Statisticians** like more **R Studio**  than any other IDE. The main Reason for this might be that, in statistical analysis till R is most popular language due to its package range for statistics.

# In[ ]:


JobrolewiseIDE2019


# # Is frequency of different IDE users depend on their Job Role ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# - $H_0$ : Frequency of different IDE users is independent of their Job Role
# - $H_1$ : Frequency of different IDE users is depending on their Job Role
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(JobrolewiseIDE2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different Job Role is having different likedness for different IDE. We can say that **Jupter Notebook** or **JupyterLab**  is not having same popularity in each age group.  **Jupter Notebook** or **JupyterLab** is most popular among **Data Scientists** but **R Studio** is most popular among **Statisticians**.

# # Which programming language Kaggler Likes?

# In[ ]:


# What programming languages do you use on a regular basis (Done)
python19 = survey2019.iloc[:,82].notnull().astype('int')
r19 = survey2019.iloc[:,83].notnull().astype('int')
sql19 = survey2019.iloc[:,84].notnull().astype('int')
c19 = survey2019.iloc[:,85].notnull().astype('int')
cplusplus19 = survey2019.iloc[:,86].notnull().astype('int')
java19 = survey2019.iloc[:,87].notnull().astype('int')
javascript19 = survey2019.iloc[:,88].notnull().astype('int')
typescript19 = survey2019.iloc[:,89].notnull().astype('int')
bash19 = survey2019.iloc[:,90].notnull().astype('int')
matlab19 = survey2019.iloc[:,91].notnull().astype('int')

languagedf19 = pd.DataFrame({
    "Python" : python19,
    "R": r19,
    "Sql": sql19,
    "C": c19,
    "C++": cplusplus19,
    "Java": java19,
    "Javascript" : javascript19,
    "Typescript": typescript19,
    "Bash": bash19,
    "Matlab" : matlab19
})
proused19 = languagedf19.sum(axis=0).sort_values(ascending=False)
proused19 = proused19.reset_index() 
proused19.columns = ["Programming Language","Number of Users"]
sns.catplot(y="Programming Language", x="Number of Users",data=proused19, 
            kind='bar', height=6, aspect=1.5);
plt.figtext(0.5, 1.1, "Figure 11 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# In[ ]:


combineLanguage19 = pd.concat([df19, languagedf19], axis = 1)


# In[ ]:


combineLanguage19


# It is very easy to conclude that, the majority of participants use **Python** .  Frequency of  users of **SQL**, **R** and **Java** is following the frequency of **Python** users. But many questions will start attacikng on our brain. Important questions are as follows :
# 
# - Is frequency of different Programming language users depends on their Age group ?
# - Is frequency of different Programming language users depends on their gender ?
# - Is frequency of different Programming language users depends on their Country?
# - Is frequency of different Programming language users depends on their Education?

# # Relationship between Age group and frequency of different Programming Language users

# In[ ]:


agewiseLan2019 = combineLanguage19.iloc[:,[0,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Age",as_index=False).sum()
meltedLan19 = agewiseLan2019.melt(id_vars="Age",var_name="Programming Language",value_name="Users")
sns.catplot(y="Age", x="Users",data=meltedLan19, 
            kind='bar', height=9, aspect=1.5,hue="Programming Language");
plt.title("Frequency of different Programming Language users by their age group");
plt.figtext(0.5, 1.1, "Figure 12 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 12** it is vivid that, in every age group **Python** is more popular than any other Programming Language. But we are left with a question that, **Is frequency of different Programming Languag users depend on their Age group ?**. If it is independent then **Python** popularity level is same in each group.

# In[ ]:


agewiseLan2019


# # Is frequency of different Programming Language users depend on their Age group ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# - $H_0$ : Frequency of different Programming Language users is independent of their age group
# - $H_1$ : Frequency of different Programming Language users is depending on their age group
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(agewiseLan2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different age group is having different likedness for different Programming Language. We can say that **Python**  is not having same popularity in each age group.

# # Relationship between Genders and frequency of different Programming Language users

# In[ ]:


genderLan2019 = combineLanguage19.iloc[:,[1,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Gender",as_index=False).sum()
genderLan2019 = genderLan2019[(genderLan2019.Gender =="Male")| (genderLan2019.Gender =="Female")]
meltedLan19 = genderLan2019.melt(id_vars="Gender",var_name="Programming Language",value_name="Users")
sns.catplot(y="Gender", x="Users",data=meltedLan19, 
            kind='bar', height=9, aspect=1.5,hue="Programming Language");
plt.title("Frequency of different Programming Language users by their Gender");
plt.figtext(0.5, 1.1, "Figure 12 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 12** it is vivid that, in every age group **Python** is more popular than any other Programming Language. But we are left with a question that, **Is frequency of different Programming Languag users depend on their Age group ?**. If it is independent then **Python** popularity level is same in each group.****

# In[ ]:


genderLan2019


# # Is frequency of different Programming Language users depend on their Gender ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# - $H_0$ : Frequency of different Programming language users is independent of their Gender
# - $H_1$ : Frequency of different Programming language users is depending on their Gender
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(genderLan2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different gender group is having different likedness for different Programming language. We can say that **Python**  is not having same popularity in each Gender group.

# # Relationship between Country and frequency of different Programming Language users

# In[ ]:


countryLan2019 = combineLanguage19.iloc[:,[2,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Country",as_index=False).sum()
countryLan2019 = countryLan2019[countryLan2019.Country.isin(top10Country.Country.to_list())]
meltedLan19 = countryLan2019.melt(id_vars="Country",var_name="Programming Language",value_name="Users")
sns.catplot(y="Country", x="Users",data=meltedLan19, 
            kind='bar', height=9, aspect=1.5,hue="Programming Language");
plt.title("Frequency of different Programming Language users by their Country");
plt.figtext(0.5, 1.1, "Figure 12 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 12** it is vivid that, in every age group **Python** is more popular than any other Programming Language. But we are left with a question that, **Is frequency of different Programming Languag users depend on their Age group ?**. If it is independent then **Python** popularity level is same in each group.****

# In[ ]:


countryLan2019


# # Is frequency of different Programming Language users depend on their Country of residance?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# - $H_0$ : Frequency of different programming language users is independent of their Country.
# - $H_1$ : Frequency of different programming language users is depending on their Country.
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(countryLan2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different Country participants is having different likedness for different Programming Language. We can say that **Python**  is not having same popularity in each age group.

# # Relationship between Education and frequency of different Programming Language users

# In[ ]:


eduLan2019 = combineLanguage19.iloc[:,[3,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Education",as_index=False).sum()
meltedLan19 = eduLan2019.melt(id_vars="Education",
                                 var_name="Programming Language",
                                 value_name="Users")
sns.catplot(y="Education", x="Users",data=meltedLan19, 
            kind='bar', height=9, aspect=1.5,hue="Programming Language");
plt.title("Frequency of different Programming Language users by their Education");
plt.figtext(0.5, 1.1, "Figure 12 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 12** it is vivid that, in every age group **Python** is more popular than any other Programming Language. But we are left with a question that, **Is frequency of different Programming Languag users depend on their Education group ?**. If it is independent then **Python** popularity level is same in each group.****

# In[ ]:


eduLan2019


# # Is frequency of different Programming Language users depend on their Education ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# - $H_0$ : Frequency of different Programming Language users is independent of their Education.
# - $H_1$ : Frequency of different Programming Language users is depending on their Education.
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(eduLan2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different education group is having different likedness for different Programming Language. We can say that **Python** is not having same popularity in each education group.

# # Relationship between Job Role and frequency of different Programming Language users

# In[ ]:


jobRoleLan2019 = combineLanguage19.iloc[:,[4,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Jobrole",as_index=False).sum()
meltedLan19 = jobRoleLan2019.melt(id_vars="Jobrole",var_name="Programming Language",value_name="Users")
sns.catplot(y="Jobrole", x="Users",data=meltedLan19, 
            kind='bar', height=9, aspect=1.5,hue="Programming Language");
plt.title("Frequency of different Programming Language users by their Jobrole");
plt.figtext(0.5, 1.1, "Figure 12 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# In **Figure 12**, **R** programmong language is most popular in **Statistician**. **SQL** is most popular in Database engineer. In rest of group  **Python** is more popular than any other Programming Language. But we are left with a question that, **Is frequency of different Programming Languag users depend on their Jobrole ?**.

# In[ ]:


jobRoleLan2019


# # Is frequency of different Programming language users depend on their JobRole ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# - $H_0$ : Frequency of different Programming Language users is independent of their job role.
# - $H_1$ : Frequency of different Programming Language users is depending on their job role.
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(jobRoleLan2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different jobrole group is having different likedness for different Programming Language.

# # Which cloud computing platforms Kaggler Use Most?

# In[ ]:


# Which of the following cloud computing platforms do you use on a regular basis (Done)

googleCloudPlatform19 = survey2019.iloc[:,168].notnull().astype('int')
amazonWebServices19 = survey2019.iloc[:,169].notnull().astype('int')
microsoftAzure19 = survey2019.iloc[:,170].notnull().astype('int')
ibmCloud19 = survey2019.iloc[:,171].notnull().astype('int')
alibabaCloud19 = survey2019.iloc[:,172].notnull().astype('int')
salesforceCloud19 = survey2019.iloc[:,173].notnull().astype('int')
oracleCloud19 = survey2019.iloc[:,174].notnull().astype('int')
sapCloud19 = survey2019.iloc[:,175].notnull().astype('int')
vmwareCloud19 = survey2019.iloc[:,176].notnull().astype('int')
redHatCloud19 = survey2019.iloc[:,177].notnull().astype('int')

cloudPlatform19 = pd.DataFrame({
    "GoogleCloudPlatform" : googleCloudPlatform19,
    "AmazonWebServices": amazonWebServices19,
    "MicrosoftAzure": microsoftAzure19,
    "IBMCloud": ibmCloud19,
    "AlibabaCloud": alibabaCloud19,
    "SalesforceCloud": salesforceCloud19,
    "OracleCloud" : oracleCloud19,
    "SAPCloud": sapCloud19,
    "VMwareCloud": vmwareCloud19,
    "RedHatCloud" : redHatCloud19
})

combinecloud19 = pd.concat([df19, cloudPlatform19], axis = 1)
cloudused19 = cloudPlatform19.sum(axis=0).sort_values(ascending=False)
cloudused19 = cloudused19.reset_index() 
cloudused19.columns = ["Cloud Tools","Number of Users"]
sns.catplot(y="Cloud Tools", x="Number of Users",data=cloudused19, 
            kind='bar', height=6, aspect=1.5);
plt.figtext(0.5, 1.1, "Figure 11 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# Figure 11 correctly depicts that Amazon Web Services and Google cloud services are most popular among Kagglers.

# # Relationship between Age group and frequency of different cloud computing platforms

# In[ ]:


ageCloud2019 = combinecloud19.iloc[:,[0,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Age",as_index=False).sum()
meltedCloud19 = ageCloud2019.melt(id_vars="Age",var_name="Cloud Tools",value_name="Users")
sns.catplot(y="Age", x="Users",data=meltedCloud19, 
            kind='bar', height=9, aspect=1.5,hue="Cloud Tools");
plt.title("Frequency of different Cloud Tools Platform users by their age group");
plt.figtext(0.5, 1.1, "Figure 12 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 12** it is vivid that, in  age group 18-21 **Google Cloud Platform** more popular than any other Cloud Platform. But in other age group Amazon Cloud Servises are more popular. But we are left with a question that, **Is frequency of different Cloud Platform Users depend on their Age group ?

# In[ ]:


ageCloud2019


# # Is frequency of different Cloud Platforms users depend on their Age group ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# - $H_0$ : Frequency of different Cloud platform users is independent of their age group
# - $H_1$ : Frequency of different Cloud platform users is depending on their age group
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(ageCloud2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different age group is having different likedness for different Cloud Platform. 

# # Relationship between Gender and frequency of different Cloud Platform users

# In[ ]:


genderCloud2019 = combinecloud19.iloc[:,[1,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Gender",as_index=False).sum()

genderCloud2019 = genderCloud2019[(genderCloud2019.Gender =="Male")| (genderCloud2019.Gender =="Female")]

melted19 = genderCloud2019.melt(id_vars="Gender",var_name="Cloud Tools",value_name="Users")
sns.catplot(y="Gender", x="Users",data=melted19, 
            kind='bar', height=9, aspect=1.5,hue="Cloud Tools");
plt.title("Frequency of different Cloud Platform users by their Gender");
plt.figtext(0.5, 1.1, "Figure 13 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 13** it is vivid that, in every gender group Amazon Web Services Platform is Popular

# In[ ]:


genderCloud2019


# # Is frequency of different Cloud Platforms users depend on their Gender ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# - $H_0$ : Frequency of different Cloud Platforms users is independent of their Gender
# - $H_1$ : Frequency of different Cloud Platforms users is depending on their Gender
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(genderCloud2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different gender group is having different likedness for different Cloud Platform. 
# 

# # Relationship between Country of residance and frequency of different Cloud Platform users

# In[ ]:


countryCloud2019 = combinecloud19.iloc[:,[2,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Country",as_index=False).sum()
melted19 = countryCloud2019.melt(id_vars="Country",var_name="Cloud Tools",value_name="Users")
sns.catplot(y="Country", x="Users",data=melted19, 
            kind='bar', height=9, aspect=1.5,hue="Cloud Tools");
plt.title("Frequency of different  Cloud platform users by their Country");
plt.figtext(0.5, 1.1, "Figure 14 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# In[ ]:


countryCloud2019


# # Is frequency of different Cloud platform users depend on their Country ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# - $H_0$ : Frequency of different Cloud platform users is independent of their country
# - $H_1$ : Frequency of different Cloud platform users is depending on their country
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(countryCloud2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different country residents is having different likedness for different Cloud Platform.

# # Relationship between Education and frequency of different Cloud platform users

# In[ ]:


eduCloud2019 = combinecloud19.iloc[:,[3,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Education",as_index=False).sum()
melted19 = eduCloud2019.melt(id_vars="Education",var_name="Cloud Tools",value_name="Users")
sns.catplot(y="Education", x="Users",data=melted19, 
            kind='bar', height=9, aspect=1.5,hue="Cloud Tools");
plt.title("Frequency of different Cloud platform users by their Education");
plt.figtext(0.5, 1.1, "Figure 15 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 15** it is vivid that, in every education group Amazon Web Services is more popular.

# In[ ]:


eduCloud2019


# # Is frequency of different Cloud platform users depend on their Education ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# - $H_0$ : Frequency of different Cloud platform users is independent of their Education
# - $H_1$ : Frequency of different Cloud platform users is depending on their Education
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(eduCloud2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different education group is having different likedness for different Cloud platforms.
# 

# # Relationship between Jobrole and frequency of different Cloud platform users

# In[ ]:


jobrole2019 = combinecloud19.iloc[:,[4,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Jobrole",as_index=False).sum()
melted19 = jobrole2019.melt(id_vars="Jobrole",var_name="Cloud Tools",value_name="Users")
sns.catplot(y="Jobrole", x="Users",data=melted19, 
            kind='bar', height=9, aspect=1.5,hue="Cloud Tools");
plt.title("Frequency of different Cloud platform users by their Jobrole");
plt.figtext(0.5, 1.1, "Figure 16 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 16** it is vivid that, in every Job Role group Amazon web services is most popular

# In[ ]:


jobrole2019


# ### Most important for Students and Not Employed No likedness for Cloud Platforms

# #                   big data / analytics products 

# In[ ]:


# Which specificff big data / analytics products do you use on a regular basis? (Done)

googleBigQuery19 = survey2019.iloc[:,194].notnull().astype('int')
awsRedshift19 = survey2019.iloc[:,195].notnull().astype('int')
databricks19 = survey2019.iloc[:,196].notnull().astype('int')
awsElasticMapReduce19 = survey2019.iloc[:,197].notnull().astype('int')
teradata19 = survey2019.iloc[:,198].notnull().astype('int')
microsoftAnalysisServices19 = survey2019.iloc[:,199].notnull().astype('int')
googleCloudDataflow19 = survey2019.iloc[:,200].notnull().astype('int')
awsAthena19 = survey2019.iloc[:,201].notnull().astype('int')
awsKinesis19 = survey2019.iloc[:,202].notnull().astype('int')
googleCloudPubSub19 = survey2019.iloc[:,203].notnull().astype('int')

dfn19 = pd.DataFrame({
    "googleBigQuery" : googleBigQuery19,
    "awsRedshift": awsRedshift19,
    "databricks": databricks19,
    "awsElasticMapReduce": awsElasticMapReduce19,
    "teradata": teradata19,
    "microsoftAnalysisServices": microsoftAnalysisServices19,
    "googleCloudDataflow" : googleCloudDataflow19,
    "awsAthena": awsAthena19,
    "awsKinesis": awsKinesis19,
    "googleCloudPubSub" : googleCloudPubSub19
})

combine19 = pd.concat([df19, dfn19], axis = 1)

dfdn = dfn19.sum(axis=0).sort_values(ascending=False)
dfdn = dfdn.reset_index() 
dfdn.columns = ["BigDataTools","Number of Users"]
sns.catplot(y="BigDataTools", x="Number of Users",data=dfdn, 
            kind='bar', height=6, aspect=1.5);
plt.figtext(0.5, 1.1, "Figure 17 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# Google Big Query is most popular among  big data / analytics products

# # Relationship between Age group and frequency of different Big Data Tools

# In[ ]:


age2019 = combine19.iloc[:,[0,5,6,
                            7,8,9,10,11,
                            12,13,14]].groupby("Age",as_index=False).sum()

melted19 = age2019.melt(id_vars="Age",var_name="BigDataTools",value_name="Users")
sns.catplot(y="Age", x="Users",data=melted19, 
            kind='bar', height=9, aspect=1.5,hue="BigDataTools");
plt.title("Frequency of different BigDataTools Platform users by their age group");
plt.figtext(0.5, 1.1, "Figure 18 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 18** it is vivid that, in every age group(Not in 70+) **Google Big Query**  is more popular than any other Big Data Tools. But in 70+ age group AWS red Shift is popular.

# In[ ]:


age2019


# # Is frequency of different Big Data Tools depend on their Age group ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# $H_0$ : Frequency of different  Big Data Tools users is independent of their age group
# $H_1$ : Frequency of different  Big Data Tools users is depending on their age group
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(age2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we can reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different age group like Google Big Query Most

# # Relationship between Gender and frequency of different  Big Data Tools

# In[ ]:


gender2019 = combine19.iloc[:,[1,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Gender",as_index=False).sum()

gender2019 = gender2019[(gender2019.Gender =="Male")| (genderCloud2019.Gender =="Female")]

melted19 = gender2019.melt(id_vars="Gender",var_name="BigDataTools",value_name="Users")
sns.catplot(y="Gender", x="Users",data=melted19, 
            kind='bar', height=9, aspect=1.5,hue="BigDataTools");
plt.title("Frequency of different Cloud Platform users by their Gender");
plt.figtext(0.5, 1.1, "Figure 19 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# Google big query is popular among Male and Female.

# In[ ]:


gender2019


# # Is frequency of different Big Data Platforms users depend on their Gender ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# $H_0$ : Frequency of different  Big Data Tools users is independent of their Gender
# $H_1$ : Frequency of different  Big Data Tools users is depending on their Gender
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


###################################################################################################

resData = sts.chi2_contingency(gender2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different gender group is having different likedness for different Big Data tools.

# # Relationship between Country of residance and frequency of different  Big Data Tools

# In[ ]:


country2019 = combine19.iloc[:,[2,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Country",as_index=False).sum()
country2019 = country2019[country2019.Country.isin(top10Country.Country.to_list())]
melted19 = country2019.melt(id_vars="Country",var_name="BigDataTools",value_name="Users")
sns.catplot(y="Country", x="Users",data=melted19, 
            kind='bar', height=9, aspect=1.5,hue="BigDataTools");
plt.title("Frequency of different  Cloud platform users by their Country");
plt.figtext(0.5, 1.1, "Figure 20 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 20** it is vivid that, in United States of America AWS Redshift is More Popular than . In Canada DataBricks is Most Popular. In China Google Cloud DataFlow is most Popular. But in India and Other Country Amazon web services is most popular.

# In[ ]:


country2019


# # Is frequency of different  Big Data Tools users depend on their Country ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# $H_0$ : Frequency of different  Big Data Tools users is independent of their country
# $H_1$ : Frequency of different  Big Data Tools users is depending on their country
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(country2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different Country group is having different likedness for different Big Data Tools. It is also very clear from Figure 20.

# # Relationship between Education and frequency of different  Big Data Tools users

# In[ ]:


edu2019 = combine19.iloc[:,[3,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Education",as_index=False).sum()
melted19 = edu2019.melt(id_vars="Education",var_name="BigDataTools",value_name="Users")
sns.catplot(y="Education", x="Users",data=melted19, 
            kind='bar', height=9, aspect=1.5,hue="BigDataTools");
plt.title("Frequency of different  Big Data Tools users by their Education");
plt.figtext(0.5, 1.1, "Figure 21 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 21** it is vivid that, in every education  group **Google Big Query** is most popular. 

# In[ ]:


edu2019


# # Is frequency of different  Big Data Tools users depend on their Education ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# $H_0$ : Frequency of different  Big Data Tools users is independent of their Education
# $H_1$ : Frequency of different  Big Data Tools users is depending on their Education
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(edu2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that **Google Big Query** is equal popular in every education group.

# # Relationship between Jobrole and frequency of different  Big Data Tools users

# In[ ]:


jobrole2019 = combine19.iloc[:,[4,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Jobrole",as_index=False).sum()
melted19 = jobrole2019.melt(id_vars="Jobrole",var_name="BigDataTools",value_name="Users")
sns.catplot(y="Jobrole", x="Users",data=melted19, 
            kind='bar', height=9, aspect=1.5,hue="BigDataTools");
plt.title("Frequency of different Cloud platform users by their Jobrole");
plt.figtext(0.5, 1.1, "Figure 22 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 22** it is vivid that, in every Job Role group **Google Big Query is more popular** 

# # Which Database

# In[ ]:


# Which Database (Done)

mySQL19 = survey2019.iloc[:,233].notnull().astype('int')
postgresSQL19 = survey2019.iloc[:,234].notnull().astype('int')
sqlite19 = survey2019.iloc[:,235].notnull().astype('int')
microsoftSQLServer19 = survey2019.iloc[:,236].notnull().astype('int')
oracleDatabase19 = survey2019.iloc[:,237].notnull().astype('int')
microsoftAccess19 = survey2019.iloc[:,238].notnull().astype('int')
awsRelationalDatabaseService19 = survey2019.iloc[:,239].notnull().astype('int')
awsDynamoDB19 = survey2019.iloc[:,240].notnull().astype('int')
azureSQLDatabase19 = survey2019.iloc[:,241].notnull().astype('int')
googleCloudSQL19 = survey2019.iloc[:,242].notnull().astype('int')

dfn19 = pd.DataFrame({
    "MySQL" : mySQL19,
    "PostgresSQL": postgresSQL19,
    "SQLite": sqlite19,
    "MicrosoftSQLServer": microsoftSQLServer19,
    "OracleDatabase": oracleDatabase19,
    "MicrosoftAccess": microsoftAccess19,
    "AWSRelationalDatabase" : awsRelationalDatabaseService19,
    "AWSDynamoDB": awsDynamoDB19,
    "AzureSQLDatabase": azureSQLDatabase19,
    "GoogleCloudSQL" : googleCloudSQL19
})

combine19 = pd.concat([df19, dfn19], axis = 1)

dfdn = dfn19.sum(axis=0).sort_values(ascending=False)
dfdn = dfdn.reset_index() 
dfdn.columns = ["DataBase","Number of Users"]
sns.catplot(y="DataBase", x="Number of Users",data=dfdn, 
            kind='bar', height=6, aspect=1.5);
plt.figtext(0.5, 1.1, "Figure 23 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# # Relationship between Age group and frequency of different Database

# In[ ]:


###################################################################################################
age2019 = combine19.iloc[:,[0,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Age",as_index=False).sum()
melted19 = age2019.melt(id_vars="Age",var_name="DataBase",value_name="Users")
sns.catplot(y="Age", x="Users",data=melted19, 
            kind='bar', height=9, aspect=1.5,hue="DataBase");
plt.title("Frequency of different DataBase Platform users by their age group");
plt.figtext(0.5, 1.1, "Figure 24 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 24** it is vivid that, in every age group **MySQL** is more popular than any other Database.

# In[ ]:


age2019


# # Is frequency of different Databases users depend on their Age group ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# $H_0$ : Frequency of different Database users is independent of their age group
# $H_1$ : Frequency of different Database users is depending on their age group
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(age2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different age group is having different likedness for different Database.

# # Relationship between Gender and frequency of different Database users

# In[ ]:


gender2019 = combine19.iloc[:,[1,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Gender",as_index=False).sum()

gender2019 = gender2019[(gender2019.Gender =="Male")| (genderCloud2019.Gender =="Female")]

melted19 = gender2019.melt(id_vars="Gender",var_name="DataBase",value_name="Users")
sns.catplot(y="Gender", x="Users",data=melted19, 
            kind='bar', height=9, aspect=1.5,hue="DataBase");
plt.title("Frequency of different Database users by their Gender");
plt.figtext(0.5, 1.1, "Figure 25 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 25** it is vivid that, in every gender group **MySQL** is more popular than any other Database. But we are left with a question that, **Is frequency of different Database users depend on their Age group ?**.

# In[ ]:


gender2019


# # Is frequency of different Databases users depend on their Gender ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# $H_0$ : Frequency of different Databases users is independent of their Gender
# $H_1$ : Frequency of different Databases users is depending on their Gender
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(gender2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different gender group is having different likedness for different Database. 

# # Relationship between Country of residance and frequency of different Database users

# In[ ]:


country2019 = combine19.iloc[:,[2,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Country",as_index=False).sum()
country2019 = country2019[country2019.Country.isin(top10Country.Country.to_list())]
melted19 = country2019.melt(id_vars="Country",var_name="DataBase",value_name="Users")
sns.catplot(y="Country", x="Users",data=melted19, 
            kind='bar', height=9, aspect=1.5,hue="DataBase");
plt.title("Frequency of different  Database users by their Country");
plt.figtext(0.5, 1.1, "Figure 26 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 26** it is vivid that, in every age group **MySQL**  is more popular than any other Database.

# In[ ]:


country2019


# # Is frequency of different Database users depend on their Country ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# $H_0$ : Frequency of different Database users is independent of their country
# $H_1$ : Frequency of different Database users is depending on their country
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(country2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different Country group is having different likedness for different Database.

# # Relationship between Education and frequency of different Database users

# In[ ]:


edu2019 = combine19.iloc[:,[3,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Education",as_index=False).sum()
melted19 = edu2019.melt(id_vars="Education",var_name="DataBase",value_name="Users")
sns.catplot(y="Education", x="Users",data=melted19, 
            kind='bar', height=9, aspect=1.5,hue="DataBase");
plt.title("Frequency of different Database users by their Education");
plt.figtext(0.5, 1.1, "Figure 27 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 27** it is vivid that, in every age group **MySQL** is more popular than any other Database.

# In[ ]:


edu2019


# # Is frequency of different Database users depend on their Education ?
# 
# We can answer this question using chi-square test. For chi-square test of independence. Following are hypothesis :
# 
# $H_0$ : Frequency of different Database users is independent of their Education
# $H_1$ : Frequency of different Database users is depending on their Education
# 
# Following codeblock perform chi-square test of independence.

# In[ ]:


resData = sts.chi2_contingency(edu2019.iloc[:,1:])
print("p-value of chi-square test is : ", resData[1])


# From analysis in codblock above, using the **p-value** of chi-square test of independence, we are not having proper evidence to reject Null hypothesis for chi-square test of independence. Hence it can be concluded that different education group is having different likedness for different Database.

# # Relationship between Jobrole and frequency of different Database users

# In[ ]:


jobrole2019 = combine19.iloc[:,[4,5,6,
                                          7,8,9,10,11,
                                          12,13,
                                          14]].groupby("Jobrole",as_index=False).sum()
melted19 = jobrole2019.melt(id_vars="Jobrole",var_name="DataBase",value_name="Users")
sns.catplot(y="Jobrole", x="Users",data=melted19, 
            kind='bar', height=9, aspect=1.5,hue="DataBase");
plt.title("Frequency of different Database users by their Jobrole");
plt.figtext(0.5, 1.1, "Figure 28 :", wrap=True, horizontalalignment='center',
            fontsize=15,color="blue",alpha=1,fontweight="black");


# From **Figure 28** it is vivid that, in every jobrole group **MySQL** is more popular than any other Databse. But in Database Enginner Microsoft SQL Server is more popular.

# In[ ]:


jobrole2019


# ### Finding from above analysis
# 
# 1. Jupyter Notebook and JupyterLab is most popular used IDE.
# 2. Python is Most popular programming language by Kagglers.
# 3. Amazon Web Services is most popular Big Data Platform.
# 4. Google Big Query is most popular Big Data Tool
# 5. MySQL is most popular Database.

# ## Future Work 
# 
# 1. Detail analysis for other tools used by Kagglers
