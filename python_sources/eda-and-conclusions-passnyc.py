#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from numpy import * # linear algebra
from pandas import *# data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib.pyplot import *
from plotly import *
from scipy.stats import *
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Importing data
school=read_csv('../input/2016 School Explorer.csv')


# In[ ]:


#To view all the column names
set_option('display.max_seq_items',None)


# In[ ]:


school.columns


# In[ ]:


school.isnull().sum()
school.drop(['Adjusted Grade','New?','Other Location Code in LCGMS'],axis=1,inplace=True)


# In[ ]:


school.dropna(inplace=True)


# In[ ]:


school.isnull().sum()


# ***Distribution of Community Schools in each City***

# In[ ]:


#Distribution of Community Schools in each City
school_city=crosstab(school['City'], school['Community School?']).sort_values('No',ascending=True)
school_city

school_city.plot(kind='barh', stacked=True,figsize=(10,10),width=0.8)

title('Distribution of Community Schools in each City ')
xlabel('Name of the City')
ylabel('Frequency of Community Schools and Non-Community Schools')
legend(title='Community School',loc="center right")
show()


# **Conclusion:** The highest number of community schools is in Bronx. Brooklyn has the second highest number of community schools. New York has the third highest number of community schools. And it can be seen that there are community schools are present only in 7 cities.

# 
# ***Comparison of Economic Index between Community schools and Non-Community schools***

# In[ ]:


sns.set(rc={'figure.figsize':(7,5)})
sns.boxplot(x="Community School?", y="Economic Need Index", data=school,dodge=False,palette="seismic",width=0.3)
show()


# ***Conclusion:*** The average economic need index of Communtiy Schools is higher than the average economic need index of non-communtiy Schools. Therefore Community Schools should be considered more when compared to non-community schools.

# 
# ***Comparison of Average ELA proficiency between Community schools and Non-Community schools***

# In[ ]:


sns.boxplot(x="Community School?", y="Average ELA Proficiency", data=school,dodge=False,palette="husl",width=0.3)
show()


# ***Conclusion:*** The average ELA proficiency of Communtiy Schools is higher than the average ELA proficiency of Non-Communtiy Schools. But it can be seen the average ELA proficiency of both Community School and Non-Communtiy Schools falls under the category "partial but insufficient"(However there are outliers in both). If the Students in Community Schools are given better facilities they will study better and score better.
# 

#  ***Comparison of Average math proficiency between Community schools and Non-Community schools***

# In[ ]:


sns.set(rc={'figure.figsize':(7,5)})
sns.boxplot(x="Community School?", y="Average Math Proficiency", data=school,dodge=False,palette="hot",width=0.3)
show()


# ***Conclusion:*** The average Math proficiency of Communtiy Schools is higher than the average Math proficiency of Non-Communtiy Schools. But it can be seen the average Math proficiency of both Community School and Non-Communtiy Schools falls under the category "partial but insufficient"(However there are outliers in both). If the Students in Community Schools are given better facilities they will study better and score better.

#  ***Comparison of Supportive Environment Rating between Community schools and Non-Community schools***

# In[ ]:


grp=crosstab(school['Supportive Environment Rating'],school["Community School?"])
grp = grp.div(grp.sum(1), axis=0)
colors = ["#2E86C1","#D68910"]
grp.plot(kind='bar', stacked=True,figsize=(15,5),width=0.3,color=colors)
index=arange(9)
xticks(rotation=0)
title('Comparison of Supportive Environment Rating between Community schools and Non-Community schools')
# xticks(index,names)

legend(loc="upper right")
show()


# ***Conclusion:*** Majority of the students who have good Supportive Environment (Exceeding Target, Meeting target) are from the Non-Community Schools. Whereas the students from community schools do not have a good Supportive Environment when compared to the students from Non- Community schools relatively. So from this also we can infer that more suppport and help is required by the  students from Community schools when compared to the students from Non-Community schools.

#  ***Comparison of Rigorous Instruction Rating between Community schools and Non-Community schools***

# In[ ]:


grp=crosstab(school['Rigorous Instruction Rating'],school["Community School?"])
grp = grp.div(grp.sum(1), axis=0)
colors = ["#2E86C1","#D68910"]
grp.plot(kind='bar', stacked=True,figsize=(15,5),width=0.3,color=colors)
index=arange(9)
xticks(rotation=0)
title('Comparison of Rigorous Instruction Rating between Community schools and Non-Community schools')
# xticks(index,names)

legend(loc="upper right")
show()


# ***Conclusion:*** Majority of the students who have rigorous instruction (Exceeding Target, Meeting target) are from the Non-Community Schools. Whereas the students from community schools do not have  rigorous instruction when compared to the students from Non- Community schools relatively. So from this we can infer that; if given teachers who give rigorous training the  students from Community schools will perform better.

#  ***Comparison of Collaborative Teachers Rating between Community schools and Non-Community schools***

# In[ ]:


grp=crosstab(school['Collaborative Teachers Rating'],school["Community School?"])
grp = grp.div(grp.sum(1), axis=0)
colors = ["#2E86C1","#D68910"]
grp.plot(kind='bar', stacked=True,figsize=(15,5),width=0.3,color=colors)
index=arange(9)
xticks(rotation=0)
title('Comparison of Collaborative Teachers Rating between Community schools and Non-Community schools')
# xticks(index,names)

legend(loc="upper right")
show()


# ***Conclusion:*** Majority of the schools which have good Collaborative Teachers Rating (Exceeding Target, Meeting target) are Non-Community Schools. Whereas the community schools do not have good Collaborative Teachers Rating when compared to  Non- Community schools. So from this also we can infer that more suppport and help is required by the  students from Community schools when compared to the students from Non-Community schools.

# **From the above insights(based on the exploratory data analysis alone; However further analysis is needed) it can be concluded that the students studying in the Community school need more help when compared to the students studying in the Non-Community school. Therefore we shall further subset the Community schools inorder to find out the schools which need more help amongst the Community schools based on the "Economic Need Index" and "Student performance"**

# **Subsetting based on "Economic Need Index" and "Student performance"**

# In[ ]:


school_community=school.copy()
school_community=school_community[school_community['Community School?']=='Yes']


# In[ ]:


sns.set(rc={'figure.figsize':(12,5)})
sns.boxplot(x="City", y="Economic Need Index", data=school_community,palette="seismic",width=0.3)
show()


# In[ ]:


school_community_eco_need=school_community.loc[(school_community['City']=='NEW YORK') | (school_community['City']=='BRONX') | (school_community['City']=='BROOKLYN'),:]
len(school_community_eco_need)


# In[ ]:


school_community_eco_need['Student Achievement Rating']


# In[ ]:


school_community_eco_need.columns


# In[ ]:


achieve_community=DataFrame(melt(school_community_eco_need,id_vars=['School Name'], value_vars=['Student Achievement Rating']))
achieve_community
del achieve_community['variable']
achieve_community


# **Every school above except 'P.S. 298 DR. BETTY SHABAZZ' is either approaching the target or meeting the target or exceeding the same. So, it can be suggested that the students from these schools can be given SHSAT admissions exams.**

# **Association between Students Chronically Missing School and Supportive Environment Rating**

# In[ ]:


sns.set(rc={'figure.figsize':(13,8)})
school['Percent of Students Chronically Absent']=school['Percent of Students Chronically Absent'].astype('str')      
school['Percent of Students Chronically Absent'] = school['Percent of Students Chronically Absent'].str.rstrip('%').astype('float') *10
 
sns.boxplot(x="Supportive Environment Rating", y="Percent of Students Chronically Absent", data=school,dodge=False,            order=["Not Meeting Target","Approaching Target","Meeting Target","Exceeding Target"],palette="Oranges",width=0.4)
title('Association between Students Chronically Missing School and Supportive Environment Rating')
show()


# The schools where the Supportive Environment given to the students is not upto the mark("Not Meeting Target", "Approacing target") is where the chronic absence of the students is more.

# ***Association between Students Chronically Missing School and Rigorous Instruction Rating***

# In[ ]:


sns.set(rc={'figure.figsize':(13,8)})

sns.boxplot(x='Rigorous Instruction Rating', y="Percent of Students Chronically Absent", data=school,dodge=False,            palette="Blues",width=0.4,order=["Not Meeting Target","Approaching Target","Meeting Target","Exceeding Target"])
title('Association between Students Chronically Missing School and Rigorous Instruction Rating')
show()


# ***Conclusion***: The schools where the Instruction given to the students is not upto the mark("Not Meeting Target", "Approacing target") is where the chronic absence of the students is more.

# **Analysing the factors which affects the chronical school missing scenarios**

# ***Association between Students Chronically Missing School and Collaborative Teachers Rating***

# In[ ]:


sns.set(rc={'figure.figsize':(13,8)})

sns.boxplot(x='Collaborative Teachers Rating', y="Percent of Students Chronically Absent", data=school,dodge=False,            palette="Greens",width=0.4,order=["Not Meeting Target","Approaching Target","Meeting Target","Exceeding Target"])
title('Association between Students Chronically Missing School and Collaborative Teachers Rating')
show()


# 
# ***Conclusion***: The schools where the Collaborative effort of the teachers is not upto the mark("Not Meeting Target", "Approacing target") is where the chronic absence of the students is more.

# **Association between Students Chronically Missing School and Strong Family-Community Ties Rating**

# In[ ]:


sns.set(rc={'figure.figsize':(13,8)})

sns.boxplot(x='Strong Family-Community Ties Rating', y="Percent of Students Chronically Absent", data=school,dodge=False,            palette="Purples",width=0.4,order=["Not Meeting Target","Approaching Target","Meeting Target","Exceeding Target"])
title('Association between Students Chronically Missing School and Strong Family-Community Ties Rating')
show()


# ***Conclusion***: The average absence of the Students increases as the Strength of Family-Community increases. But it can however be seen that the average attendence of students calculated based on  Strength of Family-Community is almost the same. Therefore it can be seen that the attendence percentage of the students is not much affected by the Strength of Family-Community.

# **Association between Students Chronically Missing School and Trust Rating**

# In[ ]:


sns.set(rc={'figure.figsize':(13,8)})

sns.boxplot(x='Trust Rating', y="Percent of Students Chronically Absent", data=school,dodge=False,            palette="Reds",width=0.4,order=["Not Meeting Target","Approaching Target","Meeting Target","Exceeding Target"])
title('Association between Students Chronically Missing School and Trust Rating')
show()


# ***Conclusion***: The schools where theTrust Rating is not upto the mark("Not Meeting Target", "Approacing target") is where the chronic absence of the students is more.
