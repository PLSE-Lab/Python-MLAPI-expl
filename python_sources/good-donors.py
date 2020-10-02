#!/usr/bin/env python
# coding: utf-8

# In[104]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **DONORS**

# In[105]:


donor = pd.read_csv("../input/Donors.csv")
print(donor.info())


# In[3]:


# sample of donors
donor[["Donor ID","Donor City","Donor State","Donor Is Teacher","Donor Zip"]].head(5)


# In[4]:


# donors missing data
missing = donor.copy()
missing = missing.T
missed = missing.isnull().sum(axis=1)
print(missed)


# In[19]:


# Donor City column has many missing data, therefore we view Donor State instead
donor["Donor State"].value_counts().nlargest(30).plot.bar(figsize=(10,8))
plt.ylabel("Number of donors")
plt.xlabel("State")
plt.title("Top 30 States")


# In[7]:


# donor is teacher or not
donor["Donor Is Teacher"].value_counts().plot(kind="pie", autopct='%1.2f%%')
plt.legend(["Non Teacher","Teacher"])
plt.title("Percentage of Teachers vs Non")


# In[9]:


# number of donors in different States
ct = donor.groupby(["Donor State","Donor Is Teacher"]).agg({"Donor Is Teacher":"count"}).rename(columns={"Donor Is Teacher" : "Teacher Donor Counts"}).reset_index()
pd.pivot_table(ct,index="Donor State",columns="Donor Is Teacher",values="Teacher Donor Counts").plot(kind="bar",figsize=(14,11))
plt.xlabel("State")
plt.ylabel("Number of Donors")
plt.title("Donors in different States")


# Donors data has two columns of missing data: Donor City and Donor Zip. Hence, we focus on Donor State column to understand the distribution of the donors at DonorsChoose.org. 90% of the donors are non-teacher, as we can see the comparison of number of donors in different States.

# **DONATIONS**

# In[106]:


donation = pd.read_csv("../input/Donations.csv")
print(donation.info())


# In[11]:


# sample of donations
donation[["Project ID","Donation ID","Donor ID","Donation Included Optional Donation","Donation Amount","Donor Cart Sequence","Donation Received Date"]].head(5)


# In[12]:


# donations missing data
missing = donation.copy()
missing = missing.T
missed = missing.isnull().sum(axis=1)
print(missed)


# In[13]:


# donations included any optional donation
donation["Donation Included Optional Donation"].value_counts().plot(kind="pie",autopct="%1.2f%%")
plt.legend(["Included","Not Included"])
plt.title("Percentage of Optional Donation")


# In[15]:


# distribution of donation amount from 2012-2018 at DonorsChoose
donation["Donation Received Date"] = pd.to_datetime(donation["Donation Received Date"])
donation["Donation Year"] = donation["Donation Received Date"].dt.year
donation["Donation Year"] = donation["Donation Year"].astype(int)
donation.groupby("Donation Year")["Donation Amount"].sum().plot(kind="bar",rot=0)
plt.xlabel("Year")
plt.ylabel("Total Donations")
plt.title("Total Donations From 2012-2018")


# In[18]:


# distribution of donors  
donation["Donor ID"].value_counts().nlargest(30).plot.bar(figsize=(10,8))
plt.xlabel("Donor ID")
plt.ylabel("Number of Donations")
plt.title("Top 30 Donors")


# In[25]:


# distribution of projects  
donation["Project ID"].value_counts().nlargest(30).plot.bar(figsize=(10,8))
plt.xlabel("Project ID")
plt.ylabel("Number of Donors")
plt.title("Top 30 Projects")


# Summary of donors involved in the projects

# In[21]:


donation.groupby(["Project ID","Donor ID","Donation Year"])["Donation Amount"].sum().reset_index()


# Donors with highest donation amount

# In[22]:


donation.groupby("Donor ID")["Donation Amount"].sum().nlargest(30).reset_index()


# Projects with highest donation amount

# In[23]:


donation.groupby("Project ID")["Donation Amount"].sum().nlargest(30).reset_index()


# The amount of donations keeps increasing from 2013 to 2017, data for 2018 is not yet complete. The top 30 donors involve in donations more than 2500 times with the highest reaches above 17500 donations. The projects can have more than 200 donations and 3 projects received more than 600 donations. 

# **RESOURCES**

# In[107]:


resource = pd.read_csv("../input/Resources.csv")
print(resource.info())


# In[27]:


# sample of resources
resource[["Project ID","Resource Item Name","Resource Quantity","Resource Unit Price","Resource Vendor Name"]].head(5)


# In[28]:


# resources missing data
missing = resource.copy()
missing = missing.T
missed = missing.isnull().sum(axis=1)
print(missed)


# In[108]:


# remove missing data
resource1 = resource.copy()
resource1 = resource1.dropna(axis=0)
# group resource by Project ID
resource1.groupby("Project ID")["Resource Quantity","Resource Unit Price"].sum().sort_values(ascending=False,by="Resource Quantity").reset_index()[:100]


# In[32]:


# Top 30 Resource Vendor Name
resource1["Resource Item Name"].value_counts().nlargest(30).plot.bar(figsize=(10,8))


# In[30]:


# Top 30 Resource Vendor Name
resource1["Resource Vendor Name"].value_counts().nlargest(30).plot.bar(figsize=(10,8))


# All columns in resources data are missing, except for Project ID. The most used vendor at DonorsChoose is Amazon Business.  

# **TEACHERS**

# In[109]:


teacher = pd.read_csv("../input/Teachers.csv")
print(teacher.info())


# In[34]:


# sample of teachers data
teacher[["Teacher ID","Teacher Prefix","Teacher First Project Posted Date"]].head(5)


# In[35]:


# teachers missing data
missing = teacher.copy()
missing = missing.T
missed = missing.isnull().sum(axis=1)
print(missed)


# In[110]:


# remove Teacher Prefix missing data
teacher["Teacher Prefix"] = teacher["Teacher Prefix"].astype(str)
teacher["Teacher Prefix"] = teacher["Teacher Prefix"].dropna()
teacher["Teacher Prefix"].value_counts().plot(kind="pie",autopct="%1.2f%%")


# In[111]:


# create Teacher Gender based on Teacher Prefix
# assume Mx., Dr., and Teacher as Male
teacher["Teacher Gender"] = teacher["Teacher Prefix"].astype(str)
teacher["Teacher Gender"] = teacher["Teacher Gender"].fillna("Mr.")
teacher["Teacher Gender"] = teacher["Teacher Gender"].replace("Mrs.","Female")
teacher["Teacher Gender"] = teacher["Teacher Gender"].replace("Ms.","Female")
teacher["Teacher Gender"] = teacher["Teacher Gender"].replace("Mr.","Male")
teacher["Teacher Gender"] = teacher["Teacher Gender"].replace("Dr.","Male")
teacher["Teacher Gender"] = teacher["Teacher Gender"].replace("Mx.","Male")
teacher["Teacher Gender"] = teacher["Teacher Gender"].replace("Teacher","Male")
teacher["Teacher Gender"] = teacher["Teacher Gender"].replace("nan","Male")
teacher["Teacher Gender"] = teacher["Teacher Gender"].astype(str)
teacher["Teacher Gender"].value_counts().plot(kind="pie",autopct="%1.2f%%")


# In[38]:


# Teacher First Project
teacher["Teacher First Project Posted Date"] = pd.to_datetime(teacher["Teacher First Project Posted Date"])
teacher["Teacher Year"] = teacher["Teacher First Project Posted Date"].dt.year
teacher["Teacher Year"] = teacher["Teacher Year"].astype(int)
ty = teacher["Teacher Year"].value_counts()#.plot.bar(figsize=(10,8),rot=0)
plt.figure(figsize=(10,8))
plt.bar(np.arange(17),ty[sorted(ty.index)].values)
plt.xticks(np.arange(17),sorted(ty.index))
plt.xlabel("Year")
plt.ylabel("Number of Contributions")
plt.title("Teacher First Project from 2002-2018")


# Most of the teachers contribute in DonorsChoose are female. The number of teachers contributed at DonorsChoose keeps increasing, with the peak happen in 2016. 

# **SCHOOLS**

# In[112]:


school = pd.read_csv("../input/Schools.csv")
print(school.info())


# In[40]:


# sample of schools
school[["School ID","School Name","School Metro Type","School Percentage Free Lunch","School State","School Zip","School City","School County","School District"]].head(5)


# In[41]:


# schools missing data
missing = school.copy()
missing = missing.T
missed = missing.isnull().sum(axis=1)
print(missed)


# In[113]:


school1 = school.copy()
# impute median to School Percentage Free Lunch column
school1["School Percentage Free Lunch"].fillna(school1["School Percentage Free Lunch"].median())
# remove missing data in schools
school1 = school1.dropna(axis=0)
print(school1.info())


# In[51]:


# School Metro Type
school1["School Metro Type"].value_counts().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of School Type")


# In[52]:


# School State
school1["School State"].value_counts().plot.bar(figsize=(9,6))
plt.xlabel("State")
plt.ylabel("Number of Contributions")
plt.title("Top 30 States")


# In[54]:


# schools provide free lunch
sc = school1["School Percentage Free Lunch"].astype(int).value_counts()
plt.figure(figsize=(14,13))
plt.bar(np.arange(50),sc[sorted(sc.index)].values[:50],width=0.45)
plt.xticks(np.arange(50),sorted(sc.index[:50]))
plt.xlabel("Percentage of Free Lunch")
plt.ylabel("Number of Schools")
plt.title("Distribution of schools provided free lunch")


# There are two columns of missing data in Schools data, we impute School Percentage Free Lunch with its median. Top 3 states contributed to DonorsChoose are California, Texas and New York. Schools involved mostly in the projects are from suburban and urban, followed by school from rural area.

# **PROJECTS**

# In[114]:


project = pd.read_csv("../input/Projects.csv")
print(project.info())


# In[56]:


# sample of projects data
project[["Project ID","School ID","Teacher ID","Teacher Project Posted Sequence","Project Type","Project Title","Project Essay","Project Short Description","Project Need Statement","Project Subject Category Tree","Project Grade Level Category","Project Resource Category","Project Cost","Project Posted Date","Project Expiration Date","Project Current Status","Project Fully Funded Date"]].head(5)


# In[57]:


# projects missing data
missing = project.copy()
missing = missing.T
missed = missing.isnull().sum(axis=1)
print(missed)


# In[115]:


# impute missing data
project1 = project.copy()
project1 = project1.dropna(axis=0)
# Project Type
project1["Project Type"] = project1["Project Type"].dropna(axis=0)
project1["Project Type"].value_counts().plot(kind="pie",autopct="%1.2f%%",figsize=(6,6))
plt.title("Percentage of Project Type")


# In[59]:


# School ID
project1["School ID"].value_counts().nlargest(30).plot.bar(figsize=(10,8))
plt.xlabel("School ID")
plt.ylabel("Number of Schools")
plt.title("Top 30 Schools involved in Projects")


# In[60]:


project1["Teacher ID"].value_counts().nlargest(30).plot.bar(figsize=(10,8))
plt.xlabel("Teacher ID")
plt.ylabel("Number of Teachers")
plt.title("Top 30 Teachers involved in Projects")


# In[79]:


project1["Project Title"].value_counts().nlargest(30).plot.bar(figsize=(10,8))
plt.xlabel("Title")
plt.ylabel("Number of Titles")
plt.title("Top 30 Titles")


# In[62]:


project1["Project Subject Category Tree"].value_counts().nlargest(30).plot.bar(figsize=(10,8))
plt.xlabel("Subject Category")
plt.ylabel("Number of Categories")
plt.title("Top 30 Subjects")


# In[63]:


project1["Project Subject Category Tree"].value_counts().nlargest(5).plot(kind="pie",autopct="%1.2f%%",figsize=(6,5))
plt.title("Percentage of Top 5 Subjects")


# In[64]:


project1["Project Subject Subcategory Tree"].value_counts().nlargest(30).plot.bar(figsize=(10,8))
plt.xlabel("Subject Subcategory")
plt.ylabel("Number of Subcategory")
plt.title("Top 30 Subcategories in Projects")


# In[65]:


project1["Project Subject Subcategory Tree"].value_counts().nlargest(5).plot(kind="pie",autopct="%1.2f%%",figsize=(6,5))
plt.title("Percentage of Top 5 Subcategories")


# In[66]:


project1["Project Resource Category"].value_counts().plot.bar(figsize=(8,6))
plt.xlabel("Resource Category")
plt.ylabel("Number of Resources")
plt.title("Top Resources")


# In[67]:


project1["Project Grade Level Category"].value_counts().plot(kind="pie",autopct="%1.2f%%",figsize=(6,5))
plt.title("Percentage of Grade Category")


# In[68]:


project1["Project Posted Date"] = pd.to_datetime(project1["Project Posted Date"])
project1["Project Posted Year"] = project1["Project Posted Date"].dt.year
sn.distplot(project1["Project Posted Year"])
plt.title("Project distribution from 2013-2018")


# In[69]:


py = project1.groupby("Project Posted Year")["Project Cost"].sum()
plt.figure(figsize=(6,5))
plt.bar(np.arange(6),py[sorted(py.index)].values)
plt.xticks(np.arange(6),sorted(py.index))
plt.xlabel("Year")
plt.ylabel("Total Cost")
plt.title("Project Cost from 2013-2018")


# In[70]:


pt = project1.groupby(["Project Posted Year","Project Subject Category Tree"])["Project Cost"].sum().reset_index()
p1 = pt[pt["Project Subject Category Tree"]=="Literacy & Language"]
p2 = pt[pt["Project Subject Category Tree"]=="Math & Science"]
idx = np.arange(6)
width = 0.35
fig,ax = plt.subplots(figsize=(8,6))
ax.bar(idx-width/2,p1["Project Cost"],width,color="SkyBlue",label="Literacy & Language")
ax.bar(idx+width/2,p2["Project Cost"],width,color="IndianRed",label="Math & Science")
ax.set_xticks(idx)
ax.set_xticklabels(p2["Project Posted Year"])
ax.set_xlabel("Posted Year")
ax.set_ylabel("Total Cost")
ax.legend(["Literacy & Language","Math & Science"])
plt.title("Project Cost from 2013-2018")


# In[71]:


project1.groupby(["Project Posted Year","Project Type"])["Project Cost"].sum()


# **PROJECTS, SCHOOLS, TEACHERS**

# In[72]:


ps = pd.concat([project1,school1,teacher],axis=1,join_axes=[project1.index]).dropna(axis=0)
ps["School Metro Type"].value_counts().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of School Type")


# In[73]:


ps["School State"].value_counts().plot.bar(figsize=(9,6))
plt.xlabel("State")
plt.ylabel("Number of Projects")
plt.title("Top States involved in DonorsChoose")


# In[74]:


ps["Project Grade Level Category"].value_counts().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Grade in Projects")


# In[75]:


ps["Project Resource Category"].value_counts().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Resource in Projects")


# In[76]:


ps["Teacher Gender"].value_counts().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Teacher Gender in Projects")


# In[77]:


ps["Teacher Prefix"].value_counts().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Teacher Prefix in Projects")


# **CALIFORNIA**

# In[87]:


cl = ps[ps["School State"]=="California"][["Project Title","Project Subject Category Tree","Project Grade Level Category","Project Resource Category","Project Cost","School Metro Type","Teacher Gender","Teacher Year"]]
cl["Teacher Year"] = cl["Teacher Year"].astype(int)
cl.groupby("Project Subject Category Tree")["Project Cost"].sum().nlargest(5).plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on different Categories")


# In[89]:


cl.groupby("Project Title")["Project Cost"].sum().nlargest(5).plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Project Title")


# In[80]:


cl.groupby("School Metro Type")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on School Type")


# In[81]:


cl.groupby("Project Grade Level Category")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Grade Level")


# In[82]:


cl.groupby("Project Resource Category")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Resource Category")


# In[83]:


cl.groupby("Teacher Gender")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Teacher Gender")


# In[84]:


cl.groupby("Teacher Year")["Project Cost"].sum().plot(kind="bar",rot=0,figsize=(10,6))
plt.xlabel("Year")
plt.ylabel("Project Cost")
plt.title("Histogram of Teacher First Project")


# **TEXAS**

# In[90]:


tx = ps[ps["School State"]=="Texas"][["Project Title","Project Subject Category Tree","Project Grade Level Category","Project Resource Category","Project Cost","School Metro Type","Teacher Gender","Teacher Year"]]
tx["Teacher Year"] = tx["Teacher Year"].astype(int)
tx.groupby("Project Subject Category Tree")["Project Cost"].sum().nlargest(5).plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on different categories")


# In[92]:


tx.groupby("Project Title")["Project Cost"].sum().nlargest(5).plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Project Title")


# In[91]:


tx.groupby("School Metro Type")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on School Type")


# In[93]:


tx.groupby("Project Grade Level Category")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Grade Level")


# In[94]:


tx.groupby("Project Resource Category")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Resource Category")


# In[95]:


tx.groupby("Teacher Gender")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Teacher Gender")


# In[96]:


tx.groupby("Teacher Year")["Project Cost"].sum().plot(kind="bar",rot=0,figsize=(10,6))
plt.xlabel("Year")
plt.ylabel("Project Cost")
plt.title("Histogram of Teacher First Project")


# **NEW YORK**

# In[97]:


ny = ps[ps["School State"]=="New York"][["Project Title","Project Subject Category Tree","Project Grade Level Category","Project Resource Category","Project Cost","School Metro Type","Teacher Gender","Teacher Year"]]
ny["Teacher Year"] = ny["Teacher Year"].astype(int)
ny.groupby("Project Subject Category Tree")["Project Cost"].sum().nlargest(5).plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on different subjects")


# In[98]:


ny.groupby("Project Title")["Project Cost"].sum().nlargest(5).plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Project Title")


# In[99]:


ny.groupby("School Metro Type")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on School Type")


# In[100]:


ny.groupby("Project Grade Level Category")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Grade Level")


# In[101]:


ny.groupby("Project Resource Category")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Resource Category")


# In[102]:


ny.groupby("Teacher Gender")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Teacher Gender")


# In[103]:


ny.groupby("Teacher Year")["Project Cost"].sum().plot(kind="bar",rot=0,figsize=(10,6))
plt.xlabel("Year")
plt.ylabel("Project Cost")
plt.title("Histogram of Teacher First Project")

