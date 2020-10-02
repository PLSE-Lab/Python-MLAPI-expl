#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt


# In[ ]:


graduate_admission_data= pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")
graduate_admission_data.head() #Inspecting first 5 rows


# In[ ]:


graduate_admission_data.shape


# In[ ]:


graduate_admission_data.describe() #Central Limit theorem is applied


# In[ ]:


graduate_admission_data.info() #No missing Values


# In[ ]:


graduate_admission_data["University Rating"]= graduate_admission_data["University Rating"]. astype("str")
graduate_admission_data.dtypes


# Since there is no missing values in this dataset and data type is perfect. There is no need for Data Cleaning

# ***DATA VISUALIZATION***

# In[ ]:


grp= graduate_admission_data.groupby("University Rating") #---Only to calculate no. of applications
x= grp["SOP"].agg(np.mean)
y= grp["CGPA"].agg(np.mean)
z= grp["Research"].agg(np.sum)
v= grp["GRE Score"].agg(np.mean)

print(x)
print(y)
print(z)
print(v)


# In[ ]:


plt.plot(x,color= "g")
plt.title("SOP strength with University Ranking")
plt.xlabel("University Ranking")
plt.ylabel("SOP strength")
plt.show()


# In[ ]:


plt.plot(y,color= "g")
plt.title("CGPA with University Ranking")
plt.xlabel("University Ranking")
plt.ylabel("CGPA")
plt.show()


# In[ ]:


plt.plot(z,color= "y")
plt.title("Researchers vs University Ranking")
plt.xlabel("University Ranking")
plt.ylabel("Researchers")
plt.show()


# In[ ]:


plt.plot(v,color= "m")
plt.title("GRE Score vs University Ranking")
plt.xlabel("University Ranking")
plt.ylabel("GRE Score")
plt.show()


# In[ ]:


graduate_admission_data.info()


# In[ ]:


print(graduate_admission_data.columns.tolist())


# In[ ]:


graduate_admission_data["Chance of Admit "].plot(kind = 'hist')


# To find:
# Reation between all the variables and chance of admit in different university rankings

# In[ ]:


Rank_1_Uni= graduate_admission_data.loc[graduate_admission_data["University Rating"]=="1",:]
Rank_1_Uni.head()
Rank_2_Uni= graduate_admission_data.loc[graduate_admission_data["University Rating"]=="2",:]
Rank_3_Uni= graduate_admission_data.loc[graduate_admission_data["University Rating"]=="3",:]
Rank_4_Uni= graduate_admission_data.loc[graduate_admission_data["University Rating"]=="4",:]
Rank_5_Uni= graduate_admission_data.loc[graduate_admission_data["University Rating"]=="5",:]


# In[ ]:


plt.scatter(x=Rank_1_Uni['GRE Score'], y = Rank_1_Uni['Chance of Admit '],color= "y")
plt.title("Relationship btw GRE Score and Chance of acceptance in University Ranking= 1")
plt.show()


# In[ ]:


plt.scatter(x=Rank_5_Uni['GRE Score'], y = Rank_5_Uni['Chance of Admit '],color= "b")
plt.title("Relationship btw GRE Score and Chance of acceptance in University Rank= 5")
plt.show()


# In[ ]:


plt.scatter(x=Rank_5_Uni['CGPA'], y = Rank_5_Uni['Chance of Admit '],color= "b")
plt.title("Relationship btw CGPA and Chance of acceptance in University Rank= 5")
plt.show()


# In[ ]:


plt.scatter(x=Rank_5_Uni['TOEFL Score'], y = Rank_5_Uni['Chance of Admit '],color= "b")
plt.title("Relationship btw TOEFL Score and Chance of acceptance in University Rank= 5")
plt.show()


# In[ ]:


plt.scatter(x=Rank_5_Uni['SOP'], y = Rank_5_Uni['Chance of Admit '],color= "b")
plt.title("Relationship btw SOP and Chance of acceptance in University Rank= 5")
plt.show()


# In[ ]:


plt.scatter(x=Rank_5_Uni['LOR '], y = Rank_5_Uni['Chance of Admit '],color= "b")
plt.title("Relationship btw LOR and Chance of acceptance in University Rank= 5")
plt.show()


# In[ ]:


plt.scatter(x=Rank_5_Uni['Research'], y = Rank_5_Uni['Chance of Admit '],color= "b")
plt.title("Relationship btw Research and Chance of acceptance in University Rank= 5")
plt.show()


# In[ ]:




