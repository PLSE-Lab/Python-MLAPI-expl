#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing Dataset
import numpy as np
import pandas as pd 
import os
print(os.listdir("../input"))


# **Importing dataset of different format**

# In[ ]:


#Setting input directory
os.chdir("../input")

#Importing from *.txt
Depart_Info = pd.read_csv("Department_Information.txt" , sep =  "|")
Depart_Info.head()


# In[ ]:


Employee_Info= pd.read_csv("Employee_Information.txt" , sep =  "|")
Employee_Info.head()
Employee_Info.info()


# In[ ]:


Student_cons_info= pd.read_csv("Student_Counceling_Information.txt" , sep =  "|")

#reteriving observations of top five rows
Student_cons_info.head()


# In[ ]:


#Importing data from *.csv
Depart_inf = pd.read_csv("Department_Information.csv")
Depart_inf.head()


# In[ ]:


#Importing data *.xlsx
Employ_Info = pd.read_excel("Employee_Information.xlsx")
Employ_Info.head()


# In[ ]:


#Importing from *.sas7bdat
department_info= pd.read_sas('department_information.sas7bdat')
department_info.head()


# **Exporting dataset **

# In[ ]:


#Export to  *.txt
Student_cons_info.to_csv("../Student_Counceling_Information.txt")


# In[ ]:


#Export to *.csv 
Depart_inf.to_csv("../Department_Information.csv")


# In[ ]:


#Export to *.xlsx
Employ_Info.to_excel("../Employee_Information.xlsx")


# In[ ]:


#Export to *.sas7bdat 
department_info.to_sas("../department_information.sas7bdat")

