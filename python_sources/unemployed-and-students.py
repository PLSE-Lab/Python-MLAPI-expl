#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# In this kernel we analized three diferent subjects from the Kaggle Survey 2018. In the first subject we will try to find the causes of unemployment. In the second subject we will try to  analize how much are the students involved with Data Science and Machine Learning. In the last subject we will try to study the advantages of become a Data Scientist.
# 

# In[ ]:


import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))


# **Data**
# 
# From The 2018 Kaggle Machine Learning & Data Science Survey we will study the **multipleChoiceResponse** dataset which have 50 questions and multiple choice answers. We took the questions which only have one answer.  This is the main file we will be working with.
# 

# In[ ]:


multiple = pd.read_csv('../input/multipleChoiceResponses.csv', dtype=np.object)


# **Unemployment **
# 
# We realized that people who finished their undergradute major and is thinking about attain a higgest level of formal education have the highest unemployment rate. Most of them are thinking about studying a master's degree and they are between 22 and 29 years old, which means that most of them are students. They are from India, United States and Russia.
# 
# 

# In[ ]:


ne = multiple[multiple.Q6 == "Not employed"]
q4_ne = ne[["Q4"]]
q4_nep = q4_ne.Q4.value_counts().plot(kind="pie", autopct='%1.1f%%',figsize = (10,10))


# In[ ]:


q2_ne = ne[["Q2"]]
q2_nep = q2_ne.Q2.value_counts().plot(kind="pie",autopct='%1.1f%%',figsize = (10,10))


# In[ ]:


q3_ne = ne[["Q3"]]
q3_ne.Q3.value_counts().plot(kind="pie",autopct='%1.1f%%',figsize = (10,10))


# **What about the students?**
# 
# The age of the most of the students who answered the kaggle's survey" is between 18-24 years old which is the 75% of the total of the students. They are from India, United States and China. Most of them have a undergrate major which could be described as "Computer science". The most popular tool which they analize data are Local or hosted development enviroments like RStudio,JupyterLab, etc. They also use python as programming lenguage and they spend enough time practicing and improving their programming skills. They also think that the best way to practice is working in independent projects.  They think in a near future they could become in DataScientist.
# 
# 

# In[ ]:


st = multiple[multiple.Q6 == "Student"]
q1_st = st[["Q2"]]
q1_st.Q2.value_counts().plot(kind="pie",label=False,title = "Age of students who answered the kaggle's survey" ,autopct='%1.1f%%',figsize = (10,10))
plt.legend()
plt.show()


# In[ ]:


q3_st = st[["Q3"]]
q3_st.Q3.value_counts().plot(kind="pie",title= "", autopct='%1.1f%%',figsize = (10,10))


# In[ ]:


q12_st = st[["Q12_MULTIPLE_CHOICE"]]
q12_st.Q12_MULTIPLE_CHOICE.value_counts().plot(kind="pie",title= "", autopct='%1.1f%%',figsize = (10,10))


# In[ ]:


q17_st = st[["Q17"]]
q17_st.Q17.value_counts().plot(kind="pie",title= "the specific programming lenguage used most often", autopct='%1.1f%%',figsize = (10,10))


# In[ ]:


q23_st = st[["Q23"]]
q23_st.Q23.value_counts().plot(kind="pie",title= "Percent of the time that students spent actively coding", autopct='%1.1f%%',figsize = (10,10))


# In[ ]:


q26_st = st[["Q26"]]
q26_st.Q26.value_counts().plot(kind="pie",title= "", autopct='%1.1f%%',figsize = (10,10))


# 

# In[ ]:





# In[ ]:




