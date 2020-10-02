#!/usr/bin/env python
# coding: utf-8

# ### 2012313961 Yoon Seungrok

# ###  Date:      2019. 11. 19

# #### ---------------------------------------------------------------------------------------------------------------------------------------------------------
# 
# 
# # HW2
# 
# Due 11-21-2019 Thursday 11:59:59pm
# Instruction: Download output.txt from http://momgoose.cs.sunykorea.ac.kr/cybersec/hw2/
# Write a python code and turn in all your source code and answers.
# 
# Q1. How many failed password attempt for root in output.txt file?
# 
# Q2. List all the authorized users who logged in successfully? Provide the lines from the outputfile?
# 
# Q3. List all the invalid username attempts.
# 
# Q4. List all the IPs and its city, country to access the server as a root Q5. Was any root login attempt successful?
# 
# Why?
# 
# Why not?
# 
# Justify your answers.
# #### ---------------------------------------------------------------------------------------------------------------------------------------------------------
# 
# 

# ### Preprocessing

# In[ ]:


import pandas as pd 
import os


# In[ ]:


output= pd.read_table("../input/output/output.txt",header=None, sep='\t')
output= output.rename(columns= {0:"Log"})
new= output["Log"].str.split(" " ,n = 5, expand = True)
output["Date"]= new[0].astype(str) + ' '+new[1]+' '+new[2]
output["Type"]=new[3] +" "+ new[4]
new1=new[5].str.split(";",n=1,expand=True)
new2=new1[0].str.split("from",n=1,expand=True)
output["Msg"]= new2[0]
output["IP&PORT"]=new2[1]
output["Detail"]=new1[1]
output=output.drop(columns=['Log'])


# ### Save the output file as csv for the later use.

# In[ ]:


output.to_excel("output_excel.xlsx")


# ### Shape of the DataFrame

# There are 266279 logs in the output.txt. 

# In[ ]:


output.shape


# In[ ]:


output.head(10) 


# # Q1. How many failed password attempt for root in output.txt file?
# 
# - There were 71025 failed password attempts for root in output.txt file.

# In[ ]:


q1=output[output["Msg"].str.contains("Failed password for root", regex=False, case=False, na=False)]
display(q1)


# In[ ]:


output.tail()


# ### Save failed password attempts

# In[ ]:


q1.to_excel("q1_failedPasswordAttempts_for_root.xlsx")


# # Q2. List all the authorized users who logged in successfully? Provide the lines from the outputfile? 
# 
# ### <Authorized user name & IP >
# - youngmin.kwon(10.1.1.3 , 10.12.13.151), 
# - root(114.79.146.115),
# - simonwoo(115.145.188.81 )

# However......who is root from 114.79.146.115?

# ### Successful login logs

# In[ ]:


q2= output[output["Msg"].str.contains("Accepted password" ,regex=False, case=False, na=False)]
display(q2)


# In[ ]:


authorizedUserList= set()
for i in q2["Msg"].str.split("for"):
    authorizedUserList.add(i[1])
print("Authorized users are: ", '\t'.join(i  for i in authorizedUserList ))


# # Q3 List all the invalid username attempts.

# First, I selected all the rows that contain  "authentication failure" ,"invalid/Invalid", and "Failed" in order to make the flow of invalid attempts more visible since most of the invalid approaches have a log flow like below ;
# - 1) input_userauth_request with invalid user ID (the request host IP is recorded in the Detail column) (not always)
# - 2) try authentication 
# - 3) Failed password for invalid user ID 

# In[ ]:


invalidAttempts=output.loc[
              (output['Msg'].str.contains("authentication failure")) |  
              (output['Msg'].str.contains("invalid",regex=False, case=False, na=False)) |  
             (output['Msg'].str.contains("Failed",regex=False, case=False, na=False))]
display(invalidAttempts)


# Below are invalid username attempts.

# In[ ]:


invalidUserNameAttempts=  invalidAttempts[invalidAttempts["Msg"].str.contains("invalid user") & (invalidAttempts["IP&PORT"].notnull())]
invalidUserNameAttempts


# Below is an invalidUserNameList attempted so far. In total, 8444 attempts were carried out with below listed names.

# In[ ]:


invalidUserNameList= pd.DataFrame((invalidUserNameAttempts['Msg'].str.split(" ",n=5,expand=True)[5]).unique())
invalidUserNameList


# ### Save invalid username attempts

# In[ ]:


invalidUserNameAttempts.to_excel('q3_invalidUsernameAttempts.xlsx')


# # Q4 List all the IPs and its city, country to access the server as a root 
# 

# After filtering rows with failed and accepted password attempts for root, Iextracted all the IP addresses with  '.unuque()' method to avoid IP overlap.

# In[ ]:


q4=output[output["Msg"].str.contains("Failed password for root", regex=False, case=False, na=False)|
          output["Msg"].str.contains("Accepted password for root", regex=False, case=False, na=False)]


# In[ ]:





# In[ ]:


ipList= q4["IP&PORT"].str.split(" ",n=2, expand=True)
# ipList=pd.DataFrame(ipList[1].unique())
ipList=ipList[1].unique()
ipListDF=pd.DataFrame(ipList)
ipListDF=ipListDF.rename(columns={0:'IP'})
ipListDF


# In[ ]:


#save dataframe
ipListDF.to_excel('q4_country.xlsx')


# In[ ]:


get_ipython().system('pip install ip2geotools')
from ip2geotools.databases.noncommercial import DbIpCity


# In[ ]:


def location(ip):
    try:
        response = DbIpCity.get(ip, api_key='free')
        return "Country: {0}, City: {1}".format(response.country, response.city)
    except:
        print(ip , "exception")
        pass
        return None
ipListDF["Location"]= ipListDF["IP"].apply(lambda x: location(x))
display(ipListDF)
    


# In[ ]:


ipListDF.to_excel('q4_ipLocation.xlsx')


# In[ ]:





# In[ ]:





# # Q5. Was any root login attempt successful?

# In[ ]:


Nov 15 06:56:58 momgoose sudo: simonwoo : TTY=pts/0 ; PWD=/var/log ; USER=root ; COMMAND=/usr/bin/less auth.log
Nov 15 06:56:58 momgoose sudo: pam_unix(sudo:session): session opened for user root by simonwoo(uid=1013)


# In[ ]:





# In[ ]:





# In[ ]:




