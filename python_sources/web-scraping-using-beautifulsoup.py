#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
from bs4 import BeautifulSoup
import pandas as pd


# In[ ]:




temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/26228-syracuse-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



r= pd.DataFrame(temp)


# In[ ]:


r.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)


# In[ ]:


r


# In[ ]:


df=r.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)
df


# In[ ]:





# In[ ]:


df = df[df.Name != 'Welcome to Yocket']


# In[ ]:


p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)


# In[ ]:


p.head()


# In[ ]:


columns = ['GRE', 'Undergrad']
df.drop(columns, inplace=True, axis=1)


# In[ ]:


df


# In[ ]:


df.to_csv("Syracuse1.csv")


# In[ ]:


pd.read_csv("Syracuse1.csv")


# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/95-university-of-maryland-college-park/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


# In[ ]:


p.head()


# In[ ]:


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)


# In[ ]:


p


# In[ ]:


p = p[p.Name != 'Welcome to Yocket']
p


# In[ ]:


columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)


# In[ ]:


p.to_csv("Umcp.csv")


# In[ ]:


pd.read_csv("Umcp.csv")


# # UIC 

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/310-university-of-illinois-at-chicago/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


# In[ ]:


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)


# In[ ]:


p = p[p.Name != 'Welcome to Yocket']


# In[ ]:


p.head()


# In[ ]:


p.to_csv("Uic1.csv")


# In[ ]:


pd.read_csv("Uic1.csv")


# # CMU

# In[ ]:


#14-carnegie-mellon-university
temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/46196-carnegie-mellon-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


# In[ ]:


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)


# In[ ]:


p = p[p.Name != 'Welcome to Yocket']


# In[ ]:


p.head()


# In[ ]:


p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)


# In[ ]:


p.head()


# In[ ]:


columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)


# In[ ]:


p.to_csv("CMU.csv")


# In[ ]:


pd.read_csv("CMU.csv")


# # CMU 2

# In[ ]:


#14-carnegie-mellon-university
temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/14-carnegie-mellon-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


# In[ ]:


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)


# In[ ]:


p = p[p.Name != 'Welcome to Yocket']


# In[ ]:


p.head()


# In[ ]:


p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)


# In[ ]:


columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)


# In[ ]:


p.to_csv("CMU2.csv")


# # U Wash

# In[ ]:



temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/46121-university-of-washington/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


# In[ ]:


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)


# In[ ]:


p = p[p.Name != 'Welcome to Yocket']


# In[ ]:


p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)


# In[ ]:


columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)


# In[ ]:


p.to_csv("UWash.csv")


# # U WASH 2 (IM)

# In[ ]:



temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/46140-university-of-washington/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


# In[ ]:


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)


# In[ ]:


p = p[p.Name != 'Welcome to Yocket']


# In[ ]:


p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)


# In[ ]:


columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)


# In[ ]:


p.to_csv("Uwash2.csv")


# # TAMU MIS

# In[ ]:



temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/49-texas-a-and-m-university-college-station/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


# In[ ]:


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)


# In[ ]:


p = p[p.Name != 'Welcome to Yocket']


# In[ ]:


p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)


# In[ ]:


columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)


# In[ ]:


p.to_csv("TAMU.csv")


# # Kelley -Indiana MIS

# In[ ]:



temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/368-indiana-university-bloomington/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


# In[ ]:


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)


# In[ ]:



p = p[p.Name != 'Welcome to Yocket']


# In[ ]:



p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)


# In[ ]:


columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)


# In[ ]:


p.to_csv("kelley.csv")


# # U CINN MIS

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/158-university-of-cincinnati/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


# In[ ]:


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)


# In[ ]:


p.to_csv("UCinn.csv")


# # UTD ITM

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/26891-university-of-texas-dallas/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


# In[ ]:




p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("UTD.csv")


# # NEU MIS

# In[ ]:



temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/913-northeastern-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


# In[ ]:


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("NEU.csv")


# # Eller mis

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/26239-university-of-arizona/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Eller.csv")


# # Rutgers Mis

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/1417-rutgers-university-new-brunswick/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Rutgers1.csv")


# # Rutgers newark

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/358-rutgers-university-newark/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Rutgers2.csv")


# # minnesota Twin cities

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/423-university-of-minnesota-twin-cities/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("minnesota.csv")


# # georgia state uni
# 

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/545-georgia-state-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Rutgers2.csv")


# In[ ]:





# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/358-rutgers-university-newark/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Rutgers2.csv")


# In[ ]:





# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/358-rutgers-university-newark/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("gsu.csv")


# # worcester 

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/816-worcester-polytechnic-institute/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("wpi.csv")


# # rensselaer-polytechnic-institute

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/461-rensselaer-polytechnic-institute/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("RPI.csv")


# In[ ]:





# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/358-rutgers-university-newark/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Rutgers2.csv")


# # Boston

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/6-boston-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("BU.csv")


# # UC Irvine

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/1043-university-of-california-irvine/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("UCI.csv")


# # Suny B

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/128-state-university-of-new-york-at-buffalo/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("SunyB.csv")


# # northwestern

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/318-northwestern-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("northwestern.csv")


# # ASU

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/253-arizona-state-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("asu.csv")


# # U florida

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/77-university-of-florida/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Uflorida.csv")


# # UNCC

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/363-university-of-north-carolina-at-charlotte/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Uncc.csv")


# In[ ]:





# # Stevens

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/1451-stevens-institute-of-technology/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Stevens.csv")


# # U Iowa

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/1249-university-of-iowa/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("UIowa.csv")


# # RIT

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/179-rochester-institute-of-technology/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("RIT.csv")


# # ut Arlington

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/349-university-of-texas-arlington/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("UT_arlington.csv")


# # santa clara

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/622-santa-clara-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("SantaClara.csv")


# # ITT

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/414-illinois-institute-of-technology/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("IIT.csv")


# # university-of-delaware

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/402-university-of-delaware/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("delaware.csv")


# # university-of-utah

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/26240-university-of-utah/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("UUtah.csv")


# # drexel-university

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/1272-drexel-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("drexel.csv")


# # university-of-texas-austin

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/104-university-of-texas-austin/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("UT_Austin.csv")


# # new-york-university

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/589-new-york-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("NYU.csv")


# # pennsylvania-state-university

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/1063-pennsylvania-state-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Penn_State.csv")


# # university-of-pennsylvania

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/278-university-of-pennsylvania/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("UPenn.csv")


# # iowa-state-university

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/326-iowa-state-university/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("Iowa_State.csv")


# # university-of-california-los-angeles

# In[ ]:


temp=[]


for i in range(1,20):
    page = requests.get("https://yocket.in/applications-admits-rejects/57-university-of-california-los-angeles/{}".format(i))

    soup = BeautifulSoup(page.content, 'html.parser')



    name_containers = soup.find_all('div', class_ = 'col-sm-6')
    for i in name_containers:
        k =(i.div.text)
        t=[i for i in k.strip().split("\n") if len(i) is not 0]
        temp.append(t)



p= pd.DataFrame(temp)


p.rename(columns={0: 'Name', 'newName2': 'University', 1: 'University', 2: 'Year', 3: 'Status',4: 'GRE',5: 'GRE_SCORE',6: 'Eng_test',7:'Test_score',8: 'Undergrad',9: 'Undergrad_score',11: 'work_ex'}, inplace=True)

p = p[p.Name != 'Welcome to Yocket']

p=p.drop([10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ], axis=1)

columns = ['GRE', 'Undergrad']
p.drop(columns, inplace=True, axis=1)

p.to_csv("UCLA.csv")


# In[ ]:




