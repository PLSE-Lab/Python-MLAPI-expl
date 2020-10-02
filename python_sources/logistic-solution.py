#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Download the dataset from google drive 
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
import torchvision.models as models
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm.notebook import tqdm

#Download the dataset by GDrive
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

download = drive.CreateFile({'id': '1gwxg6Moi2Q7xjt87M2_C1YaxC5QjSd5J'})
download.GetContentFile('dataset.zip')


# Getting the dataset

# In[ ]:


get_ipython().system('7za x dataset.zip -o\\dataset')


# In[ ]:


PATH = "dataset/"
sla = pd.read_excel(PATH + "SLA_matrix.xlsx")
dom = pd.read_csv(PATH + "delivery_orders_march.csv")


# In[ ]:


sla


# In[ ]:


dom.shape[0]


# In[ ]:


#Applying Lowercase for all the addresses
dom = dom.apply(lambda x: x.astype(str).str.lower())


# In[ ]:


dom["buyeraddress"] = dom["buyeraddress"].str.replace(" ","")
dom["selleraddress"] = dom["selleraddress"].str.replace(" ","")
len(dom[dom["buyeraddress"].str.contains("metromanila")]) + len(dom[dom["buyeraddress"].str.contains("luzon")]) +  len(dom[dom["buyeraddress"].str.contains("visayas")]) +len(dom[dom["buyeraddress"].str.contains("mindanao")])


# In[ ]:


#Since we have that the total is less than the real value => there are some mismatch name
dom.loc[dom["buyeraddress"].str.contains("metromanila"),"buyeraddress"] = "metromanila"
dom.loc[dom["buyeraddress"].str.contains("luzon"),"buyeraddress"] = "luzon"
dom.loc[dom["buyeraddress"].str.contains("visayas"),"buyeraddress"] = "visayas"
dom.loc[dom["buyeraddress"].str.contains("mindanao"),"buyeraddress"] = "mindanao"


# In[ ]:


#Convert into only consider the place
dom.loc[dom["selleraddress"].str.contains("metromanila"),"selleraddress"] = "metromanila"
dom.loc[dom["selleraddress"].str.contains("luzon"),"selleraddress"] = "luzon"
dom.loc[dom["selleraddress"].str.contains("visayas"),"selleraddress"] = "visayas"
dom.loc[dom["selleraddress"].str.contains("mindanao"),"selleraddress"] = "mindanao"


# In[ ]:


dom


# In[ ]:


dom['sla_time'] = 7


# In[ ]:


#Concat the route
dom["route"] = dom["buyeraddress"] + " " + dom["selleraddress"]


# In[ ]:


dom.loc[dom["route"] == "metromanila metromanila","sla_time"] = 3
dom.loc[dom["route"] == "luzon luzon","sla_time"] = 5
dom.loc[dom["route"] == "metromanila luzon","sla_time"] = 5
dom.loc[dom["route"] == "luzon metromanila","sla_time"] = 5


# In[ ]:


dom[dom["sla_time"] == 7]


# In[ ]:


cols = ["selleraddress","buyeraddress","route"]
dom.drop(cols,inplace = True,axis = 1)


# In[ ]:


dom['pick'] = pd.to_datetime(dom['pick'],unit='s').dt.date


# In[ ]:


dom['weekday_pick'] = pd.to_datetime(dom['pick']).dt.weekday


# In[ ]:


dom['1st_deliver_attempt'] = pd.to_datetime(dom['1st_deliver_attempt'],unit='s').dt.date
dom['weekday_1'] = pd.to_datetime(dom['1st_deliver_attempt']).dt.weekday


# In[ ]:


dom['2nd_deliver_attempt'] = pd.to_datetime(dom['2nd_deliver_attempt'],unit='s').dt.date
dom['weekday_2'] = pd.to_datetime(dom['2nd_deliver_attempt']).dt.weekday


# In[ ]:


dom


# In[ ]:


dom["diff1"] = dom["1st_deliver_attempt"] - dom["pick"]
dom["diff2"] = dom["2nd_deliver_attempt"] - dom["1st_deliver_attempt"]


# In[ ]:


duration_in_s = duration.total_seconds() 


# In[ ]:


dom["diff1"] = dom["diff1"].dt.days


# In[ ]:


dom


# In[ ]:


from datetime import date, timedelta

def all_sundays(year):
# January 1st of the given year
       dt = date(year, 1, 1)
# First Sunday of the given year       
       dt += timedelta(days = 6 - dt.weekday())  
       while dt.year == year:
          yield dt
          dt += timedelta(days = 7)
          
for s in all_sundays(2020):
   print(s)


# In[ ]:


dt = date(2020,3,25)
list_sundays = all_sundays(2020)


# In[ ]:


list_sundays = list(list_sundays)


# In[ ]:


list_sundays.append(dt)


# In[ ]:


list_sundays


# In[ ]:


dt1 = date(2020,3,30)
dt2 = date(2020,3,31)
list_sundays.append(dt1)


# In[ ]:


list_sundays.append(dt2)


# In[ ]:


list_sundays


# In[ ]:


dom_2nd = dom[pd.notnull(dom['2nd_deliver_attempt'])]


# In[ ]:





# In[ ]:


dom_2nd["is_late"] = 0
from tqdm.notebook import tqdm
for i in tqdm(range(0, dom_2nd.shape[0])):
  count = 0
  count_2 = 0
  pick = dom_2nd["pick"].iloc[i]
  first = dom_2nd["1st_deliver_attempt"].iloc[i]
  second = dom_2nd["2nd_deliver_attempt"].iloc[i]
  for single_date in pd.date_range(pick, first):
      if(single_date in list_sundays):
        count+=1
  dom_2nd["sla_time"].iloc[i] += count

  if(dom_2nd["diff1"].iloc[i] > dom_2nd["sla_time"].iloc[i]):
    dom_2nd["is_late"].iloc[i] = 1
    continue
  else:
    for single_date in pd.date_range(first,second):
      if(single_date in list_sundays):
        count_2+=1
    if(dom_2nd["diff2"].iloc[i] > (3 + count_2)):
      dom_2nd["is_late"].iloc[i] = 1
    
    




# In[ ]:


dom_1st = dom[pd.isnull(dom['2nd_deliver_attempt'])]
dom_1st["is_late"] = 0
for i in tqdm(range(0, dom_1st.shape[0])):
  count = 0
  pick = dom_1st["pick"].iloc[i]
  first = dom_1st["1st_deliver_attempt"].iloc[i]
  for single_date in pd.date_range(pick, first):
      if(single_date in list_sundays):
        count+=1
  dom_1st["sla_time"].iloc[i] += count

  if(dom_1st["diff1"].iloc[i] > dom_1st["sla_time"].iloc[i]):
    dom_1st["is_late"].iloc[i] = 1
    continue


# In[ ]:


dom_1st[dom_1st["is_late"]==1]


# In[ ]:


submit_1 = dom_1st.drop(["pick","1st_deliver_attempt","2nd_deliver_attempt","sla_time","weekday","weekday_pick","weekday_2","weekday_1","diff1","diff2"],axis = 1)


# In[ ]:


submit_1


# In[ ]:


submit_2 = dom_2nd.drop(["pick","1st_deliver_attempt","2nd_deliver_attempt","sla_time","weekday","weekday_pick","weekday_2","weekday_1","diff1","diff2"],axis = 1)


# In[ ]:


submit_2[submit_2["is_late"] == 1]


# In[ ]:


pd.concat([submit_1, submit_2], ignore_index=True).to_csv("submission.csv",index= False)


# In[ ]:




