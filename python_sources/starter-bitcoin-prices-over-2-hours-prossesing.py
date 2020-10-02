#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This Is a little Preprossesing example where I will be Graphing The Btc Price Data

# ## Imports
# matplotlib - pyplot

# In[ ]:


from matplotlib import pyplot as plt
import os


# Load The Data From The Data Set and Seperate it to be used
# 

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        Path = os.path.join(dirname, filename)  #Gets Dataset

with open(Path,"r") as raw:
	lined = raw.readlines()                     #Reads The Txt into Array
	lined = [l.replace("\n","") for l in lined] #removes the Spaceing Bettween Values
	Formated = [l.split("|",2) for l in lined]  #Splits The Values Into [[Date],[Time],[Price From USD]]

	Done = []                                   #Makes Empty Array

	for I,FMD in enumerate(Formated):
		FMD.pop(0)                              #Removes Date From Every Line
		FMD.pop(0)                              #Removes Time From Every Line
		FMD = FMD[0].replace("$","")            #Removes $ Character From every Price Value
		Done.append(float(FMD))                 #Appends Prices To Done Array and Converts The String To A Float

	lined = None                                #Clears Array To Save Memory
	Formated = None                             #Clears Array To Save Memory


# Plots Data With pyplot

# In[ ]:


plt.plot(Done)   #Plots Data
plt.show()       #Shows Graph

print(len(Done)) #Prints Length Of Dataset


# ## Conclusion
# This is A Very Easy To Use And Format Btc Price Dataset That can Be Used To Predict Prices And see Developmets
