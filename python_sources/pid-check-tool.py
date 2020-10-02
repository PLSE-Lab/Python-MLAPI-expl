#!/usr/bin/env python
# coding: utf-8

# Check all running processes:

# In[ ]:


import psutil

for proc in psutil.process_iter():
        print(proc)


# Check some PID:

# In[ ]:


PID = 1

if PID in psutil.pids():
    print("Process with pid =", PID, "is currentrly running")
else:
    print("Process was not found")


# In[ ]:


PID = 2

if PID in psutil.pids():
    print("Process with pid =", PID, "is currentrly running")
else:
    print("Process was not found")


# Interacrive mode (need to Copy&Edit the Notebook):

# In[ ]:


PID = int(input("Enter your PID: "))

if PID in psutil.pids():
    print("Process with pid =", PID, "is currentrly running")
else:
    print("Process was not found")

