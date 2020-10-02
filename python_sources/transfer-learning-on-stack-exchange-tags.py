#!/usr/bin/env python
# coding: utf-8

# Transfer Learning on Stack Exchange Tags
# ========================================

# In[ ]:


# Imports
import numpy as np
import pandas as pd
import tensorflow as tf
from bs4 import BeautifulSoup
from IPython.display import display

# Convert csv files into dataframes
biology_pd = pd.read_csv('../input/biology.csv')
cooking_pd = pd.read_csv('../input/cooking.csv')
cryptology_pd = pd.read_csv('../input/crypto.csv')
diy_pd = pd.read_csv('../input/diy.csv')
robotics_pd = pd.read_csv('../input/robotics.csv')
travel_pd = pd.read_csv('../input/travel.csv')
test_pd = pd.read_csv('../input/test.csv')

# Print dataframe heads
print('Biology: %i questions' % biology_pd.shape[0])
display(biology_pd.head())
print('Cooking: %i questions' % cooking_pd.shape[0])
display(cooking_pd.head())
print('Crytology: %i questions' % cryptology_pd.shape[0])
display(cryptology_pd.head())
print('DIY: %i questions' % diy_pd.shape[0])
display(diy_pd.head())
print('Robotics: %i questions' % robotics_pd.shape[0])
display(robotics_pd.head())
print('Travel: %i questions' % travel_pd.shape[0])
display(travel_pd.head())
print('Test: %i questions' % test_pd.shape[0])
display(test_pd.head())


# In[ ]:


# Convert dataframes to ndarrays
biology_np = biology_pd[['title', 'content', 'tags']].as_matrix()
cooking_np = cooking_pd[['title', 'content', 'tags']].as_matrix()
cryptology_np = cryptology_pd[['title', 'content', 'tags']].as_matrix()
diy_np = diy_pd[['title', 'content', 'tags']].as_matrix()
robotics_np = robotics_pd[['title', 'content', 'tags']].as_matrix()
travel_np = travel_pd[['title', 'content', 'tags']].as_matrix()
test_np = test_pd[['title', 'content']].as_matrix()

# Parse html
def parse_html(data_np):    
    for i in range(data_np.shape[0]):
        soup = BeautifulSoup(data_np[i,1], 'html.parser')
        soup = soup.get_text()
        soup = BeautifulSoup(soup, 'html.parser')
        soup = soup.decode('utf8')
        data_np[i,1] = soup.replace('\n', ' ')


parse_html(biology_np)
parse_html(cooking_np)
parse_html(cryptology_np)
parse_html(diy_np)
parse_html(robotics_np)
parse_html(travel_np)
parse_html(test_np)


# In[ ]:


biology_x = np.sum(biology_np[:,0] + ' ' + biology_np[:,1])
biology_y = np.sum(biology_np[:,2])


# In[ ]:


cooking_x = np.sum(cooking_np[:,0] + ' ' + cooking_np[:,1])
cooking_y = np.sum(cooking_np[:,2])


# In[ ]:


cryptology_x = np.sum(cryptology_np[:,0] + ' ' + cryptology_np[:,1])
cryptology_y = np.sum(cryptology_np[:,2])


# In[ ]:


diy_x = np.sum(diy_np[:,0] + ' ' + diy_np[:,1])
diy_y = np.sum(diy_np[:,2])


# In[ ]:


robotics_x = np.sum(robotics_np[:,0] + ' ' + robotics_np[:,1])
robotics_y = np.sum(robotics_np[:,2])


# In[ ]:


travel_x = np.sum(travel_np[:,0] + ' ' + travel_np[:,1])
travel_y = np.sum(travel_np[:,2])


# In[ ]:


test_temp = test_np[:,0] + ' ' + test_np[:,1]


# In[ ]:


for i in range(len(test_temp)):
    test_temp[i] = test_temp[i].split(' ')


# In[ ]:


#print(test_temp)
test_x = np.sum(test_temp)

