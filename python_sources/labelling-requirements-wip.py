#!/usr/bin/env python
# coding: utf-8

# ### labelling requirements 
# 
# This is work in progress code for labelling requirements with different classes which ultimately will lead to solving an nlp problem. 

# In[1]:


import pandas as pd
import os 
import glob
import json
import re 


# In[5]:


ls ../input/cityofla/CityofLA/


# In[19]:


BP = '../input/cityofla/CityofLA/'
def ji(a):
    """quick join cause I don't like typing"""
    return os.path.join(BP, a)

def load_file(fname):
    lis = list()
    with open(fname) as f:
        for i in f:
            lis.append(i)
    return lis


# In[20]:



file_names = glob.glob(ji('Job Bulletins/*'))
file_names = sorted(file_names)
data_dict = dict()

end_words = ['NOTE', 'PROCESS', 'DUTIES', 'SALARY', 'APPLY']

missed_fnames = list()

for fname in file_names:
    try:
        lst = load_file(fname)
        found_title = False
        dct = dict()
        for e, line in enumerate(lst):
            line = line.strip()
            line = re.sub(r'[\t\n]*', '', line)
            if not found_title and line != '\n': 
                dct['job_title'] = lst[e].strip() 
                found_title = True
            elif "Class Code:" in line:
                dct['class_code'] = int(line.split(' ')[-1].strip())
            elif 'REQUIREMENT' in line:
                # now we are in important part
                new_list = lst[e:]
                break


        done = False
        requirements_list = list()
        for line in new_list:
            # start a new bullet point
            # note this misses multiline bullet points
            for word in end_words:
                if word in line:
                    done = True
            if done:
                break
            requirements_list.append(line)

        dct['requirements'] = requirements_list
        data_dict[dct['class_code']] = dct
    except:
        missed_fnames.append(fname)
        print('didnt work')

# do some cleaning 
keys = list(data_dict.keys())
for key in keys:
    dd = data_dict[key]
    new_list = list()
    for e, i in enumerate(dd['requirements']):
        if e == 0:
            continue
        elif len(i) < 10:
            continue
        else:
            new_list.append(i.strip())
    data_dict[key]['requirements'] = new_list

with open('requirements_data.json', 'w') as f:
    json.dump(data_dict, f)


# In[ ]:





# In[22]:


with open('requirements_data.json') as f:
    req_dict = json.load(f) 


# In[23]:


# start labelling here. 


# In[24]:


label_dict = dict()


# In[25]:


# create iterator so we can get one key at a time
keys = iter(sorted(list(req_dict.keys())))
current_key = 0


# In[26]:


# running this cell prints out requirements for one bulletin
key = next(keys)
# increment key position: helps if we need to restart
current_key += 1
tmp = req_dict[key]['requirements']
for i in tmp:
    print(i, '\n')
print('position of current key: {}'.format(current_key))
print(key)


# In[ ]:


label_dict[key] = ['implicit_req']


# In[ ]:


label_dict[key] = ['explicit_req']


# In[ ]:


label_dict[key] = ['degree_and_implicit_req']


# In[ ]:


label_dict[key] = ['degree_and_implicit_req', 'explicit_req', 'explicit_req']


# In[ ]:


label_dict[key] = ['explicit_req', 'explicit_req']


# In[ ]:


label_dict[key] = ['referral', 'letter_of_rec']


# In[ ]:


label_dict[key] = ['explicit_req', 'letter_of_rec']


# In[ ]:


label_dict[key] = ['explicit_req', 'implicit_req']


# In[ ]:


label_dict[key] = ['explicit_req', 'degree_req']


# In[ ]:


label_dict[key] = ['degree_and_unit_req', 'degree_and_unit_req']


# In[ ]:


label_dict[key] = ['degree_req', 'certification_req', 'explicit_req']


# In[ ]:


label_dict[key] = ['degree_and_unit_req', 'explicit_req', 'null']


# In[ ]:


label_dict[key] = ['course_req', 'implicit_req']


# In[ ]:


label_dict[key] = ['explicit_req', 'course_unit_req']


# In[ ]:


label_dict[key] = ['implicit_req', 'course_unit_req']


# In[ ]:


label_dict[key] = ['implicit_req','explicit_req','course_req']


# In[ ]:


label_dict[key] = ['explicit_req', 'implicit_req', 'course_req']


# In[ ]:


label_dict


# In[ ]:


with open('requirements_labelled.json', 'w') as f:
    json.dump(label_dict, f)


# In[ ]:


# in case we need to restart
for i in range(current_key):
    key = next(keys)

