#!/usr/bin/env python
# coding: utf-8

# We have built a topic-based index to all the papers in the COVID-19 Open Research Dataset ([http://aipano.cse.ust.hk/covid/noncomm_1000/](http://aipano.cse.ust.hk/covid/noncomm_1000/)).  The system cateogrizes papers to topics that have been automatically detected among the papers.  
# 
# A hierarchy of topics is shown at the home page (see first picture below).  Each topic node can be expanded to show its child topics.  When a particular topic node is clicked, a list of papers belonging to that topic will be shown (see second picture below).  This can be used to find the papers for a particular topic.
# 
# We hope that the system can be helpful to researchers who wish to know what is in COVID-19 but do not have anything specific in mind to search for.

# In[ ]:


from IPython.display import Image
Image("/kaggle/input/20200416covid19/covid19-home.20200416.png")


# In[ ]:


Image("/kaggle/input/20200416covid19/covid19-z46.20200416.png")

