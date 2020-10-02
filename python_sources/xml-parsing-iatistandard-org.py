#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from lxml import etree
url_iatistandard_org = 'http://datastore.iatistandard.org/api/1/access/activity.xml'
# https://lxml.de/parsing.html
root = etree.parse(url_iatistandard_org)


# In[ ]:


root


# In[ ]:


etree.tostring(root)


# In[ ]:


# print(etree.tostring(root, pretty_print=True))
# https://www.journaldev.com/18043/python-lxml
print(etree.tostring(root, pretty_print=True).decode("utf-8"))


# In[ ]:


result = root.getroot()


# In[ ]:


result


# In[ ]:


# https://python101.pythonlibrary.org/chapter31_lxml.html
for appt in result.getchildren():
        for elem in appt.getchildren():
            #print(elem)
            print('elem.tag : ' + elem.tag)
            print('elem[0].tag : ' + elem[0].tag)
            print('elem[0].text : ' + elem[0].text)


# In[ ]:


# https://python101.pythonlibrary.org/chapter31_lxml.html
for appt in result.getchildren():
        for elem in appt.getchildren():
            for in_elem in elem.getchildren():
                print(in_elem.tag)
                if in_elem.text:
                    if isinstance(in_elem.text, str):
                        print('in_elem.text : '+ in_elem.text)


# In[ ]:


result.getchildren()[0].getnext().tag


# In[ ]:


# https://lxml.de/api.html
[ el.tag for el in root.iter() ]


# In[ ]:


len([ el.tag for el in root.iter() ])


# In[ ]:


[ el.text for el in root.iter('narrative') ]


# In[ ]:


len([ el.text for el in root.iter('narrative') ])


# In[ ]:


[ el.text for el in root.iter('participating-org') ]


# In[ ]:


len([ el.text for el in root.iter('participating-org') ])


# In[ ]:


[ el.text for el in root.iter('title') ]


# In[ ]:


[ el.text for el in root.iter('description') ]


# In[ ]:


[ el.text for el in root.iter('title', 'participating-org') ]

