#!/usr/bin/env python
# coding: utf-8

# In[ ]:


all_files = get_ipython().getoutput('ls /kaggle/input/open-access-karl-fristons-papers/*.xml')


# In[ ]:


import lxml.etree as le
import lxml.html, lxml.html.clean
from textwrap import wrap

cleaner = lxml.html.clean.Cleaner(allow_tags=[], remove_unknown_tags=False)

for xml_file in all_files:
    with open(xml_file,'r') as f:
        doc=le.parse(f)
        body = doc.xpath("./body")[0]
        for elem in body.xpath(".//disp-formula") + body.xpath(".//inline-formula") + body.xpath(".//fig") + body.xpath(".//title") + body.xpath(".//sub") + body.xpath(".//sup") + body.xpath(".//italic"):
            parent=elem.getparent()
            parent.remove(elem)
        lxml.etree.strip_tags(body, "*")
        with open(xml_file.split("/")[-1].replace(".xml", ".txt"),'w') as f_out:
            text = "\n".join(wrap(body.text))
            print(text)
            f_out.write(text)


# In[ ]:




