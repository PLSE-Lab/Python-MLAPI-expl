#!/usr/bin/env python
# coding: utf-8

# This notebook reads in the html file, parses it, gets just the text and then writes it out to a .txt file.

# In[ ]:


from bs4 import BeautifulSoup

# open the HTML file
with open("../input/simple") as f:
    text = f.read()

# parse our text file
parsed_html = BeautifulSoup(text, "lxml")

# print all the text elements to an output file
with open("pypi_packages.txt", "w+") as f:
    f.write(''.join(parsed_html.html.findAll(text=True)))


# In[ ]:


# print the first 25 lines of our output file
get_ipython().system('head -n 25 pypi_packages.txt')

