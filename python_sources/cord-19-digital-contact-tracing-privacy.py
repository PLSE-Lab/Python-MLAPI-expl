#!/usr/bin/env python
# coding: utf-8

# CORD-19 Digital Contact Tracing Privacy
# ======
# ![](https://images.unsplash.com/photo-1590935216595-f9fc9b65179d?ixlib=rb-1.2.1&auto=format&fit=crop&w=900&q=53)
# *Photo Credit: [@marcuswinkler](https://unsplash.com/@markuswinkler) on [Unsplash](https://unsplash.com/photos/A0iDWXTrQEY)*
# 
# Contact tracing is one of the key intervention methods to mitigate the spread of COVID-19. Explore CORD-19 articles related to digital contact tracing methods, privacy concerns and protections being put in place. Go beyond the headlines and read the studies for yourself.
# 

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'from cord19reports import install\n\n# Install report dependencies\ninstall()')


# In[ ]:


get_ipython().run_cell_magic('capture', '--no-display', 'from cord19reports import report, render\n\ntask = """\nname: query\n\ndigital contact tracing privacy:\n    query: digital mobile contact tracing +privacy\n    columns:\n        - name: Date\n        - name: Study\n        - name: Study Link\n        - name: Journal\n        - name: Study Type\n        - {name: Tracing Method, query: wifi network bluetooth blockchain beacon os app, question: What technical method}\n        - {name: Privacy Issues, query: privacy concerns, question: What privacy concerns}\n        - {name: Privacy Protections, query: privacy protection mitigations, question: What methods to preserve privacy, snippet: True}\n        - name: Sample Size\n        - name: Study Population\n        - name: Matches\n        - name: Entry\n"""\n\n# Build and render report\nreport(task)\nrender("query")')

