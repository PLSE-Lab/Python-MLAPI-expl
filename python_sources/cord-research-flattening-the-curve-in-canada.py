#!/usr/bin/env python
# coding: utf-8

# # Flattening the Curve

# ![covidblue.png](attachment:covidblue.png)

# In[ ]:


get_ipython().system('pip install git+https://github.com/dgunning/cord19.git')


# ## CORD Research Engine

# In[ ]:


from cord import ResearchPapers
papers = ResearchPapers.load(index='text')


# ## Viewing the Research Papers

# In[ ]:


papers


# In[ ]:


covid_papers = papers.since_sarscov2()


# In[ ]:


covid_papers.searchbar('relationships between testing tracing efforts and public health outcomes')


# In[ ]:


papers.match('.*Romer, ?P', column='authors')


# In[ ]:


papers[32213]

