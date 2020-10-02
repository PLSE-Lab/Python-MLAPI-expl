#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook is a concept test to prove the utility of having diagrams easily created with some code.
# 
# [PlantUML](http://plantuml.com/) is an open-source tool that uses simple textual descriptions to draw beautiful UML diagrams.
# 
# 
# 

# In[ ]:


get_ipython().system('pip install iplantuml')
import iplantuml


# In[ ]:


get_ipython().run_cell_magic('plantuml', '', '\n@startuml\nAlice -> Bob: Authentication Request\nBob --> Alice: Authentication Response\n@enduml')


# In[ ]:


get_ipython().run_cell_magic('plantuml', '', '\n@startuml\n\nframe "training phase" {\n  \n  folder "preprocessing" as TRPPP {\n    card labels as L\n    file "training data" as TRD\n    rectangle "feature extraction" as TRFE\n  }\n  \n  folder learning {\n    component "machine learning algorithm" as MLA\n  }\n  \n  L --> MLA\n  TRD --> TRFE\n  TRFE --> MLA\n}\n\nframe "test phase" {\n  \n  folder "preprocessing" as TSPPP {\n    file "test data" as TSD\n    rectangle "feature extraction" as TSFE\n  }\n  \n  folder evaluating {\n    component "model" as M\n  }\n  \n  TSD --> TSFE\n  TSFE --> M\n}\n\nMLA -right-> M\n\n@enduml')

