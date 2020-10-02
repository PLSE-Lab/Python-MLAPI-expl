#!/usr/bin/env python
# coding: utf-8

# # Combinando Data Frames

# In[ ]:


import pandas as pd


# ## Append

# In[ ]:


Nombres = pd.DataFrame({'id':[1,2,3,4], 'Nombre': ["Ana", "Juan", "Carolina", "Pedro"]})
Nombres


# In[ ]:


Nombres2 = pd.DataFrame({'id':[5,6], 'Nombre': ["Julia", "Alberto"]})
Nombres2


# In[ ]:


Nombres.append(Nombres2)


# ## Merge

# ### Por una columna

# In[ ]:


Nombres = pd.DataFrame({'id':[1,2,3,4], 'Nombre': ["Ana", "Juan", "Carolina", "Pedro"]})
Nombres


# In[ ]:


Edad = pd.DataFrame({'id':[1,2,3,4], 'Edad':[11,21,8,15]})
Edad


# In[ ]:


pd.merge(Nombres, Edad)


# ### Especificando columna comun

# In[ ]:


Nombres = pd.DataFrame({'id_Nombres':[1,2,3,4], 'Nombre': ["Ana", "Juan", "Carolina", "Pedro"]})
Edad = pd.DataFrame({'id_edades':[1,2,3,4], 'Edad':[11,21,8,15]})


# In[ ]:


pd.merge(Nombres, Edad, left_on='id_Nombres', right_on='id_edades')


# ### How (inner, outer, left, right,...)

# In[ ]:


Nombres = pd.DataFrame({'id':[3,4,5,6], 'Nombre': ["Ana", "Juan", "Carolina", "Pedro"]})
Edad = pd.DataFrame({'id':[1,2,3,4], 'Edad':[11,21,8,15]})


# In[ ]:


pd.merge(Nombres, Edad, how='inner')


# In[ ]:


pd.merge(Nombres, Edad, how='outer')


# In[ ]:


pd.merge(Nombres, Edad, how='left')


# In[ ]:


pd.merge(Nombres, Edad, how='right')

