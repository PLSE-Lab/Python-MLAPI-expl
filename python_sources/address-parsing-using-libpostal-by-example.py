#!/usr/bin/env python
# coding: utf-8

# **LibPostal** is A C library for parsing/normalizing street addresses around the world. Powered by statistical NLP and open geo data.

# In[ ]:


# Installing dependencies
get_ipython().system('conda install -c conda-forge postal -y')


# In[ ]:


# Address parser using Libpostal

from postal.parser import parse_address
from postal.expand import expand_address


# In[ ]:


import json


# LibPostal (Pypostal for python) give us as a response a dictionary so. As I want it as a json, I'll convert it with the following function (I prefere with identation).

# In[ ]:


def convert_json(address):
    new_address = {k: v for (v, k) in address}
    json_address = json.dumps(new_address, sort_keys=True,ensure_ascii = False, indent=1)
    return json_address


# Now, I use parse_address and expand address and return in json format.

# In[ ]:


def address_parser(address):
    
    # I use first position as default even though sometimes expand address returns more than one
    # Expand address tries to expand some 
    expanded_address = expand_address( address )[0]
    parsed_address = parse_address( expanded_address )
    json_address = convert_json(  parsed_address )
    
    return json_address
    
#     return convert_json(
#         parse_address( 
#             expand_address(address)[0] 
#             )
#         )


# In[ ]:


print(address_parser("10600 N Tantau Ave")) # Apple Address

