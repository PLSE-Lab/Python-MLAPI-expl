#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('git clone https://github.com/rblcoder/curvature.git')


# In[ ]:


get_ipython().system('pip install osmium')


# In[ ]:


get_ipython().system('pip install msgpack-python')


# In[ ]:


import os
os.chdir('curvature')


# In[ ]:


get_ipython().system('./processing_chains/adams_default.sh -h')


# In[ ]:


get_ipython().system('wget http://download.geofabrik.de/north-america/us/alabama-140101.osm.pbf')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


#Download from http://download.geofabrik.de/north-america/us/
get_ipython().system('./processing_chains/adams_default.sh -v alabama-140101.osm.pbf')


# In[ ]:


get_ipython().system("bin/curvature-collect -v --highway_types 'motorway,trunk,primary,secondary,tertiary,unclassified,residential,service,motorway_link,trunk_link,primary_link,secondary_link,service' alabama-140101.osm.pbf     | bin/curvature-collect  $verbose $input_file     | bin/curvature-pp filter_out_ways_with_tag --tag surface --values 'unpaved,dirt,gravel,fine_gravel,sand,grass,ground,pebblestone,mud,clay,dirt/sand,soil'     | bin/curvature-pp filter_out_ways_with_tag --tag service --values 'driveway,parking_aisle,drive-through,parking,bus,emergency_access'     | bin/curvature-pp add_segments     | bin/curvature-pp add_segment_length_and_radius     | bin/curvature-pp add_segment_curvature     | bin/curvature-pp filter_segment_deflections     | bin/curvature-pp split_collections_on_straight_segments --length 2414     | bin/curvature-pp roll_up_length     | bin/curvature-pp roll_up_curvature     | bin/curvature-pp filter_collections_by_curvature --min 300     | bin/curvature-pp sort_collections_by_sum --key curvature --direction DESC     > alabama-140101.msgpack")


# In[ ]:


get_ipython().system('ls -l')

