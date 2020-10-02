#!/usr/bin/env python
# coding: utf-8

# It looks like the training data has some peculiar behaviour. If the ip, device and os fields are plotted for each click against the corresponding time, you can see that new ips, device and os ids appear every 24 hours or so.
# The download clicks are not distributed uniformly over these new values. 

# In[7]:


import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

sample_data = pd.read_csv( '../input/train_sample.csv' )

sample_data[ 'click_time' ] = pd.to_datetime( sample_data[ 'click_time' ] )

clicktime = np.array( [ t.timestamp() for t in sample_data[ 'click_time' ] ] )
clicktime -= np.min( clicktime )

sample_data[ 'time' ] = clicktime.astype( int )

sample_data.sort_values( by = [ 'time' ], inplace = True )
sample_data.reset_index( drop = True, inplace = True )

idx_download = sample_data.index[ sample_data[ 'is_attributed' ] == 1 ]

def plot_attribute_vs_time( sdata, idx, attr_name ):
    plt.figure(figsize=(16, 12))
    plt.plot( sdata[ 'time' ] / 3600, sdata[ attr_name ], 'b.', alpha = 0.1, label = 'is_attributed = 0' )
    plt.plot( sdata[ 'time' ][idx] / 3600, sdata[ attr_name ][idx], 'ro', alpha = 0.6, label = 'is_attributed = 1' )
    plt.xlabel( 'Time [hours]', fontsize = 16 )
    plt.ylabel( attr_name, fontsize = 16 )
    leg  = plt.legend( fontsize = 16 )
    for l in leg.get_lines():
        l.set_alpha( 1 )
        l.set_marker( '.' )
    plt.show()

plot_attribute_vs_time( sample_data, idx_download, 'ip' )
plot_attribute_vs_time( sample_data, idx_download, 'device' )
plot_attribute_vs_time( sample_data, idx_download, 'os' )


# In[ ]:




